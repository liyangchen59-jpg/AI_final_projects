# coding: UTF-8
import os
import re
import string
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
CHINESE_PUNCTUATION = '，。！？；：、（）【】《》“”‘’—…·『』「」﹏￥·～-'
PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation + CHINESE_PUNCTUATION)


def preprocess_text(text, config=None):
    if config is None or not getattr(config, 'use_preprocessing', False):
        return text.strip()
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text.translate(PUNCTUATION_TABLE)


def get_tokenizer(ues_word):
    if ues_word:
        return lambda x: x.split(' ')
    return lambda x: [y for y in x]


def build_vocab(file_path, tokenizer, max_size, min_freq, config=None):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.rsplit('\t', 1)[0]
            content = preprocess_text(content, config)
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        # 将构建的词汇表按照降序排序
        vocab_list = sorted([_ for _ in vocab_dic.items()
                             if _[1] >= min_freq],
                            key=lambda x: x[1], reverse=True)[
                     :max_size]
        # 词汇表词典
        vocab_dic = {word_count[0]: idx for idx,
                    word_count in enumerate(vocab_list)}
        #  # 未知字，padding填充
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def load_dataset(path, vocab, config, tokenizer, pad_size=32):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.rsplit('\t', 1)
            processed_content = preprocess_text(content, config)
            words_line = []
            token = tokenizer(processed_content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] *
                                 (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(label), seq_len, content, processed_content))
    return contents


def build_dataset(config, ues_word):
    tokenizer = get_tokenizer(ues_word)
    vocab = build_vocab(config.train_path,
                        tokenizer=tokenizer,
                        max_size=MAX_VOCAB_SIZE, min_freq=1,
                        config=config)

    train = load_dataset(config.train_path, vocab, config, tokenizer, config.pad_size)
    dev = load_dataset(config.dev_path, vocab, config, tokenizer, config.pad_size)
    test = load_dataset(config.test_path, vocab, config, tokenizer, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = len(batches) % batch_size != 0
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2]
                                for _ in datas]).to(self.device)
        raw_texts = [_[3] for _ in datas]
        processed_texts = [_[4] for _ in datas]
        return (x, seq_len), y, raw_texts, processed_texts

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size
                                   : len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size:
                                   (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config, predict):
    if predict == True:
        config.batch_size = 1
    iter = DatasetIterater(dataset, config.batch_size,
                           config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
