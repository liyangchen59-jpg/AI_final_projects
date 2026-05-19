# coding:utf-8

import torch
import numpy as np
from utils import PAD, get_tokenizer, preprocess_text


def load_dataset(text, vocab, config, ues_word=False):
    contents = []
    tokenizer = get_tokenizer(ues_word)
    for line in text:
        lin = line.strip()
        if not lin:
            continue
        processed_content = preprocess_text(lin, config)
        words_line = []
        token = tokenizer(processed_content)
        seq_len = len(token)
        if config.pad_size:
            if len(token) < config.pad_size:
                token.extend([PAD] * (config.pad_size - len(token)))
            else:
                token = token[:config.pad_size]
                seq_len = config.pad_size
        for word in token:
            words_line.append(vocab.get(word, vocab.get('<UNK>')))

        contents.append((words_line, int(0), seq_len, lin, processed_content))
    return contents


def match_label(pred, config):
    label_list = config.class_list
    return label_list[pred]


def final_predict(config, model, data_iter):
    model.eval()
    predict_all = np.array([])
    with torch.no_grad():
        for texts, _, _, _ in data_iter:
            outputs = model(texts)
            pred = torch.max(outputs.data, 1)[1].cpu().numpy()
            pred_label = [match_label(i, config) for i in pred]
            predict_all = np.append(predict_all, pred_label)
    return predict_all
