# coding: UTF-8
import os
import time
import torch
import numpy as np
from importlib import import_module
from train import train, init_network
from predict import load_dataset, final_predict
from utils import build_dataset, build_iterator, get_time_dif


class TransformerPredict:
    def __init__(self, config, model, vocab):
        self.config = config
        self.model = model
        self.vocab = vocab

    def predict(self, text):
        content = load_dataset(text, self.vocab, self.config)
        predict_iter = build_iterator(content, self.config, predict=True)
        self.config.n_vocab = len(self.vocab)

        result = final_predict(self.config, self.model, predict_iter)
        for i, j in enumerate(result):
            print('text:{}'.format(text[i]), '\t', 'label:{}'.format(j))


if __name__ == '__main__':
    dataset = 'THUCNews'
    embedding = 'random'
    model_name = 'Transformer'
    task_name = os.environ.get('TASK_NAME', 'A').upper()
    if task_name not in {'A', 'B'}:
        raise ValueError('TASK_NAME must be A or B')

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    config.task_name = task_name
    config.use_preprocessing = task_name == 'B'
    config.misclassified_path = f"{dataset}/data/task{task_name}.misclassified"

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print(f"Running Task {task_name}")
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, False)
    train_iter = build_iterator(train_data, config, False)
    dev_iter = build_iterator(dev_data, config, False)
    test_iter = build_iterator(test_data, config, False)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    train(config, model, train_iter, dev_iter, test_iter)

    tp = TransformerPredict(config, model, vocab)
    test = ['北京冬奥会所有竞赛项目都已圆满结束，'
            '中国代表团最终以9金4银2铜的成绩位列奖牌榜第三位。'
            '本届冬奥会，中国代表团的金牌数、'
            '奖牌数和奖牌榜排名均创造了我们1980年参加冬奥会以来的最佳成绩',
            '自然·天文学》在线发表了一项月球科学研究的重要成果。'
            '利用嫦娥五号月球“土特产”的同位素年龄和着陆区撞击坑统计结果，'
            '中科院空天信息创新研究院等单位的研究人员在目前常用月球年代函数的'
            '基础上建立了新的更精确的月球年代函数模型',
            '近日，教育部印发了《幼儿园保育教育质量评估指南》'
            '（以下简称《评估指南》）。'
            '教育部基础教育司负责人就《评估指南》有关内容回答了记者提问',
            '张涵予，1964年出生于北京，籍贯陕西蓝田，是中国内地男演员，'
            '1988年毕业于中央戏剧学院表演系，中国文学艺术界联合会第十届全委会委员，'
            '中国电影家协会副主席。在部队大院长大，读高中时，'
            '就到中央电视台译制片组给外国影片配音，1983年，张涵予成为专业配音演员',
            '甲醇期货今日挂牌上市继上半年焦炭、铅期货上市后，'
            '酝酿已久的甲醇期货将在今日正式挂牌交易。'
            '基准价均为3050元／吨继上半年焦炭、铅期货上市后，'
            '酝酿已久的甲醇期货将在今日正式挂牌交易。']
    tp.predict(test)
