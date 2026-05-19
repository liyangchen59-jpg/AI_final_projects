# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tqdm import tqdm


def write_misclassified(config, misclassified_records):
    with open(config.misclassified_path, 'w', encoding='utf-8') as f:
        for idx, record in enumerate(misclassified_records, 1):
            f.write(f"sample: {idx}\n")
            f.write(f"text: {record['text']}\n")
            f.write(f"processed_text: {record['processed_text']}\n")
            f.write(f"true_label: {record['true_label']}\n")
            f.write(f"pred_label: {record['pred_label']}\n")
            f.write(f"true_class: {record['true_class']}\n")
            f.write(f"pred_class: {record['pred_class']}\n")
            f.write("--------------------\n")


# 权重初始化，默认xavier
def init_network(model, method='xavier',
                 exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate)

    total_batch = 0
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]\nTraining:'.format(epoch + 1, config.num_epochs))
        model.train()
        for trains, labels, _, _ in tqdm(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_batch += 1

        true = labels.data.cpu()
        predic = torch.max(outputs.data, 1)[1].cpu()
        train_acc = metrics.accuracy_score(true, predic)
        print("Evaluating:")
        dev_acc, dev_loss = evaluate(config, model, dev_iter)
        time_dif = get_time_dif(start_time)
        msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  ' \
                'Train Acc: {2:>6.2%},  Val Loss: {3:>5.2}, ' \
                ' Val Acc: {4:>6.2%},  Time: {5}'
        print(msg.format(total_batch, loss.item(),
                            train_acc, dev_loss, dev_acc,
                            time_dif))

    print("Testing:")
    test(config, model, test_iter)


def test(config, model, test_iter):
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion, misclassified_records = \
        evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    write_misclassified(config, misclassified_records)
    print(f"Misclassified samples saved to: {config.misclassified_path}")
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    misclassified_records = []
    with torch.no_grad():
        for texts, labels, raw_texts, processed_texts in tqdm(data_iter):
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            if test:
                for raw_text, processed_text, label, pred in zip(raw_texts, processed_texts, labels, predic):
                    if label != pred:
                        misclassified_records.append({
                            'text': raw_text,
                            'processed_text': processed_text,
                            'true_label': int(label),
                            'pred_label': int(pred),
                            'true_class': config.class_list[int(label)],
                            'pred_class': config.class_list[int(pred)],
                        })

    acc = metrics.accuracy_score(labels_all, predict_all)
    avg_loss = loss_total / len(data_iter)
    if test:
        report = metrics.classification_report(labels_all,
                                               predict_all,
                            target_names=config.class_list,
                                               digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, avg_loss, report, confusion, misclassified_records
    return acc, avg_loss
