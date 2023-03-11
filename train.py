import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch import nn
from tqdm import tqdm
from data import train_set, test_set
from model import Model
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score
import pycm
from matplotlib import pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']


def get_performance(y_pred, y_true):
    y_pred = y_pred.argmax(axis=1)
    acc = accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
    recall = recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
    return acc, f1, recall


def plot(performance):
    # train set
    for k in performance:
        for p in performance[k]:
            plt.figure()
            plt.plot(range(len(performance[k][p])), performance[k][p])
            plt.title(f'{k}_{p}')
            plt.savefig(f'output/{k}_{p}.jpg')


def confusion_matrix(y_true, y_pred):
    class_name = os.listdir('haichong-10/train')
    cm = pycm.ConfusionMatrix(y_true, y_pred, classes=class_name)
    cm.plot(cmap=plt.cm.Blues, number_label=True)

    plt.savefig('output/cm.jpg')


def train():
    # 加载数据
    batch_size = 32
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # 定义超参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 10
    lr = 1e-3
    classes = 10
    model = Model(classes).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    performance = {
        'train': {
            'acc': [],
            'f1': [],
            'recall': []
        },
        'test': {
            'acc': [],
            'f1': [],
            'recall': []
        }
    }

    # 开始训练
    best_loss = float('inf')
    train_batch = 0
    test_batch = 0
    cm_true = []
    cm_pred = []
    for epoch in range(epochs):
        # 训练集
        loss_list = []
        acc_list = []
        model.train()
        train_iter = tqdm(enumerate(train_loader), total=len(train_loader))
        temp_model = model
        for index, (x_batch, y_batch) in train_iter:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward
            output = model(x_batch)
            loss = criterion(output, y_batch)

            # 计算模型性能
            if epoch == epochs - 1:
                cm_true.extend(y_batch.cpu().numpy())
                cm_pred.extend(output.argmax(axis=1).cpu().numpy())
            if train_batch % 20 == 0:
                acc, f1, recall = get_performance(output, y_batch)
                performance['train']['acc'].append(acc)
                performance['train']['f1'].append(f1)
                performance['train']['recall'].append(recall)
            train_batch += 1

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            acc = accuracy_score(y_batch.cpu().numpy(), output.argmax(axis=1).cpu().numpy())
            acc_list.append(acc)
            train_iter.desc = f'Epoch: {epoch}'
            train_iter.postfix = f'Loss: {loss.item():.3f} Acc: {acc}'

        print(f'Epoch: {epoch} / {epochs} TrainSet: loss: {np.mean(loss_list):.3f} Acc: {np.mean(acc_list)}')

        # 测试集
        loss_list = []
        acc_list = []
        temp_model.eval()
        test_iter = tqdm(enumerate(test_loader), total=len(test_loader))
        for index, (x_batch, y_batch) in test_iter:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward
            output = temp_model(x_batch)
            loss = criterion(output, y_batch)

            # 计算模型性能
            if test_batch % 10 == 0:
                acc, f1, recall = get_performance(output, y_batch)
                performance['test']['acc'].append(acc)
                performance['test']['f1'].append(f1)
                performance['test']['recall'].append(recall)
            test_batch += 1

            loss_list.append(loss.item())
            acc = accuracy_score(y_batch.cpu().numpy(), output.argmax(axis=1).cpu().numpy())
            acc_list.append(acc)
            test_iter.desc = f'Epoch: {epoch} '
            test_iter.postfix = f'Loss: {loss.item():.3f} Acc: {acc}'

        if np.mean(loss_list) < best_loss:
            torch.save(model, 'model.pth')
        print(f'Epoch: {epoch} / {epochs} TestSet: loss: {np.mean(loss_list):.3f} Acc: {np.mean(acc_list)}')

    # 保存模型评估指标
    torch.save(performance, 'output/performance.pkl')
    torch.save([cm_true, cm_pred], 'output/target.pkl')

    # 训练结束，对结果进行可视化
    plot(performance)
    confusion_matrix(cm_true, cm_pred)


if __name__ == '__main__':
    train()
