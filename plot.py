import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

performance = torch.load('output/performance.pkl')
train_acc = performance['train']['acc']
test_acc = performance['test']['acc']

act_train_acc = [0]
act_test_acc = [0]
for i in range(0, len(train_acc), 10):
    act_train_acc.append(np.mean([train_acc[i+j] for j in range(10)]))

for i in range(0, len(test_acc), 5):
    act_test_acc.append(np.mean([train_acc[i+j] for j in range(5)]))

plt.figure()
plt.plot(range(len(act_train_acc)), act_train_acc, label='train')
plt.plot(range(len(act_test_acc)), act_test_acc, label='test')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.savefig('output/acc.jpg')
plt.show()


train_f1 = performance['train']['f1']
test_f1 = performance['test']['f1']

act_train_f1 = [0]
act_test_f1 = [0]
for i in range(0, len(train_f1), 10):
    act_train_f1.append(np.mean([train_f1[i+j] for j in range(10)]))

for i in range(0, len(test_f1), 5):
    act_test_f1.append(np.mean([test_f1[i+j] for j in range(5)]))

plt.plot(range(len(act_train_f1)), act_train_f1, label='train')
plt.plot(range(len(act_test_f1)), act_test_f1, label='test')
plt.xlabel('epoch')
plt.ylabel('f1-score')
plt.legend()
plt.savefig('output/f1.jpg')
plt.show()


train_recall = performance['train']['recall']
test_recall = performance['test']['recall']

act_train_recall = [0]
act_test_recall = [0]
for i in range(0, len(train_recall), 10):
    act_train_recall.append(np.mean([train_recall[i+j] for j in range(10)]))

for i in range(0, len(test_recall), 5):
    act_test_recall.append(np.mean([test_recall[i+j] for j in range(5)]))

plt.plot(range(len(act_train_recall)), act_train_recall, label='train')
plt.plot(range(len(act_test_recall)), act_test_recall, label='test')
plt.xlabel('epoch')
plt.ylabel('recall')
plt.legend()
plt.savefig('output/recall.jpg')
plt.show()


print(f'Train: max acc: {max(act_train_acc)}')

