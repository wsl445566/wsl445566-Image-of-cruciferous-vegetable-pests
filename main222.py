
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np
import random
import os
import torchvision
from torch import optim
import modddd
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'


#设置随机种子
os.environ["CUDA_VISIBLE_DECICES"]="0,1"
def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True
   os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(10)

root = './data1000/'

# Hyper parameters
num_epochs =20  #循环次数
batch_size = 1         #每次投喂数据量
learning_rate = 0.0002  #学习率
momentum = 0.2      #变化率
num_classes = 10       #几分类
correct = 0
total = 0
correct1 = 0
total1 = 0
num_correct = 0
num_correct2 =0






class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, datatxt, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片 彩色图片则为RGB
        img = img.resize((224,224))
        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform
        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

# 根据自己定义的那个类MyDataset来创建数据集！注意是数据集！而不是loader迭代器
train_data = MyDataset(datatxt=root + 'train.txt', transform=transforms.ToTensor())
test_data = MyDataset(datatxt=root + 'test.txt', transform=transforms.ToTensor())

#然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = batch_size, shuffle=False)


model = modddd.ResNet50()
print(model)##这里的输出模型，这里没问题，下面都在运行了

# Device configuration  判断能否使用cuda加速
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = model.to(device)


criterion = nn.CrossEntropyLoss()
#criterion =CELoss()
optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,betas=(0.5,0.999))
#optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum = momentum )

train_losss = []
train_accc = []
test_losss = []
test_accc = []
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    train_acc = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = net(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        pred2 = torch.max(outputs, 1)[1]
        num_correct2 = (pred2 == labels).sum()
        train_acc += num_correct2.item()

        loss.backward()
        optimizer.step()
        #if (i + 1) % 190 == 0:
    print('Epoch [{}/{}],Train_Loss: {:.4f},Train_acc: {:.4f}'
          .format(epoch + 1, num_epochs, running_loss/len(train_data), train_acc/len(train_data)))
    train_accc.append(train_acc/len(train_data))
    train_losss.append(running_loss / len(train_data))

    net.eval()
    test_loss = 0.
    test_acc = 0.
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = net(batch_x)
            loss2 = criterion(out, batch_y)
            test_loss += loss2.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            test_acc += num_correct.item()
        test_accc.append(test_acc/len(test_data))
        test_losss.append(test_loss/len(test_data))
        print('Epoch :{}, Test Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, test_loss / (len(
            test_data)), test_acc / (len(test_data))))

iters = np.arange(1,num_epochs+1,1)
def draw_train_process(title, iters, label_cost):
    plt.figure()
    plt.legend(['Train Loss'])
    plt.title(title, fontsize=24)
    plt.plot(iters, label_cost, '-', label=label_cost)
    plt.ylabel(str(title),fontsize=24)
    plt.ylim(0, 1)
    my_y_ticks = np.arange(0, 1, 0.1)
    plt.yticks(my_y_ticks)
    plt.xlabel('Epoch Number')
    plt.show()
draw_train_process('Train_loss', iters, train_losss)

def draw_train_process(title, iters, label_cost):
    plt.figure()
    plt.legend(['Test Loss'])
    plt.title(title, fontsize=24)
    plt.plot(iters, label_cost, '-', label=label_cost)
    plt.ylabel(str(title),fontsize=24)
    plt.ylim(0, 1)
    my_y_ticks = np.arange(0, 1, 0.1)
    plt.yticks(my_y_ticks)
    plt.xlabel('Epoch Number')
    plt.show()
draw_train_process('Test_loss', iters, test_losss)
def draw_train_acc(title, iters, label_cost):
    plt.figure()

    plt.title(title, fontsize=24)
    plt.legend(['Train_acc'])
    plt.plot(iters, label_cost, '-', label=label_cost)
    plt.ylabel(str(title),fontsize=24)
    plt.ylim(0, 1)
    my_y_ticks = np.arange(0, 1, 0.1)
    plt.yticks(my_y_ticks)
    plt.xlabel('Epoch Number')
    plt.show()
draw_train_acc('Train_acc', iters, train_accc)

def draw_test_process(title, iters, label_cost):
    plt.figure()
    plt.legend(['Test_accuracy'])
    plt.title(title, fontsize=24)
    plt.plot(iters, label_cost, '-', label=label_cost)
    plt.ylabel(str(title),fontsize=24)
    plt.ylim(0, 1)
    my_y_ticks = np.arange(0, 1, 0.1)
    plt.yticks(my_y_ticks)
    plt.xlabel('Epoch Number')
    plt.show()

draw_test_process('Test_accuracy', iters, test_accc)

