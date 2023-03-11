import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split

# 数据增强
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(),
    transforms.GaussianBlur(3),
    transforms.RandomRotation(20),
    transforms.Resize((50, 50)),
    transforms.ColorJitter(0.2, ),

])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((50, 50))
])


# 加载数据集
dataset = ImageFolder(
    root='./haichong-10/train',
    transform=transform,
)

train_set, test_set = random_split(dataset, [0.8, 0.2])
