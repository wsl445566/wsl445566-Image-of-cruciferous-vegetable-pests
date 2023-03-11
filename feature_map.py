import matplotlib.pyplot as plt
import os

import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image, make_grid

# 加载模型
model = torch.load('model.pth')
# print(model)
# 选择要使用的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

FEATURE_FOLDER = "./output/features"
if not os.path.exists(FEATURE_FOLDER):
    os.mkdir(FEATURE_FOLDER)

feature_list = list()
count = 0
idx = 0


def get_image_path_for_hook(module):
    global count
    image_name = feature_list[count] + ".png"
    count += 1
    image_path = os.path.join(FEATURE_FOLDER, image_name)
    return image_path


def hook_func(module, input, output):
    image_path = get_image_path_for_hook(module)
    data = output.clone().detach()
    global idx
    idx += 1
    data = data.data.permute(1, 0, 2, 3)
    data = make_grid(data)
    # data = plt.cm.viridis(data.cpu().numpy().transpose(1, 2, 0))
    data = plt.cm.viridis(data[0].cpu().numpy())
    plt.imshow(data)
    plt.savefig(image_path)


for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        feature_list.append(name)
        module.register_forward_hook(hook_func)

# 加载图片
img = Image.open('haichong-10/train/小地老虎/0.jpg')
to_tensor = transforms.PILToTensor()
img = to_tensor(img) / 255.0
img = torch.unsqueeze(img, 0).to(device)

# 运行模型并获取特征图
res = model(img)
