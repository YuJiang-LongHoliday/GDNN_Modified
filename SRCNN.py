# @Time    : 2024/6/8 17:35
# @Author  : YuJiang-LongHoliday
# @Project : SRCNN.py

#todo
# 注释：SRCNN,超分辨率卷积神经网络,图像超分辨率是指提高图像分辨率，这意味着增加图像的尺寸并添加更多细节。
# SRCNN 通过一个三层卷积神经网络来实现这一目标，该网络学习如何将低分辨率图像映射到高分辨率图像。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, target_folder, degraded_folder, transform=None):  # target_folder：存储高分辨率目标图像的文件夹路径
        self.target_folder = target_folder                               # degraded_folder：存储低分辨率退化图像的文件夹路径
        self.degraded_folder = degraded_folder
        self.transform = transform
        self.filenames = os.listdir(target_folder)  # 获取目标文件夹中所有文件的文件名列表，并将其存储在 self.filenames 中

    def __len__(self):
        return len(self.filenames)

    # 获取数据集元素
    def __getitem__(self, idx):
        target_img_path = os.path.join(self.target_folder, self.filenames[idx])
        degraded_img_path = os.path.join(self.degraded_folder, self.filenames[idx])

        target_img = Image.open(target_img_path).convert('L')  # 转为灰度图像（'L' 表示灰度模式）
        degraded_img = Image.open(degraded_img_path).convert('L')

        if self.transform:
            target_img = self.transform(target_img)
            degraded_img = self.transform(degraded_img)

        return degraded_img, target_img

# 训练参数和数据加载
BATCH_SIZE = 8
EPOCHS = 2000
LR = 0.0001
WEIGHT_DECAY = 1e-8  # 权重衰减：一种正则化技术，用于防止过拟合。它在每次更新时对权重施加一个小的惩罚。
SAVE_INTERVAL = 100  # 保存模型权重的间隔，即每训练 200 个 epoch 保存一次模型权重

# 数据转换
transform = transforms.Compose([  # transforms.Compose：将多个图像变换操作组合在一起。
    transforms.ToTensor(),  # transforms.ToTensor()：将图像从 PIL Image 或 numpy.ndarray 转换为形状为 (C,H,W) 的张量，并将像素值从 [0, 255] 缩放到 [0, 1]。
    transforms.RandomHorizontalFlip(),  # 以 50% 的概率对图像进行水平翻转，这是一种数据增强方法，可以帮助模型更好地泛化。
])

# 自定义数据集的实例化
train_dataset = CustomDataset(
    degraded_folder=r"F:\Python\Super_Resolution Microscopy by Grating and Deep Neural Network\DataSets\208_3\Degraded_Image_without_Grating",
    target_folder=r"F:\Python\Super_Resolution Microscopy by Grating and Deep Neural Network\DataSets\208_3\Target_Image",
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # model.parameters传递给优化器的参数是模型的所有参数，这些参数会在训练过程中被更新。

# # 训练模型
# for epoch in range(EPOCHS):  # 遍历所有训练轮次，从 0 到 EPOCHS-1
#     model.train()  # 将模型设置为训练模式。这将启用 dropout 和 batch normalization 的训练行为。
#     running_loss = 0.0
#     for degraded_imgs, target_imgs in train_loader:
#         degraded_imgs, target_imgs = degraded_imgs.to(device), target_imgs.to(device)
#
#         optimizer.zero_grad()  # 清零优化器中的所有梯度，以避免梯度累积
#
#         outputs = model(degraded_imgs)  # 通过模型进行前向传播
#
#         loss1 = nn.MSELoss()
#
#         loss = loss1(outputs, target_imgs)  # 计算模型输出与目标图像之间的损失
#         loss.requires_grad_(True)  # 确保损失具有梯度信息，以便进行反向传播
#
#         loss.backward()  # 计算损失相对于模型参数的梯度
#         optimizer.step()  # 更新模型参数
#
#         # degraded_imgs.size(0)：获取当前批次的图像数量，以便正确累加
#         running_loss += loss.item() * degraded_imgs.size(0)
#
#     epoch_loss = running_loss / len(train_loader.dataset)
#     print(f"Epoch {epoch + 1}/{EPOCHS} - MSE Loss: {epoch_loss:.4f}")
#
#     # 每SAVE_INTERVAL次保存一次权重
#     if (epoch + 1) % SAVE_INTERVAL == 0:
#         torch.save(model, "SRCNN_Net_208_3_With_Grating_{}.pth".format(epoch))  # 模型训练时保存的是整个模型对象而不是整个模型参数
#         print(f"Saved model weights at epoch {epoch + 1}")
#
# print("Training Complete!")



