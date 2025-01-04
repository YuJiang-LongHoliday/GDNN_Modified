# @Time    : 2024/6/9 21:16
# @Author  : YuJiang-LongHoliday
# @Project : SRGAN.py

#todo
# 注释：SRGAN 是在 SRResNet 基础上引入生成对抗网络（GAN）框架的超分辨率模型。
# GAN 由一个生成器（Generator）和一个判别器（Discriminator）组成，生成器的目标是生成逼真的高分辨率图像(生成器部分使用 SRResNet 结构)，而判别器的目标是区分真实图像和生成图像。

import torch  # 导入 PyTorch 主库，用于处理张量操作和构建神经网络
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，用于构建网络层
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # 导入 torchvision 的 transforms 模块，用于数据预处理和增强。
from PIL import Image
import os
from MS_SSIM_L1_Loss import MS_SSIM_L1_LOSS
from Binary_Loss import improved_binary_loss
from SSIM import SSIM
from torch.optim.lr_scheduler import StepLR  # 导入 PyTorch 的学习率调度器，用于在训练过程中调整学习率。
from torch.cuda.amp import GradScaler, autocast  # 导入 PyTorch 的学习率调度器，用于在训练过程中调整学习率。

class DenoiseModule(nn.Module):
    def __init__(self, in_channels):  # 接受一个参数 in_channels，表示输入图像的通道数
        super(DenoiseModule, self).__init__()
        self.denoise = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 批归一化有助于加速训练，并稳定模型。
            nn.ReLU(inplace=True),  # inplace=True 表示将直接修改输入，节省内存
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.denoise(x)


class ResBlock(nn.Module):
    """优化后的残差模块"""
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),  # PReLU（Parametric ReLU）激活函数，具有可学习参数的 ReLU 函数
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        residual = x  # 保存输入 x，作为残差连接的部分
        x = self.block(x)  # 将输入 x 传递给 self.block，即通过定义好的 nn.Sequential 容器按顺序执行所有子模块，得到中间结果 x
        x += residual  # 将中间结果 x 与输入 residual 相加，实现残差连接。这一步引入了残差学习机制，缓解深层神经网络中的梯度消失问题。
        return x

class SRResNet(nn.Module):
    """优化后的SRResNet模型(4x)"""
    def __init__(self):
        super(SRResNet, self).__init__()
        self.denoise_module = DenoiseModule(1)  # 使用之前定义的 DenoiseModule，初始化时指定输入通道数为 1（假设输入图像是灰度图像）
        self.conv_input = nn.Conv2d(64, 64, kernel_size=9, padding=4, padding_mode='reflect')
        self.relu = nn.PReLU()

        # todo 定义3个残差块, 此处与论文中有差异
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(3)])  # note:定义多少残差块

        # 定义中间卷积层和批归一化层
        self.conv_mid = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.BatchNorm2d(64)

        # 子像素卷积层实现上采样
        self.upsample1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2),  # nn.PixelShuffle(2)：像素重排层，将通道数减小 4 倍，同时将空间分辨率提高 2 倍
            nn.PReLU()
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        # 定义输出卷积层
        self.conv_output = nn.Conv2d(64, 1, kernel_size=9, padding=4, padding_mode='reflect')

    def forward(self, x):
        x = self.denoise_module(x)
        x = self.relu(self.conv_input(x))
        residual = x

        x = self.res_blocks(x)

        x = self.bn_mid(self.conv_mid(x))
        x += residual  # 应用残差连接

        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.conv_output(x)
        return x



# 这个判别器模型的主要功能是接收一个输入图像（如 64x64 像素的灰度图像），并通过一系列卷积、批归一化和激活函数操作，输出一个概率值，表示输入图像是真实的（非生成的）概率
class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            # 输入尺寸: [input_channels x 64 x 64]
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # todo 与论文中的模型有差异
            nn.AdaptiveAvgPool2d(1),  # nn.AdaptiveAvgPool2d(1) 将特征图的大小调整为 1x1
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
            # 输出尺寸: [1]
        )

    def forward(self, x):
        return self.network(x)

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, target_folder, degraded_folder, transform=None):
        self.target_folder = target_folder
        self.degraded_folder = degraded_folder
        self.transform = transform  # 存储图像转换操作
        self.filenames = os.listdir(target_folder)  # self.filenames 存储目标图像文件夹中的文件名列表

    # __len__ 方法返回数据集的长度，即文件名列表的长度，表示数据集中图像的数量
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # target_img_path 和 degraded_img_path 分别获取目标图像和降质图像的完整路径。
        target_img_path = os.path.join(self.target_folder, self.filenames[idx])
        degraded_img_path = os.path.join(self.degraded_folder, self.filenames[idx])

        # 使用 PIL 库的 Image.open 方法打开图像，并通过 convert('L') 方法将其转换为灰度图像。
        target_img = Image.open(target_img_path).convert('L')
        degraded_img = Image.open(degraded_img_path).convert('L')

        if self.transform:
            target_img = self.transform(target_img)
            degraded_img = self.transform(degraded_img)

        return degraded_img, target_img

BATCH_SIZE = 8
EPOCHS = 400
LR = 0.001
WEIGHT_DECAY = 1e-5
SAVE_INTERVAL = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化生成器
model = SRResNet().to(device)

# 假设Discriminator类已经定义
discriminator = Discriminator().to(device)

# 数据处理和加载
transform = transforms.Compose([
    # 将图像从 PIL 图像或者 NumPy 数组转换为 PyTorch 的张量 (Tensor)。
    # 转换后的张量会自动将图像的像素值归一化到 [0, 1] 范围内
    # 同时，这个转换还会将图像的通道顺序从 HWC (Height, Width, Channels) 转换为 CHW (Channels, Height, Width)，这与 PyTorch 张量的默认格式一致。
    transforms.ToTensor(),
    # 数据增强
    # 这个操作会随机地水平翻转图像（左右翻转），以增强模型的泛化能力。
    # 随机水平翻转的概率是 0.5，即每次加载图像时，有 50% 的概率会对图像进行水平翻转
    transforms.RandomHorizontalFlip(),
])

train_dataset = CustomDataset(
    degraded_folder = "F:\\Python\\Super_Resolution Microscopy by Grating and Deep Neural Network\\DataSets\\SRCNN_DataSets\\archive\\sr_tumor_imgs",
    target_folder="F:\\Python\\Super_Resolution Microscopy by Grating and Deep Neural Network\\DataSets\\SRCNN_DataSets\\archive\\hr_tumor_imgs",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化生成器和判别器的优化器
optimizer_G = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# 为判别器和生成器定义损失函数
criterion_GAN = nn.BCELoss()  # nn.BCELoss() 是 PyTorch 中的二元交叉熵损失函数，用于二分类任务。
# BCELoss（Binary Cross Entropy Loss）是用于计算真实标签和预测标签之间的误差的标准损失函数之一。
criterion_content = MS_SSIM_L1_LOSS()


# 训练
for epoch in range(EPOCHS):
    model.train()  # 设置生成器为训练模式
    discriminator.train()  # 设置判别器为训练模式
    running_loss_G = 0.0
    running_loss_D = 0.0

    for degraded_imgs, target_imgs in train_loader:  # 通过数据加载器 train_loader 迭代加载每批次的退化图像和目标图像
        degraded_imgs, target_imgs = degraded_imgs.to(device), target_imgs.to(device)  # 将加载的图像数据移动到指定设备（如 GPU）上

        valid = torch.ones((degraded_imgs.size(0), 1), requires_grad=False).to(device)  # 创建一个全为 1 的张量，表示真实图像的标签
        fake = torch.zeros((degraded_imgs.size(0), 1), requires_grad=False).to(device)  # 创建一个全为 0 的张量，表示生成图像的标签

        # -----------------
        #  训练判别器
        # -----------------
        optimizer_D.zero_grad()

        # 训练真实图像
        real_output = discriminator(target_imgs)  # 通过判别器对真实图像进行前向传播，获得输出
        real_output_squeeze = torch.squeeze(real_output, -1)  # 去掉最后一个维度的大小为 1 的维度
        real_output_squeeze = torch.squeeze(real_output_squeeze, -1)  # 再次去掉一个大小为 1 的维度
        real_loss = criterion_GAN(real_output_squeeze, valid)  # 计算真实图像的损失

        # 训练生成的图像
        fake_imgs = model(degraded_imgs).detach()  # 通过生成器生成假图像，并使用 detach() 方法防止梯度传播到生成器
        fake_output = discriminator(fake_imgs)   # 通过判别器对假图像进行前向传播，获得输出
        fake_output_squeezed = torch.squeeze(fake_output, -1)  # 去除多余的维度
        fake_output_squeezed = torch.squeeze(fake_output_squeezed, -1)
        fake_loss = criterion_GAN(fake_output_squeezed, fake)  # 计算生成图像的损失

        # 判别器总损失
        loss_D = (real_loss + fake_loss) / 2
        loss_D.backward()  # 反向传播，计算梯度
        optimizer_D.step()  # 更新判别器的参数

        running_loss_D += loss_D.item()  # 累加判别器的损失

        # -----------------
        #  训练生成器
        # -----------------
        optimizer_G.zero_grad()  # 清零生成器的梯度

        # 生成器的目标是骗过判别器
        # print(discriminator(fake_output_squeezed).size())  # 在训练生成器部分中打印经过判别器后的假图像输出的尺寸
        fake_imgs = model(degraded_imgs)  # 通过生成器生成假图像
        g_loss = criterion_GAN(discriminator(fake_output_squeezed), valid)  # 计算生成器的对抗损失，即希望判别器认为生成的假图像是真实的

        # 内容损失
        content_loss = criterion_content(fake_imgs, target_imgs)  # 计算生成图像和真实图像之间的内容损失

        # 生成器总损失
        loss_G = g_loss + content_loss  # 计算生成器的总损失
        loss_G.backward()
        optimizer_G.step()

        running_loss_G += loss_G.item()  # 累加生成器的损失

    epoch_loss_G = running_loss_G / len(train_loader.dataset)  # 计算一个 epoch 中生成器的平均损失
    epoch_loss_D = running_loss_D / len(train_loader.dataset)  # 计算一个 epoch 中判别器的平均损失
    print(f"Epoch {epoch + 1}/{EPOCHS} - Generator Loss: {epoch_loss_G:.4f} - Discriminator Loss: {epoch_loss_D:.4f}")  # 打印当前 epoch 的生成器和判别器的损失

    # 每SAVE_INTERVAL次保存一次权重
    if (epoch + 1) % SAVE_INTERVAL == 0:
        torch.save(model.state_dict(), "SRGAN_Generator_w{epoch + 1}.pth")  # 保存生成器的权重
        torch.save(discriminator.state_dict(), "SRGAN_Discriminator_w{epoch + 1}.pth")  # 保存判别器的权重
        print(f"Saved model weights at epoch {epoch + 1}")
