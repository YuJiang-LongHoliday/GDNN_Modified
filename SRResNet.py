# @Time    : 2024/6/10 15:09
# @Author  : YuJiang-LongHoliday
# @Project : SRResNet.py

#todo
# 注释：SRResNet,

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from PSNR import PSNR_Loss
from MS_SSIM_L1_Loss import MS_SSIM_L1_LOSS
from Binary_Loss import improved_binary_loss
from SSIM import SSIM
from torch.optim.lr_scheduler import StepLR  # 导入学习率调度器模块中的StepLR类，用于在训练过程中调整学习率
from torch.cuda.amp import GradScaler, autocast  # 导入混合精度训练模块中的GradScaler和autocast类，用于在训练中实现自动混合精度，以加速计算和减少显存占用
from torchvision.utils import save_image


Save_Image = r"F:\YuJiang\Python\Super_Resolution Microscopy by Grating and Deep Neural Network\Train_Image\SRResNet_624_3_Without_Grating"  # Save_Image：用于保存训练期间生成图像的目录。

class DenoiseModule(nn.Module):
    def __init__(self, in_channels):
        super(DenoiseModule, self).__init__()
        self.denoise = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 批量归一化层，作用于64个通道，有助于加速训练和提高稳定性
            nn.ReLU(inplace=True),
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
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        residual = x
        x = self.block(x)
        x += residual
        return x


class SRResNet(nn.Module):
    """优化后的SRResNet模型(4x)"""
    def __init__(self):
        super(SRResNet, self).__init__()
        self.denoise_module = DenoiseModule(1)
        self.conv_input = nn.Conv2d(64, 64, kernel_size=9, padding=4, padding_mode='reflect')
        self.relu = nn.PReLU()

        # 定义4个残差块
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(2)])

        self.conv_mid = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.BatchNorm2d(64)

        # 子像素卷积层实现上采样
        self.upsample1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

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

class CustomDataset(Dataset):
    def __init__(self, target_folder, degraded_folder, transform=None):
        self.target_folder = target_folder
        self.degraded_folder = degraded_folder
        self.transform = transform
        self.filenames = os.listdir(target_folder)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        target_img_path = os.path.join(self.target_folder, self.filenames[idx])
        degraded_img_path = os.path.join(self.degraded_folder, self.filenames[idx])

        # Convert the images to grayscale
        target_img = Image.open(target_img_path).convert('L')
        degraded_img = Image.open(degraded_img_path).convert('L')

        # Resize target_img to match the output size
        target_img = target_img.resize((128, 128), Image.BICUBIC)

        if self.transform:
            target_img = self.transform(target_img)
            degraded_img = self.transform(degraded_img)

        return degraded_img, target_img

# 参数
BATCH_SIZE = 8
EPOCHS = 2000
LR = 0.0001
WEIGHT_DECAY = 1e-8

SAVE_INTERVAL = 100

# 数据处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    #数据增强
    transforms.RandomHorizontalFlip(),
])

train_dataset = CustomDataset(
    degraded_folder=r"F:\YuJiang\Python\Super_Resolution Microscopy by Grating and Deep Neural Network\DataSets_128_64\624_3\Degraded_Image_without_Grating",
    target_folder=r"F:\YuJiang\Python\Super_Resolution Microscopy by Grating and Deep Neural Network\DataSets_128_64\624_3\Target_Image",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRResNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# 训练
for epoch in range(EPOCHS):
    # 在每个epoch开始时更新注意力权重
    model.train()
    running_loss = 0.0
    for degraded_imgs, target_imgs in train_loader:
        degraded_imgs, target_imgs = degraded_imgs.to(device), target_imgs.to(device)

        optimizer.zero_grad()
        outputs = model(degraded_imgs)

        # 将目标图像的大小调整为与输出图像的大小一致
        target_imgs = transforms.functional.resize(target_imgs, outputs.shape[2:])

        # # 打印输入和输出的形状
        # print(f'Degraded images shape: {degraded_imgs.shape}')
        # print(f'Target images shape: {target_imgs.shape}')
        # print(f'Output images shape: {outputs.shape}')
        # 1 - ssim是为了将它转换为损失，因为ssim的完美相似度得分是1
        #loss1 = SSIM()
        loss1 = MS_SSIM_L1_LOSS() #第一阶段训练效果最好,权重为z1000
        #loss1 = nn.MSELoss()
        #loss = psnr_loss(outputs,target_imgs)
        loss = loss1(outputs, target_imgs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * degraded_imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Fusion Loss: {epoch_loss:.4f}")

    # 每SAVE_INTERVAL次保存一次权重
    if (epoch + 1) % SAVE_INTERVAL == 0:
        #torch.save(model, "SRResNet_208_3_Without_Grating_{}.pth".format(epoch + 1))

        # 指定保存文件夹和文件名
        save_folder = r'F:\YuJiang\Python\Super_Resolution Microscopy by Grating and Deep Neural Network\Params'  # 替换为你要保存的文件夹路径
        save_filename = "SRResNet_624_3_Without_Grating_{}.pth".format(epoch + 1)
        save_path = os.path.join(save_folder, save_filename)

        # 确保保存文件夹存在
        os.makedirs(save_folder, exist_ok=True)

        # 保存模型
        torch.save(model, save_path)
        print(f"Saved model weights at epoch {epoch + 1}")

    # 保存训练时的生成图像
    _Image = degraded_imgs[0]  # 取出当前批次中的第一个输入图像
    _Segment_Image = target_imgs[0]  # 取出当前批次中的第一个标签图像
    _Out_Image = outputs[0]  # 取出当前批次中的第一个输出图像

    # 将输入图像和目标图像调整为与输出图像相同的大小
    _Image = transforms.functional.resize(_Image, _Out_Image.shape[1:])
    _Segment_Image = transforms.functional.resize(_Segment_Image, _Out_Image.shape[1:])

    img = torch.stack([_Image, _Segment_Image, _Out_Image], dim=0)  # 将输入图像、标签图像和输出图像堆叠在一起
    # torch.stack 函数用于沿着新的维度将一系列张量堆叠在一起
    # 由于 dim=0，新的维度将插入到第0维，结果张量 img 的形状为 (3, 3, 256, 256)。这里的每个元素代表：
    # 第0维度：3个不同的图像（输入图像、标签图像和输出图像）。
    # 第1维度：每个图像的通道数（3个通道，通常是RGB）。
    # 第2维度和第3维度：图像的高度和宽度（256x256像素）

    save_image(img, f'{Save_Image}/{epoch + 1}.png')  # 堆叠后的图像被保存到指定路径

print("Training Complete!")
