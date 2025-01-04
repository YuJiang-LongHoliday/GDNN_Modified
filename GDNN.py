# @Time    : 2024/6/10 22:00
# @Author  : YuJiang-LongHoliday
# @Project : GDNN.py
# @Function:

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from SSIM import SSIM
from torch.optim.lr_scheduler import StepLR  # 导入学习率调度器，可以逐步调整学习率。
from torch.cuda.amp import GradScaler, autocast  # 导入 AMP（自动混合精度）的 GradScaler 和 autocast，用于加速训练并减少显存使用，特别是在 GPU 上进行训练时。
from torchvision.utils import save_image

Save_Image = r'F:\Python\Super_Resolution Microscopy by Grating and Deep Neural Network\Train_Image\410nm\Polarization\With_Grating\300nm\Unfocus\410nm_Polarization_With_Grating_300_Unfocus'  # Save_Image：用于保存训练期间生成图像的目录。

# 实现了一个空间注意力机制（Spatial Attention），用于图像处理任务中的卷积神经网络
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # 添加可训练的权重参数
        self.attention_weight = nn.Parameter(torch.zeros(1), requires_grad=True)  # # 定义一个可训练的注意力权重参数，初始值为 0，允许梯度更新

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入张量 x 在通道维度（dim=1）进行平均池化，结果保留维度（keepdim=True）
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入张量 x 在通道维度（dim=1）进行最大池化，结果保留维度（keepdim=True）
        attention = torch.cat([avg_out, max_out], dim=1)  # 将平均池化结果和最大池化结果在通道维度上拼接，生成一个新的张量，形状为 (batch_size, 2, height, width)
        attention = self.conv1(attention)  # 将拼接后的张量通过卷积层，输出一个单通道的注意力图
        # 使用权重调整注意力的强度
        attention = self.sigmoid(attention) * torch.sigmoid(self.attention_weight)  # 对卷积输出的注意力图应用 sigmoid 激活函数，同时乘以注意力权重（通过 sigmoid 激活），以调整注意力的强度
        return x * attention  # 将输入张量 x 与注意力图逐元素相乘，返回加权后的张量


# 实现了一个通道注意力机制（Channel Attention），用于图像处理任务中的卷积神经网络
class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):  # num_channels：输入特征图的通道数,reduction_ratio=8：通道缩减比率，用于减少参数量和计算量，默认为 8
        super(ChannelAttention, self).__init__()  # 调用父类 nn.Module 的构造函数，以确保正确初始化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 定义一个自适应平均池化层，输出大小为 1×1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 定义一个自适应最大池化层，输出大小为 1×1
        self.fc = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # 注意力权重
        self.attention_weight = nn.Parameter(torch.zeros(1),requires_grad=True)  # 定义一个可训练的注意力权重参数，初始值为 0，允许梯度更新

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # 对输入张量 x 进行自适应平均池化，输出大小为 1×1，然后通过全连接层序列，生成通道注意力图
        max_out = self.fc(self.max_pool(x))  # 对输入张量 x 进行自适应最大池化，输出大小为 1×1，然后通过全连接层序列，生成通道注意力图
        out = avg_out + max_out  # 将平均池化结果和最大池化结果相加，融合这两种信息
        scale = self.sigmoid(out)*torch.sigmoid(self.attention_weight)  # 对融合后的注意力图应用 sigmoid 激活函数，同时乘以注意力权重（通过 sigmoid 激活），以调整注意力的强度
        return x*scale


# 实现了一个残差块（Residual Block），并且可以选择性地使用通道注意力和空间注意力机制
# 注意力机制能够自适应地调整特征图的通道和空间权重，提升模型性能
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True):
        super(ResidualBlock, self).__init__()

        self.use_attention = use_attention
        if self.use_attention:
            self.channel_attention = ChannelAttention(out_channels)
            self.spatial_attention = SpatialAttention()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)  # 改变特征图的通道数从 in_channels 到 out_channels
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活函数，inplace=True 表示直接在输入上进行操作，节省内存
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.1)  # Dropout 层，随机丢弃 10% 的神经元，用于防止过拟合

        self.skip = nn.Sequential()  # 定义跳跃连接，默认情况下为空的顺序容器
        if stride != 1 or in_channels != out_channels:  # 如果步幅不为 1 或输入和输出通道数不相等，则在跳跃连接中添加一个 1×1 的卷积层和批归一化层，以匹配输入和输出的维度
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_attention:
            out = self.channel_attention(out)
            out = self.spatial_attention(out)

        out += residual  # 将卷积输出与残差项相加，完成跳跃连接
        out = self.relu(out)

        return out

class UpSampleBlock(nn.Module):
    """上采样块，用于解码器"""
    def __init__(self, in_channels, out_channels,use_attention=True):
        super(UpSampleBlock, self).__init__()
        self.use_attention = use_attention
        if self.use_attention:  # 如果 use_attention 为 True，则实例化通道注意力机制（Channel Attention）和空间注意力机制（Spatial Attention）
            self.channel_attention = ChannelAttention(out_channels)
            self.spatial_attention = SpatialAttention()
        self.up_sample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

    def forward(self, x):
        out = self.up_sample(x)
        if self.use_attention:
            out = self.channel_attention(out)
            out = self.spatial_attention(out)
        return out

# 自定义神经网路
class SimpleResNet(nn.Module):
    """带有注意力机制的对称编解码器结构"""
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, 16),
            nn.MaxPool2d(2, 2),
            ResidualBlock(16, 32),
            nn.MaxPool2d(2, 2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, 2),
        )

        self.decoder = nn.Sequential(
            UpSampleBlock(64, 32),
            UpSampleBlock(32, 16),
            UpSampleBlock(16, 8),

        )

        self.final_conv = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        x = nn.functional.interpolate(x, size=(x.shape[2], x.shape[3]), mode='bilinear',align_corners=False)  # 强制调整输出尺寸
        x = self.activation(x)
        return x

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, target_folder, degraded_folder, transform=None, target_size=(128, 128)):  # 记得要改目标图像！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        self.target_folder = target_folder
        self.degraded_folder = degraded_folder
        self.transform = transform
        self.target_size = target_size
        self.filenames = os.listdir(target_folder)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        target_img_path = os.path.join(self.target_folder, self.filenames[idx])
        degraded_img_path = os.path.join(self.degraded_folder, self.filenames[idx])

        # Convert the images to grayscale
        target_img = Image.open(target_img_path).convert('L')
        degraded_img = Image.open(degraded_img_path).convert('L')

        # Resize target image to match model output size
        target_img = target_img.resize(self.target_size, Image.BILINEAR)
        degraded_img = degraded_img.resize(self.target_size, Image.BILINEAR)

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
    # 数据增强
    transforms.RandomHorizontalFlip(),
])

train_dataset = CustomDataset(
    degraded_folder=r"F:\Python\Super_Resolution Microscopy by Grating and Deep Neural Network\DataSets_128_128\Degrade_Image\410nm\Polarization\With_Grating\Unfocus\300nm\1040",
    target_folder= r"F:\Python\Super_Resolution Microscopy by Grating and Deep Neural Network\DataSets_128_128\Target_Image\1040",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleResNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def update_attention_weights(epoch, model, init_weight=0.0, final_weight=1.0, total_epochs=2000):
    """
    根据当前epoch渐进式增加注意力权重。

    参数:
    - epoch: 当前的epoch。
    - model: PyTorch模型。
    - init_weight: 注意力权重的初始值。
    - final_weight: 注意力权重的最终值。
    - total_epochs: 总的训练epochs。
    """
    # 计算当前权重，线性从init_weight增加到final_weight
    current_weight = (final_weight - init_weight) * (epoch / total_epochs) + init_weight
    current_weight = min(current_weight, final_weight)  # 确保不超过final_weight

    # 遍历模型中的所有模块，更新注意力权重
    for module in model.modules():
        if isinstance(module, (ChannelAttention, SpatialAttention)):
            module.attention_weight.data.fill_(current_weight)


# 训练模型
for epoch in range(EPOCHS):
    # 在每个epoch开始时更新注意力权重
    update_attention_weights(epoch, model,init_weight=0.0, final_weight=1.0, total_epochs=EPOCHS)
    model.train()
    running_loss = 0.0
    for degraded_imgs, target_imgs in train_loader:
        degraded_imgs, target_imgs = degraded_imgs.to(device), target_imgs.to(device)

        optimizer.zero_grad()

        outputs = model(degraded_imgs)

        # 1 - ssim是为了将它转换为损失，因为ssim的完美相似度得分是1
        lloss = SSIM()
        loss = 1 - lloss(outputs, target_imgs)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * degraded_imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS} - SSIM Loss: {epoch_loss:.4f}")

    # 每SAVE_INTERVAL次保存一次权重
    if (epoch + 1) % SAVE_INTERVAL == 0:
        # torch.save(model, "Attention_Net_208_3_With_2_Grating_{}.pth".format(epoch + 1))

        # 指定保存文件夹和文件名
        save_folder = r'F:\YuJiang\Python\Super_Resolution Microscopy by Grating and Deep Neural Network\Params\640nm\Unpolarization\With_Grating\900nm\Unfocus'  # 替换为你要保存的文件夹路径
        save_filename = "Attention_Net_640nm_Unpolarization_With_Grating_900nm_Unfocus_{}.pth".format(epoch + 1)
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

    img = torch.stack([_Image, _Segment_Image, _Out_Image], dim=0)  # 将输入图像、标签图像和输出图像堆叠在一起
    # torch.stack 函数用于沿着新的维度将一系列张量堆叠在一起
    # 由于 dim=0，新的维度将插入到第0维，结果张量 img 的形状为 (3, 3, 256, 256)。这里的每个元素代表：
    # 第0维度：3个不同的图像（输入图像、标签图像和输出图像）。
    # 第1维度：每个图像的通道数（3个通道，通常是RGB）。
    # 第2维度和第3维度：图像的高度和宽度（256x256像素）

    save_image(img, f'{Save_Image}/{epoch + 1}.png')  # 堆叠后的图像被保存到指定路径

print("Training Complete!")