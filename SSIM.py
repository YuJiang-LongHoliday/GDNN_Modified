# @Time    : 2024/6/6 20:21
# @Author  : YuJiang-LongHoliday
# @Project : SSIM.py

#todo
# 注释：SSIM,Structural Similarity Index,一种衡量两张图片之间相似度的指标。
# SSIM的计算基于滑动窗口实现，即每次计算均从图片上取一个尺寸为N × N的窗口，基于窗口计算SSIM指标，遍历整张图像后再将所有窗口的数值取平均值，作为整张图像的SSIM指标。

import torch
import torch.nn.functional as F
from math import exp
import numpy as np
from torchvision import transforms
import cv2

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
                         dtype=torch.float32)
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window.cuda()


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel).cuda()
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel).cuda()

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


# # todo 使用示例
# # 定义图像转换
# # 读取两张图片
# img1_path = 'F:\\Python\\Super_Resolution Microscopy by Grating and Deep Neural Network\\DataSets\\SRCNN_DataSets\\test\\1.png'
# img2_path = 'F:\\Python\\Super_Resolution Microscopy by Grating and Deep Neural Network\\DataSets\\SRCNN_DataSets\\test\\Targets\\1.png'
# # img2_path = 'F:\\Python\\Super_Resolution Microscopy by Grating and Deep Neural Network\\DataSets\\SRCNN_DataSets\\test\\1.png'
# # F:\\Python\\Super_Resolution Microscopy by Grating and Deep Neural Network\\DataSets\\SRCNN_DataSets\\test\\Degrade\\1.png
#
#
# img1 = cv2.imread(img1_path)
# img2 = cv2.imread(img2_path)
#
# # 将图片转换为 PyTorch 的 Tensor 格式，并调整形状和数据类型
# #                                0       1        2
# # 在NumPy中，多维数组的维度顺序是 (height, width, channels)，即行数（高度）、列数（宽度）和通道数。
# # 而在PyTorch中，张量的维度顺序是 (batch_size, channels, height, width)，即批量大小、通道数、高度和宽度。
# # .unsqueeze(0): 这一步在第0维度上增加一个维度，即在最前面增加一个维度，将其变为一个单一的样本。
# img1_tensor = torch.tensor(img1.transpose((2, 0, 1))).unsqueeze(0).float()
# img2_tensor = torch.tensor(img2.transpose((2, 0, 1))).unsqueeze(0).float()
#
# # 将两个 Tensor 移动到同一设备上
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# img1_tensor = img1_tensor.to(device)
# img2_tensor = img2_tensor.to(device)
#
# # 创建 SSIM 实例
# ssim_module = SSIM(window_size=11, size_average=True)
#
# # 计算 SSIM
# ssim_value = ssim_module(img1_tensor, img2_tensor)
#
# print("SSIM value:", ssim_value.item())
#
# # 显示图像
# cv2.imshow('Image 1', img1)
# cv2.imshow('Image 2', img2)
# # 等待按键按下
# cv2.waitKey(0)
# # 关闭所有窗口
# cv2.destroyAllWindows()