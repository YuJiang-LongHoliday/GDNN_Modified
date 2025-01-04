# @Time    : 2024/6/6 19:35
# @Author  : YuJiang-LongHoliday
# @Project : PSNR.py

#todo
# 注释：PSNR, Peak Signal-to-Noise Ratio,峰值信噪比,较高的 PSNR 表示图像重建质量较好。

import torch
import torch.nn.functional as F
from PIL import Image  # PIL（Python Imaging Library）库用于加载和处理图像
import numpy as np  # numpy库用于处理图像数据，特别是执行数值计算

# max_pixel：图像中像素的最大值，默认为 1.0，适用于归一化到 [0, 1] 范围的图像
def PSNR_Loss(target, input, max_pixel=1.0):
    MSE_Loss = F.mse_loss(input, target)  # 使用 PyTorch 的 F.mse_loss 函数计算输入图像和目标图像之间的均方误差（MSE）

    # 如果 MSE 为零，意味着两张图像完全相同。此时返回正无穷大（表示无穷大的 PSNR）
    if MSE_Loss == 0:
        return torch.tensor(float('inf'), device=target.device)

    # 将 max_pixel 转换为张量，并确保它在与输入和目标图像相同的设备上（CPU 或 GPU）和相同的数据类型
    max_pixel_tensor = torch.tensor(max_pixel, device=target.device, dtype=target.dtype)

    PSNR = 20 * torch.log10(max_pixel_tensor) - 10 * torch.log10(MSE_Loss)
    return -PSNR  # 返回负的 PSNR 值，因为在深度学习中我们通常最小化损失函数，而较高的 PSNR 表示较好的图像质量。

def Calculate_PSNR(img1, img2):
    # 加载图像
    image1 = Image.open(image1_path).convert('L')  # 转换为灰度图像以便比较,'L' 代表灰度模式（Luminance）
    image2 = Image.open(image2_path).convert('L')

    # 确保两张图像尺寸相同
    if image1.size != image2.size:
        raise ValueError("Images must have the same dimensions.")

    # 将图像转换为numpy数组
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # 计算MSE
    MSE_Value = np.mean((image1_array - image2_array) ** 2)

    if MSE_Value == 0:
        return float('inf')
    max_pixel = 255.0
    PSNR = 20 * np.log10(max_pixel) - 10 * np.log10(MSE_Value)
    return PSNR

# # todo 使用示例
# image1_path = 'F:\\Python\\Super_Resolution Microscopy by Grating and Deep Neural Network\\DataSets\\SRCNN_DataSets\\test\\1.png'  # 输入第一张图像路径
# image2_path = 'F:\\Python\\Super_Resolution Microscopy by Grating and Deep Neural Network\\DataSets\\SRCNN_DataSets\\test\\Targets\\1.png'  # 输入第二张图像路径
#
# PSNR_Value = Calculate_PSNR(image1_path, image2_path)
# print(f"The PSNR value between the two images is: {PSNR_Value:.2f} dB")

