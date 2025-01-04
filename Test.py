# @Time    : 2024/6/10 22:57
# @Author  : YuJiang-LongHoliday
# @Project : GDNN_Test.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from Attention_Net import SimpleResNet, CustomDataset, ResidualBlock, ChannelAttention, SpatialAttention,UpSampleBlock

def infer_images(model_path, input_folder_path, output_folder_path):
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    # model = SimpleResNet().to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model = torch.load(model_path, map_location=device)
    model.eval()  # 设置为评估模式

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 确保输出文件夹存在
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 遍历输入文件夹中的所有图像
    for filename in os.listdir(input_folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            input_image_path = os.path.join(input_folder_path, filename)
            output_image_path = os.path.join(output_folder_path, filename)

            # 加载和预处理图像
            image = Image.open(input_image_path).convert('L')  # 转换为灰度图像
            image = transform(image).unsqueeze(0).to(device)  # 添加batch维度

            # 推断
            with torch.no_grad():
                reconstructed_image = model(image)

            # 将输出转换为图像并保存
            save_image = transforms.ToPILImage()(reconstructed_image.squeeze().cpu())  # 移除batch维度
            save_image.save(output_image_path)
            print(f"Processed and saved: {output_image_path}")

# 使用方法
model_path = r"F:\Python\Super_Resolution Microscopy by Grating and Deep Neural Network\Params\410nm\Polarization\With_Grating\300nm\Focus\Attention_Net_1040_With_Grating_300nm_focus_128_128_2000.pth"
input_folder_path = r"C:\Users\Kawhi Leonard\Desktop\410_polar_WithGrating_Focus_300nm"
output_folder_path = r"F:\Python\U-Net_Test_5\DataSets\ReConstruction"
infer_images(model_path, input_folder_path, output_folder_path)