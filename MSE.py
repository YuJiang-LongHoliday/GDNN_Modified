# @Time    : 2024/6/6 10:16
# @Author  : YuJiang-LongHoliday
# @Project : MSE.py
#todo
# 注释：MSE,Mean Squared Error,计算两张图像之间的均方误差,均方误差是衡量两张图像差异的一种方式，数值越大表示差异越大。

from PIL import Image  # PIL（Python Imaging Library）库用于加载和处理图像
import numpy as np  # numpy库用于处理图像数据，特别是执行数值计算

def Calculate_MSE(image1_path, image2_path):
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

    return MSE_Value

# todo 使用示例
image1_path = 'F:\\Python\\Super_Resolution Microscopy by Grating and Deep Neural Network\\1.png'
image2_path = 'F:\\Python\\Super_Resolution Microscopy by Grating and Deep Neural Network\\3.png'
MSE_Value = Calculate_MSE(image1_path, image2_path)
print(f"The Mean Squared Error (MSE) between the two images is: {MSE_Value}")


