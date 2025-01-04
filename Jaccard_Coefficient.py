# @Time    : 2024/6/6 22:05
# @Author  : YuJiang-LongHoliday
# @Project : Jaccard_Coefficient.py

#todo
# 注释：Jaccard系数（Jaccard Coefficient）是一种用于衡量两个集合之间相似性的指标。在图像处理中，Jaccard系数通常用于衡量两个二值图像之间的相似程度。
# Jaccard系数越接近1，表示两个集合的重叠部分越多，相似度越高。

import cv2
import numpy as np

def Jaccard_Coefficient(img1, img2):
    # 确保两张图像大小相同
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions.")

    # 计算交集：两个图像对应像素都为1的点
    intersection = np.logical_and(img1, img2)

    # 计算并集：两个图像对应像素至少有一个为1的点
    union = np.logical_or(img1, img2)

    # 计算交集和并集中为True的像素个数
    intersection_count = np.sum(intersection)
    union_count = np.sum(union)

    # 避免除以0的情况
    if union_count == 0:
        return 0

    # 计算并返回杰卡德相似系数
    jaccard_sim = intersection_count / union_count
    return jaccard_sim

# 读取两张二值化图像
image_path1 = 'F:\\Python\\Super_Resolution Microscopy by Grating and Deep Neural Network\\1.png'
image_path2 = 'F:\\Python\\Super_Resolution Microscopy by Grating and Deep Neural Network\\2.png'

# 小于等于128的像素值被转换为False，但在转换为 np.uint8 类型时，False 被转换为0，True 被转换为1。
# 所以在这种情况下，大于128的部分会变成255（即白色），小于等于128的部分会变成0（即黑色）
img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
img1 = (img1 > 50).astype(np.uint8)  # 假设阈值为128，将图像二值化

img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
img2 = (img2 > 50).astype(np.uint8)  # 假设阈值为128，将图像二值化

# 计算并打印杰卡德相似系数
Jaccard_Coefficient_Value = Jaccard_Coefficient(img1, img2)

print(f'Jaccard Similarity: {Jaccard_Coefficient_Value}')