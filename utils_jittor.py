import jittor as jt
from jittor import nn
import jittor.nn as F
import numpy as np
from PIL import Image
import cv2
import os

def randrot(image, label=None):
    """随机旋转图像"""
    angle = np.random.randint(-30, 30)
    
    # 检查图像维度
    if len(image.shape) < 2:
        return image if label is None else (image, label)
    
    h, w = image.shape[-2:]
    if h <= 0 or w <= 0:
        return image if label is None else (image, label)
        
    center = (w // 2, h // 2)
    
    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 旋转图像
    if len(image.shape) == 4:  # batch dimension
        image_np = image.squeeze(0).numpy()
        if len(image_np.shape) == 3:
            image_np = image_np.transpose(1, 2, 0)
        rotated = cv2.warpAffine(image_np, M, (w, h))
        if len(rotated.shape) == 3:
            rotated = rotated.transpose(2, 0, 1)
        rotated = jt.array(rotated).unsqueeze(0)
    elif len(image.shape) == 3:
        image_np = image.numpy().transpose(1, 2, 0)
        rotated = cv2.warpAffine(image_np, M, (w, h))
        rotated = jt.array(rotated.transpose(2, 0, 1))
    else:
        image_np = image.numpy()
        rotated = cv2.warpAffine(image_np, M, (w, h))
        rotated = jt.array(rotated)
    
    if label is not None:
        if len(label.shape) == 4:
            label_np = label.squeeze(0).numpy()
        else:
            label_np = label.numpy()
        if len(label_np.shape) > 2:
            label_np = label_np.squeeze()
        label_rotated = cv2.warpAffine(label_np, M, (w, h))
        if len(label.shape) == 4:
            label_rotated = jt.array(label_rotated).unsqueeze(0)
        else:
            label_rotated = jt.array(label_rotated)
        return rotated, label_rotated
    
    return rotated

def randfilp(image, label=None):
    """随机翻转图像"""
    # 随机水平翻转
    if np.random.random() > 0.5:
        image = jt.flip(image, dim=-1)
        if label is not None:
            label = jt.flip(label, dim=-1)
    
    # 随机垂直翻转
    if np.random.random() > 0.5:
        image = jt.flip(image, dim=-2)
        if label is not None:
            label = jt.flip(label, dim=-2)
    
    if label is not None:
        return image, label
    return image

def RGB2YCrCb(input_im):
    """RGB转YCrCb颜色空间"""
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = jt.array([[0.299, 0.587, 0.114], 
                    [-0.169, -0.331, 0.5], 
                    [0.5, -0.419, -0.081]])
    bias = jt.array([0., 128., 128.])
    temp = im_flat.mm(mat.t()) + bias
    out = temp.reshape(input_im.shape[0], input_im.shape[2], input_im.shape[3], 3).transpose(1, 3).transpose(2, 3)
    return out

def YCrCb2RGB(input_im):
    """YCrCb转RGB颜色空间"""
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = jt.array([[1.0, 0.0, 1.402], 
                    [1.0, -0.344, -0.714], 
                    [1.0, 1.772, 0.0]])
    bias = jt.array([0., -128., -128.])
    temp = (im_flat + bias).mm(mat.t())
    out = temp.reshape(input_im.shape[0], input_im.shape[2], input_im.shape[3], 3).transpose(1, 3).transpose(2, 3)
    return out

def tensor_to_image(tensor):
    """将张量转换为PIL图像"""
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    
    if tensor.shape[0] == 1:
        # 灰度图像
        image = tensor.squeeze(0).numpy()
        image = (image * 255).astype(np.uint8)
        return Image.fromarray(image, mode='L')
    elif tensor.shape[0] == 3:
        # RGB图像
        image = tensor.numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        return Image.fromarray(image, mode='RGB')
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

def image_to_tensor(image):
    """将PIL图像转换为张量"""
    if image.mode == 'L':
        # 灰度图像
        array = np.array(image).astype(np.float32) / 255.0
        return jt.array(array).unsqueeze(0)
    elif image.mode == 'RGB':
        # RGB图像
        array = np.array(image).astype(np.float32) / 255.0
        return jt.array(array.transpose(2, 0, 1))
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")

def save_image(tensor, path):
    """保存张量为图像文件"""
    image = tensor_to_image(tensor)
    image.save(path)

def load_image(path, mode='RGB'):
    """加载图像文件为张量"""
    image = Image.open(path).convert(mode)
    return image_to_tensor(image)

def calculate_psnr(img1, img2, max_val=1.0):
    """计算PSNR"""
    mse = jt.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * jt.log10(max_val / jt.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, window_size=11, max_val=1.0):
    """计算SSIM"""
    from losses_jittor import mssim
    ssim_val, _ = mssim(img1, img2, window_size=window_size, val_range=max_val)
    return ssim_val.item()

def normalize_tensor(tensor, mean=None, std=None):
    """标准化张量"""
    if mean is None:
        mean = jt.mean(tensor)
    if std is None:
        std = jt.std(tensor)
    
    return (tensor - mean) / std

def denormalize_tensor(tensor, mean, std):
    """反标准化张量"""
    return tensor * std + mean

def resize_tensor(tensor, size, mode='bilinear'):
    """调整张量尺寸"""
    return F.interpolate(tensor, size=size, mode=mode, align_corners=True)

def pad_tensor(tensor, pad_size, mode='reflect'):
    """填充张量"""
    if mode == 'reflect':
        return F.pad(tensor, pad_size, mode='reflect')
    elif mode == 'constant':
        return F.pad(tensor, pad_size, mode='constant', value=0)
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")

def crop_tensor(tensor, crop_size):
    """裁剪张量"""
    h, w = tensor.shape[-2:]
    crop_h, crop_w = crop_size
    
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    return tensor[..., start_h:start_h+crop_h, start_w:start_w+crop_w]

def compute_gradient(tensor):
    """计算梯度"""
    # Sobel算子
    sobel_x = jt.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0)
    sobel_y = jt.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0)
    
    if tensor.shape[1] > 1:
        # 多通道图像，转换为灰度
        gray = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
    else:
        gray = tensor
    
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    
    gradient = jt.sqrt(grad_x ** 2 + grad_y ** 2)
    return gradient

def compute_laplacian(tensor):
    """计算拉普拉斯算子"""
    laplacian_kernel = jt.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).float().unsqueeze(0).unsqueeze(0)
    
    if tensor.shape[1] > 1:
        # 多通道图像，转换为灰度
        gray = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
    else:
        gray = tensor
    
    laplacian = F.conv2d(gray, laplacian_kernel, padding=1)
    return laplacian

def histogram_matching(source, template):
    """直方图匹配"""
    # 将张量转换为numpy数组
    source_np = source.numpy()
    template_np = template.numpy()
    
    # 计算直方图
    source_hist, source_bins = np.histogram(source_np.flatten(), 256, [0, 1])
    template_hist, template_bins = np.histogram(template_np.flatten(), 256, [0, 1])
    
    # 计算累积分布函数
    source_cdf = source_hist.cumsum()
    template_cdf = template_hist.cumsum()
    
    # 归一化
    source_cdf = source_cdf / source_cdf[-1]
    template_cdf = template_cdf / template_cdf[-1]
    
    # 创建映射表
    mapping = np.interp(source_cdf, template_cdf, np.linspace(0, 1, 256))
    
    # 应用映射
    matched = np.interp(source_np.flatten(), np.linspace(0, 1, 256), mapping)
    matched = matched.reshape(source_np.shape)
    
    return jt.array(matched)

def create_gaussian_kernel(size, sigma):
    """创建高斯核"""
    coords = jt.arange(size).float()
    coords -= size // 2
    
    g = jt.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    
    return g.unsqueeze(0) * g.unsqueeze(1)

def gaussian_blur(tensor, kernel_size=5, sigma=1.0):
    """高斯模糊"""
    kernel = create_gaussian_kernel(kernel_size, sigma)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    # 对每个通道分别应用卷积
    channels = tensor.shape[1]
    kernel = kernel.repeat(channels, 1, 1, 1)
    
    blurred = F.conv2d(tensor, kernel, padding=kernel_size//2, groups=channels)
    return blurred

def edge_preserving_filter(tensor, d=9, sigma_color=75, sigma_space=75):
    """边缘保持滤波（使用双边滤波近似）"""
    # 这里使用简化的实现，实际应用中可能需要更复杂的算法
    return gaussian_blur(tensor, kernel_size=d, sigma=sigma_space/10)

def adaptive_histogram_equalization(tensor, clip_limit=2.0, tile_grid_size=(8, 8)):
    """自适应直方图均衡化"""
    # 简化实现，实际应用中需要更复杂的CLAHE算法
    # 这里只进行全局直方图均衡化
    tensor_np = tensor.numpy()
    
    # 对每个通道分别处理
    result = np.zeros_like(tensor_np)
    for c in range(tensor_np.shape[1]):
        channel = tensor_np[0, c]
        # 计算直方图
        hist, bins = np.histogram(channel.flatten(), 256, [0, 1])
        # 计算累积分布函数
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]
        # 应用均衡化
        result[0, c] = np.interp(channel.flatten(), np.linspace(0, 1, 256), cdf_normalized).reshape(channel.shape)
    
    return jt.array(result)

def multi_scale_decomposition(tensor, levels=3):
    """多尺度分解"""
    pyramid = [tensor]
    current = tensor
    
    for i in range(levels):
        # 下采样
        current = F.avg_pool2d(current, kernel_size=2, stride=2)
        pyramid.append(current)
    
    return pyramid

def multi_scale_reconstruction(pyramid):
    """多尺度重建"""
    result = pyramid[-1]
    
    for i in range(len(pyramid) - 2, -1, -1):
        # 上采样
        result = F.interpolate(result, size=pyramid[i].shape[-2:], mode='bilinear', align_corners=True)
        # 融合
        result = (result + pyramid[i]) / 2
    
    return result

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = jt.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = jt.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = jt.diag(self.confusionMatrix) / self.confusionMatrix.sum(dim=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        # 过滤掉无穷大的值
        valid_mask = classAcc < float('inf')
        if valid_mask.sum() > 0:
            meanAcc = classAcc[valid_mask].mean()
        else:
            meanAcc = jt.array(0.0)
        return meanAcc  # 返回单个值

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = jt.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = jt.sum(self.confusionMatrix, dim=1) + jt.sum(self.confusionMatrix, dim=0) - jt.diag(
            self.confusionMatrix)  # dim = 1表示混淆矩阵行的值，返回列表； dim = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        # 过滤掉无穷大的值
        valid_mask = IoU < float('inf')
        if valid_mask.sum() > 0:
            mIoU = IoU[valid_mask].mean()
        else:
            mIoU = jt.array(0.0)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel, ignore_labels):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :param ignore_labels:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        for IgLabel in ignore_labels:
            mask &= (imgLabel != IgLabel)
        
        # 获取有效的预测和标签
        valid_pred = imgPredict[mask]
        valid_label = imgLabel[mask]
        
        # 手动计算混淆矩阵，替代bincount
        confusionMatrix = jt.zeros((self.numClass, self.numClass))
        for i in range(self.numClass):
            for j in range(self.numClass):
                # 计算预测为i，真实标签为j的像素数量
                count = ((valid_pred == i) & (valid_label == j)).sum()
                confusionMatrix[j, i] = count
        
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel, ignore_labels):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = jt.zeros((self.numClass, self.numClass))