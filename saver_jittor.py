import os
import jittor as jt
from jittor import nn
import numpy as np
from PIL import Image
import cv2
from utils_jittor import tensor_to_image

class Saver:
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt.result_dir
        self.model_dir = os.path.join(self.save_dir, 'models')
        self.result_dir = os.path.join(self.save_dir, 'results')
        
        # 创建保存目录
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
    
    def save_checkpoint(self, state, epoch):
        """保存模型检查点"""
        if isinstance(epoch, int):
            filename = f'checkpoint_epoch_{epoch:04d}.pkl'
        else:
            filename = f'checkpoint_{epoch}.pkl'
        
        filepath = os.path.join(self.model_dir, filename)
        jt.save(state, filepath)
        print(f'Checkpoint saved: {filepath}')
    
    def load_checkpoint(self, filepath):
        """加载模型检查点"""
        if os.path.exists(filepath):
            checkpoint = jt.load(filepath)
            print(f'Checkpoint loaded: {filepath}')
            return checkpoint
        else:
            print(f'Checkpoint not found: {filepath}')
            return None
    
    def save_model(self, model, filename='model.pkl'):
        """保存模型"""
        filepath = os.path.join(self.model_dir, filename)
        jt.save(model.state_dict(), filepath)
        print(f'Model saved: {filepath}')
    
    def load_model(self, model, filepath):
        """加载模型"""
        if os.path.exists(filepath):
            state_dict = jt.load(filepath)
            model.load_state_dict(state_dict)
            print(f'Model loaded: {filepath}')
            return model
        else:
            print(f'Model file not found: {filepath}')
            return model
    
    def save_image(self, tensor, filename, normalize=True):
        """保存图像张量"""
        if normalize:
            # 归一化到[0, 1]
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        
        # 转换为PIL图像并保存
        image = tensor_to_image(tensor)
        filepath = os.path.join(self.result_dir, filename)
        image.save(filepath)
        print(f'Image saved: {filepath}')
    
    def save_fusion_result(self, vis_img, ir_img, fused_img, filename_prefix):
        """保存融合结果"""
        # 保存可见光图像
        self.save_image(vis_img, f'{filename_prefix}_vis.png')
        
        # 保存红外图像
        self.save_image(ir_img, f'{filename_prefix}_ir.png')
        
        # 保存融合图像
        self.save_image(fused_img, f'{filename_prefix}_fused.png')
        
        # 创建拼接图像
        self.save_concatenated_image([vis_img, ir_img, fused_img], 
                                   f'{filename_prefix}_comparison.png')
    
    def save_concatenated_image(self, tensors, filename):
        """保存拼接图像"""
        # 将所有张量转换为相同尺寸
        h, w = tensors[0].shape[-2:]
        normalized_tensors = []
        
        for tensor in tensors:
            # 调整尺寸
            if tensor.shape[-2:] != (h, w):
                tensor = jt.nn.interpolate(tensor.unsqueeze(0), size=(h, w), 
                                         mode='bilinear', align_corners=True).squeeze(0)
            
            # 归一化
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
            normalized_tensors.append(tensor)
        
        # 水平拼接
        concatenated = jt.concat(normalized_tensors, dim=-1)
        
        # 保存
        self.save_image(concatenated, filename, normalize=False)
    
    def save_segmentation_result(self, image, prediction, label, filename_prefix):
        """保存分割结果"""
        # 保存原图
        self.save_image(image, f'{filename_prefix}_image.png')
        
        # 保存预测结果
        if len(prediction.shape) == 4:
            # 如果是概率图，取最大值索引
            prediction = jt.argmax(prediction, dim=1).float()
        self.save_image(prediction, f'{filename_prefix}_prediction.png')
        
        # 保存标签（如果有）
        if label is not None:
            self.save_image(label.float(), f'{filename_prefix}_label.png')
        
        # 创建对比图
        if label is not None:
            self.save_concatenated_image([image, prediction.unsqueeze(0), label.unsqueeze(0).float()], 
                                       f'{filename_prefix}_seg_comparison.png')
        else:
            self.save_concatenated_image([image, prediction.unsqueeze(0)], 
                                       f'{filename_prefix}_seg_comparison.png')
    
    def save_training_log(self, log_data, filename='training_log.txt'):
        """保存训练日志"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'a') as f:
            f.write(log_data + '\n')
    
    def save_config(self, config, filename='config.txt'):
        """保存配置信息"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            for key, value in vars(config).items():
                f.write(f'{key}: {value}\n')
    
    def create_result_summary(self, results, filename='results_summary.txt'):
        """创建结果摘要"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            f.write('Results Summary\n')
            f.write('=' * 50 + '\n')
            for key, value in results.items():
                f.write(f'{key}: {value}\n')

def multi_task_tester(model, vis_img, ir_img, save_path, img_name):
    """多任务测试器"""
    model.eval()
    
    with jt.no_grad():
        # 模型推理
        semantic_out, binary_out, boundary_out, fused_img, vi_img, ir_img_pred = model(vis_img, ir_img)
        
        # 保存融合结果
        save_fusion_image(fused_img, save_path, img_name)
        
        # 保存分割结果（如果需要）
        if semantic_out is not None:
            save_segmentation_image(semantic_out, save_path, img_name)
        
        return fused_img, semantic_out

def save_fusion_image(fused_img, save_path, img_name):
    """保存融合图像"""
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 归一化到[0, 255]
    fused_img = jt.clamp(fused_img, 0, 1)
    fused_img = (fused_img * 255).byte()
    
    # 转换为numpy数组
    if len(fused_img.shape) == 4:
        fused_img = fused_img.squeeze(0)
    
    if fused_img.shape[0] == 1:
        # 灰度图像
        img_array = fused_img.squeeze(0).numpy()
        image = Image.fromarray(img_array, mode='L')
    else:
        # RGB图像
        img_array = fused_img.numpy().transpose(1, 2, 0)
        image = Image.fromarray(img_array, mode='RGB')
    
    # 保存图像
    save_file = os.path.join(save_path, f'{img_name}_fused.png')
    image.save(save_file)
    print(f'Fusion result saved: {save_file}')

def save_segmentation_image(seg_out, save_path, img_name):
    """保存分割图像"""
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 获取预测类别
    if len(seg_out.shape) == 4:
        seg_pred = jt.argmax(seg_out, dim=1)
    else:
        seg_pred = seg_out
    
    # 转换为numpy数组
    if len(seg_pred.shape) == 3:
        seg_pred = seg_pred.squeeze(0)
    
    seg_array = seg_pred.numpy().astype(np.uint8)
    
    # 应用颜色映射（可选）
    colored_seg = apply_color_map(seg_array)
    
    # 保存图像
    save_file = os.path.join(save_path, f'{img_name}_seg.png')
    Image.fromarray(colored_seg).save(save_file)
    print(f'Segmentation result saved: {save_file}')

def apply_color_map(seg_array, num_classes=9):
    """应用颜色映射到分割结果"""
    # 定义颜色映射
    colors = [
        [0, 0, 0],       # 背景
        [128, 0, 0],     # 类别1
        [0, 128, 0],     # 类别2
        [128, 128, 0],   # 类别3
        [0, 0, 128],     # 类别4
        [128, 0, 128],   # 类别5
        [0, 128, 128],   # 类别6
        [128, 128, 128], # 类别7
        [64, 0, 0],      # 类别8
    ]
    
    h, w = seg_array.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(min(num_classes, len(colors))):
        mask = seg_array == i
        colored[mask] = colors[i]
    
    return colored

def load_pretrained_model(model, pretrained_path, strict=True):
    """加载预训练模型"""
    if os.path.exists(pretrained_path):
        print(f'Loading pretrained model from {pretrained_path}')
        state_dict = jt.load(pretrained_path)
        
        # 如果是完整的检查点，提取模型状态字典
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # 加载状态字典
        if strict:
            model.load_state_dict(state_dict)
        else:
            # 非严格模式，只加载匹配的参数
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        
        print('Pretrained model loaded successfully')
    else:
        print(f'Pretrained model not found: {pretrained_path}')
    
    return model