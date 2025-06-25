import os
import cv2
import numpy as np
from PIL import Image
import jittor as jt
from jittor.dataset import Dataset
from jittor.transform import *
import jittor.transform as transforms
from utils_jittor import *
from natsort import natsorted

class MSRSData(Dataset):
    def __init__(self, opts, is_train=True, crop_size=256):
        super(MSRSData, self).__init__()
        self.is_train = is_train
        self.crop_size = crop_size
        
        # 设置数据路径 - 根据实际MSRS数据集结构
        if is_train:
            self.vis_dir = os.path.join(opts.dataroot, 'train', 'vi')
            self.ir_dir = os.path.join(opts.dataroot, 'train', 'ir')
            self.label_dir = os.path.join(opts.dataroot, 'train', 'label')
            self.bi_dir = os.path.join(opts.dataroot, 'train', 'bi')
            self.bd_dir = os.path.join(opts.dataroot, 'train', 'bd')
            self.mask_dir = os.path.join(opts.dataroot, 'train', 'mask')
        else:
            self.vis_dir = os.path.join(opts.dataroot, 'test', 'vi')
            self.ir_dir = os.path.join(opts.dataroot, 'test', 'ir')
            self.label_dir = os.path.join(opts.dataroot, 'test', 'label')
        
        # 获取文件列表 - 使用label目录作为基准
        self.file_list = natsorted(os.listdir(self.label_dir))
        print(f"Dataset size: {len(self.file_list)}")
        
        # 数据变换
        self.crop = transforms.RandomCrop(crop_size)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        # 获取图像名称
        image_name = self.file_list[index]
        
        # 构建路径
        vis_path = os.path.join(self.vis_dir, image_name)
        ir_path = os.path.join(self.ir_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name)
        
        # 读取图像
        vis = self.imread(path=vis_path)
        ir = self.imread(path=ir_path, vis_flag=False)
        label = self.imread(path=label_path, label_flag=True)
        
        if self.is_train:
            # 训练时读取额外数据
            bi_path = os.path.join(self.bi_dir, image_name)
            bd_path = os.path.join(self.bd_dir, image_name)
            mask_path = os.path.join(self.mask_dir, image_name)
            bi = self.imread(path=bi_path, label_flag=True)
            bd = self.imread(path=bd_path, label_flag=True)
            mask = self.imread(path=mask_path, vis_flag=False)
            
            # 数据增强
            # 确保所有张量都去除batch维度
            if len(vis.shape) == 4:
                vis = vis.squeeze(0)
            if len(ir.shape) == 4:
                ir = ir.squeeze(0)
            if len(label.shape) == 4:
                label = label.squeeze(0)
            if len(bi.shape) == 4:
                bi = bi.squeeze(0)
            if len(bd.shape) == 4:
                bd = bd.squeeze(0)
            if len(mask.shape) == 4:
                mask = mask.squeeze(0)
                
            # 沿着通道维度拼接 [C, H, W]
            vis_ir = jt.concat([vis, ir, label, bi, bd, mask], dim=0)
            
            # 检查图像尺寸，确保足够大
            h, w = vis_ir.shape[-2], vis_ir.shape[-1]
            if h < 256 or w < 256:
                # 计算需要的尺寸，保持宽高比
                scale = max(256.0 / h, 256.0 / w)
                new_h, new_w = int(h * scale), int(w * scale)
                vis_ir = transforms.resize(vis_ir, (new_h, new_w))
            
            # 数据增强
            vis_ir = randfilp(vis_ir)
            vis_ir = randrot(vis_ir)
            
            # 对于RandomCrop，需要将张量转换为PIL图像格式
            # 先转换为numpy数组，然后使用PIL处理
            vis_ir_np = vis_ir.numpy()
            # 转换维度从CHW到HWC
            vis_ir_np = np.transpose(vis_ir_np, (1, 2, 0))
            
            # 处理多通道数据：如果通道数大于4，PIL无法直接处理
            # 我们需要分别处理可见光(前3通道)和其他通道
            if vis_ir_np.shape[2] > 4:
                # 分离可见光图像(前3通道)和其他通道
                vis_part = vis_ir_np[:, :, :3]  # RGB通道
                other_part = vis_ir_np[:, :, 3:]  # 其他通道
                
                # 转换为PIL图像(只处理RGB部分)
                vis_pil = Image.fromarray((vis_part * 255).astype(np.uint8))
            else:
                # 如果通道数<=4，直接转换
                vis_ir_pil = Image.fromarray((vis_ir_np * 255).astype(np.uint8))
            
            # 使用PIL的crop功能
            crop_size = 256
            if vis_ir_np.shape[2] > 4:
                # 处理多通道情况
                w, h = vis_pil.size
                if w >= crop_size and h >= crop_size:
                    # 随机裁剪
                    left = np.random.randint(0, w - crop_size + 1)
                    top = np.random.randint(0, h - crop_size + 1)
                    vis_pil_cropped = vis_pil.crop((left, top, left + crop_size, top + crop_size))
                    # 对其他通道应用相同的裁剪
                    other_part_cropped = other_part[top:top+crop_size, left:left+crop_size, :]
                else:
                    vis_pil_cropped = vis_pil
                    other_part_cropped = other_part
                
                # 重新组合所有通道
                vis_part_cropped = np.array(vis_pil_cropped).astype(np.float32) / 255.0
                patch_np = np.concatenate([vis_part_cropped, other_part_cropped], axis=2)
            else:
                # 处理通道数<=4的情况
                w, h = vis_ir_pil.size
                if w >= crop_size and h >= crop_size:
                    left = np.random.randint(0, w - crop_size + 1)
                    top = np.random.randint(0, h - crop_size + 1)
                    vis_ir_pil = vis_ir_pil.crop((left, top, left + crop_size, top + crop_size))
                patch_np = np.array(vis_ir_pil).astype(np.float32) / 255.0
            
            # 转换回张量格式
            patch_np = np.transpose(patch_np, (2, 0, 1))  # HWC到CHW
            patch = jt.array(patch_np)
            
            vis, ir, label, bi, bd, mask = jt.split(patch, [3, 1, 1, 1, 1, 1], dim=0)
            

            
            label = label.long()
            bi = (bi / 255.0).long()
            bd = (bd / 255.0).long()
            
            # 移除调试信息，直接返回正确维度的张量
            # 张量形状应该是 [C, H, W]，不需要squeeze第0维
            return ir.squeeze(0), vis, label.squeeze(0), bi.squeeze(0), bd.squeeze(0), mask.squeeze(0)
        else:
            label = label.long()
            return ir.squeeze(0), vis.squeeze(0), label.squeeze(0), image_name
    
    @staticmethod
    def imread(path, label_flag=False, vis_flag=True):
        """读取图像并转换为张量"""
        if label_flag:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # 转换为float32并归一化到[0,1]，然后转换为张量
            img = img.astype(np.float32) / 255.0
            im_ts = jt.array(img).unsqueeze(0).unsqueeze(0) * 255  # [1, 1, H, W]
        else:
            if vis_flag:  # 可见光图像，RGB通道
                img = np.array(Image.open(path).convert('RGB'))
                # 转换为float32并归一化到[0,1]
                img = img.astype(np.float32) / 255.0
                # 转换维度从HWC到CHW
                img = np.transpose(img, (2, 0, 1))
                im_ts = jt.array(img).unsqueeze(0)  # [1, 3, H, W]
            else:  # 红外图像，单通道
                img = np.array(Image.open(path).convert('L'))
                img = img.astype(np.float32) / 255.0
                im_ts = jt.array(img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        return im_ts

class FusionData(Dataset):
    """用于融合测试的数据集"""
    def __init__(self, data_dir):
        super(FusionData, self).__init__()
        self.data_dir = data_dir
        
        # 设置数据路径
        self.vis_dir = os.path.join(data_dir, 'vi')
        self.ir_dir = os.path.join(data_dir, 'ir')
        
        # 获取文件列表
        self.file_list = natsorted([f for f in os.listdir(self.vis_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f"Fusion test dataset size: {len(self.file_list)}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 获取文件名
        image_name = self.file_list[idx]
        
        # 构建路径
        vis_path = os.path.join(self.vis_dir, image_name)
        ir_path = os.path.join(self.ir_dir, image_name)
        
        # 读取图像
        vis = MSRSData.imread(path=vis_path, vis_flag=True)
        ir = MSRSData.imread(path=ir_path, vis_flag=False)
        
        return vis.squeeze(0), ir.squeeze(0), image_name

def get_loader(opts, is_train=True, batch_size=1, num_workers=4):
    """获取数据加载器"""
    if is_train:
        dataset = MSRSData(opts, is_train=True)
        shuffle = True
        drop_last = True
    else:
        dataset = MSRSData(opts, is_train=False)
        shuffle = False
        drop_last = False
    
    from jittor.dataset import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )
    
    return loader

def get_fusion_loader(data_dir, batch_size=1, num_workers=4):
    """获取融合测试数据加载器"""
    dataset = FusionData(data_dir)
    
    from jittor.dataset import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return loader