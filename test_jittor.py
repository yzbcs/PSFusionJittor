import jittor as jt
import jittor.nn as nn
from jittor import transform
from jittor.dataset import DataLoader
import os
import time
import numpy as np
from PIL import Image
import argparse

from PSF_jittor import PSF
from create_dataset_jittor import get_fusion_loader
from utils_jittor import *
from saver_jittor import Saver
from options import TestOptions

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PSFusion Testing with Jittor')
    
    # 数据相关
    parser.add_argument('--dataroot', type=str, required=True, help='数据集根目录')
    parser.add_argument('--crop_size', type=int, default=480, help='输入图像尺寸')
    parser.add_argument('--n_classes', type=int, default=9, help='分割类别数')
    
    # 模型相关
    parser.add_argument('--pretrained_path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--model_name', type=str, default='PSF', help='模型名称')
    
    # 测试相关
    parser.add_argument('--batch_size', type=int, default=1, help='批大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--save_dir', type=str, default='./test_results', help='结果保存目录')
    parser.add_argument('--save_seg', action='store_true', help='是否保存分割结果')
    
    # GPU设置
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs')
    
    return parser.parse_args()

def setup_jittor(gpu_ids):
    """设置Jittor环境"""
    if gpu_ids and gpu_ids != '-1':
        # 设置使用GPU
        jt.flags.use_cuda = 1
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"Using GPU: {gpu_ids}")
    else:
        # 使用CPU
        jt.flags.use_cuda = 0
        print("Using CPU")

def load_model(model_path, n_classes=9):
    """加载预训练模型"""
    print(f"Loading model from: {model_path}")
    
    # 创建模型
    model = PSF(n_classes=n_classes)
    
    # 加载权重
    if os.path.exists(model_path):
        checkpoint = jt.load(model_path)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.eval()
    return model

def create_test_dataloader(dataroot, batch_size, num_workers):
    """创建测试数据加载器"""
    test_loader = get_fusion_loader(
        data_dir=dataroot,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return test_loader

def test_model(model, test_loader, save_dir, save_seg=False):
    """测试模型"""
    model.eval()
    
    # 创建保存目录
    fusion_dir = os.path.join(save_dir, 'fusion')
    os.makedirs(fusion_dir, exist_ok=True)
    
    if save_seg:
        seg_dir = os.path.join(save_dir, 'segmentation')
        os.makedirs(seg_dir, exist_ok=True)
    
    total_time = 0
    num_images = 0
    
    print("Starting testing...")
    
    with jt.no_grad():
        for i, batch in enumerate(test_loader):
            # 获取数据
            vis_img, ir_img, img_name = batch
            
            # 记录推理时间
            start_time = time.time()
            
            # 模型推理
            outputs = model(vis_img, ir_img)
            
            if isinstance(outputs, tuple):
                semantic_out, binary_out, boundary_out, fusion_img, vi_img, ir_img = outputs
                seg_result = semantic_out if save_seg else None
            else:
                fusion_img = outputs
                seg_result = None
            
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            num_images += fusion_img.shape[0]
            
            # 保存结果
            for j in range(fusion_img.shape[0]):
                if isinstance(img_name, list):
                    name = img_name[j] if j < len(img_name) else f"image_{i}_{j}"
                else:
                    name = f"image_{i}_{j}"
                
                # 保存融合图像
                fusion_path = os.path.join(fusion_dir, f"{name}_fusion.png")
                save_fusion_image(fusion_img[j], fusion_path)
                
                # 保存分割结果
                if save_seg and seg_result is not None:
                    seg_path = os.path.join(seg_dir, f"{name}_seg.png")
                    save_segmentation_result(seg_result[j], seg_path)
            
            # 打印进度
            if (i + 1) % 10 == 0:
                avg_time = total_time / num_images
                print(f"Processed {i + 1}/{len(test_loader)} batches, "
                      f"Avg inference time: {avg_time:.4f}s")
    
    # 计算平均推理时间
    avg_inference_time = total_time / num_images if num_images > 0 else 0
    print(f"\nTesting completed!")
    print(f"Total images: {num_images}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Average inference time: {avg_inference_time:.4f}s")
    print(f"FPS: {1.0/avg_inference_time:.2f}" if avg_inference_time > 0 else "FPS: N/A")
    
    return avg_inference_time

def save_fusion_image(fusion_tensor, save_path):
    """保存融合图像"""
    # 转换为numpy数组
    if isinstance(fusion_tensor, jt.Var):
        fusion_np = fusion_tensor.detach().numpy()
    else:
        fusion_np = fusion_tensor
    
    # 处理维度
    if len(fusion_np.shape) == 4:
        fusion_np = fusion_np[0]  # 移除batch维度
    
    if len(fusion_np.shape) == 3:
        if fusion_np.shape[0] == 1:  # 灰度图像
            fusion_np = fusion_np[0]
        elif fusion_np.shape[0] == 3:  # RGB图像
            fusion_np = np.transpose(fusion_np, (1, 2, 0))
    
    # 归一化到0-255
    fusion_np = np.clip(fusion_np * 255, 0, 255).astype(np.uint8)
    
    # 保存图像
    if len(fusion_np.shape) == 2:  # 灰度图像
        Image.fromarray(fusion_np, mode='L').save(save_path)
    else:  # RGB图像
        Image.fromarray(fusion_np, mode='RGB').save(save_path)

def save_segmentation_result(seg_tensor, save_path):
    """保存分割结果"""
    # 转换为numpy数组
    if isinstance(seg_tensor, jt.Var):
        seg_np = seg_tensor.detach().numpy()
    else:
        seg_np = seg_tensor
    
    # 处理维度
    if len(seg_np.shape) == 4:
        seg_np = seg_np[0]  # 移除batch维度
    
    if len(seg_np.shape) == 3:
        # 如果是概率图，取argmax
        seg_np = np.argmax(seg_np, axis=0)
    
    # 转换为uint8
    seg_np = seg_np.astype(np.uint8)
    
    # 应用颜色映射（可选）
    colored_seg = apply_color_map(seg_np)
    
    # 保存图像
    Image.fromarray(colored_seg, mode='RGB').save(save_path)

def apply_color_map(seg_map, n_classes=9):
    """为分割图应用颜色映射"""
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
    
    # 扩展颜色列表以适应更多类别
    while len(colors) < n_classes:
        colors.append([np.random.randint(0, 256) for _ in range(3)])
    
    h, w = seg_map.shape
    colored_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(n_classes):
        mask = seg_map == i
        colored_map[mask] = colors[i]
    
    return colored_map

def save_test_report(save_dir, avg_inference_time, num_images):
    """保存测试报告"""
    report_path = os.path.join(save_dir, 'test_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("PSFusion Test Report\n")
        f.write("===================\n\n")
        f.write(f"Total images processed: {num_images}\n")
        f.write(f"Average inference time: {avg_inference_time:.4f}s\n")
        f.write(f"FPS: {1.0/avg_inference_time:.2f}\n" if avg_inference_time > 0 else "FPS: N/A\n")
        f.write(f"Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Test report saved to: {report_path}")

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置Jittor环境
    setup_jittor(args.gpu_ids)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args.pretrained_path, args.n_classes)
    
    # 创建数据加载器
    test_loader = create_test_dataloader(
        args.dataroot, 
        args.batch_size, 
        args.num_workers
    )
    
    # 测试模型
    avg_inference_time = test_model(
        model, 
        test_loader, 
        args.save_dir, 
        args.save_seg
    )
    
    # 保存测试报告
    save_test_report(args.save_dir, avg_inference_time, len(test_loader.dataset))
    
    print(f"\nAll results saved to: {args.save_dir}")

if __name__ == '__main__':
    main()