import os
import argparse
import time
import jittor as jt
from jittor import nn
from jittor.dataset import DataLoader
import numpy as np
from PIL import Image
from PSF_jittor import PSF
from create_dataset_jittor import get_fusion_loader, get_loader
from options import TestOptions
from saver_jittor import Saver, multi_task_tester, load_pretrained_model
from utils_jittor import *

# 设置Jittor使用GPU
jt.flags.use_cuda = 1

def test_fusion():
    """测试图像融合"""
    # 解析命令行参数
    opt = TestOptions().parse()
    
    # 创建保存器
    saver = Saver(opt)
    
    # 创建模型
    model = PSF(n_classes=opt.n_classes)
    
    # 加载预训练模型
    if opt.pretrained_path and os.path.exists(opt.pretrained_path):
        model = load_pretrained_model(model, opt.pretrained_path, strict=False)
    else:
        print("Warning: No pretrained model loaded")
    
    # 设置为评估模式
    model.eval()
    
    # 创建测试数据加载器
    test_loader = get_fusion_loader(
        data_dir=opt.dataroot,
        batch_size=1,
        num_workers=1
    )
    
    print(f"Testing on {len(test_loader)} images...")
    
    # 创建结果保存目录
    result_dir = os.path.join(opt.save_dir, 'fusion_results')
    os.makedirs(result_dir, exist_ok=True)
    
    # 测试循环
    total_time = 0
    with jt.no_grad():
        for i, batch in enumerate(test_loader):
            ir_img, vis_img, img_name = batch
            print(f"Processing {i+1}/{len(test_loader)}: {img_name[0]}")
            
            # 记录开始时间
            start_time = time.time()
            
            # 模型推理
            semantic_out, binary_out, boundary_out, fused_img, vi_img, ir_img_pred = model(vis_img, ir_img)
            
            # 记录结束时间
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            
            print(f"Inference time: {inference_time:.4f}s")
            
            # 保存融合结果
            img_basename = os.path.splitext(img_name[0])[0]
            multi_task_tester(model, vis_img, ir_img, result_dir, img_basename)
            
            # 保存单独的融合图像
            save_fusion_image(fused_img, result_dir, img_basename)
            
            # 可选：保存分割结果
            if opt.save_seg and semantic_out is not None:
                save_segmentation_result(semantic_out, result_dir, img_basename)
    
    # 计算平均推理时间
    avg_time = total_time / len(test_loader)
    print(f"\nTesting completed!")
    print(f"Average inference time: {avg_time:.4f}s")
    print(f"Total time: {total_time:.4f}s")
    print(f"Results saved to: {result_dir}")
    
    # 保存测试报告
    save_test_report(opt, len(test_loader), total_time, avg_time, result_dir)

def save_fusion_image(fused_img, save_dir, img_name):
    """保存融合图像"""
    # 归一化到[0, 1]
    fused_img = jt.clamp(fused_img, 0, 1)
    
    # 转换为PIL图像
    if len(fused_img.shape) == 4:
        fused_img = fused_img.squeeze(0)
    
    if fused_img.shape[0] == 1:
        # 灰度图像
        img_array = (fused_img.squeeze(0) * 255).byte().numpy()
        image = Image.fromarray(img_array, mode='L')
    else:
        # RGB图像
        img_array = (fused_img * 255).byte().numpy().transpose(1, 2, 0)
        image = Image.fromarray(img_array, mode='RGB')
    
    # 保存
    save_path = os.path.join(save_dir, f'{img_name}_fused.png')
    image.save(save_path)

def save_segmentation_result(seg_out, save_dir, img_name):
    """保存分割结果"""
    # 获取预测类别
    if len(seg_out.shape) == 4:
        seg_pred = jt.argmax(seg_out, dim=1)
    else:
        seg_pred = seg_out
    
    # 转换为numpy数组
    if len(seg_pred.shape) == 3:
        seg_pred = seg_pred.squeeze(0)
    
    seg_array = seg_pred.numpy().astype(np.uint8)
    
    # 保存原始分割图
    seg_image = Image.fromarray(seg_array, mode='L')
    save_path = os.path.join(save_dir, f'{img_name}_seg.png')
    seg_image.save(save_path)
    
    # 保存彩色分割图
    colored_seg = apply_color_map(seg_array)
    colored_image = Image.fromarray(colored_seg, mode='RGB')
    save_path_colored = os.path.join(save_dir, f'{img_name}_seg_colored.png')
    colored_image.save(save_path_colored)

def apply_color_map(seg_array, num_classes=9):
    """应用颜色映射"""
    # 定义颜色映射
    colors = np.array([
        [0, 0, 0],       # 背景
        [128, 0, 0],     # 类别1 - 红色
        [0, 128, 0],     # 类别2 - 绿色
        [128, 128, 0],   # 类别3 - 黄色
        [0, 0, 128],     # 类别4 - 蓝色
        [128, 0, 128],   # 类别5 - 紫色
        [0, 128, 128],   # 类别6 - 青色
        [128, 128, 128], # 类别7 - 灰色
        [64, 0, 0],      # 类别8 - 深红色
    ], dtype=np.uint8)
    
    h, w = seg_array.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(min(num_classes, len(colors))):
        mask = seg_array == i
        colored[mask] = colors[i]
    
    return colored

def save_test_report(opt, num_images, total_time, avg_time, result_dir):
    """保存测试报告"""
    report_path = os.path.join(result_dir, 'test_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("PSFusion Test Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: PSF\n")
        f.write(f"Pretrained Model: {opt.pretrained_path}\n")
        f.write(f"Test Data: {opt.dataroot}\n")
        f.write(f"Number of Images: {num_images}\n")
        f.write(f"Crop Size: {opt.crop_size}\n")
        f.write(f"Number of Classes: {opt.n_classes}\n")
        f.write("\n")
        f.write("Performance Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Inference Time: {total_time:.4f}s\n")
        f.write(f"Average Inference Time: {avg_time:.4f}s\n")
        f.write(f"FPS: {1.0/avg_time:.2f}\n")
        f.write("\n")
        f.write(f"Results saved to: {result_dir}\n")
    
    print(f"Test report saved to: {report_path}")

def batch_test_fusion(data_dir, model_path, save_dir, crop_size=480):
    """批量测试图像融合"""
    # 创建模型
    model = PSF(n_classes=9)
    
    # 加载模型
    if os.path.exists(model_path):
        model = load_pretrained_model(model, model_path, strict=False)
        print(f"Model loaded from: {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        return
    
    model.eval()
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取图像文件列表
    vis_dir = os.path.join(data_dir, 'vi')
    ir_dir = os.path.join(data_dir, 'ir')
    
    if not os.path.exists(vis_dir) or not os.path.exists(ir_dir):
        print(f"Data directories not found: {vis_dir}, {ir_dir}")
        return
    
    vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Found {len(vis_files)} visible images and {len(ir_files)} infrared images")
    
    # 处理每对图像
    with jt.no_grad():
        for i, (vis_file, ir_file) in enumerate(zip(vis_files, ir_files)):
            print(f"Processing {i+1}/{len(vis_files)}: {vis_file}")
            
            # 加载图像
            vis_path = os.path.join(vis_dir, vis_file)
            ir_path = os.path.join(ir_dir, ir_file)
            
            vis_img = load_image(vis_path, mode='RGB').unsqueeze(0)
            ir_img = load_image(ir_path, mode='L').unsqueeze(0)
            
            # 调整尺寸
            vis_img = jt.nn.interpolate(vis_img, size=(crop_size, crop_size), mode='bilinear', align_corners=True)
            ir_img = jt.nn.interpolate(ir_img, size=(crop_size, crop_size), mode='bilinear', align_corners=True)
            
            # 模型推理
            semantic_out, binary_out, boundary_out, fused_img, vi_img, ir_img_pred = model(vis_img, ir_img)
            
            # 保存结果
            img_basename = os.path.splitext(vis_file)[0]
            save_fusion_image(fused_img, save_dir, img_basename)
    
    print(f"Batch testing completed! Results saved to: {save_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PSFusion Testing')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'batch'],
                       help='Testing mode: test (with dataset) or batch (direct image processing)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pretrained model')
    parser.add_argument('--save_dir', type=str, default='./test_results',
                       help='Directory to save results')
    parser.add_argument('--crop_size', type=int, default=480,
                       help='Input image size')
    parser.add_argument('--n_classes', type=int, default=9,
                       help='Number of segmentation classes')
    parser.add_argument('--save_seg', action='store_true',
                       help='Save segmentation results')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        # 使用TestOptions进行标准测试
        test_fusion()
    elif args.mode == 'batch':
        # 批量处理模式
        batch_test_fusion(args.data_dir, args.model_path, args.save_dir, args.crop_size)

if __name__ == '__main__':
    main()