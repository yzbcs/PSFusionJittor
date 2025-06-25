import jittor as jt
from jittor import nn
from jittor import optim
import numpy as np
import json
import os
from datetime import datetime
from PSF_jittor import PSF
from create_dataset_jittor import get_loader
from losses_jittor import Fusionloss, OhemCELoss
from optimizer_jittor import Optimizer
from saver_jittor import Saver
from utils_jittor import SegmentationMetric
from options import TrainOptions

# 设置Jittor使用GPU
jt.flags.use_cuda = 1

def main():
    # 解析命令行参数
    opt = TrainOptions().parse()
    
    # 创建保存器
    saver = Saver(opt)
    
    # 创建数据加载器
    train_loader = get_loader(
        opt, 
        is_train=True, 
        batch_size=opt.batch_size, 
        num_workers=opt.num_workers
    )
    
    val_loader = get_loader(
        opt, 
        is_train=False, 
        batch_size=1, 
        num_workers=opt.num_workers
    )
    
    # 创建模型
    model = PSF(n_classes=opt.class_nb)
    
    # 创建损失函数
    fusion_loss = Fusionloss()
    seg_loss = OhemCELoss(ignore_label=255)
    
    # 创建优化器
    optimizer = Optimizer(
        model=model,
        lr=opt.lr,
        weight_decay=1e-4,  # 设置默认值
        epochs=opt.max_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3  # 设置默认值
    )
    
    # 创建日志文件
    log_dir = os.path.join(opt.result_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_log_{timestamp}.json')
    
    # 初始化日志数据
    log_data = {
        'train_losses': [],
        'train_fusion_losses': [],
        'train_seg_losses': [],
        'val_losses': [],
        'val_mious': [],
        'val_accs': [],
        'epochs': []
    }
    
    # 训练循环
    for epoch in range(opt.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_fusion_loss = 0.0
        epoch_seg_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            if len(batch) == 6:  # 训练数据
                ir_img, vis_img, label, bi, bd, mask = batch
            else:  # 测试数据
                ir_img, vis_img, label, _ = batch
                bi = bd = mask = None
            
            # 确保张量有正确的维度 [N, C, H, W]
            # vis_img已经是[N, C, H, W]格式
            # ir_img需要添加通道维度，从[N, H, W]变为[N, 1, H, W]
            if len(ir_img.shape) == 3:  # [N, H, W]
                ir_img = ir_img.unsqueeze(1)  # [N, 1, H, W]
            
            # 前向传播
            semantic_out, binary_out, boundary_out, fused_img, vi_img, ir_img_pred = model(vis_img, ir_img)
            # 为了兼容原有的损失计算，将语义分割结果组合
            seg_result = (semantic_out, binary_out, boundary_out)
            fusion_img = fused_img
            
            # 计算损失
            fusion_loss_val = fusion_loss(vis_img, ir_img, fusion_img)
            seg_loss_val = seg_loss(seg_result, label)
            total_loss = fusion_loss_val + seg_loss_val
            
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(total_loss)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_fusion_loss += fusion_loss_val.item()
            epoch_seg_loss += seg_loss_val.item()
            
            if (i + 1) % opt.print_freq == 0:
                print(f'Epoch [{epoch+1}/{opt.max_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {total_loss.item():.4f}, Fusion: {fusion_loss_val.item():.4f}, '
                      f'Seg: {seg_loss_val.item():.4f}')
        
        # 记录训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        avg_fusion_loss = epoch_fusion_loss / len(train_loader)
        avg_seg_loss = epoch_seg_loss / len(train_loader)
        
        log_data['train_losses'].append(avg_train_loss)
        log_data['train_fusion_losses'].append(avg_fusion_loss)
        log_data['train_seg_losses'].append(avg_seg_loss)
        log_data['epochs'].append(epoch + 1)
        
        # 验证
        if epoch % opt.val_freq == 0:
            model.eval()
            val_loss = 0.0
            seg_metric = SegmentationMetric(opt.class_nb)
            lb_ignore = [255]
            
            with jt.no_grad():
                for batch in val_loader:
                    if len(batch) == 6:
                        ir_img, vis_img, label, bi, bd, mask = batch
                    else:
                        ir_img, vis_img, label, _ = batch
                    
                    # 确保ir_img有正确的维度
                    if len(ir_img.shape) == 3:
                        ir_img = ir_img.unsqueeze(1)
                    
                    semantic_out, binary_out, boundary_out, fused_img, vi_img, ir_img_pred = model(vis_img, ir_img)
                    
                    # 计算语义分割预测结果
                    seg_result_pred = jt.argmax(semantic_out, dim=1)[0]  # 取第一个元素
                    seg_result_pred = seg_result_pred.unsqueeze(1)  # 添加通道维度
                    seg_metric.addBatch(seg_result_pred, label, lb_ignore)
                    
                    # 为了兼容原有的损失计算，将语义分割结果组合
                    seg_result = (semantic_out, binary_out, boundary_out)
                    fusion_img = fused_img
                    fusion_loss_val = fusion_loss(vis_img, ir_img, fusion_img)
                    seg_loss_val = seg_loss(seg_result, label)
                    total_loss_val = fusion_loss_val + seg_loss_val
                    val_loss += total_loss_val.item()
            
            # 计算评估指标
            mIoU = seg_metric.meanIntersectionOverUnion().item()
            Acc = seg_metric.pixelAccuracy().item()
            avg_val_loss = val_loss / len(val_loader)
            
            # 记录验证指标
            log_data['val_losses'].append(avg_val_loss)
            log_data['val_mious'].append(mIoU)
            log_data['val_accs'].append(Acc)
            
            print(f'Validation - Loss: {avg_val_loss:.4f}, mIoU: {mIoU:.4f}, Acc: {Acc:.4f}')
        
        # 保存模型
        if epoch % opt.model_save_freq == 0:
            saver.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss
            }, epoch)
        
        # 保存日志数据
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f'Epoch [{epoch+1}/{opt.max_epochs}] completed, Average Loss: {avg_train_loss:.4f}')
    
    print('Training completed!')
    
    # 保存最终模型
    saver.save_checkpoint({
        'epoch': opt.max_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_train_loss
    }, 'final')

if __name__ == '__main__':
    main()