import jittor as jt
from jittor import optim
import math

class Optimizer:
    def __init__(self, model, lr=0.01, momentum=0.9, weight_decay=1e-4, epochs=100, steps_per_epoch=100, pct_start=0.3):
        """
        自定义优化器，包含学习率调度
        
        Args:
            model: 要优化的模型
            lr: 初始学习率
            momentum: 动量
            weight_decay: 权重衰减
            epochs: 总训练轮数
            steps_per_epoch: 每轮的步数
            pct_start: 预热阶段占总训练的比例
        """
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        
        # 创建SGD优化器
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # 计算总步数和预热步数
        self.total_steps = epochs * steps_per_epoch
        self.warmup_steps = int(self.total_steps * pct_start)
        
        # 当前步数
        self.current_step = 0
        
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()
    
    def backward(self, loss):
        """反向传播"""
        self.optimizer.backward(loss)
    
    def step(self):
        """执行一步优化"""
        # 更新学习率
        self._update_lr()
        
        # 执行优化步骤
        self.optimizer.step()
        
        # 更新步数
        self.current_step += 1
    
    def _update_lr(self):
        """更新学习率"""
        if self.current_step < self.warmup_steps:
            # 预热阶段：线性增长
            lr = self.lr * (self.current_step / self.warmup_steps)
        else:
            # 余弦衰减阶段
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.lr * 0.5 * (1 + math.cos(math.pi * progress))
        
        # 设置新的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self):
        """获取优化器状态"""
        return {
            'optimizer': self.optimizer.state_dict(),
            'current_step': self.current_step,
            'lr': self.lr,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'steps_per_epoch': self.steps_per_epoch,
            'pct_start': self.pct_start
        }
    
    def load_state_dict(self, state_dict):
        """加载优化器状态"""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.current_step = state_dict['current_step']
        self.lr = state_dict['lr']
        self.momentum = state_dict['momentum']
        self.weight_decay = state_dict['weight_decay']
        self.epochs = state_dict['epochs']
        self.steps_per_epoch = state_dict['steps_per_epoch']
        self.pct_start = state_dict['pct_start']
        
        # 重新计算总步数和预热步数
        self.total_steps = self.epochs * self.steps_per_epoch
        self.warmup_steps = int(self.total_steps * self.pct_start)

class CosineAnnealingWarmRestarts:
    """余弦退火学习率调度器（带热重启）"""
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.T_cur + 1
            self.T_cur = epoch
        else:
            self.T_cur = epoch
            
        if self.T_cur >= self.T_i:
            # 重启
            self.T_cur = self.T_cur - self.T_i
            self.T_i = self.T_i * self.T_mult
            
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * \
                               (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2

class LinearWarmupCosineAnnealingLR:
    """线性预热 + 余弦退火学习率调度器"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=0, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # 预热阶段
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.warmup_start_lr + \
                                   (base_lr - self.warmup_start_lr) * epoch / self.warmup_epochs
        else:
            # 余弦退火阶段
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * \
                                   (1 + math.cos(math.pi * (epoch - self.warmup_epochs) / 
                                                (self.max_epochs - self.warmup_epochs))) / 2

def get_optimizer(model, opt_type='sgd', lr=0.01, momentum=0.9, weight_decay=1e-4):
    """获取优化器"""
    if opt_type.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif opt_type.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif opt_type.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

def get_scheduler(optimizer, scheduler_type='cosine', **kwargs):
    """获取学习率调度器"""
    if scheduler_type.lower() == 'cosine':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 1),
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_type.lower() == 'warmup_cosine':
        return LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=kwargs.get('warmup_epochs', 10),
            max_epochs=kwargs.get('max_epochs', 100),
            warmup_start_lr=kwargs.get('warmup_start_lr', 0),
            eta_min=kwargs.get('eta_min', 0)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")