import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import os
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model import get_model
from utils import get_data_loaders

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, device, writer):
    """训练模型并返回每轮的验证准确率"""
    logger.info("开始模型训练")
    best_model_wts = model.state_dict()
    best_val_acc = 0.0
    val_accs = []

    for epoch in range(num_epochs):
        logger.info(f'第 {epoch}/{num_epochs - 1} 轮')
        logger.info('-' * 10)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            logger.info(f'{phase} 损失: {epoch_loss:.4f} 准确率: {epoch_acc:.4f}')
            if phase == 'val':
                val_accs.append(epoch_acc.item())
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = model.state_dict()
        scheduler.step()
    model.load_state_dict(best_model_wts)
    return model, best_val_acc, val_accs

def plot_boxplot(val_accs_by_param, param_name, param_values, output_path):
    """绘制验证准确率的箱式图"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    sns.set(style="whitegrid", context="notebook", font_scale=1.2)
    
    plt.figure(figsize=(12, 7), dpi=100)
    data = []
    for param_val, accs in zip(param_values, val_accs_by_param.values()):
        data.extend([(param_val, acc) for acc in accs])
    df = pd.DataFrame(data, columns=[param_name, 'val_acc'])
    
    palette = sns.color_palette("Set2", n_colors=len(val_accs_by_param))
    sns.boxplot(x=param_name, y='val_acc', data=df, palette=palette, width=0.6, fliersize=5,
                boxprops=dict(alpha=0.8),
                whiskerprops=dict(linestyle='--', linewidth=1.5),
                capprops=dict(linewidth=1.5))
    
    sns.stripplot(x=param_name, y='val_acc', data=df, color='black', size=4, alpha=0.5, jitter=True)
    
    counts = df.groupby(param_name).size()
    for i, val in enumerate(counts.index):
        plt.text(i, df['val_acc'].max() + 0.02, f'n={counts[val]}', ha='center', fontsize=10, color='black')
    
    plt.title(f'Validation Accuracy vs. {param_name.replace("_", " ").title()}', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel(param_name.replace("_", " ").title(), fontsize=14, labelpad=10)
    plt.ylabel('Validation Accuracy', fontsize=14, labelpad=10)
    
    # 计算箱体范围（基于所有数据的四分位数）
    q1 = df['val_acc'].quantile(0.25)
    q3 = df['val_acc'].quantile(0.75)
    iqr = q3 - q1
    margin = iqr * 0.1  # 增加 10% 的 IQR 作为边界扩展
    ymin = q1 - 1.5 * iqr - margin
    ymax = q3 + 1.5 * iqr + margin
    plt.ylim(ymin, ymax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"箱式图已保存至 {output_path}")
    plt.close()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 数据路径
    data_dir = '/home/spoil/cv/assignment02/task1/data'  # 替换为实际数据路径
    
    # 超参数组合
    hyperparams = {
        'lr_base': [0.01, 0.001, 0.0001],
        'batch_size': [16, 32, 64],
        'weight_decay': [0.01, 0.001, 0.0001],
        'num_epochs': [10, 20, 30]
    }
    
    # 基准超参数
    base_params = {
        'lr_base': 0.001,
        'batch_size': 32,
        'weight_decay': 0.001,
        'num_epochs': 20,
        'lr_fc': 0.01
    }
    
    # 为每个超参数进行实验
    for param_name in hyperparams.keys():
        logger.info(f"开始实验超参数: {param_name}")
        val_accs_by_param = {}
        param_values = hyperparams[param_name]
        
        for param_value in param_values:
            logger.info(f"训练 {param_name}={param_value}")
            
            # 更新实验参数
            exp_params = base_params.copy()
            exp_params[param_name] = param_value
            
            # 获取数据加载器
            dataloaders, dataset_sizes = get_data_loaders(data_dir, exp_params['batch_size'])
            
            # 获取模型和优化器
            model, criterion, optimizer, scheduler = get_model(
                device,
                lr_base=exp_params['lr_base'],
                lr_fc=exp_params['lr_fc'],
                weight_decay=exp_params['weight_decay']
            )
            
            # TensorBoard 记录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            writer = SummaryWriter(f'runs/train_{timestamp}_{param_name}_{param_value}')
            
            # 训练模型
            model, best_val_acc, val_accs = train_model(
                model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
                exp_params['num_epochs'], device, writer
            )
            
            val_accs_by_param[param_value] = val_accs
            writer.close()
            logger.info(f"{param_name}={param_value} 最佳验证准确率: {best_val_acc:.4f}")
        
        # 绘制箱式图
        output_path = f'images/boxplot_val_acc_{param_name}.png'
        plot_boxplot(val_accs_by_param, param_name, param_values, output_path)

if __name__ == '__main__':
    main()