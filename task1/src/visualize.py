import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging
import torch
import argparse
import torch.nn as nn
from utils import load_model_weights, get_data_loaders
from test import compute_class_accuracy
from torchvision import datasets, models, transforms
from model import get_model

logger = logging.getLogger(__name__)

def visualize_hyperparam_results(results, save_path='images/hyperparam_heatmap.png'):
    lr_base_values = sorted(set(r[0] for r in results))
    lr_fc_values = sorted(set(r[1] for r in results))
    heatmap_data = np.zeros((len(lr_fc_values), len(lr_base_values)))
    for lr_base, lr_fc, _, _, test_acc in results:
        i = lr_fc_values.index(lr_fc)
        j = lr_base_values.index(lr_base)
        heatmap_data[i, j] = test_acc
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, xticklabels=lr_base_values, yticklabels=lr_fc_values, annot=True, fmt='.4f', cmap='viridis')
    plt.title('Test Accuracy vs. Learning Rates')
    plt.xlabel('Base Learning Rate')
    plt.ylabel('FC Learning Rate')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"超参数热图保存至 {save_path}")

def visualize_conv1_kernels(model, save_path='images/conv1_kernels.png', num_kernels=16):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    conv1_weight = model.conv1.weight.data.cpu().numpy()
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        if i < num_kernels:
            kernel = conv1_weight[i, 0, :, :]
            kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
            ax.imshow(kernel, cmap='gray')
            ax.axis('off')
            ax.set_title(f'Kernel {i}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"第一层卷积核可视化保存至 {save_path}")

def visualize_fc_weights(model, save_path='images/fc_weights.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fc_weight = model.fc.weight.data.cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(fc_weight, cmap='viridis')
    plt.title('Fully Connected Layer Weights')
    plt.xlabel('Input Features')
    plt.ylabel('Classes')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"全连接层权重热图保存至 {save_path}")

def visualize_param_distribution(model, save_path='images/weight_distribution.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            params.append(param.data.cpu().numpy().flatten())
    params = np.concatenate(params)
    plt.figure(figsize=(8, 6))
    plt.hist(params, bins=50, density=True)
    plt.title('Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"权重分布直方图保存至 {save_path}")

def visualize_training_history(train_losses, val_losses, train_accs, val_accs, save_path='images/training_history.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in train_accs], 'b-', label='Train Accuracy')
    plt.plot(epochs, [acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in val_accs], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"训练历史曲线保存至 {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/spoil/cv/assignment02/task1/data', help='数据集目录')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--lr-base', type=float, default=0.001, help='基础学习率')
    parser.add_argument('--lr-fc', type=float, default=0.01, help='全连接层学习率')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='权重衰减')
    parser.add_argument('--from-scratch', action='store_true', help='从头训练')
    parser.add_argument('--grid-search', action='store_true', help='执行超参数网格搜索')
    args = parser.parse_args()

    logger.info(f"参数: {vars(args)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    dataloaders, dataset_sizes = get_data_loaders(args.data_dir, args.batch_size)
    
    model = models.resnet18(weights=None)  # 不加载预训练权重
    model.fc = nn.Linear(model.fc.in_features, 101)  # 适配 Caltech-101 的 101 个类别
    model = model.to(device)

    weight_path = 'models/caltech101_resnet18_scratch.pth'  # 权重文件路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model_weights(model, weight_path, device)
    # 可视化
    visualize_conv1_kernels(model, 'images/conv1_kernels.png')
    visualize_fc_weights(model, 'images/fc_weights.png')
    visualize_param_distribution(model, 'images/weight_distribution.png')
    class_acc_finetune = compute_class_accuracy(model, dataloaders['test'], dataset_sizes, device, 'images/class_accuracy.png')

if __name__ == '__main__':
    main()