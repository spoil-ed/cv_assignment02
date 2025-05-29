import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np 
import logging
import time
import uuid
from datetime import datetime
from train import train_model, grid_search_hyperparams
from test import evaluate_test_accuracy, compute_class_accuracy
from utils import get_data_loaders, save_training_metrics
from model import get_model
from visualize import visualize_hyperparam_results, visualize_conv1_kernels, visualize_fc_weights, visualize_param_distribution, visualize_training_history

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/spoil/cv/assignment02/task1/data', help='数据集目录')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
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

    if args.grid_search:
        lr_base_list = [0.01, 0.001, 0.0001]
        lr_fc_list = [0.1, 0.01, 0.001]
        weight_decay_list = [0.001]
        results_finetune = grid_search_hyperparams(dataloaders, dataset_sizes, device, args.epochs, lr_base_list, lr_fc_list, weight_decay_list, from_scratch=False)
        visualize_hyperparam_results(results_finetune, 'images/hyperparam_heatmap_finetune.png')
        best_result = max(results_finetune, key=lambda x: x[4])
        logger.info(f"微调最佳超参数: lr_base={best_result[0]}, lr_fc={best_result[1]}, weight_decay={best_result[2]}, 验证准确率={best_result[3]:.4f}, 测试准确率={best_result[4]:.4f}")
    else:
        # 微调模型
        logger.info("加载预训练的 ResNet-18 模型")
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 101)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.ones(101).to(device), reduction='mean')
        optimizer = optim.SGD([
            {'params': [p for name, p in model.named_parameters() if 'fc' not in name], 'lr': args.lr_base},
            {'params': model.fc.parameters(), 'lr': args.lr_fc}
        ], momentum=0.9, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter(f'/home/spoil/cv/assignment02/task1/runs/finetune_{timestamp}_{uuid.uuid4()}')
        model, best_val_acc, train_losses, val_losses, train_accs, val_accs = train_model(
            model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
            args.epochs, device, writer, save_path='models/caltech101_resnet18_finetune.pth'
        )
        test_acc_finetune = evaluate_test_accuracy(model, dataloaders['test'], dataset_sizes, device)
        writer.close()

        # 保存微调模型的指标
        save_training_metrics(
            best_val_acc, 
            train_losses, 
            val_losses, 
            train_accs, 
            val_accs, 
            save_dir='experiments'
        )

        # 绘制微调模型的训练历史
        visualize_training_history(train_losses, val_losses, train_accs, val_accs, 'images/training_history_finetune.png')

        # 可视化
        visualize_conv1_kernels(model, 'images/conv1_kernels_finetune.png')
        visualize_fc_weights(model, 'images/fc_weights_finetune.png')
        visualize_param_distribution(model, 'images/weight_distribution_finetune.png')
        class_acc_finetune = compute_class_accuracy(model, dataloaders['test'], dataset_sizes, device, 'images/class_accuracy_finetune.png')

        # 从头训练模型
        logger.info("加载随机初始化的 ResNet-18 模型")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 101)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.ones(101).to(device), reduction='mean')
        optimizer = optim.SGD(model.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter(f'/home/spoil/cv/assignment02/task1/runs/scratch_{timestamp}_{uuid.uuid4()}')
        model, best_val_acc_scratch, train_losses_scratch, val_losses_scratch, train_accs_scratch, val_accs_scratch = train_model(
            model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
            args.epochs, device, writer, save_path='models/caltech101_resnet18_scratch.pth'
        )
        test_acc_scratch = evaluate_test_accuracy(model, dataloaders['test'], dataset_sizes, device)
        writer.close()

        # 保存从头训练的指标
        save_training_metrics(
            best_val_acc_scratch, 
            train_losses_scratch, 
            val_losses_scratch, 
            train_accs_scratch, 
            val_accs_scratch, 
            save_dir='experiments'
        )

        # 绘制从头训练的训练历史
        visualize_training_history(train_losses_scratch, val_losses_scratch, train_accs_scratch, val_accs_scratch, 'images/training_history_scratch.png')

        visualize_conv1_kernels(model, 'images/conv1_kernels_scratch.png')
        visualize_fc_weights(model, 'images/fc_weights_scratch.png')
        visualize_param_distribution(model, 'images/weight_distribution_scratch.png')
        class_acc_scratch = compute_class_accuracy(model, dataloaders['test'], dataset_sizes, device, 'images/class_accuracy_scratch.png')

if __name__ == '__main__':
    main()