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

def get_data_loaders(data_dir, batch_size=32):
    logger.info("开始加载 Caltech-101 数据集")
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    logger.info(f"从 {data_dir} 加载数据集")
    start_time = time.time()
    dataset = datasets.Caltech101(root=data_dir, download=True)
    logger.info(f"加载数据集耗时: {time.time() - start_time:.2f} 秒")

    # 分割数据集：训练、验证、测试集
    indices = list(range(len(dataset)))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42)  # 80%训练，20%验证
    logger.info(f"训练样本: {len(train_idx)}, 验证样本: {len(val_idx)}, 测试样本: {len(test_idx)}")

    # 验证索引有效性
    assert max(train_idx) < len(dataset), "训练索引超出数据集范围"
    assert max(val_idx) < len(dataset), "验证索引超出数据集范围"
    assert max(test_idx) < len(dataset), "测试索引超出数据集范围"

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    dataset.transform = None
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    test_dataset.dataset.transform = data_transforms['test']

    num_workers = 2
    try:
        DataLoader(train_dataset, batch_size=1, num_workers=num_workers)
    except Exception as e:
        logger.warning(f"多线程加载失败: {e}，切换到单线程")
        num_workers = 0

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    }

    dataset_sizes = {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)}
    logger.info("数据加载器创建成功")
    return dataloaders, dataset_sizes

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, device, writer, patience=5, delta=0, save_path='models/best_model.pth'):
    logger.info("开始模型训练")
    best_val_acc = 0.0
    best_model_wts = model.state_dict()
    counter = 0
    early_stop = False

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        logger.info(f'第 {epoch}/{num_epochs - 1} 轮')
        logger.info('-' * 10)
        if early_stop:
            break
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
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(best_model_wts, save_path)
                    logger.info(f"保存最佳模型到 {save_path}, 验证准确率: {best_val_acc:.4f}")
                    counter = 0
                else:
                    counter += 1
                    if counter > patience:
                        early_stop = True
                        logger.info(f"早停触发，停止训练")
        if phase == 'train':
            scheduler.step()
    logger.info(f'最佳验证准确率: {best_val_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, best_val_acc, train_losses, val_losses, train_accs, val_accs

def evaluate_test_accuracy(model, test_loader, dataset_sizes, device):
    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Test'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    test_acc = running_corrects.double() / dataset_sizes['test']
    logger.info(f'测试准确率: {test_acc:.4f}')
    return test_acc

def grid_search_hyperparams(dataloaders, dataset_sizes, device, num_epochs, lr_base_list, lr_fc_list, weight_decay_list, from_scratch=False):
    results = []
    for lr_base in lr_base_list:
        for lr_fc in lr_fc_list:
            for weight_decay in weight_decay_list:
                logger.info(f"测试超参数: lr_base={lr_base}, lr_fc={lr_fc}, weight_decay={weight_decay}")
                weights = None if from_scratch else ResNet18_Weights.IMAGENET1K_V1
                model = models.resnet18(weights=weights)
                model.fc = nn.Linear(model.fc.in_features, 101)
                model = model.to(device)
                criterion = nn.CrossEntropyLoss(weight=torch.ones(101).to(device), reduction='mean')
                optimizer = optim.SGD([
                    {'params': [p for name, p in model.named_parameters() if 'fc' not in name], 'lr': lr_base},
                    {'params': model.fc.parameters(), 'lr': lr_fc}
                ], momentum=0.9, weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                writer = SummaryWriter(f'/home/spoil/cv/assignment02/task1/runs/{timestamp}_lr_base_{lr_base}_lr_fc_{lr_fc}_wd_{weight_decay}')
                model, best_val_acc, _, _, _, _ = train_model(
                    model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
                    num_epochs, device, writer, save_path=f'models/model_lr{lr_base}_fc{lr_fc}_wd{weight_decay}.pth'
                )
                test_acc = evaluate_test_accuracy(model, dataloaders['test'], dataset_sizes, device)
                results.append((lr_base, lr_fc, weight_decay, best_val_acc, test_acc))
                writer.close()
    return results

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

def compute_class_accuracy(model, test_loader, dataset_sizes, device, save_path='images/class_accuracy.png'):
    model.eval()
    class_correct = torch.zeros(101).to(device)
    class_total = torch.zeros(101).to(device)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for label, pred in zip(labels, preds):
                class_correct[label] += (pred == label).item()
                class_total[label] += 1
    class_acc = class_correct / (class_total + 1e-10)
    class_acc = class_acc.cpu().numpy()
    plt.figure(figsize=(12, 6))
    plt.bar(range(101), class_acc)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Class Index')
    plt.ylabel('Accuracy')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"类别准确率图保存至 {save_path}")
    return class_acc

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

    if args.grid_search:
        lr_base_list = [0.01, 0.001, 0.0001]
        lr_fc_list = [0.1, 0.01, 0.001]
        weight_decay_list = [0.01, 0.001, 0.0001]
        results_finetune = grid_search_hyperparams(dataloaders, dataset_sizes, device, args.epochs, lr_base_list, lr_fc_list, weight_decay_list, from_scratch=False)
        results_scratch = grid_search_hyperparams(dataloaders, dataset_sizes, device, args.epochs, lr_base_list, lr_fc_list, weight_decay_list, from_scratch=True)
        visualize_hyperparam_results(results_finetune, 'images/hyperparam_heatmap_finetune.png')
        visualize_hyperparam_results(results_scratch, 'images/hyperparam_heatmap_scratch.png')
        best_result = max(results_finetune, key=lambda x: x[4])
        logger.info(f"微调最佳超参数: lr_base={best_result[0]}, lr_fc={best_result[1]}, weight_decay={best_result[2]}, 验证准确率={best_result[3]:.4f}, 测试准确率={best_result[4]:.4f}")
        best_result_scratch = max(results_scratch, key=lambda x: x[4])
        logger.info(f"从头训练最佳超参数: lr_base={best_result_scratch[0]}, lr_fc={best_result_scratch[1]}, weight_decay={best_result_scratch[2]}, 验证准确率={best_result_scratch[3]:.4f}, 测试准确率={best_result_scratch[4]:.4f}")
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

        visualize_conv1_kernels(model, 'images/conv1_kernels_scratch.png')
        visualize_fc_weights(model, 'images/fc_weights_scratch.png')
        visualize_param_distribution(model, 'images/weight_distribution_scratch.png')
        class_acc_scratch = compute_class_accuracy(model, dataloaders['test'], dataset_sizes, device, 'images/class_accuracy_scratch.png')

if __name__ == '__main__':
    main()