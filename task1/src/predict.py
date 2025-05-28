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
from tqdm import tqdm
import numpy as np
import logging
import time

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
    # 数据预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224), # 随机裁剪
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),  # 随机翻转
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 根据 Image-Net 均值和标准差实现标准化
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 加载 Caltech-101 数据集
    logger.info(f"从 {data_dir} 加载数据集")
    start_time = time.time()
    dataset = datasets.Caltech101(root=data_dir, download=True)
    logger.info(f"加载数据集耗时: {time.time() - start_time:.2f} 秒")

    # 分割数据集
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    logger.info(f"训练样本: {len(train_idx)}, 测试样本: {len(test_idx)}")

    # 验证索引有效性
    assert max(train_idx) < len(dataset), "训练索引超出数据集范围"
    assert max(test_idx) < len(dataset), "测试索引超出数据集范围"

    # 创建训练和测试数据集，分别应用变换
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    # 为数据集应用变换
    dataset.transform = None  # 确保原始数据集无变换
    train_dataset.dataset.transform = data_transforms['train']
    test_dataset.dataset.transform = data_transforms['test']

    # 动态设置 num_workers
    num_workers = 4
    try:
        DataLoader(train_dataset, batch_size=1, num_workers=num_workers)
    except Exception as e:
        logger.warning(f"多线程加载失败: {e}，切换到单线程")
        num_workers = 0

    # 创建 DataLoader
    train_loaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loaders = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_idx),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(test_idx),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    }

    dataset_sizes = {'train': len(train_idx), 'test': len(test_idx)}
    logger.info("数据加载器创建成功")
    return train_loaders, test_loaders, dataset_sizes

def train_model(model, train_loaders, test_loaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, device, writer, patience=5, delta=0):
    logger.info("开始模型训练")
    best_train_acc = 0.0
    best_test_acc = 0.0
    best_model_wts = model.state_dict()

    counter = 0 # 记录早停
    early_stop = False

    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        logger.info(f'第 {epoch}/{num_epochs - 1} 轮')
        logger.info('-' * 10)
        if early_stop:
            break
        for phase in ["train", "test"]:
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(train_loaders if phase == "train" else test_loaders, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                logger.debug(f"加载批次: 输入形状 {inputs.shape}, 标签形状 {labels.shape}")

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            logger.info(f'{phase} 损失: {epoch_loss:.4f} 准确率: {epoch_acc:.4f}')
            if phase == "train":
                train_accs.append(epoch_acc)
                train_losses.append(epoch_loss)
                if epoch_acc > best_train_acc:
                    best_train_acc = epoch_acc
            else:
                test_accs.append(epoch_acc)
                test_losses.append(epoch_loss)
                if epoch_acc > best_test_acc:
                    best_test_acc = epoch_acc
                    counter = 0
                if epoch_acc < best_test_acc + delta:
                    counter += 1
                    if counter > patience:
                        early_stop = True
    logger.info(f'最佳测试准确率: {best_test_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/spoil/cv/assignment02/task1/data', help='数据集目录')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--lr-base', type=float, default=0.001, help='基础学习率')
    parser.add_argument('--lr-fc', type=float, default=0.01, help='全连接层学习率')
    parser.add_argument('--from-scratch', action='store_true', help='从头训练')
    args = parser.parse_args()

    logger.info(f"参数: {vars(args)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    train_loaders, test_loaders, dataset_sizes = get_data_loaders(args.data_dir, args.batch_size)

    ### 预训练
    logger.info("加载预训练的 ResNet-18 模型")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 101)
    logger.info(f"修改全连接层: {num_ftrs} -> 101")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD([
        {'params': [p for name, p in model.named_parameters() if 'fc' not in name], 'lr': args.lr_base},
        {'params': model.fc.parameters(), 'lr': args.lr_fc}
    ], momentum=0.9)
    logger.info(f"微调优化器: 基础学习率={args.lr_base}, 全连接层学习率={args.lr_fc}")

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    logger.info("学习率调度器初始化")

    writer = SummaryWriter()
    logger.info("TensorBoard writer 初始化")

    model = train_model(
        model, train_loaders, test_loaders, dataset_sizes, criterion, optimizer, scheduler,
        args.epochs, device, writer
    )

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/caltech101_resnet18.pth')
    logger.info("模型保存至 models/caltech101_resnet18.pth")

    writer.close()

    ### 随机初始化
    logger.info("加载随机初始化的 ResNet-18 模型")
    weights = None 
    model = models.resnet18(weights=weights)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 101)
    logger.info(f"修改全连接层: {num_ftrs} -> 101")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr_base, momentum=0.9)
    logger.info(f"从头训练优化器: 学习率={args.lr_base}")

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    logger.info("学习率调度器初始化")

    writer = SummaryWriter()
    logger.info("TensorBoard writer 初始化")

    model = train_model(
        model, train_loaders, test_loaders, dataset_sizes, criterion, optimizer, scheduler,
        args.epochs, device, writer
    )

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/caltech101_resnet18.pth')
    logger.info("模型保存至 models/caltech101_resnet18.pth")

    writer.close()

if __name__ == '__main__':
    main()