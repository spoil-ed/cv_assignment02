import torch
import os
from datetime import datetime
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import time
import logging

logger = logging.getLogger(__name__)

def save_training_metrics(best_val_acc, train_losses, val_losses, train_accs, val_accs, save_dir='experiments'):
    """
    保存训练指标到指定目录，文件名包含时间戳
    
    参数:
        best_val_acc: 最佳验证准确率
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
        save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存路径
    save_path = os.path.join(save_dir, f'metrics_scratch_{timestamp}.pt')
    
    # 准备保存的数据
    metrics = {
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    # 保存到文件
    try:
        torch.save(metrics, save_path)
        logger.info(f"训练指标已保存至 {save_path}")
    except Exception as e:
        logger.error(f"保存训练指标失败: {e}")
        raise

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

    indices = list(range(len(dataset)))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42)
    logger.info(f"训练样本: {len(train_idx)}, 验证样本: {len(val_idx)}, 测试样本: {len(test_idx)}")

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

def load_model_weights(model, weight_path, device='cpu'):
    """
    从本地加载模型权重
    
    参数:
        model: 未加载权重的模型实例
        weight_path: 权重文件的路径（如 'path/to/model_weights.pth'）
        device: 加载权重的设备（'cpu' 或 'cuda'）
    
    返回:
        加载了权重的模型
    """
    try:
        # 确保模型在正确的设备上
        model = model.to(device)
        
        # 加载权重文件
        checkpoint = torch.load(weight_path, map_location=device)
        
        # 如果权重文件包含整个模型状态字典
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 直接加载权重
            model.load_state_dict(checkpoint)
        
        # 设置模型为评估模式
        model.eval()
        logger.info(f"成功从 {weight_path} 加载模型权重")
        return model
    
    except Exception as e:
        logger.error(f"加载模型权重失败: {e}")
        raise