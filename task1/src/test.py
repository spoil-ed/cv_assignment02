import torch
import torch.nn as nn
import torchvision.models as models
import os
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
    return class_acc

def test_pretrained_model(weight_path, test_loader, dataset_sizes, device, class_acc_save_path='images/class_accuracy_test.png'):
    """
    加载本地预训练模型权重并进行测试评估
    
    参数:
        weight_path: 预训练模型权重的路径（如 'models/caltech101_resnet18_finetune.pth'）
        test_loader: 测试集数据加载器
        dataset_sizes: 数据集大小字典，包含 'test' 键
        device: 运行设备（'cpu' 或 'cuda'）
        class_acc_save_path: 类别准确率图保存路径
    
    返回:
        test_acc: 测试集总体准确率
        class_acc: 每个类别的准确率
    """
    try:
        # 初始化 ResNet-18 模型
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 101)  # 适配 Caltech-101 的 101 个类别
        model = model.to(device)
        
        # 加载权重
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        logger.info(f"成功从 {weight_path} 加载模型权重")
        
        # 计算测试集准确率
        test_acc = evaluate_test_accuracy(model, test_loader, dataset_sizes, device)
        
        # 计算类别准确率并保存图表
        class_acc = compute_class_accuracy(model, test_loader, dataset_sizes, device, save_path=class_acc_save_path)
        
        # 绘制类别准确率图
        plt.figure(figsize=(12, 6))
        plt.bar(range(101), class_acc)
        plt.title('Per-Class Accuracy')
        plt.xlabel('Class Index')
        plt.ylabel('Accuracy')
        os.makedirs(os.path.dirname(class_acc_save_path), exist_ok=True)
        plt.savefig(class_acc_save_path)
        plt.close()
        logger.info(f"类别准确率图保存至 {class_acc_save_path}")
        
        return test_acc, class_acc
    
    except Exception as e:
        logger.error(f"测试预训练模型失败: {e}")
        raise

# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test.log'),
            logging.StreamHandler()
        ]
    )
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 假设已有数据加载器和数据集大小
    # dataloaders, dataset_sizes = get_data_loaders('/home/spoil/cv/assignment02/task1/data', batch_size=32)
    # test_loader = dataloaders['test']
    # dataset_sizes = dataset_sizes
    
    # 替换为实际的测试数据加载器和数据集大小
    # 以下为占位符，需替换为实际数据
    test_loader = None  # 替换为实际的 test_loader
    dataset_sizes = {'test': 0}  # 替换为实际的 dataset_sizes
    
    # 权重文件路径
    weight_path = 'models/best_model.pth'
    
    # 运行测试
    test_acc, class_acc = test_pretrained_model(
        weight_path=weight_path,
        test_loader=test_loader,
        dataset_sizes=dataset_sizes,
        device=device,
        class_acc_save_path='images/class_accuracy_test.png'
    )