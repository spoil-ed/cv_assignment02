import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from test import evaluate_test_accuracy
from model import get_model

logger = logging.getLogger(__name__)

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

def grid_search_hyperparams(dataloaders, dataset_sizes, device, num_epochs, lr_base_list, lr_fc_list, weight_decay_list, from_scratch=False):
    results = []
    for lr_base in lr_base_list:
        for lr_fc in lr_fc_list:
            for weight_decay in weight_decay_list:
                logger.info(f"测试超参数: lr_base={lr_base}, lr_fc={lr_fc}, weight_decay={weight_decay}")
                model, criterion, optimizer, scheduler = get_model(device, num_classes=101, from_scratch=from_scratch, lr_base=lr_base, lr_fc=lr_fc, weight_decay=weight_decay)
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