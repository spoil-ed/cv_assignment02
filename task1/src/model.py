import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights

def get_model(device, num_classes=101, from_scratch=False, lr_base=0.001, lr_fc=0.01, weight_decay=0.001):
    weights = None if from_scratch else ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=torch.ones(num_classes).to(device), reduction='mean')
    optimizer = optim.SGD([
        {'params': [p for name, p in model.named_parameters() if 'fc' not in name], 'lr': lr_base},
        {'params': model.fc.parameters(), 'lr': lr_fc}
    ], momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    return model, criterion, optimizer, scheduler