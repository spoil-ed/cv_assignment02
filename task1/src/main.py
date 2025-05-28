from PIL import Image
import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.models as models

def resize_images(input_dir, output_dir, size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = Image.open(img_path).resize(size)
            img.save(os.path.join(output_category_path, img_name))

resize_images('101_ObjectCategories', 'resized_101')

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
dataset = ImageFolder('101_ObjectCategories', transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '101_ObjectCategories',
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

# 加载预训练 ResNet
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 101)  # 修改分类头为 101 类
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
model.train()
for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')