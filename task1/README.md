# CV_Assignment02 - Task 1: Caltech101 Image Classification

## 概述
本项目为计算机视觉课程（CV_Assignment02）的 Task 1，基于 Caltech101 数据集，使用 ResNet18 模型实现图像分类。任务包括：
- 微调 ImageNet 预训练的 ResNet18 模型，适配 Caltech101 的 101 个类别。
- 从头训练（scratch）ResNet18 模型，进行性能对比。
- 分析超参数（如学习率、批量大小、权重衰减、训练轮数）对性能的影响。
- 可视化训练过程中的损失、准确率、模型权重及超参数效果。

## 任务描述
- **数据集**：使用 [Caltech101 数据集](https://data.caltech.edu/records/mzrjq-6wc02)，按标准划分训练集、验证集和测试集（60%/20%/20%）。
- **模型**：修改 ResNet18 的全连接层（输出 101 类），微调时使用 ImageNet 预训练权重，从头训练时使用随机初始化。
- **超参数**：测试学习率（基础和全连接层）、批量大小、权重衰减和训练轮数，绘制箱式图和热图分析影响。
- **可视化**：使用 TensorBoard 记录训练/验证的损失和准确率，生成卷积核、全连接层权重、权重分布和超参数效果的可视化。

## 依赖项
- Python 3.9.21
- 依赖库（见 `requirements.txt`）：
  ```bash
  pip install -r requirements.txt
  ```
- **注意**：若使用 GPU，请确保安装支持 CUDA 的 PyTorch 版本（例如 `torch==1.12.1+cu113`）。可参考 [PyTorch 官网](https://pytorch.org/get-started/previous-versions/)。

## 项目结构
```plaintext
├── boxplot.log
├── data
│   ├── README.md
│   └── caltech101
│       ├── 101_ObjectCategories
│       ├── Annotations
│       └── show_annotation.m
├── experiments
│   ├── finetune_20250529_034950
│   └── ...
├── images
│   ├── boxplot_lr_base.png
│   └── ...
├── models
│   ├── best_model.pth
│   ├── caltech101_resnet18_finetune.pth
│   └── caltech101_resnet18_scratch.pth
├── report
├── runs
├── src
│   ├── boxplot.py
│   ├── main.py
│   ├── model.py
│   ├── predict.py
│   ├── test.py
│   ├── train.py
│   ├── utils.py
│   └── visualize.py
└── training.log
```

## 运行说明

1. **克隆仓库**：
   ```bash
   git clone https://github.com/spoil-ed/cv_assignment02.git
   cd cv_assignment02/task1
   ```

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

3. **准备数据集**：
   - 下载 [Caltech101 数据集](https://data.caltech.edu/records/mzrjq-6wc02) 并解压到 `data/caltech101/`。
   - 确保目录结构为 `data/caltech101/101_ObjectCategories`。

4. **训练模型**：
   - **微调和从头训练**：
     ```bash
     python src/main.py --data-dir data --batch-size 32 --epochs 30 --lr-base 0.001 --lr-fc 0.01 --weight-decay 0.001
     ```
     - 添加 `--from-scratch` 进行从头训练。
     - 添加 `--grid-search` 进行超参数网格搜索。
   - 输出：模型权重保存至 `models/`，训练指标保存至 `experiments/`，可视化图像保存至 `experiments/images/`。

5. **测试模型**：
   ```bash
   python src/test.py
   ```
   - 修改 `test.py` 中的 `weight_path` 和 `data_dir` 以加载特定模型权重和数据集。
   - 输出：测试准确率和类别准确率图。

6. **生成箱式图**：
   ```bash
   python src/boxplot.py --data-dir data --batch-size 32 --epochs 20 --lr-base 0.001 --lr-fc 0.01 --weight-decay 0.001
   ```
   - 输出：超参数的箱式图保存至 `images/`。

7. **可视化训练过程**：
   ```bash
   tensorboard --logdir runs
   ```
   - 在浏览器访问 `http://localhost:6006` 查看损失和准确率曲线。

## 模型权重
- 微调模型：`caltech101_resnet18_finetune.pth`
- 从头训练模型：`caltech101_resnet18_scratch.pth`
- 最佳模型：`best_model.pth`
- 下载链接：[百度云链接](https://pan.baidu.com/s/16ltCGOIotZC5y1wReYtsqg?pwd=best)
- 说明：权重文件下载后放置在 `models/` 目录下，调用`test.py`实现测试。

## 实验报告
- 报告（PDF 格式）位于 `report/` 目录，包含模型描述、数据集说明、实验结果、TensorBoard 损失/准确率曲线及超参数分析。
- GitHub 仓库：[https://github.com/spoil-ed/cv_assignment02](https://github.com/spoil-ed/cv_assignment02)

## 注意事项
- 确保 `data/caltech101/` 包含正确的数据集文件。
- GPU 推荐使用 CUDA 11.3 或兼容版本。
- 箱式图和可视化图像需要 Matplotlib 和 Seaborn 支持，确保环境正确配置。
- 所有代码仅用于学术用途。

---

**版权所有 © 2025 spoil-ed**