# 在 PASCAL VOC 数据集上训练和测试 Mask R-CNN 和 Sparse R-CNN

本仓库包含使用 MMDetection 框架 在 PASCAL VOC 数据集上训练和测试 Mask R-CNN 和 Sparse R-CNN 模型的代码和资源。项目包括数据预处理、模型训练、测试和结果可视化的脚本，涵盖了两个模型的提案框（proposal box）和最终预测结果。

## 项目结构

```
├── checkpoints
├── configs
│   ├── _base_
│   ├── mask_rcnn
│   │   └── mask-rcnn_r50-caffe_fpn_ms-poly-1x_voc_to_coco.py
│   └── sparse_rcnn
│       └── sparse-rcnn_r50_fpn_1x_voc_to_coco.py
├── data
│   ├── README.md
│   ├── VOCdevkit
│   └── download.py
├── experiments
│   └── tutorial_exps
│       ├── mask_rcnn_voc
│       └── sparse_rcnn_voc
├── models
├── report
├── runs
└── src
    ├── Mask_R-CNN_VOC_demo.ipynb
    ├── Sparse_R-CNN_VOC_demo.ipynb
    ├── __pycache__
    ├── convert_voc_to_coco.py
    ├── split.py
    ├── test.py
    ├── train.py
    ├── visualize_sparse.py
    └── visualize_mask.py
```

- **checkpoints**: 存储模型权重（不在仓库中，下载链接见下文）。
- **configs**: Mask R-CNN 和 Sparse R-CNN 的配置文件。
- **data**: 包含 PASCAL VOC 数据集和下载脚本。
- **experiments**: 存储训练日志和模型检查点。
- **models**: 存储训练好的模型权重（不在仓库中）。
- **report**: 实验报告（PDF 格式）。
- **runs**: TensorBoard 日志目录。
- **src**: 数据预处理、训练、测试和可视化的源代码。

## 前置要求

### 依赖

安装 `requirements.txt` 中列出的依赖：

```bash
pip install -r requirements.txt
```

### 环境设置

- **Python**: 3.8 或更高版本
- **CUDA**: 11.0 或更高版本（若使用 GPU）
- **PyTorch**: 确保与 MMDetection 兼容（见 `requirements.txt`）。
- **MMDetection**: 按以下步骤安装。

### 安装 MMDetection

克隆 MMDetection 仓库并安装：

```bash
cd task2
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

### 数据集准备

1. 使用 `data` 目录中的 `download.py` 下载 PASCAL VOC 2012 数据集：

   ```bash
   cd data
   python download.py
   ```

   这会将数据集下载并解压到 `data/VOCdevkit/VOC2012`。

2. 将数据集划分为训练、验证和测试集：

   ```bash
   cd src
   python split.py
   ```

   这会在 `data/VOCdevkit/VOC2012/split` 中生成 `train.txt`、`val.txt` 和 `test.txt`。

3. 将 VOC 格式的标注转换为 COCO 格式：

   ```bash
   python convert_voc_to_coco.py
   ```

   这会在 `data/VOCdevkit/VOC2012/Annotations` 中生成 `instances_train2012.json`、`instances_val2012.json` 和 `instances_test2012.json`。

## 模型权重

训练好的 Mask R-CNN 和 Sparse R-CNN 模型权重可通过以下链接下载：

- **Mask R-CNN**: 下载 (epoch_24.pth)
- **Sparse R-CNN**: 下载 (epoch_48.pth)

下载后，将权重文件放置在以下路径：

- Mask R-CNN: `experiments/tutorial_exps/mask_rcnn_voc/20250529_164921/epoch_24.pth`
- Sparse R-CNN: `experiments/tutorial_exps/sparse_rcnn_voc/20250529_062936/epoch_48.pth`

## 训练

使用 `train.py` 脚本和相应的配置文件进行模型训练。

### Mask R-CNN

```bash
cd src
python train.py ../configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_voc_to_coco.py --work-dir ../experiments/tutorial_exps/mask_rcnn_voc
```

Mask R-CNN 具有预训练权重，下载方式：
```bash
mkdir -p checkpoints
wget -c https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth -O checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth
```

### Sparse R-CNN

```bash
python train.py ../configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc_to_coco.py --work-dir ../experiments/tutorial_exps/sparse_rcnn_voc
```

**训练设置**：

- **数据集**: PASCAL VOC 2012（训练集：70%，验证集：15%，测试集：15%）
- **网络结构**: Mask R-CNN（ResNet-50 主干），Sparse R-CNN（ResNet-50 主干）
- **批次大小**: 2（配置文件中默认）
- **学习率**: Mask R-CNN 为 0.02，Sparse R-CNN 为 0.0001（动态调整）
- **优化器**: SGD（动量=0.9，权重衰减=0.0001）
- **训练轮数**: Mask R-CNN 为 24 轮，Sparse R-CNN 为 48 轮
- **损失函数**: CrossEntropyLoss（分类），L1Loss 和 GIoULoss（边界框回归），CrossEntropyLoss（掩码预测，仅 Mask R-CNN）
- **评价指标**: mAP（COCO 格式的平均精度）

## 测试

使用 `test.py` 脚本测试模型性能。

### Mask R-CNN

```bash
python test.py ../configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_voc_to_coco.py ../experiments/tutorial_exps/mask_rcnn_voc/20250529_164921/epoch_24.pth --work-dir ../experiments/tutorial_exps/mask_rcnn_voc --out results_mask_rcnn.pkl
```

### Sparse R-CNN

```bash
python test.py ../configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc_to_coco.py ../experiments/tutorial_exps/sparse_rcnn_voc/20250529_062936/epoch_48.pth --work-dir ../experiments/tutorial_exps/sparse_rcnn_voc --out results_sparse_rcnn.pkl
```

## 可视化

### 可视化 Mask R-CNN 的提案框和最终预测

将测试集中的 4 张图像放入 `visualizations/in` 目录，然后运行：

```bash
python visualize_mask.py
```

这会生成：

- RPN 提案框：`visualizations/out/proposals`
- 最终预测（包括边界框、实例分割、类别标签和得分）：`visualizations/out/predictions`

### 可视化 Sparse R-CNN 的预测

将测试集中的 4 张图像放入 `visualizations/in` 目录，然后运行：

```bash
python visualize_sparse.py
```

这会生成最终预测（包括边界框、类别标签和得分）：`visualizations/out/predictions`

### 非 VOC 数据集图像测试

将 3 张包含 VOC 类别物体的非 VOC 数据集图像放入 `visualizations/in` 目录，分别运行 `visualize_mask.py` 和 `visualize.py`，结果将保存到 `visualizations/out/predictions`。

## TensorBoard 可视化

训练过程中会生成 TensorBoard 日志，位于 `runs` 目录。查看训练和验证集的损失曲线及验证集的 mAP 曲线：

```bash
tensorboard --logdir runs
```

访问 `http://localhost:6006` 查看可视化结果。

## 实验报告

实验报告（PDF 格式）位于 `report` 目录，包含：

- 模型和数据集介绍
- 实验设置（训练测试划分、网络结构、超参数等）
- TensorBoard 损失和 mAP 曲线
- 可视化对比（Mask R-CNN 的提案框与最终预测，Mask R-CNN 和 Sparse R-CNN 的结果）
- 非 VOC 数据集图像的测试结果

## 运行示例

1. 准备数据集并转换格式。
2. 下载模型权重并放置到指定路径。
3. 运行训练脚本（`train.py`）进行模型训练。
4. 运行测试脚本（`test.py`）评估模型性能。
5. 运行可视化脚本（`visualize.py` 和 `visualize_mask.py`）生成结果。

## 注意事项

- 确保 GPU 可用并正确配置 CUDA 环境。
- 检查配置文件中的路径是否与本地环境一致。
- 非 VOC 数据集图像需包含 VOC 类别（如 person、car 等）以获得有意义的预测结果。

## 联系方式

如有问题，请在 GitHub Issues 中反馈或联系 \[your-email@example.com\]。