# CV_Assignment02 - Task 2: Object Detection and Segmentation on PASCAL VOC

## 概述
本项目为计算机视觉课程（CV_Assignment02）的 Task 2，基于 PASCAL VOC 2012 数据集，使用 MMDetection 框架实现 Mask R-CNN 和 Sparse R-CNN 的目标检测和实例分割。任务包括：
- 微调 Mask R-CNN 和 Sparse R-CNN 模型，适配 PASCAL VOC 的 20 个类别。
- 数据预处理，将 VOC 格式转换为 COCO 格式。
- 对比 Mask R-CNN 和 Sparse R-CNN 的性能（mAP）。
- 可视化训练过程中的损失、mAP 以及 Mask R-CNN 的 RPN 提案框和最终预测结果。

## 任务描述
- **数据集**：使用 [PASCAL VOC 2012 数据集](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)，按 70%/15%/15% 划分为训练集、验证集和测试集。
- **模型**：基于 ResNet-50 主干的 Mask R-CNN（带实例分割）和 Sparse R-CNN（仅目标检测）。
- **超参数**：固定超参数设置（学习率、批量大小、权重衰减、训练轮数），配置文件位于 `configs/`。
- **可视化**：使用 TensorBoard 记录训练/验证的损失和 mAP，生成 Mask R-CNN 的 RPN 提案框、最终预测（边界框、实例分割、类别标签和得分）以及 Sparse R-CNN 的预测结果。

## 依赖项
- Python 3.8 或更高版本
- CUDA 11.0 或更高版本（若使用 GPU）
- 依赖库（见 `requirements.txt`）：
  ```bash
  pip install -r requirements.txt
  ```
- **注意**：若使用 GPU，请确保安装支持 CUDA 的 PyTorch 版本（例如 `torch==1.12.1+cu113`）。可参考 [PyTorch 官网](https://pytorch.org/get-started/previous-versions/)。

## 项目结构
```plaintext
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
    ├── visualize_mask.py
    └── visualize_sparse.py
```

- **checkpoints**: 存储预训练模型权重（不在仓库中，下载链接见下文）。
- **configs**: Mask R-CNN 和 Sparse R-CNN 的配置文件。
- **data**: 包含 PASCAL VOC 数据集和下载脚本。
- **experiments**: 存储训练日志和模型检查点。
- **models**: 存储训练好的模型权重（不在仓库中）。
- **report**: 实验报告（PDF 格式）。
- **runs**: TensorBoard 日志目录。
- **src**: 数据预处理、训练、测试和可视化的源代码。

## 运行说明

1. **克隆仓库**：
   ```bash
   git clone https://github.com/spoil-ed/cv_assignment02.git
   cd cv_assignment02/task2
   ```

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

3. **安装 MMDetection**：
   ```bash
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   pip install -v -e .
   cd ..
   ```

4. **准备数据集**：
   - 下载 [PASCAL VOC 2012 数据集](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 并解压到 `data/VOCdevkit/VOC2012`：
     ```bash
     cd data
     python download.py
     ```
   - 划分数据集（70% 训练，15% 验证，15% 测试）：
     ```bash
     cd src
     python split.py
     ```
     这会在 `data/VOCdevkit/VOC2012/split` 中生成 `train.txt`、`val.txt` 和 `test.txt`。
   - 将 VOC 格式转换为 COCO 格式：
     ```bash
     python convert_voc_to_coco.py
     ```
     这会在 `data/VOCdevkit/VOC2012/Annotations` 中生成 `instances_train2012.json`、`instances_val2012.json` 和 `instances_test2012.json`。

5. **下载预训练权重**：
   - Mask R-CNN 预训练权重：
     ```bash
     mkdir -p checkpoints
     wget -c https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth -O checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.pth
     ```

6. **训练模型**：
   - **Mask R-CNN**：
     ```bash
     cd src
     python train.py ../configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_voc_to_coco.py --work-dir ../experiments/tutorial_exps/mask_rcnn_voc
     ```
   - **Sparse R-CNN**：
     ```bash
     python train.py ../configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc_to_coco.py --work-dir ../experiments/tutorial_exps/sparse_rcnn_voc
     ```
   - **训练设置**：
     - 数据集：PASCAL VOC 2012（训练集：70%，验证集：15%，测试集：15%）
     - 网络结构：Mask R-CNN 和 Sparse R-CNN（均基于 ResNet-50 主干）
     - 批次大小：2（配置文件默认）
     - 学习率：Mask R-CNN 为 0.02，Sparse R-CNN 为 0.0001（动态调整）
     - 优化器：SGD（动量=0.9，权重衰减=0.0001）
     - 训练轮数：Mask R-CNN 为 24 轮，Sparse R-CNN 为 48 轮
     - 损失函数：CrossEntropyLoss（分类），L1Loss 和 GIoULoss（边界框回归），CrossEntropyLoss（掩码预测，仅 Mask R-CNN）
     - 评价指标：mAP（COCO 格式的平均精度）
   - 输出：模型权重保存至 `experiments/tutorial_exps/{model}/`，训练日志保存至 `runs/`。

7. **测试模型**：
   - **Mask R-CNN**：
     ```bash
     python test.py ../configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_voc_to_coco.py ../experiments/tutorial_exps/mask_rcnn_voc/20250529_164921/epoch_24.pth --work-dir ../experiments/tutorial_exps/mask_rcnn_voc --out results_mask_rcnn.pkl
     ```
   - **Sparse R-CNN**：
     ```bash
     python test.py ../configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc_to_coco.py ../experiments/tutorial_exps/sparse_rcnn_voc/20250529_062936/epoch_48.pth --work-dir ../experiments/tutorial_exps/sparse_rcnn_voc --out results_sparse_rcnn.pkl
     ```
   - 输出：测试结果（mAP）保存至 `--out` 指定的 `.pkl` 文件。

8. **可视化结果**：
   - **Mask R-CNN 的 RPN 提案框和最终预测**：
     将测试集中的 4 张图像放入 `visualizations/in/`，然后运行：
     ```bash
     python visualize_mask.py
     ```
     输出：
     - RPN 提案框：`visualizations/out/proposals/`
     - 最终预测（边界框、实例分割、类别标签和得分）：`visualizations/out/predictions/`
   - **Sparse R-CNN 的预测**：
     将测试集中的 4 张图像放入 `visualizations/in/`，然后运行：
     ```bash
     python visualize_sparse.py
     ```
     输出：最终预测（边界框、类别标签和得分）：`visualizations/out/predictions/`
   - **非 VOC 数据集图像测试**：
     将 3 张包含 VOC 类别（如 person、car 等）的非 VOC 数据集图像放入 `visualizations/in/`，分别运行 `visualize_mask.py` 和 `visualize_sparse.py`，结果保存至 `visualizations/out/predictions/`。

9. **TensorBoard 可视化**：
   ```bash
   tensorboard --logdir runs
   ```
   在浏览器访问 `http://localhost:6006` 查看训练/验证的损失和 mAP 曲线。

## 模型权重
- **Mask R-CNN**：`experiments/tutorial_exps/mask_rcnn_voc/20250529_164921/epoch_24.pth`
- **Sparse R-CNN**：`experiments/tutorial_exps/sparse_rcnn_voc/20250529_062936/epoch_48.pth`
- **预训练权重**：Mask R-CNN 的预训练权重（COCO 数据集）：
  - [下载链接](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth)
  - 保存至 `checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.pth`
- **训练权重**：训练后的权重可通过以下链接下载（需替换为实际链接）：
  - [百度云链接](https://pan.baidu.com/s/your-link?pwd=your-password)
  - 下载后放置在 `experiments/tutorial_exps/{model}/` 目录下。

## 实验报告
- 报告（PDF 格式）位于 `report/` 目录，包含：
  - 模型和数据集介绍
  - 实验设置（训练/测试划分、网络结构、超参数等）
  - TensorBoard 损失和 mAP 曲线
  - 可视化对比（Mask R-CNN 的 RPN 提案框与最终预测，Mask R-CNN 和 Sparse R-CNN 的结果）
  - 非 VOC 数据集图像的测试结果
- GitHub 仓库：[https://github.com/spoil-ed/cv_assignment02](https://github.com/spoil-ed/cv_assignment02)

## 注意事项
- 确保 `data/VOCdevkit/VOC2012/` 包含正确的数据集文件。
- GPU 推荐使用 CUDA 11.0 或兼容版本。
- 检查配置文件中的路径，确保与本地环境一致。
- 非 VOC 数据集图像需包含 VOC 类别（如 person、car 等）以获得有意义的预测结果。
- 所有代码仅用于学术用途。

## 联系方式
如有问题，请在 GitHub Issues 中反馈或联系 [your-email@example.com]。

---

**版权所有 © 2025 spoil-ed**