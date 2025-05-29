import os
import mmcv
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmcv.transforms import Compose
import torch
import shutil

# 配置
config_file = 'configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc_to_coco.py'
checkpoint_file = 'checkpoints/Sparse_R-CNN_best_weight.pth'
data_root = 'data/VOCdevkit/VOC2012'
test_ann_file = 'data/VOCdevkit/VOC2012/split/test.txt'
base_out_dir = 'visualization/sparse_rcnn_voc/'
input_dir = os.path.join(base_out_dir, 'in')
predictions_dir = os.path.join(base_out_dir, 'out', 'predictions') 
backend_args = None

# 创建目录
os.makedirs(input_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

# 初始化模型
try:
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
except FileNotFoundError as e:
    print(f"错误：{e}。请检查 config_file 和 checkpoint_file 路径是否正确。")
    exit(1)

# 加载 input_dir 中的所有图像
img_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
img_paths = [os.path.join(input_dir, img_id) for img_id in img_files]
selected_imgs = [os.path.splitext(img_id)[0] for img_id in img_files]  # 移除 .jpg 后缀作为 img_id

if not img_paths:
    print(f"错误：{input_dir} 中没有找到 .jpg 文件，请确保目录中已有图像。")
    exit(1)

# 初始化可视化器
visualizer_cfg = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend', save_dir=base_out_dir),
        dict(type='TensorboardVisBackend', save_dir=os.path.join(base_out_dir, 'tensorboard'))
    ],
    name='visualizer'
)
visualizer = VISUALIZERS.build(visualizer_cfg)
visualizer.dataset_meta = {
    'classes': ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
    'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192), (197, 226, 255),
                (190, 153, 153), (180, 165, 180), (90, 86, 231), (210, 120, 180), (102, 121, 66),
                (0, 255, 0), (0, 0, 142), (0, 60, 100), (0, 0, 230), (0, 80, 100),
                (46, 191, 191), (81, 0, 21), (220, 20, 60), (255, 245, 0), (139, 0, 0)]
}

# 定义测试管道（保持与Sparse R-CNN兼容）
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
test_pipeline = Compose(test_pipeline)

# 预处理图像
def preprocess_image(img_path):
    try:
        data = dict(img_path=img_path, img_id=os.path.basename(img_path).split('.')[0])
        data = test_pipeline(data)
        img_tensor = data['inputs'].to('cuda:0')
        data_sample = data['data_samples']
        img = mmcv.imread(img_path)
        return img_tensor, data_sample, img
    except Exception as e:
        print(f"预处理 {img_path} 时出错：{e}")
        return None, None, None

# 可视化最终预测
def visualize_predictions(result, img, out_file, score_thr=0.3):
    if result is None or 'pred_instances' not in result:
        print(f"无效的预测结果，跳过 {out_file}")
        return
    
    # 确保分数和类别存在
    instances = result.pred_instances
    if 'scores' not in instances or 'labels' not in instances:
        print(f"预测结果缺少必要字段，跳过 {out_file}")
        return
    
    print(f"预测结果 {out_file}：bboxes={len(instances.bboxes)}")
    try:
        visualizer.add_datasample(
            name='predictions',
            image=img,
            data_sample=result,
            draw_gt=False,
            show=False,
            out_file=out_file,
            pred_score_thr=score_thr
        )
        print(f"成功生成预测可视化：{out_file}")
    except Exception as e:
        print(f"可视化预测 {out_file} 时出错：{e}")

# 处理每张图片
for idx, (img_path, img_id) in enumerate(zip(img_paths, selected_imgs)):
    if not os.path.exists(img_path):
        print(f"错误：图片 {img_path} 未找到，跳过。")
        continue
    
    # 预处理
    img_tensor, data_sample, img = preprocess_image(img_path)
    if img_tensor is None:
        continue
    
    # 直接获取最终预测
    try:
        result = inference_detector(model, img_path)
        pred_out_file = os.path.join(predictions_dir, f'{img_id}_predictions.jpg')
        visualize_predictions(result, img, pred_out_file, score_thr=0.3)
    except Exception as e:
        print(f"处理 {img_path} 的预测时出错：{e}")

print(f"输入图片已保存到 {input_dir}")
print(f"预测可视化已保存到 {predictions_dir}")