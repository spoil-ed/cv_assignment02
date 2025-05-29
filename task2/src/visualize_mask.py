import os
import mmcv
import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmcv.transforms import Compose
import logging
from pathlib import Path
from typing import Optional, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置
CONFIG_FILE = 'configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_voc_to_coco.py'
CHECKPOINT_FILE = 'experiments/tutorial_exps/mask_rcnn_voc/20250529_164921/epoch_24.pth'
IMG_DIR = 'visualizations/in'
BASE_OUT_DIR = 'visualizations'
PROPOSALS_DIR = os.path.join(BASE_OUT_DIR, 'out', 'proposals')
PREDICTIONS_DIR = os.path.join(BASE_OUT_DIR, 'out', 'predictions')
BACKEND_ARGS = None
SCORE_THR_PROPOSALS = 0.001  # RPN 提案分数阈值
SCORE_THR_PREDICTIONS = 0.3   # 最终预测分数阈值

# 创建输出目录
Path(PROPOSALS_DIR).mkdir(parents=True, exist_ok=True)
Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)

# 加载所有图片
img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
img_paths = [os.path.join(IMG_DIR, f) for f in img_files]
logger.info(f"找到 {len(img_paths)} 张图片进行处理")

# 初始化模型
try:
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')
except FileNotFoundError as e:
    logger.error(f"模型加载失败：{e}。请检查 config_file 和 checkpoint_file 路径是否正确。")
    exit(1)
except Exception as e:
    logger.error(f"初始化模型时发生未知错误：{e}")
    exit(1)

# 初始化可视化器
visualizer_cfg = {
    'type': 'DetLocalVisualizer',
    'vis_backends': [
        {'type': 'LocalVisBackend', 'save_dir': BASE_OUT_DIR},
        {'type': 'TensorboardVisBackend', 'save_dir': os.path.join(BASE_OUT_DIR, 'tensorboard')}
    ],
    'name': 'visualizer'
}
try:
    visualizer = VISUALIZERS.build(visualizer_cfg)
    visualizer.dataset_meta = {
        'classes': model.dataset_meta['classes'],
        'palette': model.dataset_meta['palette']
    }
except Exception as e:
    logger.error(f"初始化可视化器失败：{e}")
    exit(1)

# 定义测试管道
test_pipeline = Compose([
    {'type': 'LoadImageFromFile', 'backend_args': BACKEND_ARGS},
    {'type': 'Resize', 'scale': (1333, 800), 'keep_ratio': True},
    {'type': 'PackDetInputs', 'meta_keys': ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')}
])

def preprocess_image(img_path: str) -> Optional[Tuple[torch.Tensor, DetDataSample, np.ndarray]]:
    """预处理图像"""
    try:
        data = {'img_path': img_path, 'img_id': Path(img_path).stem}
        data = test_pipeline(data)
        img_tensor = data['inputs'].to('cuda:0').float() / 255.0
        data_sample = data['data_samples']
        img = mmcv.imread(img_path, channel_order='rgb')
        if img is None:
            raise ValueError(f"无法加载图片：{img_path}")
        return img_tensor, data_sample, img
    except Exception as e:
        logger.error(f"预处理 {img_path} 时出错：{e}")
        return None, None, None

def get_rpn_proposals(model, img_tensor: torch.Tensor, data_sample: DetDataSample) -> Optional[InstanceData]:
    """获取 RPN 提案"""
    try:
        model.eval()
        with torch.no_grad():
            feats = model.extract_feat(img_tensor.unsqueeze(0))
            rpn_results_list = model.rpn_head.predict(feats, [data_sample], rescale=False)
            instances = InstanceData(
                bboxes=rpn_results_list[0].bboxes,
                scores=rpn_results_list[0].scores,
                labels=torch.zeros_like(rpn_results_list[0].scores, dtype=torch.long)
            )
            logger.info(f"RPN 提案数量（{data_sample.metainfo['img_id']}）：{len(instances.bboxes)}")
            return instances
    except Exception as e:
        logger.error(f"提取 RPN 提案时出错（{data_sample.metainfo['img_id']}）：{e}")
        return None

def visualize_proposals(img: np.ndarray, instances: InstanceData, out_file: str, score_thr: float = SCORE_THR_PROPOSALS) -> None:
    """可视化 RPN 提案"""
    if instances is None or len(instances.bboxes) == 0:
        logger.warning(f"由于提案无效或为空，跳过 {out_file} 的可视化")
        return
    try:
        data_sample = DetDataSample(pred_instances=instances)
        visualizer.add_datasample(
            name='proposals',
            image=img,
            data_sample=data_sample,
            draw_gt=False,
            show=False,
            out_file=out_file,
            pred_score_thr=score_thr
        )
        logger.info(f"成功生成提案可视化：{out_file}")
    except Exception as e:
        logger.error(f"可视化提案 {out_file} 时出错：{e}")

def visualize_predictions(result: DetDataSample, img: np.ndarray, out_file: str, score_thr: float = SCORE_THR_PREDICTIONS) -> None:
    """可视化最终预测"""
    if result is None or len(result.pred_instances.bboxes) == 0:
        logger.warning(f"由于预测结果无效或为空，跳过 {out_file} 的可视化")
        return
    logger.info(f"预测结果 {out_file}：bboxes={len(result.pred_instances.bboxes)}, masks={len(result.pred_instances.masks)}")
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
        logger.info(f"成功生成预测可视化：{out_file}")
    except Exception as e:
        logger.error(f"可视化预测 {out_file} 时出错：{e}")

def main():
    """主处理函数"""
    for img_path in img_paths:
        if not os.path.exists(img_path):
            logger.warning(f"图片 {img_path} 未找到，跳过。")
            continue

        # 预处理图像
        img_tensor, data_sample, img = preprocess_image(img_path)
        if img_tensor is None:
            continue

        img_id = Path(img_path).stem
        logger.info(f"处理图片：{img_id}")

        # 获取 RPN 提案
        instances = get_rpn_proposals(model, img_tensor, data_sample)
        proposal_out_file = os.path.join(PROPOSALS_DIR, f'{img_id}_proposals.jpg')
        visualize_proposals(img, instances, proposal_out_file)

        # 获取最终预测
        try:
            result = inference_detector(model, img_path)
            pred_out_file = os.path.join(PREDICTIONS_DIR, f'{img_id}_predictions.jpg')
            visualize_predictions(result, img, pred_out_file)
        except Exception as e:
            logger.error(f"处理 {img_path} 的预测时出错：{e}")

if __name__ == '__main__':
    main()
    logger.info(f"提案可视化已保存到 {PROPOSALS_DIR}")
    logger.info(f"预测可视化已保存到 {PREDICTIONS_DIR}")