import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmengine.visualization import Visualizer

config = '../mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_voc_to_coco.py'
img = mmcv.imread('/home/spoil/cv/assignment02/task2/data/VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg',channel_order='rgb')
checkpoint_file = '../experiments/tutorial_exps/mask_rcnn_voc/epoch_3.pth'
model = init_detector(config, checkpoint_file, device='cpu')
new_result = inference_detector(model, img)
print(new_result)

# init visualizer(run the block only once in jupyter notebook)
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

# get built visualizer
visualizer_now = Visualizer.get_current_instance()
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer_now.dataset_meta = model.dataset_meta
# show the results
visualizer_now.add_datasample(
    'new_result',
    img,
    data_sample=new_result,
    draw_gt=False,
    wait_time=0,
    show=False,
    pred_score_thr=0.5
)
visualizer_now.show()