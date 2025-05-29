import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import cv2

# 配置路径
voc_root = "data/VOCdevkit/VOC2012"
annotation_dir = os.path.join(voc_root, "Annotations")
image_dir = os.path.join(voc_root, "JPEGImages")
seg_dir = os.path.join(voc_root, "SegmentationObject")
imageset_dir = os.path.join(voc_root, "split")
output_train_json = os.path.join(voc_root, "Annotations/instances_train2012.json")
output_val_json = os.path.join(voc_root, "Annotations/instances_val2012.json")
output_test_json = os.path.join(voc_root, "Annotations/instances_test2012.json")

# VOC 类别列表
categories = [
    {"id": i + 1, "name": name, "supercategory": "none"}
    for i, name in enumerate([
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ])
]

# 类别映射：掩码像素值到 VOC 类别名的映射
category_mapping = {i + 1: cat["name"] for i, cat in enumerate(categories)}

# 初始化 COCO 格式
def init_coco_format():
    return {
        "info": {
            "description": "VOC to COCO conversion",
            "version": "1.0",
            "year": 2025,
            "date_created": "2025-05-28"
        },
        "images": [],
        "annotations": [],
        "categories": categories
    }

# 辅助函数：从掩码生成多边形坐标
def parse_voc_annotation(xml_file):
    """
    解析 PASCAL VOC 格式的 XML 文件，提取边界框和类别信息。
    
    参数:
        xml_file: VOC 格式的 XML 文件路径。
    
    返回:
        annotations: 包含边界框和类别信息的列表，每个元素为 {'name': 类别名, 'bbox': [xmin, ymin, xmax, ymax]}
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        bbox = [
            float(bndbox.find('xmin').text),
            float(bndbox.find('ymin').text),
            float(bndbox.find('xmax').text),
            float(bndbox.find('ymax').text)
        ]
        annotations.append({
            'name': name,
            'bbox': bbox  # [xmin, ymin, xmax, ymax]
        })
    
    return annotations

def polygon_to_bbox(polygon):
    """
    将多边形转换为边界框 [xmin, ymin, width, height]。
    
    参数:
        polygon: 多边形点列表 [x1, y1, x2, y2, ...]
    
    返回:
        bbox: [xmin, ymin, width, height]
    """
    points = np.array(polygon).reshape(-1, 2)
    xmin = np.min(points[:, 0])
    ymin = np.min(points[:, 1])
    xmax = np.max(points[:, 0])
    ymax = np.max(points[:, 1])
    return [xmin, ymin, xmax - xmin, ymax - ymin]

def compute_iou(box1, box2):
    """
    计算两个边界框的 IoU。
    
    参数:
        box1: [xmin, ymin, w, h]
        box2: [xmin, ymin, xmax, ymax]（VOC 格式）
    
    返回:
        iou: 交并比值
    """
    x1, y1, w1, h1 = box1
    x2, y2, x2_max, y2_max = box2
    
    # 计算交集
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2_max)
    y_bottom = min(y1 + h1, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算并集
    area1 = w1 * h1
    area2 = (x2_max - x2) * (y2_max - y2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def mask_to_polygon_with_voc(mask, voc_xml_file, category_mapping):
    """
    将掩码转换为多边形，并与 VOC annotation 的检测框匹配，选择 IoU 最大的轮廓（不考虑类别）。
    
    参数:
        mask: 输入掩码图像（单通道，类别由像素值表示）。
        voc_xml_file: PASCAL VOC 格式的 XML 文件路径。
        category_mapping: 掩码像素值到 VOC 类别名的映射（例如 {1: 'cat', 2: 'dog'}）。
    
    返回:
        results: 列表，每个元素为 {'category_id': int, 'category_name': str, 'polygons': [[x1, y1, x2, y2, ...]], 'bbox': [xmin, ymin, w, h], 'iou': float, 'voc_annotation': dict}
    """
    # 解析 VOC annotation
    voc_annotations = parse_voc_annotation(voc_xml_file)
    
    # 确保掩码是 numpy 数组
    mask = np.array(mask)
    
    # 获取所有唯一的像素值（类别）
    unique_values = np.unique(mask)
    unique_values = unique_values[unique_values != 0]  # 排除背景
    
    results = []
    for value in unique_values:
        # 创建当前类别的二值化掩码
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        binary_mask[mask == value] = 255
        
        # 确保掩码是二值化的
        _, binary_mask = cv2.threshold(binary_mask, 0, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) < 3:  # 确保轮廓点数足够
                continue
            # 将轮廓展平为 [x1, y1, x2, y2, ...]
            polygon = contour.reshape(-1).tolist()
            
            # 将多边形转换为边界框
            polygon_bbox = polygon_to_bbox(polygon)
            
            # 与 VOC annotation 的检测框计算 IoU
            max_iou = 0.2
            best_annotation = None
            for ann in voc_annotations:
                voc_bbox = ann['bbox']  # [xmin, ymin, xmax, ymax]
                iou = compute_iou(polygon_bbox, voc_bbox)
                if iou > max_iou:
                    max_iou = iou
                    best_annotation = ann
            
            # 如果找到匹配的 annotation，记录结果
            if best_annotation is not None and max_iou > 0:
                results.append({
                    'category_id': value,  # 保留掩码的原始像素值（类别）
                    'category_name': category_mapping.get(value, 'unknown'),
                    'polygons': [polygon],
                    'bbox': polygon_bbox,
                    'iou': max_iou,
                    'voc_annotation': best_annotation
                })
    
    # 按 IoU 排序并筛选出每个 VOC annotation 的最佳匹配
    final_results = []
    used_annotations = set()
    results = sorted(results, key=lambda x: x['iou'], reverse=True)  # 按 IoU 降序排序
    
    for result in results:
        ann_id = id(result['voc_annotation'])
        if ann_id not in used_annotations:
            # 使用 VOC annotation 的类别，而不是掩码的 category_id
            voc_category_name = result['voc_annotation']['name']
            voc_category_id = next((cat['id'] for cat in categories if cat['name'] == voc_category_name), None)
            if voc_category_id is not None:
                result['category_id'] = voc_category_id
                result['category_name'] = voc_category_name
                final_results.append(result)
                used_annotations.add(ann_id)
    
    return final_results

# 读取 ImageSets/Segmentation 中的 train.txt, trainval.txt, val.txt 文件
def get_image_lists(imageset_dir):
    train_images = set()
    val_images = set()
    test_images = set()
    
    # 读取 train.txt
    train_file = os.path.join(imageset_dir, "train.txt")
    if os.path.exists(train_file):
        with open(train_file, "r") as f:
            train_images.update(line.strip() for line in f if line.strip())
    
    # 读取 val.txt
    val_file = os.path.join(imageset_dir, "val.txt")
    if os.path.exists(val_file):
        with open(val_file, "r") as f:
            val_images.update(line.strip() for line in f if line.strip())
    
    # 读取 test.txt
    test_file = os.path.join(imageset_dir, "test.txt")
    if os.path.exists(test_file):
        with open(test_file, "r") as f:
            test_images.update(line.strip() for line in f if line.strip())
    
    return list(train_images), list(val_images), list(test_images)

# 自定义 JSON 编码器以避免换行
class NoNewlineEncoder(json.JSONEncoder):
    def encode(self, obj):
        return json.dumps(obj, separators=(',', ':'))

# 转换函数
def convert_voc_to_coco(annotation_dir, image_dir, seg_dir, image_list, output_json):
    coco_format = init_coco_format()
    image_id = 1
    annotation_id = 1

    for img_name in image_list:
        xml_file = img_name + ".xml"
        xml_path = os.path.join(annotation_dir, xml_file)
        if not os.path.exists(xml_path):
            print(f"XML file {xml_file} not found, skipping...")
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 提取图像信息
        filename = root.find("filename").text
        img_path = os.path.join(image_dir, filename)
        try:
            img = Image.open(img_path)
            width, height = img.size
        except Exception as e:
            print(f"Failed to open image {filename}: {e}, skipping...")
            continue

        coco_image = {
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        }
        coco_format["images"].append(coco_image)

        # 加载分割掩码
        seg_filename = os.path.splitext(filename)[0] + ".png"
        seg_path = os.path.join(seg_dir, seg_filename)
        if not os.path.exists(seg_dir) or not os.path.exists(seg_path):
            print(f"分割掩码 {seg_filename} 未找到（图像 {filename}），跳过...")
            continue

        mask = np.array(Image.open(seg_path).convert("L"))  # 转换为灰度图

        # 使用 mask_to_polygon_with_voc 获取多边形
        results = mask_to_polygon_with_voc(mask, xml_path, category_mapping)
        if not results:
            continue

        # 为每个结果创建 COCO 标注
        used_polygons = set()
        for result in results:
            if id(result) in used_polygons:
                continue

            category_id = result['category_id']  # 已经更新为 VOC 标注的类别
            category_name = result['category_name']
            
            # 边界框使用 VOC 标注的边界框
            voc_bbox = result['voc_annotation']['bbox']
            xmin, ymin, xmax, ymax = voc_bbox
            width = xmax - xmin
            height = ymax - ymin

            # 创建 COCO 标注
            coco_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "segmentation": result['polygons'],  # 直接使用匹配的多边形
                "iscrowd": 0
            }

            coco_format["annotations"].append(coco_annotation)
            used_polygons.add(id(result))
            annotation_id += 1

        image_id += 1
    print(image_id-1)
    # 保存 COCO JSON，使用自定义编码器以避免换行
    with open(output_json, "w") as f:
        json.dump(coco_format, f, cls=NoNewlineEncoder)
    print(f"COCO JSON saved to {output_json}")

# 执行转换
if __name__ == "__main__":
    train_images, val_images, test_images = get_image_lists(imageset_dir)
    if not train_images:
        raise ValueError("No train images found in ImageSets/Segmentation")
    if not val_images:
        raise ValueError("No val images found in ImageSets/Segmentation")

    convert_voc_to_coco(annotation_dir, image_dir, seg_dir, train_images, output_train_json)
    convert_voc_to_coco(annotation_dir, image_dir, seg_dir, val_images, output_val_json)
    convert_voc_to_coco(annotation_dir, image_dir, seg_dir, test_images, output_test_json)