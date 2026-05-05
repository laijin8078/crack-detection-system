#!/usr/bin/env python3
"""
建筑裂缝检测 - 推理脚本
支持单张图像和批量图像推理
输出详细的检测结果：类别、位置坐标、置信度、分割掩码
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime


def predict_image(model, image_path, conf_threshold=0.25, iou_threshold=0.7, save_results=True):
    """
    对单张图像进行推理

    Args:
        model: YOLO模型
        image_path: 图像路径
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU阈值
        save_results: 是否保存结果

    Returns:
        detections: 检测结果列表
    """
    # 推理
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        save=save_results,
        project='outputs/predictions',
        name='inference',
        exist_ok=True
    )

    detections = []

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            print(f"未检测到裂缝: {image_path}")
            continue

        boxes = r.boxes
        masks = r.masks

        for i, box in enumerate(boxes):
            # 提取边界框信息
            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            xywh = box.xywh[0].cpu().numpy()  # [cx, cy, w, h]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = r.names[cls]

            # 提取分割掩码
            mask_polygon = None
            if masks is not None and i < len(masks):
                mask_xy = masks[i].xy[0]  # 多边形坐标
                mask_polygon = mask_xy.tolist()

            detection = {
                'class': class_name,
                'class_id': cls,
                'confidence': round(conf, 4),
                'bbox': {
                    'x1': round(float(xyxy[0]), 2),
                    'y1': round(float(xyxy[1]), 2),
                    'x2': round(float(xyxy[2]), 2),
                    'y2': round(float(xyxy[3]), 2)
                },
                'center': {
                    'x': round(float(xywh[0]), 2),
                    'y': round(float(xywh[1]), 2)
                },
                'size': {
                    'width': round(float(xywh[2]), 2),
                    'height': round(float(xywh[3]), 2)
                },
                'mask_polygon': mask_polygon
            }

            detections.append(detection)

    return detections


def batch_predict(model, source_dir, conf_threshold=0.25, iou_threshold=0.7, save_json=True):
    """
    批量推理

    Args:
        model: YOLO模型
        source_dir: 图像目录
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU阈值
        save_json: 是否保存JSON结果
    """
    source_path = Path(source_dir)

    if not source_path.exists():
        print(f"错误: 路径不存在 {source_dir}")
        return

    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    if source_path.is_file():
        # 单张图像
        image_files = [source_path]
    else:
        # 目录中的所有图像
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_path.glob(f'*{ext}'))
            image_files.extend(source_path.glob(f'*{ext.upper()}'))

    if not image_files:
        print(f"未找到图像文件: {source_dir}")
        return

    print(f"找到 {len(image_files)} 张图像")
    print("=" * 50)

    all_results = {}

    for img_path in image_files:
        print(f"\n处理: {img_path.name}")

        detections = predict_image(model, str(img_path), conf_threshold, iou_threshold)

        result = {
            'image': img_path.name,
            'timestamp': datetime.now().isoformat(),
            'num_cracks': len(detections),
            'detections': detections
        }

        all_results[img_path.name] = result

        # 打印检测结果
        if detections:
            print(f"检测到 {len(detections)} 个裂缝:")
            for j, det in enumerate(detections, 1):
                print(f"  裂缝 {j}:")
                print(f"    类别: {det['class']}")
                print(f"    置信度: {det['confidence']:.2%}")
                print(f"    位置: ({det['center']['x']:.0f}, {det['center']['y']:.0f})")
                print(f"    尺寸: {det['size']['width']:.0f} x {det['size']['height']:.0f}")
        else:
            print("  未检测到裂缝")

    # 保存JSON结果
    if save_json:
        output_dir = Path('outputs/predictions/inference')
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存到: {json_path}")

    # 统计信息
    total_cracks = sum(r['num_cracks'] for r in all_results.values())
    images_with_cracks = sum(1 for r in all_results.values() if r['num_cracks'] > 0)

    print("\n" + "=" * 50)
    print("统计信息")
    print("=" * 50)
    print(f"总图像数: {len(all_results)}")
    print(f"有裂缝的图像: {images_with_cracks}")
    print(f"总裂缝数: {total_cracks}")
    print(f"平均每张图像裂缝数: {total_cracks / len(all_results):.2f}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='建筑裂缝检测推理')
    parser.add_argument('--model', type=str, default='outputs/runs/crack_detection/weights/best.pt',
                        help='模型权重路径')
    parser.add_argument('--source', type=str, required=True,
                        help='图像路径或目录')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='NMS IoU阈值')
    parser.add_argument('--save-json', action='store_true',
                        help='保存JSON结果')

    args = parser.parse_args()

    # 加载模型
    print("加载模型...")
    model = YOLO(args.model)
    print(f"模型加载成功: {args.model}")

    # 推理
    batch_predict(
        model=model,
        source_dir=args.source,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_json=args.save_json
    )
