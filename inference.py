#!/usr/bin/env python3
"""
建筑裂缝检测 - 推理脚本
支持三种模式：
  image:          单张/批量图像推理，单图检测与基础统计
  video:          视频输入，启用连续帧 tracker，输出唯一 track_id 数量
  image_sequence: 多张连续墙面图像，启用跨图去重逻辑
"""

import argparse
import json
import yaml
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

from utils.crack_postprocess import extract_crack_features, filter_results_by_class
from utils.crack_tracker import CrackTracker
from utils.crack_dedup import deduplicate_cracks
from utils.crack_report import (
    build_image_report,
    build_tracking_report,
    build_dedup_report,
    save_report,
    NumpyEncoder,
)


def _patch_spdconv():
    """在运行时向已安装的 ultralytics 注入 SPDConv 模块"""
    import torch.nn as nn
    from ultralytics.nn.modules import conv as conv_module

    if hasattr(conv_module, "SPDConv"):
        return

    class SPDConv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
            super().__init__()
            self.scale = s
            self.conv = conv_module.Conv(c1 * (s**2), c2, k, 1, p, g, act=act)

        def forward(self, x):
            return self.conv(nn.PixelUnshuffle(self.scale)(x))

    SPDConv.__module__ = "ultralytics.nn.modules.conv"
    SPDConv.__qualname__ = "SPDConv"

    conv_module.SPDConv = SPDConv
    if "SPDConv" not in conv_module.__all__:
        conv_module.__all__ = (*conv_module.__all__, "SPDConv")

    import ultralytics.nn.modules as modules_pkg
    modules_pkg.SPDConv = SPDConv
    if "SPDConv" not in modules_pkg.__all__:
        modules_pkg.__all__ = (*modules_pkg.__all__, "SPDConv")

    import ultralytics.nn.tasks as tasks_module
    tasks_module.SPDConv = SPDConv

    import inspect, textwrap
    source = inspect.getsource(tasks_module.parse_model)
    source = textwrap.dedent(source)
    patterns = [
        ("SCDown,\n            C2fCIB,", "SCDown,\n            SPDConv,\n            C2fCIB,"),
        ("SCDown,\n                C2fCIB,", "SCDown,\n                SPDConv,\n                C2fCIB,"),
    ]
    for old, new in patterns:
        if old in source:
            source = source.replace(old, new)
            break
    else:
        raise RuntimeError("无法找到 base_modules 注入点，请检查 ultralytics 版本兼容性")
    code = compile(source, tasks_module.__file__, "exec")
    ns = dict(tasks_module.__dict__)
    exec(code, ns)
    tasks_module.parse_model = ns["parse_model"]


def load_inference_config(config_path='configs/inference_config.yaml'):
    """加载推理配置文件"""
    path = Path(config_path)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def predict_single_image(model, image_path, config, save_images=True):
    """
    对单张图像进行推理并提取裂缝特征

    Returns:
        cracks: list of crack feature dicts
        image: np.ndarray 原始图像
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图像: {image_path}")
        return [], None

    pp_cfg = config.get('postprocess', {})
    min_area = pp_cfg.get('min_area_px', 50)
    ds_ratio = pp_cfg.get('mask_downsample_ratio', 4)
    target_cls = pp_cfg.get('target_class_ids', None)

    results = model.predict(
        source=image,
        conf=config.get('model', {}).get('conf_threshold', 0.15),
        iou=config.get('model', {}).get('iou_threshold', 0.7),
        imgsz=config.get('model', {}).get('imgsz', 640),
        verbose=False,
    )
    cracks = extract_crack_features(results, image.shape, min_area, ds_ratio,
                                    target_class_ids=target_cls)

    # 手动保存标注图片（避免 YOLO numpy 模式下只存一张的问题）
    if save_images:
        filter_results_by_class(results, target_cls)
        save_dir = Path('outputs/predictions/inference')
        save_dir.mkdir(parents=True, exist_ok=True)
        annotated = results[0].plot()
        fname = Path(image_path).stem + '.jpg'
        cv2.imwrite(str(save_dir / fname), annotated)

    return cracks, image


# ==================== Mode: image ====================

def run_image_mode(model, source, config, save_json=True, save_images=True):
    """
    单张/批量图像推理模式
    不做去重，仅做单图检测与基础统计
    """
    source_path = Path(source)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    if source_path.is_file():
        image_files = [source_path]
    else:
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_path.glob(f'*{ext}'))
            image_files.extend(source_path.glob(f'*{ext.upper()}'))
        image_files = list(dict.fromkeys(image_files))  # Windows 大小写不敏感去重

    if not image_files:
        print(f"未找到图像文件: {source}")
        return

    print(f"找到 {len(image_files)} 张图像")
    print(f"模式: image（单图检测）")
    print("=" * 50)

    all_reports = []
    total_raw = 0

    pp_cfg = config.get('postprocess', {})
    min_area = pp_cfg.get('min_area_px', 50)
    ds_ratio = pp_cfg.get('mask_downsample_ratio', 4)

    model_cfg = config.get('model', {})
    conf = model_cfg.get('conf_threshold', 0.15)
    iou = model_cfg.get('iou_threshold', 0.7)

    for img_path in image_files:
        print(f"\n处理: {img_path.name}")
        cracks, image = predict_single_image(model, img_path, config, save_images=save_images)
        total_raw += len(cracks)

        report = build_image_report(
            source_id=img_path.name,
            cracks=cracks,
            model_name=config.get('model', {}).get('name', 'yolov8n-seg-cracks-joints'),
        )
        all_reports.append(report)

        if cracks:
            print(f"  检测到 {len(cracks)} 个裂缝")
            for j, c in enumerate(cracks, 1):
                print(f"    裂缝 {j}: conf={c['confidence']:.2%}, "
                      f"center=({c['center_xy'][0]:.0f}, {c['center_xy'][1]:.0f}), "
                      f"area={c['area_px']}px, angle={c['orientation_angle']:.1f}°")
        else:
            print("  未检测到裂缝")

    # 汇总
    print("\n" + "=" * 50)
    print("统计信息 (image 模式)")
    print("=" * 50)
    print(f"总图像数: {len(image_files)}")
    print(f"总裂缝检测数 (raw): {total_raw}")
    print(f"有裂缝的图像: {sum(1 for r in all_reports if r['summary']['unique_crack_count'] > 0)}")
    print("=" * 50)

    # 保存
    if save_json:
        output_dir = Path('outputs/reports')
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        summary_report = {
            'source': str(source),
            'mode': 'image',
            'total_images': len(image_files),
            'total_raw_cracks': total_raw,
            'per_image': all_reports,
        }
        json_path = output_dir / f'image_summary_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"\n报告已保存到: {json_path}")
        return str(json_path)

    return None


# ==================== Mode: video ====================

def run_video_mode(model, source, config, save_json=True, save_video=False):
    """
    视频推理模式
    启用连续帧 tracker，输出唯一 track 数量
    """
    print(f"打开视频文件: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("错误: 无法打开视频文件")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")
    print(f"模式: video（连续帧跟踪）")

    # 初始化 tracker
    tcfg = config.get('tracker', {})
    tcfg['image_diag'] = np.sqrt(width**2 + height**2)
    tracker = CrackTracker(tcfg)

    pp_cfg = config.get('postprocess', {})
    min_area = pp_cfg.get('min_area_px', 50)
    ds_ratio = pp_cfg.get('mask_downsample_ratio', 4)
    target_cls = pp_cfg.get('target_class_ids', None)

    model_cfg = config.get('model', {})
    conf = model_cfg.get('conf_threshold', 0.15)
    iou = model_cfg.get('iou_threshold', 0.7)

    frame_idx = 0
    frame_raw_counts = []

    # 视频写入器
    video_writer = None
    if save_video:
        output_dir = Path('outputs/predictions/video')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{Path(source).stem}_tracked.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"输出视频: {output_path}")

    print("\n处理视频帧...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLO 推理
        results = model.predict(
            source=frame, conf=conf, iou=iou, verbose=False,
            imgsz=config.get('model', {}).get('imgsz', 640),
        )
        cracks = extract_crack_features(results, frame.shape, min_area, ds_ratio,
                                        target_class_ids=target_cls)
        frame_raw_counts.append(len(cracks))

        # 更新 tracker
        assignments = tracker.update(cracks, frame_idx)

        # 可视化
        filter_results_by_class(results, target_cls)
        annotated = results[0].plot()
        for det_idx, tid in assignments.items():
            c = cracks[det_idx]
            cx, cy = int(c['center_xy'][0]), int(c['center_xy'][1])
            cv2.putText(annotated, f'ID:{tid}', (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.putText(annotated, f'Frame: {frame_idx}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f'Unique tracks: {tracker.unique_track_count()}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if video_writer:
            video_writer.write(annotated)

        if frame_idx % 100 == 0:
            progress = frame_idx / total_frames * 100 if total_frames else 0
            print(f"  进度: {progress:.1f}% ({frame_idx}/{total_frames}), "
                  f"当前帧检测: {len(cracks)}, 累计唯一裂缝: {tracker.unique_track_count()}")

    cap.release()
    if video_writer:
        video_writer.release()

    # 生成跟踪报告
    raw_total = tracker.total_detection_count()
    tracker_summary = tracker.summary()
    report = build_tracking_report(
        source_id=Path(source).name,
        tracker_summary=tracker_summary,
        raw_detection_count=raw_total,
        model_name=config.get('model', {}).get('name', 'yolov8n-seg-cracks-joints'),
    )

    print("\n" + "=" * 50)
    print("视频跟踪统计")
    print("=" * 50)
    print(f"总帧数: {frame_idx}")
    print(f"原始检测总数 (raw): {raw_total}")
    print(f"有效 track 检测数: {tracker_summary.get('valid_track_detections', raw_total)}")
    print(f"短轨迹过滤检测数: {tracker_summary.get('short_track_detections', 0)}")
    print(f"唯一裂缝数 (unique tracks): {tracker.unique_track_count()}")
    print(f"重复识别去除数: {raw_total - tracker.unique_track_count()}")
    print("=" * 50)

    if save_json:
        filepath = save_report(report, filename=f'video_track_{Path(source).stem}.json')
        print(f"\n跟踪报告已保存到: {filepath}")
        return filepath

    return None


# ==================== Mode: image_sequence ====================

def run_image_sequence_mode(model, source, config, wall_id=None, save_json=True, save_images=True):
    """
    多图序列推理模式
    对多张图像进行跨图去重，为同一裂缝分配统一 crack_id

    Args:
        wall_id: 墙面标识符。同 wall_id 的图像才参与去重匹配。
                 未指定时默认使用输入目录名。
                 不同墙面的图像应使用不同 wall_id 分开处理。
    """
    source_path = Path(source)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    if source_path.is_file():
        image_files = [source_path]
    else:
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_path.glob(f'*{ext}'))
            image_files.extend(source_path.glob(f'*{ext.upper()}'))
        image_files = list(dict.fromkeys(image_files))  # Windows 大小写不敏感去重

    if not image_files:
        print(f"未找到图像文件: {source}")
        return

    # wall_id 默认值：输入目录名
    if wall_id is None:
        wall_id = source_path.name if source_path.is_dir() else source_path.stem
        print(f"注意: 未指定 --wall-id，默认使用 '{wall_id}' 作为墙面标识")
        print(f"不同墙面的图像请勿混在同一目录，应分目录存放并分别为每组指定 wall_id")

    print(f"Wall ID: {wall_id}")
    print(f"找到 {len(image_files)} 张图像")
    print(f"模式: image_sequence（跨图去重）")
    print("=" * 50)

    all_cracks_per_image = []
    image_ids = []
    image_paths = []

    for img_path in image_files:
        print(f"\n处理: {img_path.name}")
        cracks, image = predict_single_image(model, img_path, config, save_images=save_images)
        all_cracks_per_image.append(cracks)
        image_ids.append(img_path.name)
        image_paths.append(str(img_path))

        if cracks:
            print(f"  检测到 {len(cracks)} 个裂缝")
        else:
            print("  未检测到裂缝")

    # 跨图去重
    print("\n" + "=" * 50)
    print("执行跨图去重...")
    dedup_cfg = config.get('dedup', {})
    debug_dedup = config.get('dedup', {}).get('debug_dedup', False)
    dedup_result = deduplicate_cracks(all_cracks_per_image, image_ids, dedup_cfg,
                                      debug=debug_dedup, image_paths=image_paths)

    print(f"原始检测总数: {dedup_result['raw_detection_count']}")
    print(f"唯一裂缝数: {dedup_result['unique_crack_count']}")
    print(f"重复去除数: {dedup_result['duplicate_removed_count']}")
    print(f"匹配组数: {dedup_result['matched_groups']}")
    print(f"无匹配单条: {dedup_result['unmatched_singles']}")
    print("=" * 50)

    # 打印匹配组信息
    if dedup_result['matched_groups'] > 0:
        print("\n跨图匹配裂缝组:")
        for crack in dedup_result['cracks']:
            if crack['is_duplicate']:
                images = [a['image_id'] for a in crack['appearances']]
                print(f"  {crack['crack_id']}: 出现在 {len(images)} 张图像中: {', '.join(images)}")

    # 生成报告
    report = build_dedup_report(
        source_id=str(source),
        dedup_result=dedup_result,
        wall_id=wall_id,
        model_name=config.get('model', {}).get('name', 'yolov8n-seg-cracks-joints'),
    )

    if save_json:
        filepath = save_report(report, filename=f'dedup_{Path(source).name if source_path.is_dir() else source_path.stem}.json')
        print(f"\n去重报告已保存到: {filepath}")
        return filepath

    return None


# ==================== Main ====================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='建筑裂缝检测推理')
    parser.add_argument('--model', type=str,
                        default='runs/segment/outputs/runs/crack_detection/weights/yolov8n-seg-cracks-joints.pt',
                        help='模型权重路径')
    parser.add_argument('--source', type=str, default='测试图片',
                        help='输入路径：图像文件、图像目录、视频文件')
    parser.add_argument('--mode', type=str, default='image',
                        choices=['image', 'video', 'image_sequence'],
                        help='推理模式: image=单图检测, video=视频跟踪, image_sequence=跨图去重')
    parser.add_argument('--conf', type=float, default=None,
                        help='置信度阈值（覆盖配置文件）')
    parser.add_argument('--iou', type=float, default=None,
                        help='NMS IoU阈值（覆盖配置文件）')
    parser.add_argument('--config', type=str, default='configs/inference_config.yaml',
                        help='推理配置文件路径')
    parser.add_argument('--save-json', action='store_true', default=True,
                        help='保存JSON结果')
    parser.add_argument('--save-video', action='store_true',
                        help='保存跟踪标注视频（仅 video 模式）')
    parser.add_argument('--wall-id', type=str, default=None,
                        help='墙面/序列标识（仅 image_sequence 模式），未指定时默认使用输入目录名')
    parser.add_argument('--no-save', action='store_true',
                        help='不保存JSON结果')
    parser.add_argument('--advice', action='store_true',
                        help='检测完成后自动调用 DeepSeek 生成维修建议')
    parser.add_argument('--advice-pdf', action='store_true',
                        help='生成维修建议的同时输出 PDF 报告（需同时指定 --advice）')
    parser.add_argument('--save-images', action='store_true', default=True,
                        help='保存带标注框和 mask 的预测图片到 outputs/predictions/（默认开启，使用 --no-save-images 禁用）')
    parser.add_argument('--no-save-images', action='store_false', dest='save_images',
                        help='不保存预测图片')
    parser.add_argument('--debug-dedup', action='store_true',
                        help='输出跨图去重匹配得分明细和骨架可视化（仅 image_sequence 模式）')

    args = parser.parse_args()

    # 加载配置
    config = load_inference_config(args.config)

    # 命令行参数覆盖配置文件
    if args.conf is not None:
        config.setdefault('model', {})['conf_threshold'] = args.conf
    if args.iou is not None:
        config.setdefault('model', {})['iou_threshold'] = args.iou

    save_json = not args.no_save

    # debug-dedup 传入 config
    if args.debug_dedup:
        config.setdefault('dedup', {})['debug_dedup'] = True

    # 自动推断 mode
    if args.mode == 'image':
        source_path = Path(args.source)
        if source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            print(f"检测到视频文件，自动切换为 video 模式")
            args.mode = 'video'

    # 加载模型
    print(f"加载模型...")
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        print("请指定正确的模型路径，例如: python inference.py --model runs/segment/outputs/runs/crack_detection/weights/yolov8n-seg-cracks-joints.pt")
        exit(1)
    _patch_spdconv()
    model = YOLO(str(model_path))
    print(f"模型加载成功: {model_path}")
    print(f"推理模式: {args.mode}")

    # 按模式运行
    report_path = None
    if args.mode == 'image':
        report_path = run_image_mode(model, args.source, config, save_json=save_json, save_images=args.save_images)
    elif args.mode == 'video':
        report_path = run_video_mode(model, args.source, config, save_json=save_json, save_video=args.save_video)
    elif args.mode == 'image_sequence':
        report_path = run_image_sequence_mode(model, args.source, config, wall_id=args.wall_id, save_json=save_json, save_images=args.save_images)
    else:
        print(f"未知模式: {args.mode}")

    # 自动生成维修建议
    if args.advice and report_path:
        from utils.deepseek_advisor import generate_advice
        print("\n" + "=" * 50)
        print("自动生成 DeepSeek 维修建议...")
        print("=" * 50)
        try:
            advice, advice_path = generate_advice(report_path)
            print(f"\n维修建议已保存到: {advice_path}")
            if args.advice_pdf:
                from utils.advice_pdf import generate_advice_pdf
                pdf_path = generate_advice_pdf(advice_path)
                print(f"PDF 报告已保存到: {pdf_path}")
        except RuntimeError as e:
            print(f"维修建议生成失败: {e}")
