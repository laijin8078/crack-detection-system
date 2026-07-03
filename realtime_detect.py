#!/usr/bin/env python3
"""
建筑裂缝检测 - 实时检测脚本
支持多种输入源：
- USB摄像头 (启用裂缝跟踪)
- 网络摄像头/IP摄像头（RTSP流）(启用裂缝跟踪)
- 视频文件 (启用裂缝跟踪)
- 图像文件/文件夹 (单图检测)
"""

import argparse
import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import yaml

from utils.crack_postprocess import extract_crack_features, filter_results_by_class
from utils.crack_tracker import CrackTracker
from utils.crack_report import build_tracking_report, build_image_report, save_report


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


class CrackDetector:
    def __init__(self, model_path, conf_threshold=0.15, iou_threshold=0.7,
                 use_tracker=True, config_path='configs/inference_config.yaml'):
        """
        初始化裂缝检测器

        Args:
            model_path: 模型权重路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
            use_tracker: 是否启用裂缝跟踪（视频/摄像头模式下）
            config_path: 推理配置文件路径
        """
        print("加载模型...")
        _patch_spdconv()
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        print(f"模型加载成功: {model_path}")

        # 加载配置
        self.config = load_inference_config(config_path)
        pp_cfg = self.config.get('postprocess', {})
        self.min_area = pp_cfg.get('min_area_px', 50)
        self.ds_ratio = pp_cfg.get('mask_downsample_ratio', 4)
        self.target_cls = pp_cfg.get('target_class_ids', None)

        # tracker
        self.use_tracker = use_tracker
        self.tracker = None

        # FPS计算
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def _init_tracker(self, frame_shape):
        """延迟初始化 tracker（需要知道图像尺寸）"""
        if self.tracker is not None:
            return
        H, W = frame_shape[:2]
        tcfg = self.config.get('tracker', {})
        tcfg['image_diag'] = np.sqrt(W**2 + H**2)
        self.tracker = CrackTracker(tcfg)

    def process_frame(self, frame, frame_idx=None):
        """
        处理单帧图像

        Returns:
            annotated_frame: 标注后的帧
            num_cracks: 当前帧检测到的裂缝数量 (raw)
            unique_tracks: 当前累计的唯一裂缝数 (tracker 模式下有意义)
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        # 提取裂缝特征
        cracks = extract_crack_features(results, frame.shape, self.min_area, self.ds_ratio,
                                        target_class_ids=self.target_cls)
        num_cracks = len(cracks)

        # tracker 更新
        track_assignments = {}
        unique_tracks = num_cracks
        if self.use_tracker:
            self._init_tracker(frame.shape)
            if frame_idx is not None:
                track_assignments = self.tracker.update(cracks, frame_idx)
            else:
                track_assignments = self.tracker.update(cracks)
            unique_tracks = self.tracker.unique_track_count()

        # 获取标注后的图像（先过滤非目标类别）
        filter_results_by_class(results, self.target_cls)
        annotated_frame = results[0].plot()

        # 绘制 track ID
        for det_idx, tid in track_assignments.items():
            if det_idx < len(cracks):
                c = cracks[det_idx]
                cx, cy = int(c['center_xy'][0]), int(c['center_xy'][1])
                cv2.putText(annotated_frame, f'ID:{tid}', (cx - 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 更新FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

        # 显示信息
        cv2.putText(annotated_frame, f'FPS: {self.fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self.use_tracker:
            cv2.putText(annotated_frame, f'Frame cracks: {num_cracks} | Unique: {unique_tracks}',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(annotated_frame, f'Cracks: {num_cracks}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return annotated_frame, num_cracks, unique_tracks

    def _get_tracker_report(self, source_name='camera'):
        """生成当前跟踪报告"""
        if self.tracker is None:
            return None
        tracker_summary = self.tracker.summary()
        return build_tracking_report(
            source_id=source_name,
            tracker_summary=tracker_summary,
            model_name=self.config.get('model', {}).get('name', 'yolov8n-seg-cracks-joints'),
        )

    def detect_from_camera(self, camera_id=0):
        """
        从摄像头实时检测（启用裂缝跟踪）

        Args:
            camera_id: 摄像头ID或RTSP URL
        """
        if isinstance(camera_id, str) and camera_id.startswith('rtsp://'):
            print(f"连接网络摄像头: {camera_id}")
            cap = cv2.VideoCapture(camera_id)
            source_name = camera_id.split('/')[-1] or 'rtsp_camera'
        else:
            print(f"打开USB摄像头: {camera_id}")
            cap = cv2.VideoCapture(int(camera_id))
            source_name = f'camera_{camera_id}'

        if not cap.isOpened():
            print("错误: 无法打开摄像头")
            return

        print("按 'q' 键退出 | 按 's' 保存当前帧 | 按 'r' 输出跟踪报告")
        print("裂缝跟踪已启用")

        frame_id = 0
        self.use_tracker = True

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取帧")
                break

            annotated_frame, num_cracks, unique_tracks = self.process_frame(frame, frame_id)

            cv2.imshow('Crack Detection - Real-time (Tracking)', annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                output_dir = Path('outputs/predictions/realtime')
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f'frame_{frame_id:06d}.jpg'
                cv2.imwrite(str(save_path), annotated_frame)
                print(f"保存帧: {save_path}")
            elif key == ord('r'):
                report = self._get_tracker_report(source_name)
                if report:
                    filepath = save_report(report, filename=f'track_report_{source_name}_{frame_id:06d}.json')
                    print(f"跟踪报告已保存: {filepath}")
                    print(f"  当前累计: 原始检测数={report['summary']['raw_detection_count']}, "
                          f"唯一裂缝数={report['summary']['unique_crack_count']}")

            frame_id += 1

        cap.release()
        cv2.destroyAllWindows()

        # 输出最终统计
        if self.tracker:
            print(f"\n检测结束 - 最终统计:")
            print(f"  总帧数: {frame_id}")
            print(f"  唯一裂缝数: {self.tracker.unique_track_count()}")
            report = self._get_tracker_report(source_name)
            if report:
                filepath = save_report(report, filename=f'track_report_{source_name}_final.json')
                print(f"  最终报告: {filepath}")

    def detect_from_video(self, video_path, save_output=False):
        """
        从视频文件检测（启用裂缝跟踪）

        Args:
            video_path: 视频文件路径
            save_output: 是否保存输出视频
        """
        print(f"打开视频文件: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("错误: 无法打开视频文件")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")
        print("裂缝跟踪已启用")

        self.use_tracker = True

        video_writer = None
        if save_output:
            output_dir = Path('outputs/predictions/video')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'{Path(video_path).stem}_tracked.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"输出视频: {output_path}")

        print("按 'q' 键退出 | 按 'r' 输出跟踪报告")

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            annotated_frame, num_cracks, unique_tracks = self.process_frame(frame, frame_idx)

            if frame_idx % 30 == 0:
                progress = frame_idx / total_frames * 100 if total_frames else 0
                print(f"进度: {progress:.1f}% ({frame_idx}/{total_frames}), "
                      f"帧检测: {num_cracks}, 唯一裂缝: {unique_tracks}")

            if video_writer is not None:
                video_writer.write(annotated_frame)

            cv2.imshow('Crack Detection - Video (Tracking)', annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                report = self._get_tracker_report(Path(video_path).stem)
                if report:
                    filepath = save_report(report, filename=f'track_report_video_{frame_idx:06d}.json')
                    print(f"跟踪报告已保存: {filepath}")

        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

        # 输出最终统计
        if self.tracker:
            print(f"\n检测结束 - 最终统计:")
            print(f"  总帧数: {frame_idx}")
            print(f"  唯一裂缝数: {self.tracker.unique_track_count()}")
            report = self._get_tracker_report(Path(video_path).stem)
            if report:
                filepath = save_report(report, filename=f'track_report_{Path(video_path).stem}_final.json')
                print(f"  最终报告: {filepath}")

    def detect_from_images(self, image_dir):
        """
        从图像文件夹检测（单图模式，不使用 tracker）

        Args:
            image_dir: 图像目录
        """
        image_path = Path(image_dir)

        if not image_path.exists():
            print(f"错误: 路径不存在 {image_dir}")
            return

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        if image_path.is_file():
            image_files = [image_path]
        else:
            image_files = []
            for ext in image_extensions:
                image_files.extend(image_path.glob(f'*{ext}'))
                image_files.extend(image_path.glob(f'*{ext.upper()}'))
            image_files = list(dict.fromkeys(image_files))  # Windows 大小写不敏感去重

        if not image_files:
            print(f"未找到图像文件: {image_dir}")
            return

        print(f"找到 {len(image_files)} 张图像")
        print("按 'q' 键退出, 按任意键查看下一张")

        self.use_tracker = False
        all_reports = []

        for img_file in image_files:
            print(f"\n处理: {img_file.name}")

            frame = cv2.imread(str(img_file))
            if frame is None:
                print(f"无法读取图像: {img_file}")
                continue

            annotated_frame, num_cracks, _ = self.process_frame(frame)

            # 生成单图报告
            results = self.model.predict(
                source=frame, conf=self.conf_threshold,
                iou=self.iou_threshold, verbose=False,
            )
            cracks = extract_crack_features(results, frame.shape, self.min_area, self.ds_ratio,
                                        target_class_ids=self.target_cls)
            report = build_image_report(
                source_id=img_file.name,
                cracks=cracks,
                model_name=self.config.get('model', {}).get('name', 'yolov8n-seg-cracks-joints'),
            )
            all_reports.append(report)

            print(f"  检测到 {num_cracks} 个裂缝")

            cv2.imshow('Crack Detection - Images', annotated_frame)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        print(f"\n检测结束 - 共 {len(image_files)} 张图像, "
              f"总裂缝数: {sum(r['summary']['unique_crack_count'] for r in all_reports)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='建筑裂缝实时检测')
    parser.add_argument('--model', type=str, default='runs/segment/outputs/runs/crack_detection/weights/yolov8n-seg-cracks-joints.pt',
                        help='模型权重路径')
    parser.add_argument('--source', type=str, default='0',
                        help='输入源: 0=USB摄像头, rtsp://...=网络摄像头, 图像路径, 视频路径')
    parser.add_argument('--conf', type=float, default=0.15,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='NMS IoU阈值')
    parser.add_argument('--save-video', action='store_true',
                        help='保存输出视频（仅视频输入）')
    parser.add_argument('--no-tracker', action='store_true',
                        help='禁用裂缝跟踪（视频/摄像头模式下）')
    parser.add_argument('--config', type=str, default='configs/inference_config.yaml',
                        help='推理配置文件路径')

    args = parser.parse_args()

    detector = CrackDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        use_tracker=not args.no_tracker,
        config_path=args.config,
    )

    source = args.source

    if source.isdigit():
        detector.detect_from_camera(int(source))
    elif source.startswith('rtsp://') or source.startswith('http://'):
        detector.detect_from_camera(source)
    elif Path(source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        detector.detect_from_video(source, save_output=args.save_video)
    else:
        detector.detect_from_images(source)
