#!/usr/bin/env python3
"""
建筑裂缝检测 - 实时检测脚本
支持多种输入源：
- USB摄像头
- 网络摄像头/IP摄像头（RTSP流）
- 图像文件/文件夹
- 视频文件
"""

import argparse
import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import numpy as np


class CrackDetector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.7):
        """
        初始化裂缝检测器

        Args:
            model_path: 模型权重路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
        """
        print("加载模型...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        print(f"模型加载成功: {model_path}")

        # FPS计算
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def process_frame(self, frame):
        """
        处理单帧图像

        Args:
            frame: 输入帧

        Returns:
            annotated_frame: 标注后的帧
            num_cracks: 检测到的裂缝数量
        """
        # 推理
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        # 获取标注后的图像
        annotated_frame = results[0].plot()

        # 统计裂缝数量
        num_cracks = len(results[0].boxes) if results[0].boxes is not None else 0

        # 更新FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

        # 在图像上显示FPS和裂缝数量
        cv2.putText(annotated_frame, f'FPS: {self.fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Cracks: {num_cracks}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return annotated_frame, num_cracks

    def detect_from_camera(self, camera_id=0):
        """
        从摄像头实时检测

        Args:
            camera_id: 摄像头ID或RTSP URL
        """
        # 打开摄像头
        if isinstance(camera_id, str) and camera_id.startswith('rtsp://'):
            print(f"连接网络摄像头: {camera_id}")
            cap = cv2.VideoCapture(camera_id)
        else:
            print(f"打开USB摄像头: {camera_id}")
            cap = cv2.VideoCapture(int(camera_id))

        if not cap.isOpened():
            print("错误: 无法打开摄像头")
            return

        print("按 'q' 键退出")
        print("按 's' 键保存当前帧")

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取帧")
                break

            # 处理帧
            annotated_frame, num_cracks = self.process_frame(frame)

            # 显示结果
            cv2.imshow('Crack Detection - Real-time', annotated_frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                output_dir = Path('outputs/predictions/realtime')
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f'frame_{frame_id:06d}.jpg'
                cv2.imwrite(str(save_path), annotated_frame)
                print(f"保存帧: {save_path}")
                frame_id += 1

        cap.release()
        cv2.destroyAllWindows()
        print("检测结束")

    def detect_from_video(self, video_path, save_output=False):
        """
        从视频文件检测

        Args:
            video_path: 视频文件路径
            save_output: 是否保存输出视频
        """
        print(f"打开视频文件: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("错误: 无法打开视频文件")
            return

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")

        # 创建视频写入器
        video_writer = None
        if save_output:
            output_dir = Path('outputs/predictions/video')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'{Path(video_path).stem}_detected.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"输出视频: {output_path}")

        print("按 'q' 键退出")

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # 处理帧
            annotated_frame, num_cracks = self.process_frame(frame)

            # 显示进度
            if frame_idx % 30 == 0:
                progress = frame_idx / total_frames * 100
                print(f"进度: {progress:.1f}% ({frame_idx}/{total_frames})")

            # 保存到输出视频
            if video_writer is not None:
                video_writer.write(annotated_frame)

            # 显示结果
            cv2.imshow('Crack Detection - Video', annotated_frame)

            # 键盘控制
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("检测结束")

    def detect_from_images(self, image_dir):
        """
        从图像文件夹检测

        Args:
            image_dir: 图像目录
        """
        image_path = Path(image_dir)

        if not image_path.exists():
            print(f"错误: 路径不存在 {image_dir}")
            return

        # 支持的图像格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        if image_path.is_file():
            image_files = [image_path]
        else:
            image_files = []
            for ext in image_extensions:
                image_files.extend(image_path.glob(f'*{ext}'))
                image_files.extend(image_path.glob(f'*{ext.upper()}'))

        if not image_files:
            print(f"未找到图像文件: {image_dir}")
            return

        print(f"找到 {len(image_files)} 张图像")
        print("按 'q' 键退出, 按任意键查看下一张")

        for img_file in image_files:
            print(f"\n处理: {img_file.name}")

            # 读取图像
            frame = cv2.imread(str(img_file))
            if frame is None:
                print(f"无法读取图像: {img_file}")
                continue

            # 处理图像
            annotated_frame, num_cracks = self.process_frame(frame)

            # 显示结果
            cv2.imshow('Crack Detection - Images', annotated_frame)

            # 等待按键
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        print("检测结束")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='建筑裂缝实时检测')
    parser.add_argument('--model', type=str, default='outputs/runs/crack_detection/weights/best.pt',
                        help='模型权重路径')
    parser.add_argument('--source', type=str, default='0',
                        help='输入源: 0=USB摄像头, rtsp://...=网络摄像头, 图像路径, 视频路径')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='NMS IoU阈值')
    parser.add_argument('--save-video', action='store_true',
                        help='保存输出视频（仅视频输入）')

    args = parser.parse_args()

    # 初始化检测器
    detector = CrackDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    # 判断输入源类型
    source = args.source

    if source.isdigit():
        # USB摄像头
        detector.detect_from_camera(int(source))
    elif source.startswith('rtsp://') or source.startswith('http://'):
        # 网络摄像头
        detector.detect_from_camera(source)
    elif Path(source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # 视频文件
        detector.detect_from_video(source, save_output=args.save_video)
    else:
        # 图像文件或目录
        detector.detect_from_images(source)
