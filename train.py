#!/usr/bin/env python3
"""
建筑裂缝检测 - 训练脚本
使用YOLOv8s-seg模型进行实例分割训练
支持CPU和GPU自动检测
"""

import torch
import yaml
from ultralytics import YOLO
from pathlib import Path
import argparse
from datetime import datetime


def check_environment():
    """检查训练环境"""
    print("=" * 50)
    print("环境检查")
    print("=" * 50)
    print(f"Python版本: {torch.__version__}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("将使用CPU进行训练（速度较慢）")
    print("=" * 50)


def load_config(config_path):
    """加载训练配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_model(config_path='configs/train_config.yaml',
                aug_config_path='configs/augmentation_config.yaml',
                resume=False):
    """
    训练YOLOv8s-seg模型

    Args:
        config_path: 训练配置文件路径
        aug_config_path: 数据增强配置文件路径
        resume: 是否从上次中断处继续训练
    """
    # 检查环境
    check_environment()

    # 加载配置
    print("\n加载配置文件...")
    train_config = load_config(config_path)
    aug_config = load_config(aug_config_path)

    # 合并配置
    config = {**train_config, **aug_config}

    # 自动检测设备
    if config['device'] == '':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"使用设备: {config['device']}")
    print(f"模型: {config['model']}")
    print(f"数据集: {config['data']}")
    print(f"训练轮数: {config['epochs']}")
    print(f"批大小: {config['batch']}")

    # 初始化模型
    print("\n初始化模型...")
    if resume:
        # 从上次中断处继续
        model = YOLO('outputs/weights/last.pt')
        print("从上次训练继续...")
    else:
        # 加载预训练权重
        model = YOLO(config['model'])
        print(f"加载预训练权重: {config['model']}")

    # 开始训练1
    print("\n" + "=" * 50)
    print(f"开始训练 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    results = model.train(
        data=config['data'],
        epochs=config['epochs'],
        batch=config['batch'],
        imgsz=config['imgsz'],
        device=config['device'],
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        lrf=config['lrf'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        warmup_epochs=config['warmup_epochs'],
        warmup_momentum=config['warmup_momentum'],
        warmup_bias_lr=config['warmup_bias_lr'],
        patience=config['patience'],
        save=config['save'],
        save_period=config['save_period'],
        workers=config['workers'],
        val=config['val'],
        plots=config['plots'],
        # 数据增强参数
        hsv_h=config['hsv_h'],
        hsv_s=config['hsv_s'],
        hsv_v=config['hsv_v'],
        degrees=config['degrees'],
        translate=config['translate'],
        scale=config['scale'],
        shear=config['shear'],
        perspective=config['perspective'],
        flipud=config['flipud'],
        fliplr=config['fliplr'],
        mosaic=config['mosaic'],
        mixup=config['mixup'],
        copy_paste=config['copy_paste'],
        # 项目名称
        project='outputs/runs',
        name='crack_detection',
        exist_ok=True,
        resume=resume
    )

    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)

    # 在验证集上评估
    print("\n在验证集上评估...")
    metrics = model.val()

    # 打印关键指标
    print("\n" + "=" * 50)
    print("验证集性能指标")
    print("=" * 50)
    print(f"Box mAP@0.5: {metrics.box.map50:.4f}")
    print(f"Box mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p:.4f}")
    print(f"Recall: {metrics.box.r:.4f}")
    print(f"Mask mAP@0.5: {metrics.seg.map50:.4f}")
    print(f"Mask mAP@0.5:0.95: {metrics.seg.map:.4f}")
    print("=" * 50)

    # 保存最终模型路径
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    print(f"\n最佳模型保存在: {best_model_path}")
    print(f"训练日志保存在: {results.save_dir}")

    return results, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练建筑裂缝检测模型')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='训练配置文件路径')
    parser.add_argument('--aug-config', type=str, default='configs/augmentation_config.yaml',
                        help='数据增强配置文件路径')
    parser.add_argument('--resume', action='store_true',
                        help='从上次中断处继续训练')

    args = parser.parse_args()

    try:
        results, metrics = train_model(
            config_path=args.config,
            aug_config_path=args.aug_config,
            resume=args.resume
        )
        print("\n训练成功完成！")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        raise

