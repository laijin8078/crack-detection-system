#!/usr/bin/env python3
"""
建筑裂缝检测 - 训练脚本
使用YOLOv8-seg模型进行实例分割训练
支持CPU和GPU自动检测，支持SPD-Conv改进模型
"""

import torch
import yaml
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
from datetime import datetime


def _patch_spdconv():
    """在运行时向已安装的 ultralytics 注入 SPDConv 模块"""
    import torch.nn as nn
    from ultralytics.nn.modules import conv as conv_module

    if hasattr(conv_module, "SPDConv"):
        return

    class SPDConv(nn.Module):
        """Space-to-Depth Convolution module."""

        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
            super().__init__()
            self.scale = s
            self.conv = conv_module.Conv(c1 * (s**2), c2, k, 1, p, g, act=act)

        def forward(self, x):
            return self.conv(nn.PixelUnshuffle(self.scale)(x))

    # 让 pickle 能找到 SPDConv：伪装成来自 ultralytics.nn.modules.conv
    SPDConv.__module__ = "ultralytics.nn.modules.conv"
    SPDConv.__qualname__ = "SPDConv"

    # 注入到 conv 模块
    conv_module.SPDConv = SPDConv
    if "SPDConv" not in conv_module.__all__:
        conv_module.__all__ = (*conv_module.__all__, "SPDConv")

    # 注入到 nn.modules 包
    import ultralytics.nn.modules as modules_pkg
    modules_pkg.SPDConv = SPDConv
    if "SPDConv" not in modules_pkg.__all__:
        modules_pkg.__all__ = (*modules_pkg.__all__, "SPDConv")

    # 注入到 tasks 模块（YAML 解析器通过 globals()[m] 查找）
    import ultralytics.nn.tasks as tasks_module
    tasks_module.SPDConv = SPDConv

    # 重定义 parse_model，在 base_modules 中加入 SPDConv
    import inspect
    import textwrap
    source = inspect.getsource(tasks_module.parse_model)
    source = textwrap.dedent(source)

    # 尝试在 base_modules frozenset 里加入 SPDConv
    patterns = [
        ("SCDown,\n            C2fCIB,", "SCDown,\n            SPDConv,\n            C2fCIB,"),
        ("SCDown,\n                C2fCIB,", "SCDown,\n                SPDConv,\n                C2fCIB,"),
    ]
    patched = False
    for old, new in patterns:
        if old in source:
            source = source.replace(old, new)
            patched = True
            break

    if not patched:
        raise RuntimeError("无法找到 base_modules 注入点，请检查 ultralytics 版本兼容性")

    code = compile(source, tasks_module.__file__, "exec")
    ns = dict(tasks_module.__dict__)
    exec(code, ns)
    tasks_module.parse_model = ns["parse_model"]


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
                resume=False,
                model_path=None,
                use_spd=False):
    """
    训练YOLOv8-seg模型

    Args:
        config_path: 训练配置文件路径
        aug_config_path: 数据增强配置文件路径
        resume: 是否从上次中断处继续训练
        model_path: 覆盖配置文件中的模型路径
        use_spd: 使用SPD-Conv改进的模型结构
    """
    # 检查环境
    check_environment()

    # 加载配置
    print("\n加载配置文件...")
    train_config = load_config(config_path)
    aug_config = load_config(aug_config_path)

    # 合并配置
    config = {**train_config, **aug_config}

    # 确定模型
    if model_path:
        config['model'] = model_path
    elif use_spd:
        config['model'] = 'ultralytics/ultralytics/cfg/models/v8/yolov8s-seg-spd.yaml'

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
        last_pt = Path('runs/segment/outputs/runs/crack_detection/weights/last.pt')
        model = YOLO(str(last_pt))
        print(f"从上次训练继续...({last_pt})")
    elif use_spd:
        # 注入 SPDConv 到已安装的 ultralytics
        print("使用SPD-Conv改进模型结构...")
        _patch_spdconv()
        model = YOLO(config['model'])
        print(f"模型结构已加载: {config['model']}")
    else:
        # 加载预训练权重
        model = YOLO(config['model'])
        print(f"加载预训练权重: {config['model']}")

    # 开始训练
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
        cos_lr=config['cos_lr'],
        dropout=config['dropout'],
        multi_scale=config['multi_scale'],
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
        mask_ratio=config['mask_ratio'],
        box=config['box'],
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

    def safe_mean(x):
        """将 Ultralytics 返回的标量/数组安全转为 float 平均值"""
        if x is None:
            return 0.0
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            return 0.0
        return float(np.nanmean(arr))

    # 打印关键指标
    print("\n" + "=" * 50)
    print("验证集性能指标")
    print("=" * 50)
    print(f"Box mAP@0.5: {metrics.box.map50:.4f}")
    print(f"Box mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Box Precision: {safe_mean(metrics.box.p):.4f}")
    print(f"Box Recall: {safe_mean(metrics.box.r):.4f}")
    print(f"Mask mAP@0.5: {metrics.seg.map50:.4f}")
    print(f"Mask mAP@0.5:0.95: {metrics.seg.map:.4f}")
    print(f"Mask Precision: {safe_mean(metrics.seg.p):.4f}")
    print(f"Mask Recall: {safe_mean(metrics.seg.r):.4f}")
    print("=" * 50)

    # 保存最终模型路径
    best_model_path = Path(results.save_dir) / 'weights' / 'best1.pt'
    print(f"\n最佳模型保存在: {best_model_path}")
    print(f"训练日志保存在: {results.save_dir}")

    return results, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练建筑裂缝检测模型')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='训练配置文件路径')
    parser.add_argument('--aug-config', type=str, default='configs/augmentation_config.yaml',
                        help='数据增强配置文件路径')
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径（覆盖配置文件中的设置）')
    parser.add_argument('--spd', action='store_true',
                        help='使用SPD-Conv改进的模型结构')
    parser.add_argument('--resume', action='store_true',
                        help='从上次中断处继续训练')

    args = parser.parse_args()

    try:
        results, metrics = train_model(
            config_path=args.config,
            aug_config_path=args.aug_config,
            resume=args.resume,
            model_path=args.model,
            use_spd=args.spd
        )
        print("\n训练成功完成！")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        raise

