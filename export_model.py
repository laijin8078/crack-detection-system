#!/usr/bin/env python3
"""
模型导出：YOLOv8-seg → ONNX / FP16

用法：
  python export_model.py                          # 导出 ONNX (FP32)
  python export_model.py --half                   # 导出 ONNX (FP16)
  python export_model.py --format openvino         # 导出 OpenVINO (Intel CPU 加速)
"""

import argparse
from pathlib import Path
import torch


def patch_spdconv():
    """注入 SPDConv 模块"""
    import torch.nn as nn
    from ultralytics.nn.modules import conv as conv_module

    if hasattr(conv_module, "SPDConv"):
        return

    class SPDConv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
            super().__init__()
            self.scale = s
            self.conv = conv_module.Conv(c1 * (s ** 2), c2, k, 1, p, g, act=act)

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
        raise RuntimeError("无法找到 base_modules 注入点")
    code = compile(source, tasks_module.__file__, "exec")
    ns = dict(tasks_module.__dict__)
    exec(code, ns)
    tasks_module.parse_model = ns["parse_model"]


def main():
    parser = argparse.ArgumentParser(description="YOLOv8-seg 模型导出")
    parser.add_argument("--model", type=str,
                        default="runs/segment/outputs/runs/crack_detection/weights/yolov8n-seg-cracks-joints.pt",
                        help="模型权重路径")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "openvino", "engine", "tflite"],
                        help="导出格式")
    parser.add_argument("--half", action="store_true",
                        help="导出 FP16（体积减半，CPU 上提升有限，GPU 上提升明显）")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="导出时的输入尺寸")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset 版本")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return

    patch_spdconv()

    print(f"加载模型: {model_path}")
    from ultralytics import YOLO
    model = YOLO(str(model_path))

    # 若导出 FP16，先将模型转为 half
    if args.half:
        print("转换为 FP16（半精度）...")
        model.model.half()
        # 保存 FP16 的 .pt 文件
        half_pt = model_path.with_stem(model_path.stem + "_fp16")
        torch.save(model.model.state_dict(), half_pt)
        print(f"FP16 权重已保存: {half_pt}")

    print(f"开始导出: format={args.format}, imgsz={args.imgsz}, half={args.half}")
    export_path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        opset=args.opset,
        simplify=True,          # ONNX 图简化
        workspace=4,            # TensorRT 工作空间 (GB)
    )

    print(f"导出成功: {export_path}")

    # 打印大小对比
    pt_size = model_path.stat().st_size / 1024 / 1024
    export_file = Path(export_path)
    export_size = export_file.stat().st_size / 1024 / 1024
    print(f"\n大小对比:")
    print(f"  原始 .pt:  {pt_size:.1f} MB")
    print(f"  导出模型:  {export_size:.1f} MB")


if __name__ == "__main__":
    main()
