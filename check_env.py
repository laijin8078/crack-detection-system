#!/usr/bin/env python3
"""
环境验证脚本
检查所有必需的依赖是否正确安装
"""

import sys

def check_package(package_name, import_name=None):
    """检查包是否安装"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"[OK] {package_name:20s} 已安装")
        return True
    except ImportError:
        print(f"[NO] {package_name:20s} 未安装")
        return False

def main():
    print("=" * 50)
    print("建筑裂缝检测系统 - 环境验证")
    print("=" * 50)

    # 检查Python版本
    print(f"\nPython版本: {sys.version}")

    # 核心依赖
    print("\n核心深度学习框架:")
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"),
    ]

    all_ok = True
    for pkg, imp in packages:
        if not check_package(pkg, imp):
            all_ok = False

    # 数据处理
    print("\n数据处理:")
    packages = [
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
        ("pillow", "PIL"),
        ("pyyaml", "yaml"),
    ]

    for pkg, imp in packages:
        if not check_package(pkg, imp):
            all_ok = False

    # 可视化
    print("\n可视化与监控:")
    packages = [
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("tensorboard", "tensorboard"),
        ("tqdm", "tqdm"),
    ]

    for pkg, imp in packages:
        if not check_package(pkg, imp):
            all_ok = False

    # 评估与分析
    print("\n评估与分析:")
    packages = [
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
    ]

    for pkg, imp in packages:
        if not check_package(pkg, imp):
            all_ok = False

    # Web框架
    print("\nWeb框架:")
    packages = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
    ]

    for pkg, imp in packages:
        if not check_package(pkg, imp):
            all_ok = False

    # 数据库
    print("\n数据库:")
    packages = [
        ("sqlalchemy", "sqlalchemy"),
    ]

    for pkg, imp in packages:
        if not check_package(pkg, imp):
            all_ok = False

    # 报告生成
    print("\n报告生成:")
    packages = [
        ("reportlab", "reportlab"),
        ("fpdf2", "fpdf"),
    ]

    for pkg, imp in packages:
        if not check_package(pkg, imp):
            all_ok = False

    # PyTorch详细信息
    print("\n" + "=" * 50)
    print("PyTorch详细信息:")
    print("=" * 50)
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        else:
            print("使用CPU模式（训练速度较慢）")
    except:
        pass

    # 总结
    print("\n" + "=" * 50)
    if all_ok:
        print("[SUCCESS] 所有依赖安装成功！")
        print("=" * 50)
        print("\n下一步:")
        print("1. 训练模型: python train.py")
        print("2. 查看快速启动指南: type QUICKSTART.md")
        return 0
    else:
        print("[FAILED] 部分依赖未安装")
        print("=" * 50)
        print("\n请运行以下命令安装缺失的依赖:")
        print("pip install -r requirements.txt")
        return 1

if __name__ == '__main__':
    sys.exit(main())
