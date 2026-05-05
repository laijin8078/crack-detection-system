@echo off
REM 建筑裂缝检测系统 - Windows依赖安装脚本

echo ================================
echo 建筑裂缝检测系统 - 环境配置
echo ================================

REM 检查Python版本
echo.
echo 检查Python版本...
python --version

REM 升级pip
echo.
echo 升级pip...
python -m pip install --upgrade pip

REM 选择安装模式
echo.
echo 请选择安装模式：
echo 1. GPU版本（需要NVIDIA GPU和CUDA）
echo 2. CPU版本
set /p choice="请输入选项 (1/2): "

if "%choice%"=="1" (
    echo.
    echo 安装PyTorch GPU版本...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else if "%choice%"=="2" (
    echo.
    echo 安装PyTorch CPU版本...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else (
    echo 无效选项，退出安装
    exit /b 1
)

REM 安装Ultralytics YOLOv8
echo.
echo 安装Ultralytics YOLOv8...
pip install ultralytics

REM 安装其他依赖
echo.
echo 安装其他依赖包...
pip install -r requirements.txt

REM 验证安装
echo.
echo ================================
echo 验证安装
echo ================================

echo.
echo 检查PyTorch...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

echo.
echo 检查Ultralytics...
python -c "from ultralytics import YOLO; print('Ultralytics安装成功')"

echo.
echo 检查其他关键包...
python -c "import yaml; import cv2; import fastapi; print('所有依赖包安装成功')"

echo.
echo ================================
echo 安装完成！
echo ================================
echo.
echo 下一步：
echo 1. 训练模型: python train.py
echo 2. 查看快速启动指南: type QUICKSTART.md
echo.
pause
