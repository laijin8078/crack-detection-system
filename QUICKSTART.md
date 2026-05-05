# 快速启动指南

## 第一步：安装依赖

```bash
# 1. 安装PyTorch（根据你的系统选择合适的版本）
# GPU版本（推荐，需要NVIDIA GPU）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU版本（如果没有GPU）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. 安装YOLOv8
pip install ultralytics

# 3. 安装其他依赖
pip install -r requirements.txt
```

## 第二步：验证环境

```bash
# 检查PyTorch是否安装成功
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

# 检查Ultralytics是否安装成功
python -c "from ultralytics import YOLO; print('Ultralytics安装成功')"
```

## 第三步：训练模型

```bash
# 开始训练（自动下载预训练权重）
python train.py

# 训练过程中可以在另一个终端查看TensorBoard
tensorboard --logdir outputs/runs
```

**训练时间估计：**
- GPU（RTX 3060）：3-5小时
- CPU：20-30小时

**训练完成后，最佳模型保存在：**
```
outputs/runs/crack_detection/weights/best.pt
```

## 第四步：测试模型

### 4.1 单张图像测试

```bash
python inference.py --source data/crack-seg/test/images/1616.rf.c868709931a671796794fdbb95352c5a.jpg --save-json
```

### 4.2 批量测试

```bash
python inference.py --source data/crack-seg/test/images/ --save-json
```

### 4.3 评估模型

```bash
python evaluate.py --models outputs/runs/crack_detection/weights/best.pt --save
```

## 第五步：实时检测

### 5.1 摄像头检测

```bash
# USB摄像头
python realtime_detect.py --source 0

# 按 'q' 退出
# 按 's' 保存当前帧
```

### 5.2 视频文件检测

```bash
python realtime_detect.py --source video.mp4 --save-video
```

## 第六步：启动Web服务

```bash
# 启动Web服务
uvicorn app:app --reload

# 访问Web界面
# 浏览器打开: http://localhost:8000/static/index.html
```

## 常用命令速查

### 训练相关
```bash
# 基础训练
python train.py

# 继续训练
python train.py --resume

# 查看训练日志
tensorboard --logdir outputs/runs
```

### 推理相关
```bash
# 图像推理
python inference.py --source image.jpg

# 实时检测
python realtime_detect.py --source 0

# 视频检测
python realtime_detect.py --source video.mp4
```

### 评估相关
```bash
# 评估模型
python evaluate.py --models best.pt --save

# 对比模型
python evaluate.py --models model1.pt model2.pt --save
```

### Web服务
```bash
# 开发模式
uvicorn app:app --reload

# 生产模式
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## 故障排除

### 问题1：显存不足
**解决方案：**
1. 减小batch size：编辑`configs/train_config.yaml`，将`batch: 16`改为`batch: 8`或更小
2. 使用更小的模型：将`model: yolov8s-seg.pt`改为`model: yolov8n-seg.pt`

### 问题2：模型文件不存在
**解决方案：**
首次运行时，YOLOv8会自动下载预训练权重。如果下载失败，可以手动下载：
```bash
# 下载YOLOv8s-seg预训练权重
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt
```

### 问题3：Web服务无法访问
**解决方案：**
1. 确认服务已启动：`uvicorn app:app --reload`
2. 检查端口是否被占用：`netstat -ano | findstr :8000`
3. 尝试使用其他端口：`uvicorn app:app --port 8080`

### 问题4：摄像头无法打开
**解决方案：**
1. 检查摄像头是否被其他程序占用
2. 尝试不同的摄像头ID：`--source 1`或`--source 2`
3. 检查摄像头权限

## 下一步

1. **优化模型**：调整训练参数，提高检测精度
2. **扩展功能**：实现多类别裂缝分类（横向、纵向、网状）
3. **部署应用**：将系统部署到服务器或边缘设备
4. **集成系统**：与现有的建筑管理系统集成

## 获取帮助

- 查看完整文档：`README.md`
- 查看API文档：http://localhost:8000/docs
- 查看实施计划：`.claude/plans/rosy-zooming-gray.md`
