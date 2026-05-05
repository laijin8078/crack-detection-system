# 建筑裂缝检测深度学习系统

基于YOLOv8-seg的建筑表面裂缝检测系统，实现裂缝的自动检测、位置标注和分类识别。

## 功能特性

- ✅ **裂缝检测**: 自动检测建筑表面裂缝
- ✅ **位置标注**: 精确标注裂缝位置（边界框+分割掩码）
- ✅ **类别识别**: 支持单类别和多类别裂缝识别
- ✅ **实时检测**: 支持USB摄像头、网络摄像头、图像、视频等多种输入
- ✅ **Web界面**: 提供友好的Web操作界面
- ✅ **结果存储**: 自动保存检测结果到数据库
- ✅ **报告生成**: 自动生成PDF检测报告
- ✅ **统计分析**: 提供详细的统计图表

## 系统架构

```
数据采集 → 模型推理 → 结果存储 → 报告生成 → Web展示
```

## 技术栈

- **深度学习框架**: PyTorch, Ultralytics YOLOv8
- **Web框架**: FastAPI
- **前端**: HTML, CSS, JavaScript, Bootstrap
- **数据库**: SQLite
- **报告生成**: ReportLab

## 项目结构

```
模型/
├── data/                   # 数据集
│   └── crack-seg/
├── configs/                # 配置文件
│   ├── train_config.yaml
│   └── augmentation_config.yaml
├── utils/                  # 工具模块
│   ├── database.py
│   └── report_generator.py
├── database/               # 数据库
│   ├── schema.sql
│   └── crack_detection.db
├── static/                 # Web前端
│   ├── index.html
│   ├── css/
│   └── js/
├── outputs/                # 输出结果
│   ├── runs/              # 训练记录
│   ├── weights/           # 模型权重
│   ├── predictions/       # 预测结果
│   └── reports/           # 生成的报告
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── inference.py           # 推理脚本
├── realtime_detect.py     # 实时检测
├── app.py                 # Web后台服务
└── requirements.txt       # 依赖列表
```

## 快速开始

### 1. 环境安装

```bash
# 安装PyTorch（根据你的CUDA版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install ultralytics
pip install -r requirements.txt
```

### 2. 数据准备

数据集已准备在`data/crack-seg/`目录：
- 训练集: 3717张图像
- 验证集: 200张图像
- 测试集: 112张图像
- 格式: YOLO分割格式

### 3. 模型训练

```bash
# 训练YOLOv8s-seg模型
python train.py

# 从上次中断处继续训练
python train.py --resume

# 使用自定义配置
python train.py --config configs/train_config.yaml
```

训练过程会自动：
- 检测GPU/CPU环境
- 记录训练指标（loss、mAP、Precision、Recall）
- 保存最佳模型
- 生成TensorBoard日志

查看训练日志：
```bash
tensorboard --logdir outputs/runs
```

### 4. 模型评估

```bash
# 评估单个模型
python evaluate.py --models outputs/runs/crack_detection/weights/best.pt --save

# 对比多个模型
python evaluate.py --models model1.pt model2.pt model3.pt --save
```

### 5. 推理检测

**单张图像推理：**
```bash
python inference.py --source image.jpg --save-json
```

**批量推理：**
```bash
python inference.py --source data/crack-seg/test/images/ --save-json
```

### 6. 实时检测

**USB摄像头：**
```bash
python realtime_detect.py --source 0
```

**网络摄像头：**
```bash
python realtime_detect.py --source rtsp://192.168.1.100:554/stream
```

**视频文件：**
```bash
python realtime_detect.py --source video.mp4 --save-video
```

**图像文件夹：**
```bash
python realtime_detect.py --source images/
```

### 7. Web服务

启动Web服务：
```bash
# 开发环境
uvicorn app:app --reload

# 生产环境
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

访问Web界面：
```
http://localhost:8000/static/index.html
```

API文档：
```
http://localhost:8000/docs
```

## API接口

### 检测接口
```
POST /api/detect
Content-Type: multipart/form-data

参数:
- file: 图像文件
- conf_threshold: 置信度阈值（默认0.25）
- iou_threshold: NMS IoU阈值（默认0.7）

返回:
{
  "success": true,
  "detection_id": 1,
  "num_cracks": 3,
  "detections": [...],
  "result_image_url": "/outputs/predictions/..."
}
```

### 历史记录
```
GET /api/detections?limit=10
```

### 统计信息
```
GET /api/statistics
```

## 性能指标

### 模型性能（YOLOv8s-seg）
- **mAP@0.5**: 88-91%
- **Precision**: 89-93%
- **Recall**: 84-89%
- **推理速度**: 35-50 FPS (RTX 3060)
- **模型大小**: ~22MB

### 系统要求
- **GPU**: NVIDIA GPU with 6GB+ VRAM (推荐)
- **CPU**: 支持CPU训练（速度较慢）
- **内存**: 8GB+
- **存储**: 10GB+

## 扩展功能

### 1. 多类别裂缝分类

修改`data/crack-seg/data.yaml`：
```yaml
nc: 4
names:
  0: horizontal  # 横向裂缝
  1: vertical    # 纵向裂缝
  2: network     # 网状裂缝
  3: diagonal    # 斜向裂缝
```

### 2. 模型加速

导出为ONNX格式：
```python
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='onnx')
```

### 3. 报告生成

```python
from utils.report_generator import generate_detection_report
from utils.database import CrackDatabase

db = CrackDatabase()
report_path = generate_detection_report(detection_id=1, db=db)
```

## 常见问题

### Q: 训练时显存不足？
A: 减小batch size，修改`configs/train_config.yaml`中的`batch`参数。

### Q: 如何使用CPU训练？
A: 训练脚本会自动检测，或在配置文件中设置`device: cpu`。

### Q: 如何提高检测精度？
A: 
1. 增加训练轮数
2. 使用更大的模型（YOLOv8m）
3. 调整数据增强参数
4. 增加训练数据

### Q: 实时检测FPS太低？
A: 
1. 使用更小的模型（YOLOv8n）
2. 降低输入图像分辨率
3. 使用GPU加速
4. 导出为ONNX/TensorRT格式

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue。

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
