# 建筑裂缝检测深度学习系统

基于 YOLOv8s-seg 的建筑表面裂缝检测系统，支持裂缝实例分割、视频跟踪去重、跨图去重、DeepSeek AI 维修建议生成。

## 功能特性

- **裂缝检测**: YOLOv8s-seg 实例分割，输出 bbox + mask + 骨架 + 几何特征
- **视频跟踪**: 轻量级 tracker，为连续帧中的同一条裂缝分配稳定 track_id
- **跨图去重**: 基于骨架形态相似度的跨图像裂缝去重，区分同图/跨图匹配规则
- **AI 维修建议**: 接入 DeepSeek API，根据检测结果自动生成结构化修补方案
- **PDF 报告**: 一键生成排版精美的中英文维修建议 PDF 报告
- **Web 服务**: FastAPI 后端，支持图像上传检测
- **实时检测**: 支持 USB 摄像头、网络摄像头、视频文件

## 系统架构

```
墙面图像/视频 → YOLOv8s-seg 检测 → 后处理(骨架/跟踪/去重) → 检测 JSON
                                                              ↓
                                            DeepSeek API → 维修建议 JSON + PDF
```

## 项目结构

```
模型/
├── data/                    # 数据集
│   └── crack-seg/           # 训练/验证/测试集 (YOLO 分割格式)
├── configs/                 # 配置文件
│   ├── train_config.yaml    # 训练参数
│   ├── augmentation_config.yaml  # 数据增强
│   └── inference_config.yaml     # 推理/跟踪/去重参数
├── utils/                   # 核心模块
│   ├── crack_postprocess.py # Mask 二值化、骨架提取、几何特征
│   ├── crack_tracker.py     # 视频连续帧裂缝跟踪 (轻量级)
│   ├── crack_dedup.py       # 跨图骨架形态去重 (v2)
│   ├── crack_report.py      # 结构化 JSON 报告输出
│   ├── deepseek_advisor.py  # DeepSeek API 维修建议生成
│   ├── advice_pdf.py        # PDF 维修报告生成
│   ├── database.py          # SQLite 检测记录存储
│   └── report_generator.py  # PDF 检测报告生成
├── docs/                    # 文档
│   ├── crack_dedup_tracking_usage.md  # 去重/跟踪使用说明
│   ├── dedup_optimization_notes.md    # 去重优化技术说明
│   ├── deepseek_advice_usage.md       # DeepSeek 接入说明
│   └── end_to_end_workflow.md         # 端到端流程
├── tests/                   # 测试
│   └── test_crack_dedup_regression.py # 去重回归测试
├── test_cases/              # 测试用例
│   ├── same_crack_sequence_001/       # 正例：同裂缝合并
│   └── different_parallel_cracks_001/ # 反例：不同裂缝防误合并
├── outputs/                 # 输出
│   ├── reports/             # 检测 JSON 报告
│   ├── advice/              # DeepSeek 建议 JSON + PDF
│   ├── predictions/         # 标注图片
│   └── runs/                # 训练记录
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
├── inference.py             # 推理脚本（三种模式）
├── realtime_detect.py       # 实时检测（摄像头/视频）
├── generate_advice.py       # DeepSeek 建议生成 CLI
├── app.py                   # FastAPI Web 服务
└── requirements.txt         # 依赖
```

## 快速开始

### 1. 安装

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install -r requirements.txt
```

### 2. 模型训练

```bash
python train.py
# 模型保存在 outputs/runs/crack_detection/weights/best.pt
```

### 3. 推理检测

```bash
# 单图检测
python inference.py --source image.jpg --mode image

# 视频跟踪（统计唯一裂缝数）
python inference.py --source video.mp4 --mode video

# 同一墙面多图去重（推荐）
python inference.py --source ./wall_photos/ --mode image_sequence --wall-id 3F东墙
```

### 4. 一键检测 + AI 建议 + PDF

```bash
# 配置 DeepSeek API Key
set DEEPSEEK_API_KEY=sk-your-key-here

# 一条命令完成全部
python inference.py --source ./wall_photos/ --mode image_sequence \
  --wall-id 3F东墙 --advice --advice-pdf --save-images
```

输出文件:

```
outputs/reports/dedup_3F东墙.json    # 检测报告
outputs/advice/advice_dedup_3F东墙.json  # AI 建议
outputs/advice/advice_dedup_3F东墙.pdf   # PDF 报告
outputs/predictions/inference/*.jpg      # 标注图片
```

### 5. 实时检测

```bash
# USB 摄像头（自动启用裂缝跟踪）
python realtime_detect.py --source 0

# 视频文件
python realtime_detect.py --source video.mp4 --save-video
```

### 6. Web 服务

```bash
uvicorn app:app --reload
# 访问 http://localhost:8000/docs 查看 API 文档
```

## 三种推理模式

| 模式 | 场景 | 去重方式 | 关键输出 |
|------|------|----------|----------|
| `image` | 单张/批量图像 | 不做去重，单图统计 | `crack_id` |
| `video` | 视频/摄像头连续帧 | 帧间 tracker，分配 `track_id` | `track_id`, `unique_crack_count` |
| `image_sequence` | 同一墙面多角度图像 | 骨架形态 + 几何特征跨图去重 | `wall_id`, `is_duplicate`, `matched_with` |

## 输出 JSON 核心字段

```json
{
  "source_type": "image_sequence",
  "wall_id": "3F东墙",
  "summary": {
    "raw_detection_count": 45,
    "unique_crack_count": 12,
    "duplicate_removed_count": 33
  },
  "cracks": [{
    "crack_id": "C1",
    "confidence": 0.82,
    "bbox_xyxy": [120, 200, 380, 260],
    "center_xy": [250, 230],
    "area_px": 3614,
    "length_px_est": 314.0,
    "orientation_angle": 90.0,
    "is_duplicate": true,
    "matched_with": ["img_001.jpg", "img_003.jpg"]
  }],
  "limitations": ["缺少尺度标定，无法得到真实毫米级尺寸"]
}
```

## DeepSeek AI 维修建议

配置 Key，检测完成自动生成：

```bash
export DEEPSEEK_API_KEY="sk-your-key"
python inference.py --source ./wall/ --mode image_sequence --wall-id test --advice --advice-pdf
```

DeepSeek 输出: 检测概况、风险等级、可能原因、分步修补方案、建议材料、是否需人工复核。

详见 [`docs/deepseek_advice_usage.md`](docs/deepseek_advice_usage.md)

## 回归测试

```bash
python tests/test_crack_dedup_regression.py
# 预期: 2 PASS, 0 FAIL
```

## 跨图去重技术说明

v2 版本用骨架形态相似度替代中心点距离硬门控，解决了同一裂缝在不同角度拍摄时因位置偏移大而被误判为不同裂缝的问题。同图内通过端点距离和 mask 重叠防止不同裂缝误合并。

详见 [`docs/dedup_optimization_notes.md`](docs/dedup_optimization_notes.md)

## 性能指标 (YOLOv8s-seg)

- mAP@0.5: 88-91%
- 推理速度: 35-50 FPS (RTX 3060)
- 模型大小: ~22MB

## 常见问题

**Q: 训练时显存不足？** 减小 `configs/train_config.yaml` 中的 `batch` 参数。

**Q: 不同墙面图片放在一起导致过度合并？** 不同墙面应分目录存放，并分别指定 `--wall-id`。`image_sequence` 模式仅适合同一墙面。

**Q: 未检测到 DEEPSEEK_API_KEY？** 运行前需配置环境变量: `set DEEPSEEK_API_KEY=sk-xxx` (Windows) 或 `export DEEPSEEK_API_KEY=sk-xxx` (Linux/Mac)。

**Q: 实时检测 FPS 太低？** 使用更小的模型 (YOLOv8n-seg)，降低输入分辨率，或导出 ONNX/TensorRT。

## 当前局限

1. 裂缝尺寸为像素值，真实物理尺寸需尺度标定
2. 跨图去重依赖 mask 质量，强透视变化建议后续接入 Homography
3. 不同墙面图像必须通过 `wall_id` 分开，不能混入同一 `image_sequence`
4. AI 建议不判断结构安全，仅供参考

## 许可证

MIT License
