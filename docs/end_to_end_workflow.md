# 端到端工作流程说明

## 整体流程

```
墙面图像/视频 → YOLOv8s-seg 检测 → 后处理(跟踪/去重) → 检测 JSON → DeepSeek → 维修建议 JSON
```

* 训练代码：`train.py`（不需要在推理流程中运行）
* 检测推理：`inference.py`（三种模式）
* 建议生成：`generate_advice.py`

---

## 命令速查

### 场景 A：单图检测 + 建议生成

```bash
# 1. 检测
python inference.py --source wall.jpg --mode image

# 2. 生成建议
python generate_advice.py --report outputs/reports/image_summary_*.json
```

### 场景 B：图像序列跨图去重 + 建议生成

```bash
# 1. 检测 + 去重（同一墙面）
python inference.py \
  --source ./wall_A_photos/ \
  --mode image_sequence \
  --wall-id wall_A

# 2. 生成建议
python generate_advice.py \
  --report outputs/reports/dedup_wall_A.json
```

### 场景 C：视频跟踪 + 建议生成

```bash
# 1. 检测 + 跟踪
python inference.py --source inspection.mp4 --mode video

# 2. 生成建议
python generate_advice.py \
  --report outputs/reports/video_track_inspection.json
```

### 场景 D：调试验证（不调用 DeepSeek API）

```bash
# 仅验证 prompt 构建，不实际调用 API
python generate_advice.py --report outputs/reports/dedup_wall_A.json --dry-run
```

---

## 输出文件路径说明

| 步骤 | 输出路径 | 内容 |
|------|----------|------|
| 检测(image) | `outputs/reports/image_summary_<timestamp>.json` | 单图/批量检测汇总 |
| 检测(video) | `outputs/reports/video_track_<name>.json` | 视频跟踪报告 |
| 检测(image_sequence) | `outputs/reports/dedup_<dir_name>.json` | 跨图去重报告 |
| 建议生成 | `outputs/advice/advice_<report_stem>.json` | DeepSeek 维修建议 |

---

## 端到端验证记录

以下为实际运行的验证结果（2026-05-26）：

```bash
# 输入：5 张墙面测试图像（同一墙面，2 次重复 = 10 张）
$ python inference.py --source /tmp/test_wall --mode image_sequence --wall-id wall_e2e_test

# 检测结果
raw_detection_count:    28
unique_crack_count:      9
duplicate_removed_count: 19

# 检测 JSON
outputs/reports/dedup_test_wall.json

# 建议 JSON (dry-run)
outputs/advice/advice_dedup_test_wall.json
```

### 验证通过的检查项

- [x] 检测 JSON 包含 `source_type`、`wall_id`、`summary`、`cracks`、`limitations`
- [x] `summary.raw_detection_count`、`unique_crack_count`、`duplicate_removed_count` 正确
- [x] 建议 JSON 包含全部 12 个必需字段
- [x] `detected_summary` 与检测报告的 `summary` 一致
- [x] system prompt 包含：不虚构宽度/深度、不判断结构安全、尺度标定提醒、建议人工复核
- [x] user prompt 包含完整检测数据与像素单位转换警告
- [x] 缺失 DEEPSEEK_API_KEY 时给出清晰错误提示
- [x] 报告文件不存在时给出清晰错误提示
- [x] `--dry-run` 模式不调用 API 但完整构建 prompt

---

## 常见错误

| 错误现象 | 原因 | 解决方法 |
|----------|------|----------|
| `未检测到 DEEPSEEK_API_KEY` | 未配置环境变量 | `export DEEPSEEK_API_KEY=sk-xxx` |
| `报告文件不存在` | `--report` 路径错误 | 检查 JSON 文件名，用 `ls outputs/reports/` 确认 |
| DeepSeek 返回 HTTP 4xx | API Key 无效或过期 | 检查 Key 是否正确，是否有余额 |
| DeepSeek 返回 HTTP 5xx | DeepSeek 服务端故障 | 稍后重试 |
| LLM 输出解析失败 | DeepSeek 返回了非 JSON 格式 | 模块已内置三层容错，通常能恢复；检查原始响应 |
| `UnicodeDecodeError` | Windows GBK 编码问题 | 确保 JSON 文件以 UTF-8 编码保存 |

---

## 组会汇报可用说明

> 当前系统已实现从墙面图像输入到 DeepSeek 维修建议输出的完整端到端流程。
>
> **检测阶段**：使用 YOLOv8s-seg 对墙面图像进行裂缝实例分割，在视频场景下通过轻量级 tracker 分配稳定 track_id 统计唯一裂缝数，在多图场景下通过 wall_id 分组和几何特征匹配进行跨图去重。
>
> **建议阶段**：将结构化检测 JSON 输入 DeepSeek，由 LLM 根据裂缝数量、长度、面积、方向角、跨图重复情况等信息，生成包含风险等级、可能原因、分步修补方案、建议材料、是否需要人工复核的结构化维修建议。
>
> **关键约束**：LLM prompt 中明确禁止虚构真实物理尺寸和不判断结构安全——所有物理尺寸均为像素值，需后续尺度标定才能换算为毫米级数据。
>
> 整个过程通过两个命令即可完成：`python inference.py --mode image_sequence --wall-id <id>` 生成检测 JSON → `python generate_advice.py --report <path>` 生成维修建议 JSON。

---

## 当前系统不能解决的问题

1. **真实物理尺寸** — YOLO 检测输出为像素值，无法换算为毫米级裂缝宽度和深度，需要尺度标定（如已知参照物尺寸或激光测距）
2. **结构安全性判断** — 系统不会直接判断建筑是否安全，仅提供风险等级参考和人工复核建议
3. **裂缝深度测量** — 2D 图像无法测量裂缝深度，需要 3D 扫描或超声波等检测手段
4. **环境因素关联** — 系统不支持分析温度、湿度、荷载等环境因素与裂缝的因果关系
5. **实时长期监测** — 当前不支持裂缝的长期变化监测（需要时序对比和变化检测功能）
