# DeepSeek 裂缝修补建议生成使用说明

## 功能概述

基于已有的裂缝检测 JSON 报告，调用 DeepSeek API 自动生成结构化墙面裂缝修补建议，包括：

- 检测概况与风险等级评估
- 裂缝可能原因分析
- 分步修补方案
- 建议材料清单
- 是否需要人工复核及原因

## 前提条件

1. 已生成裂缝检测 JSON 报告（通过 `inference.py` 生成）
2. 已配置 DeepSeek API Key

## 配置 DeepSeek API Key

```bash
# Linux / Mac
export DEEPSEEK_API_KEY="sk-your-key-here"

# Windows (CMD)
set DEEPSEEK_API_KEY=sk-your-key-here

# Windows (PowerShell)
$env:DEEPSEEK_API_KEY="sk-your-key-here"
```

如果未配置环境变量，运行 `generate_advice.py` 时会提示：

```
错误: 未检测到 DEEPSEEK_API_KEY，请先配置环境变量后再运行。
```

## 运行命令

### 基本用法

```bash
# 为跨图去重报告生成建议
python generate_advice.py --report outputs/reports/dedup_wall_A1.json

# 为视频跟踪报告生成建议
python generate_advice.py --report outputs/reports/video_track_inspection.json

# 为单图汇总报告生成建议
python generate_advice.py --report outputs/reports/image_summary_20260526_141546.json
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--report` | **必填** 裂缝检测 JSON 报告路径 | - |
| `--output-dir` | 维修建议输出目录 | `outputs/advice` |
| `--dry-run` | 仅构建 prompt 不调用 API，用于调试 | false |
| `--model` | DeepSeek 模型名称 | `deepseek-chat` |

### 调试模式

```bash
# 不实际调用 API，仅查看 prompt 构建结果
python generate_advice.py --report outputs/reports/dedup_wall_A1.json --dry-run
```

### 完整工作流示例

```bash
# 第1步：运行 image_sequence 推理 + 跨图去重
python inference.py \
  --source ./wall_3F_east/ \
  --mode image_sequence \
  --wall-id wall_3F_east

# 第2步：基于去重报告调用 DeepSeek 生成修补建议
python generate_advice.py \
  --report outputs/reports/dedup_wall_3F_east.json
```

## 输入 JSON 格式

`generate_advice.py` 自动兼容两种报告格式：

### 格式1：单报告（video / image_sequence / 单图 image）

```json
{
  "source_type": "image_sequence",
  "wall_id": "wall_A1",
  "summary": {
    "raw_detection_count": 45,
    "unique_crack_count": 12,
    "duplicate_removed_count": 33
  },
  "cracks": [
    {
      "crack_id": "C1",
      "confidence": 0.82,
      "area_px": 3614,
      "length_px_est": 314.0,
      "orientation_angle": 90.0,
      "is_duplicate": true,
      "matched_with": ["img_001.jpg", "img_003.jpg"]
    }
  ],
  "limitations": [...]
}
```

### 格式2：批量汇总（image 模式 batch 输出）

```json
{
  "source": "data/images/",
  "mode": "image",
  "per_image": [
    {
      "source_type": "image",
      "summary": {"raw_detection_count": 3, ...},
      "cracks": [...],
      "limitations": [...]
    }
  ]
}
```

## 输出 JSON 格式

```json
{
  "source_report": "outputs/reports/dedup_wall_A1.json",
  "generated_at": "2026-05-26T15:30:00",
  "model_used": "deepseek-chat",
  "overall_assessment": "本次检测发现...",
  "risk_level": "中等",
  "detected_summary": {
    "raw_detection_count": 45,
    "unique_crack_count": 12,
    "duplicate_removed_count": 33
  },
  "possible_causes": [
    "温度变化导致材料热胀冷缩",
    "地基不均匀沉降"
  ],
  "repair_plan": [
    {
      "step": 1,
      "title": "裂缝表面清理",
      "description": "使用钢丝刷清除裂缝表面松散物..."
    }
  ],
  "materials": [
    "环氧树脂注浆料",
    "聚合物水泥砂浆"
  ],
  "need_manual_review": true,
  "manual_review_reason": "裂缝数量较多，建议专业人员现场评估",
  "limitations": [
    "缺少尺度标定，无法得到真实毫米级尺寸"
  ]
}
```

## 当前局限

1. **DeepSeek API 依赖网络**：需要外网访问 `api.deepseek.com`
2. **DeepSeek 响应格式**：虽然 prompt 要求输出纯 JSON，但 LLM 输出可能包含额外文本，模块已内置容错解析（尝试直接解析 → 提取 ```json``` 块 → 提取 `{ }` 块 → 兜底）
3. **裂缝物理尺寸**：检测 JSON 中的 `length_px_est` 和 `area_px` 是像素值，DeepSeek 不会将其误认为真实物理尺寸（已在 prompt 中明确约束）
4. **结构安全判断**：DeepSeek 不会输出建筑结构是否安全的结论（已在 prompt 中明确约束）
5. **模型限制**：默认使用 `deepseek-chat`，可根据需要切换为其他 DeepSeek 模型

## 后续扩展

### 接入 FastAPI 后端接口

如果需要通过 Web API 调用 DeepSeek 建议生成，可以：

1. 在 `app.py` 新增路由：
```python
@app.post("/api/generate-advice")
async def generate_advice_api(report_path: str):
    from utils.deepseek_advisor import generate_advice
    advice, path = generate_advice(report_path)
    return {"success": True, "advice": advice, "saved_to": path}
```

2. 前端上传报告后自动调用 `/api/generate-advice` 获取修补建议并展示。

3. 注意事项：
   - API 调用可能耗时较长（DeepSeek 响应通常 5-30 秒），建议使用异步任务队列（如 Celery）
   - 需要在前端展示 `need_manual_review` 和 `risk_level` 等关键决策字段
   - `limitations` 字段应始终展示，提醒用户结果存在局限
