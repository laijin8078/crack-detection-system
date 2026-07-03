# 裂缝去重与跟踪使用说明

## 功能概述

基于 yolov8n-seg-cracks-joints.pt 的墙面裂缝检测系统，支持三种推理模式解决裂缝重复识别问题：

| 模式 | 场景 | 去重方式 | 输出关键字段 |
|------|------|----------|-------------|
| `image` | 单张/批量图像 | 不做去重，单图统计 | `crack_id` (C1, C2...) |
| `video` | 视频/摄像头连续帧 | 帧间 tracker 跟踪，分配 `track_id` | `track_id`, `unique_crack_count` |
| `image_sequence` | 同一墙面多角度图像 | 跨图几何特征匹配去重，分配统一 `crack_id` | `wall_id`, `is_duplicate`, `matched_with` |

---

## 运行命令

### 1. 单图推理（image 模式）

```bash
# 单张图像
python inference.py --source image.jpg --mode image

# 多张图像（各自独立，不跨图去重）
python inference.py --source ./wall_photos/ --mode image

# 无裂缝图像
python inference.py --source clean_wall.jpg --mode image
```

### 2. 视频跟踪（video 模式）

```bash
# 视频文件，启用裂缝跟踪
python inference.py --source inspection.mp4 --mode video

# 保存带 track_id 标注的输出视频
python inference.py --source inspection.mp4 --mode video --save-video

# 实时摄像头（自动启用跟踪）
python realtime_detect.py --source 0
```

视频模式下，按 `r` 键可随时输出当前跟踪报告 JSON。

### 3. 跨图去重（image_sequence 模式）

```bash
# 同一墙面多张图像，显式指定 wall_id
python inference.py --source ./wall_A/images/ --mode image_sequence --wall-id wall_A

# 未指定 wall_id 时，默认使用输入目录名
python inference.py --source ./wall_001/ --mode image_sequence
# → wall_id 自动设为 "wall_001"
```

---

## 输出 JSON 字段说明

### 顶层字段

| 字段 | 说明 |
|------|------|
| `image_or_video_id` | 输入标识（文件名/目录名） |
| `source_type` | `"image"` / `"video"` / `"image_sequence"` |
| `model_name` | 固定 `"yolov8n-seg-cracks-joints"` |
| `wall_id` | 墙面标识（仅 `image_sequence` 模式） |
| `timestamp` | ISO 时间戳 |

### summary 字段

| 字段 | image 模式 | video 模式 | image_sequence 模式 |
|------|-----------|-----------|-------------------|
| `raw_detection_count` | 单图检测数 | 所有帧检测总和 | 所有图像检测总和 |
| `unique_crack_count` | 同 raw（不去重） | 唯一 track_id 数量 | 跨图去重后唯一裂缝数 |
| `duplicate_removed_count` | 0 | raw - unique | raw - unique |
| `overall_confidence` | 平均置信度 | 平均置信度 | 平均置信度 |

### cracks[] 每个裂缝的字段

| 字段 | 说明 |
|------|------|
| `crack_id` | 唯一裂缝 ID（`"C1"`, `"C2"` ...） |
| `track_id` | 视频跟踪 ID（仅 video 模式） |
| `frame_index` | 帧索引（仅 video 模式） |
| `confidence` | 置信度 |
| `bbox_xyxy` | 边界框 `[x1, y1, x2, y2]` |
| `center_xy` | 中心点 `[cx, cy]` |
| `area_px` | mask 像素面积 |
| `length_px_est` | 估计长度（像素） |
| `orientation_angle` | 方向角（度，范围 0-180） |
| `is_duplicate` | 是否为跨图重复（仅 image_sequence） |
| `matched_with` | 匹配到的其他图像列表（仅 image_sequence） |
| `first_frame` / `last_frame` / `frame_count` | 轨迹帧范围（仅 video） |

### limitations 字段

说明当前版本的已知局限（详见下方）。

---

## 当前版本局限

1. **跨图去重仅适合同一墙面、连续拍摄、有明显重叠区域的图像序列**。不同墙面必须分目录存放或指定不同 `wall_id`，否则会产生错误合并。
2. **未指定 `--wall-id` 时，默认使用输入目录名作为 wall_id**。请确保不同墙面图像存放在不同目录。
3. **无尺度标定**，`length_px_est` 和 `area_px` 为像素值，无法直接换算为真实物理尺寸。
4. **视频跟踪的 `min_track_frames=3`** 会过滤仅出现 1-2 帧的短时裂缝检测，短时出现的目标可能被忽略。
5. **跨图去重纯几何匹配**，未接入 Homography 图像配准，透视变形较大的墙面图像去重精度受限。
6. **贪心匹配非全局最优**，密集裂缝场景可能有次优分配。

---

## 后续改进方向

- 接入 Homography 图像配准，将不同视角图像变换到统一坐标后做 mask IoU 匹配
- 引入 ByteTrack / BoT-SORT 等更强的多目标跟踪器
- 接入 DeepSeek 等 LLM 根据结构化 JSON 结果生成修补建议报告
- 增加尺度标定功能，输出裂缝真实物理长度和宽度

### 跨图去重优化 (v2)

2026-05 完成了跨图去重核心优化，详见 [`docs/dedup_optimization_notes.md`](dedup_optimization_notes.md)：骨架形态相似度替代中心点距离；区分同图/跨图规则。


## 回归测试

```bash
python tests/test_crack_dedup_regression.py
# 预期: 2 PASS, 0 FAIL
```

## Debug 模式

```bash
python inference.py --source test_cases/same_crack_sequence_001/ \
  --mode image_sequence --wall-id test --debug-dedup
```

输出每对裂缝的 skeleton_score、angle_score、endpoint_distance、mask_iou、bbox_iou 和 matched/rejected reason。

---

## 建议汇报表述

> 当前已实现重复识别问题的第一版工程闭环：
>
> - **视频场景**：通过连续帧 tracker 分配稳定 `track_id`，以唯一 track 数量替代每帧检测数量，避免同一条裂缝在多帧中被重复统计；
> - **多图场景**：通过 `wall_id` 分组和几何特征匹配（中心点、方向角、面积比、bbox 宽高比）进行跨图去重，为疑似同一裂缝分配统一 `crack_id`；
> - **结果输出**：以结构化 JSON 格式输出，包含 `raw_detection_count`、`unique_crack_count`、`duplicate_removed_count` 及每条裂缝的详细几何特征，可直接作为后续 DeepSeek 等 LLM 的输入，生成修补建议报告。

---

## 接入 DeepSeek 的数据入口

如果需要将检测结果接入 DeepSeek 生成修补建议，应读取以下 JSON 文件作为输入：

```
outputs/reports/dedup_<wall_id>.json     # 跨图去重结果
outputs/reports/video_track_<name>.json  # 视频跟踪结果
outputs/reports/image_summary_*.json     # 单图汇总结果
```

每个 JSON 中 `cracks[]` 数组包含每条唯一裂缝的完整几何特征，可直接序列化为 prompt 传给 LLM。
