# 跨图去重优化说明

## 原始问题

同一墙面裂缝在多角度、多距离、不同裁剪范围拍摄时，被 YOLO 检测后误判为不同裂缝，导致 `unique_crack_count` 偏高，裂缝总数统计不准确。

### 根本原因

旧版本 `_match_score()` 使用 `center_distance_threshold=100px` 作为**硬门控**。中心点距离超过 100px 直接返回 score=0，拒绝匹配。

实测数据：同一裂缝在三张图中中心点分别偏移 256px、412px、230px，全部被硬门控拒绝。

```
img_01 C0 <-> img_02 C0: dist=256 > 100 → REJECTED
img_01 C0 <-> img_03 C0: dist=412 > 100 → REJECTED
img_02 C0 <-> img_03 C0: dist=230 > 100 → REJECTED
raw=4, unique=4, removed=0 ← 完全没去重
```

## 优化方法

### 核心思路

从"中心点位置匹配"转向"裂缝形态匹配"。同一条裂缝无论从哪个角度拍，其骨架曲线形态（曲率、走向、长宽比）基本不变。

### 具体改动

| 改动 | 说明 |
|------|------|
| Zhang-Suen 骨架细化 | 从 mask 中提取裂缝中轴骨架（纯 numpy，零依赖） |
| 骨架归一化 | 平移到原点 + 缩放到单位尺度，消除位置和尺度差异 |
| Chamfer 距离 | 双向最近点距离衡量骨架形态相似度，替代中心点距离 |
| 中心点降级 | 从硬门控改为弱权重特征（0.10），允许大范围偏移 |
| 同图保护规则 | 同一图像内合并需端点距离<50px 或 mask/bbox 有重叠 |
| 跨图宽松匹配 | 跨图像匹配允许位置大偏移，主要依据骨架形态+方向角 |

### 匹配策略对比

| 维度 | 旧版 | 新版(跨图) | 新版(同图) |
|------|------|-----------|-----------|
| 核心特征 | 中心点距离 | 骨架形态 | 骨架+端点距离 |
| 中心点 | 硬门控 100px | 弱权重 0.10 | 不参与 |
| 方向角 | 硬门控 30° | 门控 45° | 门控 45° |
| 面积比 | 硬门控 0.3~3.0 | 门控 0.15 | 门控 0.2 |
| 端点距离 | 无 | 无 | 必须<50px 或有重叠 |

## 测试结果

### 成功样例：same_crack_sequence_001

同一裂缝在三张不同角度/距离的照片中：

```
Before:  raw=4, unique=4, removed=0  (完全没去重)
After:   raw=4, unique=1, removed=3  (正确合并为1条)
```

### 反例样例：different_parallel_cracks_001

img_B 包含两条不同但方向、形态相似的裂缝：

```
C0 <-> C1 (不同裂缝): score=0.0000, REJECTED
  reason: same_image_endpoints_too_far_no_overlap
  skel_sim=0.9997, ep_dist=733px, mask_iou=0, bbox_iou=0

C1 <-> C3 (同一裂缝的两次检测): score=0.9621, ACCEPTED
  reason: same_image_merge
  skel_sim=0.9999, ep_dist=0px, mask_iou=0.80, bbox_iou=0.62

raw=7, unique=2, removed=5
```

关键观察：即使骨架相似度高达 0.9997，只要端点距离>50px 且无 mask/bbox 重叠，同图匹配就会被拒绝。

## 回归测试

```bash
python tests/test_crack_dedup_regression.py
```

输出示例：

```
same_crack_sequence_001                  PASS | raw=4 unique=1 removed=3 | expected_unique=1
different_parallel_cracks_001            PASS | raw=7 unique=2 removed=5 | expected_min_unique>=2
总计: 2 PASS, 0 FAIL
```

## Debug 模式

```bash
python inference.py --source test_cases/same_crack_sequence_001/ \
  --mode image_sequence --wall-id test --debug-dedup
```

debug 输出每条匹配对的详细信息：骨架得分、角度分、端点距离、mask IoU、bbox IoU、匹配原因。

## 当前局限

1. **骨架质量依赖 mask**：YOLO mask 边缘粗糙时骨架形态会失真
2. **不同墙面需 wall_id 分离**：骨架形态相似的不同墙面裂缝仍可能跨图误合并
3. **强透视变化**：Homography 配准可进一步提高精度（接口已预留，未启用）
4. **端点检测不稳定**：骨架细化对 mask 边缘敏感，同一条裂缝在不同图中端点可能偏移
