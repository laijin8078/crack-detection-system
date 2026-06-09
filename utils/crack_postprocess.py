"""
裂缝实例特征提取模块
从 YOLOv8-seg 推理结果中提取每条裂缝的几何与形态特征
"""

import cv2
import numpy as np
import torch


def polygon_to_mask(polygon, image_shape):
    """
    将多边形轮廓转为二值 mask

    Args:
        polygon: (N, 2) 多边形坐标数组
        image_shape: (H, W) 原始图像尺寸

    Returns:
        mask: (H, W) 二值 mask
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if polygon is None or len(polygon) < 3:
        return mask
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_orientation(mask):
    """
    使用 PCA / minAreaRect 计算裂缝主方向角

    对 mask 轮廓点拟合最小外接旋转矩形，取长边方向作为裂缝方向。

    Args:
        mask: (H, W) 二值 mask

    Returns:
        angle: 方向角（度），范围 [0, 180)，0 表示水平向右
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    # 取最大轮廓
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return 0.0
    rect = cv2.minAreaRect(largest)
    # rect[2] 是 OpenCV 返回的角度：水平边与矩形长边夹角，范围 [-90, 0)
    angle = rect[2]
    # 标准化到 [0, 180)
    if angle < 0:
        angle += 180.0
    if angle >= 180.0:
        angle -= 180.0
    return angle


def estimate_length(mask):
    """
    估算裂缝长度（像素）

    取 mask 轮廓的最小外接矩形长边长度作为长度估计。

    Args:
        mask: (H, W) 二值 mask

    Returns:
        length_px: 长度估计值（像素）
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return 0.0
    _, (w, h), _ = cv2.minAreaRect(largest)
    return max(w, h)


def compute_area(mask):
    """计算 mask 像素面积"""
    return int(np.sum(mask > 0))


def _zhang_suen_thinning(binary):
    """
    Zhang-Suen 骨架细化算法（纯 numpy/opencv，不依赖 skimage）

    Args:
        binary: (H, W) 二值图像，前景=1, 背景=0

    Returns:
        skeleton: (H, W) 细化后的二值图像
    """
    skel = binary.copy().astype(np.uint8)
    while True:
        # 子迭代1：删除东南边界点
        marker = np.zeros_like(skel)
        rows, cols = np.where(skel == 1)
        for r, c in zip(rows, cols):
            if r == 0 or r == skel.shape[0]-1 or c == 0 or c == skel.shape[1]-1:
                continue
            p2 = skel[r-1, c]
            p3 = skel[r-1, c+1]
            p4 = skel[r, c+1]
            p5 = skel[r+1, c+1]
            p6 = skel[r+1, c]
            p7 = skel[r+1, c-1]
            p8 = skel[r, c-1]
            p9 = skel[r-1, c-1]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            n_nonzero = sum(neighbors)
            transitions = sum(1 for i in range(8) if neighbors[i] == 0 and neighbors[(i+1)%8] == 1)
            if 2 <= n_nonzero <= 6 and transitions == 1:
                if p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
                    marker[r, c] = 1
        skel[marker == 1] = 0
        if marker.sum() == 0:
            break

        # 子迭代2：删除西北边界点
        marker = np.zeros_like(skel)
        rows, cols = np.where(skel == 1)
        for r, c in zip(rows, cols):
            if r == 0 or r == skel.shape[0]-1 or c == 0 or c == skel.shape[1]-1:
                continue
            p2 = skel[r-1, c]
            p3 = skel[r-1, c+1]
            p4 = skel[r, c+1]
            p5 = skel[r+1, c+1]
            p6 = skel[r+1, c]
            p7 = skel[r+1, c-1]
            p8 = skel[r, c-1]
            p9 = skel[r-1, c-1]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            n_nonzero = sum(neighbors)
            transitions = sum(1 for i in range(8) if neighbors[i] == 0 and neighbors[(i+1)%8] == 1)
            if 2 <= n_nonzero <= 6 and transitions == 1:
                if p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
                    marker[r, c] = 1
        skel[marker == 1] = 0
        if marker.sum() == 0:
            break
    return skel


def extract_skeleton(mask):
    """
    从二值 mask 中提取裂缝骨架（中轴）

    使用 Zhang-Suen 细化算法，返回骨架二值图和骨架点集。

    Args:
        mask: (H, W) 二值 mask

    Returns:
        skel_mask: (H, W) 骨架二值图 (255=骨架, 0=背景)
        skel_pts: (N, 2) 骨架点坐标列表
    """
    if mask is None or np.sum(mask) == 0:
        return np.zeros_like(mask, dtype=np.uint8), np.zeros((0, 2))

    binary = (mask > 0).astype(np.uint8)
    skel = _zhang_suen_thinning(binary)
    skel_mask = skel.astype(np.uint8) * 255
    pts = np.column_stack(np.where(skel > 0))
    pts = pts[:, ::-1]  # (row,col) -> (x,y)
    return skel_mask, pts


def skeleton_endpoints(skel_mask):
    """
    从骨架图中检测端点（邻域内只有1个骨架像素的端点）

    Args:
        skel_mask: (H, W) 骨架二值图

    Returns:
        endpoints: list of (x, y) 或空列表
    """
    binary = (skel_mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(binary, -1, kernel) - binary
    # 端点：自身为1，8邻域中恰好有1个邻居
    endpoint_mask = (binary == 1) & (neighbors == 1)
    pts = np.column_stack(np.where(endpoint_mask))
    if len(pts) == 0:
        return []
    pts = pts[:, ::-1]
    return pts.tolist()


def normalize_skeleton_pts(skel_pts, center, scale):
    """
    将骨架点归一化：平移到原点、缩放到单位尺度

    Args:
        skel_pts: (N, 2) 骨架点坐标
        center: (cx, cy) 中心点
        scale: float 缩放因子（通常取 bbox 对角线长度或 mask 主轴长度）

    Returns:
        norm_pts: (N, 2) 归一化后的骨架点
    """
    if len(skel_pts) == 0 or scale <= 0:
        return np.zeros((0, 2))
    pts = np.array(skel_pts, dtype=np.float32)
    centered = pts - np.array(center, dtype=np.float32)
    return centered / scale


def compute_mask_centroid(mask):
    """
    计算 mask 质心坐标

    Returns:
        (cx, cy) 或 None
    """
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return (cx, cy)


def filter_results_by_class(results, target_class_ids):
    """
    从 YOLO 推理结果中过滤掉不需要的类别，原地修改 results 列表中的 boxes 和 masks。

    必须在 results[0].plot() 之前调用，否则标注图中会包含被过滤的类别。

    Args:
        results: ultralytics Results 对象列表
        target_class_ids: 要保留的类别 ID 列表，如 [0]

    Returns:
        results: 原地修改后的 results（同时返回以方便链式调用）
    """
    if target_class_ids is None or results is None:
        return results
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        if not hasattr(r.boxes, 'cls') or r.boxes.cls is None:
            continue
        # 构建布尔掩码：保留 target_class_ids 中的类别
        keep = torch.zeros(len(r.boxes.cls), dtype=torch.bool, device=r.boxes.cls.device)
        for cid in target_class_ids:
            keep = keep | (r.boxes.cls == cid)
        if keep.sum() < len(r.boxes):
            r.boxes = r.boxes[keep]
            if r.masks is not None and r.masks.data is not None:
                r.masks.data = r.masks.data[keep]
    return results


def extract_crack_features(results, image_shape, min_area_px=50, mask_downsample_ratio=4,
                          target_class_ids=None):
    """
    从 YOLOv8-seg 推理结果中提取每条裂缝的特征

    Args:
        results: ultralytics Results 对象（单张图推理结果）
        image_shape: (H, W) 原始图像尺寸
        min_area_px: 最小面积过滤阈值
        mask_downsample_ratio: mask 降采样比例
        target_class_ids: 要保留的类别 ID 列表，None 表示保留全部。
                          例如 [0] 表示只保留裂缝，过滤掉接缝等其他类别。

    Returns:
        cracks: list of dict，每条裂缝包含：
            - bbox_xyxy: [x1, y1, x2, y2]
            - center_xy: [cx, cy]  (mask 质心)
            - confidence: float
            - area_px: int
            - length_px_est: float
            - orientation_angle: float (度)
            - mask_polygon: list 或 None
            - mask_binary: np.ndarray (可选，用于去重/跟踪)
    """
    cracks = []
    if results is None:
        return cracks

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes
        masks = r.masks
        H, W = image_shape[:2]

        # 计算降采样尺寸
        ds_H, ds_W = H // mask_downsample_ratio, W // mask_downsample_ratio

        for i, box in enumerate(boxes):
            # ---- 类别过滤：只保留目标类别 ----
            if target_class_ids is not None:
                cls_id = int(box.cls[0]) if hasattr(box, 'cls') and box.cls is not None else 0
                if cls_id not in target_class_ids:
                    continue
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            # 从多边形或 box 生成 mask
            mask_polygon = None
            mask_full = None

            if masks is not None and i < len(masks):
                try:
                    mask_xy = masks[i].xy
                    if mask_xy and len(mask_xy) > 0 and len(mask_xy[0]) >= 3:
                        mask_polygon = mask_xy[0].tolist()
                        mask_full = polygon_to_mask(mask_polygon, (H, W))
                except Exception:
                    pass

            # 如果没有有效 mask polygon，用 bbox 作为近似 mask
            if mask_full is None or np.sum(mask_full) == 0:
                x1, y1, x2, y2 = map(int, xyxy)
                mask_full = np.zeros((H, W), dtype=np.uint8)
                mask_full[y1:y2, x1:x2] = 1

            # 降采样 mask（后续计算全部在降采样 mask 上运行，大幅提速）
            mask_ds = cv2.resize(mask_full, (ds_W, ds_H), interpolation=cv2.INTER_NEAREST)
            ds_area = compute_area(mask_ds)

            # 面积过滤（降采样后面积 = 原面积 / ratio²）
            area = ds_area * (mask_downsample_ratio ** 2)
            if area < min_area_px:
                continue

            # ---- 以下计算均在降采样 mask 上执行，然后还原到原始尺度 ----
            ratio = mask_downsample_ratio

            # 质心（降采样）
            centroid_ds = compute_mask_centroid(mask_ds)
            center_xy = [centroid_ds[0] * ratio, centroid_ds[1] * ratio] if centroid_ds else [
                float(xyxy[0] + xyxy[2]) / 2,
                float(xyxy[1] + xyxy[3]) / 2
            ]

            # 方向角（角度与尺度无关，无需还原）
            orientation = compute_orientation(mask_ds)

            # 长度估计（降采样，需还原）
            length_ds = estimate_length(mask_ds)
            length_est = length_ds * ratio

            # 骨架提取（在降采样 mask 上执行，大幅提速）
            skel_mask_ds, skel_pts_ds = extract_skeleton(mask_ds)

            # 骨架点还原到原始尺度
            if len(skel_pts_ds) > 0:
                skel_pts = (skel_pts_ds.astype(np.float32) * ratio).astype(np.int32)
            else:
                skel_pts = np.zeros((0, 2), dtype=np.int32)

            # 骨架二值图还原（用于端点检测和可视化）
            skel_mask = cv2.resize(skel_mask_ds, (W, H), interpolation=cv2.INTER_NEAREST)

            # 归一化骨架（基于降采样点 + 原始尺度 center/length）
            scale = max(length_est, 1.0)
            skel_norm = normalize_skeleton_pts(skel_pts, center_xy, scale)
            endpoints = skeleton_endpoints(skel_mask)

            crack = {
                'bbox_xyxy': [round(float(v), 2) for v in xyxy],
                'center_xy': [round(float(v), 2) for v in center_xy],
                'confidence': round(conf, 4),
                'area_px': area,
                'length_px_est': round(length_est, 2),
                'orientation_angle': round(orientation, 2),
                'mask_polygon': mask_polygon,
                '_mask_ds': mask_ds,           # 内部使用
                '_mask_full': mask_full,       # 内部使用
                '_skeleton_mask': skel_mask,   # 骨架二值图
                '_skeleton_pts': skel_pts,     # (N,2) 骨架点
                '_skeleton_norm': skel_norm,   # 归一化骨架点
                '_endpoints': endpoints,       # 骨架端点
            }
            cracks.append(crack)

    return cracks
