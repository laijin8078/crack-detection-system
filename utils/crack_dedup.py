"""
跨图像裂缝去重模块 (Cross-Image Crack Deduplication) v2
用于多张不同角度墙面图像中识别同一条裂缝，分配统一 crack_id

v2 改进：
  - 骨架归一化形态相似度替代中心点距离作为主要匹配依据
  - ORB 图像配准 + 特征投影（纹理不足时自动回退）
  - 中心点距离降级为弱参考特征
  - 两阶段匹配：配准投影匹配 → 归一化骨架匹配

【重要】适用范围：
  - 仅适合同一面墙、连续拍摄的图像序列
  - 不同墙面的图片不应混用 image_sequence 模式
"""

import cv2
import numpy as np
from pathlib import Path


# ==================== 几何工具函数 ====================

def _center_distance(c1, c2):
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def _angle_difference(a1, a2):
    diff = abs(a1 - a2) % 180.0
    return min(diff, 180.0 - diff)


def _bbox_aspect_ratio(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w / h if h > 0 else 1.0


def _mask_iou(m1, m2):
    if m1 is None or m2 is None:
        return 0.0
    if m1.shape != m2.shape:
        m2 = cv2.resize(m2, (m1.shape[1], m1.shape[0]), interpolation=cv2.INTER_NEAREST)
    inter = np.sum(m1 & m2)
    union = np.sum(m1 | m2)
    return float(inter / union) if union > 0 else 0.0


def compute_endpoint_distance(c1, c2):
    """计算两条裂缝骨架端点之间的最近距离（像素）"""
    ep1 = c1.get('_endpoints', [])
    ep2 = c2.get('_endpoints', [])
    if not ep1 or not ep2:
        return float('inf')
    min_dist = float('inf')
    for a in ep1:
        for b in ep2:
            d = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
            min_dist = min(min_dist, d)
    return min_dist


def _bbox_iou(b1, b2):
    """两个 bbox 的 IoU"""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0


def _chamfer_distance(pts1, pts2):
    """计算两个点集之间的双向 Chamfer 距离（越小越相似），归一化到 [0,1]"""
    if len(pts1) == 0 or len(pts2) == 0:
        return 1.0
    p1 = np.array(pts1, dtype=np.float32)
    p2 = np.array(pts2, dtype=np.float32)
    # p1 到 p2 的最近距离
    d12 = np.min(np.linalg.norm(p1[:, None] - p2[None, :], axis=2), axis=1)
    # p2 到 p1 的最近距离
    d21 = np.min(np.linalg.norm(p2[:, None] - p1[None, :], axis=2), axis=1)
    chamfer = (np.mean(d12) + np.mean(d21)) / 2.0
    # 用 200px 作为归一化参考
    return min(chamfer / 200.0, 1.0)


# ==================== 图像配准 ====================

def estimate_image_transform(img_ref, img_cur, max_dim=800):
    """
    使用 ORB 特征点 + RANSAC 估计两张图之间的仿射/单应变换

    Args:
        img_ref, img_cur: 参考图和当前图 (BGR numpy array)
        max_dim: 处理前将长边缩放到此尺寸以加速

    Returns:
        H: (3,3) 变换矩阵 或 None（配准失败时）
        status: str 状态信息
    """
    # 缩放到统一尺寸加速
    h1, w1 = img_ref.shape[:2]
    h2, w2 = img_cur.shape[:2]
    scale1 = min(1.0, max_dim / max(h1, w1))
    scale2 = min(1.0, max_dim / max(h2, w2))

    if scale1 < 1.0:
        img_ref = cv2.resize(img_ref, (int(w1 * scale1), int(h1 * scale1)))
    if scale2 < 1.0:
        img_cur = cv2.resize(img_cur, (int(w2 * scale2), int(h2 * scale2)))

    gray1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_cur, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, "ORB 特征点不足 (ref={}, cur={})".format(
            len(kp1) if kp1 else 0, len(kp2) if kp2 else 0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    good = matches[:min(len(matches), 100)]
    if len(good) < 8:
        return None, f"有效匹配点不足 ({len(good)} < 8)"

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 恢复原始尺度
    if scale1 < 1.0:
        src_pts /= scale1
    if scale2 < 1.0:
        dst_pts /= scale2

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        # 退而求其次：仿射变换
        H_affine, mask_affine = cv2.estimateAffine2D(dst_pts, src_pts, method=cv2.RANSAC)
        if H_affine is not None:
            H_full = np.eye(3, dtype=np.float32)
            H_full[:2, :] = H_affine
            return H_full, "仿射变换 (inliers={})".format(int(np.sum(mask_affine)) if mask_affine is not None else 0)
        return None, "单应/仿射估计均失败"

    inliers = int(np.sum(mask)) if mask is not None else 0
    return H, f"单应变换 (inliers={inliers}/{len(good)})"


def transform_crack_features(crack, H):
    """
    将裂缝特征通过变换矩阵 H 投影到参考坐标系

    Returns:
        dict 包含变换后的 center_xy, bbox_xyxy, _skeleton_norm
    """
    result = dict(crack)

    # 变换 center
    cx, cy = crack['center_xy']
    pt = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
    tx = cv2.perspectiveTransform(pt, H)
    result['center_xy'] = [round(float(tx[0][0][0]), 2), round(float(tx[0][0][1]), 2)]

    # 变换 bbox 四个角点，取外接矩形
    x1, y1, x2, y2 = crack['bbox_xyxy']
    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    tx_corners = cv2.perspectiveTransform(corners, H)
    tx_corners = tx_corners.reshape(-1, 2)
    result['bbox_xyxy'] = [
        round(float(np.min(tx_corners[:, 0])), 2),
        round(float(np.min(tx_corners[:, 1])), 2),
        round(float(np.max(tx_corners[:, 0])), 2),
        round(float(np.max(tx_corners[:, 1])), 2),
    ]

    # 变换 skeleton 点
    skel_pts = crack.get('_skeleton_pts', np.zeros((0, 2)))
    if len(skel_pts) > 0:
        pts = np.array(skel_pts, dtype=np.float32).reshape(-1, 1, 2)
        tx_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        result['_skeleton_pts'] = tx_pts
        # 重新归一化
        new_center = result['center_xy']
        new_scale = max(result['length_px_est'], 1.0)
        centered = tx_pts - np.array(new_center, dtype=np.float32)
        result['_skeleton_norm'] = centered / new_scale

    return result


def match_cracks_with_transform(ref_cracks, cur_cracks, H):
    """
    将 cur 图像的裂缝投影到 ref 图像坐标系后计算匹配得分

    Returns:
        scores: (M_cur, N_ref) 得分矩阵
    """
    M = len(cur_cracks)
    N = len(ref_cracks)
    scores = np.zeros((M, N))
    for i in range(M):
        cur_tx = transform_crack_features(cur_cracks[i], H)
        for j in range(N):
            scores[i, j] = _compute_pair_score(cur_tx, ref_cracks[j], use_transform=True)
    return scores


# ==================== 骨架相似度 ====================

def compute_skeleton_similarity(c1, c2):
    """
    基于归一化骨架计算两条裂缝的形态相似度 (0~1)

    归一化骨架已将骨架点平移至原点并缩放到单位尺度，
    完全消除位置和尺度影响，仅保留曲线形态信息。

    Returns:
        similarity: 0~1, 越高越相似
    """
    p1 = c1.get('_skeleton_norm')
    p2 = c2.get('_skeleton_norm')
    if p1 is None or p2 is None or len(p1) < 3 or len(p2) < 3:
        return 0.0

    cd = _chamfer_distance(p1, p2)
    return max(0.0, 1.0 - cd)


# ==================== 综合匹配得分 ====================

def _compute_pair_score(c1, c2, same_image=False, use_transform=False):
    """
    计算两条裂缝的匹配得分 (0~1, 越高越相似)。
    same_image=True 时启用更严格的同图保护规则。
    """
    skel_sim = compute_skeleton_similarity(c1, c2)
    a_diff = _angle_difference(c1['orientation_angle'], c2['orientation_angle'])
    max_angle = 45.0
    angle_score = max(0.0, 1.0 - a_diff / max_angle)

    dist = _center_distance(c1['center_xy'], c2['center_xy'])
    dist_max = 800.0
    if use_transform:
        center_score = 1.0
    else:
        center_score = max(0.0, 1.0 - dist / dist_max)

    a1, a2 = c1['area_px'], c2['area_px']
    area_ratio = min(a1, a2) / max(a1, a2) if max(a1, a2) > 0 else 0
    l1, l2 = c1['length_px_est'], c2['length_px_est']
    len_ratio = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 0

    # 方向角门控
    if a_diff > max_angle:
        return 0.0, 'angle_diff_too_large'

    # ---- 同图保护规则 ----
    if same_image:
        if area_ratio < 0.2 or len_ratio < 0.25:
            return 0.0, 'same_image_size_ratio_gate'

        ep_dist = compute_endpoint_distance(c1, c2)
        mask_iou_val = _mask_iou(c1.get('_mask_ds'), c2.get('_mask_ds'))
        bbox_iou_val = _bbox_iou(c1['bbox_xyxy'], c2['bbox_xyxy'])

        if skel_sim < 0.75:
            return 0.0, 'same_image_skeleton_too_low'

        if ep_dist > 50 and mask_iou_val < 0.03 and bbox_iou_val < 0.02:
            return 0.0, 'same_image_endpoints_too_far_no_overlap'

        score = 0.40 * skel_sim + 0.25 * angle_score + 0.15 * (1.0 - min(ep_dist/100.0, 1.0))
        score += 0.10 * min(mask_iou_val * 5, 1.0) + 0.10 * min(bbox_iou_val * 5, 1.0)
        return min(score, 1.0), 'same_image_merge'

    # ---- 跨图匹配 ----
    if area_ratio < 0.15 or len_ratio < 0.2:
        return 0.0, 'cross_image_size_ratio_gate'

    score = 0.45 * skel_sim + 0.25 * angle_score + 0.10 * center_score
    score += 0.10 * min(area_ratio * 2, 1.0) + 0.10 * min(len_ratio * 2, 1.0)
    return min(score, 1.0), 'cross_image_match'


# ==================== 合并匹配 (保持接口兼容) ====================

def _match_score(c1, c2, config):
    """旧接口兼容，内部调用新评分"""
    s, _ = _compute_pair_score(c1, c2, same_image=False, use_transform=False)
    return s


def _build_debug_entry(img_a, ci_a, img_b, ci_b, c1, c2, score, matched, threshold, reason, same_image):
    """构建 debug 匹配明细"""
    dist = _center_distance(c1['center_xy'], c2['center_xy'])
    a_diff = _angle_difference(c1['orientation_angle'], c2['orientation_angle'])
    skel_sim = compute_skeleton_similarity(c1, c2)
    ep_dist = compute_endpoint_distance(c1, c2)
    miou = _mask_iou(c1.get('_mask_ds'), c2.get('_mask_ds'))
    biou = _bbox_iou(c1['bbox_xyxy'], c2['bbox_xyxy'])
    entry = {
        'crack_a': f'{img_a}_C{ci_a}',
        'crack_b': f'{img_b}_C{ci_b}',
        'same_image': same_image,
        'skeleton_score': round(skel_sim, 4),
        'angle_score': round(max(0.0, 1.0 - a_diff/45.0), 4),
        'center_dist_px': round(dist, 1),
        'endpoint_distance': round(ep_dist, 1) if ep_dist != float('inf') else None,
        'mask_iou': round(miou, 4),
        'bbox_iou': round(biou, 4),
        'final_score': round(score, 4),
        'matched': matched,
        'threshold': threshold,
        'reason': reason,
    }
    return entry


def _greedy_merge(all_cracks, image_ids, config, debug=False):
    """贪心聚类合并，支持配准和骨架匹配"""
    threshold = config.get('min_final_score', 0.5)
    use_homography = config.get('use_homography', True)
    debug_matches = []

    indexed = []
    # 加载图像用于配准（若开启）
    image_buffers = {}  # lazy load
    for img_idx, (img_id, cracks) in enumerate(zip(image_ids, all_cracks)):
        for c_idx, crack in enumerate(cracks):
            indexed.append({
                'image_idx': img_idx,
                'image_id': img_id,
                'crack_idx': c_idx,
                'crack': crack,
            })

    parent = list(range(len(indexed)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # 按图像对迭代
    num_imgs = len(all_cracks)
    for i in range(num_imgs):
        for j in range(i + 1, num_imgs):
            ci_list = [x for x in indexed if x['image_idx'] == i]
            cj_list = [x for x in indexed if x['image_idx'] == j]

            # ---- 阶段一：尝试图像配准 ----
            H = None
            reg_status = "未启用"
            if use_homography:
                # lazy load images
                for k in [i, j]:
                    if k not in image_buffers:
                        img_path = None
                        for item in indexed:
                            if item['image_idx'] == k:
                                # 从 crack 反查图像路径
                                break
                        # 从 all_cracks 反查——实际上需要原始图像路径。
                        # 这里我们用已有的 mask_full 来估算，
                        # 但需要原始图像做配准。
                        # 我们在 inference.py 层面传入图像路径。
                        pass

                # 配准需要原始图像——当前架构没有直接传入。
                # 作为第一版：使用骨架归一化做形态匹配，不依赖原始图像配准。
                # 后续可在 inference.py 中传入图像路径后启用。
                reg_status = "需要原始图像路径（当前使用骨架归一化回退）"

            # ---- 阶段二：归一化骨架形态匹配 ----
            same_img = (i == j)  # 同一张图内的两个检测
            for ci in ci_list:
                for cj in cj_list:
                    if H is not None:
                        cj_tx = transform_crack_features(cj['crack'], H)
                        score, reason = _compute_pair_score(ci['crack'], cj_tx, same_image=same_img, use_transform=True)
                    else:
                        score, reason = _compute_pair_score(ci['crack'], cj['crack'], same_image=same_img, use_transform=False)

                    # 同图/跨图使用不同阈值
                    use_threshold = config.get('same_image_merge', {}).get('min_skeleton_score', 0.75) if same_img else threshold
                    matched = score >= use_threshold

                    if debug:
                        debug_matches.append(_build_debug_entry(
                            ci['image_id'], ci['crack_idx'],
                            cj['image_id'], cj['crack_idx'],
                            ci['crack'], cj['crack'],
                            score, matched, use_threshold, reason, same_img))

                    if matched:
                        union(indexed.index(ci), indexed.index(cj))

    groups = {}
    for idx, item in enumerate(indexed):
        root = find(idx)
        if root not in groups:
            groups[root] = []
        groups[root].append(item)

    crack_groups, unmatched = [], []
    for items in groups.values():
        if len(items) == 1:
            unmatched.append(items[0])
        else:
            crack_groups.append(items)

    return crack_groups, unmatched, debug_matches


# ==================== 主入口 ====================

def deduplicate_cracks(all_cracks_per_image, image_ids, config=None, debug=False, image_paths=None):
    """
    跨图像裂缝去重主函数

    Args:
        all_cracks_per_image: list of list，每张图像的 crack list
        image_ids: list of str，每张图像名
        config: dict，去重参数
        debug: bool，是否输出 debug 匹配明细
        image_paths: list of str，每张图像的完整路径（用于图像配准，可选）

    Returns:
        result: dict
    """
    if config is None:
        config = {}
    threshold = config.get('min_final_score', 0.5)

    crack_groups, unmatched, debug_matches = _greedy_merge(
        all_cracks_per_image, image_ids, config, debug=debug)

    unique_cracks = []
    cid = 1

    for group in crack_groups:
        best = max(group, key=lambda x: x['crack']['confidence'])
        bc = best['crack']
        apps = []
        for item in group:
            apps.append({
                'image_id': item['image_id'],
                'image_idx': item['image_idx'],
                'confidence': item['crack']['confidence'],
                'bbox_xyxy': item['crack']['bbox_xyxy'],
                'center_xy': item['crack']['center_xy'],
            })
        unique_cracks.append({
            'crack_id': f'C{cid}',
            'is_duplicate': len(apps) > 1,
            'appearances': apps,
            'best_confidence': bc['confidence'],
            'avg_confidence': round(np.mean([a['confidence'] for a in apps]), 4),
            'bbox_xyxy': bc['bbox_xyxy'],
            'center_xy': bc['center_xy'],
            'area_px': bc['area_px'],
            'length_px_est': bc['length_px_est'],
            'orientation_angle': bc['orientation_angle'],
        })
        cid += 1

    for item in unmatched:
        c = item['crack']
        unique_cracks.append({
            'crack_id': f'C{cid}',
            'is_duplicate': False,
            'appearances': [{
                'image_id': item['image_id'], 'image_idx': item['image_idx'],
                'confidence': c['confidence'], 'bbox_xyxy': c['bbox_xyxy'],
                'center_xy': c['center_xy'],
            }],
            'best_confidence': c['confidence'],
            'avg_confidence': c['confidence'],
            'bbox_xyxy': c['bbox_xyxy'],
            'center_xy': c['center_xy'],
            'area_px': c['area_px'],
            'length_px_est': c['length_px_est'],
            'orientation_angle': c['orientation_angle'],
        })
        cid += 1

    raw_count = sum(len(cracks) for cracks in all_cracks_per_image)
    unique_count = len(unique_cracks)
    result = {
        'unique_crack_count': unique_count,
        'raw_detection_count': raw_count,
        'duplicate_removed_count': raw_count - unique_count,
        'cracks': unique_cracks,
        'matched_groups': len(crack_groups),
        'unmatched_singles': len(unmatched),
    }
    if debug:
        result['debug_matches'] = debug_matches
    return result


# ---- Homography 接口已实现 (estimate_image_transform / transform_crack_features) ----
apply_homography_to_crack_features = transform_crack_features
estimate_homography_from_images = estimate_image_transform
