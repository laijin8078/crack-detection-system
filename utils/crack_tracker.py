"""
轻量级裂缝跟踪器 (Crack Tracker)
用于视频连续帧中为同一条裂缝分配稳定的 track_id

匹配策略：bbox IoU + mask IoU + 中心点距离 + 方向角差异的加权打分 + 贪心匹配
"""

import numpy as np
from collections import OrderedDict


def bbox_iou(box1, box2):
    """
    计算两个 bbox 的 IoU

    Args:
        box1, box2: [x1, y1, x2, y2]

    Returns:
        iou: float
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def mask_iou(mask1, mask2):
    """
    计算两个二值 mask 的 IoU

    mask1/mask2 应为同尺寸的 np.ndarray (uint8 或 bool)

    Returns:
        iou: float
    """
    if mask1 is None or mask2 is None:
        return 0.0
    if mask1.shape != mask2.shape:
        return 0.0
    inter = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return float(inter / union) if union > 0 else 0.0


def center_distance(c1, c2):
    """计算两个中心点的欧氏距离"""
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def angle_difference(a1, a2):
    """
    计算两个角度（度）之间的最小差值 [0, 90]
    裂缝方向具有 180° 周期性，所以取 min(|diff|, 180 - |diff|)
    """
    diff = abs(a1 - a2) % 180.0
    return min(diff, 180.0 - diff)


def _greedy_match(cost_matrix, cost_threshold=0.7):
    """
    贪心匹配：按最小代价逐个分配，不重复匹配
    返回 (row_indices, col_indices) 即 (det_idx, track_idx)

    Args:
        cost_matrix: (M, N) 代价矩阵，M=detections, N=tracks，值越小越匹配
        cost_threshold: 代价超过此值则不匹配

    Returns:
        matches: list of (det_idx, track_idx)
        unmatched_dets: list of det_idx
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])) if cost_matrix.ndim == 2 else []

    M, N = cost_matrix.shape
    # 创建 (代价, i, j) 三元组并排序
    candidates = []
    for i in range(M):
        for j in range(N):
            if cost_matrix[i, j] <= cost_threshold:
                candidates.append((cost_matrix[i, j], i, j))
    candidates.sort(key=lambda x: x[0])

    matched_dets = set()
    matched_tracks = set()
    matches = []

    for cost, i, j in candidates:
        if i not in matched_dets and j not in matched_tracks:
            matches.append((i, j))
            matched_dets.add(i)
            matched_tracks.add(j)

    unmatched_dets = [i for i in range(M) if i not in matched_dets]
    return matches, unmatched_dets


class CrackTracker:
    """
    裂缝视频跟踪器

    为连续帧中的裂缝分配稳定的 track_id，统计唯一裂缝数量。
    """

    def __init__(self, config=None):
        """
        Args:
            config: dict，包含跟踪参数：
                - max_lost_frames: int
                - min_track_frames: int
                - bbox_iou_weight: float
                - center_dist_weight: float
                - orientation_weight: float
                - bbox_iou_threshold: float
                - center_distance_threshold: float
                - angle_threshold: float
                - image_diag: float 图像对角线长度（用于归一化距离）
        """
        cfg = config or {}
        self.max_lost_frames = cfg.get('max_lost_frames', 10)
        self.min_track_frames = cfg.get('min_track_frames', 3)
        self.w_iou = cfg.get('bbox_iou_weight', 0.3)
        self.w_mask = cfg.get('mask_iou_weight', 0.2)
        self.w_dist = cfg.get('center_dist_weight', 0.3)
        self.w_angle = cfg.get('orientation_weight', 0.2)
        self.iou_gate = cfg.get('bbox_iou_threshold', 0.2)
        self.mask_iou_gate = cfg.get('mask_iou_threshold', 0.1)
        self.dist_gate = cfg.get('center_distance_threshold', 80)
        self.angle_gate = cfg.get('angle_threshold', 30.0)
        self.image_diag = cfg.get('image_diag', 1000.0)

        self.next_id = 1
        self.tracks = OrderedDict()           # track_id -> track_info (当前活跃 + 丢失中)
        self.completed_tracks = []            # 已删除但满足 min_track_frames 的轨迹
        self.frame_idx = 0
        self.all_time_detection_count = 0     # 所有帧的累计检测次数（含被过滤的短轨迹）
        self.active_ids = set()

    def _match_cost(self, crack, track):
        """
        计算一条检测与一个跟踪轨迹的匹配代价 (0~1, 越小越好)

        匹配维度：mask IoU + bbox IoU + 中心点距离 + 方向角差异
        门控：如果任一单项指标极差，直接返回代价 1.0（不可匹配）
        """
        last = track['last_crack']
        box1, box2 = crack['bbox_xyxy'], last['bbox_xyxy']
        c1, c2 = crack['center_xy'], last['center_xy']
        a1, a2 = crack['orientation_angle'], last['orientation_angle']

        # ---- 门控检查 ----
        iou = bbox_iou(box1, box2)
        if iou < self.iou_gate:
            return 1.0

        dist = center_distance(c1, c2)
        norm_dist = dist / self.image_diag
        if dist > self.dist_gate:
            return 1.0

        a_diff = angle_difference(a1, a2)
        if a_diff > self.angle_gate:
            return 1.0

        # ---- mask IoU（降采样 mask 加速计算） ----
        m_iou = mask_iou(crack.get('_mask_ds'), last.get('_mask_ds'))
        has_mask = (crack.get('_mask_ds') is not None and last.get('_mask_ds') is not None)
        if has_mask and m_iou < self.mask_iou_gate:
            return 1.0

        # ---- 加权代价 ----
        iou_cost = 1.0 - iou
        mask_cost = 1.0 - m_iou if has_mask else 0.0
        dist_cost = min(norm_dist / (self.dist_gate / self.image_diag), 1.0)
        angle_cost = a_diff / self.angle_gate

        # 无 mask 时重新分配权重
        if has_mask:
            cost = (self.w_iou * iou_cost +
                    self.w_mask * mask_cost +
                    self.w_dist * dist_cost +
                    self.w_angle * angle_cost)
        else:
            w_sum = self.w_iou + self.w_dist + self.w_angle
            cost = ((self.w_iou / w_sum) * iou_cost +
                    (self.w_dist / w_sum) * dist_cost +
                    (self.w_angle / w_sum) * angle_cost)
        return cost

    def update(self, cracks, frame_idx=None):
        """
        用当前帧的检测结果更新跟踪状态

        Args:
            cracks: list of crack feature dicts（来自 crack_postprocess.extract_crack_features）
            frame_idx: 当前帧索引

        Returns:
            frame_assignments: dict，{det_idx: track_id}
        """
        if frame_idx is not None:
            self.frame_idx = frame_idx
        else:
            self.frame_idx += 1

        # 清除上一帧的 active 标记
        self.active_ids.clear()

        self.all_time_detection_count += len(cracks)

        if not cracks:
            # 没有检测到裂缝，所有 track 丢失一帧
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['lost_frames'] += 1
                if self.tracks[tid]['lost_frames'] > self.max_lost_frames:
                    self._archive_track(tid)
            return {}

        M = len(cracks)
        active_tracks = [(tid, t) for tid, t in self.tracks.items() if t['lost_frames'] == 0]
        N = len(active_tracks)

        frame_assignments = {}

        if N > 0:
            # 构建代价矩阵
            cost_matrix = np.zeros((M, N))
            for i, crack in enumerate(cracks):
                for j, (tid, track) in enumerate(active_tracks):
                    cost_matrix[i, j] = self._match_cost(crack, track)

            # 贪心匹配
            matches, unmatched_dets = _greedy_match(cost_matrix, cost_threshold=0.7)

            for i, j in matches:
                tid = active_tracks[j][0]
                self._update_track(tid, cracks[i])
                frame_assignments[i] = tid
        else:
            unmatched_dets = list(range(M))

        # 为未匹配的检测创建新 track
        for i in unmatched_dets:
            tid = self._create_track(cracks[i])
            frame_assignments[i] = tid

        # 更新丢失帧计数（活跃 track 中本次未匹配到的）
        matched_tids = set(frame_assignments.values())
        for tid in list(self.tracks.keys()):
            if tid not in matched_tids and self.tracks[tid]['lost_frames'] == 0:
                self.tracks[tid]['lost_frames'] += 1
                if self.tracks[tid]['lost_frames'] > self.max_lost_frames:
                    self._archive_track(tid)

        return frame_assignments

    def _create_track(self, crack):
        """创建新的跟踪轨迹"""
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = {
            'track_id': tid,
            'first_frame': self.frame_idx,
            'last_frame': self.frame_idx,
            'frame_count': 1,
            'lost_frames': 0,
            'last_crack': crack,
            'crack_history': [(self.frame_idx, crack)],
            'total_confidence': crack['confidence'],
        }
        self.active_ids.add(tid)
        return tid

    def _update_track(self, tid, crack):
        """更新已有的跟踪轨迹"""
        track = self.tracks[tid]
        track['last_frame'] = self.frame_idx
        track['frame_count'] += 1
        track['lost_frames'] = 0
        track['last_crack'] = crack
        track['crack_history'].append((self.frame_idx, crack))
        track['total_confidence'] += crack['confidence']
        self.active_ids.add(tid)

    def get_valid_tracks(self):
        """
        获取所有有效跟踪轨迹（出现帧数 >= min_track_frames），包含已完成的和活跃的

        Returns:
            list of track dicts
        """
        valid = []
        # 活跃/丢失中的 track
        for tid, track in self.tracks.items():
            if track['frame_count'] >= self.min_track_frames:
                valid.append(self._format_track(track))
        # 已归档的 completed track
        for track in self.completed_tracks:
            if track['frame_count'] >= self.min_track_frames:
                valid.append(self._format_track(track))
        return valid

    def unique_track_count(self):
        """返回唯一裂缝数量（所有历史有效 track 数，包括已完成的）"""
        return len(self.get_valid_tracks())

    def total_detection_count(self):
        """返回原始检测总数（所有帧所有检测之和，含短轨迹的被过滤帧）"""
        return self.all_time_detection_count

    def summary(self):
        """返回跟踪摘要信息"""
        valid = self.get_valid_tracks()
        valid_detections = sum(t['frame_count'] for t in valid)
        return {
            'total_detections': self.all_time_detection_count,  # 全部帧的检测总数
            'unique_tracks': len(valid),
            'valid_track_detections': valid_detections,          # 有效 track 对应的检测数
            'short_track_detections': self.all_time_detection_count - valid_detections,  # 短轨迹检测数
            'total_frames': self.frame_idx + 1,
            'tracks': valid,
        }

    def _archive_track(self, tid):
        """将 track 从活跃列表移到归档列表（保留已完成的有效 track）"""
        track = self.tracks.pop(tid)
        self.completed_tracks.append(dict(track))

    def _format_track(self, track):
        """格式化 track 信息用于报告"""
        return {
            'track_id': track['track_id'],
            'first_frame': track['first_frame'],
            'last_frame': track['last_frame'],
            'frame_count': track['frame_count'],
            'avg_confidence': round(track['total_confidence'] / track['frame_count'], 4),
            'crack_history': track['crack_history'],
        }
