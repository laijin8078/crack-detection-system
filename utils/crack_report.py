"""
结构化裂缝检测报告生成模块
按统一 JSON schema 输出检测、跟踪、去重结果
"""

import json
from datetime import datetime
from pathlib import Path
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """自定义 JSON encoder，处理 numpy 类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def build_standard_report(
    source_id,
    source_type,
    model_name,
    raw_detection_count,
    unique_crack_count,
    duplicate_removed_count,
    cracks_detail,
    extra_info=None,
    wall_id=None,
    extra_limitations=None,
):
    """
    构建符合统一 schema 的结构化报告

    Args:
        source_id: 图像/视频标识符
        source_type: 'image' | 'video' | 'image_sequence'
        model_name: 模型名称
        raw_detection_count: 原始检测数（所有帧/图合计）
        unique_crack_count: 去重后唯一裂缝数
        duplicate_removed_count: 去除的重复数
        cracks_detail: list of dict，每条裂缝的详细信息
        extra_info: dict，额外信息（帧数、图像数等）

    Returns:
        report: dict
    """
    overall_conf = 0.0
    if cracks_detail:
        confs = [c.get('confidence', c.get('avg_confidence', 0)) for c in cracks_detail]
        overall_conf = round(sum(confs) / len(confs), 4)

    report = {
        'image_or_video_id': source_id,
        'source_type': source_type,
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'raw_detection_count': raw_detection_count,
            'unique_crack_count': unique_crack_count,
            'duplicate_removed_count': duplicate_removed_count,
            'overall_confidence': overall_conf,
        },
        'cracks': cracks_detail,
        'limitations': [
            '当前去重结果依赖图像质量和检测 mask 精度',
            '跨视角严格去重建议后续加入 Homography 图像配准',
            '裂缝真实长度和宽度需要尺度标定',
            '跨图去重当前仅适合拍摄角度变化较小的场景：不同墙面的图片不应混用 image_sequence 模式',
            '视频跟踪的 min_track_frames 参数会过滤只出现 1~2 帧的检测，短时出现的目标可能被忽略',
        ],
    }

    if wall_id:
        report['wall_id'] = wall_id
    elif source_type == 'image_sequence':
        report['wall_id'] = None

    if extra_limitations:
        report['limitations'].extend(extra_limitations)

    if extra_info:
        report['extra_info'] = extra_info

    return report


def build_crack_entry(
    crack_id,
    confidence,
    bbox_xyxy,
    center_xy,
    area_px,
    length_px_est,
    orientation_angle,
    track_id=None,
    frame_index=None,
    is_duplicate=False,
    matched_with=None,
):
    """
    构建单条裂缝的结构化条目

    Args:
        crack_id: 唯一裂缝 ID (如 'C1', 'C2')
        confidence: 置信度
        bbox_xyxy: [x1, y1, x2, y2]
        center_xy: [cx, cy]
        area_px: mask 像素面积
        length_px_est: 估计长度（像素）
        orientation_angle: 方向角（度）
        track_id: 视频跟踪 ID（可选）
        frame_index: 帧索引（可选）
        is_duplicate: 是否为跨图重复
        matched_with: 匹配到的其他裂缝 ID

    Returns:
        entry: dict
    """
    return {
        'crack_id': crack_id,
        'track_id': track_id,
        'frame_index': frame_index,
        'confidence': confidence,
        'bbox_xyxy': bbox_xyxy,
        'center_xy': center_xy,
        'area_px': area_px,
        'length_px_est': length_px_est,
        'orientation_angle': orientation_angle,
        'is_duplicate': is_duplicate,
        'matched_with': matched_with,
    }


def build_tracking_report(source_id, tracker_summary, raw_detection_count=None, model_name='yolov8n-seg-cracks-joints'):
    """
    从 tracker 摘要构建视频跟踪报告

    Args:
        source_id: 视频文件名
        tracker_summary: CrackTracker.summary() 返回值
        raw_detection_count: 覆盖 summary 中的 total_detections（因为 summary 只含 valid tracks）
        model_name: 模型名称

    Returns:
        report: dict
    """
    raw_count = raw_detection_count if raw_detection_count is not None else tracker_summary['total_detections']
    cracks_detail = []
    for track in tracker_summary['tracks']:
        # 取最近一帧的特征作为代表
        last_frame, last_crack = track['crack_history'][-1]
        entry = build_crack_entry(
            crack_id=f'C{track["track_id"]}',
            track_id=track['track_id'],
            frame_index=last_frame,
            confidence=track['avg_confidence'],
            bbox_xyxy=last_crack['bbox_xyxy'],
            center_xy=last_crack['center_xy'],
            area_px=last_crack['area_px'],
            length_px_est=last_crack['length_px_est'],
            orientation_angle=last_crack['orientation_angle'],
            is_duplicate=False,
        )
        # 附加帧范围信息
        entry['first_frame'] = track['first_frame']
        entry['last_frame'] = track['last_frame']
        entry['frame_count'] = track['frame_count']
        cracks_detail.append(entry)

    return build_standard_report(
        source_id=source_id,
        source_type='video',
        model_name=model_name,
        raw_detection_count=raw_count,
        unique_crack_count=tracker_summary['unique_tracks'],
        duplicate_removed_count=max(0, raw_count - tracker_summary['unique_tracks']),
        cracks_detail=cracks_detail,
        extra_info={
            'total_frames': tracker_summary['total_frames'],
            'valid_track_detections': tracker_summary.get('valid_track_detections', raw_count),
            'short_track_detections': tracker_summary.get('short_track_detections', 0),
        },
    )


def build_dedup_report(source_id, dedup_result, wall_id=None, model_name='yolov8n-seg-cracks-joints'):
    """
    从去重结果构建跨图去重报告

    Args:
        source_id: 图像目录或批次标识
        dedup_result: deduplicate_cracks() 返回值
        wall_id: 墙面标识符（仅 image_sequence 模式），不同 wall_id 之间不进行去重
        model_name: 模型名称

    Returns:
        report: dict
    """
    extra_limitations = []
    if wall_id is None:
        extra_limitations.append(
            '未显式指定 wall_id，默认使用输入目录名；不同墙面图像不应混合输入同一 image_sequence'
        )
    cracks_detail = []
    for crack in dedup_result['cracks']:
        matched_with = None
        if crack['is_duplicate']:
            matched_with = [a['image_id'] for a in crack['appearances']]

        entry = build_crack_entry(
            crack_id=crack['crack_id'],
            track_id=None,
            frame_index=None,
            confidence=crack['avg_confidence'],
            bbox_xyxy=crack['bbox_xyxy'],
            center_xy=crack['center_xy'],
            area_px=crack['area_px'],
            length_px_est=crack['length_px_est'],
            orientation_angle=crack['orientation_angle'],
            is_duplicate=crack['is_duplicate'],
            matched_with=matched_with,
        )
        # 附加跨图出现信息
        entry['appearances'] = [
            {'image_id': a['image_id'], 'confidence': a['confidence']}
            for a in crack['appearances']
        ]
        cracks_detail.append(entry)

    return build_standard_report(
        source_id=source_id,
        source_type='image_sequence',
        model_name=model_name,
        raw_detection_count=dedup_result['raw_detection_count'],
        unique_crack_count=dedup_result['unique_crack_count'],
        duplicate_removed_count=dedup_result['duplicate_removed_count'],
        cracks_detail=cracks_detail,
        wall_id=wall_id,
        extra_limitations=extra_limitations,
        extra_info={
            'matched_groups': dedup_result['matched_groups'],
            'unmatched_singles': dedup_result['unmatched_singles'],
        },
    )


def build_image_report(source_id, cracks, model_name='yolov8n-seg-cracks-joints'):
    """
    单张图像的基础报告（不做去重）

    Args:
        source_id: 图像文件名
        cracks: list of crack feature dicts
        model_name: 模型名称

    Returns:
        report: dict
    """
    cracks_detail = []
    for i, c in enumerate(cracks):
        entry = build_crack_entry(
            crack_id=f'C{i + 1}',
            track_id=None,
            frame_index=None,
            confidence=c['confidence'],
            bbox_xyxy=c['bbox_xyxy'],
            center_xy=c['center_xy'],
            area_px=c['area_px'],
            length_px_est=c['length_px_est'],
            orientation_angle=c['orientation_angle'],
            is_duplicate=False,
            matched_with=None,
        )
        cracks_detail.append(entry)

    return build_standard_report(
        source_id=source_id,
        source_type='image',
        model_name=model_name,
        raw_detection_count=len(cracks),
        unique_crack_count=len(cracks),
        duplicate_removed_count=0,
        cracks_detail=cracks_detail,
    )


def save_report(report, output_dir='outputs/reports', filename=None):
    """
    保存报告到 JSON 文件

    Args:
        report: dict 报告
        output_dir: 输出目录
        filename: 自定义文件名，默认使用时间戳

    Returns:
        filepath: 保存路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'crack_report_{timestamp}.json'

    filepath = output_path / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    return str(filepath)
