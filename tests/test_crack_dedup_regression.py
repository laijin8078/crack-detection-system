#!/usr/bin/env python3
"""
裂缝去重回归测试脚本
验证跨图去重优化后，关键测试用例仍满足预期结果。

用法:
  python tests/test_crack_dedup_regression.py
  python tests/test_crack_dedup_regression.py --verbose
  python tests/test_crack_dedup_regression.py --case test_cases/same_crack_sequence_001/
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
import cv2
import numpy as np
import torch.nn as nn
from ultralytics.nn.modules import conv as conv_module

from utils.crack_postprocess import extract_crack_features
from utils.crack_dedup import deduplicate_cracks


def _patch_spd():
    """SPD 运行时注入（与训练代码保持一致）"""
    if hasattr(conv_module, "SPDConv"):
        return
    class SPDConv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
            super().__init__()
            self.scale = s
            self.conv = conv_module.Conv(c1 * (s ** 2), c2, k, 1, p, g, act=act)
        def forward(self, x):
            return self.conv(nn.PixelUnshuffle(self.scale)(x))
    SPDConv.__module__ = "ultralytics.nn.modules.conv"
    conv_module.SPDConv = SPDConv
    import ultralytics.nn.modules as mp; mp.SPDConv = SPDConv
    import ultralytics.nn.tasks as tm; tm.SPDConv = SPDConv


def load_expected(case_dir):
    """加载测试用例期望值"""
    expected_path = case_dir / "expected.json"
    if not expected_path.exists():
        return None
    with open(expected_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_dedup_on_case(case_dir, model, config=None):
    """对测试用例运行去重，返回结果"""
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(sorted(case_dir.glob(f'*{ext}')))
        image_files.extend(sorted(case_dir.glob(f'*{ext.upper()}')))
    image_files = list(dict.fromkeys(image_files))

    if not image_files:
        return None, f"未找到图像文件: {case_dir}"

    all_cracks = []
    image_ids = []

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        results = model.predict(source=img, conf=0.15, iou=0.7, verbose=False)
        cracks = extract_crack_features(results, img.shape, min_area_px=50)
        all_cracks.append(cracks)
        image_ids.append(img_path.name)

    if config is None:
        config = {
            'min_final_score': 0.55,
            'max_angle_diff': 45.0,
            'same_image_merge': {
                'enabled': True,
                'min_skeleton_score': 0.75,
                'min_mask_iou': 0.03,
                'max_endpoint_distance': 50,
                'min_bbox_iou': 0.02,
                'require_endpoint_proximity': True,
            },
            'cross_image_merge': {
                'min_final_score': 0.55,
                'allow_large_center_shift': True,
                'max_angle_diff': 45.0,
            },
        }

    result = deduplicate_cracks(all_cracks, image_ids, config, debug=False)
    return result, None


def test_case(case_dir, model, verbose=False):
    """测试单个用例，返回 (PASS/FAIL, details)"""
    case_name = case_dir.name
    expected = load_expected(case_dir)

    result, error = run_dedup_on_case(case_dir, model)
    if error:
        return False, f"ERROR: {error}", {}

    raw = result['raw_detection_count']
    unique = result['unique_crack_count']
    removed = result['duplicate_removed_count']

    details = {
        'case_name': case_name,
        'raw_detection_count': raw,
        'unique_crack_count': unique,
        'duplicate_removed_count': removed,
        'matched_groups': result['matched_groups'],
        'unmatched_singles': result['unmatched_singles'],
    }

    if expected is None:
        return True, f"PASS (no expected.json) | raw={raw} unique={unique}", details

    exp_unique = expected.get('expected_unique_crack_count')
    exp_min_unique = expected.get('expected_min_unique_crack_count')

    if exp_unique is not None:
        passed = unique == exp_unique
        status = "PASS" if passed else "FAIL"
        msg = f"{status} | raw={raw} unique={unique} removed={removed} | expected_unique={exp_unique}"
        return passed, msg, details

    if exp_min_unique is not None:
        passed = unique >= exp_min_unique
        status = "PASS" if passed else "FAIL"
        msg = f"{status} | raw={raw} unique={unique} removed={removed} | expected_min_unique>={exp_min_unique}"
        return passed, msg, details

    return True, f"PASS (no numeric expectation) | raw={raw} unique={unique}", details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="裂缝去重回归测试")
    parser.add_argument("--case", type=str, default=None,
                        help="仅测试指定用例目录")
    parser.add_argument("--model", type=str,
                        default="runs/segment/outputs/runs/crack_detection/weights/best.pt",
                        help="YOLO 模型路径")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="详细输出")
    args = parser.parse_args()

    # 加载模型
    print("加载模型...")
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        sys.exit(1)
    _patch_spd()
    model = YOLO(str(model_path))
    print(f"模型加载成功\n")

    # 确定测试用例
    test_cases_dir = PROJECT_ROOT / "test_cases"
    if args.case:
        case_dirs = [Path(args.case)]
    else:
        case_dirs = sorted(test_cases_dir.glob("*/"))

    if not case_dirs:
        print("未找到测试用例。请确认 test_cases/ 目录存在。")
        sys.exit(1)

    # 运行测试
    passed = 0
    failed = 0
    all_details = []

    print("=" * 70)
    print(f"{'Case':<40} {'Result':<30}")
    print("=" * 70)

    for case_dir in case_dirs:
        if not case_dir.is_dir():
            continue
        ok, msg, details = test_case(case_dir, model, verbose=args.verbose)
        if ok:
            passed += 1
        else:
            failed += 1

        case_name = case_dir.name
        print(f"{case_name:<40} {msg}")

        if args.verbose and details:
            for k, v in details.items():
                print(f"  {k}: {v}")

    print("=" * 70)
    print(f"总计: {passed} PASS, {failed} FAIL")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
