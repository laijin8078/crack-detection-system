#!/usr/bin/env python3
"""
建筑裂缝检测 - 评估脚本
评估模型性能，计算详细指标，对比不同模型
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import numpy as np


def evaluate_model(model_path, data_yaml='data/crack-seg/data.yaml'):
    """
    评估单个模型

    Args:
        model_path: 模型权重路径
        data_yaml: 数据集配置文件

    Returns:
        metrics_dict: 指标字典
    """
    print(f"\n评估模型: {model_path}")
    print("=" * 50)

    # 加载模型
    model = YOLO(model_path)

    # 在测试集上评估
    metrics = model.val(data=data_yaml, split='test')

    # 提取指标
    def safe_mean(x):
        """把 Ultralytics 返回的标量/数组安全转成 float 平均值"""
        if x is None:
            return 0.0

        arr = np.asarray(x, dtype=float)

        if arr.size == 0:
            return 0.0

        return float(np.nanmean(arr))

    box_p = np.asarray(metrics.box.p, dtype=float)
    box_r = np.asarray(metrics.box.r, dtype=float)
    box_f1 = 2 * box_p * box_r / (box_p + box_r + 1e-6)

    seg_p = np.asarray(metrics.seg.p, dtype=float)
    seg_r = np.asarray(metrics.seg.r, dtype=float)
    seg_f1 = 2 * seg_p * seg_r / (seg_p + seg_r + 1e-6)

    metrics_dict = {
        'model': Path(model_path).stem,

        # Box metrics
        'box_map50': float(metrics.box.map50),
        'box_map': float(metrics.box.map),
        'box_precision': safe_mean(box_p),
        'box_recall': safe_mean(box_r),
        'box_f1': safe_mean(box_f1),

        # Segmentation metrics
        'seg_map50': float(metrics.seg.map50),
        'seg_map': float(metrics.seg.map),
        'seg_precision': safe_mean(seg_p),
        'seg_recall': safe_mean(seg_r),
        'seg_f1': safe_mean(seg_f1),
    }

    # 打印指标
    print("\n边界框检测指标:")
    print(f"  mAP@0.5: {metrics_dict['box_map50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics_dict['box_map']:.4f}")
    print(f"  Precision: {metrics_dict['box_precision']:.4f}")
    print(f"  Recall: {metrics_dict['box_recall']:.4f}")
    print(f"  F1-Score: {metrics_dict['box_f1']:.4f}")

    print("\n分割掩码指标:")
    print(f"  mAP@0.5: {metrics_dict['seg_map50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics_dict['seg_map']:.4f}")
    print(f"  Precision: {metrics_dict['seg_precision']:.4f}")
    print(f"  Recall: {metrics_dict['seg_recall']:.4f}")
    print(f"  F1-Score: {metrics_dict['seg_f1']:.4f}")

    return metrics_dict


def compare_models(model_paths, data_yaml='data/crack-seg/data.yaml', save_results=True):
    """
    对比多个模型

    Args:
        model_paths: 模型路径列表
        data_yaml: 数据集配置文件
        save_results: 是否保存结果
    """
    print("\n" + "=" * 50)
    print("模型对比评估")
    print("=" * 50)

    results_list = []

    for model_path in model_paths:
        if not Path(model_path).exists():
            print(f"警告: 模型不存在 {model_path}")
            continue

        try:
            metrics = evaluate_model(model_path, data_yaml)
            results_list.append(metrics)
        except Exception as e:
            print(f"评估失败: {model_path}, 错误: {e}")

    if not results_list:
        print("没有成功评估的模型")
        return

    # 创建DataFrame
    df = pd.DataFrame(results_list)

    # 打印对比表格
    print("\n" + "=" * 50)
    print("模型对比结果")
    print("=" * 50)
    print(df.to_string(index=False))

    if save_results:
        # 保存CSV
        output_dir = Path('outputs/evaluation')
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / 'model_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n结果已保存到: {csv_path}")

        # 绘制对比图表
        plot_comparison(df, output_dir)

    return df


def plot_comparison(df, output_dir):
    """
    绘制模型对比图表

    Args:
        df: 结果DataFrame
        output_dir: 输出目录
    """
    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
    plt.rcParams['axes.unicode_minus'] = False

    # 1. mAP对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Box mAP
    ax = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df['box_map50'], width, label='mAP@0.5', alpha=0.8)
    ax.bar(x + width/2, df['box_map'], width, label='mAP@0.5:0.95', alpha=0.8)
    ax.set_xlabel('模型')
    ax.set_ylabel('mAP')
    ax.set_title('边界框检测 - mAP对比')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Seg mAP
    ax = axes[0, 1]
    ax.bar(x - width/2, df['seg_map50'], width, label='mAP@0.5', alpha=0.8)
    ax.bar(x + width/2, df['seg_map'], width, label='mAP@0.5:0.95', alpha=0.8)
    ax.set_xlabel('模型')
    ax.set_ylabel('mAP')
    ax.set_title('分割掩码 - mAP对比')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision & Recall (Box)
    ax = axes[1, 0]
    ax.bar(x - width/2, df['box_precision'], width, label='Precision', alpha=0.8)
    ax.bar(x + width/2, df['box_recall'], width, label='Recall', alpha=0.8)
    ax.set_xlabel('模型')
    ax.set_ylabel('分数')
    ax.set_title('边界框检测 - Precision & Recall')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision & Recall (Seg)
    ax = axes[1, 1]
    ax.bar(x - width/2, df['seg_precision'], width, label='Precision', alpha=0.8)
    ax.bar(x + width/2, df['seg_recall'], width, label='Recall', alpha=0.8)
    ax.set_xlabel('模型')
    ax.set_ylabel('分数')
    ax.set_title('分割掩码 - Precision & Recall')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'model_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"对比图表已保存到: {plot_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估建筑裂缝检测模型')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['runs/segment/outputs/runs/crack_detection/weights/best.pt'],
                        help='模型权重路径列表')
    parser.add_argument('--data', type=str, default='data/crack-seg/data.yaml',
                        help='数据集配置文件')
    parser.add_argument('--save', action='store_true',
                        help='保存评估结果')

    args = parser.parse_args()

    if len(args.models) == 1:
        # 评估单个模型
        evaluate_model(args.models[0], args.data)
    else:
        # 对比多个模型
        compare_models(args.models, args.data, save_results=args.save)
