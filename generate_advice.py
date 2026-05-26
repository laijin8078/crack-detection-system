#!/usr/bin/env python3
"""
DeepSeek 裂缝修补建议生成脚本
读取裂缝检测 JSON 报告，调用 DeepSeek API 生成结构化维修建议

用法:
  python generate_advice.py --report outputs/reports/dedup_wall_3F_east.json
  python generate_advice.py --report outputs/reports/video_track_inspection.json --output-dir outputs/advice
  python generate_advice.py --report outputs/reports/image_summary.json --dry-run
"""

import argparse
import sys
from pathlib import Path

from utils.deepseek_advisor import generate_advice, get_api_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="根据裂缝检测 JSON 报告，调用 DeepSeek 生成维修建议"
    )
    parser.add_argument(
        "--report", type=str, required=True,
        help="裂缝检测 JSON 报告路径 (如 outputs/reports/dedup_wall_001.json)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/advice",
        help="维修建议输出目录 (默认: outputs/advice)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅构建 prompt 不调用 API，用于调试"
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-chat",
        help="DeepSeek 模型名称 (默认: deepseek-chat)"
    )
    parser.add_argument(
        "--pdf", action="store_true",
        help="同时生成 PDF 格式的维修建议报告"
    )

    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"错误: 报告文件不存在: {args.report}")
        print("请先运行 inference.py 生成检测报告，再调用本脚本。")
        sys.exit(1)

    try:
        advice, output_path = generate_advice(
            report_path=str(report_path),
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )

        # 打印摘要
        print("\n" + "=" * 50)
        print("维修建议摘要")
        print("=" * 50)
        print(f"风险等级: {advice.get('risk_level', 'N/A')}")
        print(f"需要人工复核: {advice.get('need_manual_review', True)}")
        print(f"可能原因: {len(advice.get('possible_causes', []))} 条")
        print(f"修补方案: {len(advice.get('repair_plan', []))} 步")
        print(f"建议材料: {len(advice.get('materials', []))} 种")
        print("=" * 50)

        # 生成 PDF
        if args.pdf:
            from utils.advice_pdf import generate_advice_pdf
            pdf_path = generate_advice_pdf(output_path)
            print(f"\nPDF 报告已保存到: {pdf_path}")

    except RuntimeError as e:
        print(f"错误: {e}")
        sys.exit(1)
