# -*- coding: utf-8 -*-
"""
后端检测流水线
读取 uploads/ 中的自动截图和手动截图，执行：
  1. YOLO 裂缝检测（自动截图跨图去重，手动截图单图检测）
  2. 保存标注图片到 results/processed_images/
  3. 生成检测报告（JSON + PDF）+ DeepSeek 维修建议（JSON + PDF）到 results/reports/
"""

import json
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))

from utils.crack_postprocess import extract_crack_features, filter_results_by_class
from utils.crack_dedup import deduplicate_cracks
from utils.crack_report import (
    build_image_report,
    build_dedup_report,
    save_report,
    NumpyEncoder,
)

BASE_DIR = Path(__file__).parent
AUTO_DIR = BASE_DIR / "uploads" / "auto"
MANUAL_DIR = BASE_DIR / "uploads" / "manual"
RESULTS_DIR = BASE_DIR / "results"
PROCESSED_DIR = RESULTS_DIR / "processed_images"
REPORTS_DIR = RESULTS_DIR / "reports"

MODEL_PATH = BASE_DIR / "runs" / "segment" / "outputs" / "runs" / "crack_detection" / "weights" / "yolov8n-seg-cracks-joints.pt"
MODEL_PATH_PT = MODEL_PATH
CONFIG_PATH = BASE_DIR / "configs" / "inference_config.yaml"


def _patch_spdconv():
    """向 ultralytics 注入 SPDConv 模块"""
    import torch.nn as nn
    from ultralytics.nn.modules import conv as conv_module

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
    SPDConv.__qualname__ = "SPDConv"
    conv_module.SPDConv = SPDConv
    if "SPDConv" not in conv_module.__all__:
        conv_module.__all__ = (*conv_module.__all__, "SPDConv")

    import ultralytics.nn.modules as modules_pkg
    modules_pkg.SPDConv = SPDConv
    if "SPDConv" not in modules_pkg.__all__:
        modules_pkg.__all__ = (*modules_pkg.__all__, "SPDConv")

    import ultralytics.nn.tasks as tasks_module
    tasks_module.SPDConv = SPDConv

    import inspect, textwrap
    source = inspect.getsource(tasks_module.parse_model)
    source = textwrap.dedent(source)
    patterns = [
        ("SCDown,\n            C2fCIB,", "SCDown,\n            SPDConv,\n            C2fCIB,"),
        ("SCDown,\n                C2fCIB,", "SCDown,\n                SPDConv,\n                C2fCIB,"),
    ]
    for old, new in patterns:
        if old in source:
            source = source.replace(old, new)
            break
    else:
        raise RuntimeError("无法找到 base_modules 注入点，请检查 ultralytics 版本兼容性")
    code = compile(source, tasks_module.__file__, "exec")
    ns = dict(tasks_module.__dict__)
    exec(code, ns)
    tasks_module.parse_model = ns["parse_model"]


def load_config():
    path = Path(CONFIG_PATH)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_image_files(directory):
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    files = []
    d = Path(directory)
    if not d.exists():
        return files
    for ext in exts:
        files.extend(d.glob(f"*{ext}"))
        files.extend(d.glob(f"*{ext.upper()}"))
    return sorted(list(dict.fromkeys(files)))


def extract_location(filename):
    """从文件名提取拍摄位置"""
    name = Path(filename).stem
    for key in ['起点', '终点', '途径点']:
        if key in name:
            return key
    return "手动拍摄"


def detect_single_image(model, image_path, config, save_annotated=True):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  无法读取图像: {image_path}")
        return [], None

    pp_cfg = config.get("postprocess", {})
    min_area = pp_cfg.get("min_area_px", 50)
    ds_ratio = pp_cfg.get("mask_downsample_ratio", 4)
    target_cls = pp_cfg.get("target_class_ids", None)

    model_cfg = config.get("model", {})
    results = model.predict(
        source=image,
        conf=model_cfg.get("conf_threshold", 0.15),
        iou=model_cfg.get("iou_threshold", 0.7),
        imgsz=model_cfg.get("imgsz", 640),
        verbose=False,
    )
    filter_results_by_class(results, target_cls)
    cracks = extract_crack_features(results, image.shape, min_area, ds_ratio,
                                    target_class_ids=target_cls)
    annotated = results[0].plot()

    if save_annotated:
        save_name = f"{Path(image_path).stem}_detected.jpg"
        save_path = PROCESSED_DIR / save_name
        cv2.imwrite(str(save_path), annotated)

    return cracks, annotated


def process_auto_images(model, config, wall_id):
    import time
    image_files = get_image_files(AUTO_DIR)
    if not image_files:
        print("📷 自动截图：无图片，跳过")
        return None

    print(f"\n{'='*50}")
    print(f"📷 处理自动截图 ({len(image_files)} 张) - 模式: 跨图去重")
    print(f"{'='*50}")

    all_cracks = []
    image_ids = []
    image_paths = []

    t_total = time.time()
    for img_path in image_files:
        t_img = time.time()
        print(f"  检测: {img_path.name}")
        cracks, _ = detect_single_image(model, img_path, config, save_annotated=True)
        all_cracks.append(cracks)
        image_ids.append(img_path.name)
        image_paths.append(str(img_path))
        elapsed = (time.time() - t_img) * 1000
        if cracks:
            print(f"    检测到 {len(cracks)} 个裂缝 ({elapsed:.0f}ms)")
        else:
            print(f"    未检测到裂缝 ({elapsed:.0f}ms)")

    t_infer = time.time()
    print(f"  推理总计: {(t_infer - t_total):.1f}s")

    dedup_cfg = config.get("dedup", {})
    dedup_result = deduplicate_cracks(
        all_cracks, image_ids, dedup_cfg,
        debug=dedup_cfg.get("debug_dedup", False),
        image_paths=image_paths,
    )
    t_dedup = time.time()
    print(f"  去重耗时: {(t_dedup - t_infer):.2f}s")
    print(f"  总耗时: {(t_dedup - t_total):.1f}s")

    print(f"\n  原始检测总数: {dedup_result['raw_detection_count']}")
    print(f"  去重后唯一裂缝: {dedup_result['unique_crack_count']}")
    print(f"  去除重复: {dedup_result['duplicate_removed_count']}")

    report = build_dedup_report(
        source_id="自动截图",
        dedup_result=dedup_result,
        wall_id=wall_id,
        model_name=config.get("model", {}).get("name", "yolov8n-seg-cracks-joints"),
    )
    report["image_type"] = "auto"

    # 为每条裂缝补充位置信息（从 appearances 中的 image_id 解析）
    for crack in report.get("cracks", []):
        appearances = crack.get("appearances", [])
        locations = list(dict.fromkeys(
            extract_location(a["image_id"]) for a in appearances
        ))
        crack["locations"] = locations
        crack["location_label"] = "、".join(locations)

    return report


def process_manual_images(model, config):
    image_files = get_image_files(MANUAL_DIR)
    if not image_files:
        print("📷 手动截图：无图片，跳过")
        return None

    print(f"\n{'='*50}")
    print(f"📷 处理手动截图 ({len(image_files)} 张) - 模式: 单图检测")
    print(f"{'='*50}")

    all_reports = []
    total_cracks = 0

    for img_path in image_files:
        print(f"  检测: {img_path.name}")
        cracks, _ = detect_single_image(model, img_path, config, save_annotated=True)
        total_cracks += len(cracks)

        location = extract_location(img_path.name)
        report = build_image_report(
            source_id=img_path.name,
            cracks=cracks,
            model_name=config.get("model", {}).get("name", "yolov8n-seg-cracks-joints"),
        )
        # 为每条裂缝标注位置
        for crack in report.get("cracks", []):
            crack["location"] = location
        report["location"] = location
        all_reports.append(report)

        if cracks:
            print(f"    检测到 {len(cracks)} 个裂缝")
        else:
            print("    未检测到裂缝")

    return {
        "source": "手动截图",
        "source_type": "image",
        "mode": "batch_image",
        "image_type": "manual",
        "total_images": len(image_files),
        "total_raw_cracks": total_cracks,
        "per_image": all_reports,
    }


def generate_advice_report(report_path):
    try:
        from utils.deepseek_advisor import generate_advice
        advice, advice_path = generate_advice(str(report_path), output_dir=str(REPORTS_DIR))
        print(f"  维修建议已保存: {advice_path}")
        return advice, advice_path
    except Exception as e:
        print(f"  ⚠️ 跳过维修建议: {e}")
        return None, None


def build_summary_report(auto_report_raw, manual_report_raw, wall_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def count_from_report(r):
        if r is None:
            return 0, 0
        if "per_image" in r:
            total = sum(img["summary"]["unique_crack_count"] for img in r["per_image"])
            raw = sum(img["summary"]["raw_detection_count"] for img in r["per_image"])
            return total, raw
        s = r.get("summary", {})
        return s.get("unique_crack_count", 0), s.get("raw_detection_count", 0)

    auto_unique, auto_raw = count_from_report(auto_report_raw)
    manual_unique, manual_raw = count_from_report(manual_report_raw)

    # 收集所有裂缝详情
    all_crack_details = []

    # 自动截图裂缝（含跨图位置信息）
    if auto_report_raw:
        for c in auto_report_raw.get("cracks", []):
            all_crack_details.append({
                "crack_id": c.get("crack_id"),
                "source": "自动截图",
                "locations": c.get("locations", []),
                "confidence": c.get("confidence"),
                "area_px": c.get("area_px"),
                "length_px_est": c.get("length_px_est"),
                "orientation_angle": c.get("orientation_angle"),
                "is_duplicate": c.get("is_duplicate", False),
            })

    # 手动截图裂缝
    if manual_report_raw:
        for img_report in manual_report_raw.get("per_image", []):
            loc = img_report.get("location", "手动拍摄")
            for c in img_report.get("cracks", []):
                all_crack_details.append({
                    "crack_id": c.get("crack_id"),
                    "source": img_report.get("image_or_video_id", ""),
                    "locations": [loc],
                    "confidence": c.get("confidence"),
                    "area_px": c.get("area_px"),
                    "length_px_est": c.get("length_px_est"),
                    "orientation_angle": c.get("orientation_angle"),
                    "is_duplicate": False,
                })

    summary = {
        "title": f"裂缝检测报告 - {wall_id}",
        "wall_id": wall_id,
        "generated_at": datetime.now().isoformat(),
        "auto_screenshots": {
            "unique_cracks": auto_unique,
            "raw_detections": auto_raw,
            "dedup_removed": auto_raw - auto_unique,
        },
        "manual_screenshots": {
            "unique_cracks": manual_unique,
            "raw_detections": manual_raw,
        },
        "total_unique_cracks": auto_unique + manual_unique,
        "total_raw_detections": auto_raw + manual_raw,
        "risk_assessment": _assess_risk(auto_unique + manual_unique),
        "all_cracks": all_crack_details,
    }

    summary_path = REPORTS_DIR / f"summary_{wall_id}_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\n📋 汇总报告已保存: {summary_path}")

    return summary, summary_path


def _assess_risk(total_cracks):
    if total_cracks == 0:
        return {"level": "无风险", "action": "墙面状况良好，无需处理"}
    elif total_cracks <= 3:
        return {"level": "轻微", "action": "建议定期观察，必要时进行表面修补"}
    elif total_cracks <= 8:
        return {"level": "中等", "action": "建议尽快进行专业评估和修复"}
    else:
        return {"level": "较严重", "action": "建议立即联系专业机构进行检测和加固处理"}


# ==================== PDF 生成 ====================

def _find_chinese_font():
    candidates = [
        ("SimHei", "C:/Windows/Fonts/simhei.ttf"),
        ("SimSun", "C:/Windows/Fonts/simsun.ttf"),
        ("Microsoft YaHei", "C:/Windows/Fonts/msyh.ttc"),
        ("Microsoft YaHei Bold", "C:/Windows/Fonts/msyhbd.ttc"),
        ("KaiTi", "C:/Windows/Fonts/simkai.ttf"),
    ]
    for name, path in candidates:
        try:
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            pdfmetrics.registerFont(TTFont(name, path))
            return name
        except Exception:
            continue
    return None


def _build_pdf_styles(font_name):
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    styles = getSampleStyleSheet()
    title_s = ParagraphStyle(
        'CNTitle', parent=styles['Title'],
        fontName=font_name, fontSize=20, leading=28,
        textColor=colors.HexColor('#1a1a2e'), spaceAfter=6 * mm, alignment=1,
    )
    h1_s = ParagraphStyle(
        'CNH1', parent=styles['Heading1'],
        fontName=font_name, fontSize=15, leading=22,
        textColor=colors.HexColor('#16213e'), spaceBefore=8 * mm, spaceAfter=4 * mm,
    )
    h2_s = ParagraphStyle(
        'CNH2', parent=styles['Heading2'],
        fontName=font_name, fontSize=12, leading=18,
        textColor=colors.HexColor('#0f3460'), spaceBefore=5 * mm, spaceAfter=2 * mm,
    )
    body_s = ParagraphStyle(
        'CNBody', parent=styles['BodyText'],
        fontName=font_name, fontSize=10, leading=16,
        textColor=colors.HexColor('#333333'),
    )
    small_s = ParagraphStyle(
        'CNSmall', parent=styles['BodyText'],
        fontName=font_name, fontSize=8, leading=12,
        textColor=colors.HexColor('#888888'),
    )
    return title_s, h1_s, h2_s, body_s, small_s


def _risk_color(level):
    from reportlab.lib import colors
    mapping = {
        '无风险': colors.HexColor('#27ae60'),
        '轻微': colors.HexColor('#27ae60'),
        '中等': colors.HexColor('#f39c12'),
        '较严重': colors.HexColor('#e74c3c'),
        '需人工复核': colors.HexColor('#8e44ad'),
    }
    return mapping.get(level, colors.HexColor('#888888'))


def generate_detection_pdf(report_path, wall_id=""):
    """将检测报告 JSON 转为 PDF"""
    font_name = _find_chinese_font()
    if font_name is None:
        print("  ⚠️ 未找到中文字体，跳过检测报告 PDF 生成")
        return None

    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
    )

    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    title_s, h1_s, h2_s, body_s, small_s = _build_pdf_styles(font_name)

    pdf_path = Path(report_path).with_suffix('.pdf')
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    story = []

    # 标题
    image_type = report.get("image_type", "")
    type_label = "自动截图（跨图去重）" if image_type == "auto" else "手动截图（单图检测）"
    story.append(Paragraph(f"裂缝检测报告 - {type_label}", title_s))
    story.append(Paragraph(
        f"墙面: {wall_id or report.get('wall_id', '-')}  |  "
        f"来源: {report.get('image_or_video_id', '')}  |  "
        f"时间: {report.get('timestamp', '')}",
        small_s
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
    story.append(Spacer(1, 5 * mm))

    # 检测统计
    story.append(Paragraph("一、检测统计", h1_s))
    s = report.get("summary", {})
    stats_data = [
        ["原始检测数", str(s.get("raw_detection_count", 0)),
         "去重后唯一裂缝", str(s.get("unique_crack_count", 0))],
        ["去除重复数", str(s.get("duplicate_removed_count", 0)),
         "平均置信度", f"{s.get('overall_confidence', 0) * 100:.1f}%"],
    ]
    t = Table(stats_data, colWidths=[3.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ecf0f1')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    # 裂缝详情
    cracks = report.get("cracks", [])
    if cracks:
        story.append(Paragraph("二、裂缝详情", h1_s))
        header = ["ID", "位置", "置信度", "面积(px²)", "长度(px)", "角度(°)", "跨图重复"]
        table_data = [header]
        for c in cracks:
            dup = "是" if c.get("is_duplicate") else "否"
            loc = c.get("location_label") or c.get("location", "")
            table_data.append([
                str(c.get("crack_id", "-")),
                loc,
                f"{c.get('confidence', 0) * 100:.1f}%",
                str(c.get("area_px", 0)),
                f"{c.get('length_px_est', 0):.0f}",
                f"{c.get('orientation_angle', 0):.0f}",
                dup,
            ])
        ct = Table(table_data, colWidths=[1.5 * cm, 2.8 * cm, 1.8 * cm, 1.8 * cm, 1.6 * cm, 1.6 * cm, 1.8 * cm])
        ct.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(ct)

    # 局限性
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("三、检测局限性", h1_s))
    for lim in report.get("limitations", []):
        story.append(Paragraph(f"- {lim}", body_s))

    # 页脚
    story.append(Spacer(1, 10 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')))
    story.append(Paragraph(
        "本报告由 yolov8n-seg-cracks-joints 自动生成，所有尺寸为像素值，未经尺度标定。仅供参考。", small_s
    ))

    doc.build(story)
    print(f"  📄 检测报告 PDF 已保存: {pdf_path}")
    return str(pdf_path)


def generate_summary_pdf(summary_path, advice_path=None, annotated_images=None):
    """将汇总报告 JSON 转为 PDF，可选嵌入 DeepSeek 维修建议和检测标注图"""
    font_name = _find_chinese_font()
    if font_name is None:
        print("  ⚠️ 未找到中文字体，跳过汇总报告 PDF 生成")
        return None

    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable, PageBreak, Image
    )

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    advice = None
    if advice_path and Path(advice_path).exists():
        try:
            with open(advice_path, 'r', encoding='utf-8') as f:
                advice = json.load(f)
            print(f"  📄 已加载维修建议: {advice_path}")
        except Exception as e:
            print(f"  ⚠️ 加载维修建议失败: {e}")

    title_s, h1_s, h2_s, body_s, small_s = _build_pdf_styles(font_name)

    pdf_path = Path(summary_path).with_suffix('.pdf')
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    story = []

    # 标题
    story.append(Paragraph(f"墙面裂缝检测汇总报告", title_s))
    story.append(Paragraph(
        f"墙面: {summary.get('wall_id', '-')}  |  "
        f"时间: {summary.get('generated_at', '')}",
        small_s
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
    story.append(Spacer(1, 5 * mm))

    # 检测统计
    story.append(Paragraph("一、检测总览", h1_s))
    auto = summary.get("auto_screenshots", {})
    manual = summary.get("manual_screenshots", {})
    overview_data = [
        ["类别", "原始检测数", "唯一裂缝数", "去除重复"],
        ["自动截图", str(auto.get("raw_detections", 0)), str(auto.get("unique_cracks", 0)),
         str(auto.get("dedup_removed", 0))],
        ["手动截图", str(manual.get("raw_detections", 0)), str(manual.get("unique_cracks", 0)), "-"],
        ["合计", str(summary.get("total_raw_detections", 0)),
         str(summary.get("total_unique_cracks", 0)),
         str(auto.get("dedup_removed", 0))],
    ]
    t = Table(overview_data, colWidths=[3 * cm, 3.5 * cm, 3.5 * cm, 3 * cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 5 * mm))

    # 风险等级
    risk = summary.get("risk_assessment", {})
    story.append(Paragraph("二、风险评估", h1_s))
    risk_level = risk.get("level", "未知")
    r_color = _risk_color(risk_level)
    story.append(Paragraph(
        f"<font color='{r_color}'><b>风险等级: {risk_level}</b></font>", body_s
    ))
    story.append(Paragraph(f"建议措施: {risk.get('action', '-')}", body_s))

    # 全部裂缝清单
    all_cracks = summary.get("all_cracks", [])
    if all_cracks:
        story.append(Spacer(1, 5 * mm))
        story.append(Paragraph("三、全部裂缝清单", h1_s))
        header = ["编号", "位置", "置信度", "面积(px)", "长度(px)", "角度"]
        col_w = [1.2*cm, 3.2*cm, 2.0*cm, 2.0*cm, 2.0*cm, 2.0*cm]
        table_data = [header]
        for i, c in enumerate(all_cracks, 1):
            locs = c.get("locations", [])
            loc_str = "、".join(locs) if locs else "-"
            table_data.append([
                str(i),
                loc_str,
                f"{c.get('confidence', 0)*100:.1f}%",
                str(c.get("area_px", 0)),
                f"{c.get('length_px_est', 0):.0f}",
                f"{c.get('orientation_angle', 0):.0f}",
            ])
        ct = Table(table_data, colWidths=col_w)
        ct.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(ct)

    # ===== 检测标注图 =====
    if annotated_images:
        story.append(PageBreak())
        story.append(Paragraph("四、检测标注图", h1_s))
        max_w = 15 * cm
        max_h = 10 * cm
        for img_path in annotated_images:
            if not os.path.isfile(img_path):
                continue
            try:
                story.append(Paragraph(f"{Path(img_path).name}", h2_s))
                img = Image(img_path, width=max_w, height=max_h)
                img.hAlign = 'CENTER'
                story.append(img)
                story.append(Spacer(1, 5 * mm))
            except Exception as e:
                story.append(Paragraph(f"（无法加载图片: {e}）", small_s))

    # ===== DeepSeek 维修建议（嵌入报告） =====
    if advice:
        story.append(Spacer(1, 8 * mm))
        story.append(Paragraph("五、AI 维修建议（DeepSeek）", h1_s))

        assessment = advice.get("overall_assessment", "")
        if assessment:
            story.append(Paragraph("<b>综合评估：</b>" + assessment, body_s))
            story.append(Spacer(1, 3 * mm))

        risk_level = advice.get("risk_level", "")
        if risk_level:
            r_color = _risk_color(risk_level)
            story.append(Paragraph(
                f"<b>AI风险等级：</b><font color='{r_color}'>{risk_level}</font>", body_s
            ))

        causes = advice.get("possible_causes", [])
        if causes:
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph("<b>可能原因：</b>", body_s))
            for cause in causes:
                story.append(Paragraph(f"  - {cause}", body_s))

        repair = advice.get("repair_plan", [])
        if repair:
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph("<b>修补方案：</b>", body_s))
            for step in repair:
                if isinstance(step, dict):
                    story.append(Paragraph(
                        f"  步骤{step.get('step', '')}: {step.get('title', '')} — {step.get('description', '')}", body_s
                    ))
                else:
                    story.append(Paragraph(f"  - {step}", body_s))

        materials = advice.get("materials", [])
        if materials:
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph("<b>建议材料：</b>", body_s))
            for m in materials:
                story.append(Paragraph(f"  - {m}", body_s))

        need_review = "是" if advice.get("need_manual_review", True) else "否"
        reason = advice.get("manual_review_reason", "")
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(f"<b>需要人工复核：</b>{need_review}  |  {reason}", body_s))

    # 页脚
    story.append(Spacer(1, 15 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')))
    story.append(Paragraph(
        "本报告由 yolov8n-seg-cracks-joints 模型自动检测 + DeepSeek AI 生成建议，所有尺寸为像素值，未经尺度标定。"
        "结构安全性需专业人员现场评估。",
        small_s
    ))

    doc.build(story)
    print(f"  📄 汇总报告 PDF 已保存: {pdf_path}")
    return str(pdf_path)


def _generate_advice_pdf(advice_path):
    """调用 utils.advice_pdf 将 advice JSON 转为 PDF"""
    try:
        from utils.advice_pdf import generate_advice_pdf
        pdf_path = generate_advice_pdf(str(advice_path))
        print(f"  📄 维修建议 PDF 已保存: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"  ⚠️ 维修建议 PDF 生成失败: {e}")
        return None


# ==================== 主入口 ====================

def cleanup_results():
    """清理上一次的检测结果"""
    for folder in [PROCESSED_DIR, REPORTS_DIR]:
        if folder.exists():
            for f in folder.glob("*"):
                try:
                    if f.is_file():
                        f.unlink()
                except Exception as e:
                    print(f"  [cleanup] 清理失败 {f}: {e}")
    print("[cleanup] 已清理上一次的检测报告和预测图")


def process_uploads(wall_id="未命名", status_dict=None):
    """
    主入口：处理 uploads/ 中的所有图片
    由 server.py 在图片传输完成后调用（后台线程）

    Args:
        wall_id: 墙面标识
        status_dict: 可选，server.py 传入的共享字典，用于状态上报。
                     格式: status_dict[wall_id] = {"status": "processing|done|error", "files": [], "summary": {}}
    """
    print(f"\n{'#'*60}")
    print(f"# 后端检测流水线启动")
    print(f"# 墙面标识: {wall_id}")
    print(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 清理上一次的检测结果
    cleanup_results()

    # 收集所有生成的文件
    generated_files = []

    config = load_config()
    model_cfg = config.get("model", {})
    use_half = model_cfg.get("half", False)
    model_weights = model_cfg.get("weights", str(MODEL_PATH))
    model_path = BASE_DIR / model_weights if not Path(model_weights).is_absolute() else Path(model_weights)
    if not model_path.exists():
        print(f"   配置模型未找到 ({model_path})，回退到默认 ONNX 模型")
        model_path = MODEL_PATH if MODEL_PATH.exists() else MODEL_PATH_PT

    print(f"\n🔧 加载 YOLO 模型: {model_path}")
    print(f"   半精度(half): {use_half}")

    is_onnx = model_path.suffix == ".onnx"
    if is_onnx:
        print("   使用 ONNX Runtime 推理 (CPU 加速)")
        model = YOLO(str(model_path))
    else:
        _patch_spdconv()
        model = YOLO(str(model_path))

    # 半精度（仅 GPU 有效，CPU 上跳过）
    if use_half and not is_onnx:
        import torch
        if torch.cuda.is_available():
            try:
                model.model.half()
                print("   已启用 FP16 半精度 (GPU)")
            except Exception as e:
                print(f"   ⚠️ FP16 不可用: {e}")
        else:
            print("   ⚠️ 未检测到 GPU，跳过 FP16（CPU 不支持半精度推理）")

    print("   模型加载成功")

    try:
        # === 处理自动截图（跨图去重） ===
        auto_report = process_auto_images(model, config, wall_id)
        # === 处理手动截图（单图检测） ===
        manual_report = process_manual_images(model, config)

        if auto_report is None and manual_report is None:
            print("\n⚠️ 没有检测到任何图片，流水线结束")
            if status_dict is not None:
                status_dict[wall_id] = {"status": "done", "files": [], "summary": None}
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # === 汇总报告 JSON ===
        summary, summary_json = build_summary_report(auto_report, manual_report, wall_id)
        print(f"\n📊 报告数据确认:")
        print(f"   自动截图 - 唯一裂缝: {summary['auto_screenshots']['unique_cracks']}, "
              f"原始检测: {summary['auto_screenshots']['raw_detections']}")
        print(f"   手动截图 - 唯一裂缝: {summary['manual_screenshots']['unique_cracks']}, "
              f"原始检测: {summary['manual_screenshots']['raw_detections']}")
        print(f"   全部裂缝清单条目: {len(summary.get('all_cracks', []))}")
        print(f"   裂缝总数: {summary['total_unique_cracks']}")

        # === DeepSeek 维修建议 JSON ===
        print("🤖 生成 DeepSeek 维修建议...")
        _, advice_path = generate_advice_report(str(summary_json))

        # === 收集标注图片路径 ===
        annotated_images = sorted(
            [str(p) for p in PROCESSED_DIR.glob("*") if p.is_file()],
            key=lambda x: os.path.getmtime(x), reverse=True
        )

        # === 汇总报告 PDF（合并标注图 + DeepSeek 建议） ===
        pdf_path = generate_summary_pdf(str(summary_json), advice_path, annotated_images)
        if pdf_path:
            generated_files.append(pdf_path)

        # === 上报状态 ===
        if status_dict is not None:
            status_dict[wall_id] = {
                "status": "done",
                "timestamp": timestamp,
                "files": generated_files,
                "summary": {
                    "total_unique_cracks": summary.get("total_unique_cracks", 0),
                    "risk_level": summary.get("risk_assessment", {}).get("level", ""),
                },
            }

    except Exception as e:
        print(f"\n❌ 流水线处理异常: {e}")
        import traceback
        traceback.print_exc()
        if status_dict is not None:
            status_dict[wall_id] = {
                "status": "error",
                "error": str(e),
                "files": generated_files,
            }
        return

    # === 流水线完成 ===
    print(f"\n{'#'*60}")
    print(f"# 流水线处理完成")
    print(f"# 标注图片: {PROCESSED_DIR}")
    print(f"# 检测报告: {REPORTS_DIR}")
    print(f"# 唯一裂缝总数: {summary['total_unique_cracks']}")
    print(f"# 风险等级: {summary['risk_assessment']['level']}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="后端检测流水线")
    parser.add_argument("--wall-id", type=str, default="未命名", help="墙面标识")
    args = parser.parse_args()
    process_uploads(args.wall_id)
