"""
DeepSeek 维修建议 PDF 报告生成模块
读取 advice JSON，生成格式化的 PDF 维修建议报告
"""

import json
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus.flowables import KeepTogether


def _find_chinese_font():
    """自动寻找可用的中文字体"""
    candidates = [
        ("SimHei", "C:/Windows/Fonts/simhei.ttf"),
        ("SimSun", "C:/Windows/Fonts/simsun.ttf"),
        ("Microsoft YaHei", "C:/Windows/Fonts/msyh.ttc"),
        ("Microsoft YaHei", "C:/Windows/Fonts/msyhbd.ttc"),
        ("KaiTi", "C:/Windows/Fonts/simkai.ttf"),
        ("FangSong", "C:/Windows/Fonts/simfang.ttf"),
        ("Arial Unicode MS", "C:/Windows/Fonts/ARIALUNI.TTF"),
        ("NotoSansCJK", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ("NotoSansSC", "/usr/share/fonts/noto-cjk/NotoSansSC-Regular.otf"),
    ]
    for name, path in candidates:
        try:
            pdfmetrics.registerFont(TTFont(name, path))
            return name
        except Exception:
            continue
    return None


def _build_styles(font_name):
    """构建 PDF 样式"""
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CNTitle', parent=styles['Title'],
        fontName=font_name, fontSize=20, leading=28,
        textColor=colors.HexColor('#1a1a2e'),
        spaceAfter=6 * mm, alignment=1,
    )
    h1_style = ParagraphStyle(
        'CNH1', parent=styles['Heading1'],
        fontName=font_name, fontSize=15, leading=22,
        textColor=colors.HexColor('#16213e'),
        spaceBefore=8 * mm, spaceAfter=4 * mm,
    )
    h2_style = ParagraphStyle(
        'CNH2', parent=styles['Heading2'],
        fontName=font_name, fontSize=12, leading=18,
        textColor=colors.HexColor('#0f3460'),
        spaceBefore=5 * mm, spaceAfter=2 * mm,
    )
    body_style = ParagraphStyle(
        'CNBody', parent=styles['BodyText'],
        fontName=font_name, fontSize=10, leading=16,
        textColor=colors.HexColor('#333333'),
    )
    small_style = ParagraphStyle(
        'CNSmall', parent=styles['BodyText'],
        fontName=font_name, fontSize=8, leading=12,
        textColor=colors.HexColor('#888888'),
    )
    risk_style = ParagraphStyle(
        'CNRisk', parent=styles['BodyText'],
        fontName=font_name, fontSize=12, leading=18,
        textColor=colors.HexColor('#c0392b'),
    )
    return title_style, h1_style, h2_style, body_style, small_style, risk_style


def _risk_color(level):
    """风险等级对应颜色"""
    mapping = {
        '轻微': colors.HexColor('#27ae60'),
        '中等': colors.HexColor('#f39c12'),
        '较严重': colors.HexColor('#e74c3c'),
        '需人工复核': colors.HexColor('#8e44ad'),
    }
    return mapping.get(level, colors.HexColor('#888888'))


def generate_advice_pdf(advice_path, output_path=None):
    """
    将维修建议 JSON 生成 PDF 报告

    Args:
        advice_path: advice JSON 文件路径
        output_path: PDF 输出路径（默认与 JSON 同目录同名 .pdf）
    """
    font_name = _find_chinese_font()
    if font_name is None:
        raise RuntimeError(
            "未找到可用的中文字体，无法生成 PDF。\n"
            "请确认系统中安装了 SimHei、SimSun 或 Microsoft YaHei 等中文字体。"
        )

    # 加载 advice JSON
    with open(advice_path, 'r', encoding='utf-8') as f:
        advice = json.load(f)

    # 输出路径
    if output_path is None:
        output_path = Path(advice_path).with_suffix('.pdf')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 样式
    title_s, h1_s, h2_s, body_s, small_s, risk_s = _build_styles(font_name)

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    story = []

    # ---- 标题 ----
    story.append(Paragraph("墙面裂缝检测与维修建议报告", title_s))
    story.append(Paragraph(
        f"生成时间: {advice.get('generated_at', '')}  |  模型: {advice.get('model_used', '')}  |  "
        f"来源: {advice.get('source_report', '')}",
        small_s
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
    story.append(Spacer(1, 5 * mm))

    # ---- 检测概况 ----
    story.append(Paragraph("一、检测概况", h1_s))
    ds = advice.get('detected_summary', {})
    overview_data = [
        ['原始检测数', str(ds.get('raw_detection_count', 0)),
         '去重后唯一裂缝数', str(ds.get('unique_crack_count', 0))],
        ['去除重复数', str(ds.get('duplicate_removed_count', 0)),
         '风险等级', advice.get('risk_level', 'N/A')],
    ]
    t = Table(overview_data, colWidths=[3.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ecf0f1')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#ecf0f1'), colors.HexColor('#ffffff')]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    # 高亮风险等级单元格
    r_color = _risk_color(advice.get('risk_level', ''))
    t.setStyle(TableStyle([('TEXTCOLOR', (3, 0), (3, 1), r_color),
                           ('FONTSIZE', (3, 0), (3, 1), 11)]))
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    assessment = advice.get('overall_assessment', '')
    if assessment:
        story.append(Paragraph("综合评估：", h2_s))
        story.append(Paragraph(assessment, body_s))

    # ---- 风险等级 ----
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("二、风险等级", h1_s))
    risk_text = f"<b>{advice.get('risk_level', 'N/A')}</b>"
    story.append(Paragraph(risk_text, risk_s))

    # ---- 可能原因 ----
    causes = advice.get('possible_causes', [])
    if causes:
        story.append(Paragraph("三、可能原因", h1_s))
        for i, cause in enumerate(causes, 1):
            story.append(Paragraph(f"{i}. {cause}", body_s))

    # ---- 修补方案 ----
    repair = advice.get('repair_plan', [])
    if repair:
        story.append(Paragraph("四、修补方案", h1_s))
        for step_item in repair:
            if isinstance(step_item, dict):
                title = step_item.get('title', '')
                desc = step_item.get('description', '')
                step_num = step_item.get('step', '')
                story.append(Paragraph(
                    f"<b>步骤 {step_num}: {title}</b>", h2_s
                ))
                story.append(Paragraph(desc, body_s))
            else:
                story.append(Paragraph(str(step_item), body_s))

    # ---- 建议材料 ----
    materials = advice.get('materials', [])
    if materials:
        story.append(Paragraph("五、建议材料", h1_s))
        for m in materials:
            story.append(Paragraph(f"- {m}", body_s))

    # ---- 人工复核 ----
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("六、人工复核建议", h1_s))
    need_review = "是" if advice.get('need_manual_review', True) else "否"
    reason = advice.get('manual_review_reason', '')
    review_data = [
        ['是否需要人工复核', need_review],
        ['复核原因', reason],
    ]
    rt = Table(review_data, colWidths=[4 * cm, 12 * cm])
    rt.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
        ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(rt)

    # ---- 局限性 ----
    limitations = advice.get('limitations', [])
    if limitations:
        story.append(Paragraph("七、当前检测局限性", h1_s))
        for lim in limitations:
            story.append(Paragraph(f"- {lim}", body_s))

    # ---- 页脚 ----
    story.append(Spacer(1, 10 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')))
    story.append(Paragraph(
        "本报告由 YOLOv8s-seg 模型自动检测 + DeepSeek AI 生成建议，仅供参考。"
        "所有尺寸为像素值，未经尺度标定。结构安全性需专业人员现场评估。",
        small_s
    ))

    doc.build(story)
    return str(output_path)
