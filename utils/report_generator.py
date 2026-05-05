"""
报告生成模块
生成PDF格式的检测报告
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import io


class ReportGenerator:
    def __init__(self):
        """初始化报告生成器"""
        # 注册中文字体（如果有的话）
        try:
            pdfmetrics.registerFont(TTFont('SimHei', 'SimHei.ttf'))
            self.chinese_font = 'SimHei'
        except:
            self.chinese_font = 'Helvetica'

        self.styles = getSampleStyleSheet()

    def generate_report(self, detection_data, output_path):
        """
        生成检测报告

        Args:
            detection_data: 检测数据字典
            output_path: 输出PDF路径
        """
        # 创建PDF文档
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )

        # 构建报告内容
        story = []

        # 标题
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=1  # 居中
        )
        title = Paragraph("建筑裂缝检测报告", title_style)
        story.append(title)

        # 基本信息
        info_data = [
            ['报告生成时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['检测图像', detection_data.get('image_name', '-')],
            ['检测ID', str(detection_data.get('detection_id', '-'))],
            ['处理时间', f"{detection_data.get('processing_time', 0):.3f}秒"],
            ['模型名称', detection_data.get('model_name', 'YOLOv8s-seg')]
        ]

        info_table = Table(info_data, colWidths=[5*cm, 10*cm])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), self.chinese_font),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(info_table)
        story.append(Spacer(1, 20))

        # 检测概览
        overview_style = self.styles['Heading2']
        story.append(Paragraph("检测概览", overview_style))
        story.append(Spacer(1, 10))

        num_cracks = detection_data.get('num_cracks', 0)
        avg_conf = detection_data.get('avg_confidence', 0)

        overview_text = f"""
        本次检测共发现 <b>{num_cracks}</b> 个裂缝，
        平均置信度为 <b>{avg_conf*100:.2f}%</b>。
        """

        story.append(Paragraph(overview_text, self.styles['BodyText']))
        story.append(Spacer(1, 20))

        # 检测结果图像
        if detection_data.get('result_path'):
            story.append(Paragraph("检测结果图像", overview_style))
            story.append(Spacer(1, 10))

            try:
                img = Image(detection_data['result_path'], width=15*cm, height=15*cm)
                story.append(img)
                story.append(Spacer(1, 20))
            except:
                story.append(Paragraph("无法加载结果图像", self.styles['BodyText']))

        # 裂缝详情表格
        if detection_data.get('detections'):
            story.append(Paragraph("裂缝详情", overview_style))
            story.append(Spacer(1, 10))

            crack_data = [['序号', '类别', '置信度', '中心坐标', '尺寸']]

            for i, det in enumerate(detection_data['detections'], 1):
                crack_data.append([
                    str(i),
                    det.get('class', '-'),
                    f"{det.get('confidence', 0)*100:.2f}%",
                    f"({det['center']['x']:.0f}, {det['center']['y']:.0f})",
                    f"{det['size']['width']:.0f}×{det['size']['height']:.0f}"
                ])

            crack_table = Table(crack_data, colWidths=[2*cm, 3*cm, 3*cm, 4*cm, 3*cm])
            crack_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), self.chinese_font),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(crack_table)
            story.append(Spacer(1, 20))

        # 结论和建议
        story.append(Paragraph("结论与建议", overview_style))
        story.append(Spacer(1, 10))

        if num_cracks == 0:
            conclusion = "本次检测未发现明显裂缝，建筑表面状况良好。"
        elif num_cracks <= 2:
            conclusion = "检测到少量裂缝，建议定期观察，必要时进行修补。"
        elif num_cracks <= 5:
            conclusion = "检测到多处裂缝，建议尽快进行专业评估和修复。"
        else:
            conclusion = "检测到大量裂缝，建议立即进行专业检测和加固处理。"

        story.append(Paragraph(conclusion, self.styles['BodyText']))

        # 生成PDF
        doc.build(story)

        return output_path


def generate_detection_report(detection_id, db, output_dir='outputs/reports'):
    """
    为指定检测记录生成报告

    Args:
        detection_id: 检测记录ID
        db: 数据库对象
        output_dir: 输出目录

    Returns:
        report_path: 报告文件路径
    """
    # 获取检测数据
    result = db.get_detection(detection_id)
    if not result:
        raise ValueError(f"检测记录不存在: {detection_id}")

    detection = result['detection']
    cracks = result['cracks']

    # 准备报告数据
    report_data = {
        'detection_id': detection['id'],
        'image_name': detection['image_name'],
        'processing_time': detection['processing_time'],
        'model_name': detection['model_name'],
        'num_cracks': detection['num_cracks'],
        'avg_confidence': detection['avg_confidence'],
        'result_path': detection['result_path'],
        'detections': []
    }

    for crack in cracks:
        report_data['detections'].append({
            'class': crack['class_name'],
            'confidence': crack['confidence'],
            'center': {'x': crack['center_x'], 'y': crack['center_y']},
            'size': {'width': crack['width'], 'height': crack['height']}
        })

    # 生成报告
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_filename = f'report_{detection_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    report_path = output_path / report_filename

    generator = ReportGenerator()
    generator.generate_report(report_data, report_path)

    return report_path


if __name__ == '__main__':
    # 测试报告生成
    from utils.database import CrackDatabase

    db = CrackDatabase()
    detections = db.get_recent_detections(limit=1)

    if detections:
        detection_id = detections[0]['id']
        report_path = generate_detection_report(detection_id, db)
        print(f"报告已生成: {report_path}")
    else:
        print("没有检测记录")
