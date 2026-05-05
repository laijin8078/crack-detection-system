"""
数据库操作工具
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import json


class CrackDatabase:
    def __init__(self, db_path='database/crack_detection.db'):
        """初始化数据库连接"""
        self.db_path = db_path
        self.conn = None
        self.init_database()

    def init_database(self):
        """初始化数据库，创建表结构"""
        # 确保数据库目录存在
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # 连接数据库
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        # 读取并执行schema.sql
        schema_path = Path('database/schema.sql')
        if schema_path.exists():
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
                self.conn.executescript(schema_sql)
                self.conn.commit()

    def save_detection(self, image_name, image_path, detections, result_path=None,
                       processing_time=None, model_name='yolov8s-seg'):
        """
        保存检测结果

        Args:
            image_name: 图像文件名
            image_path: 图像路径
            detections: 检测结果列表
            result_path: 结果图像路径
            processing_time: 处理时间（秒）
            model_name: 模型名称

        Returns:
            detection_id: 检测记录ID
        """
        cursor = self.conn.cursor()

        # 计算平均置信度
        avg_conf = sum(d['confidence'] for d in detections) / len(detections) if detections else 0

        # 插入检测记录
        cursor.execute('''
            INSERT INTO detections (image_name, image_path, result_path, num_cracks,
                                    avg_confidence, processing_time, model_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (image_name, image_path, result_path, len(detections), avg_conf, processing_time, model_name))

        detection_id = cursor.lastrowid

        # 插入裂缝详情
        for det in detections:
            bbox = det.get('bbox', {})
            center = det.get('center', {})
            size = det.get('size', {})
            mask_polygon = json.dumps(det.get('mask_polygon')) if det.get('mask_polygon') else None

            cursor.execute('''
                INSERT INTO crack_details (detection_id, class_name, class_id, confidence,
                                           bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                                           center_x, center_y, width, height, mask_polygon)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (detection_id, det['class'], det.get('class_id', 0), det['confidence'],
                  bbox.get('x1'), bbox.get('y1'), bbox.get('x2'), bbox.get('y2'),
                  center.get('x'), center.get('y'), size.get('width'), size.get('height'),
                  mask_polygon))

        self.conn.commit()
        return detection_id

    def get_detection(self, detection_id):
        """获取检测记录"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM detections WHERE id = ?', (detection_id,))
        detection = cursor.execute('SELECT * FROM detections WHERE id = ?', (detection_id,)).fetchone()

        if detection:
            # 获取裂缝详情
            cracks = cursor.execute(
                'SELECT * FROM crack_details WHERE detection_id = ?',
                (detection_id,)
            ).fetchall()

            return {
                'detection': dict(detection),
                'cracks': [dict(crack) for crack in cracks]
            }
        return None

    def get_recent_detections(self, limit=10):
        """获取最近的检测记录"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM detections
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self):
        """获取统计信息"""
        cursor = self.conn.cursor()

        # 总体统计
        stats = cursor.execute('''
            SELECT
                COUNT(*) as total_images,
                SUM(num_cracks) as total_cracks,
                AVG(num_cracks) as avg_cracks_per_image,
                AVG(avg_confidence) as avg_confidence
            FROM detections
        ''').fetchone()

        # 每日统计
        daily_stats = cursor.execute('''
            SELECT * FROM daily_statistics LIMIT 7
        ''').fetchall()

        # 类别分布
        class_dist = cursor.execute('''
            SELECT * FROM crack_class_distribution
        ''').fetchall()

        return {
            'overall': dict(stats) if stats else {},
            'daily': [dict(row) for row in daily_stats],
            'class_distribution': [dict(row) for row in class_dist]
        }

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
