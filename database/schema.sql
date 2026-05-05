-- 建筑裂缝检测系统数据库结构
-- SQLite数据库

-- 检测记录表
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_name TEXT NOT NULL,
    image_path TEXT NOT NULL,
    result_path TEXT,
    num_cracks INTEGER DEFAULT 0,
    avg_confidence REAL,
    processing_time REAL,
    model_name TEXT,
    notes TEXT
);

-- 裂缝详情表
CREATE TABLE IF NOT EXISTS crack_details (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    class_id INTEGER,
    confidence REAL NOT NULL,
    bbox_x1 REAL,
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    center_x REAL,
    center_y REAL,
    width REAL,
    height REAL,
    mask_polygon TEXT,
    area REAL,
    FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE
);

-- 创建索引以提高查询性能
CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_num_cracks ON detections(num_cracks);
CREATE INDEX IF NOT EXISTS idx_crack_details_detection_id ON crack_details(detection_id);
CREATE INDEX IF NOT EXISTS idx_crack_details_class_name ON crack_details(class_name);
CREATE INDEX IF NOT EXISTS idx_crack_details_confidence ON crack_details(confidence);

-- 统计视图：每日检测统计
CREATE VIEW IF NOT EXISTS daily_statistics AS
SELECT
    DATE(timestamp) as date,
    COUNT(*) as total_images,
    SUM(num_cracks) as total_cracks,
    AVG(num_cracks) as avg_cracks_per_image,
    AVG(avg_confidence) as avg_confidence
FROM detections
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- 统计视图：裂缝类别分布
CREATE VIEW IF NOT EXISTS crack_class_distribution AS
SELECT
    class_name,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence,
    MIN(confidence) as min_confidence,
    MAX(confidence) as max_confidence
FROM crack_details
GROUP BY class_name
ORDER BY count DESC;
