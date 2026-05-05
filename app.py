#!/usr/bin/env python3
"""
建筑裂缝检测 - Web后台服务
使用FastAPI实现RESTful API
支持图像上传、实时检测、结果查询等功能
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import time
from utils.database import CrackDatabase
import json

# 创建FastAPI应用
app = FastAPI(
    title="建筑裂缝检测API",
    description="基于YOLOv8的建筑表面裂缝检测系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# 全局变量
model = None
db = None


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型和初始化数据库"""
    global model, db

    # 加载模型
    model_path = 'outputs/runs/crack_detection/weights/best.pt'
    if not Path(model_path).exists():
        print(f"警告: 模型文件不存在 {model_path}")
        print("请先训练模型或指定正确的模型路径")
    else:
        print("加载模型...")
        model = YOLO(model_path)
        print("模型加载成功")

    # 初始化数据库
    db = CrackDatabase()
    print("数据库初始化成功")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global db
    if db:
        db.close()


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "name": "建筑裂缝检测API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }


@app.post("/api/detect")
async def detect_crack(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7
):
    """
    检测图像中的裂缝

    Args:
        file: 上传的图像文件
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU阈值

    Returns:
        检测结果JSON
    """
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    try:
        # 读取图像
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无效的图像文件")

        # 记录开始时间
        start_time = time.time()

        # 推理
        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        # 计算处理时间
        processing_time = time.time() - start_time

        # 提取检测结果
        detections = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes = r.boxes
            masks = r.masks

            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                xywh = box.xywh[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = r.names[cls]

                mask_polygon = None
                if masks is not None and i < len(masks):
                    mask_xy = masks[i].xy[0]
                    mask_polygon = mask_xy.tolist()

                detection = {
                    'class': class_name,
                    'class_id': cls,
                    'confidence': round(conf, 4),
                    'bbox': {
                        'x1': round(float(xyxy[0]), 2),
                        'y1': round(float(xyxy[1]), 2),
                        'x2': round(float(xyxy[2]), 2),
                        'y2': round(float(xyxy[3]), 2)
                    },
                    'center': {
                        'x': round(float(xywh[0]), 2),
                        'y': round(float(xywh[1]), 2)
                    },
                    'size': {
                        'width': round(float(xywh[2]), 2),
                        'height': round(float(xywh[3]), 2)
                    },
                    'mask_polygon': mask_polygon
                }

                detections.append(detection)

        # 生成可视化图像
        annotated = results[0].plot()

        # 保存结果图像
        output_dir = Path('outputs/predictions/api')
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_filename = f'{timestamp}_{file.filename}'
        result_path = output_dir / result_filename
        cv2.imwrite(str(result_path), annotated)

        # 保存到数据库
        detection_id = db.save_detection(
            image_name=file.filename,
            image_path=str(file.filename),
            detections=detections,
            result_path=str(result_path),
            processing_time=processing_time,
            model_name='yolov8s-seg'
        )

        # 返回结果
        return {
            'success': True,
            'detection_id': detection_id,
            'timestamp': datetime.now().isoformat(),
            'processing_time': round(processing_time, 3),
            'num_cracks': len(detections),
            'detections': detections,
            'result_image_url': f'/outputs/predictions/api/{result_filename}'
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@app.get("/api/detections")
async def get_detections(limit: int = 10):
    """获取最近的检测记录"""
    try:
        detections = db.get_recent_detections(limit=limit)
        return {
            'success': True,
            'count': len(detections),
            'detections': detections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/api/detection/{detection_id}")
async def get_detection(detection_id: int):
    """获取指定检测记录的详细信息"""
    try:
        result = db.get_detection(detection_id)
        if result is None:
            raise HTTPException(status_code=404, detail="检测记录不存在")
        return {
            'success': True,
            'data': result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/api/statistics")
async def get_statistics():
    """获取统计信息"""
    try:
        stats = db.get_statistics()
        return {
            'success': True,
            'statistics': stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'database_connected': db is not None
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
