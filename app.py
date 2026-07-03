#!/usr/bin/env python3
"""
建筑裂缝检测 - Web后台服务
使用FastAPI实现RESTful API
支持图像上传、实时检测、结果查询等功能
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import time
import asyncio
from utils.database import CrackDatabase
from utils.crack_postprocess import extract_crack_features, filter_results_by_class
from utils.crack_report import build_image_report, build_dedup_report
from utils.crack_dedup import deduplicate_cracks
from utils.deepseek_advisor import generate_advice
from utils.advice_pdf import generate_advice_pdf
import json
import yaml

app = FastAPI(
    title="建筑裂缝智能检测系统",
    description="基于 yolov8n-seg-cracks-joints 的墙面裂缝实例分割、跨图去重、DeepSeek AI 维修建议系统",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

model = None
db = None


@app.on_event("startup")
async def startup_event():
    """服务启动：加载 YOLO 模型和数据库"""
    global model, db

    model_path = 'runs/segment/outputs/runs/crack_detection/weights/yolov8n-seg-cracks-joints.pt'
    if not Path(model_path).exists():
        print(f"警告: 模型文件不存在 {model_path}")
    else:
        print("正在加载模型...")
        model = YOLO(model_path)
        print("模型加载成功")

    db = CrackDatabase()
    print("数据库初始化成功")


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭：清理数据库连接"""
    global db
    if db:
        db.close()


# ==================== 基础接口 ====================

@app.get("/", tags=["系统信息"], summary="获取服务状态")
async def 服务状态():
    """返回 API 名称、版本、运行状态和模型加载情况"""
    return {
        "系统": "建筑裂缝智能检测系统",
        "版本": "2.0.0",
        "运行状态": "正常",
        "模型已加载": model is not None,
    }


@app.get("/api/health", tags=["系统信息"], summary="健康检查")
async def 健康检查():
    """检查模型和数据库是否正常连接"""
    return {
        "服务状态": "正常",
        "模型已加载": model is not None,
        "数据库已连接": db is not None,
    }


# ==================== 单帧检测 ====================

@app.post("/api/detect", tags=["裂缝检测"], summary="单张图像裂缝检测")
async def 单帧裂缝检测(
    file: UploadFile = File(..., description="墙面图像文件（jpg/png）"),
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.7,
):
    """
    上传一张墙面图像，返回裂缝检测结果。

    - **file**: 图像文件（支持 jpg、png、bmp）
    - **conf_threshold**: 置信度阈值（0-1），值越高结果越严格
    - **iou_threshold**: NMS IoU 阈值（0-1）
    """
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载，请检查模型文件是否存在")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无法解析图像文件，请确认上传的是有效图片")

        start_time = time.time()

        results = model.predict(source=image, conf=conf_threshold, iou=iou_threshold, verbose=False)
        processing_time = time.time() - start_time

        filter_results_by_class(results, [1])
        cracks = extract_crack_features(results, image.shape, min_area_px=50, mask_downsample_ratio=4,
                                        target_class_ids=[1])

        detections = []
        for i, c in enumerate(cracks):
            detections.append({
                'class': 'crack',
                'class_id': 0,
                'confidence': c['confidence'],
                'bbox': {
                    'x1': c['bbox_xyxy'][0], 'y1': c['bbox_xyxy'][1],
                    'x2': c['bbox_xyxy'][2], 'y2': c['bbox_xyxy'][3],
                },
                'center': {'x': c['center_xy'][0], 'y': c['center_xy'][1]},
                'size': {
                    'width': round(c['bbox_xyxy'][2] - c['bbox_xyxy'][0], 2),
                    'height': round(c['bbox_xyxy'][3] - c['bbox_xyxy'][1], 2),
                },
                'mask_polygon': c.get('mask_polygon'),
                'area_px': c['area_px'],
                'length_px_est': c['length_px_est'],
                'orientation_angle': c['orientation_angle'],
            })

        report = build_image_report(
            source_id=file.filename, cracks=cracks, model_name='yolov8n-seg-cracks-joints',
        )

        annotated = results[0].plot()
        output_dir = Path('outputs/predictions/api')
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_filename = f'{timestamp}_{file.filename}'
        result_path = output_dir / result_filename
        cv2.imwrite(str(result_path), annotated)

        detection_id = db.save_detection(
            image_name=file.filename, image_path=str(file.filename),
            detections=detections, result_path=str(result_path),
            processing_time=processing_time, model_name='yolov8n-seg-cracks-joints',
        )

        return {
            '检测成功': True,
            '记录ID': detection_id,
            '时间戳': datetime.now().isoformat(),
            '处理耗时(秒)': round(processing_time, 3),
            '裂缝数量': len(cracks),
            '检测详情': detections,
            '检测报告': report,
            '标注图地址': f'/outputs/predictions/api/{result_filename}',
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


# ==================== 批量序列检测（核心接口） ====================

@app.post("/api/detect-sequence", tags=["裂缝检测"], summary="批量图像序列检测 + 跨图去重 + AI 建议")
async def 批量序列检测(
    files: list[UploadFile] = File(..., description="同一墙面拍摄的多张图像"),
    wall_id: str = Form(default="default_wall", description="墙面标识名称"),
):
    """
    上传同一墙面多角度拍摄的图像，自动完成：

    1. 逐帧 YOLO 裂缝检测
    2. 跨图像裂缝去重（骨架形态相似度匹配）
    3. DeepSeek AI 维修建议生成
    4. 维修建议 PDF 报告

    通过 SSE（Server-Sent Events）实时回传处理进度。

    - **files**: 同一次巡检拍摄的全部墙面图像（支持同时上传多张）
    - **wall_id**: 给这面墙起个名字，如 "3楼东墙"、"wall_A"，用于区分不同墙面
    """
    import yaml as _yaml

    dedup_cfg = {}
    cfg_path = Path("configs/inference_config.yaml")
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as _f:
            raw_cfg = _yaml.safe_load(_f)
            dedup_cfg = raw_cfg.get("dedup", {})
            dedup_cfg.pop("debug_dedup", None)

    async def 处理并推送进度():
        try:
            total = len(files)
            all_cracks = []
            image_ids = []
            annotated_urls = []
            detection_id = None
            result_dir = Path("outputs/predictions/sequence")
            result_dir.mkdir(parents=True, exist_ok=True)

            # ---- 阶段1: 逐帧检测 ----
            for idx, f in enumerate(files):
                yield f'data: {{"事件":"检测进度","当前":{idx+1},"总数":{total},"图片":"{f.filename}"}}\n\n'
                await asyncio.sleep(0.01)

                contents = await f.read()
                nparr = np.frombuffer(contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    continue

                results = model.predict(source=image, conf=0.15, iou=0.7, verbose=False)
                filter_results_by_class(results, [1])
                cracks = extract_crack_features(results, image.shape, min_area_px=50, mask_downsample_ratio=4,
                                        target_class_ids=[1])
                all_cracks.append(cracks)
                image_ids.append(f.filename)

                annotated = results[0].plot()
                save_name = f"{datetime.now().strftime('%H%M%S')}_{idx}_{f.filename}"
                cv2.imwrite(str(result_dir / save_name), annotated)
                annotated_urls.append(f"/outputs/predictions/sequence/{save_name}")

            yield f'data: {{"事件":"检测进度","当前":{total},"总数":{total},"状态":"检测完成，正在跨图去重..."}}\n\n'
            await asyncio.sleep(0.01)

            # ---- 阶段2: 跨图去重 ----
            dedup_result = deduplicate_cracks(all_cracks, image_ids, dedup_cfg, debug=False)
            yield f'data: {{"事件":"去重完成","原始检测数":{dedup_result["raw_detection_count"]},"唯一裂缝数":{dedup_result["unique_crack_count"]},"去除重复数":{dedup_result["duplicate_removed_count"]}}}\n\n'
            await asyncio.sleep(0.01)

            # ---- 生成去重报告 ----
            report = build_dedup_report(
                source_id=wall_id, dedup_result=dedup_result,
                wall_id=wall_id, model_name="yolov8n-seg-cracks-joints",
            )
            report_path = Path("outputs/reports") / f"去重报告_{wall_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as _f:
                json.dump(report, _f, indent=2, ensure_ascii=False)

            try:
                detection_id = db.save_detection(
                    image_name=f"sequence_{wall_id}", image_path=str(report_path),
                    detections=[{
                        "class": "crack", "class_id": 0, "confidence": c["avg_confidence"],
                        "bbox": {"x1": c["bbox_xyxy"][0], "y1": c["bbox_xyxy"][1],
                                 "x2": c["bbox_xyxy"][2], "y2": c["bbox_xyxy"][3]},
                        "center": {"x": c["center_xy"][0], "y": c["center_xy"][1]},
                        "size": {"width": c["bbox_xyxy"][2]-c["bbox_xyxy"][0],
                                 "height": c["bbox_xyxy"][3]-c["bbox_xyxy"][1]},
                        "mask_polygon": None,
                        "area_px": c["area_px"], "length_px_est": c["length_px_est"],
                        "orientation_angle": c["orientation_angle"],
                    } for c in dedup_result["cracks"]],
                    processing_time=0, model_name="yolov8n-seg-cracks-joints",
                )
            except Exception:
                pass

            # ---- 每条裂缝的代表图 ----
            rep_images = []
            for crack in dedup_result["cracks"]:
                apps = crack.get("appearances", [])
                if apps:
                    best_app = max(apps, key=lambda a: a["confidence"])
                    rep_img_idx = best_app.get("image_idx", 0)
                    if rep_img_idx < len(annotated_urls):
                        rep_images.append(annotated_urls[rep_img_idx])

            # ---- 阶段3: AI 维修建议 ----
            advice = None
            advice_pdf = None
            try:
                advice_data, advice_path = generate_advice(str(report_path))
                risk = advice_data.get("risk_level", "未知")
                yield f'data: {{"事件":"AI建议","风险等级":"{risk}","保存至":"{advice_path}"}}\n\n'
                await asyncio.sleep(0.01)
                advice = advice_data
                pdf_path = generate_advice_pdf(advice_path)
                advice_pdf = f"/outputs/advice/{Path(pdf_path).name}"
            except RuntimeError as e:
                yield f'data: {{"事件":"AI建议","状态":"跳过","原因":"{str(e)[:100]}"}}\n\n'
                await asyncio.sleep(0.01)

            # ---- 最终结果 ----
            result = {
                "事件": "完成",
                "检测成功": True,
                "记录ID": detection_id,
                "墙面标识": wall_id,
                "检测汇总": {
                    "原始检测总数": dedup_result["raw_detection_count"],
                    "去重后唯一裂缝数": dedup_result["unique_crack_count"],
                    "已去除重复数": dedup_result["duplicate_removed_count"],
                },
                "裂缝列表": dedup_result["cracks"],
                "代表标注图": rep_images,
                "检测报告地址": f"/outputs/reports/{report_path.name}",
                "AI维修建议": advice,
                "维修建议PDF": advice_pdf,
                "全部标注图": annotated_urls,
            }
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f'data: {{"事件":"错误","信息":"{str(e)}"}}\n\n'

    return StreamingResponse(
        处理并推送进度(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==================== 历史记录 ====================

@app.get("/api/detections", tags=["历史记录"], summary="查询最近检测记录")
async def 最近检测记录(limit: int = 10):
    """获取数据库中最新的 N 条检测记录"""
    try:
        detections = db.get_recent_detections(limit=limit)
        return {
            '查询成功': True,
            '记录数量': len(detections),
            '记录列表': detections,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/api/detection/{detection_id}", tags=["历史记录"], summary="查询指定检测详情")
async def 检测记录详情(detection_id: int):
    """根据记录 ID 查询单次检测的完整信息（含每条裂缝的详情）"""
    try:
        result = db.get_detection(detection_id)
        if result is None:
            raise HTTPException(status_code=404, detail="该检测记录不存在")
        return {
            '查询成功': True,
            '数据': result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/api/statistics", tags=["历史记录"], summary="检测数据统计")
async def 检测统计():
    """汇总统计：总检测次数、总裂缝数、每日趋势等"""
    try:
        stats = db.get_statistics()
        return {
            '查询成功': True,
            '统计数据': stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
