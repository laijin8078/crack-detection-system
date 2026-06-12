# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, Response, send_file
import os
import json
import time
import socket
import struct
import threading
import zipfile
import io
import re
from datetime import datetime

app = Flask(__name__)

# 基于脚本所在目录的绝对路径，避免工作目录不一致
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
AUTO_UPLOAD_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, "auto")
MANUAL_UPLOAD_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, "manual")

os.makedirs(AUTO_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MANUAL_UPLOAD_FOLDER, exist_ok=True)

RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
PROCESSED_IMAGES_FOLDER = os.path.join(RESULTS_FOLDER, "processed_images")
REPORTS_FOLDER = os.path.join(RESULTS_FOLDER, "reports")
os.makedirs(PROCESSED_IMAGES_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# 流水线状态共享字典：{ wall_id: {"status": "processing|done|error", "files": [...], "summary": {...}} }
processing_status = {}

# TCP 推送目标（上位机B），启动时可通过命令行参数覆盖
TARGET_HOST = os.environ.get("RESULT_TARGET_HOST", "")
TARGET_PORT = int(os.environ.get("RESULT_TARGET_PORT", "0"))


STANDARD_PDF_NAME_RE = re.compile(r"^\d{3,}_pdf_\d{14}_\d+\.pdf$", re.IGNORECASE)


def find_order_serial(*values):
    for value in values:
        text = str(value or "")
        for pattern in (r"工单\s*([0-9]{3,})", r"([0-9]{3,})_pdf_\d{14}_\d+\.pdf"):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
    return "000"


def iter_upload_names():
    for folder in (AUTO_UPLOAD_FOLDER, MANUAL_UPLOAD_FOLDER):
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            yield name


def build_tcp_pdf_names(wall_id, pdf_files):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    order_serial = find_order_serial(
        wall_id,
        *(os.path.basename(fpath) for fpath in pdf_files),
        *iter_upload_names(),
    )
    return {
        fpath: f"{order_serial}_pdf_{timestamp}_{idx}.pdf"
        for idx, fpath in enumerate(pdf_files, 1)
    }


def send_results_via_tcp(wall_id):
    """
    通过 TCP 将 PDF 报告推送给上位机 B。

    协议需与 receiver.py 保持一致：
      [4字节 meta JSON长度] + [meta JSON]
      循环 file_count 次：
        [4字节 文件名长度] + [文件名 UTF-8]
        [4字节 文件内容长度] + [文件二进制内容]
      [4字节 0] 作为结束标记
    """
    info = processing_status.get(wall_id, {})
    all_files = info.get("files", [])

    if not TARGET_HOST or not TARGET_PORT:
        print(f"[tcp] 未配置目标，跳过推送")
        return

    # 只发 PDF
    pdf_files = [f for f in all_files if f.lower().endswith(".pdf") and os.path.isfile(f)]
    if not pdf_files:
        print("[tcp] 没有 PDF 文件，跳过推送")
        return

    tcp_pdf_names = build_tcp_pdf_names(wall_id, pdf_files)

    print(f"[tcp] 连接上位机B {TARGET_HOST}:{TARGET_PORT}, 共 {len(pdf_files)} 个 PDF")

    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(30)
        sock.connect((TARGET_HOST, TARGET_PORT))
        print(f"[tcp] 已连接")

        meta = {
            "wall_id": wall_id,
            "summary": info.get("summary") or {},
            "file_count": len(pdf_files),
            "files": [
                {
                    "filename": tcp_pdf_names[fpath],
                    "source_filename": os.path.basename(fpath),
                    "size": os.path.getsize(fpath),
                    "type": "pdf",
                }
                for fpath in pdf_files
            ],
        }
        meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        sock.sendall(struct.pack(">I", len(meta_bytes)))
        sock.sendall(meta_bytes)
        total = 4 + len(meta_bytes)
        print(f"[tcp] 已发送元数据: {len(meta_bytes)}字节, file_count={len(pdf_files)}")

        for i, fpath in enumerate(pdf_files, 1):
            filename = tcp_pdf_names[fpath]
            name_bytes = filename.encode("utf-8")
            with open(fpath, "rb") as f:
                content = f.read()

            print(f"[tcp] 发送第{i}个: name={filename}, "
                  f"content={len(content)}字节 (0x{len(content):08X})")
            sock.sendall(struct.pack(">I", len(name_bytes)))
            sock.sendall(name_bytes)
            sock.sendall(struct.pack(">I", len(content)))
            sock.sendall(content)
            total += 4 + len(name_bytes) + 4 + len(content)

        sock.sendall(struct.pack(">I", 0))
        total += 4

        print(f"[tcp] 全部发送完成, 共{len(pdf_files)}个PDF, 总计{total}字节")

    except Exception as e:
        print(f"[tcp] 推送失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if sock:
            sock.close()


def cleanup_uploads():
    """清理上一次传输的所有图片"""
    print(f"[cleanup] 开始清理，auto目录: {AUTO_UPLOAD_FOLDER}")
    print(f"[cleanup] manual目录: {MANUAL_UPLOAD_FOLDER}")
    for folder in [AUTO_UPLOAD_FOLDER, MANUAL_UPLOAD_FOLDER]:
        if os.path.exists(folder):
            files = os.listdir(folder)
            print(f"[cleanup] {folder} 中有 {len(files)} 个文件")
            for f in files:
                path = os.path.join(folder, f)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        print(f"[cleanup] 已删除: {f}")
                except Exception as e:
                    print(f"[cleanup] 删除失败 {path}: {e}")
        else:
            print(f"[cleanup] 目录不存在: {folder}")
    print("[cleanup] 清理完成")


def run_pipeline_async(wall_id):
    """在后台线程中运行检测流水线，完成后通过 TCP 推送给上位机 B"""
    print(f"[pipeline] 准备启动后台流水线, wall_id={wall_id}")
    processing_status[wall_id] = {"status": "processing", "files": [], "summary": None}
    try:
        from pipeline import process_uploads
        print(f"[pipeline] 成功导入 pipeline.process_uploads")
        process_uploads(wall_id, status_dict=processing_status)
        print(f"[pipeline] 流水线完成, status={processing_status.get(wall_id, {}).get('status')}")

        # 流水线完成后，通过 TCP 推送给上位机 B
        info = processing_status.get(wall_id, {})
        if info.get("status") == "done" and info.get("files"):
            print(f"[pipeline] 开始 TCP 推送结果到上位机 B...")
            send_results_via_tcp(wall_id)
    except Exception as e:
        import traceback
        print(f"[pipeline] 后台处理失败: {e}")
        traceback.print_exc()
        processing_status[wall_id] = {"status": "error", "error": str(e), "files": []}


# ==================== API 端点 ====================

@app.route('/api/detect-sequence', methods=['POST'])
def receive_images():
    print(f"\n{'='*60}")
    print(f"收到请求时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    wall_id = request.form.get('wall_id', '未命名')
    print(f"墙面标识：{wall_id}")

    file_data_list = []
    files = request.files.getlist('files')
    print(f"收到图片数量：{len(files)}")

    if len(files) == 0:
        return jsonify({
            "status": "error",
            "message": "没有收到任何图片文件"
        }), 400

    # 传输前清理上一次的图片
    cleanup_uploads()

    for file in files:
        if file.filename == '':
            continue
        file_data = file.read()
        file_data_list.append({
            "name": file.filename,
            "data": file_data
        })

    def generate():
        saved_count = 0
        total = len(file_data_list)

        for idx, item in enumerate(file_data_list):
            try:
                filename = f"{wall_id}_{item['name']}"

                if any(key in item['name'] for key in ['起点', '终点', '途径点']):
                    save_path = os.path.join(AUTO_UPLOAD_FOLDER, filename)
                    file_type = "自动截图"
                else:
                    save_path = os.path.join(MANUAL_UPLOAD_FOLDER, filename)
                    file_type = "手动截图"

                with open(save_path, 'wb') as f:
                    f.write(item['data'])

                saved_count += 1
                print(f"保存成功【{file_type}】：{filename} ({idx + 1}/{total})")

            except Exception as e:
                print(f"保存失败：{item['name']} - {str(e)}")
                continue

            time.sleep(0.1)
            result = {
                "status": "success",
                "index": idx + 1,
                "total": total,
                "filename": filename,
                "wall_id": wall_id,
                "num_cracks": 0,
                "message": f"图片{idx + 1}处理完成"
            }
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

        final_result = {
            "status": "complete",
            "wall_id": wall_id,
            "total_received": total,
            "total_saved": saved_count,
            "message": f"传输完成，成功保存{saved_count}张图片，后台开始检测处理",
            "result_query": f"/api/result-status?wall_id={wall_id}",
            "result_download": f"/api/download-results?wall_id={wall_id}"
        }
        yield f"data: {json.dumps(final_result, ensure_ascii=False)}\n\n"

        print(f"\n所有图片处理完成，共保存 {saved_count} 张")
        print(f"自动截图保存至：{AUTO_UPLOAD_FOLDER}")
        print(f"手动截图保存至：{MANUAL_UPLOAD_FOLDER}")
        print(f"{'='*60}\n")

        # 启动后台检测流水线
        if saved_count > 0:
            print(f"[server] 所有图片保存完毕(saved_count={saved_count}), 启动后台流水线...")
            t = threading.Thread(target=run_pipeline_async, args=(wall_id,), daemon=True)
            t.start()
            print(f"[server] 后台线程已启动, thread={t.name}")
        else:
            print(f"[server] saved_count=0, 不启动流水线")

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/result-status', methods=['GET'])
def result_status():
    """查询流水线处理状态"""
    wall_id = request.args.get('wall_id', '')
    if not wall_id:
        # 返回所有 wall_id 的状态
        return jsonify({"status": "ok", "jobs": processing_status})

    info = processing_status.get(wall_id)
    if info is None:
        return jsonify({"status": "not_found", "message": f"未找到 wall_id={wall_id} 的处理记录"})

    return jsonify({
        "wall_id": wall_id,
        "processing_status": info["status"],
        "summary": info.get("summary"),
        "file_count": len(info.get("files", [])),
        "files": [os.path.relpath(f, RESULTS_FOLDER) for f in info.get("files", [])],
    })


@app.route('/api/download-results', methods=['GET'])
def download_results():
    """将所有结果打包为 zip 下载"""
    wall_id = request.args.get('wall_id', '')
    if not wall_id:
        return jsonify({"status": "error", "message": "缺少 wall_id 参数"}), 400

    info = processing_status.get(wall_id)
    if info is None:
        return jsonify({"status": "error", "message": f"未找到 wall_id={wall_id} 的处理记录"}), 404

    if info["status"] == "processing":
        return jsonify({"status": "pending", "message": "处理尚未完成，请稍后再试"}), 202

    if info["status"] == "error":
        return jsonify({"status": "error", "message": info.get("error", "处理失败")}), 500

    # 打包所有文件为 zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fpath in info.get("files", []):
            if os.path.isfile(fpath):
                arcname = os.path.relpath(fpath, RESULTS_FOLDER)
                zf.write(fpath, arcname)

    buf.seek(0)
    zip_name = f"results_{wall_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return send_file(
        buf,
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_name
    )


@app.route('/api/download-file', methods=['GET'])
def download_file():
    """下载单个结果文件"""
    rel_path = request.args.get('path', '')
    if not rel_path:
        return jsonify({"status": "error", "message": "缺少 path 参数"}), 400

    # 安全检查：防止目录穿越
    full_path = os.path.normpath(os.path.join(RESULTS_FOLDER, rel_path))
    if not full_path.startswith(os.path.normpath(RESULTS_FOLDER)):
        return jsonify({"status": "error", "message": "非法路径"}), 403

    if not os.path.isfile(full_path):
        return jsonify({"status": "error", "message": f"文件不存在: {rel_path}"}), 404

    return send_file(full_path, as_attachment=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="建筑裂缝检测服务端")
    parser.add_argument("--port", type=int, default=8000, help="HTTP 服务端口 (默认 8000)")
    parser.add_argument("--target-host", type=str, default="", help="上位机B的IP地址（结果推送目标）")
    parser.add_argument("--target-port", type=int, default=0, help="上位机B的TCP端口（结果推送目标）")
    args = parser.parse_args()

    if args.target_host:
        TARGET_HOST = args.target_host
    if args.target_port:
        TARGET_PORT = args.target_port

    print("服务器启动")
    print(f"  HTTP 端口: {args.port}")
    print(f"  自动截图: {AUTO_UPLOAD_FOLDER}")
    print(f"  手动截图: {MANUAL_UPLOAD_FOLDER}")
    print(f"  检测结果: {RESULTS_FOLDER}")
    if TARGET_HOST and TARGET_PORT:
        print(f"  结果推送: {TARGET_HOST}:{TARGET_PORT} (TCP)")
    else:
        print(f"  结果推送: 未配置（使用 --target-host --target-port 指定）")
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
