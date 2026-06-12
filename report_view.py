# -*- coding: utf-8 -*-
"""
接收地址：192.168.137.118:8888
保存目录：C:/Users/11/Desktop/car_detect_final/reports  # 改为正斜杠避免转义
完全对齐标准recv_all循环读逻辑，支持接收任意数量PDF，适配TCP_NODELAY
修改：保存文件时自动加上首页中正在执行的工单编号（如 001_pdf_...）
"""

import socket
import struct
import os
import sys
import json
import re
from datetime import datetime
from PySide6.QtWidgets import *
from PySide6.QtCore import *

STANDARD_PDF_NAME_RE = re.compile(r"^\d{3,}_pdf_\d{14}_\d+\.pdf$", re.IGNORECASE)


class PDFReceiverThread(QThread):
    pdf_received = Signal(str)
    log_msg = Signal(str)
    finished = Signal()

    def __init__(self, listen_port=8888, save_dir=r"C:\Users\11\Desktop\car_detect_final\reports", order_serial=None):
        super().__init__()
        self.listen_port = listen_port
        self.save_dir = save_dir
        self.order_serial = order_serial   # 工单编号（如 "001"），None 表示不加
        self.running = True
        os.makedirs(self.save_dir, exist_ok=True)
        self.MAX_FILE_SIZE = 200 * 1024 * 1024
        self.MAX_META_SIZE = 1024 * 1024

    def recv_all(self, sock, n):
        data = b""
        while len(data) < n:
            chunk = sock.recv(min(n - len(data), 65536))
            if not chunk:
                raise ConnectionError(f"connection closed while receiving {n} bytes, got {len(data)}")
            data += chunk
        return data

    def recv_u32(self, sock):
        return struct.unpack(">I", self.recv_all(sock, 4))[0]

    def next_payload_looks_like_json(self, sock):
        marker = sock.recv(1, socket.MSG_PEEK)
        return marker in (b"{", b"[")

    def safe_filename(self, filename):
        filename = os.path.basename(filename or "")
        if not filename:
            filename = f"pdf_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        return filename

    def add_order_prefix(self, filename):
        if self.order_serial and not filename.startswith(f"{self.order_serial}_"):
            return f"{self.order_serial}_{filename}"
        return filename

    def resolve_order_serial(self, source_filename=""):
        if self.order_serial:
            return str(self.order_serial)

        for pattern in (r"工单\s*([0-9]{3,})", r"([0-9]{3,})_pdf_\d{14}_\d+\.pdf"):
            match = re.search(pattern, source_filename or "", re.IGNORECASE)
            if match:
                return match.group(1)

        return "000"

    def build_pdf_filename(self, source_filename, file_index, timestamp):
        source_filename = self.safe_filename(source_filename)
        if STANDARD_PDF_NAME_RE.match(source_filename):
            return source_filename

        order_serial = self.resolve_order_serial(source_filename)
        return f"{order_serial}_pdf_{timestamp}_{file_index}.pdf"

    def save_pdf(self, filename, file_content, file_index=1, timestamp=None):
        timestamp = timestamp or datetime.now().strftime('%Y%m%d%H%M%S')
        final_name = self.build_pdf_filename(filename, file_index, timestamp)
        save_path = os.path.join(self.save_dir, final_name)
        with open(save_path, "wb") as f:
            f.write(file_content)
        self.pdf_received.emit(save_path)
        return save_path

    def receive_new_protocol(self, conn, meta_len):
        meta = json.loads(self.recv_all(conn, meta_len).decode("utf-8"))
        file_count = int(meta.get("file_count", 0))
        wall_id = meta.get("wall_id", "")
        self.log_msg.emit(f"解析到服务器元数据: wall_id={wall_id}, file_count={file_count}")

        received_count = 0
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        for index in range(1, file_count + 1):
            self.log_msg.emit(f"\n---------- 开始接收第 {index} 个 PDF ----------")

            name_len = self.recv_u32(conn)
            if name_len == 0:
                self.log_msg.emit("收到结束标记，停止接收")
                break
            if name_len > 4096:
                raise ValueError(f"文件名长度异常: {name_len}")

            filename = self.recv_all(conn, name_len).decode("utf-8", errors="replace")
            file_length = self.recv_u32(conn)
            self.log_msg.emit(f"文件名: {filename}")
            self.log_msg.emit(f"解析文件长度: {file_length} 字节")

            if file_length <= 0 or file_length > self.MAX_FILE_SIZE:
                raise ValueError(f"文件大小异常: {file_length} 字节")

            file_content = self.recv_all(conn, file_length)
            save_path = self.save_pdf(filename, file_content, index, timestamp)
            received_count += 1
            self.log_msg.emit(f"第 {index} 个文件接收完成，保存路径: {save_path}")

        try:
            end_mark = self.recv_u32(conn)
            if end_mark != 0:
                self.log_msg.emit(f"警告: 结束标记不是 0，而是 {end_mark}")
        except Exception as e:
            self.log_msg.emit(f"警告: 未读到结束标记: {e}")

        return received_count

    def receive_legacy_protocol(self, conn, first_file_length):
        current_file = 0
        file_length = first_file_length
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        while True:
            current_file += 1
            self.log_msg.emit(f"\n---------- 开始接收第 {current_file} 个 PDF(旧协议) ----------")
            self.log_msg.emit(f"解析文件长度: {file_length} 字节")

            if file_length <= 0 or file_length > self.MAX_FILE_SIZE:
                raise ValueError(f"文件大小异常: {file_length} 字节")

            file_content = self.recv_all(conn, file_length)
            base_name = f"pdf_{datetime.now().strftime('%Y%m%d%H%M%S')}_{current_file}.pdf"
            save_path = self.save_pdf(base_name, file_content, current_file, timestamp)
            self.log_msg.emit(f"第 {current_file} 个文件接收完成，保存路径: {save_path}")

            try:
                file_length = self.recv_u32(conn)
            except Exception:
                break

        return current_file

    def run(self):
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind(("0.0.0.0", self.listen_port))
            server_sock.listen(1)
            server_sock.settimeout(1)

            self.log_msg.emit("========================================")
            self.log_msg.emit("✅ 接收服务启动成功")
            self.log_msg.emit(f"📍 接收地址：192.168.137.118:{self.listen_port}")
            self.log_msg.emit("📄 支持接收任意数量 PDF")
            self.log_msg.emit(f"📍 保存目录：{self.save_dir}")
            if self.order_serial:
                self.log_msg.emit(f"🏷️  当前工单编号：{self.order_serial}（将添加至文件名前缀）")
            else:
                self.log_msg.emit("⚠️  未检测到正在执行的工单，文件名将不加前缀")
            self.log_msg.emit("========================================")

            while self.running:
                try:
                    conn, client_addr = server_sock.accept()
                    conn.settimeout(30)
                    # 开启TCP_NODELAY禁用Nagle算法，解决小包合并导致的延迟问题
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.log_msg.emit(f"🔗 客户端已连接：{client_addr}")

                    first_len = self.recv_u32(conn)
                    if 0 < first_len <= self.MAX_META_SIZE and self.next_payload_looks_like_json(conn):
                        received = self.receive_new_protocol(conn, first_len)
                    else:
                        self.log_msg.emit("未检测到元数据，按旧协议接收 PDF")
                        received = self.receive_legacy_protocol(conn, first_len)
                    self.log_msg.emit(f"本次连接接收完成，共 {received} 个文件")
                    conn.close()
                    continue

                    current_file = 0
                    while True:
                        current_file += 1
                        self.log_msg.emit(f"\n---------- 开始接收第 {current_file} 个 PDF ----------")

                        header_data = self.recv_all(conn, 4)
                        if len(header_data) != 4:
                            self.log_msg.emit(f"ℹ️ 对方已断开连接，本次共接收 {current_file-1} 个文件")
                            break

                        file_length = struct.unpack(">I", header_data)[0]
                        self.log_msg.emit(f"📏 解析文件长度：{file_length} 字节")

                        if file_length <= 0 or file_length > self.MAX_FILE_SIZE:
                            self.log_msg.emit(f"❌ 第{current_file}个文件：文件大小异常，丢弃数据")
                            continue

                        file_content = self.recv_all(conn, file_length)
                        if len(file_content) != file_length:
                            self.log_msg.emit(f"❌ 第{current_file}个文件：文件内容接收不完整")
                            continue

                        # 生成基础文件名：pdf_20260602120000_1.pdf
                        base_name = f"pdf_{datetime.now().strftime('%Y%m%d%H%M%S')}_{current_file}.pdf"
                        # 如果存在工单编号，则添加前缀
                        if self.order_serial:
                            final_name = f"{self.order_serial}_{base_name}"
                        else:
                            final_name = base_name
                        save_path = os.path.join(self.save_dir, final_name)

                        with open(save_path, "wb") as f:
                            f.write(file_content)

                        self.pdf_received.emit(save_path)
                        self.log_msg.emit(f"✅ 第{current_file}个文件接收完成，保存路径：{save_path}")

                    conn.close()

                except socket.timeout:
                    continue
                except Exception as e:
                    self.log_msg.emit(f"⚠️ 连接异常：{str(e)}")

        except Exception as e:
            self.log_msg.emit(f"❌ 服务启动失败：{str(e)}")
        finally:
            if 'server_sock' in locals() and server_sock:
                server_sock.close()
            self.finished.emit()

    def stop(self):
        self.running = False
        self.wait()


class ReportViewWidget(QWidget):
    def __init__(self):
        super().__init__()
        # 统一报告目录，使用原始字符串r前缀避免转义
        self.save_dir = r"C:\Users\11\Desktop\car_detect_final\reports"
        self.listen_port = 8888
        self.local_ip = "192.168.137.118"
        self.current_file_path = None
        self.work_thread = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("PDF 接收工具（自动加工单编号）")
        self.setStyleSheet("background-color:#ffffff;")
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(12)

        title_label = QLabel("📄 PDF 文件接收工具")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size:22px; font-weight:bold; color:#2c3e50;")
        main_layout.addWidget(title_label)

        addr_tip = QLabel(f"本机接收地址：{self.local_ip}:{self.listen_port}\n支持接收任意数量 PDF 文件\n保存目录：{self.save_dir}")
        addr_tip.setAlignment(Qt.AlignCenter)
        addr_tip.setStyleSheet("font-size:15px; color:#27ae60; padding:10px; border:1px solid #dddddd; border-radius:8px;")
        main_layout.addWidget(addr_tip)

        self.btn_start = QPushButton("▶ 开始监听")
        self.btn_stop = QPushButton("■ 停止监听")
        self.btn_open_file = QPushButton("📄 打开最新PDF")
        self.btn_open_folder = QPushButton("📂 打开保存文件夹")
        self.btn_clear_log = QPushButton("🧹 清空日志")

        btn_style = """
        QPushButton{
            font-size:14px;
            padding:9px;
            border-radius:6px;
            background-color:#3498db;
            color:#ffffff;
        }
        QPushButton:disabled{
            background-color:#95a5a6;
        }
        """
        for btn in [self.btn_start, self.btn_stop, self.btn_open_file, self.btn_open_folder, self.btn_clear_log]:
            btn.setStyleSheet(btn_style)
            main_layout.addWidget(btn)

        self.btn_stop.setEnabled(False)
        self.btn_open_file.setEnabled(False)

        log_title = QLabel("运行日志")
        log_title.setStyleSheet("font-size:14px; font-weight:bold;")
        main_layout.addWidget(log_title)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        main_layout.addWidget(self.log_edit)

        self.file_path_label = QLabel("当前无接收文件")
        self.file_path_label.setStyleSheet("background:#f8f9fa; padding:8px; border-radius:6px;")
        main_layout.addWidget(self.file_path_label)

        self.setLayout(main_layout)
        self.resize(850, 680)

        self.btn_start.clicked.connect(self.start_listen)
        self.btn_stop.clicked.connect(self.stop_listen)
        self.btn_open_file.clicked.connect(self.open_latest_file)
        self.btn_open_folder.clicked.connect(self.open_save_folder)
        self.btn_clear_log.clicked.connect(self.log_edit.clear)

    def get_current_order_serial(self):
        """
        向上查找父窗口，获取 CarController 中的 active_task_order_serial
        若未找到或工单未开始，返回 None
        """
        parent = self.parent()
        while parent:
            if hasattr(parent, 'active_task_order_serial'):
                serial = parent.active_task_order_serial
                if serial:
                    return serial
                else:
                    return None
            parent = parent.parent()
        return None

    def start_listen(self):
        if self.work_thread and self.work_thread.isRunning():
            return

        # 获取当前正在执行的工单编号（用于文件名前缀）
        order_serial = self.get_current_order_serial()
        if order_serial:
            self.log_edit.append(f"🏷️  检测到当前工单编号：{order_serial}，后续接收的文件将自动添加此前缀")
        else:
            self.log_edit.append("⚠️  未检测到正在执行的工单，接收的文件将不加编号前缀")

        # 确保目录存在
        os.makedirs(self.save_dir, exist_ok=True)

        self.work_thread = PDFReceiverThread(self.listen_port, self.save_dir, order_serial)
        self.work_thread.log_msg.connect(self.log_edit.append)
        self.work_thread.pdf_received.connect(self.on_file_received)
        self.work_thread.finished.connect(lambda: self.log_edit.append("✅ 监听服务已停止"))
        self.work_thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_edit.append("⏳ 监听已启动，等待客户端连接...")

    def stop_listen(self):
        if self.work_thread:
            self.work_thread.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def on_file_received(self, file_path):
        self.current_file_path = file_path
        self.file_path_label.setText(f"最新文件：{file_path}")
        self.btn_open_file.setEnabled(True)
        QMessageBox.information(self, "接收成功", f"文件已成功保存：\n{file_path}")

    def open_latest_file(self):
        if self.current_file_path and os.path.isfile(self.current_file_path):
            try:
                if sys.platform == "win32":
                    os.startfile(self.current_file_path)
                else:
                    import subprocess
                    subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", self.current_file_path])
            except Exception as e:
                QMessageBox.warning(self, "打开失败", f"无法打开文件：{str(e)}")
        else:
            QMessageBox.warning(self, "文件不存在", "最新接收的PDF文件不存在，请检查保存路径！")

    def open_save_folder(self):
        os.startfile(self.save_dir)

    def closeEvent(self, event):
        self.stop_listen()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ReportViewWidget()
    win.show()
    sys.exit(app.exec())
