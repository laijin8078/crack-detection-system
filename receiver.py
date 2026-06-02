# -*- coding: utf-8 -*-
"""
上位机 B - 检测结果接收程序（带界面）
监听 TCP 端口，接收 server.py 推送的报告和图片，界面展示并支持点击打开
"""

import socket
import struct
import json
import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

RECEIVE_DIR = "received_results"


def open_file(filepath):
    """用系统默认程序打开文件"""
    if not os.path.exists(filepath):
        messagebox.showerror("错误", f"文件不存在:\n{filepath}")
        return
    try:
        if sys.platform == "win32":
            os.startfile(filepath)
        elif sys.platform == "darwin":
            subprocess.run(["open", filepath])
        else:
            subprocess.run(["xdg-open", filepath])
    except Exception as e:
        messagebox.showerror("错误", f"无法打开文件:\n{e}")


class ReceiverApp:
    def __init__(self, host="0.0.0.0", port=9000):
        self.host = host
        self.port = port
        self.server_running = True
        self.current_meta = None
        self.received_files = []
        self.history = []  # 历史接收记录

        os.makedirs(RECEIVE_DIR, exist_ok=True)

        # ===== 主窗口 =====
        self.root = tk.Tk()
        self.root.title("上位机B - 裂缝检测结果接收")
        self.root.geometry("780x560")
        self.root.configure(bg="#f0f0f0")

        self._build_ui()

        # ===== 后台 TCP 监听线程 =====
        self.tcp_thread = threading.Thread(target=self._tcp_listen, daemon=True)
        self.tcp_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # 顶部：状态栏
        top_frame = tk.Frame(self.root, bg="#2c3e50", height=50)
        top_frame.pack(fill=tk.X)
        top_frame.pack_propagate(False)

        self.status_label = tk.Label(
            top_frame, text=f"监听中  |  端口: {self.port}  |  等待 server 推送结果...",
            fg="white", bg="#2c3e50", font=("Microsoft YaHei", 12)
        )
        self.status_label.pack(pady=12)

        # 中部：摘要信息
        info_frame = tk.LabelFrame(self.root, text=" 检测摘要 ", font=("Microsoft YaHei", 10, "bold"), bg="#f0f0f0")
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.info_text = tk.Text(info_frame, height=6, font=("Microsoft YaHei", 11),
                                 bg="white", relief=tk.FLAT, state=tk.DISABLED)
        self.info_text.pack(fill=tk.X, padx=5, pady=5)

        # 中部：文件列表
        list_frame = tk.LabelFrame(self.root, text=" 接收文件（双击打开） ", font=("Microsoft YaHei", 10, "bold"), bg="#f0f0f0")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        columns = ("#", "文件名", "大小", "类型")
        self.file_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=12)
        self.file_tree.heading("#", text="#", command=lambda: None)
        self.file_tree.heading("文件名", text="文件名")
        self.file_tree.heading("大小", text="大小")
        self.file_tree.heading("类型", text="类型")
        self.file_tree.column("#", width=40, anchor=tk.CENTER)
        self.file_tree.column("文件名", width=340)
        self.file_tree.column("大小", width=100, anchor=tk.CENTER)
        self.file_tree.column("类型", width=80, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=scrollbar.set)
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.file_tree.bind("<Double-1>", self._on_file_double_click)

        # 底部：历史记录 + 按钮
        bottom_frame = tk.Frame(self.root, bg="#f0f0f0")
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)

        self.history_label = tk.Label(bottom_frame, text="历史记录: 暂无", font=("Microsoft YaHei", 9),
                                      fg="#888", bg="#f0f0f0", anchor=tk.W)
        self.history_label.pack(side=tk.LEFT, padx=5)

        btn_frame = tk.Frame(bottom_frame, bg="#f0f0f0")
        btn_frame.pack(side=tk.RIGHT)

        tk.Button(btn_frame, text="打开文件夹", font=("Microsoft YaHei", 9),
                  command=self._open_receive_dir, width=12).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="清空列表", font=("Microsoft YaHei", 9),
                  command=self._clear_list, width=10).pack(side=tk.LEFT, padx=3)

    def _tcp_listen(self):
        """后台线程：TCP 监听"""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((self.host, self.port))
            srv.listen(5)
        except OSError as e:
            self._update_status(f"端口 {self.port} 被占用: {e}")
            return

        self._update_status(f"监听中  |  端口: {self.port}  |  等待 server 推送结果...")

        while self.server_running:
            try:
                srv.settimeout(1)
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except Exception:
                break

            self._update_status(f"接收中...  |  来自: {addr[0]}")
            try:
                self._handle_connection(conn)
            except Exception as e:
                self._update_status(f"接收失败: {e}")
            finally:
                conn.close()

        srv.close()

    def _handle_connection(self, conn):
        """处理一次 TCP 连接，解析协议并保存/展示"""
        conn.settimeout(30)

        # 1. 读元数据
        meta = self._recv_meta(conn)
        self.current_meta = meta
        self.received_files = []

        wall_id = meta.get("wall_id", "未知")
        summary = meta.get("summary", {})
        file_count = meta.get("file_count", 0)

        wall_dir = os.path.join(RECEIVE_DIR, wall_id)
        os.makedirs(wall_dir, exist_ok=True)

        # 更新界面摘要
        self.root.after(0, self._show_summary, meta)

        # 2. 逐个接收文件
        for i in range(file_count):
            filename, data = self._recv_file(conn)
            if filename is None:
                break

            save_path = os.path.join(wall_dir, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(data)

            self.received_files.append((filename, len(data), save_path))
            self.root.after(0, self._add_file_row, i + 1, filename, len(data), filename.split(".")[-1].upper())

        # 确认结束标记
        try:
            end = struct.unpack(">I", self._recv_exact(conn, 4))[0]
        except Exception:
            end = -1

        timestamp = datetime.now().strftime("%H:%M:%S")
        total = summary.get("total_unique_cracks", 0)
        risk = summary.get("risk_level", "N/A")
        self.history.append(f"[{timestamp}] {wall_id} | 裂缝:{total} | 风险:{risk} | 文件:{len(self.received_files)}")

        self.root.after(0, self._on_receive_done, wall_id)

    def _recv_meta(self, conn):
        meta_len = struct.unpack(">I", self._recv_exact(conn, 4))[0]
        return json.loads(self._recv_exact(conn, meta_len).decode("utf-8"))

    def _recv_file(self, conn):
        """接收一个文件，返回 (filename, data) 或 (None, None) 表示结束"""
        name_len = struct.unpack(">I", self._recv_exact(conn, 4))[0]
        if name_len == 0:
            return None, None  # 结束标记
        filename = self._recv_exact(conn, name_len).decode("utf-8")
        data_len = struct.unpack(">I", self._recv_exact(conn, 4))[0]
        data = self._recv_exact(conn, data_len)
        return filename, data

    @staticmethod
    def _recv_exact(sock, n):
        data = b""
        while len(data) < n:
            chunk = sock.recv(min(n - len(data), 65536))
            if not chunk:
                raise ConnectionError("连接断开")
            data += chunk
        return data

    # ===== GUI 更新方法（主线程调用） =====

    def _update_status(self, text):
        self.root.after(0, lambda: self.status_label.config(text=text))

    def _show_summary(self, meta):
        summary = meta.get("summary", {})
        lines = [
            f"墙面标识:   {meta.get('wall_id', '未知')}",
            f"接收时间:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"裂缝总数:   {summary.get('total_unique_cracks', 0)} 条",
            f"风险等级:   {summary.get('risk_level', 'N/A')}",
            f"接收文件:   {meta.get('file_count', 0)} 个",
        ]
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", "\n".join(lines))
        self.info_text.config(state=tk.DISABLED)

        # 清空文件列表
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)

    def _add_file_row(self, idx, filename, size, ftype):
        size_str = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"
        self.file_tree.insert("", tk.END, values=(idx, filename, size_str, ftype))

    def _on_receive_done(self, wall_id):
        self._update_status(f"接收完成  |  端口: {self.port}  |  {wall_id} - {len(self.received_files)} 个文件")
        self._update_history_label()

    def _on_file_double_click(self, event):
        selection = self.file_tree.selection()
        if not selection:
            return
        values = self.file_tree.item(selection[0], "values")
        filename = values[1]

        # 在 received_files 中查找完整路径
        for fname, _, fullpath in self.received_files:
            if fname == filename:
                open_file(fullpath)
                return

        # 兜底：用当前 wall_id 查找
        if self.current_meta:
            wall_dir = os.path.join(RECEIVE_DIR, self.current_meta.get("wall_id", ""))
            fullpath = os.path.join(wall_dir, filename)
            open_file(fullpath)

    def _open_receive_dir(self):
        if self.current_meta:
            wall_dir = os.path.join(RECEIVE_DIR, self.current_meta.get("wall_id", ""))
        else:
            wall_dir = RECEIVE_DIR
        open_file(wall_dir)

    def _clear_list(self):
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        self.received_files = []

    def _update_history_label(self):
        recent = self.history[-3:] if len(self.history) > 3 else self.history
        self.history_label.config(text=" | ".join(recent))

    def _on_close(self):
        self.server_running = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="上位机B - 检测结果接收程序（带界面）")
    parser.add_argument("--port", type=int, default=9000, help="监听端口 (默认 9000)")
    args = parser.parse_args()

    app = ReceiverApp(port=args.port)
    app.run()
