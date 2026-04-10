#!/usr/bin/env python3
"""
SO101 相机实时监控（独立进程，port 7070 camera_head）

核心原则（与 rgbd_camera.py Monitor 保持一致）：
  - 主线程专用于 cv2.imshow + cv2.waitKey，稳定 30fps
  - CameraWrapper 在后台线程中负责 WebSocket + H.264 解码
  - 分辨率：原生相机输出，不做 resize（与 rgbd_camera.py 完全一致）

用法：
    python examples/so101/so101_camera_monitor.py
    python examples/so101/so101_camera_monitor.py --port 7070 --name camera_head --fps 30

按 'q' 或 ESC 退出，Ctrl+C 同样退出。
"""

import os
import sys
import signal
import time
import threading
import argparse
import numpy as np

# ★ STEP 1：MPLBACKEND=Agg 必须最先设置，防止 matplotlib 占用 Qt
os.environ.setdefault("MPLBACKEND", "Agg")

# ★ STEP 2：import cv2（在 av/websockets/rgbd_camera 之前）
import cv2

# ── 项目根目录 ────────────────────────────────────────────────────────────────
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ─────────────────────────────────────────────────────────────────────────────
# 独立 CameraWrapper（不依赖 rgbd_camera.py，避免模块级 import av 冲突）
# 行为与 rgbd_camera.py CameraWrapper 完全一致，额外增加线程锁和安全退出
# ─────────────────────────────────────────────────────────────────────────────
class CameraWrapper:
    def __init__(self, name: str, port: int):
        self._name = name
        self.port  = port
        # 初始黑帧（480×640），真实帧到来后替换
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.image_index = 0
        self.received_first_frame = False
        self.running = False
        self.thread  = None
        # 与 rgbd_camera.py 一致：不加锁，依赖 CPython GIL 保证引用替换的原子性

    @property
    def name(self):
        return self._name

    def is_first_frame_received(self):
        return self.received_first_frame

    def start(self):
        self.running = True
        self.thread  = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        # av/websockets 懒加载（在 cv2 窗口创建之后才 import）
        import asyncio
        asyncio.run(self._do_stuff())

    async def _do_stuff(self):
        import io
        import websockets
        import av

        uri = f"ws://localhost:{self.port}"
        print(f"[CameraWrapper:{self._name}] 连接 {uri}...", flush=True)
        while self.running:
            try:
                async with websockets.connect(uri) as ws:
                    print(f"[CameraWrapper:{self._name}] 已连接", flush=True)
                    cur_pos   = 0
                    raw_data  = io.BytesIO()
                    container = None
                    while self.running:
                        data  = await ws.recv()
                        data  = data[8:]           # 去掉 8 字节时间戳
                        raw_data.write(data)
                        raw_data.seek(cur_pos)
                        if cur_pos == 0:
                            container = av.open(raw_data, mode="r")
                        for packet in container.demux():
                            if packet.size == 0:
                                continue
                            for frame in packet.decode():
                                # 与 rgbd_camera.py 完全一致：直接赋值，无锁无拷贝
                                # CPython GIL 保证 self.image = ... 是原子引用替换
                                self.image = frame.to_ndarray(format="bgr24")
                                self.image_index += 1
                                if not self.received_first_frame:
                                    self.received_first_frame = True
                                    print(f"[CameraWrapper:{self._name}] 收到第一帧 "
                                          f"{self.image.shape}", flush=True)
                        cur_pos += len(data)
            except Exception as e:
                if self.running:
                    print(f"[CameraWrapper:{self._name}] 连接异常: {e}，1s 后重连...",
                          flush=True)
                    time.sleep(1.0)

    def get_frame(self) -> tuple:
        """返回 (bgr_frame_reference, index)，与 rgbd_camera.py 完全一致，无拷贝无锁。"""
        return self.image, self.image_index

    def stop(self):
        self.running = False
        # 不调用 asyncio.get_event_loop().stop()，避免影响主线程事件循环
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Monitor（与 rgbd_camera.py Monitor 逻辑完全对等）
# 主线程专用于显示，后台线程负责解码
# ─────────────────────────────────────────────────────────────────────────────
class Monitor:
    # 相机原生分辨率（与 rgbd_camera.py / OrcaSim 一致）
    NATIVE_W = 640
    NATIVE_H = 480
    # 显示缩放倍数：窗口缩至原始分辨率的 0.7×，由 OpenCV/GPU 负责缩放
    DISP_SCALE = 1.0
    DISP_W = int(NATIVE_W * DISP_SCALE)   # 448
    DISP_H = int(NATIVE_H * DISP_SCALE)   # 336

    def __init__(self, name: str, fps: float = 30.0, port: int = 7070):
        self.fps      = fps
        self.interval = 1.0 / fps
        self.running  = False
        self.win_name = f"Camera Monitor - {name} (Port {port})"
        self.camera   = CameraWrapper(name=name, port=port)

    def _init_window(self):
        """创建/重建窗口，防止最小化。每次调用先销毁旧窗口确保重建生效。"""
        import subprocess

        # ① 先销毁旧句柄（namedWindow 对已存在窗口是 no-op，必须先 destroy）
        try:
            cv2.destroyWindow(self.win_name)
            cv2.waitKey(1)
        except Exception:
            pass

        # ② 重新创建
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.win_name, self.DISP_W, self.DISP_H)

        # ③ OpenCV 属性置顶（不够可靠，但多加一层无妨）
        try:
            cv2.setWindowProperty(self.win_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

        # ④ 显示占位黑帧，触发窗口实际出现
        cv2.imshow(self.win_name, np.zeros((self.NATIVE_H, self.NATIVE_W, 3), np.uint8))
        cv2.waitKey(1)

        # ⑤ 通过 wmctrl 设置 X11 _NET_WM_STATE_ABOVE（最可靠的 Linux 置顶方式）
        #    wmctrl 安装：sudo apt install wmctrl
        try:
            time.sleep(0.15)    # 等窗口在 X11 注册
            subprocess.run(
                ["wmctrl", "-r", self.win_name, "-b", "add,above,sticky"],
                check=False, capture_output=True, timeout=1.0,
            )
        except (FileNotFoundError, Exception):
            pass    # wmctrl 未安装时静默忽略

    def start(self):
        # ★ STEP 3：先建窗口（此时 av 尚未被 import）
        self._init_window()

        # ★ STEP 4：窗口建好后再启动相机（内部懒加载 av）
        self.camera.start()

        self.running = True
        print(f"[Monitor] 等待第一帧...", flush=True)
        print(f"[Monitor] 显示窗口: {self.DISP_W}×{self.DISP_H}"
              f"（原始 {self.NATIVE_W}×{self.NATIVE_H}，"
              f"缩放 {self.DISP_SCALE:.1f}×）", flush=True)

        try:
            while self.running:
                # ── 最小化/关闭检测 ──────────────────────────────────────────
                try:
                    visible = cv2.getWindowProperty(self.win_name,
                                                    cv2.WND_PROP_VISIBLE)
                    if visible < 0:
                        # 窗口已被关闭（用户点了×），退出
                        break
                    if visible < 1:
                        # 被最小化：销毁并重建，同时重新执行 wmctrl 置顶
                        print("[Monitor] 检测到窗口最小化，重建...", flush=True)
                        self._init_window()
                except Exception:
                    pass

                # 在帧副本上画字，不污染原始缓冲区
                frame, idx = self.camera.get_frame()
                disp = frame.copy()
                info = f"Frame:{idx}  {self.fps:.0f}fps"
                cv2.putText(disp, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(self.win_name, disp)

                # waitKey 等待 interval ms，与 rgbd_camera.py Monitor 完全一致
                key = cv2.waitKey(int(self.interval * 1000)) & 0xFF
                if key == ord("q") or key == 27:   # q 或 ESC
                    print("[Monitor] 退出键", flush=True)
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self.running = False
        cv2.destroyAllWindows()
        self.camera.stop()
        print("[Monitor] 已停止", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SO101 相机实时监控")
    parser.add_argument("--port", type=int,   default=7090,
                        help="WebSocket 端口（默认 7090）")
    parser.add_argument("--name", type=str,   default="camera_wrist",
                        help="相机名称（默认 camera_wrist）")
    parser.add_argument("--fps",  type=float, default=30.0,
                        help="显示帧率（默认 30）")
    args = parser.parse_args()

    def _sigint_handler(sig, frame):
        print("\nCtrl+C，正在退出...", flush=True)
        cv2.destroyAllWindows()
        os._exit(0)
    signal.signal(signal.SIGINT, _sigint_handler)

    print(f"[Monitor] 相机: {args.name}  端口: {args.port}  帧率: {args.fps} fps")
    print(f"[Monitor] 原始分辨率: {Monitor.NATIVE_W}×{Monitor.NATIVE_H}  "
          f"显示窗口: {Monitor.DISP_W}×{Monitor.DISP_H}（{Monitor.DISP_SCALE:.1f}×）")
    print(f"[Monitor] 按 'q' 或 ESC 退出；窗口被最小化会自动还原\n")

    monitor = Monitor(name=args.name, fps=args.fps, port=args.port)
    monitor.start()


if __name__ == "__main__":
    main()
