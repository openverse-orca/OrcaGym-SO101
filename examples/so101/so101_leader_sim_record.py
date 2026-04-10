#!/usr/bin/env python3
"""
SO101 主臂遥操作仿真数据采集脚本

功能：
  - 物理主臂（SO101 Leader）→ 仿真从臂（OrcaGym SO101）
  - 双路相机图像（camera_global:7070, camera_wrist:7080）写入 LeRobot 数据集
  - 数据保存为 LeRobot 数据集格式（兼容 LeRobot 训练流程）

实时相机监控请另开终端运行：
    python examples/so101/so101_camera_monitor.py

运行步骤：
  1. 连接主臂 USB-TTL 串口（默认 /dev/ttyACM1）
  2. 启动 OrcaStudio 并运行 SO101 场景
  3. conda activate so101
  4. （可选）另开终端：python examples/so101/so101_camera_monitor.py
  5. python examples/so101/so101_leader_sim_record.py \\
         --repo_id your_name/so101_sim_dataset \\
         --task "pick up the cube"

键盘控制（采集过程中）：
  Page Down → 结束当前 episode 并保存
  Page Up   → 丢弃当前 episode，重新录制
  ESC       → 停止全部采集，保存已有数据
"""

import os
import sys
import signal
import time
import threading
import argparse
import logging
import contextlib
import readline
import select
import termios
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

# ── 项目根目录 ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── LeRobot 源码路径 ────────────────────────────────────────────────────────
_LEROBOT_SRC = _PROJECT_ROOT / "lerobot" / "src"
if str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))

# ── SO101 XML 默认路径（可通过 --xml_path 参数覆盖）───────────────────────
_DEFAULT_XML_PATH = str(_PROJECT_ROOT / "assets" / "so101" / "so101_new_calib.xml")

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

# busy_wait 不含 av，可安全在顶层导入
from lerobot.utils.robot_utils import busy_wait
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.utils import log_say

from orca_gym.environment.orca_gym_env import RewardType
from envs.so101.so101_env import SO101Env
from envs.manipulation.dual_arm_env import ControlDevice, RunMode, ActionType

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 配置常量
# ─────────────────────────────────────────────────────────────────────────────
ENV_NAME      = "SO101LeaderSimRecord-v0"
TIME_STEP     = 0.001
TARGET_FPS    = 30.0
REALTIME_STEP = 1.0 / TARGET_FPS   # 1/30 s ≈ 0.0333 s = 30 Hz
FRAME_SKIP    = round(REALTIME_STEP / TIME_STEP)   # ≈ 33
CONTROL_FREQ  = 1.0 / REALTIME_STEP

# SO101_XML_PATH 在运行时由 --xml_path 参数或 _DEFAULT_XML_PATH 决定（见 create_env）

# agent name（影响 obs 键名）
AGENT_NAME = "so101_new_calib_usda"

# SO101 6 个自由度：5 个臂关节 + 夹爪
JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
NUM_JOINTS = len(JOINT_NAMES)   # 6

# 相机配置：名称 → WebSocket 端口
CAMERA_CONFIG = {
    "camera_global":  7070,
    "camera_wrist": 7090,
}

# 相机图像分辨率（与 rgbd_camera.py CameraWrapper 原生分辨率一致）
IMG_H, IMG_W = 480, 640    # OrcaSim 相机输出分辨率（720p，16:9 宽高比）

# ── blueblock 初始化参数 ─────────────────────────────────────────────────
# 中心坐标来自 get_blueblock_pose.py 实测（场景静置时的 block 世界坐标）
BLOCK_CENTER_X   = 1.057112   # blueblock 中心 x（m）
BLOCK_CENTER_Y   = 0.141864   # blueblock 中心 y（m）
BLOCK_SIDE_LEN   = 0.025      # 蓝块边长（m），MuJoCo geom size=0.0125 为半边长
# 5 个候选点位：中心点 + 前/后/左/右各偏移 1.5 倍边长（0.0375 m）
# 坐标系：x 正方向为"前"，y 正方向为"左"
_BLOCK_OFFSET    = 1.5 * BLOCK_SIDE_LEN   # = 0.0375 m
BLOCK_CANDIDATES = [
    (BLOCK_CENTER_X,               BLOCK_CENTER_Y              ),  # 中心
    (BLOCK_CENTER_X + _BLOCK_OFFSET, BLOCK_CENTER_Y            ),  # 前
    (BLOCK_CENTER_X - _BLOCK_OFFSET, BLOCK_CENTER_Y            ),  # 后
    (BLOCK_CENTER_X,               BLOCK_CENTER_Y + _BLOCK_OFFSET),  # 左
    (BLOCK_CENTER_X,               BLOCK_CENTER_Y - _BLOCK_OFFSET),  # 右
]

# begin_save_video 临时目录（触发 WebSocket 推流，不使用这里的 mp4）
_VIDEO_DUMP_PATH = "/tmp/so101_sim_record_stream"



# ─────────────────────────────────────────────────────────────────────────────
# 等待用户准备好后开始下一集（与 record.py 的 _wait_for_next_episode 一致）
# ─────────────────────────────────────────────────────────────────────────────
def _stdin_has_enter() -> bool:
    """非阻塞地检测 stdin 中是否有 Enter（\\n）。"""
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if not rlist:
        return False
    data = os.read(sys.stdin.fileno(), 256)
    return b"\n" in data


def _wait_for_next_episode(events: dict, current_task: str, header: str) -> str | None:
    """
    每集结束后阻塞等待用户准备好再开始下一集。

    - 直接按 Enter   → 沿用当前任务描述，开始下一集
    - 按 Shift       → 预填当前任务到输入框，编辑后按 Enter 开始
    - 按 Esc         → 停止全部采集，返回 None
    """
    from pynput import keyboard as kb

    shift_triggered = threading.Event()

    def _on_key(key):
        if key in (kb.Key.shift, kb.Key.shift_l, kb.Key.shift_r):
            shift_triggered.set()

    W = 62
    print(f"\n{'═' * W}")
    print(f"  {header}")
    print(f"{'─' * W}")
    print("  重置完成后，选择下一步操作：")
    print("    · 直接按 Enter   → 开始下一集（沿用当前任务）")
    print("    · 按 Shift       → 修改任务描述后按 Enter 开始")
    print("    · 按 Esc         → 停止全部采集")
    print(f"{'─' * W}")
    print(f"  当前任务：{current_task}")
    print(f"{'═' * W}")
    print("  > 等待按键…", flush=True)

    # 打印提示后再 flush stdin，避免读到 Page Up/Down 的转义序列残留
    time.sleep(0.15)
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass

    listener = kb.Listener(on_press=_on_key)
    listener.start()

    try:
        while True:
            if events.get("stop_recording"):
                return None

            if shift_triggered.is_set():
                listener.stop()
                shift_triggered.clear()

                print("  修改任务描述（当前内容已预填，可直接编辑）：")
                readline.set_startup_hook(lambda: readline.insert_text(current_task))
                try:
                    new_text = input("  > ").strip()
                finally:
                    readline.set_startup_hook(None)
                    events["exit_early"] = False
                    events["rerecord_episode"] = False
                    try:
                        termios.tcflush(sys.stdin, termios.TCIFLUSH)
                    except Exception:
                        pass

                if new_text:
                    current_task = new_text
                print(f"\n  ✓ 任务已更新：{current_task}")
                print("  准备好后按 Enter 开始录制…\n")

                while True:
                    if events.get("stop_recording"):
                        return None
                    if _stdin_has_enter():
                        return current_task

            if _stdin_has_enter():
                return current_task

    finally:
        if listener.is_alive():
            listener.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 后台视频编码器（与 lerobot/src/lerobot/record.py 的 BackgroundVideoEncoder 一致）
# ─────────────────────────────────────────────────────────────────────────────
class BackgroundVideoEncoder:
    """
    将 PNG→MP4 编码移到单独的后台线程，让主线程在编码进行中就能立即开始录制下一集。

    设计要点
    --------
    * **单 worker**：``ThreadPoolExecutor(max_workers=1)`` 保证编码任务顺序执行，
      不会出现多任务并发争抢 CPU（SVT-AV1 内部已多线程）。
    * **episode 0 的写竞争**：``encode_episode_videos(0)`` 在编码结束后会调用
      ``meta.update_video_info() + write_info()``；而主线程在保存 episode 1 的元数据时
      也会调用 ``meta.save_episode() + write_info()``。
      两者会同时写 ``info.json``，造成数据丢失。
      解决方法：在保存任意 episode > 0 的数据之前调用 ``ensure_ep0_done()``，
      等待 episode 0 编码完成（实践中此调用几乎总是立即返回，
      因为 ep0 编码 ≈10-30 s，而录 ep1 需要 60 s+）。
    * **优雅退出**：正常结束时调用 ``wait_all()``；Ctrl+C 时同样调用 ``wait_all()``
      并支持再次按 Ctrl+C 强制跳过。
    """

    def __init__(self, dataset: LeRobotDataset) -> None:
        self._dataset = dataset
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bg_video_enc"
        )
        self._futures: list[Future] = []
        self._ep0_done: bool = False  # 快速路径标志，ep0 确认编码完成后恒为 True

    def submit(self, episode_index: int) -> None:
        """提交编码任务，立即返回，不阻塞主线程。"""
        future = self._executor.submit(self._encode, episode_index)
        self._futures.append(future)

    def _encode(self, episode_index: int) -> None:
        try:
            logging.info(f"⏳ [后台编码] Episode {episode_index} 开始编码...")
            self._dataset.encode_episode_videos(episode_index)
            logging.info(f"✓  [后台编码] Episode {episode_index} 编码完成")
        except Exception:
            logging.exception(f"✗  [后台编码] Episode {episode_index} 编码失败")

    def ensure_ep0_done(self) -> None:
        """
        确保 episode 0 的编码已完成。

        必须在调用 ``dataset.save_episode_data_only()`` for episode > 0 之前调用，
        以避免 ``encode_episode_videos(0)`` 中的 ``write_info()`` 与
        主线程 ``meta.save_episode()`` 产生 ``info.json`` 写竞争。
        """
        if self._ep0_done:
            return
        if self._futures:
            try:
                self._futures[0].result()  # block until ep0 encoding done
            except Exception:
                pass  # error already logged in _encode
        self._ep0_done = True

    def wait_all(self) -> None:
        """
        等待所有后台编码任务完成。程序退出前必须调用。
        再次按 Ctrl+C 可强制跳过（部分视频将不完整）。
        """
        pending = sum(1 for f in self._futures if not f.done())
        if pending:
            logging.info(
                f"⏳ 等待 {pending} 个后台视频编码任务完成，请勿关闭程序…"
                "（再次按 Ctrl+C 可强制跳过）"
            )
        interrupted = False
        for future in self._futures:
            if interrupted:
                future.cancel()
                continue
            try:
                future.result()
            except KeyboardInterrupt:
                logging.warning("⚠  强制跳过剩余编码任务，部分视频可能不完整")
                self._executor.shutdown(wait=False, cancel_futures=True)
                interrupted = True
            except Exception:
                logging.exception("后台编码任务异常")
        if not interrupted:
            self._executor.shutdown(wait=False)
            if self._futures:
                logging.info("✓  所有后台视频编码任务已完成")

    def has_pending(self) -> bool:
        """返回是否有尚未完成的编码任务。"""
        return any(not f.done() for f in self._futures)

    def shutdown_cancel(self) -> None:
        """立即取消所有尚未开始的任务（强制退出时调用）。"""
        self._executor.shutdown(wait=False, cancel_futures=True)


# ─────────────────────────────────────────────────────────────────────────────
# LeRobot 数据集特征定义
# ─────────────────────────────────────────────────────────────────────────────
def make_dataset_features(use_video: bool = True) -> dict:
    """构建 LeRobot 数据集特征 schema，双路相机均写入。"""
    img_dtype = "video" if use_video else "image"
    return {
        "action": {
            "dtype": "float32",
            "shape": (NUM_JOINTS,),
            "names": JOINT_NAMES,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (NUM_JOINTS,),
            "names": JOINT_NAMES,
        },
        "observation.images.camera_global": {
            "dtype": img_dtype,
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera_wrist": {
            "dtype": img_dtype,
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# OrcaGym 环境创建（主臂遥操作模式）
# ─────────────────────────────────────────────────────────────────────────────
def create_env(orcagym_addr: str = "localhost:50051",
               leader_port: str = "/dev/ttyACM0",
               leader_calibration=None,
               xml_path: str = None) -> gym.Env:
    if ENV_NAME not in gym.envs.registry:
        register(
            id=ENV_NAME,
            entry_point="envs.so101.so101_env:SO101Env",
            max_episode_steps=100000,
        )

    task_config = {
        "robot_xml_path":         os.path.abspath(xml_path or _DEFAULT_XML_PATH),
        "task_type":              "pick_place",
        "use_scene_augmentation": False,
    }

    env_config = {
        "frame_skip":              FRAME_SKIP,
        "reward_type":             RewardType.SPARSE,
        "orcagym_addr":            orcagym_addr,
        "agent_names":             [AGENT_NAME],
        "pico_ports":              [],
        "time_step":               TIME_STEP,
        "run_mode":                RunMode.TELEOPERATION,
        "action_type":             ActionType.JOINT_POS,
        "ctrl_device":             ControlDevice.LEADER_ARM,
        "control_freq":            CONTROL_FREQ,
        "sample_range":            0.0,
        "task_config_dict":        task_config,
        "action_step":             1,
        "camera_config":           {},
        "leader_arm_port":         leader_port,
        "leader_arm_calibration":  leader_calibration,
    }

    return gym.make(ENV_NAME, **env_config)


# ─────────────────────────────────────────────────────────────────────────────
# 相机初始化
# ─────────────────────────────────────────────────────────────────────────────
def setup_cameras(camera_config: dict) -> dict:
    from orca_gym.sensor.rgbd_camera import CameraWrapper
    cameras = {}
    for name, port in camera_config.items():
        try:
            cam = CameraWrapper(name=name, port=port)
            cam.start()
            cameras[name] = cam
            print(f"✓ 相机 {name} 已启动（端口 {port}）", flush=True)
        except Exception as e:
            print(f"✗ 相机 {name} 启动失败: {e}", flush=True)
    return cameras


def wait_for_cameras(cameras: dict, timeout: float = 30.0) -> None:
    print(f"\n等待相机就绪（最长 {timeout:.0f}s）...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        pending = [n for n, c in cameras.items() if not c.is_first_frame_received()]
        if not pending:
            print("✓ 所有相机已就绪\n", flush=True)
            return
        ready = [n for n in cameras if n not in pending]
        print(f"  就绪: {ready or '(无)'}  |  等待: {pending}", flush=True)
        time.sleep(1.0)
    print("⚠️  超时：部分相机未就绪，继续运行\n", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 从 obs / info 提取 LeRobot 帧数据
# ─────────────────────────────────────────────────────────────────────────────
def extract_state(obs: dict) -> np.ndarray:
    arm_qpos = obs[f"{AGENT_NAME}_arm_joint_qpos"].flatten()    # (5,)
    grasp    = obs[f"{AGENT_NAME}_grasp_value"].flatten()       # (1,)
    return np.concatenate([arm_qpos, grasp]).astype(np.float32)


def extract_action(info: dict) -> np.ndarray:
    act = np.asarray(info["action"], dtype=np.float32)
    return act[6:6 + NUM_JOINTS]   # (6,)


def get_camera_frame(cam) -> np.ndarray:
    """获取相机帧，返回 (H, W, 3) uint8 RGB。
    与 rgbd_camera.py 一致：不传 size，仅做 cvtColor（BGR→RGB）。
    相机原生输出 IMG_H×IMG_W（720×1280），无需 resize，与 rgbd_camera.py CameraWrapper 原生分辨率一致。
    """
    frame, _ = cam.get_frame(format="rgb24")
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# 单 episode 录制
# ─────────────────────────────────────────────────────────────────────────────
def record_episode(
    env,
    cameras: dict,
    dataset,
    events: dict,
    episode_time_s: float,
    task: str,
) -> bool:
    """
    录制单个 episode。
    Returns:
        True  → 正常保存（Page Down 或时间到）
        False → 用户要求重录（Page Up，丢弃）
    """
    frame_cnt = 0
    start_t   = time.perf_counter()
    timestamp = 0.0

    # ── 帧时序统计（ms）──────────────────────────────────────────────────
    _KEYS = ("step", "render", "cam_g", "cam_w", "write", "total")
    _stats: dict[str, list[float]] = {k: [] for k in _KEYS}
    SLOW_THR_MS = REALTIME_STEP * 1000 * 1.25   # 超过目标帧时 25% 视为慢帧

    logger.info(f"开始录制（时长 {episode_time_s}s，目标帧率 {TARGET_FPS:.0f}fps，"
                f"帧间隔 {REALTIME_STEP*1000:.1f}ms）")
    logger.info(f"任务：{task}")
    logger.info("Page Down=保存  Page Up=重录  Esc=停止")

    while timestamp < episode_time_s:
        loop_start = time.perf_counter()

        # ── 键盘事件检查 ──────────────────────────────────────────────────
        if events.get("rerecord_episode"):
            events["rerecord_episode"] = False
            events["exit_early"]       = False
            logger.info("丢弃当前 episode（Page Up）")
            _log_frame_stats(_stats, frame_cnt)
            return False

        if events.get("exit_early"):
            events["exit_early"] = False
            logger.info("提前保存当前 episode（Page Down）")
            _log_frame_stats(_stats, frame_cnt)
            return True

        if events.get("stop_recording"):
            logger.info("停止采集")
            _log_frame_stats(_stats, frame_cnt)
            return True

        # ── 仿真步进 ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape)
        )
        t1 = time.perf_counter()

        # ── 渲染 ──────────────────────────────────────────────────────────
        env.render()
        t2 = time.perf_counter()

        # ── 提取状态与动作 ────────────────────────────────────────────────
        state  = extract_state(obs)
        action = extract_action(info)

        # ── 相机帧获取 ────────────────────────────────────────────────────
        global_rgb = get_camera_frame(cameras["camera_global"])
        t3 = time.perf_counter()
        wrist_rgb  = get_camera_frame(cameras["camera_wrist"])
        t4 = time.perf_counter()

        # ── 写入 LeRobot 数据集 ──────────────────────────────────────────
        frame_data = {
            "action":                               action,
            "observation.state":                    state,
            "observation.images.camera_global":     global_rgb,
            "observation.images.camera_wrist":      wrist_rgb,
        }
        dataset.add_frame(frame_data, task=task)
        t5 = time.perf_counter()

        # ── 帧率控制 ──────────────────────────────────────────────────────
        dt_s = t5 - loop_start
        busy_wait(REALTIME_STEP - dt_s)
        t6 = time.perf_counter()

        timestamp = time.perf_counter() - start_t
        frame_cnt += 1

        # ── 逐帧时序记录 ──────────────────────────────────────────────────
        step_ms   = (t1 - t0) * 1000
        render_ms = (t2 - t1) * 1000
        cam_g_ms  = (t3 - t2) * 1000
        cam_w_ms  = (t4 - t3) * 1000
        write_ms  = (t5 - t4) * 1000
        total_ms  = (t6 - loop_start) * 1000   # 含 busy_wait

        _stats["step"].append(step_ms)
        _stats["render"].append(render_ms)
        _stats["cam_g"].append(cam_g_ms)
        _stats["cam_w"].append(cam_w_ms)
        _stats["write"].append(write_ms)
        _stats["total"].append(total_ms)

        # 慢帧：超过目标帧时 25% 则打印详细拆解
        work_ms = (t5 - loop_start) * 1000   # 不含 busy_wait
        if work_ms > SLOW_THR_MS:
            logger.warning(
                f"⚠ SLOW #{frame_cnt:5d} work={work_ms:.1f}ms "
                f"(>{SLOW_THR_MS:.1f}ms)  "
                f"step={step_ms:.1f} render={render_ms:.1f} "
                f"cam_g={cam_g_ms:.1f} cam_w={cam_w_ms:.1f} "
                f"write={write_ms:.1f}"
            )

        # 每 30 帧打印一次均值摘要
        if frame_cnt % 30 == 0:
            actual_hz = frame_cnt / max(timestamp, 1e-6)
            n = min(30, len(_stats["step"]))
            def _avg(k): return sum(_stats[k][-n:]) / n
            def _max(k): return max(_stats[k][-n:])
            logger.info(
                f"  #{frame_cnt:5d} | {timestamp:.1f}/{episode_time_s}s | "
                f"actual={actual_hz:.1f}Hz | "
                f"avg work={_avg('step')+_avg('render')+_avg('cam_g')+_avg('cam_w')+_avg('write'):.1f}ms "
                f"[step={_avg('step'):.1f} render={_avg('render'):.1f} "
                f"cam_g={_avg('cam_g'):.1f} cam_w={_avg('cam_w'):.1f} "
                f"write={_avg('write'):.1f}] "
                f"peak_work={_max('step')+_max('render')+_max('cam_g')+_max('cam_w')+_max('write'):.1f}ms"
            )

    _log_frame_stats(_stats, frame_cnt)
    actual_fps = frame_cnt / max(timestamp, 1e-6)
    logger.info(f"Episode 完成：{frame_cnt} 帧，实际 {actual_fps:.1f} fps")
    return True


def _log_frame_stats(stats: dict, frame_cnt: int) -> None:
    """打印 episode 整体帧时序统计（含 P50/P95/P99/max）。"""
    if frame_cnt == 0:
        return
    import statistics

    def _pct(lst, p):
        if not lst:
            return 0.0
        s = sorted(lst)
        idx = max(0, min(int(len(s) * p / 100 + 0.5), len(s) - 1))
        return s[idx]

    header = f"\n{'─'*70}\n  帧时序统计（共 {frame_cnt} 帧）"
    lines  = [header]
    total_avg = 0.0
    for k in ("step", "render", "cam_g", "cam_w", "write"):
        lst = stats.get(k, [])
        if not lst:
            continue
        avg = statistics.mean(lst)
        total_avg += avg
        lines.append(
            f"  {k:8s}  avg={avg:6.1f}ms  p50={_pct(lst,50):6.1f}  "
            f"p95={_pct(lst,95):6.1f}  p99={_pct(lst,99):6.1f}  "
            f"max={max(lst):6.1f}ms"
        )
    lines.append(f"  {'total_work':8s}  avg={total_avg:6.1f}ms  "
                 f"(target={REALTIME_STEP*1000:.1f}ms)")
    lines.append(f"{'─'*70}")
    logger.info("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# blueblock 随机初始化（每次 reset 后调用）
# ─────────────────────────────────────────────────────────────────────────────
def _find_block_joint_name(env) -> str | None:
    """
    找到蓝块（blueblock）的自由关节名（qpos 长度 = 7）。
      - 关节名含 'block' 且不含 'red'（排除 redblock_usda__joint_0 等红块关节）
      - qpos 长度为 7（6-DOF 自由关节）
    """
    uw = env.unwrapped
    try:
        joint_dict = uw.model.get_joint_dict()
    except Exception:
        return None

    for jn in joint_dict:
        if "block" not in jn.lower():
            continue
        if "red" in jn.lower():          # 排除红块
            continue
        try:
            qpos = uw.query_joint_qpos([jn]).get(jn)
            if qpos is not None and len(qpos) == 7:
                return jn
        except Exception:
            continue
    return None


def _randomize_block_position(env) -> None:
    """
    将蓝块（blueblock）初始化到 5 个固定候选点位之一（等概率随机选取）。
    5 个点位：中心点 + 前/后/左/右各偏移 1.5 倍边长（0.15 m）。
    Z 坐标和旋转姿态保持与 reset 后一致，不修改。
    """
    uw = env.unwrapped
    joint_name = _find_block_joint_name(env)
    if joint_name is None:
        logger.warning("⚠️  未找到蓝块自由关节，跳过位置初始化")
        return

    _LABELS = ["中心", "前", "后", "左", "右"]
    try:
        idx = np.random.randint(0, len(BLOCK_CANDIDATES))
        new_x, new_y = BLOCK_CANDIDATES[idx]
        qpos = uw.query_joint_qpos([joint_name])[joint_name].copy()
        # qpos: [x, y, z, qw, qx, qy, qz]
        qpos[0] = new_x
        qpos[1] = new_y
        uw.set_joint_qpos({joint_name: qpos})
        uw.mj_forward()
        logger.info(f"  蓝块初始化至【{_LABELS[idx]}】点位：x={new_x:.4f}  y={new_y:.4f}")
    except Exception as e:
        logger.warning(f"⚠️  蓝块位置初始化失败：{e}")


# ─────────────────────────────────────────────────────────────────────────────
# 重置阶段（等待用户复位场景，不录制）
# ─────────────────────────────────────────────────────────────────────────────
def reset_loop(env, events: dict, reset_time_s: float) -> None:
    logger.info(f"重置场景（{reset_time_s}s）…")
    start_t = time.perf_counter()

    with open(os.devnull, "w") as _null, \
         contextlib.redirect_stdout(_null), \
         contextlib.redirect_stderr(_null):
        # 先重置仿真场景（物体归位），再将 block 随机放置到中心 ±3.5cm 区域
        env.reset()
        _randomize_block_position(env)

        while (time.perf_counter() - start_t) < reset_time_s:
            loop_start = time.perf_counter()

            if events.get("exit_early"):
                events["exit_early"] = False
                break

            env.step(np.zeros(env.action_space.shape))
            env.render()

            dt_s = time.perf_counter() - loop_start
            busy_wait(REALTIME_STEP - dt_s)




# ─────────────────────────────────────────────────────────────────────────────
# 主采集流程
# ─────────────────────────────────────────────────────────────────────────────
def run_collection(
    orcagym_addr: str,
    leader_port: str,
    repo_id: str,
    root: Path | None,
    task: str,
    num_episodes: int,
    fps: float,
    episode_time_s: float,
    reset_time_s: float,
    push_to_hub: bool,
    xml_path: str = None,
) -> None:

    # ── Ctrl+C 强杀 ──────────────────────────────────────────────────────
    def _sigint_handler(sig, frame):
        print("\nCtrl+C 捕获，正在退出...", flush=True)
        os._exit(0)
    signal.signal(signal.SIGINT, _sigint_handler)

    # ── 1. 创建 OrcaGym 环境 ─────────────────────────────────────────────
    print(f"正在连接 OrcaSim（{orcagym_addr}）并初始化主臂...", flush=True)
    env = create_env(orcagym_addr, leader_port, xml_path=xml_path)
    obs, info = env.reset()
    print("✓ 环境就绪\n", flush=True)

    # ── 2. 从臂对齐主臂 ──────────────────────────────────────────────────
    print("正在读取主臂当前姿态，仿真从臂将对齐到该位置...", flush=True)
    unwrapped = env.unwrapped
    for agent in unwrapped._agents.values():
        if hasattr(agent, "align_leader_arm_zero"):
            agent.align_leader_arm_zero()
            break
    print("✓ 从臂已对齐主臂\n", flush=True)

    # ── 3. 启动相机 WebSocket 流 ─────────────────────────────────────────
    os.makedirs(_VIDEO_DUMP_PATH, exist_ok=True)
    env.unwrapped.begin_save_video(_VIDEO_DUMP_PATH, 0)
    print("✓ 视频流已启动\n", flush=True)

    cameras = setup_cameras(CAMERA_CONFIG)
    if not cameras:
        print("✗ 没有可用相机，退出", flush=True)
        env.close()
        return
    wait_for_cameras(cameras)

    # ── 4. 创建 LeRobot 数据集 ───────────────────────────────────────────
    features = make_dataset_features(use_video=True)
    if root is not None and Path(root).exists():
        print(
            f"\n❌ 数据集目录已存在：{root}\n"
            "   请使用新的 --root 路径，或手动删除旧目录后重试。\n"
            f"   例如：rm -rf {root}",
            flush=True,
        )
        env.close()
        return
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),   # 使用传入的 fps 参数（默认 30Hz）
        root=root,
        robot_type="so101",
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4 * len(CAMERA_CONFIG),
    )

    # ── 打印运行说明 ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SO101 主臂遥操作仿真数据采集")
    print(f"{'='*60}")
    print(f"  OrcaSim       : {orcagym_addr}")
    print(f"  主臂串口       : {leader_port}")
    print(f"  任务描述       : {task}")
    print(f"  目标 episodes  : {num_episodes}")
    print(f"  采集时长       : {episode_time_s}s / episode")
    print(f"  重置时间       : {reset_time_s}s")
    print(f"  帧率           : {fps} fps")
    print(f"  数据集 repo_id : {repo_id}")
    print(f"  相机分辨率     : {IMG_H}×{IMG_W}")
    print(f"{'='*60}")
    print("键盘：Page Down=保存  Page Up=重录  Esc=停止")
    print(f"{'='*60}\n")

    # ── 5. 键盘监听器 ────────────────────────────────────────────────────
    print("准备就绪，3 秒后开始……（期间请勿按 ESC / Page Up / Page Down）",
          flush=True)
    time.sleep(3.0)
    listener, events = init_keyboard_listener()
    print("✓ 键盘监听器已启动\n", flush=True)

    # ── 6. 主采集循环 ────────────────────────────────────────────────────
    recorded_episodes = 0
    current_task = task
    try:
        with VideoEncodingManager(dataset):
            bg_encoder = BackgroundVideoEncoder(dataset)

            # 第一集开始前等待用户确认
            new_task = _wait_for_next_episode(
                events, current_task,
                header=f"▶ 准备开始采集（共 {num_episodes} 集）"
            )
            if new_task is not None:
                current_task = new_task
            events["exit_early"] = False
            events["rerecord_episode"] = False

            try:
                while (recorded_episodes < num_episodes
                       and not events.get("stop_recording")):

                    episode_idx = recorded_episodes
                    ep_display  = f"Episode {episode_idx + 1} / {num_episodes}"

                    W = 62
                    print(f"\n{'═' * W}")
                    print(f"  ▶ 开始录制 {ep_display}")
                    print(f"  任务：{current_task}")
                    print(f"{'─' * W}")
                    print("  录制中按键：")
                    print("    · Page Down    提前结束并保存")
                    print("    · Page Up      丢弃本集，重新录制")
                    print("    · Esc          停止全部采集")
                    print(f"{'═' * W}\n")
                    log_say(f"开始录制 {ep_display}")

                    # 录制
                    saved = record_episode(
                        env=env,
                        cameras=cameras,
                        dataset=dataset,
                        events=events,
                        episode_time_s=episode_time_s,
                        task=current_task,
                    )

                    if not saved:
                        # Page Up：丢弃 → 重置场景 → 重录当前集
                        events["rerecord_episode"] = False
                        events["exit_early"]       = False
                        dataset.clear_episode_buffer()
                        logging.info(f"✗ {ep_display} 已丢弃，重置场景中…")
                        wait_header = f"✗ {ep_display} 已丢弃 → 将重新录制"

                        reset_loop(env=env, events=events, reset_time_s=reset_time_s)

                        events["exit_early"] = False
                        events["rerecord_episode"] = False
                        new_task = _wait_for_next_episode(events, current_task, header=wait_header)
                        if new_task is None:
                            break
                        current_task = new_task
                        events["exit_early"] = False
                        events["rerecord_episode"] = False
                        continue   # 重录同一集

                    # Page Down / 时间到：重置场景 → 异步保存（后台编码）→ 等待 Enter
                    events["rerecord_episode"] = False
                    events["exit_early"]       = False

                    reset_loop(env=env, events=events, reset_time_s=reset_time_s)

                    events["exit_early"] = False
                    events["rerecord_episode"] = False

                    # episode > 0 时确保 ep0 编码已完成，避免 info.json 写竞争
                    if recorded_episodes > 0:
                        bg_encoder.ensure_ep0_done()

                    logging.info(f"◎ {ep_display} 重置完毕，开始保存数据…")
                    episode_index = dataset.save_episode_data_only()
                    logging.info(
                        f"✓ {ep_display} 数据已落盘（parquet + 元数据），"
                        f"视频编码已提交后台线程（Episode {episode_index}）"
                    )
                    bg_encoder.submit(episode_index)
                    recorded_episodes += 1
                    wait_header = f"✓ {ep_display} 已保存（后台编码进行中）"

                    # 最后一集已完成则退出
                    if recorded_episodes >= num_episodes or events.get("stop_recording"):
                        break

                    # 等待用户确认开始下一集
                    new_task = _wait_for_next_episode(events, current_task, header=wait_header)
                    if new_task is None:
                        break
                    current_task = new_task
                    events["exit_early"] = False
                    events["rerecord_episode"] = False

            finally:
                # 无论正常结束还是异常/Ctrl+C，都等待所有后台编码完成
                # 再让 VideoEncodingManager.__exit__ 清理 PNG 目录
                bg_encoder.wait_all()

    except KeyboardInterrupt:
        logger.info("\n采集被中断（Ctrl+C）")

    except Exception as e:
        import traceback
        logger.error(f"\n⚠️  采集异常退出：{e}")
        logger.error(traceback.format_exc())

    finally:
        log_say("停止采集")
        logger.info("正在清理资源...")

        if not is_headless() and listener is not None:
            try:
                listener.stop()
            except Exception:
                pass

        try:
            env.unwrapped.stop_save_video()
        except Exception:
            pass

        # 给相机线程 2s 退出，超时则强制杀进程
        threading.Timer(6.0, lambda: os._exit(0)).start()
        for cam in cameras.values():
            cam.running = False
        for cam in cameras.values():
            if cam.thread and cam.thread.is_alive():
                cam.thread.join(timeout=2.0)

        try:
            env.close()
        except Exception:
            pass

        if push_to_hub and recorded_episodes > 0:
            logger.info("正在上传数据集到 Hugging Face Hub...")
            dataset.push_to_hub()

        print(f"\n{'='*60}")
        print("数据采集结束")
        print(f"  总 episode 数   : {dataset.meta.total_episodes}")
        print(f"  总帧数          : {dataset.meta.total_frames}")
        print(f"  数据集路径      : {dataset.root}")
        print(f"{'='*60}\n")

        os._exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SO101 主臂遥操作仿真数据采集")
    parser.add_argument("--repo_id",        type=str,   required=True,
                        help="数据集标识符，格式 username/dataset_name")
    parser.add_argument("--task",           type=str,   required=True,
                        help="任务描述（语言指令）")
    parser.add_argument("--root",           type=Path,  default=None,
                        help="数据集本地存储路径（默认 ~/.cache/huggingface/lerobot）")
    parser.add_argument("--num_episodes",   type=int,   default=50,
                        help="要采集的 episode 数量（默认 50）")
    parser.add_argument("--fps",            type=float, default=30.0,
                        help="采集帧率（默认 30）")
    parser.add_argument("--episode_time_s", type=float, default=60.0,
                        help="每个 episode 最长采集时长（秒，默认 60）")
    parser.add_argument("--reset_time_s",   type=float, default=2.0,
                        help="episode 间重置等待时间（秒，默认 30）")
    parser.add_argument("--orcagym_addr",   type=str,   default="localhost:50051",
                        help="OrcaGym gRPC 地址（默认 localhost:50051）")
    parser.add_argument("--leader_port",    type=str,   default="/dev/ttyACM0",
                        help="主臂串口路径（默认 /dev/ttyACM0）")
    parser.add_argument("--push_to_hub",      action="store_true",
                        help="采集完成后上传到 Hugging Face Hub")
    parser.add_argument("--xml_path",        type=str,   default=None,
                        help=f"SO101 XML 文件路径（默认: {_DEFAULT_XML_PATH}）")
    args = parser.parse_args()

    run_collection(
        orcagym_addr   = args.orcagym_addr,
        leader_port    = args.leader_port,
        repo_id        = args.repo_id,
        root           = args.root,
        task           = args.task,
        num_episodes   = args.num_episodes,
        fps            = args.fps,
        episode_time_s = args.episode_time_s,
        reset_time_s   = args.reset_time_s,
        push_to_hub    = args.push_to_hub,
        xml_path       = args.xml_path,
    )


if __name__ == "__main__":
    main()
