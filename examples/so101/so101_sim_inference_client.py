#!/usr/bin/env python3
"""
SO-101 + pi0.5 仿真推理【客户端脚本】（运行在 conda so101 环境）

架构：
  终端1（openpi uv 环境）  ──── WebSocket:8000 ────  终端2（conda so101 环境）
  serve_policy.py（策略服务器）                        本脚本（仿真+相机+动作执行）

步骤：
  1. 先在终端1（uv 环境）启动策略服务器：
       cd <OrcaGym根目录>/openpi
       uv run scripts/serve_policy.py policy:checkpoint \
           --policy.config=pi05_h7_lora \
           --policy.dir=models/pi05_h7_lora/h7_lora/4000

  2. 再在终端2（conda so101 环境）运行本脚本：
       conda activate so101
       cd <OrcaGym根目录>
       python examples/so101/so101_sim_inference_client.py

  3. 更多选项：
       python examples/so101/so101_sim_inference_client.py \
           --host localhost --port 8000 \
           --task "Pick up the blue block" \
           --xml_path assets/so101/so101_new_calib.xml

  pkill -f 'scripts/serve_policy.py'

键盘控制（推理过程中）：
  Page Down → 提前结束当前 chunk，立即重新采图推理
  Esc       → 停止推理
  Ctrl+C    → 强制退出
"""

import os
import sys
import time
import logging
import argparse
import contextlib
import platform
from functools import lru_cache
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

# ─── 项目根目录 ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─── openpi-client 路径（极轻量，无 JAX 依赖）────────────────────────────────
_OPENPI_CLIENT_SRC = _PROJECT_ROOT / "openpi" / "packages" / "openpi-client" / "src"
if str(_OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_CLIENT_SRC))

# ─── 导入 ─────────────────────────────────────────────────────────────────────
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from orca_gym.environment.orca_gym_env import RewardType
from envs.so101.so101_env import SO101Env
from envs.manipulation.dual_arm_env import ControlDevice, RunMode, ActionType

logging.basicConfig(level=logging.WARNING)   # 屏蔽底层框架 INFO 噪声
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_ch)
logger.propagate = False   # 不重复打印到 root logger


# ─── h11 数据集统计范围（全局，25 episodes）────────────────────────────────────
# 顺序: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
_DS_NAME        = "h11"
_DS_STATE_MIN   = np.array([-0.2582, -1.0017, -0.5197, -0.0364, -0.6987, -0.1745], dtype=np.float32)
_DS_STATE_MAX   = np.array([ 0.2457,  0.4966,  0.9410,  1.0003, -0.0179,  1.5117], dtype=np.float32)
_DS_STATE_MEAN  = np.array([-0.0046, -0.1483,  0.0431,  0.7820, -0.5597,  0.1426], dtype=np.float32)
_DS_ACTION_MIN  = np.array([-0.2601, -1.0000, -0.4848, -0.0389, -0.7292,  0.6509], dtype=np.float32)
_DS_ACTION_MAX  = np.array([ 0.2482,  0.5174,  1.0000,  1.0000, -0.3299,  1.0000], dtype=np.float32)
_DS_ACTION_MEAN = np.array([-0.0064, -0.1538,  0.0397,  0.7788, -0.5874,  0.7919], dtype=np.float32)
# h11 episode_0 第0帧真实初始状态（参考）
_DS_FRAME0_STATE = np.array([0.0054, -0.0244, 0.0315, 0.0307, -0.0179, -0.1745], dtype=np.float32)
# h11 episode_0 第0帧动作（参考）—— 注意 gripper=0.6509 说明数采一开始就张夹爪
_DS_FRAME0_ACTION = np.array([0.0332, -0.9991, 0.7599, 0.9855, -0.5732, 0.6509], dtype=np.float32)


def _log(msg: str) -> None:
    logger.debug(msg)


# ─── 键盘监听 / 帧率控制（内联，不依赖 lerobot）────────────────────────────────

@lru_cache(maxsize=1)
def is_headless() -> bool:
    try:
        import pynput  # noqa
        return False
    except Exception:
        return True


def init_keyboard_listener():
    events = {
        "exit_early":          False,
        "rerecord_episode":    False,
        "stop_recording":      False,
        "inference_start_time": None,   # 推理开始时间戳（perf_counter）
        "pagedown_times":      [],      # 每次按 Page Down 时的已用时（秒）列表
    }
    if is_headless():
        return None, events
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.page_down:
                t_start = events.get("inference_start_time")
                if t_start is not None:
                    elapsed = time.perf_counter() - t_start
                    events["pagedown_times"].append(elapsed)
                    print(
                        f"Page Down：提前结束当前 chunk...  ⏱ 已用时 {elapsed:.2f}s",
                        flush=True,
                    )
                else:
                    print("Page Down：提前结束当前 chunk...", flush=True)
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                t_start = events.get("inference_start_time")
                if t_start is not None:
                    elapsed = time.perf_counter() - t_start
                    print(
                        f"Esc：停止推理...  ⏱ 已用时 {elapsed:.2f}s",
                        flush=True,
                    )
                else:
                    print("Esc：停止推理...", flush=True)
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"键盘事件处理异常: {e}", flush=True)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def busy_wait(seconds: float) -> None:
    if seconds <= 0:
        return
    if platform.system() in ("Darwin", "Windows"):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            pass
    else:
        time.sleep(seconds)


# ─── 仿真环境常量 ──────────────────────────────────────────────────────────────
ENV_NAME        = "SO101SimInferenceClient-v0"
TIME_STEP       = 0.001
TARGET_FPS      = 30.0
REALTIME_STEP   = 1.0 / TARGET_FPS
FRAME_SKIP      = round(REALTIME_STEP / TIME_STEP)
CONTROL_FREQ    = 1.0 / REALTIME_STEP

_DEFAULT_XML_PATH = str(_PROJECT_ROOT / "assets" / "so101" / "so101_new_calib.xml")
SO101_XML_PATH  = os.environ.get("SO101_XML_PATH", _DEFAULT_XML_PATH)
AGENT_NAME      = "so101_new_calib_usda"

# 相机 WebSocket 端口
CAMERA_GLOBAL_PORT = 7070
CAMERA_WRIST_PORT  = 7090

# 相机图像分辨率（openpi 内部会 resize 到 224×224）
IMG_SIZE        = 224   # openpi 模型期望输入尺寸

# SO-101 关节顺序
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
NUM_JOINTS = len(MOTOR_NAMES)  # 6

# 仿真动作空间：12 维（[0:6]=EE pose 不用，[6:11]=5关节，[11]=夹爪）
_ACTION_DIM  = 12
_ARM_OFFSET  = 6
_GRIPPER_IDX = 11

# blueblock 随机范围
BLOCK_CENTER_X     = 1.057112
BLOCK_CENTER_Y     = 0.141864
BLOCK_RANDOM_RANGE = 0.035

_VIDEO_DUMP_PATH = "/tmp/so101_sim_inference_client_stream"


# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def extract_state(obs: dict) -> np.ndarray:
    arm_qpos = obs[f"{AGENT_NAME}_arm_joint_qpos"].flatten()   # (5,)
    grasp    = obs[f"{AGENT_NAME}_grasp_value"].flatten()      # (1,)
    return np.concatenate([arm_qpos, grasp]).astype(np.float32)


def policy_to_env_action(policy_action_6: np.ndarray) -> np.ndarray:
    """
    将策略输出的 6 维归一化动作转换为仿真环境接受的 12 维归一化动作。
    训练标签 action = scaled_action[6:12]，归一化到 [-1,1]；
    POLICY_NORMALIZED 模式下 env.step() 同样接受 [-1,1] 的归一化动作。
    """
    full = np.zeros(_ACTION_DIM, dtype=np.float32)
    full[_ARM_OFFSET : _ARM_OFFSET + 5] = policy_action_6[:5]  # 5 arm joints
    full[_GRIPPER_IDX]                  = policy_action_6[5]   # gripper
    return full


def prepare_image(img: np.ndarray) -> np.ndarray:
    """将相机帧缩放并转换为 openpi 期望的 uint8 格式（224×224）。"""
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
    )


# ─── 仿真环境 ──────────────────────────────────────────────────────────────────

def create_env(orcagym_addr: str = "localhost:50051",
               xml_path: str = None) -> gym.Env:
    if ENV_NAME not in gym.envs.registry:
        register(
            id=ENV_NAME,
            entry_point="envs.so101.so101_env:SO101Env",
            max_episode_steps=100000,
        )
    task_config = {
        "robot_xml_path":         os.path.abspath(xml_path or SO101_XML_PATH),
        "task_type":              "pick_place",
        "use_scene_augmentation": False,
    }
    env_config = {
        "frame_skip":       FRAME_SKIP,
        "reward_type":      RewardType.SPARSE,
        "orcagym_addr":     orcagym_addr,
        "agent_names":      [AGENT_NAME],
        "pico_ports":       [],
        "time_step":        TIME_STEP,
        "run_mode":         RunMode.POLICY_NORMALIZED,
        "action_type":      ActionType.JOINT_POS,
        "ctrl_device":      ControlDevice.LEADER_ARM,
        "control_freq":     CONTROL_FREQ,
        "sample_range":     0.0,
        "task_config_dict": task_config,
        "action_step":      1,
        "camera_config":    {},
    }
    return gym.make(ENV_NAME, **env_config)


# ─── 相机 ──────────────────────────────────────────────────────────────────────

def setup_cameras() -> dict:
    from orca_gym.sensor.rgbd_camera import CameraWrapper
    cameras = {}
    for name, port in [("camera_global", CAMERA_GLOBAL_PORT),
                       ("camera_wrist",  CAMERA_WRIST_PORT)]:
        try:
            cam = CameraWrapper(name=name, port=port)
            cam.start()
            cameras[name] = cam
            _log(f"✓ 相机 {name} 已启动（端口 {port}）")
        except Exception as e:
            _log(f"✗ 相机 {name} 启动失败: {e}")
    return cameras


def wait_for_cameras(cameras: dict, timeout: float = 30.0) -> None:
    _log(f"\n等待相机就绪（最长 {timeout:.0f}s）...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        pending = [n for n, c in cameras.items() if not c.is_first_frame_received()]
        if not pending:
            _log("✓ 所有相机已就绪\n")
            return
        _log(f"  等待: {pending}")
        time.sleep(1.0)
    _log("⚠️  超时：部分相机未就绪，继续运行\n")


# ─── blueblock 随机初始化 ──────────────────────────────────────────────────────

def _find_block_joint_name(env) -> str | None:
    """
    找到蓝块（blueblock）的自由关节名（qpos 长度 = 7）。
    逻辑与 so101_leader_sim_record.py 保持一致：
      - 关节名含 'block' 且不含 'red'（排除 redblock_usda__joint_0）
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


def get_block_pos(env) -> np.ndarray | None:
    """查询蓝块当前世界坐标 (x, y, z)，不可用时返回 None。"""
    uw  = env.unwrapped
    jn  = _find_block_joint_name(env)
    if jn is None:
        return None
    try:
        qpos = uw.query_joint_qpos([jn]).get(jn)
        if qpos is not None and len(qpos) >= 3:
            return np.array(qpos[:3], dtype=np.float64)
    except Exception:
        pass
    return None


def log_block_state(env, tag: str = "") -> None:
    """打印蓝块当前位姿信息（位置 + 四元数）。"""
    uw = env.unwrapped
    jn = _find_block_joint_name(env)
    prefix = f"[BLOCK{(' ' + tag) if tag else ''}]"
    if jn is None:
        _log(f"{prefix} ⚠️  未找到蓝块关节（joint_name=None）")
        return
    try:
        qpos = uw.query_joint_qpos([jn]).get(jn)
        if qpos is None:
            _log(f"{prefix} ⚠️  qpos 为 None (joint={jn})")
            return
        x, y, z = qpos[0], qpos[1], qpos[2]
        quat = qpos[3:7] if len(qpos) >= 7 else []
        _log(
            f"{prefix} joint={jn}  "
            f"pos=({x:.4f}, {y:.4f}, {z:.4f})  "
            f"center_ref=({BLOCK_CENTER_X:.4f}, {BLOCK_CENTER_Y:.4f})  "
            f"偏移=(Δx={x-BLOCK_CENTER_X:+.4f}, Δy={y-BLOCK_CENTER_Y:+.4f})  "
            + (f"quat={np.round(quat, 4).tolist()}" if len(quat) else "")
        )
    except Exception as e:
        _log(f"{prefix} ⚠️  查询失败: {e}")


def log_arm_state(state: np.ndarray, tag: str = "", actions: np.ndarray | None = None) -> None:
    """打印机械臂关节状态，并与数据集统计范围对比（h11）。"""
    prefix = f"[ARM{(' ' + tag) if tag else ''}]"
    lines  = [f"{prefix} 当前关节状态 vs {_DS_NAME} 数据集范围:"]
    lines.append(f"  {'关节':14s}  {'当前值':>8}  {'ds_min':>8}  {'ds_max':>8}  {'ds_mean':>8}  {'偏差(val-mean)':>14}  状态")
    for i, name in enumerate(MOTOR_NAMES):
        v    = state[i]
        lo   = _DS_STATE_MIN[i]
        hi   = _DS_STATE_MAX[i]
        mu   = _DS_STATE_MEAN[i]
        ref  = _DS_FRAME0_STATE[i]
        diff = v - mu
        if v < lo or v > hi:
            flag = "⚠️ OOB"
        elif abs(v - ref) > 0.1:
            flag = "△ 偏离首帧"
        else:
            flag = "✓"
        lines.append(
            f"  {name:14s}  {v:+8.4f}  {lo:+8.4f}  {hi:+8.4f}  {mu:+8.4f}  {diff:+14.4f}  {flag}"
        )
    lines.append(f"  {'[ep0 frame0参考]':14s}  " +
                 "  ".join(f"{_DS_FRAME0_STATE[i]:+8.4f}" for i in range(NUM_JOINTS)))
    _log("\n".join(lines))

    if actions is not None:
        lines2 = [f"{prefix} ── 输出动作（前3步）vs {_DS_NAME} 动作范围 ──"]
        lines2.append(f"  {'关节':14s}  {'step0':>8}  {'step1':>8}  {'step2':>8}  {'act_min':>8}  {'act_max':>8}  {'act_mean':>8}  状态")
        for i, name in enumerate(MOTOR_NAMES):
            lo   = _DS_ACTION_MIN[i]
            hi   = _DS_ACTION_MAX[i]
            mu   = _DS_ACTION_MEAN[i]
            a0   = actions[0, i]
            a1   = actions[1, i] if len(actions) > 1 else float('nan')
            a2   = actions[2, i] if len(actions) > 2 else float('nan')
            flag = "⚠️ OOB" if a0 < lo or a0 > hi else "✓"
            lines2.append(
                f"  {name:14s}  {a0:+8.4f}  {a1:+8.4f}  {a2:+8.4f}  {lo:+8.4f}  {hi:+8.4f}  {mu:+8.4f}  {flag}"
            )
        # gripper 专项诊断
        gr_cmd = actions[0, 5]
        lines2.append(f"\n  [夹爪诊断] step0 gripper={gr_cmd:+.4f}  "
                      f"数采首帧参考={_DS_FRAME0_ACTION[5]:+.4f}  "
                      f"动作最小值={_DS_ACTION_MIN[5]:+.4f}  "
                      + ("✅ 应当张开" if gr_cmd > 0.5 else "❌ 偏小，夹爪可能不张开"))
        _log("\n".join(lines2))


def randomize_block(env) -> None:
    uw = env.unwrapped
    jn = _find_block_joint_name(env)
    if jn is None:
        logger.warning("⚠️  未找到蓝块关节，跳过随机初始化")
        return
    try:
        qpos = uw.query_joint_qpos([jn])[jn].copy()
        _log(f"[BLOCK INIT] 随机前位置: x={qpos[0]:.4f}  y={qpos[1]:.4f}  z={qpos[2]:.4f}")
        qpos[0] = BLOCK_CENTER_X + np.random.uniform(-BLOCK_RANDOM_RANGE, BLOCK_RANDOM_RANGE)
        qpos[1] = BLOCK_CENTER_Y + np.random.uniform(-BLOCK_RANDOM_RANGE, BLOCK_RANDOM_RANGE)
        uw.set_joint_qpos({jn: qpos})
        uw.mj_forward()
        _log(f"[BLOCK INIT] 随机后位置: x={qpos[0]:.4f}  y={qpos[1]:.4f}  z={qpos[2]:.4f}  "
             f"(center_ref: x={BLOCK_CENTER_X:.4f}, y={BLOCK_CENTER_Y:.4f})")
    except Exception as e:
        logger.warning(f"⚠️  蓝块随机初始化失败: {e}")


# ─── 策略服务器连接 ────────────────────────────────────────────────────────────

def connect_policy_server(host: str, port: int,
                          retry: int = 10, interval: float = 3.0):
    """连接策略服务器，失败时重试。"""
    _log(f">>> 连接策略服务器 ws://{host}:{port}  （最多重试 {retry} 次）...")
    for i in range(retry):
        try:
            client = websocket_client_policy.WebsocketClientPolicy(
                host=host, port=port
            )
            _log(">>> ✓ 策略服务器已连接")
            return client
        except Exception as e:
            if i < retry - 1:
                _log(f"    第 {i+1} 次连接失败：{e}，{interval:.0f}s 后重试...")
                time.sleep(interval)
            else:
                raise RuntimeError(
                    f"无法连接策略服务器 ws://{host}:{port}，"
                    f"请确认服务器已在 openpi uv 环境中启动：\n"
                    f"  cd <OrcaGym根目录>/openpi\n"
                    f"  uv run scripts/serve_policy.py policy:checkpoint "
                    f"--policy.config=pi05_h7_lora "
                    f"--policy.dir=ckpt/pi05_h7_lora/h7_lora"
                ) from e


# ─── 任务输入 ──────────────────────────────────────────────────────────────────

def prompt_task(default_task: str) -> str:
    print(f"\n{'─' * 60}")
    print("请输入任务描述（直接回车使用默认值）：")
    print(f"  默认：{default_task}")
    print(f"{'─' * 60}")
    try:
        user_input = input("> ").strip()
        return user_input if user_input else default_task
    except EOFError:
        # 非交互式运行（后台/管道），直接使用默认任务
        print(f"（非交互式模式，使用默认任务：{default_task}）")
        return default_task


# ─── 推理主循环 ────────────────────────────────────────────────────────────────

def run_one_episode(
    client,
    env,
    cameras: dict,
    task: str,
    fps: int,
    max_steps: int,
) -> None:
    """
    仿真推理主循环（客户端侧）：
      读取观测 → 发送给策略服务器 → 接收 action chunk → 执行 → 循环
    """
    listener, events = init_keyboard_listener()
    _log(">>> 键盘监听已启动")

    step      = 0
    chunk_idx = 0
    _log(f">>> 任务：{task}")
    _log(">>> 键盘：Page Down=提前结束当前 chunk  Esc/Ctrl+C=停止")

    # 获取初始 obs，重置并随机化场景
    obs, _ = env.reset()
    with contextlib.suppress(Exception):
        randomize_block(env)
    for _ in range(5):
        obs, _, _, _, _ = env.step(np.zeros(_ACTION_DIM, dtype=np.float32))
    env.render()

    # ── 推理计时起点（warmup 结束后正式开始）─────────────────────────────────
    t_inference_start = time.perf_counter()
    events["inference_start_time"] = t_inference_start
    _log(f">>> ⏱  推理计时开始")

    # ── 重置后打印蓝块初始状态 ────────────────────────────────────────────────
    init_state = extract_state(obs)
    _log("=" * 70)
    _log(f"[TASK] prompt: \"{task}\"")
    _log("=" * 70)
    log_block_state(env, tag="reset后初始位置")
    _log(f"[ARM INIT] 推理初始关节状态 vs {_DS_NAME} ep0 frame0 参考:")
    _log(f"  {'关节':14s}  {'推理初始':>8}  {'数据集ep0f0':>11}  {'差值':>8}  状态")
    for i, m in enumerate(MOTOR_NAMES):
        v   = init_state[i]
        ref = _DS_FRAME0_STATE[i]
        d   = v - ref
        flag = "⚠️ 差异>0.1" if abs(d) > 0.1 else "✓"
        _log(f"  {m:14s}  {v:+8.4f}  {ref:+11.4f}  {d:+8.4f}  {flag}")
    _log(f"  [关键] gripper初始={init_state[5]:+.4f}  "
         f"数采首帧={_DS_FRAME0_STATE[5]:+.4f}  "
         f"数采首帧动作(应张开)={_DS_FRAME0_ACTION[5]:+.4f}")

    try:
        while not events.get("stop_recording"):
            if max_steps > 0 and step >= max_steps:
                _log(f">>> 已达最大步数 {max_steps}，停止")
                break

            # ── 1. 读取观测 ──────────────────────────────────────────────────
            _log(f"\n{'─' * 60}")
            _log(f">>> [chunk {chunk_idx}] 读取观测...")
            t_obs = time.perf_counter()

            state     = extract_state(obs)
            front_img = cameras["camera_global"].get_frame(format="rgb24")[0]
            wrist_img = cameras["camera_wrist"].get_frame(format="rgb24")[0]

            # 缩放到 224×224，与策略服务器期望格式一致
            front_224 = prepare_image(front_img)
            wrist_224 = prepare_image(wrist_img)

            _log(
                f">>> [chunk {chunk_idx}] 观测耗时 {(time.perf_counter()-t_obs)*1000:.0f} ms | "
                f"state={np.round(state, 3).tolist()}"
            )

            # 打印蓝块当前位置
            log_block_state(env, tag=f"chunk{chunk_idx}开始")
            # 打印机械臂关节状态与数据集范围对比
            log_arm_state(state, tag=f"chunk{chunk_idx}输入")

            # ── 2. 调用策略服务器 ────────────────────────────────────────────
            observation = {
                "observation/image":       front_224,   # (224, 224, 3) uint8
                "observation/wrist_image": wrist_224,   # (224, 224, 3) uint8
                "observation/state":       state,       # (6,) float32
                "prompt":                  task,
            }

            if chunk_idx == 0:
                _log(f">>> [chunk {chunk_idx}] 首次推理（服务器端 JAX JIT 约需 1~3 分钟）...")
            else:
                _log(f">>> [chunk {chunk_idx}] 推理中...")

            t_infer  = time.perf_counter()
            result   = client.infer(observation)
            t_infer  = time.perf_counter() - t_infer

            # result["actions"]: (50, 32)，前 6 维为 SO-101 归一化关节目标
            actions = result["actions"]
            _log(
                f">>> [chunk {chunk_idx}] 推理完成 | 耗时 {t_infer*1000:.0f} ms | "
                f"关节范围 [{actions[:, :NUM_JOINTS].min():.3f}, "
                f"{actions[:, :NUM_JOINTS].max():.3f}]"
            )
            # 打印动作输出 + 与数据集范围对比
            log_arm_state(state, tag=f"chunk{chunk_idx}输出", actions=actions)
            # 打印前10步各关节动作趋势（诊断偏左/夹爪不张问题）
            n_show = min(10, len(actions))
            _log(f"[TREND chunk{chunk_idx}] 前{n_show}步动作趋势:")
            _log(f"  {'step':>5}  " + "  ".join(f"{m:>13}" for m in MOTOR_NAMES))
            for si in range(n_show):
                _log(f"  {si:>5}  " + "  ".join(f"{actions[si, i]:+13.4f}" for i in range(NUM_JOINTS)))

            # ── 3. 执行 action chunk（open-loop）────────────────────────────
            _log(f">>> [chunk {chunk_idx}] 执行 {len(actions)} 步...")
            for sub_step, action_vec in enumerate(actions):
                if events.get("exit_early"):
                    events["exit_early"] = False
                    _log(f">>> [chunk {chunk_idx}] Page Down：第 {sub_step} 步提前结束，重新推理")
                    break
                if events.get("stop_recording"):
                    break
                if max_steps > 0 and step >= max_steps:
                    break

                t0 = time.perf_counter()

                env_action = policy_to_env_action(action_vec[:NUM_JOINTS])
                obs, _reward, terminated, truncated, _ = env.step(env_action)
                env.render()

                if sub_step % 5 == 0:
                    cur_state = extract_state(obs)
                    blk_pos   = get_block_pos(env)
                    blk_str   = (f"({blk_pos[0]:.4f}, {blk_pos[1]:.4f}, {blk_pos[2]:.4f})"
                                 if blk_pos is not None else "N/A")
                    # 6关节全打印（紧凑格式）
                    joint_str = "  ".join(
                        f"{MOTOR_NAMES[i][:3]}={cur_state[i]:+.3f}"
                        for i in range(NUM_JOINTS)
                    )
                    gr_ok = "✅张" if cur_state[5] > 0.3 else "❌合"
                    _log(
                        f"[EXE c{chunk_idx} s{sub_step:3d}/{len(actions)}(t={step:4d})]  "
                        f"{joint_str}  blk={blk_str}  {gr_ok}"
                        + ("  ✅SUCCESS" if terminated else "")
                    )

                step += 1
                if terminated:
                    _log(">>> 🎉 任务成功！")
                    log_block_state(env, tag="任务成功时")
                    events["stop_recording"] = True
                    break

                busy_wait(REALTIME_STEP - (time.perf_counter() - t0))

            # chunk 结束：打印蓝块位置 + 末端关节状态
            cur_state = extract_state(obs)
            log_block_state(env, tag=f"chunk{chunk_idx}结束")
            _log(f"[ARM END c{chunk_idx}] " +
                 "  ".join(f"{MOTOR_NAMES[i]}={cur_state[i]:+.4f}" for i in range(NUM_JOINTS)))
            chunk_idx += 1

    except KeyboardInterrupt:
        _log(">>> Ctrl+C 强制停止")
    finally:
        if not is_headless() and listener is not None:
            listener.stop()

    # ── 计时汇总 ──────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_inference_start
    pd_times = events.get("pagedown_times", [])
    _log(f">>> 推理结束 | 共 {chunk_idx} 个 chunk，{step} 步")
    _log(f">>> ⏱  推理总时长: {total_elapsed:.2f}s")
    if pd_times:
        pd_str = "  ".join(
            f"第{i+1}次={t:.2f}s" for i, t in enumerate(pd_times)
        )
        _log(f">>> ⏱  Page Down 按下时刻: {pd_str}")
    else:
        _log(f">>> ⏱  本轮未按 Page Down")


# ─── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SO-101 + pi0.5 仿真推理客户端（conda so101 环境）",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
前提：先在另一个终端（openpi uv 环境）启动策略服务器：
  cd <OrcaGym根目录>/openpi
  uv run scripts/serve_policy.py policy:checkpoint \\
      --policy.config=pi05_h7_lora \\
      --policy.dir=models/pi05_h7_lora/h7_lora

然后运行本脚本（conda so101 环境）：
  cd <OrcaGym根目录>
  python examples/so101/so101_sim_inference_client.py
""",
    )
    parser.add_argument("--host",         type=str,  default="localhost",
                        help="策略服务器地址（默认 localhost）")
    parser.add_argument("--port",         type=int,  default=8000,
                        help="策略服务器端口（默认 8000）")
    parser.add_argument("--task",         type=str,  default="Pick up the blue block",
                        help="默认任务描述")
    parser.add_argument("--orcagym_addr", type=str,  default="localhost:50051",
                        help="OrcaGym gRPC 地址（默认 localhost:50051）")
    parser.add_argument("--xml_path",     type=str,  default=None,
                        help=f"SO101 XML 文件路径（默认: {_DEFAULT_XML_PATH}）")
    parser.add_argument("--fps",          type=int,  default=30)
    parser.add_argument("--max_steps",    type=int,  default=0,
                        help="每轮最大步数，0=不限")
    args = parser.parse_args()

    # ── 1. 连接策略服务器 ─────────────────────────────────────────────────
    client = connect_policy_server(args.host, args.port)

    # ── 2. 创建仿真环境 ───────────────────────────────────────────────────
    _log(f">>> 连接 OrcaSim（{args.orcagym_addr}）...")
    env = create_env(args.orcagym_addr, xml_path=args.xml_path)
    _log(">>> ✓ 仿真环境就绪")

    # ── 3. 启动相机 ───────────────────────────────────────────────────────
    os.makedirs(_VIDEO_DUMP_PATH, exist_ok=True)
    env.unwrapped.begin_save_video(_VIDEO_DUMP_PATH, 0)
    cameras = setup_cameras()
    if len(cameras) < 2:
        _log("✗ 相机数量不足，退出")
        env.close()
        return
    wait_for_cameras(cameras)

    # ── 4. 打印信息 ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SO-101 + pi0.5 仿真推理（客户端）")
    print(f"{'=' * 60}")
    print(f"  策略服务器    : ws://{args.host}:{args.port}")
    print(f"  OrcaSim       : {args.orcagym_addr}")
    print(f"  控制频率      : {args.fps} Hz | 每 chunk 50 步（约 {50/args.fps:.1f}s）")
    print(f"{'=' * 60}")
    print("  键盘：Page Down=提前结束当前 chunk  Esc/Ctrl+C=停止")
    print(f"{'=' * 60}\n")

    # ── 5. 推理 ──────────────────────────────────────────────────────────
    task = prompt_task(args.task)

    try:
        run_one_episode(
            client    = client,
            env       = env,
            cameras   = cameras,
            task      = task,
            fps       = args.fps,
            max_steps = args.max_steps,
        )
    except KeyboardInterrupt:
        _log("\n>>> 强制退出（Ctrl+C）")
    finally:
        _log(">>> 正在关闭相机...")
        for cam in cameras.values():
            try:
                cam.running = False
            except Exception:
                pass
        try:
            env.unwrapped.stop_save_video()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        _log(">>> ✓ 结束")


if __name__ == "__main__":
    main()
