#!/usr/bin/env python3
"""
SO101 主从臂遥操作脚本

物理主臂（SO101 Leader）→ 仿真从臂（MuJoCo SO101）

控制逻辑（与 LeRobot 主从臂一致）：
  - 读取主臂6个关节当前角度（Feetech STS3215 串口）
  - 减去启动时对齐的零点偏置，得到相对增量
  - 直接写入仿真关节 ctrl，完全无 IK

启动步骤：
  1. 连接主臂 USB-TTL 串口（默认 /dev/ttyACM1）
  2. 启动 OrcaStudio 并运行 SO101 场景
  3. 运行本脚本，按提示对齐主从臂零点后按 Enter
  4. 拖动主臂即可控制仿真从臂

用法：
  python examples/so101/so101_leader_teleoperation.py
  python examples/so101/so101_leader_teleoperation.py --port /dev/ttyACM1
  python examples/so101/so101_leader_teleoperation.py --port /dev/ttyACM0 --max_steps 20000
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 默认 XML 路径（相对项目根目录的 assets 目录，可通过 --xml_path 参数覆盖）
_DEFAULT_XML_PATH = str(Path(project_root) / "assets" / "so101" / "so101_new_calib.xml")

import time
from datetime import datetime
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np

from orca_gym.environment.orca_gym_env import RewardType
from envs.so101.so101_env import SO101Env
from envs.manipulation.dual_arm_env import ControlDevice, RunMode, TaskStatus, ActionType

ENV_NAME = "SO101LeaderFollow-v0"

# 仿真和控制参数
TIME_STEP    = 0.001                      # 2ms 物理仿真步长
FRAME_SKIP   = 20                          # 每个控制步执行 20 次物理步
REALTIME_STEP = TIME_STEP * FRAME_SKIP     # 0.04s = 25 Hz 控制频率
CONTROL_FREQ = 1 / REALTIME_STEP           # 25 Hz


def teleoperation_session(env, max_steps: int = 20000):
    """
    主从臂遥操作主循环

    Args:
        env: SO101 环境实例
        max_steps: 最大步数
    """
    print(f"\n{'='*60}")
    print("SO101 主从臂遥操作模式")
    print(f"{'='*60}")
    print("【控制说明】")
    print("  直接拖动主臂即可控制仿真从臂")
    print("  - 5个手臂关节：1:1 角度映射")
    print("  - 夹爪：握紧/松开物理夹爪控制仿真夹爪")
    print("  - 启动时需对齐主从臂零点（按 Enter 确认）")
    print(f"{'='*60}\n")

    obs, info = env.reset()
    print(f"✅ 环境就绪，开始遥操作...\n")

    step_count  = 0
    total_reward = 0

    while step_count < max_steps:
        step_start = datetime.now()

        action = np.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        total_reward += reward
        step_count   += 1

        if step_count % 100 == 0:
            elapsed_ms = (datetime.now() - step_start).total_seconds() * 1000
            print(f"Step {step_count} | Reward: {total_reward:.3f} | 步耗时={elapsed_ms:.1f}ms (目标={REALTIME_STEP*1000:.0f}ms)", flush=True)

        # ── 实时限速：与 franka/dual_arm 参考实现一致 ──────────────────
        elapsed = (datetime.now() - step_start).total_seconds()
        if elapsed < REALTIME_STEP:
            time.sleep(REALTIME_STEP - elapsed)

        task_status = info.get('task_status', TaskStatus.NOT_STARTED)
        if task_status == TaskStatus.SUCCESS:
            print(f"\n✅ 任务完成！步数: {step_count}, 奖励: {total_reward:.3f}")
            break
        elif task_status == TaskStatus.RETRY:
            print(f"\n🔄 重置环境...")
            obs, info = env.reset()
            step_count   = 0
            total_reward = 0
            continue
        elif task_status == TaskStatus.FAILURE:
            print(f"\n❌ 任务失败")
            break

        if terminated:
            print(f"\n⚠️ 环境终止. 步数: {step_count}, 奖励: {total_reward:.3f}")
            break

    print(f"\n会话结束. 总步数: {step_count}, 总奖励: {total_reward:.3f}\n")


def create_so101_leader_env(
    orcagym_addr: str = "localhost:50051",
    leader_port:  str = "/dev/ttyACM0",
    leader_calibration=None,
    xml_path: str = None,
):
    """
    创建 SO101 主从臂遥操作环境

    Args:
        orcagym_addr:         OrcaGym 服务器地址
        leader_port:          主臂串口路径（如 /dev/ttyACM0）
        leader_calibration:   主臂标定数据（None = 使用默认值）
        xml_path:             SO101 MuJoCo XML 文件路径（None = 使用默认路径）

    Returns:
        env: 环境实例
    """
    if ENV_NAME not in gym.envs.registry:
        register(
            id=ENV_NAME,
            entry_point='envs.so101.so101_env:SO101Env',
            max_episode_steps=100000,
        )

    so101_xml_path = os.path.abspath(xml_path or _DEFAULT_XML_PATH)

    minimal_task_config = {
        'robot_xml_path':        so101_xml_path,
        'task_type':             'pick_place',
        'use_scene_augmentation': False,
    }

    env_config = {
        'frame_skip':          FRAME_SKIP,
        'reward_type':         RewardType.SPARSE,
        'orcagym_addr':        orcagym_addr,
        'agent_names':         ['so101_new_calib_usda'],
        'pico_ports':          [],
        'time_step':           TIME_STEP,
        'run_mode':            RunMode.TELEOPERATION,
        'action_type':         ActionType.JOINT_POS,
        'ctrl_device':         ControlDevice.LEADER_ARM,   # ← 主从臂模式
        'control_freq':        CONTROL_FREQ,
        'sample_range':        0.0,
        'task_config_dict':    minimal_task_config,
        'action_step':         1,
        'camera_config':       {},
        # 主臂串口参数（由 _init_leader_arm 读取）
        'leader_arm_port':         leader_port,
        'leader_arm_calibration':  leader_calibration,
    }

    env = gym.make(ENV_NAME, **env_config)
    return env


def main():
    import argparse

    parser = argparse.ArgumentParser(description='SO101 主从臂遥操作')
    parser.add_argument('--orcagym_addr', type=str, default='localhost:50051',
                        help='OrcaGym 服务器地址')
    parser.add_argument('--port', type=str, default='/dev/ttyACM1',
                        help='主臂串口路径（如 /dev/ttyACM0 或 /dev/ttyACM1）')
    parser.add_argument('--xml_path', type=str, default=None,
                        help=f'SO101 XML 文件路径（默认: {_DEFAULT_XML_PATH}）')
    parser.add_argument('--max_steps', type=int, default=20000,
                        help='最大步数')
    parser.add_argument('--continuous', action='store_true',
                        help='连续模式：完成后自动重置')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("SO101 主从臂遥操作工具")
    print(f"{'='*60}")
    print(f"OrcaGym地址 : {args.orcagym_addr}")
    print(f"主臂串口    : {args.port}")
    print(f"连续模式    : {'是' if args.continuous else '否'}")
    print(f"{'='*60}\n")

    print("正在创建环境并连接主臂...")
    env = create_so101_leader_env(
        orcagym_addr=args.orcagym_addr,
        leader_port=args.port,
        xml_path=args.xml_path,
    )
    print("✓ 环境创建成功！\n")

    # 零点对齐（在 gym.make() 完成后，有完整 TTY 时执行）
    unwrapped = env.unwrapped
    for agent in unwrapped._agents.values():
        if hasattr(agent, 'align_leader_arm_zero'):
            agent.align_leader_arm_zero()
            break

    try:
        if args.continuous:
            print("连续模式：按 Ctrl+C 退出\n")
            while True:
                teleoperation_session(env, max_steps=args.max_steps)
                time.sleep(1.0)
        else:
            teleoperation_session(env, max_steps=args.max_steps)

    except KeyboardInterrupt:
        print("\n\n遥操作被中断")

    finally:
        print("\n正在关闭环境...")
        env.close()
        print("✓ 环境已关闭\n")


if __name__ == "__main__":
    main()
