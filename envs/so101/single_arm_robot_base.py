"""
单臂机器人基类
参考 dual_arm_robot.py 的结构，适配单臂机器人
"""

import os
import numpy as np
from orca_gym.utils import rotations
from orca_gym.adapters.robosuite.controllers.controller_factory import controller_factory
import orca_gym.adapters.robosuite.controllers.controller_config as controller_config
import orca_gym.adapters.robosuite.utils.transform_utils as transform_utils
from scipy.spatial.transform import Rotation as R
from envs.manipulation.dual_arm_env import AgentBase, RunMode, ControlDevice, ActionType, TaskStatus
from orca_gym.utils.inverse_kinematics_controller import InverseKinematicsController
from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()


class SingleArmRobotBase(AgentBase):
    """单臂机器人基类，类似 DualArmRobot 但简化为单臂"""
    
    def __init__(self, env, id: int, name: str) -> None:
        super().__init__(env, id, name)
        
    def init_agent(self, id: int, config: dict):
        """初始化机器人代理"""
        self._read_config(config, id)
        self._setup_initial_info()
        self._setup_device()
        self._setup_controller()
        
    def _read_config(self, config: dict, id: int) -> None:
        """从配置读取关节、执行器等信息"""
        # 基座配置
        try:
            self._base_body_name = [self._env.body(config["base"]["base_body_name"], id)]
        except (KeyError, AttributeError) as e:
            _logger.error(f"❌ 找不到机器人基座 '{config['base']['base_body_name']}'")
            # 尝试获取关节信息用于调试
            try:
                import mujoco
                if hasattr(self._env, 'gym') and hasattr(self._env.gym, '_mjModel') and self._env.gym._mjModel is not None:
                    njnt = self._env.gym._mjModel.njnt
                    joint_names = [mujoco.mj_id2name(self._env.gym._mjModel, mujoco.mjtObj.mjOBJ_JOINT, i) 
                                  for i in range(njnt)]
                    joint_info = f"当前场景有 {njnt} 个关节: {joint_names[:5]}..."
                else:
                    joint_info = "无法获取关节信息"
            except:
                joint_info = "无法获取关节信息"
            
            raise RuntimeError(
                f"\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"❌ SO101机器人场景未加载！\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"\n"
                f"🔍 问题: 后端加载的场景中没有SO101机器人\n"
                f"\n"
                f"💡 解决方案:\n"
                f"   1. 在 OrcaStudio/OrcaLab 中打开 SO101 场景:\n"
                f"      文件: assets/so101/so101_new_calib.xml（相对项目根目录）\n"
                f"   2. 在 OrcaStudio 中点击'运行'按钮 (Ctrl+G)\n"
                f"   3. 确认仿真服务器在 localhost:50051 运行\n"
                f"   4. 重新运行此脚本\n"
                f"\n"
                f"📋 {joint_info}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            ) from e
        
        # 手臂关节配置
        try:
            # 先获取所有可用关节用于调试
            try:
                import mujoco
                # 直接从 MuJoCo 模型获取所有关节名称
                if hasattr(self._env, 'gym') and hasattr(self._env.gym, '_mjModel') and self._env.gym._mjModel is not None:
                    mj_model = self._env.gym._mjModel
                    all_joints = []
                    for i in range(mj_model.njnt):
                        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                        if joint_name:
                            all_joints.append(joint_name)
                    _logger.info(f"🔍 场景中所有关节 ({len(all_joints)}个): {all_joints}")
                    # 检查是否有我们需要的关节（不带前缀）
                    for joint_name in config["arm"]["joint_names"]:
                        if joint_name in all_joints:
                            _logger.info(f"✅ 找到关节（不带前缀）: '{joint_name}'")
                        else:
                            _logger.warning(f"⚠️  未找到关节（不带前缀）: '{joint_name}'")
                    # 检查是否有带前缀的关节
                    agent_name = self._env._agent_names[id] if id < len(self._env._agent_names) else ""
                    for joint_name in config["arm"]["joint_names"]:
                        prefixed = f"{agent_name}_{joint_name}" if agent_name else joint_name
                        if prefixed in all_joints:
                            _logger.info(f"✅ 找到关节（带前缀）: '{prefixed}'")
                else:
                    _logger.warning("无法访问 MuJoCo 模型")
            except Exception as debug_e:
                _logger.warning(f"无法获取关节列表用于调试: {debug_e}")
            
            # 先尝试使用env.joint()解析名称（会自动添加前缀并解析）
            resolved_joint_names = []
            for joint_name in config["arm"]["joint_names"]:
                resolved = self._env.joint(joint_name, id)
                resolved_joint_names.append(resolved)
                _logger.info(f"关节名称解析: '{joint_name}' -> '{resolved}'")
            
            self._arm_joint_names = resolved_joint_names
            self._arm_joint_id = [self._env.model.joint_name2id(joint_name) 
                                 for joint_name in self._arm_joint_names]
            self._jnt_address = [self._env.jnt_qposadr(joint_name) 
                                for joint_name in self._arm_joint_names]
            self._jnt_dof = [self._env.jnt_dofadr(joint_name) 
                            for joint_name in self._arm_joint_names]
        except (KeyError, AttributeError) as e:
            # 尝试获取关节信息用于调试
            try:
                import mujoco
                if hasattr(self._env, 'gym') and hasattr(self._env.gym, '_mjModel') and self._env.gym._mjModel is not None:
                    njnt = self._env.gym._mjModel.njnt
                    available_joints = [mujoco.mj_id2name(self._env.gym._mjModel, mujoco.mjtObj.mjOBJ_JOINT, i) 
                                       for i in range(njnt)]
                    joint_info = f"当前可用关节 ({njnt}个): {available_joints[:10]}"
                else:
                    joint_info = "无法获取关节信息"
                    available_joints = []
            except:
                joint_info = "无法获取关节信息"
                available_joints = []
            
            _logger.error(f"❌ 找不到SO101关节。需要: {config['arm']['joint_names']}")
            raise RuntimeError(
                f"\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"❌ SO101关节未找到！\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"\n"
                f"🔍 需要的关节: {config['arm']['joint_names']}\n"
                f"📋 {joint_info}\n"
                f"\n"
                f"💡 请确保在 OrcaStudio 中加载了 SO101 场景！\n"
                f"   文件: assets/so101/so101_new_calib.xml（相对项目根目录）\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            ) from e
        
        # 执行器配置（SO101 只有 position 执行器，所以两者相同）
        self._arm_motor_names = [self._env.actuator(config["arm"]["motor_names"][i], id) 
                                for i in range(len(config["arm"]["motor_names"]))]
        self._arm_position_names = [self._env.actuator(config["arm"]["position_names"][i], id) 
                                   for i in range(len(config["arm"]["position_names"]))]
        
        # SO101 的 XML 只有 position 执行器，所以始终使用 position_names
        # 不需要 disable_actuators，因为没有重复的执行器
        self._arm_actuator_id = [self._env.model.actuator_name2id(name) 
                                for name in self._arm_position_names]
        
        self._neutral_joint_values = np.array(config["arm"]["neutral_joint_values"])
        self._ee_site = self._env.site(config["arm"]["ee_center_site_name"], id)
        
        # 夹爪配置
        self._gripper_actuator_name = self._env.actuator(config["gripper"]["actuator_name"], id)
        self._gripper_actuator_id = self._env.model.actuator_name2id(self._gripper_actuator_name)
        # 夹爪关节名（已解析，含 agent id 后缀）
        self._gripper_joint_name = self._env.joint(config["gripper"]["joint_name"], id)
        # 夹爪相关 body 的所有 geom ID（用于接触力查询）
        # SO101 夹爪涉及两个 body：gripper（基座）和 moving_jaw_so101_v1（活动下颚）
        _gripper_body_names = [
            self._env.body("gripper", id),
            self._env.body("moving_jaw_so101_v1", id),
        ]
        self._gripper_geom_ids = [
            info["GeomId"]
            for info in self._env.model.get_geom_dict().values()
            if info["BodyName"] in _gripper_body_names
        ]
        _logger.info(f"夹爪 geom IDs: {self._gripper_geom_ids}  body 列表: {_gripper_body_names}")
        # 位置 PD 控制器参数：从 MuJoCo 模型动态读取（不再硬编码）
        # MuJoCo position actuator: gainprm[0]=kp, biasprm[2]=-kv, forcerange=[lo,hi]
        try:
            _mj = self._env.gym._mjModel
            _aid = self._gripper_actuator_id
            self._gripper_kp = float(_mj.actuator_gainprm[_aid, 0])
            self._gripper_kv = float(-_mj.actuator_biasprm[_aid, 2])
            self._gripper_forcerange = (
                float(_mj.actuator_forcerange[_aid, 0]),
                float(_mj.actuator_forcerange[_aid, 1]),
            )
        except Exception as _e:
            _logger.warning(f"从模型读取夹爪 PD 参数失败，使用默认值: {_e}")
            self._gripper_kp = 998.22
            self._gripper_kv = 2.731
            self._gripper_forcerange = (-5.0, 5.0)
        _logger.info(f"夹爪 PD 参数: kp={self._gripper_kp}  kv={self._gripper_kv}  "
                     f"forcerange={self._gripper_forcerange}")
        
    def _setup_initial_info(self):
        """设置初始状态信息"""
        # 设置中性位置
        self.set_joint_neutral()
        self._env.mj_forward()
        
        # 获取控制范围
        self._all_ctrlrange = self._env.model.get_actuator_ctrlrange()
        arm_ctrl_range = [self._all_ctrlrange[actuator_id] for actuator_id in self._arm_actuator_id]
        arm_qpos_range = self._env.model.get_joint_qposrange(self._arm_joint_names)
        
        self._setup_action_range(arm_ctrl_range)
        self._setup_obs_scale(arm_qpos_range)
        
        # 读取末端执行器初始位姿（世界坐标）
        site_dict = self._env.query_site_pos_and_quat([self._ee_site])
        initial_ee_xpos_world = site_dict[self._ee_site]['xpos']
        initial_ee_xquat_world = site_dict[self._ee_site]['xquat']
        
        # 转换为基座坐标系
        self._initial_ee_xpos, self._initial_ee_xquat = self._global_to_local(
            initial_ee_xpos_world, initial_ee_xquat_world
        )
        
        # Xbox 控制：目标位姿初始化
        self._xbox_target_xpos = self._initial_ee_xpos.copy()
        self._xbox_target_xquat = self._initial_ee_xquat.copy()
        self._grasp_value = 0.0
        
        # Pico 控制：夹爪阈值状态
        self._pico_gripper_offset_rate_clip = 0.0
        
        self._env.mj_forward()
        
    def _setup_device(self):
        """设置控制设备（VR / Xbox / 主臂）"""
        self._pico_joystick = None
        self._xbox_joystick = None
        self._xbox_joystick_manager = None
        self._leader_arm = None

        if self._env.run_mode == RunMode.TELEOPERATION:
            if self._env.ctrl_device == ControlDevice.VR:
                if self._env.joystick is None:
                    raise ValueError("VR controller is not initialized.")
                self._pico_joystick = self._env.joystick[self.name]
            elif self._env.ctrl_device == ControlDevice.XBOX:
                if self._env.xbox_joystick is None:
                    raise ValueError("Xbox controller is not initialized.")
                self._xbox_joystick = self._env.xbox_joystick
                self._xbox_joystick_manager = self._env.xbox_joystick_manager
            elif self._env.ctrl_device == ControlDevice.LEADER_ARM:
                self._init_leader_arm()

    def _init_leader_arm(self):
        """初始化物理主臂（SO101Leader via Feetech 串口）"""
        import sys as _sys
        # 优先使用本地 lerobot 源码，避免重装 ML 依赖引发版本冲突
        _lerobot_src = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../lerobot/src"
        ))
        if _lerobot_src not in _sys.path:
            _sys.path.insert(0, _lerobot_src)

        try:
            from lerobot.motors import Motor, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus
        except ImportError as e:
            raise ImportError(
                "无法导入 lerobot.motors，请确认：\n"
                f"  1. 本地源码路径存在: {_lerobot_src}\n"
                "  2. 已安装串口依赖: pip install feetech-servo-sdk pyserial\n"
            ) from e

        leader_port = getattr(self._env, 'leader_arm_port', '/dev/ttyACM1')
        leader_calibration = getattr(self._env, 'leader_arm_calibration', None)

        # STS3215 编码器参数（无需标定文件，使用 normalize=False 读原始值）
        # 原始编码器：0~4095（12-bit），中心=2048 对应 0°
        # 角度公式：deg = (raw - 2048) * 360 / 4096
        self._leader_raw_center  = 2048    # STS3215 零位编码器值
        self._leader_raw_range   = 4096    # STS3215 编码器分辨率

        self._leader_bus = FeetechMotorsBus(
            port=leader_port,
            motors={
                "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
                "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
                "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
                "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
                "gripper":       Motor(6, "sts3215", MotorNormMode.DEGREES),
            },
            calibration=leader_calibration,
        )
        self._leader_bus.connect()
        # connect() 内部 configure() 会将 Operating_Mode 设为 POSITION 并重新使能扭矩
        # 必须在此处显式禁用扭矩，否则主臂关节被锁死，无法手动拖动
        self._leader_bus.disable_torque()

        # 从电机寄存器读取内置校准数据（Homing_Offset / Min / Max Position Limit）
        # 无需外部标定文件，STS3215 出厂或经 LeRobot 标定后均将数据存入电机
        try:
            motor_cal = self._leader_bus.read_calibration()
            self._leader_bus.calibration = motor_cal
            print(f"✅ 从电机读取校准数据成功，可使用 DEGREES 模式读取绝对角度")
        except Exception as e:
            print(f"⚠️  读取电机校准数据失败: {e}，将使用原始编码器值 + 增量映射")
            motor_cal = None

        # 主臂关节顺序（与 _arm_actuator_id 一致，不含 gripper）
        self._leader_joint_order = [
            "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
        ]

        # 每帧最大关节角变化（弧度），防止仿真跳变（~10°/帧）
        self._leader_max_delta_rad = np.radians(10.0)

        # 零点对齐状态（调用 align_leader_arm_zero() 后置 True）
        self._leader_home_raw     = None   # 主臂启动时的原始编码器值（零点参考）
        self._leader_sim_home_rad = None   # 对应时刻仿真从臂各关节角（弧度）
        self._leader_zero_aligned = False

        # 连接完成，零点对齐延迟到外部调用 align_leader_arm_zero()
        self._leader_arm = True  # 标记主臂已就绪（零点待对齐）
        print(f"✅ 主臂串口已连接（端口: {leader_port}）")
                
    def _raw_to_deg(self, raw_val: float) -> float:
        """STS3215 原始编码器值 → 角度（度），中心=2048 对应 0°"""
        return (raw_val - self._leader_raw_center) * 360.0 / self._leader_raw_range

    def _read_leader_raw(self):
        """读取主臂关节位置，返回 dict {joint_name: value}
        
        如果电机已加载校准数据，返回 DEGREES 归一化值（真实角度，度）。
        否则 fallback 返回原始编码器值（0~4095），由调用方用 _raw_to_deg 换算。
        """
        has_cal = (hasattr(self._leader_bus, 'calibration')
                   and self._leader_bus.calibration is not None
                   and len(self._leader_bus.calibration) > 0)
        return self._leader_bus.sync_read("Present_Position", normalize=has_cal)

    def align_leader_arm_zero(self):
        """
        主臂零点对齐（在 gym.make() 完成后调用）。

        读取主臂当前编码器原始值作为零点基准（_leader_home_raw）。
        同时记录此时仿真从臂的关节角（_leader_sim_home_rad）。
        后续帧按增量：sim_target = sim_home + Δ(leader_now - leader_home)。
        无需标定文件，无需用户手动摆姿态。
        """
        if self._leader_arm is None or not hasattr(self, '_leader_bus'):
            print("❌ align_leader_arm_zero: 主臂未初始化，跳过。")
            return

        print("\n" + "="*60)
        print("【主臂零点对齐】读取主臂当前位置作为零点基准…")

        try:
            raw = self._read_leader_raw()
        except Exception as e:
            print(f"❌ 读取主臂失败: {e}")
            return

        # 判断是否有校准数据（决定 raw 是度数还是编码器原始值）
        has_cal = (hasattr(self._leader_bus, 'calibration')
                   and self._leader_bus.calibration is not None
                   and len(self._leader_bus.calibration) > 0)
        self._leader_has_calibration = has_cal

        # ── 读取并记录主臂当前值（零点基准）────────────────────
        raw_vals = np.array([float(raw[j]) for j in self._leader_joint_order])
        self._leader_home_raw     = raw_vals   # 零点：有校准=度，无校准=编码器值
        self._leader_gripper_home_raw = float(raw.get("gripper",
                                                       self._leader_raw_center if not has_cal else 0.0))

        if has_cal:
            # 有校准：raw_vals 已是角度（度），直接换算弧度
            home_deg   = raw_vals
            target_rad = np.radians(home_deg)
            print(f"  ✅ 使用校准角度（DEGREES 模式）")
        else:
            # 无校准：raw_vals 是编码器值，换算为度
            home_deg   = np.array([self._raw_to_deg(v) for v in raw_vals])
            target_rad = np.radians(home_deg)
            print(f"  ⚠️  无校准，使用原始编码器值换算（相对零位可能有偏差）")

        # ── 将仿真从臂跳转到主臂当前姿态 ─────────────────────
        for i, act_id in enumerate(self._arm_actuator_id):
            lo, hi = self._all_ctrlrange[act_id]
            self._env.ctrl[act_id] = float(np.clip(target_rad[i], lo, hi))

        # 记录仿真从臂手臂基准（对齐后的实际 ctrl 值）
        self._leader_sim_home_rad = np.array([
            self._env.ctrl[self._arm_actuator_id[i]]
            for i in range(len(self._arm_actuator_id))
        ])

        # ── 夹爪：对齐仿真夹爪到物理夹爪当前位置 ────────────────
        gripper_raw_now = float(raw.get("gripper",
                                        self._leader_raw_center if not has_cal else 0.0))
        self._leader_gripper_home_raw = gripper_raw_now  # 有校准=度，无校准=编码器值

        g_min = self._all_ctrlrange[self._gripper_actuator_id][0]
        g_max = self._all_ctrlrange[self._gripper_actuator_id][1]

        # 与增量映射方向一致：sim_init = radians(physical_deg)
        # 物理打开（+47°）→ sim +0.82 rad（偏向打开端）
        # 物理关闭（-54°）→ sim -0.94 rad → 截断到 g_min（-0.1745，最关闭端）
        if has_cal:
            gripper_sim_init = float(np.clip(np.radians(gripper_raw_now), g_min, g_max))
        else:
            # 无校准：使用 ctrlrange 中间值作为安全初始值
            gripper_sim_init = float((g_min + g_max) / 2.0)

        # 与手臂关节相同：把仿真夹爪跳转到与物理一致的位置
        self._env.ctrl[self._gripper_actuator_id] = gripper_sim_init
        self._leader_gripper_sim_home = gripper_sim_init

        # ── 标记对齐完成 ─────────────────────────────────────────
        self._leader_zero_aligned = True

        print(f"  主臂零点角度(°): {np.round(home_deg, 1)}")
        print(f"  仿真从臂跳转到(°): {np.round(np.degrees(self._leader_sim_home_rad), 1)}")
        print(f"  夹爪物理零点: {gripper_raw_now:.1f}{'°' if has_cal else '(raw)'}  "
              f"→ 仿真初始: {gripper_sim_init:.3f} rad "
              f"({np.degrees(gripper_sim_init):.1f}°)  范围[{g_min:.4f},{g_max:.4f}]")
        print("✅ 零点对齐完成，开始主从控制\n" + "="*60 + "\n")

    def _setup_controller(self):
        """设置控制器（OSC 或 IK）"""
        if self._env.action_use_motor():
            # OSC 控制器
            self._controller_config = controller_config.load_config("osc_pose")
            self._controller_config["robot_name"] = self.name
            self._controller_config["sim"] = self._env.gym
            self._controller_config["eef_name"] = self._ee_site
            
            qpos_offsets, qvel_offsets, _ = self._env.query_joint_offsets(self._arm_joint_names)
            self._controller_config["joint_indexes"] = {
                "joints": self._arm_joint_names,
                "qpos": qpos_offsets,
                "qvel": qvel_offsets,
            }
            
            arm_ctrl_range = [self._all_ctrlrange[actuator_id] for actuator_id in self._arm_actuator_id]
            self._controller_config["actuator_range"] = arm_ctrl_range
            self._controller_config["policy_freq"] = self._env.control_freq
            self._controller_config["ndim"] = len(self._arm_joint_names)
            self._controller_config["control_delta"] = False
            
            self._controller = controller_factory(self._controller_config["type"], self._controller_config)
            self._controller.update_initial_joints(self._neutral_joint_values)
            
            self._gripper_offset_rate_clip = 0.0
        else:
            # IK 控制器
            self._inverse_kinematics_controller = InverseKinematicsController(
                self._env, 
                self._env.model.site_name2id(self._ee_site), 
                self._jnt_dof, 
                2e-1, 
                0.075
            )
            
    def set_joint_neutral(self) -> None:
        """设置关节到中性位置"""
        arm_joint_qpos = {}
        for name, value in zip(self._arm_joint_names, self._neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])
        self._env.set_joint_qpos(arm_joint_qpos)
        
    def set_init_ctrl(self) -> None:
        """设置初始控制信号"""
        if self._env.action_use_motor():
            return
        for i in range(len(self._arm_actuator_id)):
            self._env.ctrl[self._arm_actuator_id[i]] = self._neutral_joint_values[i]
            
    def on_reset_model(self) -> None:
        """重置模型"""
        self._reset_gripper()
        
    def _reset_gripper(self) -> None:
        """重置夹爪状态"""
        self._gripper_offset_rate_clip = 0.0
        self._pico_gripper_offset_rate_clip = 0.0
        
    def get_obs(self) -> dict:
        """获取观测"""
        ee_sites = self._env.query_site_pos_and_quat_B([self._ee_site], self._base_body_name)
        ee_xvalp, ee_xvalr = self._env.query_site_xvalp_xvalr_B([self._ee_site], self._base_body_name)
        
        arm_joint_values = self._get_arm_joint_values(self._arm_joint_names)
        arm_joint_velocities = self._get_arm_joint_velocities(self._arm_joint_names)
        
        self._obs = {
            "ee_pos": ee_sites[self._ee_site]["xpos"].flatten().astype(np.float32),
            "ee_quat": ee_sites[self._ee_site]["xquat"].flatten().astype(np.float32),
            "ee_vel_linear": ee_xvalp[self._ee_site].flatten().astype(np.float32),
            "ee_vel_angular": ee_xvalr[self._ee_site].flatten().astype(np.float32),
            "arm_joint_qpos": arm_joint_values.flatten().astype(np.float32),
            "arm_joint_qpos_sin": np.sin(arm_joint_values).flatten().astype(np.float32),
            "arm_joint_qpos_cos": np.cos(arm_joint_values).flatten().astype(np.float32),
            "arm_joint_vel": arm_joint_velocities.flatten().astype(np.float32),
            "grasp_value": np.array([self._grasp_value], dtype=np.float32),
        }
        
        scaled_obs = {key: self._obs[key] * self._obs_scale[key] for key in self._obs.keys()}
        return scaled_obs
        
    def _get_arm_joint_values(self, joint_names) -> np.ndarray:
        """获取关节位置"""
        qpos_dict = self._env.query_joint_qpos(joint_names)
        return np.array([qpos_dict[joint_name] for joint_name in joint_names]).flatten()
        
    def _get_arm_joint_velocities(self, joint_names) -> np.ndarray:
        """获取关节速度"""
        qvel_dict = self._env.query_joint_qvel(joint_names)
        return np.array([qvel_dict[joint_name] for joint_name in joint_names]).flatten()
        
    def _setup_obs_scale(self, arm_qpos_range) -> None:
        """设置观测缩放"""
        ee_xpos_scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        ee_xquat_scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        max_ee_linear_vel = 2.0
        max_ee_angular_vel = np.pi
        
        arm_qpos_scale = np.array([max(abs(qpos_range[0]), abs(qpos_range[1])) 
                                   for qpos_range in arm_qpos_range], dtype=np.float32)
        max_arm_joint_vel = np.pi
        
        self._obs_scale = {
            "ee_pos": 1.0 / ee_xpos_scale,
            "ee_quat": 1.0 / ee_xquat_scale,
            "ee_vel_linear": np.ones(3, dtype=np.float32) / max_ee_linear_vel,
            "ee_vel_angular": np.ones(3, dtype=np.float32) / max_ee_angular_vel,
            "arm_joint_qpos": 1.0 / arm_qpos_scale,
            "arm_joint_qpos_sin": np.ones(len(arm_qpos_scale), dtype=np.float32),
            "arm_joint_qpos_cos": np.ones(len(arm_qpos_scale), dtype=np.float32),
            "arm_joint_vel": np.ones(len(arm_qpos_scale), dtype=np.float32) / max_arm_joint_vel,
            "grasp_value": np.ones(1, dtype=np.float32),
        }
        
    def _setup_action_range(self, arm_ctrl_range) -> None:
        """设置动作空间范围"""
        gripper_ctrl_range = [self._all_ctrlrange[self._gripper_actuator_id]]
        self._action_range = np.concatenate(
            [
                [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]],  # EE pose
                arm_ctrl_range,   # 手臂关节控制
                gripper_ctrl_range,  # 夹爪（完整 ctrlrange）
            ],
            dtype=np.float32,
            axis=0
        )
        
        self._action_range_min = self._action_range[:, 0]
        self._action_range_max = self._action_range[:, 1]
        
    def on_teleoperation_action(self) -> np.ndarray:
        """遥操作动作"""
        # ── 主臂模式：纯关节空间，完全绕过 IK ──────────────────
        if self._leader_arm is not None:
            self._process_leader_arm_move()
            # 构建 action（关节角直接来自 ctrl，EE pose 填 0）
            ctrl = np.asarray(
                self._env.ctrl[self._arm_actuator_id], dtype=np.float32
            )
            action = np.concatenate([
                np.zeros(6, dtype=np.float32),                       # 0-5: EE pose（不使用）
                ctrl,                                                 # 6-N: joint ctrl
                np.array([self._grasp_value], dtype=np.float32)      # N+1: gripper
            ]).flatten()
            return action

        if self._pico_joystick is not None:
            ee_xpos, ee_xquat = self._processe_pico_joystick_move()
            self._process_pico_joystick_operation()
        elif self._xbox_joystick is not None:
            ee_xpos, ee_xquat = self._process_xbox_joystick_move()
            self._process_xbox_joystick_operation()
        else:
            return np.zeros(len(self._arm_joint_names) + 7)  # EE pose + joints + gripper
        
        # 转换四元数为轴角
        grasp_axisangle = transform_utils.quat2axisangle(
            np.array([ee_xquat[1], ee_xquat[2], ee_xquat[3], ee_xquat[0]])
        )
        
        # 转换为全局坐标系
        ee_xpos_global, ee_xquat_global = self._local_to_global(ee_xpos, ee_xquat)
        grasp_axisangle_global = transform_utils.quat2axisangle(
            np.array([ee_xquat_global[1], ee_xquat_global[2], ee_xquat_global[3], ee_xquat_global[0]])
        )

        # ── 全局坐标系调试输出（每50步，仅 Pico 模式）──────────────────
        if self._pico_joystick is not None:
            if not hasattr(self, '_tele_print_counter'):
                self._tele_print_counter = 0
            self._tele_print_counter += 1
            if self._tele_print_counter % 100 == 0:
                base_pos, _, base_quat = self._env.get_body_xpos_xmat_xquat(self._base_body_name)
                e_base = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]
                                     ).as_euler('xyz', degrees=True)
                e_global = R.from_quat([ee_xquat_global[1], ee_xquat_global[2],
                                        ee_xquat_global[3], ee_xquat_global[0]]
                                       ).as_euler('xyz', degrees=True)
                print(f"  ┌─ 【C】基座坐标系（world）")
                print(f"  │   基座位置    : [{base_pos[0]:+.4f}  {base_pos[1]:+.4f}  {base_pos[2]:+.4f}]")
                print(f"  │   基座旋转    : roll={e_base[0]:+7.2f}°  pitch={e_base[1]:+7.2f}°  yaw={e_base[2]:+7.2f}°")
                print(f"  └─ 【D】目标末端姿态（世界坐标系 W，送入控制器）")
                print(f"      全局位置    : [{ee_xpos_global[0]:+.4f}  {ee_xpos_global[1]:+.4f}  {ee_xpos_global[2]:+.4f}]")
                print(f"      全局旋转    : roll={e_global[0]:+7.2f}°  pitch={e_global[1]:+7.2f}°  yaw={e_global[2]:+7.2f}°")
                print(f"      轴角(global): [{grasp_axisangle_global[0]:+.4f}  {grasp_axisangle_global[1]:+.4f}  {grasp_axisangle_global[2]:+.4f}]")
        
        # 控制器执行
        if self._env.action_type in [ActionType.END_EFFECTOR_OSC, ActionType.JOINT_MOTOR]:
            action_ee = np.concatenate([ee_xpos_global, grasp_axisangle_global])
            self._controller.set_goal(action_ee)
            ctrl = self._controller.run_controller()
            self._set_arm_ctrl(self._arm_actuator_id, ctrl)
        else:
            ctrl = self.set_arm_position_ctrl(ee_xpos_global, ee_xquat_global)
        
        # 构建动作（基座坐标系，用于数据记录）
        action_B = np.concatenate([ee_xpos, grasp_axisangle])
        ctrl = np.asarray(ctrl, dtype=np.float32)
        action = np.concatenate([
            np.asarray(action_B, dtype=np.float32),              # 0-5: EE pose
            ctrl,                                                 # 6-N: joint control
            np.array([self._grasp_value], dtype=np.float32)      # N+1: gripper
        ]).flatten()
        
        return action
        
    # ---------------------------------------------------------------
    # SO101 Pico 缩放系数
    #
    # SO101 机械臂参数：
    #   - 最大臂展：约 0.55–0.60 m
    #   - 有效工作半径：0.3–0.5 m
    #   - 上臂/前臂各约 0.22–0.25 m，腕部约 0.10 m
    #
    # Pico 手柄在 VR 空间中，人手的典型运动幅度约 0.6–1.0 m。
    # 为使末端在机械臂工作空间内运动，需将手柄位移按比例压缩：
    #
    #   PICO_POSITION_SCALE = 机器人工作半径 / 人手典型运动幅度
    #                       ≈ 0.4 m / 0.8 m  ≈  0.5
    #
    # 旋转通常保持 1:1（VR 旋转增量已是相对增量，无需压缩），
    # 若感觉旋转过于灵敏可将 PICO_ROTATION_SCALE 调低至 0.5–0.8。
    # ---------------------------------------------------------------
    PICO_POSITION_SCALE = 0.5   # 位置缩放：手柄移动 1 m → 末端移动 0.5 m
    PICO_ROTATION_SCALE = 0.1   # 旋转缩放：抑制生理耦合带来的寄生旋转（可按需调整，0=完全忽略旋转）

    def _processe_pico_joystick_move(self) -> tuple:
        """
        处理 Pico VR 手柄的移动输入（单臂版本）

        使用右手控制器控制末端执行器：
        - 右手位置 → 末端执行器位置（叠加到初始位置，乘以 PICO_POSITION_SCALE）
        - 右手旋转 → 末端执行器旋转（仅追踪相对于连接时刻的旋转增量，乘以 PICO_ROTATION_SCALE）

        【旋转处理说明】
        Unity 端发送的是手柄在 VR 世界空间中的**绝对旋转**，而非相对旋转增量。
        因此 PC 端在首次收到数据时记录手柄初始旋转作为"零点"，
        后续每帧计算真正的增量：delta = inv(initial_hand_rot) × current_hand_rot
        再将该增量叠加到机械臂末端初始姿态。

        缩放系数基于 SO101 有效工作半径（0.3–0.5 m）和人手典型运动幅度（0.6–1.0 m）确定。
        可在类变量 PICO_POSITION_SCALE / PICO_ROTATION_SCALE 中调整。
        """
        if self._pico_joystick.is_reset_pos():
            self._pico_joystick.set_reset_pos(False)
            self._reset_gripper()
            # 重置时同时清空手柄初始旋转，下次重新校准
            if hasattr(self, '_pico_initial_hand_rotation'):
                del self._pico_initial_hand_rotation

        transform_list = self._pico_joystick.get_transform_list()
        if transform_list is None:
            return self._initial_ee_xpos, self._initial_ee_xquat

        # 使用右手控制单臂
        right_abs_position, right_abs_rotation = self._pico_joystick.get_right_relative_move(transform_list)

        # ---- 位置：Unity 已发送相对初始连接时刻的位移，直接缩放使用 ----
        scaled_position = right_abs_position * self.PICO_POSITION_SCALE
        ee_xpos = self._initial_ee_xpos + scaled_position
        ee_xpos[2] = np.max((0.0, ee_xpos[2]))  # 确保不低于地面

        # ---- 旋转：Unity 发送的是绝对旋转，PC 端计算相对首帧的真实增量 ----
        # 首次收到数据时，记录手柄此刻的绝对旋转作为"零点"
        if not hasattr(self, '_pico_initial_hand_rotation'):
            self._pico_initial_hand_rotation = right_abs_rotation.copy()
            print(f"\n[PICO旋转校准] 记录手柄初始旋转零点: "
                  f"{R.from_quat([right_abs_rotation[1], right_abs_rotation[2], right_abs_rotation[3], right_abs_rotation[0]]).as_euler('xyz', degrees=True)}")

        # 计算真正的旋转增量：delta = inv(initial) × current
        initial_inv = rotations.quat_conjugate(self._pico_initial_hand_rotation)
        true_delta_rotation = rotations.quat_mul(initial_inv, right_abs_rotation)

        # ---- 旋转缩放（对增量做 slerp 插值缩短旋转幅度）----
        if self.PICO_ROTATION_SCALE < 1.0:
            identity = np.array([1.0, 0.0, 0.0, 0.0])
            scaled_rotation = identity + self.PICO_ROTATION_SCALE * (true_delta_rotation - identity)
            norm = np.linalg.norm(scaled_rotation)
            scaled_rotation = scaled_rotation / norm if norm > 1e-8 else identity
        else:
            scaled_rotation = true_delta_rotation

        ee_xquat = rotations.quat_mul(self._initial_ee_xquat, scaled_rotation)

        # ---------------------------------------------------------------
        # 调试输出（每50步），显示手柄姿态变化和末端位姿增量
        # ---------------------------------------------------------------
        if not hasattr(self, '_pico_print_counter'):
            self._pico_print_counter = 0
            self._pico_last_ee_xpos = self._initial_ee_xpos.copy()
            self._pico_last_ee_xquat = self._initial_ee_xquat.copy()
        self._pico_print_counter += 1

        if self._pico_print_counter % 100 == 0:
            # ── 欧拉角工具函数 (wxyz → xyz欧拉, 单位:度) ──────────────────
            def q2e(q_wxyz):
                return R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
                                   ).as_euler('xyz', degrees=True)

            e_abs   = q2e(right_abs_rotation)        # Pico 原始旋转
            e_init  = q2e(self._pico_initial_hand_rotation)  # 首帧零点
            e_delta = q2e(true_delta_rotation)        # 计算的旋转增量
            e_sc    = q2e(scaled_rotation)            # 缩放后旋转增量
            e_ee    = q2e(ee_xquat)                   # 目标末端旋转（B系）
            e_ee0   = q2e(self._initial_ee_xquat)     # 初始末端旋转（B系）

            print(f"\n{'═'*62}")
            print(f"[PICO完整链路] Step {self._pico_print_counter}")
            print(f"  ┌─ 【A】Pico输入（Unity→MuJoCo世界坐标系）")
            print(f"  │   位置偏移    : [{right_abs_position[0]:+.4f}  {right_abs_position[1]:+.4f}  {right_abs_position[2]:+.4f}] m")
            print(f"  │   原始旋转    : roll={e_abs[0]:+7.2f}°  pitch={e_abs[1]:+7.2f}°  yaw={e_abs[2]:+7.2f}°")
            print(f"  │   首帧零点    : roll={e_init[0]:+7.2f}°  pitch={e_init[1]:+7.2f}°  yaw={e_init[2]:+7.2f}°")
            print(f"  │   旋转增量    : roll={e_delta[0]:+7.2f}°  pitch={e_delta[1]:+7.2f}°  yaw={e_delta[2]:+7.2f}°")
            print(f"  │   缩放旋转(×{self.PICO_ROTATION_SCALE}): roll={e_sc[0]:+7.2f}°  pitch={e_sc[1]:+7.2f}°  yaw={e_sc[2]:+7.2f}°")
            print(f"  ├─ 【B】目标末端姿态（机器人基座坐标系 B）")
            print(f"  │   初始位置    : [{self._initial_ee_xpos[0]:+.4f}  {self._initial_ee_xpos[1]:+.4f}  {self._initial_ee_xpos[2]:+.4f}]")
            print(f"  │   初始旋转    : roll={e_ee0[0]:+7.2f}°  pitch={e_ee0[1]:+7.2f}°  yaw={e_ee0[2]:+7.2f}°")
            print(f"  │   目标位置    : [{ee_xpos[0]:+.4f}  {ee_xpos[1]:+.4f}  {ee_xpos[2]:+.4f}]")
            print(f"  │   目标旋转    : roll={e_ee[0]:+7.2f}°  pitch={e_ee[1]:+7.2f}°  yaw={e_ee[2]:+7.2f}°")
            print(f"  │   ⚠ 注意: Pico位置增量(A世界系) 直接加到 初始位置(B基座系) ← 若基座有旋转则此处有误")
            print(f"  └─ 【调试提示】若出现旋转问题，重点检查以下项：")
            print(f"       1. 缩放旋转是否接近0？(当前={np.linalg.norm(e_sc):.1f}°)")
            print(f"       2. 手柄位置增量是否已转换到基座坐标系再相加？")
            print(f"{'═'*62}\n")

            self._pico_last_ee_xpos  = ee_xpos.copy()
            self._pico_last_ee_xquat = ee_xquat.copy()

        return ee_xpos, ee_xquat

    def _process_pico_joystick_operation(self) -> None:
        """
        处理 Pico VR 手柄按键操作（单臂版本）
        
        按键映射：
        - 右手 Trigger（扳机）：夹爪闭合（指数灵敏度）
        - 右手 B 键（secondary）：调低夹爪最小值（按住缓慢闭合）
        - 右手 A 键（primary）：复位夹爪最小值
        - 左手 Grip（握持键）+ 松开：NOT_STARTED → GET_READY → BEGIN
        - 左手 Grip（任务进行中）：→ RETRY（重置）
        - 右手 Grip（任务进行中）：→ SUCCESS（完成）
        """
        joystick_state = self._pico_joystick.get_key_state()
        if joystick_state is None:
            return

        # === 夹爪控制（右手 Trigger）===
        trigger_value = joystick_state["rightHand"]["triggerValue"]  # [0, 1]
        k = np.e
        adjusted_value = (np.exp(k * trigger_value) - 1) / (np.exp(k) - 1)  # 指数映射 [0,1] → [0,1]

        # B 键（secondary）：缓慢降低夹爪上限（越按越紧）
        if joystick_state["rightHand"]["secondaryButtonPressed"]:
            self._pico_gripper_offset_rate_clip -= 0.5 * self._env.dt
            self._pico_gripper_offset_rate_clip = np.clip(self._pico_gripper_offset_rate_clip, -1.0, 0.0)
        # A 键（primary）：复位夹爪上限（回到完全打开）
        elif joystick_state["rightHand"]["primaryButtonPressed"]:
            self._pico_gripper_offset_rate_clip = 0.0

        offset_rate = -adjusted_value
        offset_rate = np.clip(offset_rate, -1.0, self._pico_gripper_offset_rate_clip)
        self.set_gripper_ctrl(offset_rate)
        self._grasp_value = offset_rate

        # === 任务状态控制（仅第一个 agent）===
        if self.id != 0:
            return
        self._set_pico_task_status(joystick_state)

    def _set_pico_task_status(self, joystick_state) -> None:
        """
        通过 Pico Grip 按键控制任务状态

        流程：
          左手 Grip（按住）→ GET_READY → 松开 → BEGIN（开始录制）
          任务进行中：
            左手 Grip 按下 → RETRY（放弃当前，重新录制）
            右手 Grip 按下 → SUCCESS（完成当前录制）
        """
        left_grip = joystick_state["leftHand"]["gripButtonPressed"]
        right_grip = joystick_state["rightHand"]["gripButtonPressed"]

        if self._env.task_status == TaskStatus.NOT_STARTED and left_grip:
            print("\n⏳ 准备中…松开左手 Grip 开始录制")
            self._env.set_task_status(TaskStatus.GET_READY)
        elif self._env.task_status == TaskStatus.GET_READY and not left_grip:
            print("\n▶️  开始录制！")
            self._env.set_task_status(TaskStatus.BEGIN)
        elif self._env.task_status == TaskStatus.BEGIN and left_grip:
            print("\n🔄 放弃当前录制，重置环境…")
            self._env.set_task_status(TaskStatus.RETRY)
        elif self._env.task_status == TaskStatus.BEGIN and right_grip:
            print("\n✅ 录制完成！")
            self._env.set_task_status(TaskStatus.SUCCESS)

    def _process_xbox_joystick_move(self) -> tuple:
        """
        处理 Xbox 手柄的移动输入（单臂版本）

        控制模式：
          - 位置（XYZ）：IK 控制（仅位置误差，不解算旋转）
          - 旋转：直接增量控制关节角，完全绕过 IK

        按键映射：
          - 左摇杆 X/Y        ：末端左右/前后平移（IK）
          - LT                ：末端向下
          - RT                ：末端向上
          - 右摇杆上下        ：关节4 Wrist Flex（腕部俯仰）
          - 右摇杆左右        ：关节5 Wrist Roll（腕部滚转）
          - R3（按下右摇杆）  ：切换为底座模式
            + 右摇杆左右      ：关节1 Base（底座旋转）
        """
        # 更新手柄状态
        self._xbox_joystick_manager.update()
        joystick_state = self._xbox_joystick.get_state()

        if joystick_state is None:
            if not hasattr(self, '_warned_no_joystick'):
                print("[警告] Xbox手柄状态为None")
                self._warned_no_joystick = True
            return self._initial_ee_xpos, self._initial_ee_xquat

        # 获取当前真实位姿（基座坐标系）
        ee_sites = self._env.query_site_pos_and_quat_B([self._ee_site], self._base_body_name)
        current_ee_xpos = ee_sites[self._ee_site]["xpos"]
        current_ee_xquat = ee_sites[self._ee_site]["xquat"]

        # ── 控制参数 ──────────────────────────────────────────────
        dt = self._env.dt
        MOVE_SPEED   = 3.125   # m/s（末端平移速度）
        JOINT_SPEED  = 0.8     # rad/s（关节旋转速度）
        CTRL_MIN     = 0.5     # 平移摇杆死区
        ROTATE_MIN   = 0.45    # 旋转摇杆死区（右摇杆左右适当加大）

        # ── 读取输入 ──────────────────────────────────────────────
        r3_pressed = joystick_state["buttons"]["RightStick"]

        # 左摇杆（末端平移）
        move_x_raw = -joystick_state["axes"]["LeftStickX"]
        move_y_raw = -joystick_state["axes"]["LeftStickY"]
        move_x = move_x_raw if abs(move_x_raw) >= CTRL_MIN else 0.0
        move_y = move_y_raw if abs(move_y_raw) >= CTRL_MIN else 0.0

        # Z 轴：LT=向下，RT=向上
        lt_normalized = max(0.0, (joystick_state["axes"]["LT"] + 1) * 0.5)
        rt_normalized = max(0.0, (joystick_state["axes"]["RT"] + 1) * 0.5)
        if lt_normalized < CTRL_MIN: lt_normalized = 0.0
        if rt_normalized < CTRL_MIN: rt_normalized = 0.0
        if rt_normalized > 0:
            move_z = rt_normalized
        elif lt_normalized > 0:
            move_z = -lt_normalized
        else:
            move_z = 0.0

        # 右摇杆（直接关节控制）
        right_x_raw = joystick_state["axes"]["RightStickX"]
        right_y_raw = joystick_state["axes"]["RightStickY"]
        right_x = right_x_raw if abs(right_x_raw) >= ROTATE_MIN else 0.0
        right_y = right_y_raw if abs(right_y_raw) >= ROTATE_MIN else 0.0

        # ── 位置控制（IK，仅位置误差）────────────────────────────
        pos_ctrl = np.array([move_y, move_x, move_z])
        for i in range(3):
            if abs(pos_ctrl[i]) < CTRL_MIN:
                pos_ctrl[i] = 0.0
        delta_pos = pos_ctrl * MOVE_SPEED * dt
        ee_xpos = current_ee_xpos + delta_pos
        ee_xpos[2] = np.max((0.0, ee_xpos[2]))

        # ── 直接关节角控制（旋转，绕过 IK）──────────────────────
        # 关节索引（0-based）：
        #   [0] shoulder_pan  = 关节1（底座）
        #   [1] shoulder_lift = 关节2
        #   [2] elbow_flex    = 关节3
        #   [3] wrist_flex    = 关节4（腕部俯仰）
        #   [4] wrist_roll    = 关节5（腕部滚转）
        delta_j = JOINT_SPEED * dt
        if r3_pressed:
            # R3 + 右摇杆左右 → 关节1 底座旋转
            if abs(right_x) > 0:
                self._env.ctrl[self._arm_actuator_id[0]] += right_x * delta_j
        else:
            # 右摇杆上下 → 关节4 Wrist Flex
            if abs(right_y) > 0:
                self._env.ctrl[self._arm_actuator_id[3]] += right_y * delta_j
            # 右摇杆左右 → 关节5 Wrist Roll
            if abs(right_x) > 0:
                self._env.ctrl[self._arm_actuator_id[4]] += right_x * delta_j

        # 关节角度限幅
        self._env.ctrl = np.clip(
            self._env.ctrl, self._all_ctrlrange[:, 0], self._all_ctrlrange[:, 1]
        )

        # ── 调试输出（每20步）────────────────────────────────────
        if not hasattr(self, '_joystick_print_counter'):
            self._joystick_print_counter = 0
        self._joystick_print_counter += 1
        if self._joystick_print_counter % 20 == 0:
            j1 = self._env.ctrl[self._arm_actuator_id[0]]
            j4 = self._env.ctrl[self._arm_actuator_id[3]]
            j5 = self._env.ctrl[self._arm_actuator_id[4]]
            dm = delta_pos * 1000
            print(f"\n[XBOX控制] Step {self._joystick_print_counter}")
            print(f"  位置增量: [{dm[0]:.1f}, {dm[1]:.1f}, {dm[2]:.1f}] mm")
            print(f"  J1(底座)={np.degrees(j1):+.1f}°  "
                  f"J4(WristFlex)={np.degrees(j4):+.1f}°  "
                  f"J5(WristRoll)={np.degrees(j5):+.1f}°")

        # 返回位置用于IK；传回当前四元数 → IK旋转误差=0 → 纯位置IK
        return ee_xpos, current_ee_xquat
        
    def _process_leader_arm_move(self) -> None:
        """
        读取物理主臂关节角度，写入仿真从臂关节 ctrl（增量映射）。

        控制逻辑：
          1. sync_read 读取主臂原始编码器值（RAW，无需标定）
          2. Δraw = raw_now - home_raw  → Δrad = Δdeg × π/180
          3. target_rad = sim_home_rad + Δrad
          4. 每帧限幅（防通信异常跳变）+ ctrlrange 夹紧
          5. 写入 self._env.ctrl
          6. 夹爪：raw 增量 → ctrlrange 线性映射
        """
        # 零点对齐完成前跳过
        if not getattr(self, '_leader_zero_aligned', False):
            return

        # ── 读主臂 ────────────────────────────────────────────────
        try:
            raw = self._read_leader_raw()
        except Exception as e:
            print(f"[主臂] 读取失败: {e}")
            return

        # ── 5 个手臂关节：计算目标角度 ───────────────────────
        raw_now = np.array([float(raw[j]) for j in self._leader_joint_order])
        if getattr(self, '_leader_has_calibration', False):
            # 有校准：raw_now 已是度数，直接换算弧度
            now_deg   = raw_now
            delta_deg = now_deg - self._leader_home_raw   # 当前-零点=增量
        else:
            # 无校准：编码器原始值→增量→度
            delta_raw = raw_now - self._leader_home_raw
            delta_deg = delta_raw * 360.0 / self._leader_raw_range
        delta_rad  = np.radians(delta_deg)
        target_rad = self._leader_sim_home_rad + delta_rad

        # ── 每帧限幅（防跳变）────────────────────────────────
        for i, act_id in enumerate(self._arm_actuator_id):
            current = self._env.ctrl[act_id]
            lo, hi  = self._all_ctrlrange[act_id]
            stepped = current + np.clip(
                target_rad[i] - current,
                -self._leader_max_delta_rad,
                self._leader_max_delta_rad
            )
            self._env.ctrl[act_id] = float(np.clip(stepped, lo, hi))

        # ── 夹爪：增量映射（与手臂关节方向相同，无需取反）──────────
        # 物理夹爪校准后：打开=正角度(+47°)，关闭=负角度(-54°)
        # 仿真 ctrlrange=[-0.1745, 1.7453]：最小值≈关闭，最大值≈打开
        # 物理角度方向与仿真 ctrl 方向一致，直接换算弧度即可
        gripper_raw_now = float(raw.get("gripper", self._leader_gripper_home_raw))
        gripper_delta   = gripper_raw_now - self._leader_gripper_home_raw
        if getattr(self, '_leader_has_calibration', False):
            # 有校准：gripper_delta 已是度数增量，直接换算
            gripper_delta_rad = np.radians(gripper_delta)
        else:
            # 无校准：gripper_delta 是编码器增量，换算为弧度
            gripper_delta_rad = np.radians(gripper_delta * 360.0 / self._leader_raw_range)
        gripper_target = self._leader_gripper_sim_home + gripper_delta_rad
        g_min = self._all_ctrlrange[self._gripper_actuator_id][0]
        g_max = self._all_ctrlrange[self._gripper_actuator_id][1]
        gripper_ctrl_final = float(np.clip(gripper_target, g_min, g_max))
        self._env.ctrl[self._gripper_actuator_id] = gripper_ctrl_final
        self._grasp_value = gripper_ctrl_final

        # ── 夹爪专项日志：每帧检测变化，超过阈值立即打印 ─────
        if not hasattr(self, '_gripper_last_ctrl'):
            self._gripper_last_ctrl = gripper_ctrl_final
        gripper_change = abs(gripper_ctrl_final - self._gripper_last_ctrl)
        if gripper_change > 0.03:   # 阈值 ~1.7°，过滤噪声
            clipped = (gripper_target < g_min or gripper_target > g_max)
            print(f"[夹爪] raw={gripper_raw_now:.1f}° "
                  f"home={self._leader_gripper_home_raw:.1f}° "
                  f"Δ={gripper_delta:+.1f}° "
                  f"delta_rad={gripper_delta_rad:+.4f} "
                  f"sim_home={self._leader_gripper_sim_home:.4f} "
                  f"target={gripper_target:.4f} "
                  f"ctrl={gripper_ctrl_final:.4f} "
                  f"{'⚠️ CLIPPED' if clipped else '✅'}", flush=True)
        self._gripper_last_ctrl = gripper_ctrl_final

        # ── 调试输出（每30帧打印一次）────────────────────────
        if not hasattr(self, '_leader_print_counter'):
            self._leader_print_counter = 0
        self._leader_print_counter += 1
        if self._leader_print_counter % 30 == 0:
            sim_now_deg = np.degrees([
                self._env.ctrl[self._arm_actuator_id[i]]
                for i in range(len(self._arm_actuator_id))
            ])
            cal_mode = "DEGREES(校准)" if getattr(self, '_leader_has_calibration', False) else "RAW(无校准)"
            print(f"\n[主臂→仿真] Step {self._leader_print_counter}  [{cal_mode}]")
            if getattr(self, '_leader_has_calibration', False):
                print(f"  主臂当前(°): {np.round(raw_now, 1)}")
            else:
                print(f"  主臂 RAW   : {np.round(raw_now, 0).astype(int)}")
            print(f"  主臂 Δ(°)  : {np.round(delta_deg, 1)}")
            print(f"  仿真目标(°): {np.round(np.degrees(target_rad), 1)}")
            print(f"  仿真实际(°): {np.round(sim_now_deg, 1)}")
            print(f"  夹爪当前   : {gripper_raw_now:.1f}°  Δ={gripper_delta:.2f}°  "
                  f"delta_rad={gripper_delta_rad:.4f}  sim_home={self._leader_gripper_sim_home:.4f}  "
                  f"target={gripper_target:.4f}  ctrl={gripper_ctrl_final:.4f}  "
                  f"范围[{g_min:.4f},{g_max:.4f}]")

            # ── 保存待打印状态，由 on_post_simulation() 在仿真后输出 ─
            # 原因：q/qv/actuator_force 必须在 do_simulation 之后才与
            # gripper_ctrl_final 属于同一帧，否则公式和仿真器数据错位。
            self._post_sim_log = {
                "step":          self._leader_print_counter,
                "gripper_ctrl":  gripper_ctrl_final,
            }

    def on_post_simulation(self) -> None:
        """
        在 do_simulation() 完成后由 so101_env.step() 调用。
        此时 q / qv / actuator_force 与 gripper_ctrl 属于同一帧，
        公式和仿真器读值可以正确对比。
        """
        pending = getattr(self, '_post_sim_log', None)
        if pending is None:
            return
        self._post_sim_log = None          # 消费后清空，避免重复打印

        step           = pending["step"]
        gripper_ctrl   = pending["gripper_ctrl"]

        # ── 夹爪力：仿真器读值 vs 公式 ────────────────────────────
        try:
            q_dict  = self._env.query_joint_qpos([self._gripper_joint_name])
            qv_dict = self._env.query_joint_qvel([self._gripper_joint_name])
            q  = float(q_dict[self._gripper_joint_name])
            qv = float(qv_dict[self._gripper_joint_name])

            tau_sim = None
            if hasattr(self._env, 'query_actuator_torques'):
                torques = self._env.query_actuator_torques([self._gripper_actuator_name])
                tau_sim = float(torques[self._gripper_actuator_name][0])
            elif hasattr(self._env, 'query_actuator_force'):
                all_forces = self._env.query_actuator_force()
                tau_sim = float(all_forces[self._gripper_actuator_id])

            # 公式：使用本帧真实 ctrl（与仿真器同步）
            tau_calc = self._gripper_kp * (gripper_ctrl - q) - self._gripper_kv * qv
            tau_calc_clipped = float(np.clip(tau_calc,
                                             self._gripper_forcerange[0],
                                             self._gripper_forcerange[1]))
            clipped_flag = "⚠️ CLIPPED" if abs(tau_calc) > abs(tau_calc_clipped) + 1e-6 else "✅"

            sim_str  = f"{tau_sim:.3f} N·m" if tau_sim is not None else "N/A"
            err_str  = ""
            if tau_sim is not None and abs(tau_calc_clipped) > 1e-4:
                err_pct = abs(tau_sim - tau_calc_clipped) / (abs(tau_calc_clipped) + 1e-9) * 100
                err_str = f"  误差={err_pct:.1f}%"
            print(f"  [夹爪力@仿后] ctrl={np.degrees(gripper_ctrl):.2f}°  "
                  f"q={np.degrees(q):.2f}°  q̇={np.degrees(qv):.2f}°/s  "
                  f"τ_仿真器={sim_str}  "
                  f"τ_公式={tau_calc:.3f}→{tau_calc_clipped:.3f} N·m  "
                  f"forcerange=[{self._gripper_forcerange[0]}, {self._gripper_forcerange[1]}]  "
                  f"{clipped_flag}{err_str}")
        except Exception as fe:
            print(f"  [夹爪力@仿后] 查询失败: {fe}")

        # ── 接触摩擦力 ────────────────────────────────────────────
        try:
            contact_list = self._env.query_contact_simple()
            gripper_contact_ids = [
                c["ID"] for c in contact_list
                if c.get("Geom1") in self._gripper_geom_ids
                or c.get("Geom2") in self._gripper_geom_ids
            ]
            if gripper_contact_ids:
                force_dict = self._env.query_contact_force(gripper_contact_ids)
                # mj_contactForce 在接触坐标系中返回 6 维向量：
                #   f[0]     → 法向力（沿接触法线，N）
                #   f[1:3]   → 切向摩擦力（接触平面内两分量，N）
                #   f[3:6]   → 扭转力矩（torsional torque，N·m）
                total_normal   = sum(float(abs(f[0]))              for f in force_dict.values())
                total_friction = sum(float(np.linalg.norm(f[1:3])) for f in force_dict.values())
                total_torque   = sum(float(np.linalg.norm(f[3:6])) for f in force_dict.values())
                print(f"  [接触力] 接触点数={len(gripper_contact_ids)}  "
                      f"法向力={total_normal:.3f} N  "
                      f"切向摩擦力={total_friction:.3f} N  "
                      f"扭转力矩={total_torque:.4f} N·m")
            else:
                print(f"  [接触力] 无接触")
        except Exception as ce:
            print(f"  [接触力] 查询失败: {ce}")

    def calc_rotate_matrix(self, yaw, pitch, roll):
        """计算旋转矩阵"""
        Rz_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])
        Ry_pitch = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rx_roll = np.array([
            [1, 0, 0],
            [0,  np.cos(roll), -np.sin(roll)],
            [0,  np.sin(roll),  np.cos(roll)]
        ])
        return np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))
    
    def _process_xbox_joystick_operation(self) -> None:
        """
        处理 Xbox 手柄的按钮操作（单臂版本）
        
        按键映射：
        - LB：夹爪开合切换
        - Start：开始任务
        - A：完成任务
        - B：重置任务
        """
        joystick_state = self._xbox_joystick.get_state()
        if joystick_state is None:
            return
        
        # 初始化状态变量
        if not hasattr(self, '_grasp_state'):
            self._grasp_state = False
            self._last_lb_pressed = False
            self._last_start_pressed = False
            self._last_a_pressed = False
            self._last_b_pressed = False
        
        # LB边沿触发：夹爪开合切换
        lb_pressed = joystick_state["buttons"]["LB"]
        if lb_pressed and not self._last_lb_pressed:
            self._grasp_state = not self._grasp_state
            print(f"[LB切换] 夹爪: {'闭合' if self._grasp_state else '松开'}")
        self._last_lb_pressed = lb_pressed
        
        # 设置夹爪值
        self._grasp_value = -1.0 if self._grasp_state else 0.0
        self.set_gripper_ctrl(self._grasp_value)
        
        # 任务状态控制（仅第一个agent）
        if self.id != 0:
            return
        
        start_pressed = joystick_state["buttons"]["Start"]
        a_pressed = joystick_state["buttons"]["A"]
        b_pressed = joystick_state["buttons"]["B"]
        
        # Start键：开始任务
        if start_pressed and not self._last_start_pressed:
            if self._env.task_status == TaskStatus.NOT_STARTED:
                print(f"\n✓✓✓ 任务开始！\n")
                self._env.set_task_status(TaskStatus.BEGIN)
            elif self._env.task_status == TaskStatus.BEGIN:
                print(f"⚠️  任务进行中，请按 A 键完成或按 B 键重置")
        
        # A键：完成任务
        if a_pressed and not self._last_a_pressed:
            if self._env.task_status == TaskStatus.BEGIN:
                print(f"\n✅ 任务完成！\n")
            self._env.set_task_status(TaskStatus.SUCCESS)
        
        # B键：重置任务
        if b_pressed and not self._last_b_pressed:
            if self._env.task_status == TaskStatus.BEGIN:
                print(f"\n🔄 重置任务...\n")
                self._env.set_task_status(TaskStatus.RETRY)
            elif self._env.task_status == TaskStatus.NOT_STARTED:
                print(f"⚠️  任务尚未开始，请先按 Start 键")
        
        # 更新按钮状态
        self._last_start_pressed = start_pressed
        self._last_a_pressed = a_pressed
        self._last_b_pressed = b_pressed
        
    def on_playback_action(self, action) -> np.ndarray:
        """回放动作（用于策略推理）"""
        self._grasp_value = action[6 + len(self._arm_joint_names)]
        self.set_gripper_ctrl(self._grasp_value)
        
        if self._env.action_type == ActionType.END_EFFECTOR_OSC:
            action_ee = self._action_B_to_action(action[:6])
            self._controller.set_goal(action_ee)
            ctrl = self._controller.run_controller()
            self._set_arm_ctrl(self._arm_actuator_id, ctrl)
            action = self.fill_arm_ctrl(action)
            
        elif self._env.action_type == ActionType.END_EFFECTOR_IK:
            action_ee = self._action_B_to_action(action[:6])
            quat = transform_utils.axisangle2quat(action_ee[3:6])
            action_ee_xquat = np.array([quat[3], quat[0], quat[1], quat[2]])
            ctrl = self.set_arm_position_ctrl(action_ee[:3], action_ee_xquat)
            self._set_arm_ctrl(self._arm_actuator_id, ctrl)
            action = self.fill_arm_ctrl(action)
            
        elif self._env.action_type in [ActionType.JOINT_POS, ActionType.JOINT_MOTOR]:
            arm_joint_action = action[6:6+len(self._arm_joint_names)]
            self._set_arm_ctrl(self._arm_actuator_id, arm_joint_action)
            
        else:
            raise ValueError("Invalid action type")
            
        return action
        
    def fill_arm_ctrl(self, action: np.ndarray) -> np.ndarray:
        """填充手臂控制信号"""
        ctrl = self._env.ctrl[self._arm_actuator_id]
        action[6:6+len(self._arm_actuator_id)] = ctrl
        return action
        
    def _set_arm_ctrl(self, arm_actuator_id, ctrl) -> None:
        """设置手臂控制"""
        for i in range(len(arm_actuator_id)):
            self._env.ctrl[arm_actuator_id[i]] = ctrl[i]
            
    def set_arm_position_ctrl(self, ee_xpos, ee_xquat):
        """IK 位置控制"""
        self._inverse_kinematics_controller.set_goal(ee_xpos, ee_xquat)
        delta = self._inverse_kinematics_controller.compute_inverse_kinematics()
        
        for i in range(len(self._arm_actuator_id)):
            self._env.ctrl[self._arm_actuator_id[i]] += delta[self._jnt_dof[i]]
            
        self._env.ctrl = np.clip(self._env.ctrl, self._all_ctrlrange[:, 0], self._all_ctrlrange[:, 1])
        return self._env.ctrl[self._arm_actuator_id]
        
    def _local_to_global(self, local_pos: np.ndarray, local_quat: np.ndarray) -> tuple:
        """局部坐标转全局坐标"""
        base_link_pos, _, base_link_quat = self._env.get_body_xpos_xmat_xquat(self._base_body_name)
        global_pos = base_link_pos + rotations.quat_rot_vec(base_link_quat, local_pos)
        global_quat = rotations.quat_mul(base_link_quat, local_quat)
        return global_pos, global_quat
        
    def _global_to_local(self, global_pos: np.ndarray, global_quat: np.ndarray) -> tuple:
        """全局坐标转局部坐标"""
        base_link_pos, _, base_link_quat = self._env.get_body_xpos_xmat_xquat(self._base_body_name)
        base_link_quat_inv = rotations.quat_conjugate(base_link_quat)
        local_pos = rotations.quat_rot_vec(base_link_quat_inv, global_pos - base_link_pos)
        local_quat = rotations.quat_mul(base_link_quat_inv, global_quat)
        return local_pos, local_quat
        
    def _action_B_to_action(self, action_B: np.ndarray) -> np.ndarray:
        """基座坐标系动作转世界坐标系"""
        ee_pos = action_B[:3]
        ee_axisangle = action_B[3:6]
        
        base_link_pos, _, base_link_quat = self._env.get_body_xpos_xmat_xquat(self._base_body_name)
        base_link_rot = R.from_quat(base_link_quat[[1, 2, 3, 0]])
        ee_pos_global = base_link_pos + base_link_rot.apply(ee_pos)
        
        ee_quat = transform_utils.axisangle2quat(ee_axisangle)
        ee_rot = R.from_quat(ee_quat)
        ee_rot_global = base_link_rot * ee_rot
        ee_axisangle_global = transform_utils.quat2axisangle(ee_rot_global.as_quat())
        
        return np.concatenate([ee_pos_global, ee_axisangle_global], dtype=np.float32).flatten()
        
    def set_gripper_ctrl(self, offset_rate) -> None:
        """设置夹爪控制"""
        raise NotImplementedError("Subclass should implement gripper control")
        
    def update_force_feedback(self) -> None:
        """更新力反馈"""
        pass
        
    def on_close(self):
        """关闭"""
        if hasattr(self, "_pico_joystick") and self._pico_joystick is not None:
            self._pico_joystick.close()
        if hasattr(self, "_leader_bus") and self._leader_bus is not None:
            try:
                self._leader_bus.disconnect()
                print("✅ 主臂串口已断开")
            except Exception as e:
                print(f"[警告] 主臂断开时出错: {e}")