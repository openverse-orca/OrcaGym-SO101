"""
SO101 单臂机器人环境
完整实现从底层控制到 LeRobot 集成
"""

import numpy as np
import importlib
from typing import Dict
from envs.manipulation.dual_arm_env import RunMode, ControlDevice, ActionType, TaskStatus
from orca_gym.adapters.robomimic.robomimic_env import RobomimicEnv
from orca_gym.adapters.robomimic.task.pick_place_task import PickPlaceTask
from orca_gym.devices.pico_joytsick import PicoJoystick
from orca_gym.utils.reward_printer import RewardPrinter
from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()


# SO101 机器人注册表
so101_robot_entries = {
    "so101": "envs.so101.so101_robot:SO101Robot",
    "ActorManipulator": "envs.so101.so101_robot:SO101Robot",  # 兼容旧场景
    "so101_new_calib_usda": "envs.so101.so101_robot:SO101Robot",  # OrcaStudio 场景中的实体名称
}


def get_so101_robot_entry(name: str):
    """获取 SO101 机器人入口"""
    for robot_name, entry in so101_robot_entries.items():
        if name.startswith(robot_name):
            return entry
    raise ValueError(f"Robot entry for {name} not found in so101_robot_entries.")


class SO101Env(RobomimicEnv):
    """
    SO101 单臂机器人环境
    支持遥操作、策略推理、LeRobot 集成
    """
    ENV_VERSION = "1.0.0"
    
    def __init__(
        self,
        frame_skip: int,
        reward_type: str,
        orcagym_addr: str,
        agent_names: list,
        pico_ports: list,
        time_step: float,
        run_mode: RunMode,
        action_type: ActionType,
        ctrl_device: ControlDevice,
        control_freq: int,
        sample_range: float,
        task_config_dict: dict,
        action_step: int = 1,
        camera_config: dict = None,
        **kwargs,
    ):
        self._run_mode = run_mode
        self._action_type = action_type
        self._sync_render = True
        self._ctrl_device = ctrl_device
        self._control_freq = control_freq
        self._sample_range = sample_range
        self._reward_type = reward_type
        
        # 初始化控制设备
        if self._ctrl_device == ControlDevice.VR and run_mode == RunMode.TELEOPERATION:
            if len(pico_ports) == 0:
                raise ValueError("VR 控制模式需要提供至少一个 pico_ports。")
            self._joystick = {}
            pico_joystick_list = []
            for port in pico_ports:
                pico_joystick_list.append(PicoJoystick(int(port)))
            for i, agent_name in enumerate(agent_names):
                # 当 agent 数量大于 pico 数量时，使用最后一个 pico
                self._joystick[agent_name] = pico_joystick_list[min(i, len(pico_joystick_list) - 1)]
        elif self._ctrl_device == ControlDevice.XBOX and run_mode == RunMode.TELEOPERATION:
            from orca_gym.devices.xbox_joystick import XboxJoystickManager
            self._xbox_joystick_manager = XboxJoystickManager()
            joystick_names = self._xbox_joystick_manager.get_joystick_names()
            if len(joystick_names) == 0:
                raise ValueError("No Xbox joystick detected.")
            self._xbox_joystick = self._xbox_joystick_manager.get_joystick(joystick_names[0])
            if self._xbox_joystick is None:
                raise ValueError("Xbox joystick not found.")
        
        self._reward_printer = RewardPrinter()
        self._config = task_config_dict
        self._config['grpc_addr'] = orcagym_addr
        self._task = PickPlaceTask(self._config)
        self._task.register_init_env_callback(self.init_env)
        kwargs["task"] = self._task
        self._teleop_counter = 0
        self._got_task = False
        
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            action_step=action_step,
            camera_config=camera_config if camera_config is not None else {},
            **kwargs,
        )
        
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        
        self.gym.opt.iterations = 150
        self.gym.opt.noslip_tolerance = 50
        self.gym.opt.ccd_iterations = 100
        self.gym.opt.sdf_iterations = 50
        self.gym.set_opt_config()
        
        self.ctrl = np.zeros(self.nu)
        self.mj_forward()
        
        # 创建机器人代理
        self._agents: Dict[str, any] = {}
        for id, agent_name in enumerate(self._agent_names):
            self._agents[agent_name] = self.create_agent(id, agent_name)
        
        assert len(self._agents) > 0, "At least one agent should be created."
        self._set_init_state()
        
        self._set_obs_space()
        self._set_action_space()
        
    def init_env(self):
        """初始化环境"""
        self.model, self.data = self.initialize_simulation()
        self._init_ctrl()
        self.init_agents()
        
    def create_agent(self, id, name):
        """创建机器人代理"""
        entry = get_so101_robot_entry(name)
        module_name, class_name = entry.rsplit(":", 1)
        module = importlib.import_module(module_name)
        class_type = getattr(module, class_name)
        agent = class_type(self, id, name)
        return agent
        
    def _init_ctrl(self):
        """初始化控制"""
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.gym.opt.iterations = 150
        self.gym.opt.noslip_tolerance = 50
        self.gym.opt.ccd_iterations = 100
        self.gym.opt.sdf_iterations = 50
        self.gym.set_opt_config()
        self.ctrl = np.zeros(self.nu)
        self.mj_forward()
        
    def _set_obs_space(self):
        """设置观测空间"""
        self.observation_space = self.generate_observation_space(self._get_obs().copy())
        
    def _set_action_space(self):
        """设置动作空间"""
        env_action_range = np.concatenate([agent.action_range for agent in self._agents.values()], axis=0)
        self.env_action_range_min = env_action_range[:, 0]
        self.env_action_range_max = env_action_range[:, 1]
        scaled_action_range = np.ones(env_action_range.shape, dtype=np.float32)
        self.action_space = self.generate_action_space(scaled_action_range)
        
    def get_env_version(self):
        """获取环境版本"""
        return SO101Env.ENV_VERSION
        
    @property
    def run_mode(self) -> RunMode:
        return self._run_mode
        
    @property
    def action_type(self) -> ActionType:
        return self._action_type
        
    @property
    def ctrl_device(self) -> ControlDevice:
        return self._ctrl_device
        
    @property
    def control_freq(self) -> int:
        return self._control_freq
        
    @property
    def task_status(self) -> TaskStatus:
        return self._task_status
        
    @property
    def joystick(self):
        if self._ctrl_device == ControlDevice.VR:
            return self._joystick
        return None

    @property
    def xbox_joystick(self):
        if self._ctrl_device == ControlDevice.XBOX:
            return self._xbox_joystick
            
    @property
    def xbox_joystick_manager(self):
        if self._ctrl_device == ControlDevice.XBOX:
            return self._xbox_joystick_manager
            
    def set_task_status(self, status):
        """设置任务状态"""
        if status == TaskStatus.SUCCESS:
            _logger.info("Task success!")
        elif status == TaskStatus.FAILURE:
            _logger.error("Task failure!")
        elif status == TaskStatus.BEGIN:
            print("Start to record task......")
        self._task_status = status
        
    def check_success(self):
        """检查任务是否成功"""
        success = self._is_success()
        return {"task": success}
        
    def _set_init_state(self) -> None:
        """设置初始状态"""
        self._task_status = TaskStatus.NOT_STARTED
        [agent.set_joint_neutral() for agent in self._agents.values()]
        self.ctrl = np.zeros(self.nu)
        for agent in self._agents.values():
            agent.set_init_ctrl()
        self.set_ctrl(self.ctrl)
        self.mj_forward()
        
    def _is_success(self) -> bool:
        """判断是否成功"""
        return self._task_status == TaskStatus.SUCCESS
        
    def _is_truncated(self) -> bool:
        """判断是否截断"""
        return self._task_status == TaskStatus.FAILURE
        
    def step(self, action) -> tuple:
        """环境步进"""
        if self._run_mode == RunMode.TELEOPERATION:
            ctrl, noscaled_action = self._teleoperation_action()
        elif self._run_mode == RunMode.POLICY_NORMALIZED:
            noscaled_action = self.denormalize_action(action, self.env_action_range_min, self.env_action_range_max)
            ctrl, noscaled_action = self._playback_action(noscaled_action)
        else:
            raise ValueError("Invalid run mode: ", self._run_mode)
            
        if self._run_mode == RunMode.TELEOPERATION:
            [agent.update_force_feedback() for agent in self._agents.values()]
            
        scaled_action = self.normalize_action(noscaled_action, self.env_action_range_min, self.env_action_range_max)
        
        self.do_simulation(ctrl, self.frame_skip)

        # 仿真完成后打印夹爪力矩/接触力日志（ctrl 与 q/qv 同帧，公式可正确对比）
        if self._run_mode == RunMode.TELEOPERATION:
            for agent in self._agents.values():
                if hasattr(agent, 'on_post_simulation'):
                    agent.on_post_simulation()

        obs = self._get_obs().copy()
        
        info = {"state": self.get_state(), "action": scaled_action}
        terminated = self._is_success()
        truncated = self._is_truncated()
        reward = self._compute_reward(info)
        
        return obs, reward, terminated, truncated, info
        
    def _teleoperation_action(self):
        """遥操作动作"""
        actions = []
        for agent in self._agents.values():
            action = agent.on_teleoperation_action()
            actions.append(action)
        combined_action = np.concatenate(actions).flatten()
        return self.ctrl, combined_action
        
    def _playback_action(self, action):
        """回放动作"""
        start_idx = 0
        actions = []
        for agent in self._agents.values():
            action_dim = agent.action_range.shape[0]
            agent_action = action[start_idx:start_idx + action_dim]
            processed_action = agent.on_playback_action(agent_action)
            actions.append(processed_action)
            start_idx += action_dim
        combined_action = np.concatenate(actions).flatten()
        return self.ctrl, combined_action
        
    def _get_obs(self) -> dict:
        """获取观测"""
        obs_dict = {}
        for agent in self._agents.values():
            agent_obs = agent.get_obs()
            for key, value in agent_obs.items():
                obs_dict[f"{agent.name}_{key}"] = value
        return obs_dict
        
    def get_state(self) -> dict:
        """获取状态"""
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }
        
    def _compute_reward(self, info) -> float:
        """计算奖励"""
        return 0.0
        
    def reset_model(self) -> tuple:
        """重置模型"""
        self._set_init_state()
        [agent.on_reset_model() for agent in self._agents.values()]
        obs = self._get_obs()
        info = {}
        return obs, info
        
    def close(self):
        """关闭环境"""
        [agent.on_close() for agent in self._agents.values()]
        # 关闭 Pico 连接
        if self._ctrl_device == ControlDevice.VR and hasattr(self, '_joystick'):
            closed = set()
            for pico in self._joystick.values():
                if id(pico) not in closed:
                    pico.close()
                    closed.add(id(pico))
        
    def action_use_motor(self):
        """判断是否使用电机控制"""
        if self._action_type in [ActionType.END_EFFECTOR_OSC, ActionType.JOINT_MOTOR]:
            return True
        elif self._action_type in [ActionType.JOINT_POS, ActionType.END_EFFECTOR_IK]:
            return False
        return False
