"""
SO101 机器人实现
继承 SingleArmRobotBase，实现 SO101 特定功能
"""

import numpy as np
from envs.so101.single_arm_robot_base import SingleArmRobotBase
from envs.so101.configs.so101_config import so101_config
from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()


class SO101Robot(SingleArmRobotBase):
    """SO101 单臂机器人"""
    
    def __init__(self, env, id: int, name: str) -> None:
        super().__init__(env, id, name)
        self.init_agent(id)
        
    def init_agent(self, id: int):
        """初始化 SO101 机器人"""
        _logger.info("SO101Robot init_agent")
        super().init_agent(id, so101_config)
        
    def set_gripper_ctrl(self, ctrl_value) -> None:
        """
        设置夹爪控制
        ctrl_value: 直接使用模型 ctrlrange [lo, hi] 内的值
                    推理时由反归一化得到，数采时由主臂直接赋值
        """
        self._env.ctrl[self._gripper_actuator_id] = np.clip(
            ctrl_value,
            self._all_ctrlrange[self._gripper_actuator_id][0],
            self._all_ctrlrange[self._gripper_actuator_id][1]
        )
