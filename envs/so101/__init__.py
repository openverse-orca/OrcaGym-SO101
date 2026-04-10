"""
SO101 环境 Gymnasium 注册
"""

from gymnasium.envs.registration import register

# 注册 SO101 环境
register(
    id='SO101PickPlace-v0',
    entry_point='envs.so101.so101_env:SO101Env',
    max_episode_steps=500,
)
