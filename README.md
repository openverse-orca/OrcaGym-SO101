# OrcaGym-SO101

> **SO101 单臂机器人仿真平台**：基于 OrcaGym + MuJoCo，支持物理主臂遥操作、LeRobot 格式数据采集和 pi0.5 模型推理。

```
物理主臂（SO101 Leader）
        │ USB 串口
        ▼
[OrcaGym 仿真环境]  ──  MuJoCo 物理引擎（OrcaStudio gRPC）
        │
        ├── 遥操作模式    →  so101_leader_teleoperation.py
        ├── 数据采集模式  →  so101_leader_sim_record.py  →  LeRobot 数据集
        └── 推理模式      →  so101_sim_inference_client.py  ←  pi0.5 策略服务器
```

---

## 快速开始

```bash
# 1. 克隆（含子模块）
git clone --recurse-submodules https://github.com/LazyGoatGoat/OrcaGym-SO101.git
cd OrcaGym-SO101

# 2. 安装
conda create -n so101 python=3.10 && conda activate so101
pip install -e .
pip install -r requirements_so101.txt
pip install -e openpi/packages/openpi-client

# 3. 放置场景资产（见 assets/so101/README.md）
# 4. 启动 OrcaStudio，加载 SO101 场景并运行

# 5. 遥操作
python examples/so101/so101_leader_teleoperation.py --port /dev/ttyACM0
```

## 详细文档

- [完整使用说明](examples/so101/README.md)
- [场景资产放置](assets/so101/README.md)
- [模型权重放置](models/pi05_h7_lora/README.md)

## 目录结构

```
OrcaGym-SO101/
├── envs/
│   ├── manipulation/          # 基础类（RunMode、AgentBase 等）
│   └── so101/                 # SO101 环境实现
│       ├── so101_env.py
│       ├── so101_robot.py
│       ├── single_arm_robot_base.py
│       └── configs/
├── orca_gym/                  # OrcaGym 核心框架
├── examples/
│   └── so101/                 # 三大运行脚本 + 使用文档
├── assets/
│   └── so101/                 # 放置 MuJoCo XML 场景文件（不进 git）
├── models/
│   └── pi05_h7_lora/          # 放置 pi0.5 模型权重（不进 git）
├── lerobot/                   # 子模块：HuggingFace LeRobot
├── openpi/                    # 子模块：Physical Intelligence openpi
├── 3rd_party/
│   └── robomimic/             # 子模块：robomimic（OrcaGym 适配版）
├── pyproject.toml
└── requirements_so101.txt
```

## 依赖

| 组件 | 版本 / 来源 |
|------|------------|
| Python | 3.10 |
| MuJoCo | 3.3.3 |
| OrcaStudio | 联系松应科技 |
| lerobot | HuggingFace（子模块，commit `882c80d`）|
| openpi | Physical Intelligence（子模块，commit `981483d`）|
