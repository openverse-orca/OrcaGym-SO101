# OrcaGym-SO101

SO101 单臂机器人仿真平台，基于 OrcaGym + MuJoCo，支持物理主臂遥操作、数据采集和 pi0.5 模型推理。

```
物理主臂（SO101 Leader）
        │ USB 串口
        ▼
[OrcaGym 仿真环境]  ──  MuJoCo 物理引擎（OrcaStudio gRPC）
        │
        ├── 遥操作    →  so101_leader_teleoperation.py
        ├── 数据采集  →  so101_leader_sim_record.py  →  LeRobot 数据集
        └── 模型推理  →  so101_sim_inference_client.py  ←  pi0.5 策略服务器
```

---

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/haitongding/OrcaGym-SO101.git
cd OrcaGym-SO101

# 2. 创建 conda 环境
conda create -n so101 python=3.10
conda activate so101

# 3. 安装依赖
pip install -e .
pip install -e 3rd_party/robomimic
pip install -e lerobot
pip install -e openpi/packages/openpi-client
pip install -r requirements_so101.txt
```

---

## 使用

### 遥操作

```bash
conda activate so101
python examples/so101/so101_leader_teleoperation.py --port /dev/ttyACM0
```

### 数据采集

```bash
conda activate so101
python examples/so101/so101_leader_sim_record.py \
    --repo_id your_name/dataset_name \
    --task "pick up the blue block" \
    --num_episodes 50
```

→ 详细说明：[examples/so101/README_record.md](examples/so101/README_record.md)

### 模型推理

```bash
# 终端 1：策略服务器（openpi uv 环境）
cd openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_h7_lora \
    --policy.dir=../OrcaGym-SO101/models/pi05_h7_lora/h7_lora/4000

# 终端 2：推理客户端（conda so101 环境）
conda activate so101
cd OrcaGym-SO101
python examples/so101/so101_sim_inference_client.py \
    --task "Pick up the blue block"
```

→ 详细说明：[examples/so101/README_inference.md](examples/so101/README_inference.md)

---

## 文件放置

| 内容 | 放置位置 | 说明 |
|------|----------|------|
| MuJoCo 场景文件 | `assets/so101/` | [查看说明](assets/so101/README.md) |
| pi0.5 模型权重 | `models/pi05_h7_lora/` | [查看说明](models/pi05_h7_lora/README.md) |
| openpi 定制配置 | `openpi_patches/` | [查看说明](openpi_patches/README.md) |

---

## 目录结构

```
OrcaGym-SO101/
├── examples/so101/
│   ├── so101_leader_teleoperation.py   # 遥操作
│   ├── so101_leader_sim_record.py      # 数据采集
│   ├── so101_sim_inference_client.py   # 模型推理
│   ├── so101_camera_monitor.py         # 相机预览
│   ├── README_record.md                # 数据采集说明
│   └── README_inference.md             # 模型推理说明
├── envs/so101/                         # SO101 环境实现
├── orca_gym/                           # OrcaGym 核心框架
├── assets/so101/                       # 放置 MuJoCo XML 场景文件（不进 git）
├── models/pi05_h7_lora/                # 放置 pi0.5 模型权重（不进 git）
├── lerobot/                            # LeRobot 源码
├── openpi/packages/openpi-client/      # openpi 客户端源码
├── openpi_patches/                     # openpi SO101 定制配置
├── 3rd_party/robomimic/                # robomimic 源码
├── pyproject.toml
└── requirements_so101.txt
```

---

## 依赖

| 组件 | 说明 |
|------|------|
| Python 3.10 | conda 环境 |
| MuJoCo 3.3.3 | 物理引擎 |
| OrcaStudio | 仿真平台（联系松应科技获取） |
| openpi 服务器 | 推理时需要，需单独安装 uv 环境 |
