# SO101 仿真遥操作 · 数据采集 · 模型推理

本目录包含 SO101 单臂机器人在 OrcaGym 仿真环境中的完整操作链路：

| 脚本 | 功能 |
|------|------|
| `so101_leader_teleoperation.py` | 物理主臂 → 仿真从臂实时遥操作 |
| `so101_leader_sim_record.py`   | 遥操作同时采集 LeRobot 格式数据集 |
| `so101_sim_inference_client.py`| 加载 pi0.5 模型在仿真中执行推理 |
| `so101_camera_monitor.py`      | 相机图像实时预览（辅助工具） |

---

## 环境准备

### 1. 克隆仓库（含子模块）

```bash
git clone --recurse-submodules https://github.com/LazyGoatGoat/OrcaGym-SO101.git
cd OrcaGym-SO101
```

### 2. 创建 conda 环境

```bash
conda create -n so101 python=3.10
conda activate so101
```

### 3. 安装依赖

```bash
# 安装 orca-gym 本包
pip install -e .

# 安装 SO101 运行依赖
pip install -r requirements_so101.txt

# 安装 openpi-client（推理流程）
pip install -e openpi/packages/openpi-client
```

### 4. 放置场景文件

将 SO101 MuJoCo 场景文件放入 `assets/so101/`，详见 [assets/so101/README.md](../../assets/so101/README.md)。

### 5. 启动 OrcaStudio

打开 OrcaStudio，加载 SO101 场景并点击"运行"（确保 gRPC 服务在 `localhost:50051`）。

---

## 流程一：遥操作

```bash
conda activate so101
python examples/so101/so101_leader_teleoperation.py \
    --port /dev/ttyACM0
```

**参数说明**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `/dev/ttyACM1` | 主臂串口路径 |
| `--xml_path` | `assets/so101/so101_new_calib.xml` | 场景 XML 路径 |
| `--orcagym_addr` | `localhost:50051` | OrcaGym gRPC 地址 |
| `--max_steps` | `20000` | 最大步数 |

---

## 流程二：数据采集

```bash
conda activate so101
python examples/so101/so101_leader_sim_record.py \
    --repo_id your_name/so101_sim_dataset \
    --task "pick up the blue block" \
    --num_episodes 50 \
    --episode_time_s 60
```

**键盘控制**

| 按键 | 功能 |
|------|------|
| `Page Down` | 提前结束当前 episode 并保存 |
| `Page Up` | 丢弃当前 episode，重新录制 |
| `Esc` | 停止全部采集 |

**可选：实时相机预览**

```bash
# 另开终端
python examples/so101/so101_camera_monitor.py
```

---

## 流程三：模型推理

### 步骤 1：启动策略服务器（openpi uv 环境）

```bash
cd openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_h7_lora \
    --policy.dir=../models/pi05_h7_lora/h7_lora/<checkpoint_step>
```

模型权重放置说明见 [models/pi05_h7_lora/README.md](../../models/pi05_h7_lora/README.md)。

### 步骤 2：运行推理客户端（conda so101 环境）

```bash
conda activate so101
python examples/so101/so101_sim_inference_client.py \
    --task "Pick up the blue block"
```

**键盘控制**

| 按键 | 功能 |
|------|------|
| `Page Down` | 提前结束当前 chunk，立即重新推理 |
| `Esc` / `Ctrl+C` | 停止推理 |

---

## 常见问题

**Q: 找不到串口 `/dev/ttyACM0`**
```bash
ls /dev/ttyACM*   # 确认串口设备
sudo chmod a+rw /dev/ttyACM0  # 授权（或将用户加入 dialout 组）
```

**Q: 连接 OrcaGym 失败**
- 确认 OrcaStudio 已启动并点击"运行"
- 确认 gRPC 端口 50051 未被占用：`lsof -i :50051`

**Q: 相机无图像**
- 确认 OrcaStudio 中相机 WebSocket 服务已启动
- 相机端口：`camera_global=7070`，`camera_wrist=7090`
