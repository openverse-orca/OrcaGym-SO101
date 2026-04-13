# 数据采集使用说明

脚本：`examples/so101/so101_leader_sim_record.py`

用物理 SO101 主臂遥操作仿真从臂，同步采集双路相机图像，保存为 [LeRobot](https://github.com/huggingface/lerobot) 格式数据集，可直接用于 pi0.5 / ACT / Diffusion Policy 等策略训练。

```
物理主臂 (USB 串口)
      │ 编码器读数
      ▼
OrcaGym 仿真从臂  ──→  关节状态 / 动作  ──→  LeRobot 数据集 (.parquet)
      │
      ▼
相机 WebSocket 流 (7070 / 7090)  ──→  数据集视频帧 (.mp4)
```

---

## 前置条件

### 硬件

| 设备 | 要求 |
|---|---|
| SO101 主臂 | 通过 USB-TTL 连接到电脑 |
| 电脑 | 运行 OrcaStudio 及采集脚本 |

### OrcaStudio

- 已打开并**正在运行** SO101 场景（要点击"运行/Play"，不是仅打开）
- gRPC 服务监听在 `localhost:50051`（默认，无需修改）
- 场景中已配置双路相机，WebSocket 端口分别为 `7070`（全局相机）和 `7090`（腕部相机）

### 场景文件

将 SO101 MuJoCo 场景文件放入 `assets/so101/`，参考 [assets/so101/README.md](../../assets/so101/README.md)。

### 主臂串口

确认串口号：

```bash
ls /dev/ttyACM*
# 通常为 /dev/ttyACM0 或 /dev/ttyACM1
```

如果权限不足：

```bash
sudo chmod 666 /dev/ttyACM0
# 或永久加入 dialout 组（重新登录生效）：
sudo usermod -aG dialout $USER
```

---

## 快速开始

**终端 1（可选）**：启动相机预览，确认画面正常：

```bash
conda activate so101
cd OrcaGym-SO101
python examples/so101/so101_camera_monitor.py
```

**终端 2**：运行采集脚本：

```bash
conda activate so101
cd OrcaGym-SO101

python examples/so101/so101_leader_sim_record.py \
    --repo_id your_name/so101_sim_dataset \
    --task "pick up the blue block" \
    --root ~/datasets/so101_sim \
    --num_episodes 50 \
    --leader_port /dev/ttyACM0
```

> `--repo_id` 和 `--task` 为必填参数。

---

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--repo_id` | **必填** | 数据集名称，格式 `用户名/数据集名` |
| `--task` | **必填** | 任务语言描述，写入每帧元数据 |
| `--root` | `~/.cache/huggingface/lerobot` | 数据集本地保存根目录 |
| `--num_episodes` | `50` | 要采集的 episode 总数 |
| `--fps` | `30.0` | 采集帧率（Hz） |
| `--episode_time_s` | `60.0` | 每个 episode 最长录制时长（秒） |
| `--reset_time_s` | `2.0` | 每个 episode 之间场景重置等待时间（秒） |
| `--leader_port` | `/dev/ttyACM0` | 主臂 USB 串口路径 |
| `--orcagym_addr` | `localhost:50051` | OrcaGym gRPC 服务地址 |
| `--xml_path` | `assets/so101/so101_new_calib.xml` | 场景 XML 路径（不需要改） |
| `--push_to_hub` | 关闭 | 采集完成后自动上传到 Hugging Face Hub |

---

## 采集流程

### 启动阶段（自动，无需操作）

```
[1] 连接 OrcaGym 仿真环境 + 主臂串口
[2] 从臂对齐主臂初始姿态
[3] 启动相机 WebSocket 流（7070 / 7090）
[4] 等待双路相机首帧就绪（最长 30 秒）
[5] 创建 LeRobot 数据集目录
[6] 等待你按 Enter 开始采集
```

### 录制阶段（每个 episode）

```
按 Enter → 开始录制（最长 episode_time_s 秒）
      │
      ├─ Page Down → 提前结束本集，保存数据
      ├─ Page Up   → 丢弃本集，重新录制
      └─ 时间到    → 自动保存，进入重置阶段
                │
                ▼
          重置场景（reset_time_s 秒）
                │
                ▼
          按 Enter 开始下一集
```

### 键盘控制

> 键盘监听为全局系统级（`pynput`），无需点击任何窗口。

| 按键 | 时机 | 效果 |
|---|---|---|
| **Enter** | 等待阶段 | 开始录制 / 开始下一集 |
| **Shift** | 等待阶段 | 输入新任务描述后按 Enter 开始 |
| **Page Down** | 录制中 | 提前结束当前 episode，**保存** |
| **Page Up** | 录制中 | 丢弃当前 episode，**重新录制** |
| **ESC** | 任意时刻 | 停止全部采集，保存已录数据，正常退出 |
| **Ctrl+C** | 任意时刻 | 强制退出（已录数据**不会**自动保存，慎用） |

---

## 数据集结构

```
<root>/your_name/so101_sim_dataset/
├── meta/
│   ├── info.json              # 数据集元信息（fps、特征 schema 等）
│   └── episodes.jsonl         # 每个 episode 的时长、任务描述等
├── data/
│   └── chunk-000/
│       └── episode_*.parquet  # 机械臂关节状态 + 动作数据
└── videos/
    └── chunk-000/
        ├── observation.images.camera_global/
        │   └── episode_*.mp4  # 全局相机视频
        └── observation.images.camera_wrist/
            └── episode_*.mp4  # 腕部相机视频
```

### 数据字段

| 字段 | 维度 | 说明 |
|---|---|---|
| `action` | (6,) | 主臂目标关节角（肩旋转、肩抬升、肘弯曲、腕弯曲、腕旋转、夹爪） |
| `observation.state` | (6,) | 仿真从臂当前关节角 |
| `observation.images.camera_global` | (480, 640, 3) | 全局相机 RGB |
| `observation.images.camera_wrist` | (480, 640, 3) | 腕部相机 RGB |

---

## 常见问题

**串口找不到 `/dev/ttyACM0`**

```bash
ls /dev/ttyACM*                    # 查看实际串口号
python ... --leader_port /dev/ttyACM1
```

**OrcaGym 连接失败（gRPC error）**

- 确认 OrcaStudio 已启动并点击"运行"
- 确认端口未被占用：`lsof -i :50051`
- 若端口不同：`--orcagym_addr localhost:<实际端口>`

**相机一直等待，超时无图像**

1. 确认 OrcaStudio 场景**正在运行**
2. 确认场景中已添加相机组件，WebSocket 端口为 `7070` / `7090`
3. 终端输出有 `✓ begin_save_video 已调用` 字样才会触发相机推流

**数据集目录已存在，脚本报错**

每次运行会新建数据集，不支持追加。换一个 `--root` 路径，或删除旧目录：

```bash
rm -rf ~/datasets/so101_sim
```

**进程卡死无法退出**

```bash
pgrep -f so101_leader_sim_record | xargs kill -9
```

---

## 完整命令示例

```bash
# 采集 20 集，每集最长 30 秒，数据存到 ~/datasets/
python examples/so101/so101_leader_sim_record.py \
    --repo_id dht/so101_pick_cube \
    --task "pick up the blue block and place it on the red pad" \
    --root ~/datasets/so101_pick_cube \
    --num_episodes 20 \
    --episode_time_s 30 \
    --reset_time_s 5 \
    --leader_port /dev/ttyACM1
```
