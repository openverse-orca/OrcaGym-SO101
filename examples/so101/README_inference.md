# 模型推理使用说明

脚本：`examples/so101/so101_sim_inference_client.py`

在 OrcaGym 仿真环境中加载 pi0.5 策略模型，自主执行抓取任务。推理服务器（openpi）和仿真客户端分别在两个终端运行，通过 WebSocket 通信。

```
终端 1（openpi uv 环境）            终端 2（conda so101 环境）
serve_policy.py（策略服务器）  ←WebSocket:8000→  so101_sim_inference_client.py
       ↑                                                  ↓
  pi0.5 模型权重                              OrcaGym 仿真 + 相机 + 动作执行
```

---

## 前置条件

### OrcaStudio

- SO101 场景已打开并**正在运行**
- gRPC 服务监听在 `localhost:50051`
- 双路相机 WebSocket 端口：`7070`（全局）/ `7090`（腕部）

### 场景文件

将 SO101 MuJoCo 场景文件放入 `assets/so101/`，参考 [assets/so101/README.md](../../assets/so101/README.md)。

### 模型权重

将训练好的模型权重放入 `models/` 目录，参考 [models/pi05_h7_lora/README.md](../../models/pi05_h7_lora/README.md)。

目录结构示例：

```
models/
└── pi05_h7_lora/
    └── h7_lora/
        └── 4000/          ← checkpoint 目录（包含 params.msgpack 等）
```

### openpi 推理环境

策略服务器需要单独的 openpi 环境（**uv 管理，与 conda so101 环境完全独立**）。

如果还没有安装 openpi 环境，执行：

```bash
# 克隆 openpi 上游仓库
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi

# 安装 uv（如果没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装 openpi 依赖（自动创建虚拟环境）
GIT_LFS_SKIP_SMUDGE=1 uv sync

# 应用 SO101 训练配置（patch 包含 pi05_h7_lora 等配置）
git apply /path/to/OrcaGym-SO101/openpi_patches/so101_openpi.patch
```

> openpi 环境安装时间较长（需要下载 JAX / CUDA 等重型依赖），请耐心等待。

---

## 运行步骤

### 步骤 1：启动策略服务器（openpi uv 环境）

```bash
cd openpi    # 进入 openpi 目录

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_h7_lora \
    --policy.dir=../OrcaGym-SO101/models/pi05_h7_lora/h7_lora/4000
```

> 将 `4000` 替换为你的实际 checkpoint 步数（目录名）。  
> 服务器启动成功后会输出 `Serving on port 8000`。

关闭服务器：

```bash
pkill -f 'scripts/serve_policy.py'
```

### 步骤 2：运行推理客户端（conda so101 环境）

**新开一个终端**：

```bash
conda activate so101
cd OrcaGym-SO101

python examples/so101/so101_sim_inference_client.py \
    --task "Pick up the blue block"
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--task` | `"Pick up the blue block"` | 任务语言描述（发给策略模型） |
| `--host` | `localhost` | 策略服务器地址 |
| `--port` | `8000` | 策略服务器 WebSocket 端口 |
| `--orcagym_addr` | `localhost:50051` | OrcaGym gRPC 地址 |
| `--fps` | `30` | 控制频率（Hz） |
| `--max_steps` | `0`（无限） | 最大推理步数，0 表示不限制 |
| `--xml_path` | `assets/so101/so101_new_calib.xml` | 场景 XML 路径（通常不需要改） |

---

## 推理流程

```
[1] 连接 OrcaGym 仿真环境
[2] 启动双路相机 WebSocket 流
[3] 连接策略服务器（localhost:8000）
[4] 循环执行：
      采图（全局 + 腕部）→ 读关节状态
          → 发给策略服务器
          → 接收 action chunk（16 步动作序列）
          → 逐步执行，直到 chunk 结束
          → 重复
```

### 键盘控制

| 按键 | 效果 |
|---|---|
| **Page Down** | 丢弃当前 chunk，立即重新采图推理 |
| **ESC** | 停止推理，正常退出 |
| **Ctrl+C** | 强制退出 |

---

## 常见问题

**策略服务器连接失败（WebSocket error）**

- 确认步骤 1 的服务器已启动，终端有 `Serving on port 8000` 输出
- 确认端口未被占用：`lsof -i :8000`
- 若使用非默认端口：`--port <实际端口>`

**模型权重路径找不到**

- 确认 `models/pi05_h7_lora/h7_lora/<步数>/` 目录存在
- 目录内应包含 `params.msgpack` 等文件
- `--policy.dir` 要指向 checkpoint 的**完整路径**

**OrcaGym 连接失败**

- 确认 OrcaStudio 已启动并点击"运行"
- 确认 gRPC 端口 50051 未被占用：`lsof -i :50051`

**相机无图像 / 推理动作异常**

- 确认 OrcaStudio 场景正在运行，相机 WebSocket 端口为 `7070` / `7090`
- 确认模型与当前数据集的归一化参数一致（脚本内 `_DS_STATE_MIN` / `_DS_STATE_MAX`）

**推理时机械臂不动或乱动**

- 检查任务描述 `--task` 是否与训练时一致
- 检查 checkpoint 步数（训练早期 checkpoint 效果差）
- 检查归一化范围是否与训练数据匹配
