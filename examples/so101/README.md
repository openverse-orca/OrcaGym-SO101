# SO101 仿真示例脚本

| 脚本 | 功能 | 说明文档 |
|------|------|----------|
| `so101_leader_teleoperation.py` | 物理主臂 → 仿真从臂实时遥操作 | 见下方 |
| `so101_leader_sim_record.py` | 遥操作同时采集 LeRobot 格式数据集 | [数据采集说明](README_record.md) |
| `so101_sim_inference_client.py` | pi0.5 模型推理，在仿真中执行任务 | [推理说明](README_inference.md) |
| `so101_camera_monitor.py` | 相机图像实时预览（辅助工具） | — |

---

## 遥操作

不采集数据，单纯用主臂控制仿真从臂：

```bash
conda activate so101
cd OrcaGym-SO101

python examples/so101/so101_leader_teleoperation.py \
    --port /dev/ttyACM0
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `/dev/ttyACM1` | 主臂串口路径 |
| `--xml_path` | `assets/so101/so101_new_calib.xml` | 场景 XML 路径 |
| `--orcagym_addr` | `localhost:50051` | OrcaGym gRPC 地址 |
| `--max_steps` | `20000` | 最大步数 |

---

## 数据采集

详见 **[README_record.md](README_record.md)**。

```bash
python examples/so101/so101_leader_sim_record.py \
    --repo_id your_name/so101_sim_dataset \
    --task "pick up the blue block" \
    --num_episodes 50
```

---

## 模型推理

详见 **[README_inference.md](README_inference.md)**。

```bash
# 终端 1：启动策略服务器（openpi uv 环境）
cd openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_h7_lora \
    --policy.dir=../OrcaGym-SO101/models/pi05_h7_lora/h7_lora/4000

# 终端 2：运行推理客户端（conda so101 环境）
conda activate so101
cd OrcaGym-SO101
python examples/so101/so101_sim_inference_client.py \
    --task "Pick up the blue block"
```
