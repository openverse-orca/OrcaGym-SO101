# openpi 自定义修改说明

本目录包含对上游 [openpi](https://github.com/Physical-Intelligence/openpi) 仓库的所有自定义修改。
修改基于 commit `981483dca0fd9acba698fea00aa6e52d56a66c58`。

## 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `src/openpi/training/config.py` | 新增 SO101 训练配置（`pi05_h7_lora` ~ `pi05_h12_lora`）及 `LeRobotH7DataConfig` 数据工厂类 |
| `pyproject.toml` | 依赖版本调整 |
| `.python-version` | Python 版本锁定 |

## 应用方式

### 方式一：打 patch（推荐）

```bash
cd openpi
git apply ../openpi_patches/so101_openpi.patch
```

### 方式二：直接覆盖文件

```bash
cp openpi_patches/src/openpi/training/config.py openpi/src/openpi/training/config.py
cp openpi_patches/pyproject.toml openpi/pyproject.toml
cp openpi_patches/.python-version openpi/.python-version
```

## 核心新增内容：LeRobotH7DataConfig

`config.py` 中新增了适配 SO101 数据集的数据工厂类，以及 6 套训练配置：

| 配置名 | 数据集 repo_id | 说明 |
|--------|--------------|------|
| `pi05_h7_lora` | `h7` | SO101 第7轮数据集 |
| `pi05_h8_lora` | `h8` | SO101 第8轮数据集 |
| `pi05_h9_lora` | `h9` | SO101 第9轮数据集 |
| `pi05_h10_lora` | `h10` | SO101 第10轮数据集 |
| `pi05_h11_lora` | `h11` | SO101 第11轮数据集 |
| `pi05_h12_lora` | `h12` | SO101 第12轮数据集（最新）|

## 启动推理服务器（patch 应用后）

```bash
cd openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_h12_lora \
    --policy.dir=../models/pi05_h12_lora/h12_lora/19999
```
