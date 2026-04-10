# pi0.5 模型权重

将训练好的 pi0.5 模型权重放置在本目录下。

## 目录结构

```
models/
└── pi05_h7_lora/
    └── h7_lora/
        └── <checkpoint_step>/     ← 例如 4000
            ├── params/
            │   └── ...
            └── model_config.json
```

## 启动推理服务器

```bash
# 在 openpi uv 环境中执行
cd <OrcaGym根目录>/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_h7_lora \
    --policy.dir=../models/pi05_h7_lora/h7_lora/<checkpoint_step>
```

## 获取模型权重

- 通过训练流程自行训练（参考 `examples/so101/README.md` 中的训练说明）
- 或联系项目维护者获取预训练权重
