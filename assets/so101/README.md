# SO101 仿真场景资产

将 SO101 MuJoCo 场景文件放置在本目录下。

## 需要的文件

从 [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) 仓库获取，
或联系 OrcaStudio 支持团队获取仿真版本。

```
assets/so101/
├── so101_new_calib.xml        ← 主场景文件（必需）
├── meshes/                    ← 网格文件目录（必需）
│   ├── base.stl
│   ├── shoulder_pan.stl
│   └── ...
└── textures/                  ← 纹理文件目录（可选）
```

## 配置路径

默认情况下，脚本会自动查找 `assets/so101/so101_new_calib.xml`。

若放置在其他位置，通过 `--xml_path` 参数指定：

```bash
python examples/so101/so101_leader_teleoperation.py \
    --xml_path /your/custom/path/so101_new_calib.xml
```

或设置环境变量：

```bash
export SO101_XML_PATH=/your/custom/path/so101_new_calib.xml
```
