#!/usr/bin/env bash
# =============================================================
# SO101 依赖初始化脚本
# 用途：clone lerobot / openpi 并安装 Python 依赖
# 用法：bash setup_deps.sh
# =============================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          OrcaGym-SO101  依赖初始化                   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. 初始化 robomimic submodule ───────────────────────────────────────
echo "▶ [1/4] 初始化 robomimic submodule..."
git submodule update --init --recursive 3rd_party/robomimic
echo "✓ robomimic 完成"
echo ""

# ── 2. clone lerobot ─────────────────────────────────────────────────────
LEROBOT_COMMIT="882c80d446a63a44868c67ae535467af32ce0e80"
if [ ! -d "lerobot/.git" ]; then
    echo "▶ [2/4] 克隆 lerobot（仅源码，depth=1 后切换到指定 commit）..."
    git clone --filter=blob:none \
        https://github.com/huggingface/lerobot.git lerobot
    cd lerobot
    git fetch --depth=1 origin "$LEROBOT_COMMIT"
    git checkout "$LEROBOT_COMMIT"
    cd "$REPO_ROOT"
    echo "✓ lerobot 完成（commit: ${LEROBOT_COMMIT:0:7}）"
else
    echo "✓ lerobot 已存在，跳过"
fi
echo ""

# ── 3. clone openpi ──────────────────────────────────────────────────────
OPENPI_COMMIT="981483dca0fd9acba698fea00aa6e52d56a66c58"
if [ ! -d "openpi/.git" ]; then
    echo "▶ [3/4] 克隆 openpi（仅源码）..."
    git clone --filter=blob:none \
        https://github.com/Physical-Intelligence/openpi.git openpi
    cd openpi
    git fetch --depth=1 origin "$OPENPI_COMMIT"
    git checkout "$OPENPI_COMMIT"
    cd "$REPO_ROOT"
    echo "✓ openpi 完成（commit: ${OPENPI_COMMIT:0:7}）"
else
    echo "✓ openpi 已存在，跳过"
fi
echo ""

# ── 4. 安装 Python 依赖 ───────────────────────────────────────────────────
echo "▶ [4/4] 安装 Python 依赖..."
echo "  安装 orca-gym 本包..."
pip install -e . -q

echo "  安装 SO101 运行依赖..."
pip install -r requirements_so101.txt -q

echo "  安装 openpi-client..."
pip install -e openpi/packages/openpi-client -q

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✓  所有依赖安装完成！                               ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  下一步：                                            ║"
echo "║  1. 将场景文件放入 assets/so101/                     ║"
echo "║     （参考 assets/so101/README.md）                  ║"
echo "║  2. 启动 OrcaStudio，加载 SO101 场景并运行           ║"
echo "║  3. 运行示例脚本（参考 examples/so101/README.md）    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
