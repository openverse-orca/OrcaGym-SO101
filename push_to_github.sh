#!/usr/bin/env bash
# =============================================================
# 推送到 GitHub 脚本
# 用法：
#   export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"   # 你的 Personal Access Token
#   bash push_to_github.sh
#
# Token 创建方法：
#   GitHub → Settings → Developer settings
#   → Personal access tokens → Tokens (classic)
#   → Generate new token → 勾选 repo 权限 → 生成
# =============================================================
set -e

GITHUB_USER="LazyGoatGoat"
REPO_NAME="OrcaGym-SO101"
REPO_DESC="SO101 robot simulation: teleoperation, data collection and pi0.5 inference based on OrcaGym"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ 请先设置 GITHUB_TOKEN 环境变量："
    echo "   export GITHUB_TOKEN=\"ghp_xxxxxxxxxxxx\""
    exit 1
fi

echo ""
echo "▶ [1/3] 在 GitHub 创建仓库 ${GITHUB_USER}/${REPO_NAME} ..."
CREATE_RESULT=$(curl -s -X POST \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    https://api.github.com/user/repos \
    -d "{
        \"name\": \"${REPO_NAME}\",
        \"description\": \"${REPO_DESC}\",
        \"private\": false,
        \"auto_init\": false
    }")

REPO_URL=$(echo "$CREATE_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('clone_url','ERROR: ' + d.get('message','unknown')))")

if [[ "$REPO_URL" == ERROR* ]]; then
    echo "⚠️  仓库可能已存在，尝试直接推送..."
    REPO_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
else
    echo "✓ 仓库创建成功：https://github.com/${GITHUB_USER}/${REPO_NAME}"
fi

echo ""
echo "▶ [2/3] 设置远程地址并推送..."
cd "$(dirname "${BASH_SOURCE[0]}")"

# 将 token 嵌入 URL 进行认证推送（之后可以删除）
AUTH_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"

git remote remove origin 2>/dev/null || true
git remote add origin "$AUTH_URL"

git push -u origin main

echo ""
echo "▶ [3/3] 恢复安全的远程地址（移除 token）..."
git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓  推送完成！                                               ║"
echo "║                                                              ║"
echo "║  仓库地址：https://github.com/${GITHUB_USER}/${REPO_NAME}   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
