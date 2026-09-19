#!/usr/bin/env bash
# 一键搭建开发环境。
#
# 为什么需要脚本而不只是 `pip install -e .`：
# 本环境（macOS + Python 3.13）下，pip 无法解包任何 sdist（.tar.gz）包，报
#     PermissionError: EEXIST: mkdir '<TMPDIR>/pip-install-*/<pkg>_<hash>'
# 已排查确认：普通 mkdir / curl / tar 均正常，纯 wheel 包安装正常，
# 问题精确定位在 pip 解包 sdist 的代码路径。
# jieba 只发布 sdist，因此必须绕行：手动下载 → 手动解压 → 从本地目录安装。
#
# 用法：
#     bash scripts/bootstrap_env.sh
#
# 幂等，可重复执行。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/Users/shihao/.workbuddy/binaries/python/versions/3.13.12/bin/python3}"
VENV="$ROOT/.venv"
TMP="$ROOT/.tmp"

echo "==> 项目根目录: $ROOT"
echo "==> Python: $PYTHON_BIN"

if [ ! -x "$VENV/bin/python" ]; then
  echo "==> 创建虚拟环境"
  "$PYTHON_BIN" -m venv "$VENV"
fi

PIP="$VENV/bin/pip"
mkdir -p "$TMP"
export TMPDIR="$TMP"

echo "==> 升级基础构建工具"
"$PIP" install --quiet --upgrade pip setuptools wheel

echo "==> 安装项目自身与开发依赖"
"$PIP" install --quiet -e ".[dev]"

# --- 绕行安装只发布 sdist 的包 ---------------------------------------------
install_sdist() {
  local pkg="$1" ver="$2"
  if "$VENV/bin/python" -c "import ${pkg}" 2>/dev/null; then
    echo "==> $pkg 已就绪，跳过"
    return 0
  fi

  echo "==> 绕行安装 $pkg==$ver（pip 无法解包 sdist）"
  rm -rf "$TMP/$pkg" "$TMP/$pkg.tar.gz"

  local url
  url=$(curl -sS "https://pypi.org/pypi/${pkg}/${ver}/json" \
    | "$VENV/bin/python" -c "import json,sys; print(json.load(sys.stdin)['urls'][0]['url'])")

  curl -sSL -o "$TMP/$pkg.tar.gz" "$url"
  mkdir -p "$TMP/$pkg"
  tar xzf "$TMP/$pkg.tar.gz" -C "$TMP/$pkg" --strip-components=1

  # --no-build-isolation 需要 venv 内已有 setuptools（上一步已装）
  "$PIP" install --quiet --no-build-isolation "$TMP/$pkg"
}

install_sdist jieba 0.42.1

# --- 解析层（体积大，可选）-------------------------------------------------
if [ "${WITH_MINERU:-0}" = "1" ]; then
  echo "==> 安装 MinerU 解析层（体积较大，请耐心等待）"
  "$PIP" install "mineru>=4.0,<5"

  echo "==> 下载 MinerU 模型（standard 档，约 2GB）"
  "$VENV/bin/mineru-kit" models download --tier standard

  echo "==> 开启本地解析服务"
  "$VENV/bin/mineru" config set parse_server.local.mode managed
else
  echo "==> 跳过 MinerU。需要解析层时执行：WITH_MINERU=1 bash scripts/bootstrap_env.sh"
fi

echo
echo "==> 完成。自检："
"$VENV/bin/python" -m pytest -q
"$VENV/bin/python" -c "import jieba; print('jieba:', '/'.join(jieba.lcut('营业收入同比变化')))"
