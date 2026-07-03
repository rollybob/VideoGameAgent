#!/usr/bin/env bash
# Build mGBA 0.10.2 official CFFI python bindings (headless core: run_frame +
# video buffer + memory in one process). F21 RAM-oracle foundation.
# Logs to /tmp/mgba_build.log. Idempotent-ish; safe to re-run.
set -euo pipefail
exec > >(tee -a /tmp/mgba_build.log) 2>&1
echo "=== build start $(date -u +%H:%M:%S) ==="

VENV="$HOME/projects/VGA/.venv"
SRC="$HOME/src/mgba"
BUILD="$SRC/build-py"

echo "--- apt deps ---"
sudo apt-get update -qq
sudo apt-get install -y -qq \
  cmake make gcc g++ pkg-config \
  zlib1g-dev libpng-dev libsqlite3-dev libzip-dev \
  libedit-dev python3-dev \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev

echo "--- pip cffi/setuptools into VGA venv ---"
"$VENV/bin/pip" install -q cffi setuptools wheel

echo "--- clone mgba 0.10.2 ---"
mkdir -p "$HOME/src"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch 0.10.2 https://github.com/mgba-emu/mgba.git "$SRC"
fi
cd "$SRC"; git log --oneline -1

echo "--- cmake configure (headless: no QT/SDL/GL/FFMPEG; python ON) ---"
rm -rf "$BUILD"; mkdir -p "$BUILD"; cd "$BUILD"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_QT=OFF -DBUILD_SDL=OFF \
  -DBUILD_GL=OFF -DBUILD_GLES2=OFF -DBUILD_GLES3=OFF \
  -DUSE_FFMPEG=ON -DUSE_DISCORD_RPC=OFF -DUSE_EPOXY=OFF \
  -DUSE_LIBZIP=OFF -DUSE_MINIZIP=OFF -DUSE_LZMA=OFF \
  -DBUILD_PYTHON=ON \
  -DPython3_EXECUTABLE="$VENV/bin/python" \
  -DPYTHON_EXECUTABLE="$VENV/bin/python"

echo "--- build ---"
make -j"$(nproc)"

echo "--- locate built python module ---"
find "$BUILD" -name "_pylib*.so" -o -name "mgba" -type d 2>/dev/null | head
echo "=== build done $(date -u +%H:%M:%S) ==="
