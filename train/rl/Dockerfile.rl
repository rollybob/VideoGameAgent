# RL training image for VGA Task 09 Phase A1 (PPO baseline on Four Swords).
#
# Built FROM the existing thor-torch:cu130 image (CUDA 13 + torch 2.12.1+cu130 on
# Blackwell/aarch64, mgba bindings already on sys.path at /opt/mgba-build/python/...)
# so env-stepping (mgba) and PPO training (torch/GPU) live in the same process with
# no client/server split. We do NOT reinstall torch/torchvision here - see below.
#
# Build:  docker build -t thor-rl:cu130 -f train/rl/Dockerfile.rl train/rl
# Run:    docker run --rm --runtime nvidia -v <repo>:/work -w /work thor-rl:cu130 python3 train/rl/train_ppo.py
FROM thor-torch:cu130

# stable-baselines3 pulls in its own gymnasium/cloudpickle/pandas/matplotlib deps.
# --no-deps on the SB3 line itself + explicit lightweight deps avoids letting pip's
# resolver touch torch/torchvision (same risk noted in serve/Dockerfile.vlm) --
# SB3's torch version pin can otherwise trigger an upgrade/downgrade that breaks the
# CUDA-13 aarch64 build. --break-system-packages: PEP 668 externally-managed python.
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        gymnasium cloudpickle pandas matplotlib tensorboard \
    && python3 -m pip install --no-cache-dir --break-system-packages --no-deps \
        stable-baselines3
