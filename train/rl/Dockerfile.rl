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

# Media/vision deps for the 3-tier drive + eval loop. Previously ABSENT from this image
# (only PIL was available), which forced the PIL->JPEG->host-ffmpeg workaround in
# drive_agent.py / run_drive.sh. Bake them in so the container can do frame ops and write
# mp4 directly:
#   - ffmpeg (apt): CLI + libav* shared libs (also back cv2/imageio video I/O)
#   - opencv-python-headless: cv2 without GUI/X libs (correct for a headless container)
#   - imageio + imageio-ffmpeg: numpy-array <-> video I/O with an ffmpeg backend
# These pull only numpy/pillow (already in the base) -- no torch/torchvision churn, same
# resolver caution as the SB3 line above.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --break-system-packages \
        opencv-python-headless imageio imageio-ffmpeg
