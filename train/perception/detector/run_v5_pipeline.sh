#!/usr/bin/env bash
# Detector v5 pipeline (composer arc, 2026-08-20): regenerate item labels
# (slot3-only), capture room4 fresh drops, build paste augmentation, train v5,
# check pre-registered bars, and -- only if ALL bars pass -- re-fire the
# composer kill-shot with DET=detector_v5.pt. One stage failing stops the run
# (thor-job notify carries the exit).
set -u
DK="docker run --rm --runtime nvidia --user 1000:1000 -v /home/timothy/projects/VGA:/vga"
DET_W="-w /vga/train/perception/detector"
RL_W="-w /vga/train/rl"
IMG=thor-rl:cu130

echo "=== [1/5] gen_items (slot3-only relabel) ==="
$DK $DET_W $IMG python3 -u gen_items.py || exit 1

echo "=== [2/5] gen_drops (room4 fresh drops) ==="
$DK $DET_W $IMG python3 -u gen_drops.py || exit 1

echo "=== [3/5] gen_paste (room-context augmentation) ==="
$DK $DET_W $IMG python3 -u gen_paste.py || exit 1

echo "=== [4/5] train_detector_v5 + bars ==="
$DK $DET_W $IMG python3 -u train_detector_v5.py | tee /tmp/v5_train.log || exit 1

if ! grep -q "^ALL_PASS" /tmp/v5_train.log; then
    echo "=== BARS FAILED -- stopping before re-fire (see _detector_v5_result.json) ==="
    exit 2
fi

echo "=== [5/5] composer re-fire with detector_v5 (chain + bank) ==="
$DK $RL_W -e MODE=chain -e ARMS=composed \
    -e DET=/vga/train/perception/detector/detector_v5.pt $IMG \
    python3 -u composer_v0.py || exit 1
$DK $RL_W -e MODE=bank -e ARMS=composed \
    -e DET=/vga/train/perception/detector/detector_v5.pt $IMG \
    python3 -u composer_v0.py || exit 1
echo "=== PIPELINE COMPLETE ==="
