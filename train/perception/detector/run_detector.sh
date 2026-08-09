#!/usr/bin/env bash
# per-room gen (enemy/item split) + keydrop item gen (both retry the mgba segfault), then train 3-class.
cd /vga/train/perception/detector
for rm in 0 1 2 3 4; do
  for try in 1 2 3 4; do
    [ -f data/room$rm.npz ] && break
    echo "=== gen room$rm try$try ==="
    python3 -u gen_data.py --room $rm && break || echo "room$rm try$try crashed, retry"
  done
done
for try in 1 2 3 4; do
  [ -f data/item_held.npz ] && break
  echo "=== gen items try$try ==="
  python3 -u gen_items.py && break || echo "gen_items try$try crashed, retry"
done
echo "=== present ==="; ls data/*.npz | xargs -n1 basename
python3 -u train_detector.py
