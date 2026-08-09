# train/rl — Task 09 fast brain (Track A: dense-reward RL sandbox)

Plan: docs/revitalization/09_fast_brain_rl.md. Phase A0 = plumbing + reward
oracle, NO GPU.

Layout:
- bench_solo.py      — solo ALttP single-core throughput probe (runs in thor-torch)
- bench_link4.py     — 4P link session throughput probe (runs in thor-torch)
- oracle.py          — RAM reward oracle (rupees/hearts/deaths/kills) [A0]
- replay_oracle.py   — replay flight tapes through the oracle for validation [A0]

Run pattern (container; bindings baked into thor-torch:cu130):
  docker run --rm --runtime nvidia -v ~/projects/VGA:/vga -w /vga/train/rl \
      thor-torch:cu130 python3 bench_solo.py

The link engine (link/link_engine.py) is imported via sys.path — the /vga
mount keeps repo paths identical inside and outside the container.
