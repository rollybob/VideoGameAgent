# vga/util -- verified utility primitives

Small, self-contained agent primitives, each with a matching test. Built overnight
2026-09-04 by subordinate models against hidden adversarial gates (validated both-ways,
re-verified 3x fresh). They are AVAILABLE primitives, not yet wired into the agent --
pull them in where a real need appears.

- nms.py            -- non-max suppression for overlapping detector boxes (nms(boxes, scores, iou_thresh))
- hud_delta.py      -- diff two HUD snapshots -> changed fields + events (damage/death/pickup)
- grid_bfs.py       -- 4-connected shortest-path length on a walled grid (shortest_path_len(grid, start, goal))
- stuck_detector.py -- position-based "agent stopped moving over last N frames" (is_stuck(positions, window, tol))
                       NOTE: complements the PIXEL-based stuck detector in run_reasoner.py (--stuck-thresh);
                       this one is world-position based (catches walking-into-a-wall where pixels still change).

Run a test:  cd vga/util && python3 test_nms.py   (each is standalone, plain asserts)
