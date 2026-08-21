"""
Unit tests for ``coverage_report.py`` covering the public helper functions:

* :func:`median`
* :func:`load_runs`
* :func:`summarize`
* :func:`render_markdown`

All tests use only the Python standard library ``unittest``.
"""

import unittest
import os
import json
import tempfile

# Import the module under test – it lives in the same directory as this file.
from coverage_report import median, load_runs, summarize, render_markdown


class TestMedian(unittest.TestCase):
    def test_empty_returns_zero_point_zero(self) -> None:
        self.assertEqual(median([]), 0.0)

    def test_odd_length_list(self) -> None:
        data = [5, 1, 3]
        # Sorted => [1, 3, 5]; middle element is 3.
        self.assertEqual(median(data), 3)

    def test_even_length_list_returns_average(self) -> None:
        data = [4, 2, 7, 1]  # Sorted => [1, 2, 4, 7]; average of 2 and 4 is 3.0.
        self.assertEqual(median(data), (2 + 4) / 2.0)

    def test_unsorted_and_floats(self) -> None:
        data = [10.5, 2.5, 8.0]
        # Sorted => [2.5, 8.0, 10.5]; middle is 8.0.
        self.assertEqual(median(data), 8.0)


class TestLoadRuns(unittest.TestCase):
    def test_missing_directory_returns_empty(self) -> None:
        # A clearly non‑existent absolute path should yield an empty list.
        self.assertEqual(load_runs("/tmp/does_not_exist_12345"), [])

    def test_load_and_sort_valid_runs(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            # Helper to write a coverage.json file inside a subdirectory.
            def write_run(name: str, payload: dict) -> None:
                sub = os.path.join(root_dir, name)
                os.makedirs(sub, exist_ok=True)
                with open(os.path.join(sub, "coverage.json"), "w") as f:
                    json.dump(payload, f)

            # Three valid runs – two share the same ``arm`` value.
            run_a = {
                "arm": 2,
                "seed": 10,
                "n_rooms": 15,
                "deaths": 3,
                "dmg_taken": 120,
                "t_s": 45,
            }
            run_b = {
                "arm": 1,
                "seed": 5,
                "n_rooms": 12,
                "deaths": 0,
                "dmg_taken": 80,
                "t_s": 30,
            }
            run_c = {
                "arm": 1,
                "seed": 8,
                "n_rooms": 14,
                "deaths": 2,
                "dmg_taken": 90,
                "t_s": 35,
            }

            write_run("runA", run_a)
            write_run("runB", run_b)
            write_run("runC", run_c)

            # Invalid run – missing required field ``arm``.
            write_run(
                "badMissing",
                {
                    "seed": 1,
                    "n_rooms": 5,
                    "deaths": 0,
                    "dmg_taken": 10,
                    "t_s": 20,
                },
            )

            # Invalid run – required field present but ``None``.
            write_run(
                "badNull",
                {
                    "arm": None,
                    "seed": 3,
                    "n_rooms": 7,
                    "deaths": 0,
                    "dmg_taken": 5,
                    "t_s": 10,
                },
            )

            # Subdirectory without a coverage.json file – should be ignored.
            os.makedirs(os.path.join(root_dir, "emptyDir"), exist_ok=True)

            runs = load_runs(root_dir)
            # Expect exactly the three valid runs sorted by (arm, seed).
            expected = [run_b, run_c, run_a]
            self.assertEqual(runs, expected)

    def test_malformed_json_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            # Bad JSON file.
            bad_dir = os.path.join(root_dir, "badjson")
            os.makedirs(bad_dir, exist_ok=True)
            with open(os.path.join(bad_dir, "coverage.json"), "w") as f:
                f.write("{ not a valid json }")

            # A well‑formed JSON file for control.
            good_dir = os.path.join(root_dir, "good")
            os.makedirs(good_dir, exist_ok=True)
            good_payload = {
                "arm": 1,
                "seed": 2,
                "n_rooms": 5,
                "deaths": 0,
                "dmg_taken": 0,
                "t_s": 10,
            }
            with open(os.path.join(good_dir, "coverage.json"), "w") as f:
                json.dump(good_payload, f)

            runs = load_runs(root_dir)
            self.assertEqual(runs, [good_payload])


class TestSummarize(unittest.TestCase):
    def test_empty_input(self) -> None:
        result = summarize([])
        expected = {"n_runs": 0, "arms": [], "by_arm": {}}
        self.assertEqual(result, expected)

    def test_multiple_arms_and_statistics(self) -> None:
        runs = [
            {"arm": 1, "seed": 5, "n_rooms": 10, "deaths": 2, "dmg_taken": 100, "t_s": 30},
            {"arm": 1, "seed": 8, "n_rooms": 12, "deaths": 0, "dmg_taken": 80, "t_s": 35},
            {"arm": 2, "seed": 3, "n_rooms": 9, "deaths": 1, "dmg_taken": 50, "t_s": 25},
        ]
        result = summarize(runs)

        # Top‑level aggregates.
        self.assertEqual(result["n_runs"], 3)
        self.assertEqual(result["arms"], [1, 2])

        # Per‑arm statistics for arm 1.
        arm1 = result["by_arm"][1]
        self.assertEqual(arm1["n"], 2)
        self.assertEqual(arm1["seeds"], [5, 8])
        self.assertEqual(arm1["median_rooms"], 11.0)  # (10 + 12) / 2
        self.assertEqual(arm1["mean_rooms"], round((10 + 12) / 2, 3))
        self.assertEqual(arm1["max_rooms"], 12)
        self.assertEqual(arm1["total_deaths"], 2)
        self.assertEqual(arm1["total_dmg"], 180)

        # Per‑arm statistics for arm 2 (single entry).
        arm2 = result["by_arm"][2]
        self.assertEqual(arm2["n"], 1)
        self.assertEqual(arm2["seeds"], [3])
        self.assertEqual(arm2["median_rooms"], 9)  # odd length → middle value
        self.assertEqual(arm2["mean_rooms"], round(9 / 1, 3))
        self.assertEqual(arm2["max_rooms"], 9)
        self.assertEqual(arm2["total_deaths"], 1)
        self.assertEqual(arm2["total_dmg"], 50)


class TestRenderMarkdown(unittest.TestCase):
    def test_empty_input_produces_empty_string(self) -> None:
        self.assertEqual(render_markdown([]), "")

    def test_render_and_summary_content(self) -> None:
        runs = [
            {"arm": 1, "seed": 5, "n_rooms": 10, "deaths": 2, "dmg_taken": 100, "t_s": 30},
            {"arm": 1, "seed": 8, "n_rooms": 12, "deaths": 0, "dmg_taken": 80, "t_s": 35},
        ]
        md = render_markdown(runs)
        lines = md.splitlines()

        # Header rows.
        self.assertEqual(lines[0], "| arm | seed | n_rooms | deaths | dmg_taken | t_s |")
        self.assertEqual(lines[1], "|-----|------|---------|--------|-----------|-----|")

        # Data rows – order must match the input list.
        self.assertEqual(lines[2], "| 1 | 5 | 10 | 2 | 100 | 30 |")
        self.assertEqual(lines[3], "| 1 | 8 | 12 | 0 | 80 | 35 |")

        # A blank line separates the table from the summary.
        self.assertEqual(lines[4], "")
        self.assertTrue(lines[5].startswith("## Summary"))

        # Bullet points – there should be exactly one for arm 1.
        bullet_lines = [ln for ln in lines if ln.startswith("- ")]
        self.assertEqual(len(bullet_lines), 1)
        self.assertIn("- 1: median rooms = 11.00 (n=2)", bullet_lines[0])


if __name__ == "__main__":
    unittest.main()
