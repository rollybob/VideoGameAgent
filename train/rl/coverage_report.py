import json
import os


def load_runs(directory):
    runs = []
    if not os.path.isdir(directory):   # merge patch (Claude): guard the subordinate's refine dropped
        return runs
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        coverage_path = os.path.join(subdir_path, 'coverage.json')
        if os.path.isdir(subdir_path) and os.path.exists(coverage_path):
            try:
                with open(coverage_path, 'r') as f:
                    data = json.load(f)
                # Skip entries missing required fields or with None values
                if ('arm' in data and data['arm'] is not None and \
                    'seed' in data and data['seed'] is not None and \
                    'n_rooms' in data and data['n_rooms'] is not None):
                    runs.append(data)
            except:
                continue
    return sorted(runs, key=lambda x: (x['arm'], x['seed']))

def median(lst):
    if not lst:
        return 0.0
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    if n % 2 == 1:
        return sorted_lst[n//2]
    else:
        return (sorted_lst[n//2 - 1] + sorted_lst[n//2]) / 2.0

def summarize(runs):
    if not runs:
        return {
            'n_runs': 0,
            'arms': [],
            'by_arm': {}
        }

    runs_by_arm = {}
    for run in runs:
        arm = run['arm']
        if arm not in runs_by_arm:
            runs_by_arm[arm] = []
        runs_by_arm[arm].append(run)

    arms = sorted(runs_by_arm.keys())
    by_arm = {}
    for arm in arms:
        arm_runs = runs_by_arm[arm]
        n = len(arm_runs)
        seeds = sorted([r['seed'] for r in arm_runs])
        n_rooms = [r['n_rooms'] for r in arm_runs]
        deaths = [r['deaths'] for r in arm_runs]
        dmg_taken = [r['dmg_taken'] for r in arm_runs]

        by_arm[arm] = {
            'n': n,
            'seeds': seeds,
            'median_rooms': median(n_rooms),
            'mean_rooms': round(sum(n_rooms) / n, 3),
            'max_rooms': max(n_rooms),
            'total_deaths': sum(deaths),
            'total_dmg': sum(dmg_taken)
        }

    return {
        'n_runs': len(runs),
        'arms': arms,
        'by_arm': by_arm
    }

def render_markdown(runs):
    if not runs:
        return ""

    lines = [
        "| arm | seed | n_rooms | deaths | dmg_taken | t_s |",
        "|-----|------|---------|--------|-----------|-----|"
    ]

    for run in runs:
        lines.append(
            f"| {run['arm']} | {run['seed']} | {run['n_rooms']} | {run['deaths']} | {run['dmg_taken']} | {run['t_s']} |"
        )

    lines.append("\n## Summary")
    summary = summarize(runs)
    for arm in summary['arms']:
        arm_data = summary['by_arm'][arm]
        lines.append(f"- {arm}: median rooms = {arm_data['median_rooms']:.2f} (n={arm_data['n']})")

    return "\n".join(lines)

if __name__ == '__main__':
    import sys
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    runs = load_runs(args.directory)
    if args.json:
        print(json.dumps(summarize(runs), indent=2))
    else:
        print(render_markdown(runs))