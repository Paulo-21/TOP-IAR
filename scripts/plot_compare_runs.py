import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator

ENVIRONMENTS = ["HalfCheetah-v5", "Hopper-v5"]
PREFERRED_TAGS = [
    "Reward/Test",
    "eval/episode_reward",
    "eval/average_return",
    "eval/mean_episode_return",
    "eval/mean_return",
    "eval/return_mean",
    "eval/ep_rew_mean",
    "episode_reward",
    "rollout/ep_rew_mean",
    "return",
    "reward",
    "mean_return",
]

# Improve default plot readability for small thumbnails and force no axis margins
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 200,
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'legend.fontsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 3.2,
    'lines.markersize': 9,
    'grid.linewidth': 0.7,
    'axes.xmargin': 0.0,
    'axes.ymargin': 0.0,
})


def find_runs(root_dir):
    """Find runs under the new layout: <root_dir>/<algorithm>/<env>[/tensorboard]

    Returns a list of dicts: {'path': path_to_event_dir, 'algorithm': algo, 'env': env}
    """
    runs = []
    if not os.path.isdir(root_dir):
        return runs
    for algo in os.listdir(root_dir):
        algo_dir = os.path.join(root_dir, algo)
        if not os.path.isdir(algo_dir):
            continue
        # Normal layout: algo/<env>[/tensorboard]
        found_env_subdir = False
        for env in os.listdir(algo_dir):
            env_dir = os.path.join(algo_dir, env)
            if not os.path.isdir(env_dir):
                continue
            found_env_subdir = True
            # prefer a tensorboard subdir if present
            tb_dir = os.path.join(env_dir, 'tensorboard')
            candidate = None
            if os.path.isdir(tb_dir):
                # check for event files inside
                if glob.glob(os.path.join(tb_dir, 'events.out.tfevents.*')):
                    candidate = tb_dir
            # fallback: check env_dir itself for event files
            if candidate is None:
                if glob.glob(os.path.join(env_dir, 'events.out.tfevents.*')):
                    candidate = env_dir
            if candidate:
                # normalize algorithm display name for readability in plots
                algorithm_display = algo
                # map only the exact learnable-quantiles folder to avoid merging other 'top_tqc' variants
                if 'top_tqc_learnable_quantiles' in algo:
                    algorithm_display = 'tqc_top_quantiles'
                runs.append({'path': candidate, 'algorithm': algorithm_display, 'env': env})

        # Legacy / alternative layout: event files directly under algo_dir
        # e.g. runs/<algorithm>/events.out.tfevents.*  -> try to infer env and include
        if not found_env_subdir:
            evt_glob = glob.glob(os.path.join(algo_dir, 'events.out.tfevents.*'))
            if evt_glob:
                # try to infer env from algorithm folder name
                inferred_env = None
                for e in ENVIRONMENTS:
                    if e in algo:
                        inferred_env = e
                        break
                # also try parent folder and first event filename as fallback
                if inferred_env is None:
                    for e in ENVIRONMENTS:
                        if e in os.path.basename(root_dir):
                            inferred_env = e
                            break
                if inferred_env:
                    algorithm_display = algo
                    if 'top_tqc_learnable_quantiles' in algo:
                        algorithm_display = 'tqc_top_quantiles'
                    runs.append({'path': algo_dir, 'algorithm': algorithm_display, 'env': inferred_env})
                else:
                    # unable to infer environment name reliably — skip but notify
                    print(f"Skipping {algo_dir}: no env subfolder and could not infer environment name from '{algo}'.")
    return runs


def choose_tag(ea: event_accumulator.EventAccumulator):
    tags = ea.Tags().get("scalars", [])
    for t in PREFERRED_TAGS:
        if t in tags:
            return t
    # fallback to first scalar tag
    return tags[0] if tags else None


def read_event_scalars(evt_file, tag=None):
    ea = event_accumulator.EventAccumulator(evt_file)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag is None:
        tag = choose_tag(ea)
    else:
        if tag not in tags:
            # requested tag not present in this file
            return None, None

    if tag is None:
        return None, None

    scalars = ea.Scalars(tag)
    steps = np.array([s.step for s in scalars], dtype=float)
    vals = np.array([s.value for s in scalars], dtype=float)
    return steps, vals


def parse_run_name(dirname: str):
    # Deprecated: run discovery now provides algorithm and env explicitly.
    # Keep for backward-compatibility but return best-effort values.
    env = None
    for e in ENVIRONMENTS:
        if e in dirname:
            env = e
            break
    # algorithm name: strip env and seed info
    alg = os.path.basename(dirname)
    if env and env in alg:
        alg = alg.replace(env, "")
    alg = re.sub(r"_?seed\d+", "", alg)
    alg = alg.strip("_-")
    if not alg:
        alg = os.path.basename(dirname)
    return alg, env


def gather_data(root_dirs, tag_override=None):
    # root_dirs may be a single path string or an iterable of paths
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    runs = []
    for rd in root_dirs:
        runs.extend(find_runs(rd))
    data = defaultdict(lambda: defaultdict(list))  # data[env][alg] = list of (steps, vals)

    for run_info in runs:
        run = run_info['path']
        alg = run_info.get('algorithm')
        env = run_info.get('env')
        if env not in ENVIRONMENTS:
            continue

        # look recursively under the run path for any event files (handles per-seed subfolders)
        evt_files = glob.glob(os.path.join(run, "**", "events.out.tfevents.*"), recursive=True)
        if not evt_files:
            print(f"No event files in {run}, skipping")
            continue

        # inspect first event file to select tag for the run
        first_file = sorted(evt_files)[0]
        ea_first = event_accumulator.EventAccumulator(first_file)
        ea_first.Reload()
        available_tags = ea_first.Tags().get("scalars", [])
        chosen_tag = None
        if tag_override:
            if tag_override in available_tags:
                chosen_tag = tag_override
            else:
                print(f"Requested tag '{tag_override}' not in first event file for run {run}. Available tags: {available_tags}")
                # proceed to try auto selection
        if chosen_tag is None:
            chosen_tag = choose_tag(ea_first)

        # Treat each event file as a separate run (this allows averaging/std
        # when multiple seeds produced separate event files). We still pick
        # the run-level chosen_tag from the first file but read each file
        # independently so they become distinct entries in data[env][alg].
        any_found = False
        for f in sorted(evt_files):
            steps, vals = read_event_scalars(f, tag=chosen_tag)
            if steps is None:
                continue
            any_found = True
            data[env][alg].append((steps, vals))

        if not any_found:
            print(f"No scalar data found in {run} for tag {chosen_tag}")
            continue
        # diagnostics (report files & chosen tag)
        print(f"Run: {run}")
        print(f"  files: {len(evt_files)}, tag: {chosen_tag}")

    return data


def gather_data_for_tag(root_dirs, tag: str):
    """Collect data for a specific scalar tag across runs.

    Returns data[env][alg] = list of (steps, vals)
    """
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    runs = []
    for rd in root_dirs:
        runs.extend(find_runs(rd))
    data = defaultdict(lambda: defaultdict(list))

    for run_info in runs:
        run = run_info['path']
        alg = run_info.get('algorithm')
        env = run_info.get('env')
        if env not in ENVIRONMENTS:
            continue

        # search recursively for event files (per-seed subfolders supported)
        evt_files = glob.glob(os.path.join(run, "**", "events.out.tfevents.*"), recursive=True)
        if not evt_files:
            continue

        any_found = False
        for f in sorted(evt_files):
            steps, vals = read_event_scalars(f, tag=tag)
            if steps is None:
                continue
            any_found = True
            data[env][alg].append((steps, vals))

        if any_found:
            print(f"Run: {run} (tag: {tag}, files: {len(evt_files)})")

    return data


def find_all_tags_with_prefix(root_dirs, prefix: str):
    tags = set()
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]
    for rd in root_dirs:
        for run_info in find_runs(rd):
            run = run_info['path']
            evt_files = glob.glob(os.path.join(run, "events.out.tfevents.*"))
            if not evt_files:
                continue
            first = sorted(evt_files)[0]
            ea = event_accumulator.EventAccumulator(first)
            ea.Reload()
            for t in ea.Tags().get('scalars', []):
                if t.startswith(prefix):
                    tags.add(t)
    return sorted(tags)


def plot_env(data_for_env, env_name, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    # fixed x grid across all plots: 0 .. 1e6
    xs = np.linspace(0.0, 1e6, 300)
    for alg, runs in sorted(data_for_env.items()):
        interp_vals = []
        for steps, vals in runs:
            # extend values to full [0,1e6] via constant extrapolation at ends
            if len(steps) == 0:
                continue
            if len(steps) == 1:
                interp_vals.append(np.full_like(xs, vals[0]))
            else:
                interp_vals.append(np.interp(xs, steps, vals, left=vals[0], right=vals[-1]))
        if not interp_vals:
            continue
        arr = np.vstack(interp_vals)
        ys_mean = arr.mean(axis=0)
        ys_std = arr.std(axis=0)
        p10 = np.percentile(arr, 10, axis=0)
        p90 = np.percentile(arr, 90, axis=0)

        line, = plt.plot(xs, ys_mean, label=f"{alg} (n={len(runs)})", linewidth=plt.rcParams['lines.linewidth'])
        color = line.get_color()
        # band: 10th-90th percentiles
        plt.fill_between(xs, p10, p90, color=color, alpha=0.15, linewidth=0)

    plt.title(f"Comparison on {env_name}")
    plt.xlabel("Step")
    plt.ylabel("Return")
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True)
    plt.xlim(0, 1e6)
    out_file = os.path.join(out_dir, f"{env_name}.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=plt.rcParams['savefig.dpi'], bbox_inches='tight')
    plt.close()
    print(f"Wrote plot: {out_file}")


def sanitize_tag(tag: str) -> str:
    return tag.replace('/', '_')


def plot_env_for_tag(data_for_env, env_name, tag, out_dir="plots"):
    """Plot a specific tag (e.g. Distributions/beta or Distributions/arm0)."""
    if not data_for_env:
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    # fixed x grid 0..1e6 for diagnostics as well
    xs = np.linspace(0.0, 1e6, 300)
    for alg, runs in sorted(data_for_env.items()):
        interp_vals = []
        for steps, vals in runs:
            if len(steps) == 0:
                continue
            if len(steps) == 1:
                interp_vals.append(np.full_like(xs, vals[0]))
            else:
                interp_vals.append(np.interp(xs, steps, vals, left=vals[0], right=vals[-1]))
        if not interp_vals:
            continue
        arr = np.vstack(interp_vals)
        ys_mean = arr.mean(axis=0)
        ys_std = arr.std(axis=0)
        p10 = np.percentile(arr, 10, axis=0)
        p90 = np.percentile(arr, 90, axis=0)

        line, = plt.plot(xs, ys_mean, label=f"{alg} (n={len(runs)})", linewidth=plt.rcParams['lines.linewidth'])
        color = line.get_color()
        plt.fill_between(xs, p10, p90, color=color, alpha=0.15, linewidth=0)

    plt.title(f"{tag} on {env_name}")
    plt.xlabel("Step")
    plt.ylabel(tag)
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True)
    plt.xlim(0, 1e6)
    safe = sanitize_tag(tag)
    out_file = os.path.join(out_dir, f"{env_name}_{safe}.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=plt.rcParams['savefig.dpi'], bbox_inches='tight')
    plt.close()
    print(f"Wrote plot: {out_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dirs", nargs="*",
                        default=["runs"],
                        help="Space-separated list of run directories to scan (default: runs)")
    parser.add_argument("--tag", default=None, help="Force a specific scalar tag (overrides auto-detection)")
    parser.add_argument("--out_dir", default="plots", help="Output plots directory")
    args = parser.parse_args()

    # Plot primary metric (auto-detected or forced via --tag)
    data = gather_data(args.runs_dirs, tag_override=args.tag)
    if not data:
        print("No data found. Make sure --runs_dir points to directories containing event files.")
    else:
        for env in ENVIRONMENTS:
            if env in data:
                plot_env(data[env], env, out_dir=args.out_dir)
            else:
                print(f"No runs for {env} in {args.runs_dirs}")

    # Plot Distributions/beta if present
    beta_data = gather_data_for_tag(args.runs_dirs, 'Distributions/beta')
    for env in ENVIRONMENTS:
        if env in beta_data and beta_data[env]:
            plot_env_for_tag(beta_data[env], env, 'Distributions/beta', out_dir=args.out_dir)

    # Plot any Distributions/arm* tags discovered
    arm_tags = find_all_tags_with_prefix(args.runs_dirs, 'Distributions/arm')
    if arm_tags:
        for tag in arm_tags:
            arm_data = gather_data_for_tag(args.runs_dirs, tag)
            for env in ENVIRONMENTS:
                if env in arm_data and arm_data[env]:
                    plot_env_for_tag(arm_data[env], env, tag, out_dir=args.out_dir)
    
    # Plot drop count metrics for learnable quantile dropping
    drop_count_tags = ['train/drop_count_total', 'train/drop_count_per_critic', 
                       'agent/drop_count_total', 'agent/drop_count_per_critic']
    for tag in drop_count_tags:
        drop_data = gather_data_for_tag(args.runs_dirs, tag)
        for env in ENVIRONMENTS:
            if env in drop_data and drop_data[env]:
                plot_env_for_tag(drop_data[env], env, tag, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
