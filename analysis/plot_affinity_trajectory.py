#!/usr/bin/env python3
"""
Plot binding affinity trajectories (on-target, off-target, selectivity)
across diffusion timesteps with mean line and std shaded region.

Usage:
    # Single result folder
    python analysis/plot_affinity_trajectory.py --sample_path ./results/theselective/id0_96_high

    # Multiple result folders (averaged)
    python analysis/plot_affinity_trajectory.py \
        --sample_path ./results/theselective/id0_96_high ./results/theselective/id0_90_low

    # All LOW TM-score pairs
    python analysis/plot_affinity_trajectory.py --mode low

    # All HIGH TM-score pairs
    python analysis/plot_affinity_trajectory.py --mode high

    # All pairs (HIGH + LOW)
    python analysis/plot_affinity_trajectory.py --mode all
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_trajectories(sample_path):
    """Load exp_on_traj and exp_off_traj from all result_*.pt files in a folder."""
    result_files = sorted(glob(os.path.join(sample_path, 'result_*.pt')))
    if not result_files:
        return None, None

    on_trajs = []
    off_trajs = []

    for rf in result_files:
        data = torch.load(rf, map_location='cpu')

        exp_on_traj = data.get('exp_on_traj', None)
        exp_off_traj = data.get('exp_off_traj', None)

        if exp_on_traj is None or exp_off_traj is None:
            continue

        # Convert to numpy
        if hasattr(exp_on_traj, 'numpy'):
            exp_on_traj = exp_on_traj.numpy()
        if hasattr(exp_off_traj, 'numpy'):
            exp_off_traj = exp_off_traj.numpy()

        on_trajs.append(exp_on_traj)
        off_trajs.append(exp_off_traj)

    if not on_trajs:
        return None, None

    # Stack: [num_samples, num_steps]
    on_trajs = np.stack(on_trajs, axis=0)
    off_trajs = np.stack(off_trajs, axis=0)

    return on_trajs, off_trajs


def collect_paths_by_mode(mode, base_result_path, pairs_file):
    """Collect result folder paths based on mode (low/high/all)."""
    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 5:
                on_id = int(parts[0])
                high_off_id = int(parts[1])
                low_off_id = int(parts[3])
                pairs.append((on_id, high_off_id, low_off_id))

    paths = []
    if mode in ('high', 'all'):
        for on_id, high_off_id, _ in pairs:
            p = os.path.join(base_result_path, f'id{on_id}_{high_off_id}_high')
            if os.path.isdir(p):
                paths.append(p)
    if mode in ('low', 'all'):
        for on_id, _, low_off_id in pairs:
            p = os.path.join(base_result_path, f'id{on_id}_{low_off_id}_low')
            if os.path.isdir(p):
                paths.append(p)

    return paths


def plot_trajectory(on_trajs, off_trajs, title='', output_path=None, alpha=0.25):
    """
    Plot on-target, off-target, and selectivity trajectories.

    Args:
        on_trajs: [num_samples, num_steps]
        off_trajs: [num_samples, num_steps]
    """
    num_steps = on_trajs.shape[1]
    timesteps = np.arange(num_steps)

    # Compute selectivity = on - off (higher = more selective)
    sel_trajs = on_trajs - off_trajs

    # Mean and std
    on_mean, on_std = on_trajs.mean(axis=0), on_trajs.std(axis=0)
    off_mean, off_std = off_trajs.mean(axis=0), off_trajs.std(axis=0)
    sel_mean, sel_std = sel_trajs.mean(axis=0), sel_trajs.std(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # --- On-target ---
    ax = axes[0]
    ax.plot(timesteps, on_mean, color='#2196F3', linewidth=2.5, label='On-target (mean)')
    ax.fill_between(timesteps, on_mean - on_std, on_mean + on_std,
                     color='#2196F3', alpha=alpha)
    ax.set_xlabel('Diffusion Time Steps')
    ax.set_ylabel('Predicted Binding Affinity')
    ax.set_title('On-target')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # --- Off-target ---
    ax = axes[1]
    ax.plot(timesteps, off_mean, color='#F44336', linewidth=2.5, label='Off-target (mean)')
    ax.fill_between(timesteps, off_mean - off_std, off_mean + off_std,
                     color='#F44336', alpha=alpha)
    ax.set_xlabel('Diffusion Time Steps')
    ax.set_ylabel('Predicted Binding Affinity')
    ax.set_title('Off-target')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # --- Selectivity ---
    ax = axes[2]
    ax.plot(timesteps, sel_mean, color='#4CAF50', linewidth=2.5, label='Selectivity (mean)')
    ax.fill_between(timesteps, sel_mean - sel_std, sel_mean + sel_std,
                     color='#4CAF50', alpha=alpha)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Diffusion Time Steps')
    ax.set_ylabel('Selectivity (On - Off)')
    ax.set_title('Selectivity')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.suptitle(title if title else 'Binding Affinity Trajectory', fontsize=20, y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f'Saved: {output_path}')

    plt.close(fig)

    # --- Combined plot (all three on one axis) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(timesteps, on_mean, color='#2196F3', linewidth=2.5, label='On-target')
    ax2.fill_between(timesteps, on_mean - on_std, on_mean + on_std,
                      color='#2196F3', alpha=alpha)

    ax2.plot(timesteps, off_mean, color='#F44336', linewidth=2.5, label='Off-target')
    ax2.fill_between(timesteps, off_mean - off_std, off_mean + off_std,
                      color='#F44336', alpha=alpha)

    ax2.plot(timesteps, sel_mean, color='#4CAF50', linewidth=2.5, label='Selectivity (On - Off)')
    ax2.fill_between(timesteps, sel_mean - sel_std, sel_mean + sel_std,
                      color='#4CAF50', alpha=alpha)

    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Diffusion Time Steps')
    ax2.set_ylabel('Predicted Score')
    ax2.set_title(title if title else 'Binding Affinity Trajectory')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        combined_path = output_path.replace('.png', '_combined.png')
        fig2.savefig(combined_path, bbox_inches='tight')
        print(f'Saved: {combined_path}')

    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description='Plot binding affinity trajectory')
    parser.add_argument('--sample_path', type=str, nargs='+', default=None,
                        help='Path(s) to result folder(s) containing result_*.pt')
    parser.add_argument('--mode', type=str, choices=['low', 'high', 'all'], default=None,
                        help='Aggregate mode: low/high/all TM-score pairs')
    parser.add_argument('--base_result_path', type=str, default='./results/theselective',
                        help='Base result directory (for --mode)')
    parser.add_argument('--pairs_file', type=str, default='./data/tmscore_extreme_pairs.txt',
                        help='TM-score pairs file (for --mode)')
    parser.add_argument('--output_dir', type=str, default='./analysis/figures',
                        help='Output directory for plots')
    parser.add_argument('--alpha', type=float, default=0.25,
                        help='Transparency of shaded std region')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which folders to process
    if args.mode:
        paths = collect_paths_by_mode(args.mode, args.base_result_path, args.pairs_file)
        if not paths:
            print(f'No result folders found for mode={args.mode}')
            return
        print(f'Mode: {args.mode}, found {len(paths)} result folders')

        # Load and aggregate all trajectories
        all_on, all_off = [], []
        for p in paths:
            on, off = load_trajectories(p)
            if on is not None:
                all_on.append(on)
                all_off.append(off)

        if not all_on:
            print('No trajectory data found (exp_on_traj/exp_off_traj missing in result files)')
            return

        # Check consistent step counts
        step_counts = set(a.shape[1] for a in all_on)
        if len(step_counts) > 1:
            min_steps = min(step_counts)
            print(f'Warning: inconsistent step counts {step_counts}, truncating to {min_steps}')
            all_on = [a[:, :min_steps] for a in all_on]
            all_off = [a[:, :min_steps] for a in all_off]

        on_trajs = np.concatenate(all_on, axis=0)
        off_trajs = np.concatenate(all_off, axis=0)

        print(f'Total samples: {on_trajs.shape[0]}, steps: {on_trajs.shape[1]}')

        title = f'TheSelective - {args.mode.upper()} TM-score pairs (n={on_trajs.shape[0]})'
        output_path = os.path.join(args.output_dir, f'affinity_trajectory_{args.mode}.png')
        plot_trajectory(on_trajs, off_trajs, title=title, output_path=output_path, alpha=args.alpha)

    elif args.sample_path:
        if len(args.sample_path) == 1:
            # Single folder
            p = args.sample_path[0]
            on_trajs, off_trajs = load_trajectories(p)
            if on_trajs is None:
                print(f'No trajectory data found in {p}')
                return
            print(f'Loaded {on_trajs.shape[0]} samples, {on_trajs.shape[1]} steps from {p}')

            folder_name = os.path.basename(p)
            title = f'TheSelective - {folder_name} (n={on_trajs.shape[0]})'
            output_path = os.path.join(args.output_dir, f'affinity_trajectory_{folder_name}.png')
            plot_trajectory(on_trajs, off_trajs, title=title, output_path=output_path, alpha=args.alpha)
        else:
            # Multiple folders - aggregate
            all_on, all_off = [], []
            for p in args.sample_path:
                on, off = load_trajectories(p)
                if on is not None:
                    all_on.append(on)
                    all_off.append(off)

            if not all_on:
                print('No trajectory data found')
                return

            step_counts = set(a.shape[1] for a in all_on)
            if len(step_counts) > 1:
                min_steps = min(step_counts)
                all_on = [a[:, :min_steps] for a in all_on]
                all_off = [a[:, :min_steps] for a in all_off]

            on_trajs = np.concatenate(all_on, axis=0)
            off_trajs = np.concatenate(all_off, axis=0)
            print(f'Total samples: {on_trajs.shape[0]}, steps: {on_trajs.shape[1]}')

            title = f'TheSelective - {len(args.sample_path)} folders (n={on_trajs.shape[0]})'
            output_path = os.path.join(args.output_dir, 'affinity_trajectory_multi.png')
            plot_trajectory(on_trajs, off_trajs, title=title, output_path=output_path, alpha=args.alpha)
    else:
        parser.print_help()
        print('\nError: specify --sample_path or --mode')


if __name__ == '__main__':
    main()
