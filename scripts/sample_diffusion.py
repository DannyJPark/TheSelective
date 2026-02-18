"""
Sampling script for TheSelective model with guidance support.

Loads model from training checkpoint and performs sampling with optional guidance.

Usage:
    # Basic sampling (no guidance)
    python scripts/sample_diffusion.py --ckpt path/to/checkpoint.pt --data_id 0

    # With guidance (dual-head selectivity)
    python scripts/sample_diffusion.py --ckpt path/to/checkpoint.pt --data_id 0 \
        --off_target_id 10 --guide_mode head1_head2_staged \
        --head1_type_grad_weight 100 --head1_pos_grad_weight 25 \
        --head2_type_grad_weight 100 --head2_pos_grad_weight 25
"""

import argparse
import os
import sys
import shutil
import time
import pickle
import gc

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_mean, scatter_sum
from tqdm.auto import tqdm

sys.path.append(os.path.abspath('./'))

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH, ProteinLigandData
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical, index_to_log_onehot
from utils.evaluation import atom_num


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    """Unbatch ligand atom type trajectory."""
    all_step_v = [[] for _ in range(n_data)]
    if not ligand_v_traj:
        return [np.array([]).reshape(0, 0) for _ in range(n_data)]

    for v in ligand_v_traj:
        v_array = v.detach().cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])

    all_step_v = [np.stack(step_v) if step_v else np.array([]).reshape(0, 0) for step_v in all_step_v]
    return all_step_v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default=None, help='Optional sampling config')
    parser.add_argument('--data_id', '-i', type=int, default=0, help='On-target test set data ID')
    parser.add_argument('--off_target_id', type=int, default=None, help='Off-target test set data ID')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--num_steps', type=int, default=None, help='Diffusion steps')
    parser.add_argument('--sample_num_atoms', type=str, default='prior', choices=['prior', 'ref'])
    parser.add_argument('--result_path', type=str, default='./outputs')
    parser.add_argument('--tag', type=str, default='')

    # Guidance options
    parser.add_argument('--guide_mode', type=str, default='head1_head2_staged',
                       choices=['no_guide', 'joint', 'dual_head_guidance', 'head1_head2_sequential', 'head1_head2_staged', 'head1_only'],
                       help='Guidance mode: no_guide, joint, dual_head_guidance (same model), head1_head2_sequential (Head1:interaction, Head2:cross-attn), head1_head2_staged (Head1 only for t>=500, Head1+Head2 for t<500), head1_only (original KGDiff)')
    parser.add_argument('--head1_type_grad_weight', type=float, default=0.,
                       help='Type gradient weight for head1 (on-target)')
    parser.add_argument('--head1_pos_grad_weight', type=float, default=0.,
                       help='Position gradient weight for head1 (on-target)')
    parser.add_argument('--head2_type_grad_weight', type=float, default=0.,
                       help='Type gradient weight for head2 (off-target)')
    parser.add_argument('--head2_pos_grad_weight', type=float, default=0.,
                       help='Position gradient weight for head2 (off-target)')
    parser.add_argument('--w_on', type=float, default=1.0, help='Weight for on-target affinity')
    parser.add_argument('--w_off', type=float, default=1.0, help='Weight for off-target penalty')

    args = parser.parse_args()

    # Create result directory
    # If result_path ends with 'high' or 'low', use it directly (from shell script)
    # Otherwise, create subdirectory based on data_id
    if args.result_path.endswith('_high') or args.result_path.endswith('_low'):
        result_path = args.result_path
    else:
        if args.off_target_id is not None:
            dir_name = f'id{args.data_id}_{args.off_target_id}'
        else:
            dir_name = f'id{args.data_id}'
        if args.tag:
            dir_name += f'_{args.tag}'
        result_path = os.path.join(args.result_path, dir_name)
    os.makedirs(result_path, exist_ok=True)

    logger = misc.get_logger('sampling', log_dir=result_path)
    logger.info(f'Arguments: {args}')

    # Load checkpoint
    logger.info(f'Loading checkpoint: {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=args.device)

    # Get config from checkpoint
    train_config = ckpt['config']
    logger.info(f'Training config: {train_config}')

    misc.seed_all(42)

    # Check dual head mode
    use_dual_head = getattr(train_config.model, 'use_dual_head_sam_pl', False)
    head2_mode = getattr(train_config.model, 'head2_mode', 'protein_query_atom')
    logger.info(f'Dual Head Mode: {use_dual_head}')
    if use_dual_head:
        logger.info(f'Head2 Mode: {head2_mode}')

    # Setup transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = train_config.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    logger.info(f'Protein feature dim: {protein_featurizer.feature_dim}')
    logger.info(f'Ligand feature dim: {ligand_featurizer.feature_dim}')

    # Load dataset
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=train_config.data,
        transform=transform
    )

    if train_config.data.name == 'pl':
        test_set = subsets['test']
    elif train_config.data.name == 'pdbbind':
        test_set = subsets['test']
    else:
        test_set = subsets.get('test', subsets.get('val', []))

    logger.info(f'Test set size: {len(test_set)}')

    if args.data_id >= len(test_set):
        logger.error(f'data_id {args.data_id} out of range (test set has {len(test_set)} samples)')
        return

    # Get on-target data
    on_target_data = test_set[args.data_id]
    logger.info(f'On-target data_id: {args.data_id}')
    if hasattr(on_target_data, 'protein_filename'):
        logger.info(f'On-target protein: {on_target_data.protein_filename}')

    # Get off-target data if specified
    off_target_data = None
    if args.off_target_id is not None:
        if args.off_target_id >= len(test_set):
            logger.error(f'off_target_id {args.off_target_id} out of range')
            return
        off_target_data = test_set[args.off_target_id]
        logger.info(f'Off-target data_id: {args.off_target_id}')
        if hasattr(off_target_data, 'protein_filename'):
            logger.info(f'Off-target protein: {off_target_data.protein_filename}')

    # Build model
    logger.info('Building model...')
    model = ScorePosNet3D(
        train_config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)

    # Load weights
    model.load_state_dict(ckpt['model'])
    model.eval()
    logger.info(f'Model loaded from iteration {ckpt.get("iteration", "unknown")}')

    # Sampling parameters
    num_steps = args.num_steps
    center_pos_mode = getattr(train_config.model, 'center_pos_mode', 'protein')

    logger.info(f'Sampling with num_steps={num_steps}, center_pos_mode={center_pos_mode}')
    logger.info(f'Guide mode: {args.guide_mode}')
    logger.info(f'Head1 weights: type={args.head1_type_grad_weight}, pos={args.head1_pos_grad_weight}')
    logger.info(f'Head2 weights: type={args.head2_type_grad_weight}, pos={args.head2_pos_grad_weight}')

    # Run sampling using model's method
    logger.info(f'Generating {args.num_samples} samples...')

    all_pred_pos, all_pred_v = [], []
    all_pred_exp_on, all_pred_exp_off = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_exp_on_traj, all_pred_exp_off_traj = [], []  # Trajectory for visualization
    time_list = []

    num_batch = int(np.ceil(args.num_samples / args.batch_size))

    for batch_idx in tqdm(range(num_batch), desc='Sampling batches'):
        n_data = args.batch_size if batch_idx < num_batch - 1 else args.num_samples - args.batch_size * (num_batch - 1)

        # Create on-target batch
        batch_on = Batch.from_data_list(
            [on_target_data.clone() for _ in range(n_data)],
            follow_batch=FOLLOW_BATCH
        ).to(args.device)

        t1 = time.time()

        # Memory optimization: Clear cache before processing
        if args.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        with torch.no_grad():
            batch_protein = batch_on.protein_element_batch

            # Determine number of atoms per sample
            if args.sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(batch_on.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
            elif args.sample_num_atoms == 'ref':
                batch_ligand_ref = batch_on.ligand_element_batch
                ligand_num_atoms = [(batch_ligand_ref == k).sum().item() for k in range(n_data)]
            else:
                raise ValueError(f"Unknown sample_num_atoms: {args.sample_num_atoms}")

            batch_ligand = torch.repeat_interleave(
                torch.arange(n_data, device=args.device),
                torch.tensor(ligand_num_atoms, device=args.device)
            )

            # Initialize ligand positions centered on protein
            center_pos_tensor = scatter_mean(batch_on.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos_tensor[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            # Initialize ligand atom types (uniform distribution)
            uniform_logits = torch.zeros(len(batch_ligand), model.num_classes, device=args.device)
            init_ligand_v_prob = log_sample_categorical(uniform_logits)
            init_ligand_v = init_ligand_v_prob.argmax(dim=-1)

        # Call model's sampling method
        r = model.sample_diffusion_with_guidance(
            protein_pos=batch_on.protein_pos,
            protein_v=batch_on.protein_atom_feature.float(),
            batch_protein=batch_protein,
            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            batch_ligand=batch_ligand,
            off_target_data=off_target_data,
            num_steps=num_steps,
            center_pos_mode=center_pos_mode,
            guide_mode=args.guide_mode,
            head1_type_grad_weight=args.head1_type_grad_weight,
            head1_pos_grad_weight=args.head1_pos_grad_weight,
            head2_type_grad_weight=args.head2_type_grad_weight,
            head2_pos_grad_weight=args.head2_pos_grad_weight,
            w_on=args.w_on,
            w_off=args.w_off
        )

        t2 = time.time()
        time_list.append(t2 - t1)

        # Unbatch results
        ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
        ligand_pos = r['pos'].detach().cpu().numpy()
        ligand_v = r['v'].detach().cpu().numpy()
        exp_on = r['exp_on'].detach().cpu() if r['exp_on'] is not None else torch.zeros(n_data)
        exp_off = r['exp_off'].detach().cpu() if r['exp_off'] is not None else torch.zeros(n_data)

        for k in range(n_data):
            all_pred_pos.append(ligand_pos[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_pred_v.append(ligand_v[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_pred_exp_on.append(exp_on[k].item())
            all_pred_exp_off.append(exp_off[k].item())

        # Unbatch trajectories
        pos_traj = r['pos_traj']
        v_traj = r['v_traj']

        all_step_pos = [[] for _ in range(n_data)]
        for p in pos_traj:
            p_array = p.detach().cpu().numpy().astype(np.float64)
            for k in range(n_data):
                all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
        all_step_pos = [np.stack(step_pos) for step_pos in all_step_pos]
        all_pred_pos_traj += all_step_pos

        all_step_v = unbatch_v_traj(v_traj, n_data, ligand_cum_atoms)
        all_pred_v_traj += all_step_v

        # Collect exp_on_traj and exp_off_traj for visualization
        if 'exp_on_traj' in r and len(r['exp_on_traj']) > 0:
            # Each element in exp_on_traj is [batch_size] tensor at that timestep
            exp_on_traj_batch = torch.stack([t.cpu() for t in r['exp_on_traj']], dim=0)  # [num_steps, batch_size]
            all_pred_exp_on_traj.append(exp_on_traj_batch)

        if 'exp_off_traj' in r and len(r['exp_off_traj']) > 0:
            exp_off_traj_batch = torch.stack([t.cpu() for t in r['exp_off_traj']], dim=0)  # [num_steps, batch_size]
            all_pred_exp_off_traj.append(exp_off_traj_batch)

        # Memory cleanup
        del batch_on, r
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    # Save results (compatible with docking script)
    for idx in range(len(all_pred_pos)):
        result_file = os.path.join(result_path, f'result_{idx}.pt')
        torch.save({
            'pos': all_pred_pos[idx],
            'v': all_pred_v[idx],
            'exp_on': all_pred_exp_on[idx],
            'exp_off': all_pred_exp_off[idx],
            'pos_traj': all_pred_pos_traj[idx] if all_pred_pos_traj else None,
            'v_traj': all_pred_v_traj[idx] if all_pred_v_traj else None,
        }, result_file)

    # Save summary
    summary_file = os.path.join(result_path, 'samples_summary.pkl')
    with open(summary_file, 'wb') as f:
        pickle.dump({
            'data_id': args.data_id,
            'off_target_id': args.off_target_id,
            'num_samples': len(all_pred_pos),
            'exp_on': all_pred_exp_on,
            'exp_off': all_pred_exp_off,
            'time': time_list,
            'args': vars(args)
        }, f)

    logger.info(f'Results saved to {result_path}')

    # Print summary
    logger.info('=== Sampling Summary ===')
    logger.info(f'Generated {len(all_pred_pos)} samples')
    logger.info(f'Average time per batch: {np.mean(time_list):.2f}s')

    exp_on_array = np.array(all_pred_exp_on)
    logger.info(f'On-target affinity - mean: {exp_on_array.mean():.4f}, std: {exp_on_array.std():.4f}')

    if args.off_target_id is not None:
        exp_off_array = np.array(all_pred_exp_off)
        logger.info(f'Off-target affinity - mean: {exp_off_array.mean():.4f}, std: {exp_off_array.std():.4f}')
        selectivity = exp_on_array - exp_off_array
        logger.info(f'Selectivity (on-off) - mean: {selectivity.mean():.4f}, std: {selectivity.std():.4f}')

    # ========================================================================
    # Generate trajectory visualizations for guidance modes
    # ========================================================================
    if args.guide_mode in ['dual_head_guidance', 'head1_head2_sequential', 'head1_head2_staged'] and args.off_target_id is not None:
        if all_pred_exp_on_traj and all_pred_exp_off_traj:
            try:
                logger.info('Generating Head1/Head2 affinity trajectory visualizations...')
                from pathlib import Path
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                # Concatenate trajectories from all batches: [num_steps, total_samples]
                exp_on_traj = torch.cat(all_pred_exp_on_traj, dim=1).numpy()  # [num_steps, num_samples]
                exp_off_traj = torch.cat(all_pred_exp_off_traj, dim=1).numpy()

                num_steps = exp_on_traj.shape[0]
                num_samples = exp_on_traj.shape[1]
                timesteps = np.arange(num_steps)

                # Limit visualization to first 10 samples
                max_vis_samples = min(num_samples, 10)
                if num_samples > 10:
                    logger.info(f'Limiting visualization to first 10 samples (out of {num_samples})')

                # Create visualization directory
                vis_dir = Path(result_path) / 'visualizations'
                vis_dir.mkdir(exist_ok=True)

                # Determine head labels based on guide mode
                if args.guide_mode == 'head1_head2_sequential':
                    head1_label = 'Head1 (On-target, WITH interaction)'
                    head2_label = 'Head2 (Off-target, NO interaction, cross-attn)'
                elif args.guide_mode == 'head1_head2_staged':
                    head1_label = 'Head1 (On-target, WITH interaction, ALL steps)'
                    head2_label = 'Head2 (Off-target, NO interaction, t<500 only)'
                else:
                    head1_label = 'Head1 (On-target)'
                    head2_label = 'Head2 (Off-target)'

                # ====================================================================
                # Plot 1: Separate trajectories for Head1 and Head2
                # ====================================================================
                fig, axes = plt.subplots(2, 1, figsize=(14, 10))

                # Head1 (On-target) plot
                ax = axes[0]
                for i in range(max_vis_samples):
                    ax.plot(timesteps, exp_on_traj[:, i], label=f'Sample {i+1}', alpha=0.7, linewidth=2)
                ax.set_xlabel('Iteration (Timestep 999→0)', fontsize=12)
                ax.set_ylabel('Normalized Affinity (Vina scale)', fontsize=12)
                ax.set_title(f'{head1_label} BA Prediction During Sampling', fontsize=14, fontweight='bold')
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)

                # Add secondary Y-axis for Vina scale
                y_min_on, y_max_on = exp_on_traj[:, :max_vis_samples].min(), exp_on_traj[:, :max_vis_samples].max()
                y_range_on = max(y_max_on - y_min_on, 0.01)
                ax.set_ylim(max(0, y_min_on - 0.05*y_range_on), min(1, y_max_on + 0.05*y_range_on))
                yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 6)
                ax.set_yticks(yticks)
                ax.set_yticklabels([f'{v:.3f} ({-v*16:.1f})' for v in yticks])

                # Head2 (Off-target) plot
                ax = axes[1]
                for i in range(max_vis_samples):
                    ax.plot(timesteps, exp_off_traj[:, i], label=f'Sample {i+1}', alpha=0.7, linewidth=2)
                ax.set_xlabel('Iteration (Timestep 999→0)', fontsize=12)
                ax.set_ylabel('Normalized Affinity (Vina scale)', fontsize=12)
                ax.set_title(f'{head2_label} BA Prediction During Sampling', fontsize=14, fontweight='bold')
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)

                y_min_off, y_max_off = exp_off_traj[:, :max_vis_samples].min(), exp_off_traj[:, :max_vis_samples].max()
                y_range_off = max(y_max_off - y_min_off, 0.01)
                ax.set_ylim(max(0, y_min_off - 0.05*y_range_off), min(1, y_max_off + 0.05*y_range_off))
                yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 6)
                ax.set_yticks(yticks)
                ax.set_yticklabels([f'{v:.3f} ({-v*16:.1f})' for v in yticks])

                plt.tight_layout()
                plt.savefig(vis_dir / 'head1_head2_trajectories_separate.png', dpi=300, bbox_inches='tight')
                plt.close()

                # ====================================================================
                # Plot 2: Selectivity trajectory (Head1 - Head2)
                # ====================================================================
                fig, ax = plt.subplots(figsize=(14, 6))
                all_selectivity = []
                for i in range(max_vis_samples):
                    selectivity = exp_on_traj[:, i] - exp_off_traj[:, i]
                    ax.plot(timesteps, selectivity, label=f'Sample {i+1}', linewidth=2.5, alpha=0.8)
                    all_selectivity.extend([selectivity.min(), selectivity.max()])

                ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Zero selectivity')
                ax.set_xlabel('Iteration (Timestep 999→0)', fontsize=12)
                ax.set_ylabel('Selectivity (Head1 - Head2)', fontsize=12)
                ax.set_title(f'Selectivity Evolution: {head1_label} - {head2_label}', fontsize=13, fontweight='bold')
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)

                # Color regions
                y_min_sel, y_max_sel = min(all_selectivity), max(all_selectivity)
                y_range_sel = max(y_max_sel - y_min_sel, 0.01)
                ax.set_ylim(y_min_sel - 0.05*y_range_sel, y_max_sel + 0.05*y_range_sel)
                ax.axhspan(0, ax.get_ylim()[1], alpha=0.1, color='green')
                ax.axhspan(ax.get_ylim()[0], 0, alpha=0.1, color='red')

                plt.tight_layout()
                plt.savefig(vis_dir / 'selectivity_trajectory.png', dpi=300, bbox_inches='tight')
                plt.close()

                # ====================================================================
                # Plot 3: Combined comparison per sample
                # ====================================================================
                n_cols = 2
                n_rows = (max_vis_samples + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
                axes = axes.flatten() if max_vis_samples > 1 else [axes]

                for i in range(max_vis_samples):
                    ax = axes[i]
                    line_on = ax.plot(timesteps, exp_on_traj[:, i], label='Head1 (On-target)', color='blue', linewidth=2.5, alpha=0.8)
                    line_off = ax.plot(timesteps, exp_off_traj[:, i], label='Head2 (Off-target)', color='red', linewidth=2.5, alpha=0.8)

                    # Secondary axis for selectivity
                    selectivity = exp_on_traj[:, i] - exp_off_traj[:, i]
                    ax2 = ax.twinx()
                    line_sel = ax2.plot(timesteps, selectivity, label='Selectivity', color='green', linewidth=2, alpha=0.6, linestyle='--')
                    ax2.set_ylabel('Selectivity', fontsize=11, color='green')
                    ax2.tick_params(axis='y', labelcolor='green')
                    ax2.axhline(y=0, color='black', linestyle=':', alpha=0.3)

                    ax.set_xlabel('Iteration (Timestep 999→0)', fontsize=11)
                    ax.set_ylabel('Normalized Affinity', fontsize=11)
                    ax.set_title(f'Sample {i+1}: Head1 vs Head2', fontsize=12, fontweight='bold')

                    lines = line_on + line_off + line_sel
                    labels = [l.get_label() for l in lines]
                    ax.legend(lines, labels, loc='upper left', fontsize=9)
                    ax.grid(True, alpha=0.3)

                    # Stats text
                    textstr = f'Init: H1={exp_on_traj[0, i]:.3f}, H2={exp_off_traj[0, i]:.3f}, Sel={selectivity[0]:.3f}\n'
                    textstr += f'Final: H1={exp_on_traj[-1, i]:.3f}, H2={exp_off_traj[-1, i]:.3f}, Sel={selectivity[-1]:.3f}'
                    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                # Hide unused subplots
                for i in range(max_vis_samples, len(axes)):
                    axes[i].set_visible(False)

                plt.tight_layout()
                plt.savefig(vis_dir / 'head1_head2_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()

                # ====================================================================
                # Plot 4: Mean trajectory with confidence interval
                # ====================================================================
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                # Head1 mean
                ax = axes[0]
                mean_on = exp_on_traj.mean(axis=1)
                std_on = exp_on_traj.std(axis=1)
                ax.plot(timesteps, mean_on, 'b-', linewidth=2.5, label='Head1 Mean')
                ax.fill_between(timesteps, mean_on - std_on, mean_on + std_on, alpha=0.3, color='blue')
                ax.set_xlabel('Iteration', fontsize=12)
                ax.set_ylabel('Normalized Affinity', fontsize=12)
                ax.set_title(f'{head1_label}\nMean ± Std', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Head2 mean
                ax = axes[1]
                mean_off = exp_off_traj.mean(axis=1)
                std_off = exp_off_traj.std(axis=1)
                ax.plot(timesteps, mean_off, 'r-', linewidth=2.5, label='Head2 Mean')
                ax.fill_between(timesteps, mean_off - std_off, mean_off + std_off, alpha=0.3, color='red')
                ax.set_xlabel('Iteration', fontsize=12)
                ax.set_ylabel('Normalized Affinity', fontsize=12)
                ax.set_title(f'{head2_label}\nMean ± Std', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Selectivity mean
                ax = axes[2]
                selectivity_all = exp_on_traj - exp_off_traj
                mean_sel = selectivity_all.mean(axis=1)
                std_sel = selectivity_all.std(axis=1)
                ax.plot(timesteps, mean_sel, 'g-', linewidth=2.5, label='Selectivity Mean')
                ax.fill_between(timesteps, mean_sel - std_sel, mean_sel + std_sel, alpha=0.3, color='green')
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.set_xlabel('Iteration', fontsize=12)
                ax.set_ylabel('Selectivity (H1 - H2)', fontsize=12)
                ax.set_title('Selectivity (Head1 - Head2)\nMean ± Std', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(vis_dir / 'mean_trajectory_with_std.png', dpi=300, bbox_inches='tight')
                plt.close()

                # ====================================================================
                # Plot 5: Guidance Effect Analysis
                # ====================================================================
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Cumulative change from initial
                ax = axes[0]
                for i in range(min(5, max_vis_samples)):
                    delta_on = exp_on_traj[:, i] - exp_on_traj[0, i]
                    delta_off = exp_off_traj[:, i] - exp_off_traj[0, i]
                    ax.plot(timesteps, delta_on, linestyle='-', linewidth=2, label=f'H1 Sample {i+1}', alpha=0.8)
                    ax.plot(timesteps, delta_off, linestyle='--', linewidth=2, label=f'H2 Sample {i+1}', alpha=0.8)
                ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
                ax.set_xlabel('Iteration', fontsize=12)
                ax.set_ylabel('Change from Initial', fontsize=12)
                ax.set_title('Cumulative Affinity Change\n(Solid: Head1, Dashed: Head2)', fontsize=12, fontweight='bold')
                ax.legend(fontsize=8, ncol=2)
                ax.grid(True, alpha=0.3)

                # Rate of change
                ax = axes[1]
                window = min(50, num_steps // 10)
                if window > 1:
                    for i in range(min(3, max_vis_samples)):
                        rate_on = np.gradient(exp_on_traj[:, i])
                        rate_off = np.gradient(exp_off_traj[:, i])
                        rate_on_smooth = np.convolve(rate_on, np.ones(window)/window, mode='same')
                        rate_off_smooth = np.convolve(rate_off, np.ones(window)/window, mode='same')
                        ax.plot(timesteps, rate_on_smooth, linestyle='-', linewidth=2, label=f'H1 Sample {i+1}')
                        ax.plot(timesteps, rate_off_smooth, linestyle='--', linewidth=2, label=f'H2 Sample {i+1}')
                ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
                ax.set_xlabel('Iteration', fontsize=12)
                ax.set_ylabel('Rate of Change', fontsize=12)
                ax.set_title(f'Rate of Affinity Change\n(Moving Avg, window={window})', fontsize=12, fontweight='bold')
                ax.legend(fontsize=8, ncol=2)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(vis_dir / 'guidance_effect_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f'Trajectory visualizations saved to: {vis_dir}')
                logger.info(f'  - head1_head2_trajectories_separate.png')
                logger.info(f'  - selectivity_trajectory.png')
                logger.info(f'  - head1_head2_comparison.png')
                logger.info(f'  - mean_trajectory_with_std.png')
                logger.info(f'  - guidance_effect_analysis.png')

            except Exception as e:
                logger.warning(f'Failed to generate trajectory visualizations: {e}')
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()
