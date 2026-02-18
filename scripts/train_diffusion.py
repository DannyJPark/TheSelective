"""
Training script for TheSelective model.

Supports both single-head and dual-head modes.

Config options:
- use_dual_head_sam_pl: False -> Single head
- use_dual_head_sam_pl: True  -> Dual head with head2_mode
    - head2_mode: protein_query_atom | ligand_query_atom | bidirectional_query_atom

Usage:
    python scripts/train_diffusion.py --config configs/training.yml

    # With wandb
    python scripts/train_diffusion.py --config configs/training.yml --wandb --wandb_project theselective
"""

import os
import sys
import argparse
import shutil
import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy import stats
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

sys.path.append(os.path.abspath('./'))

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans

from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_auroc(y_true, y_pred, feat_mode, logger=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        if logger:
            logger.info(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


def get_pearsonr(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return stats.pearsonr(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/training.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    # Wandb options
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='theselective', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity (team/username)')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name (default: auto-generated)')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        exit()
    args = parser.parse_args()

    # Load checkpoint or config
    if args.ckpt:
        print(f'Loading checkpoint: {args.ckpt}...')
        ckpt = torch.load(args.ckpt, map_location=args.device)
        config = ckpt['config']
    else:
        config = misc.load_config(args.config)

    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Check dual head mode
    use_dual_head = getattr(config.model, 'use_dual_head_sam_pl', False)
    head2_mode = getattr(config.model, 'head2_mode', 'protein_query_atom')

    # Training mode: 'default' (with affinity), 'generation_only' (no affinity, like TargetDiff)
    train_mode = getattr(config.train, 'mode', 'default')

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    logger.info(args)
    logger.info(config)
    logger.info(f'Training Mode: {train_mode}')
    logger.info(f'Dual Head Mode: {use_dual_head}')
    if use_dual_head:
        logger.info(f'Head2 Mode: {head2_mode}')

    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Initialize wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        logger.warning('wandb not installed. Install with: pip install wandb')

    if use_wandb:
        wandb_run_name = args.wandb_name or f"{config_name}_{args.tag}" if args.tag else config_name
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config={
                'model': dict(config.model),
                'train': dict(config.train),
                'data': dict(config.data),
                'use_dual_head': use_dual_head,
                'head2_mode': head2_mode if use_dual_head else None,
                'train_mode': train_mode,
            },
            dir=log_dir,
            reinit=True
        )
        logger.info(f'Wandb initialized: {wandb.run.url}')

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
        trans.NormalizeVina(config.data.name)
    ]

    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
    )

    if config.data.name == 'pl':
        train_set, val_set, test_set = subsets['train'], subsets['test'], []
    elif config.data.name == 'pdbbind':
        train_set, val_set, test_set = subsets['train'], subsets['val'], subsets['test']
    else:
        raise ValueError(f'Unknown dataset: {config.data.name}')

    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)} Test: {len(test_set)}')

    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(
        val_set, config.train.batch_size, shuffle=False,
        follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys
    )

    # Model
    logger.info('Building model...')
    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)

    logger.info(f'Protein feature dim: {protein_featurizer.feature_dim}')
    logger.info(f'Ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    start_it = 0
    if args.ckpt:
        model.load_state_dict(ckpt['model'])
        logger.info('Model weights loaded from checkpoint')

    def train(it):
        model.train()
        optimizer.zero_grad()

        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            # Add noise to protein positions
            protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise

            if train_mode == 'generation_only':
                # Generation-only mode (TargetDiff style, no affinity loss)
                results = model.get_diffusion_loss_generation_only(
                    protein_pos=gt_protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch.protein_element_batch,
                    ligand_pos=batch.ligand_pos,
                    ligand_v=batch.ligand_atom_feature_full,
                    batch_ligand=batch.ligand_element_batch
                )
                loss = results['loss']
                loss_pos = results['loss_pos']
                loss_v = results['loss_v']
                loss_exp = results['loss_exp']
                loss_head1 = torch.tensor(0.0)
                loss_head2 = torch.tensor(0.0)
            elif use_dual_head:
                # Dual head mode
                results = model.get_diffusion_loss_dual_head(
                    protein_pos=gt_protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch.protein_element_batch,
                    ligand_pos=batch.ligand_pos,
                    ligand_v=batch.ligand_atom_feature_full,
                    batch_ligand=batch.ligand_element_batch,
                    on_target_affinity=batch.affinity.float()
                )
                loss = results['loss']
                loss_pos = results['loss_pos']
                loss_v = results['loss_v']
                loss_exp = results['loss_exp']
                loss_head1 = results['loss_head1']
                loss_head2 = results['loss_head2']
            else:
                # Single head mode (Original KGDiff)
                results = model.get_diffusion_loss(
                    protein_pos=gt_protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    affinity=batch.affinity.float(),
                    batch_protein=batch.protein_element_batch,
                    ligand_pos=batch.ligand_pos,
                    ligand_v=batch.ligand_atom_feature_full,
                    batch_ligand=batch.ligand_element_batch
                )
                loss = results['loss']
                loss_pos = results['loss_pos']
                loss_v = results['loss_v']
                loss_exp = results['loss_exp']
                loss_head1 = loss_exp
                loss_head2 = torch.tensor(0.0)

            loss = loss / config.train.n_acc_batch
            loss.backward()

        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % args.train_report_iter == 0:
            if train_mode == 'generation_only':
                logger.info(
                    '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f) | Lr: %.6f | Grad: %.4f' % (
                        it, loss * config.train.n_acc_batch, loss_pos, loss_v,
                        optimizer.param_groups[0]['lr'], orig_grad_norm
                    )
                )
            elif use_dual_head:
                logger.info(
                    '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | exp %.6f | h1 %.6f | h2 %.6f) | Lr: %.6f | Grad: %.4f' % (
                        it, loss * config.train.n_acc_batch, loss_pos, loss_v, loss_exp,
                        loss_head1, loss_head2,
                        optimizer.param_groups[0]['lr'], orig_grad_norm
                    )
                )
            else:
                logger.info(
                    '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | exp %.6f) | Lr: %.6f | Grad: %.4f' % (
                        it, loss * config.train.n_acc_batch, loss_pos, loss_v, loss_exp,
                        optimizer.param_groups[0]['lr'], orig_grad_norm
                    )
                )

            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()

            # Wandb logging
            if use_wandb:
                wandb_log = {
                    'train/loss': float(loss * config.train.n_acc_batch),
                    'train/loss_pos': float(loss_pos),
                    'train/loss_v': float(loss_v),
                    'train/loss_exp': float(loss_exp),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/grad_norm': float(orig_grad_norm),
                    'iteration': it,
                }
                if use_dual_head:
                    wandb_log['train/loss_head1'] = float(loss_head1)
                    wandb_log['train/loss_head2'] = float(loss_head2)
                wandb.log(wandb_log, step=it)

    def validate(it):
        sum_loss, sum_loss_pos, sum_loss_v, sum_loss_exp, sum_n = 0, 0, 0, 0, 0
        sum_loss_head1, sum_loss_head2 = 0, 0
        all_pred_v, all_true_v = [], []
        all_pred_exp_head1, all_pred_exp_head2, all_true_exp = [], [], []

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs

                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)

                    if train_mode == 'generation_only':
                        # Generation-only mode (no affinity)
                        results = model.get_diffusion_loss_generation_only(
                            protein_pos=batch.protein_pos,
                            protein_v=batch.protein_atom_feature.float(),
                            batch_protein=batch.protein_element_batch,
                            ligand_pos=batch.ligand_pos,
                            ligand_v=batch.ligand_atom_feature_full,
                            batch_ligand=batch.ligand_element_batch,
                            time_step=time_step
                        )
                    elif use_dual_head:
                        results = model.get_diffusion_loss_dual_head(
                            protein_pos=batch.protein_pos,
                            protein_v=batch.protein_atom_feature.float(),
                            batch_protein=batch.protein_element_batch,
                            ligand_pos=batch.ligand_pos,
                            ligand_v=batch.ligand_atom_feature_full,
                            batch_ligand=batch.ligand_element_batch,
                            on_target_affinity=batch.affinity.float(),
                            time_step=time_step
                        )
                        pred_exp_head1 = results['pred_head1_exp']
                        pred_exp_head2 = results['pred_head2_exp']
                        loss_head1 = results['loss_head1']
                        loss_head2 = results['loss_head2']

                        sum_loss_head1 += float(loss_head1) * batch_size
                        sum_loss_head2 += float(loss_head2) * batch_size
                        all_pred_exp_head1.append(pred_exp_head1.detach().cpu().numpy())
                        all_pred_exp_head2.append(pred_exp_head2.detach().cpu().numpy())
                    else:
                        results = model.get_diffusion_loss(
                            protein_pos=batch.protein_pos,
                            protein_v=batch.protein_atom_feature.float(),
                            affinity=batch.affinity.float(),
                            batch_protein=batch.protein_element_batch,
                            ligand_pos=batch.ligand_pos,
                            ligand_v=batch.ligand_atom_feature_full,
                            batch_ligand=batch.ligand_element_batch,
                            time_step=time_step
                        )
                        pred_exp = results['pred_exp']
                        all_pred_exp_head1.append(pred_exp.detach().cpu().numpy())

                    loss = results['loss']
                    loss_pos = results['loss_pos']
                    loss_v = results['loss_v']
                    loss_exp = results['loss_exp']

                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_loss_exp += float(loss_exp) * batch_size
                    sum_n += batch_size

                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())
                    if train_mode != 'generation_only':
                        all_true_exp.append(batch.affinity.float().detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        avg_loss_exp = sum_loss_exp / sum_n

        atom_auroc = get_auroc(
            np.concatenate(all_true_v),
            np.concatenate(all_pred_v, axis=0),
            feat_mode=config.data.transform.ligand_atom_mode,
            logger=logger
        )

        # Pearson correlation (only for modes with affinity prediction)
        if train_mode != 'generation_only' and len(all_pred_exp_head1) > 0:
            pred_exp_head1_all = np.concatenate(all_pred_exp_head1, axis=0)
            true_exp_all = np.concatenate(all_true_exp, axis=0)
            exp_pearsonr_head1 = get_pearsonr(true_exp_all, pred_exp_head1_all)
        else:
            exp_pearsonr_head1 = (0.0, 1.0)  # Placeholder for generation_only mode

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        if train_mode == 'generation_only':
            # Generation-only mode logging (no affinity metrics)
            logger.info(
                '[Validate] Iter %05d | Loss %.6f | pos %.6f | v %.6f | auroc %.4f' % (
                    it, avg_loss, avg_loss_pos, avg_loss_v, atom_auroc
                )
            )
        elif use_dual_head:
            avg_loss_head1 = sum_loss_head1 / sum_n
            avg_loss_head2 = sum_loss_head2 / sum_n
            pred_exp_head2_all = np.concatenate(all_pred_exp_head2, axis=0)
            exp_pearsonr_head2 = get_pearsonr(true_exp_all, pred_exp_head2_all)

            logger.info(
                '[Validate] Iter %05d | Loss %.6f | pos %.6f | v %.6f | exp %.6f | h1 %.6f (pcc %.4f) | h2 %.6f (pcc %.4f) | auroc %.4f' % (
                    it, avg_loss, avg_loss_pos, avg_loss_v, avg_loss_exp,
                    avg_loss_head1, exp_pearsonr_head1[0],
                    avg_loss_head2, exp_pearsonr_head2[0],
                    atom_auroc
                )
            )
            writer.add_scalar('val/loss_head1', avg_loss_head1, it)
            writer.add_scalar('val/loss_head2', avg_loss_head2, it)
            writer.add_scalar('val/pcc_head1', exp_pearsonr_head1[0], it)
            writer.add_scalar('val/pcc_head2', exp_pearsonr_head2[0], it)
        else:
            logger.info(
                '[Validate] Iter %05d | Loss %.6f | pos %.6f | v %.6f | exp %.6f | pcc %.4f | auroc %.4f' % (
                    it, avg_loss, avg_loss_pos, avg_loss_v, avg_loss_exp,
                    exp_pearsonr_head1[0], atom_auroc
                )
            )

        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v', avg_loss_v, it)
        writer.add_scalar('val/atom_auroc', atom_auroc, it)

        if train_mode != 'generation_only':
            writer.add_scalar('val/loss_exp', avg_loss_exp, it)
            writer.add_scalar('val/pcc', exp_pearsonr_head1[0], it)

            # Plot correlation (only for modes with affinity)
            fig = sns.lmplot(
                data=pd.DataFrame({'pred': pred_exp_head1_all, 'true': true_exp_all}),
                x='pred', y='true'
            ).set(title=f'Head1 PCC: {exp_pearsonr_head1[0]:.4f}').figure
            writer.add_figure('val/pcc_fig_head1', fig, it)

            if use_dual_head:
                fig2 = sns.lmplot(
                    data=pd.DataFrame({'pred': pred_exp_head2_all, 'true': true_exp_all}),
                    x='pred', y='true'
                ).set(title=f'Head2 PCC: {exp_pearsonr_head2[0]:.4f}').figure
                writer.add_figure('val/pcc_fig_head2', fig2, it)

        # Wandb validation logging
        if use_wandb:
            wandb_val_log = {
                'val/loss': avg_loss,
                'val/loss_pos': avg_loss_pos,
                'val/loss_v': avg_loss_v,
                'val/atom_auroc': atom_auroc,
            }
            if train_mode != 'generation_only':
                wandb_val_log.update({
                    'val/loss_exp': avg_loss_exp,
                    'val/pcc': exp_pearsonr_head1[0],
                })
                if use_dual_head:
                    wandb_val_log.update({
                        'val/loss_head1': avg_loss_head1,
                        'val/loss_head2': avg_loss_head2,
                        'val/pcc_head1': exp_pearsonr_head1[0],
                        'val/pcc_head2': exp_pearsonr_head2[0],
                    })
                    # Log correlation plots to wandb
                    wandb.log({
                        'val/pcc_fig_head1': wandb.Image(fig),
                        'val/pcc_fig_head2': wandb.Image(fig2),
                    }, step=it)
                else:
                    wandb.log({'val/pcc_fig': wandb.Image(fig)}, step=it)

            wandb.log(wandb_val_log, step=it)

        plt.close('all')
        writer.flush()
        return avg_loss

    try:
        best_loss, best_iter = None, None
        best_ckpt_path = None
        for it in range(start_it, config.train.max_iters):
            train(it)

            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)

                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')

                    # Delete previous best checkpoint if exists
                    if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                        os.remove(best_ckpt_path)
                        logger.info(f'[Checkpoint] Deleted previous best checkpoint: {best_ckpt_path}')

                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    best_ckpt_path = ckpt_path
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                    logger.info(f'[Checkpoint] Saved new best checkpoint: {ckpt_path}')

                    # Log best model info to wandb
                    if use_wandb:
                        wandb.run.summary['best_val_loss'] = best_loss
                        wandb.run.summary['best_iter'] = best_iter
                else:
                    logger.info(f'[Validate] Val loss not improved. Best: {best_loss:.6f} at iter {best_iter}')

    except KeyboardInterrupt:
        logger.info('Training interrupted by user.')

    finally:
        # Save final model artifact to wandb
        if use_wandb:
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                artifact = wandb.Artifact(
                    name=f'model-{wandb.run.id}',
                    type='model',
                    description=f'Best model at iter {best_iter}, val_loss={best_loss:.6f}'
                )
                artifact.add_file(best_ckpt_path)
                wandb.log_artifact(artifact)
                logger.info(f'Model artifact saved to wandb')

            wandb.finish()
            logger.info('Wandb run finished')


if __name__ == '__main__':
    main()
