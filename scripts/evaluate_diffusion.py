import argparse
import os
import sys
# Add current directory to path for internal module imports
sys.path.append(os.path.abspath('./'))

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

# Evaluation, reconstruction, and transformation utility imports
from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask

# Helper function for parsing affinity weights
def parse_affinity_weight(weight_str):
    """
    Parse affinity weight from string format to float
    Handles cases like '[0.]', '0.5', etc.
    """
    try:
        # Try direct float conversion first
        return float(weight_str)
    except ValueError:
        # Handle string representations of arrays like '[0.]'
        import ast
        try:
            # Parse as Python literal (list, tuple, etc.)
            parsed = ast.literal_eval(weight_str)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                return float(parsed[0])
            else:
                return 0.0
        except:
            # Fallback: extract numbers using regex
            import re
            numbers = re.findall(r'-?\d+\.?\d*', weight_str)
            if numbers:
                return float(numbers[0])
            else:
                return 0.0

# Helper function for clean dictionary printing
def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')

# Helper function for printing ring size distribution
def print_ring_ratio(all_ring_sizes, logger):
    ring_info = {}
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')
        ring_info[ring_size] = f'{n_mol / len(all_ring_sizes):.3f}'
    return ring_info

def main():
     # --- 1. Setup and Configuration ---
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # WARN: important turn on when evaluate pdbbind related proteins
    ################
    parser.add_argument('--eval_pdbbind', action='store_true')
    ################
    
    parser.add_argument('--sample_path', type=str, default='./test_poc/')
    parser.add_argument('--verbose', type=eval, default=True)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--save_complex', action='store_true', default=False, help='Save protein-ligand complex')
    parser.add_argument('--protein_root', type=str, default='./data/test_set/')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, default='vina_dock', choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=16)
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        exit()
    args = parser.parse_args()

    # Set up result path and logger
    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # --- 2. Load Generated Data ---

    # Load generated data
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    # Initialize evaluation metrics
    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()

    # --- 3. Main Evaluation Loop ---

    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        

        r = torch.load(r_name)

        # Unified format: one sample per file with keys pos, v, pos_traj, v_traj, exp_on, exp_off
        pos = r['pos']      # [num_atoms, 3]
        v = r['v']          # [num_atoms]
        pos_traj = r.get('pos_traj', None)   # [num_steps, num_atoms, 3]
        v_traj = r.get('v_traj', None)       # [num_steps, num_atoms]

        if pos_traj is not None:
            all_pred_ligand_pos = [pos_traj]
            all_pred_ligand_v = [v_traj]
        else:
            all_pred_ligand_pos = [np.expand_dims(pos, axis=0)]
            all_pred_ligand_v = [np.expand_dims(v, axis=0)]

        all_pred_exp_score = [r.get('exp_on', 0.0)]
        all_pred_exp_atom_traj = [np.ones_like(all_pred_ligand_v[0])]

        num_samples += len(all_pred_ligand_pos)

        for sample_idx, (pred_pos, pred_v, pred_exp_score, pred_exp_atom_weight) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v, all_pred_exp_score, all_pred_exp_atom_traj)):

            pred_pos, pred_v, pred_exp, pred_exp_atom_weight = pred_pos[args.eval_step], pred_v[args.eval_step], pred_exp_score, pred_exp_atom_weight[args.eval_step]

            # --- 3a. Atom Stability Check ---

            # stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
            all_atom_types += Counter(pred_atom_type)

            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist

            # --- 3b. Molecule Reconstruction ---

            # reconstruction
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic, pred_exp_atom_weight)
                smiles = Chem.MolToSmiles(mol)
            except reconstruct.MolReconsError:
                if args.verbose:
                    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                continue
            n_recon_success += 1
            

            if '.' in smiles:
                continue
            n_complete += 1
            
            # --- 3c. Chemical Properties and Docking Score ---

            # chemical and docking check
            # try:

            chem_results = scoring_func.get_chem(mol)

            # Docking evaluation (use dock_generated_ligands.py for full docking pipeline)
            vina_results = None
            if args.docking_mode != 'none':
                try:
                    protein_fn = r.get('protein_filename', None)
                    ligand_fn = r.get('ligand_filename', None)

                    if protein_fn is None and ligand_fn is not None:
                        if args.eval_pdbbind:
                            protein_fn = os.path.join(
                                os.path.dirname(ligand_fn),
                                os.path.basename(ligand_fn)[:4] + '_protein.pdb'
                            )
                        else:
                            protein_fn = os.path.join(
                                os.path.dirname(ligand_fn),
                                os.path.basename(ligand_fn)[:10] + '.pdb'
                            )

                    if protein_fn is not None:
                        if args.docking_mode == 'qvina':
                            vina_task = QVinaDockingTask.from_generated_mol(
                                mol, protein_fn, protein_root=args.protein_root)
                            vina_results = vina_task.run_sync()
                        elif args.docking_mode in ['vina_score', 'vina_dock']:
                            vina_task = VinaDockingTask.from_generated_mol(
                                mol, protein_fn, protein_root=args.protein_root)
                            score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                            minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                            vina_results = {
                                'score_only': score_only_results,
                                'minimize': minimize_results
                            }
                            if args.docking_mode == 'vina_dock':
                                docking_save_path = None
                                if args.save_complex:
                                    complex_path = os.path.join(result_path, 'complexes')
                                    os.makedirs(complex_path, exist_ok=True)
                                    docking_save_path = os.path.join(complex_path, f'{example_idx}_{sample_idx}_dock.pdb')
                                docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness, save_path=docking_save_path)
                                vina_results['dock'] = docking_results

                            sdf_path = os.path.join(result_path, f"sdf_{r_name[:-3].split('_')[-1]}")
                            os.makedirs(sdf_path, exist_ok=True)
                            writer = Chem.SDWriter(os.path.join(sdf_path, f'res_{sample_idx}.sdf'))
                            writer.write(mol)
                            writer.close()
                    else:
                        if sample_idx == 0 and example_idx == 0:
                            logger.info('No protein filename in result files. Skipping docking. '
                                        'Use dock_generated_ligands.py for docking evaluation.')
                except Exception as e:
                    if args.verbose:
                        logger.warning(f'Docking failed for {example_idx}_{sample_idx}: {e}')

            n_eval_success += 1

            # now we only consider complete molecules as success
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist

            success_pair_dist += pair_dist
            success_atom_types += Counter(pred_atom_type)

            results.append({
                'mol': mol,
                'smiles': smiles,
                'pred_pos': pred_pos,
                'pred_v': pred_v,
                'chem_results': chem_results,
                'vina': vina_results,
                'pred_exp': pred_exp,
            })
    logger.info(f'Evaluate done! {num_samples} samples in total.')

# --- 4. Final Result Aggregation and Logging ---

    fraction_mol_stable = all_mol_stable / num_samples
    fraction_atm_stable = all_atom_stable / all_n_atom
    fraction_recon = n_recon_success / num_samples
    fraction_eval = n_eval_success / num_samples
    fraction_complete = n_complete / num_samples
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
        'complete': fraction_complete
    }
    print_dict(validity_dict, logger)

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    logger.info('JS bond distances of complete mols: ')
    print_dict(c_bond_length_dict, logger)

    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    print_dict(success_js_metrics, logger)

    atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
    if np.isnan(atom_type_js):
        logger.info('Atom type JS: NaN (no valid atoms found)')
    else:
        logger.info('Atom type JS: %.4f' % atom_type_js)

    if args.save:
        eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                            metrics=success_js_metrics,
                                            save_path=os.path.join(result_path, f'pair_dist_hist_{args.eval_step}.png'))

    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        n_recon_success, n_complete, len(results)))

    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    if args.docking_mode == 'qvina':
        vina = [r['vina'][0]['affinity'] for r in results]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif args.docking_mode in ['vina_dock', 'vina_score']:
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if args.docking_mode == 'vina_dock':
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

    # check ring distribution
    ring_info = print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)

    if args.save:
        torch.save({
            'info': 'Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        n_recon_success, n_complete, len(results)),
            'ring_info': ring_info,
            'stability': validity_dict,
            'c_bond_length_dict': c_bond_length_dict,
            'success_js_metrics': success_js_metrics,
            'atom_type_js': atom_type_js,
            'bond_length': all_bond_dist,
            'all_results': results
        }, os.path.join(result_path, f'metrics_{args.eval_step}_wo_vina.pt'))

if __name__ == '__main__':
    main()