#!/usr/bin/env python3
"""
Analyze docking results for LOW TM-score pairs in tmscore_extreme_pairs.txt.
Format: on_id,high_off_id,high_tmscore,HIGH_off_id,HIGH_tmscore
This script analyzes the LOW TM-score pairs (columns 4-5).

FILTERED VERSION:
1. Handles off-target scores from docking_results.off_target_X.affinity
2. Skips v2 models that don't have docking results yet
3. Properly handles disconnected fragment failures
4. Calculates QED and SA from SMILES for AR/Pocket2Mol results
5. FILTERS OUT DOCKING FAILURES: Any molecule with on-target OR off-target score >= 0 is excluded
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.evaluation import sascorer


def calculate_qed_sa_from_smiles(smiles):
    """Calculate QED and SA scores from SMILES string.

    Returns:
    - QED: 0-1 range (higher is better)
    - SA: 0-1 range (higher is better/easier to synthesize) - NORMALIZED score
         Original SA: 1-10 (lower is better) -> Normalized: (10 - SA) / 9
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        qed = QED.qed(mol)

        # Calculate SA score and normalize to 0-1 range (higher = easier to synthesize)
        sa_raw = sascorer.calculateScore(mol)
        sa_normalized = (10 - sa_raw) / 9

        return qed, sa_normalized
    except Exception as e:
        print(f"Error calculating QED/SA for SMILES {smiles}: {e}")
        return None, None


def load_low_tmscore_pairs(file_path):
    """Load LOW TM-score pairs from tmscore_extreme_pairs.txt file."""
    pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse format: on_id,high_off_id,high_tmscore,LOW_off_id,LOW_tmscore
            # We want columns: 0, 3, 4 (on_id, LOW_off_id, LOW_tmscore)
            parts = line.split(',')
            if len(parts) >= 5:
                on_id = int(parts[0])
                low_off_id = int(parts[3])  # Column 3: LOW off-target ID
                low_tmscore = float(parts[4])  # Column 4: LOW TM-score
                pairs.append((on_id, low_off_id, low_tmscore))

    return pairs


def parse_docking_results_json(file_path):
    """Parse docking_results.json file."""
    if not Path(file_path).exists():
        return None

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Check if this is a list format (from dock_generated_ligands.py)
        if isinstance(data, list):
            # Convert list format to dict format
            converted = {}
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    continue

                mol_id = item.get('ligand_name', f'molecule_{idx}')

                # Extract QED and SA from chem_results if available
                qed = item.get('qed')
                sa = item.get('sa')

                # Check in chem_results at top level
                if 'chem_results' in item and item['chem_results'] is not None and isinstance(item['chem_results'], dict):
                    qed = item['chem_results'].get('qed', qed)
                    sa = item['chem_results'].get('sa', sa)

                # Check in docking_results.on_target.chem_results
                if 'docking_results' in item and item['docking_results'] is not None:
                    if isinstance(item['docking_results'], dict) and 'on_target' in item['docking_results']:
                        on_target_data = item['docking_results']['on_target']
                        if isinstance(on_target_data, dict) and 'chem_results' in on_target_data:
                            chem_res = on_target_data['chem_results']
                            if chem_res is not None and isinstance(chem_res, dict):
                                qed = chem_res.get('qed', qed)
                                sa = chem_res.get('sa', sa)

                converted[mol_id] = {
                    'on_target': {
                        'vina_score': item.get('on_target_affinity')
                    },
                    'off_targets': {},
                    'qed': qed,
                    'sa': sa,
                    'docking_results': item.get('docking_results', {})  # Keep original docking_results
                }

                # Get off-target affinities (handle both list and dict formats)
                off_affinities = item.get('off_target_affinities', [])
                if off_affinities:
                    # Handle dict format (BInD): {'off_target_1': -10.692}
                    if isinstance(off_affinities, dict):
                        for off_idx, (off_key, off_score) in enumerate(off_affinities.items()):
                            converted[mol_id]['off_targets'][f'off_{off_idx}'] = {
                                'vina_score': off_score
                            }
                    # Handle list format: [-10.692]
                    else:
                        for off_idx, off_score in enumerate(off_affinities):
                            converted[mol_id]['off_targets'][f'off_{off_idx}'] = {
                                'vina_score': off_score
                            }

            return converted

        # Check if this is the new format from dock_benchmark_tmscore_pairs.py
        # Format: {"on_id": X, "off_id": Y, "results": [...]}
        if isinstance(data, dict) and 'results' in data:
            # Return the results array directly
            return data['results']

        return data
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def extract_scores_from_results(results, max_molecules=None, is_reference=False, filter_failed_docking=True):
    """Extract on-target and off-target scores from parsed results.

    Args:
        results: Dictionary of molecule results
        max_molecules: Maximum number of molecules to analyze (default: None = all molecules)
        is_reference: Whether this is Reference model (to normalize SA from 1-10 to 0-1)
        filter_failed_docking: If True, exclude molecules with on-target OR off-target score >= 0 (docking failure)

    Returns:
        Tuple of (on_target_scores, off_target_scores, qed_scores, sa_scores,
                  vina_scores, vina_mins, filtered_count)
        where vina_scores and vina_mins are dicts with 'on_target' and 'off_target' keys
        filtered_count is the number of molecules filtered out due to docking failure
    """
    on_target_scores = []
    off_target_scores = []
    qed_scores = []
    sa_scores = []

    # Track vina score_only and minimize separately
    vina_score_on = []
    vina_score_off = []
    vina_min_on = []
    vina_min_off = []

    # Track filtered molecules count
    filtered_count = 0

    if not results:
        return on_target_scores, off_target_scores, qed_scores, sa_scores, \
               {'on_target': [], 'off_target': []}, {'on_target': [], 'off_target': []}, filtered_count

    # Handle both list and dict formats
    if isinstance(results, list):
        # List format from dock_benchmark_tmscore_pairs.py
        results_list = results
    else:
        # Dict format from other scripts
        results_list = list(results.items())

    molecule_count = 0
    for item in results_list:
        # Handle both (mol_id, mol_data) tuple and mol_data dict
        if isinstance(item, tuple):
            mol_id, mol_data = item
        else:
            mol_data = item
            mol_id = f"molecule_{molecule_count}"
        if mol_id == 'summary':
            continue

        if not isinstance(mol_data, dict):
            continue

        # Stop if we've reached the maximum number of molecules (if specified)
        if max_molecules is not None and molecule_count >= max_molecules:
            break

        molecule_count += 1

        # Extract QED and SA from multiple possible locations
        qed_value = None
        sa_value = None

        # First try direct keys
        if 'qed' in mol_data and mol_data['qed'] is not None:
            try:
                qed_value = float(mol_data['qed'])
            except:
                pass
        # Then try docking_results.on_target.chem_results
        elif 'docking_results' in mol_data and mol_data['docking_results']:
            if 'on_target' in mol_data['docking_results']:
                on_target_data = mol_data['docking_results']['on_target']
                if isinstance(on_target_data, dict) and 'chem_results' in on_target_data:
                    chem_results = on_target_data['chem_results']
                    if chem_results and 'qed' in chem_results and chem_results['qed'] is not None:
                        try:
                            qed_value = float(chem_results['qed'])
                        except:
                            pass

        if 'sa' in mol_data and mol_data['sa'] is not None:
            try:
                sa_value = float(mol_data['sa'])
            except:
                pass
        # Then try docking_results.on_target.chem_results
        elif 'docking_results' in mol_data and mol_data['docking_results']:
            if 'on_target' in mol_data['docking_results']:
                on_target_data = mol_data['docking_results']['on_target']
                if isinstance(on_target_data, dict) and 'chem_results' in on_target_data:
                    chem_results = on_target_data['chem_results']
                    if chem_results and 'sa' in chem_results and chem_results['sa'] is not None:
                        try:
                            sa_value = float(chem_results['sa'])
                        except:
                            pass

        # If QED or SA not found and we have SMILES, calculate them
        if (qed_value is None or sa_value is None) and 'smiles' in mol_data:
            smiles = mol_data['smiles']
            if smiles:
                calc_qed, calc_sa = calculate_qed_sa_from_smiles(smiles)
                if qed_value is None and calc_qed is not None:
                    qed_value = calc_qed
                if sa_value is None and calc_sa is not None:
                    sa_value = calc_sa

        # Append the scores if we have them
        if qed_value is not None:
            qed_scores.append(qed_value)
        if sa_value is not None:
            # Reference model: SA is raw score (1-10 range, lower is better)
            # Need to normalize to 0-1 range (higher = easier to synthesize)
            if is_reference and sa_value > 1.0:
                # Raw SA score detected (1-10 range), normalize it
                sa_normalized = (10 - sa_value) / 9.0
                sa_scores.append(sa_normalized)
            else:
                # Already normalized (0-1 range) or other models
                sa_scores.append(sa_value)

        # === FIX 1: Extract on-target scores ===
        on_score = None

        # Try direct on_target key (for dock_benchmark_tmscore_pairs.py format)
        if 'on_target' in mol_data:
            if isinstance(mol_data['on_target'], dict):
                # Try 'affinity' first (new format)
                on_score = mol_data['on_target'].get('affinity')
                # Then try 'vina_score' (old format)
                if on_score is None:
                    on_score = mol_data['on_target'].get('vina_score')
            else:
                on_score = mol_data['on_target']

        # Try on_target_affinity
        if on_score is None and 'on_target_affinity' in mol_data:
            on_score = mol_data['on_target_affinity']

        # Try docking_results.on_target.affinity
        if on_score is None and 'docking_results' in mol_data and mol_data['docking_results']:
            if isinstance(mol_data['docking_results'], dict) and 'on_target' in mol_data['docking_results']:
                on_target_data = mol_data['docking_results']['on_target']
                if isinstance(on_target_data, dict):
                    on_score = on_target_data.get('affinity')

        if on_score is not None and on_score != 'N/A':
            try:
                on_target_scores.append(float(on_score))
            except (ValueError, TypeError):
                pass

        # === FIX 2: Extract off-target scores properly ===
        # First try off_target dict (new format from dock_benchmark_tmscore_pairs.py)
        if 'off_target' in mol_data:
            if isinstance(mol_data['off_target'], dict):
                # Try 'affinity' first (new format)
                score = mol_data['off_target'].get('affinity')
                # Then try 'vina_score' (old format)
                if score is None:
                    score = mol_data['off_target'].get('vina_score')
                if score is not None and score != 'N/A':
                    try:
                        off_target_scores.append(float(score))
                    except (ValueError, TypeError):
                        pass
            else:
                # Direct value
                if mol_data['off_target'] is not None and mol_data['off_target'] != 'N/A':
                    try:
                        off_target_scores.append(float(mol_data['off_target']))
                    except (ValueError, TypeError):
                        pass

        # Then try off_targets dict (multi off-target format)
        elif 'off_targets' in mol_data and mol_data['off_targets']:
            for off_id, off_data in mol_data['off_targets'].items():
                if isinstance(off_data, dict):
                    score = off_data.get('vina_score')
                    if score is not None and score != 'N/A':
                        try:
                            off_target_scores.append(float(score))
                        except (ValueError, TypeError):
                            pass

        # Then try off_target_affinities array or dict
        elif 'off_target_affinities' in mol_data and mol_data['off_target_affinities']:
            off_affs = mol_data['off_target_affinities']
            # Handle dict format (BInD): {'off_target_1': -6.37}
            if isinstance(off_affs, dict):
                for score in off_affs.values():
                    if score is not None:
                        try:
                            off_target_scores.append(float(score))
                        except (ValueError, TypeError):
                            pass
            # Handle list format: [-6.37]
            else:
                for score in off_affs:
                    if score is not None:
                        try:
                            off_target_scores.append(float(score))
                        except (ValueError, TypeError):
                            pass

        # Finally try docking_results.off_target_X.affinity
        if not off_target_scores and 'docking_results' in mol_data and mol_data['docking_results']:
            dr = mol_data['docking_results']
            if isinstance(dr, dict):
                # Try off_target_1, off_target_2, etc.
                for key in sorted(dr.keys()):
                    if key.startswith('off_target'):
                        off_data = dr[key]
                        if isinstance(off_data, dict):
                            off_score = off_data.get('affinity')
                            if off_score is not None and off_score != 'N/A':
                                try:
                                    off_target_scores.append(float(off_score))
                                except (ValueError, TypeError):
                                    pass

        # === NEW: Extract vina score_only and minimize scores ===
        # Handle two formats:
        # 1. Head2 models: mol_data['docking_results']['on_target']['vina_results']
        # 2. TargetDiff/KGDiff: mol_data['on_target']['vina_results']

        # Try format 1: docking_results wrapper (head2 models)
        if 'docking_results' in mol_data and mol_data['docking_results']:
            dr = mol_data['docking_results']
            if isinstance(dr, dict):
                # Extract on-target vina scores
                if 'on_target' in dr and isinstance(dr['on_target'], dict):
                    vina_results = dr['on_target'].get('vina_results', {})
                    if isinstance(vina_results, dict):
                        # score_only (filter outliers > 1000)
                        if 'score_only' in vina_results and vina_results['score_only']:
                            try:
                                score = vina_results['score_only'][0].get('affinity')
                                if score is not None and float(score) <= 1000:
                                    vina_score_on.append(float(score))
                            except:
                                pass
                        # minimize (filter outliers > 1000)
                        if 'minimize' in vina_results and vina_results['minimize']:
                            try:
                                score = vina_results['minimize'][0].get('affinity')
                                if score is not None and float(score) <= 1000:
                                    vina_min_on.append(float(score))
                            except:
                                pass

                # Extract off-target vina scores
                for key in sorted(dr.keys()):
                    if key.startswith('off_target'):
                        off_data = dr[key]
                        if isinstance(off_data, dict):
                            vina_results = off_data.get('vina_results', {})
                            if isinstance(vina_results, dict):
                                # score_only (filter outliers > 1000)
                                if 'score_only' in vina_results and vina_results['score_only']:
                                    try:
                                        score = vina_results['score_only'][0].get('affinity')
                                        if score is not None and float(score) <= 1000:
                                            vina_score_off.append(float(score))
                                    except:
                                        pass
                                # minimize (filter outliers > 1000)
                                if 'minimize' in vina_results and vina_results['minimize']:
                                    try:
                                        score = vina_results['minimize'][0].get('affinity')
                                        if score is not None and float(score) <= 1000:
                                            vina_min_off.append(float(score))
                                    except:
                                        pass

        # Try format 2: direct on_target/off_target (TargetDiff/KGDiff from dock_benchmark_tmscore_pairs.py)
        else:
            # Extract on-target vina scores
            if 'on_target' in mol_data and isinstance(mol_data['on_target'], dict):
                vina_results = mol_data['on_target'].get('vina_results', {})
                if isinstance(vina_results, dict):
                    # score_only (filter outliers > 1000)
                    if 'score_only' in vina_results and vina_results['score_only']:
                        try:
                            score = vina_results['score_only'][0].get('affinity')
                            if score is not None and float(score) <= 1000:
                                vina_score_on.append(float(score))
                        except:
                            pass
                    # minimize (filter outliers > 1000)
                    if 'minimize' in vina_results and vina_results['minimize']:
                        try:
                            score = vina_results['minimize'][0].get('affinity')
                            if score is not None and float(score) <= 1000:
                                vina_min_on.append(float(score))
                        except:
                            pass

            # Extract off-target vina scores
            if 'off_target' in mol_data and isinstance(mol_data['off_target'], dict):
                vina_results = mol_data['off_target'].get('vina_results', {})
                if isinstance(vina_results, dict):
                    # score_only (filter outliers > 1000)
                    if 'score_only' in vina_results and vina_results['score_only']:
                        try:
                            score = vina_results['score_only'][0].get('affinity')
                            if score is not None and float(score) <= 1000:
                                vina_score_off.append(float(score))
                        except:
                            pass
                    # minimize (filter outliers > 1000)
                    if 'minimize' in vina_results and vina_results['minimize']:
                        try:
                            score = vina_results['minimize'][0].get('affinity')
                            if score is not None and float(score) <= 1000:
                                vina_min_off.append(float(score))
                        except:
                            pass

    vina_scores = {'on_target': vina_score_on, 'off_target': vina_score_off}
    vina_mins = {'on_target': vina_min_on, 'off_target': vina_min_off}

    # === FILTER DOCKING FAILURES ===
    # If filter_failed_docking is True, exclude molecules where on-target OR off-target score >= 0
    if filter_failed_docking and on_target_scores and off_target_scores:
        filtered_on = []
        filtered_off = []
        filtered_qed = []
        filtered_sa = []
        filtered_vina_score_on = []
        filtered_vina_score_off = []
        filtered_vina_min_on = []
        filtered_vina_min_off = []

        min_len = min(len(on_target_scores), len(off_target_scores))

        for i in range(min_len):
            on_val = on_target_scores[i]
            off_val = off_target_scores[i]

            # Check if either score is >= 0 (docking failure)
            if on_val >= 0 or off_val >= 0:
                filtered_count += 1
                continue  # Skip this molecule

            # Keep this molecule
            filtered_on.append(on_val)
            filtered_off.append(off_val)

            # Also keep corresponding QED/SA if they exist
            if i < len(qed_scores):
                filtered_qed.append(qed_scores[i])
            if i < len(sa_scores):
                filtered_sa.append(sa_scores[i])

            # Keep vina scores if they exist
            if i < len(vina_score_on):
                filtered_vina_score_on.append(vina_score_on[i])
            if i < len(vina_score_off):
                filtered_vina_score_off.append(vina_score_off[i])
            if i < len(vina_min_on):
                filtered_vina_min_on.append(vina_min_on[i])
            if i < len(vina_min_off):
                filtered_vina_min_off.append(vina_min_off[i])

        on_target_scores = filtered_on
        off_target_scores = filtered_off
        qed_scores = filtered_qed if filtered_qed else qed_scores[:len(filtered_on)]
        sa_scores = filtered_sa if filtered_sa else sa_scores[:len(filtered_on)]
        vina_scores = {'on_target': filtered_vina_score_on, 'off_target': filtered_vina_score_off}
        vina_mins = {'on_target': filtered_vina_min_on, 'off_target': filtered_vina_min_off}

    return on_target_scores, off_target_scores, qed_scores, sa_scores, vina_scores, vina_mins, filtered_count


def calculate_selectivity(on_scores, off_scores):
    """Calculate selectivity for paired scores."""
    if not on_scores or not off_scores:
        return []

    # Pair up scores (assumes same length or use minimum length)
    min_len = min(len(on_scores), len(off_scores))
    selectivities = []

    for i in range(min_len):
        # Selectivity = off_target - on_target (positive = selective for on-target)
        selectivity = off_scores[i] - on_scores[i]
        selectivities.append(selectivity)

    return selectivities


def analyze_pair(on_id, off_id, tmscore, filter_failed_docking=True):
    """Analyze docking results for a specific on-target/off-target pair.

    Args:
        on_id: On-target ID
        off_id: Off-target ID
        tmscore: TM-score value
        filter_failed_docking: If True, exclude molecules with vina score >= 0 (docking failure)
    """
    results_dir = Path('./results')

    pair_results = {
        'on_target_id': on_id,
        'off_target_id': off_id,
        'tmscore': tmscore,
        'models': {},
        'reference_on_target': None,  # Store reference on-target for molecule-level comparison
        'reference_selectivity': None  # Store reference selectivity (off - on) for comparison
    }

    models = {
        # === Reference ligand MUST be processed FIRST to set reference_on_target ===
        'Reference': results_dir / 'Reference_tmscore_evaluation' / f'id{on_id}_low_{off_id}' / 'docking_results.json',
        # === Main comparison models ===
        'TargetDiff': Path('./results/unified_targetdiff_retrain') / f'cd2020_{on_id}_targetdiff_retrain' / f'id{on_id}' / 'docking_results' / f'on{on_id}_off{off_id}' / 'docking_results.json',
        'KGDiff': Path('./results/unified_kgdiff_retrain') / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        'KGDiff2': Path('./results/cd2020_kgdiff_original') / f'cd2020_{on_id}_kgdiff_original' / 'docking_results' / f'on{on_id}_off{off_id}' / 'docking_results.json',
        'BInD': results_dir / 'BInD_tmscore_evaluation' / f'id{on_id}_low_{off_id}' / 'docking_results.json',
        # === NEW: Benchmark models (AR and Pocket2Mol) ===
        'AR': results_dir / 'ar_vina_docked' / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        'Pocket2Mol': results_dir / 'pocket2mol_vina_docked' / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        # === NEW: Dual head models with w2_1_100_25_100_0 (re-docked) ===
        'dual_1p_w2_1': results_dir / 'redock_dual_head2_1p_all_w2_1' / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        'dual_atom_w2_1': results_dir / 'redock_dual_head2_atom_w2_1' / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        # === NEW: Unified model results ===
        'Unified_H1H2_BiNG': Path('./results/unified_h1h2_bi_noguide') / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        'Unified_H1_BiOn': Path('./results/unified_h1_bi_ori') / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        'Unified_H2_BiOff': Path('./results/unified_h2_bi_offguide') / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        # === NEW: Unified model variants ===
        'Unified_ProType': results_dir / 'unified_h1h2_pro_type' / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        'Unified_ProPos': results_dir / 'unified_h1h2_pro_pos' / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        'Unified_LigType': results_dir / 'unified_h1h2_lig_type' / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        'Unified_LigPos': results_dir / 'unified_h1h2_lig_pos' / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        'Unified_BiType': results_dir / 'unified_h1h2_bi_type' / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        'Unified_BiPos': results_dir / 'unified_h1h2_bi_pos' / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        'Unified_BiPos_v2': Path('./results/unified_h1h2_bi_pos_v2') / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        'Unified_BiPos_500': Path('./results/unified_h1h2_bi_pos_500') / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        'Unified_BiPos_500_v2': Path('./results/unified_h1h2_bi_pos_500_v2') / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
        'TheSelective': results_dir / 'theselective' / f'id{on_id}_{off_id}_low' / 'docking_results' / 'docking_results.json',
    }
    for model_name, file_path in models.items():
        results = parse_docking_results_json(file_path)

        if results:
            # Only normalize SA for Reference model
            is_reference = (model_name == 'Reference')
            # Limit TargetDiff, KGDiff, BInD, and Unified models to 8 molecules per pair for fair comparison
            if model_name in ['TargetDiff', 'KGDiff', 'KGDiff2', 'BInD', 'Unified_H1H2_BiNG', 'Unified_H1_BiOn', 'Unified_H2_BiOff', 'Unified_ProType', 'Unified_ProPos', 'Unified_LigType', 'Unified_LigPos', 'Unified_BiType', 'Unified_BiPos', 'Unified_BiPos_500', 'Unified_BiPos_500_v2', 'TheSelective'] or model_name.startswith('head2_atom') or model_name.startswith('head2_1p'):
                max_mols = 8
            else:
                max_mols = None
            on_scores, off_scores, qed_scores, sa_scores, vina_scores, vina_mins, filtered_count = extract_scores_from_results(results, max_molecules=max_mols, is_reference=is_reference, filter_failed_docking=filter_failed_docking)
            selectivities = calculate_selectivity(on_scores, off_scores)

            # Track filtered molecules count (docking failures with score >= 0)
            docking_failures = filtered_count

            # Store reference on-target score and selectivity for later comparison
            if model_name == 'Reference' and on_scores:
                pair_results['reference_on_target'] = on_scores[0] if len(on_scores) > 0 else None
                # Calculate reference selectivity (off - on)
                if len(off_scores) > 0:
                    pair_results['reference_selectivity'] = off_scores[0] - on_scores[0]
                else:
                    pair_results['reference_selectivity'] = None

            # Calculate vina score and min differences (off - on) - positive means better selectivity
            vina_score_diff = None
            vina_min_diff = None
            vina_score_diff_median = None
            vina_min_diff_median = None

            if vina_scores['on_target'] and vina_scores['off_target']:
                vina_score_diff = np.mean(vina_scores['off_target']) - np.mean(vina_scores['on_target'])
                vina_score_diff_median = np.median(vina_scores['off_target']) - np.median(vina_scores['on_target'])

            if vina_mins['on_target'] and vina_mins['off_target']:
                vina_min_diff = np.mean(vina_mins['off_target']) - np.mean(vina_mins['on_target'])
                vina_min_diff_median = np.median(vina_mins['off_target']) - np.median(vina_mins['on_target'])

            # Calculate percentage of ligands with off-target score > -7 (weak binding = good selectivity)
            # NOTE: All percentages are calculated from VALID molecules only (docking success, filtered by score >= 0)
            off_target_weak_binding_pct = None
            if off_scores:
                weak_binding_count = sum(1 for score in off_scores if score > -7.0)
                off_target_weak_binding_pct = (weak_binding_count / len(off_scores)) * 100  # Out of valid molecules

            # Calculate percentage of ligands with on-target < -7 AND off-target > -7 (specific binding)
            specific_ligands_pct = None
            if on_scores and off_scores:
                min_len = min(len(on_scores), len(off_scores))
                specific_count = sum(1 for i in range(min_len) if on_scores[i] < -7.0 and off_scores[i] > -7.0)
                specific_ligands_pct = (specific_count / min_len) * 100  # Out of valid molecules

            # Calculate percentage of ligands with on-target < -7 AND off-target > -6 (stricter selectivity)
            strict_selective_pct = None
            if on_scores and off_scores:
                min_len = min(len(on_scores), len(off_scores))
                strict_count = sum(1 for i in range(min_len) if on_scores[i] < -7.0 and off_scores[i] > -6.0)
                strict_selective_pct = (strict_count / min_len) * 100  # Out of valid molecules

            # Calculate percentage of ligands with selectivity > 1 (off-target - on-target > 1)
            high_selectivity_pct = None
            if on_scores and off_scores:
                min_len = min(len(on_scores), len(off_scores))
                high_sel_count = sum(1 for i in range(min_len) if (off_scores[i] - on_scores[i]) > 1.0)
                high_selectivity_pct = (high_sel_count / min_len) * 100  # Out of valid molecules

            # Calculate percentage of ligands satisfying BOTH: better than reference AND off-target > -6
            both_conditions_pct = None
            if off_scores and on_scores and pair_results['reference_on_target'] is not None:
                ref_score = pair_results['reference_on_target']
                # Count molecules where on<ref (better than reference) AND off>-6 (weak off-target binding)
                both_count = sum(1 for on, off in zip(on_scores, off_scores)
                                if on < ref_score and off > -6.0)
                both_conditions_pct = (both_count / len(on_scores)) * 100  # Out of valid molecules

            # Calculate percentage of ligands where on<ref AND on<off (better than reference AND on better than off)
            ref_and_selective_pct = None
            if off_scores and on_scores and pair_results['reference_on_target'] is not None:
                ref_score = pair_results['reference_on_target']
                # Count molecules where on<ref (better than reference) AND on<off (on is better than off)
                ref_sel_count = sum(1 for on, off in zip(on_scores, off_scores)
                                   if on < ref_score and on < off)
                ref_and_selective_pct = (ref_sel_count / len(on_scores)) * 100  # Out of valid molecules

            # Calculate percentage of ligands with selectivity > reference selectivity
            better_selectivity_pct = None
            if off_scores and on_scores and pair_results['reference_selectivity'] is not None:
                ref_selectivity = pair_results['reference_selectivity']
                # Count molecules where (off - on) > ref_selectivity
                better_sel_count = sum(1 for on, off in zip(on_scores, off_scores)
                                      if (off - on) > ref_selectivity)
                better_selectivity_pct = (better_sel_count / len(on_scores)) * 100  # Out of valid molecules

            # Calculate percentage of ligands with better selectivity AND stronger on-target binding than reference
            better_sel_and_on_pct = None
            if off_scores and on_scores and pair_results['reference_selectivity'] is not None and pair_results['reference_on_target'] is not None:
                ref_selectivity = pair_results['reference_selectivity']
                ref_on_target = pair_results['reference_on_target']
                # Count molecules where (off - on) > ref_selectivity AND on < ref_on_target (lower = stronger binding)
                better_sel_and_on_count = sum(1 for on, off in zip(on_scores, off_scores)
                                              if (off - on) > ref_selectivity and on < ref_on_target)
                better_sel_and_on_pct = (better_sel_and_on_count / len(on_scores)) * 100  # Out of valid molecules

            # Calculate percentage of ligands with selectivity > 1 AND on-target < reference
            sel_gt1_and_on_lt_ref_pct = None
            if off_scores and on_scores and pair_results['reference_on_target'] is not None:
                ref_on_target = pair_results['reference_on_target']
                # Count molecules where (off - on) > 1.0 AND on < ref_on_target (lower = stronger binding)
                sel_gt1_on_lt_ref_count = sum(1 for on, off in zip(on_scores, off_scores)
                                               if (off - on) > 1.0 and on < ref_on_target)
                sel_gt1_and_on_lt_ref_pct = (sel_gt1_on_lt_ref_count / len(on_scores)) * 100  # Out of valid molecules

            # Calculate percentage of ligands where on<ref AND on<off (better than reference AND stronger than off-target)
            on_lt_ref_and_off_pct = None
            if off_scores and on_scores and pair_results['reference_on_target'] is not None:
                ref_on_target = pair_results['reference_on_target']
                # Count molecules where on < ref_on_target AND on < off (stronger binding than both ref and off)
                on_lt_ref_off_count = sum(1 for on, off in zip(on_scores, off_scores)
                                           if on < ref_on_target and on < off)
                on_lt_ref_and_off_pct = (on_lt_ref_off_count / len(on_scores)) * 100  # Out of valid molecules

            # Calculate percentage of ligands where on<ref, on<off AND selectivity>ref_selectivity
            on_lt_ref_off_sel_gt_ref_pct = None
            if off_scores and on_scores and pair_results['reference_on_target'] is not None and pair_results['reference_selectivity'] is not None:
                ref_on_target = pair_results['reference_on_target']
                ref_selectivity = pair_results['reference_selectivity']
                # Count molecules where on < ref_on_target AND on < off AND (off - on) > ref_selectivity
                on_lt_ref_off_sel_count = sum(1 for on, off in zip(on_scores, off_scores)
                                               if on < ref_on_target and on < off and (off - on) > ref_selectivity)
                on_lt_ref_off_sel_gt_ref_pct = (on_lt_ref_off_sel_count / len(on_scores)) * 100  # Out of valid molecules

            # Calculate percentage of ligands with selectivity > 0
            sel_gt0_pct = None
            if off_scores and on_scores:
                sel_gt0_count = sum(1 for on, off in zip(on_scores, off_scores) if (off - on) > 0.0)
                sel_gt0_pct = (sel_gt0_count / len(on_scores)) * 100  # Out of valid molecules

            # Calculate percentage of ligands with selectivity > 1
            sel_gt1_pct = None
            if off_scores and on_scores:
                sel_gt1_count = sum(1 for on, off in zip(on_scores, off_scores) if (off - on) > 1.0)
                sel_gt1_pct = (sel_gt1_count / len(on_scores)) * 100  # Out of valid molecules

            # Calculate percentage of ligands with selectivity > 2
            sel_gt2_pct = None
            if off_scores and on_scores:
                sel_gt2_count = sum(1 for on, off in zip(on_scores, off_scores) if (off - on) > 2.0)
                sel_gt2_pct = (sel_gt2_count / len(on_scores)) * 100  # Out of valid molecules

            # Calculate percentage of ligands with selectivity > 3
            sel_gt3_pct = None
            if off_scores and on_scores:
                sel_gt3_count = sum(1 for on, off in zip(on_scores, off_scores) if (off - on) > 3.0)
                sel_gt3_pct = (sel_gt3_count / len(on_scores)) * 100  # Out of valid molecules

            # Calculate percentage of ligands with selectivity > 4
            sel_gt4_pct = None
            if off_scores and on_scores:
                sel_gt4_count = sum(1 for on, off in zip(on_scores, off_scores) if (off - on) > 4.0)
                sel_gt4_pct = (sel_gt4_count / len(on_scores)) * 100  # Out of valid molecules

            # Set _pct_total same as _pct (all calculated from valid molecules only)
            off_target_weak_binding_pct_total = off_target_weak_binding_pct
            specific_ligands_pct_total = specific_ligands_pct
            strict_selective_pct_total = strict_selective_pct
            high_selectivity_pct_total = high_selectivity_pct
            both_conditions_pct_total = both_conditions_pct
            ref_and_selective_pct_total = ref_and_selective_pct
            better_selectivity_pct_total = better_selectivity_pct
            better_sel_and_on_pct_total = better_sel_and_on_pct
            sel_gt1_and_on_lt_ref_pct_total = sel_gt1_and_on_lt_ref_pct
            on_lt_ref_and_off_pct_total = on_lt_ref_and_off_pct
            on_lt_ref_off_sel_gt_ref_pct_total = on_lt_ref_off_sel_gt_ref_pct
            sel_gt0_pct_total = sel_gt0_pct
            sel_gt1_pct_total = sel_gt1_pct
            sel_gt2_pct_total = sel_gt2_pct
            sel_gt3_pct_total = sel_gt3_pct
            sel_gt4_pct_total = sel_gt4_pct

            pair_results['models'][model_name] = {
                'on_target_mean': np.mean(on_scores) if on_scores else None,
                'on_target_median': np.median(on_scores) if on_scores else None,
                'off_target_mean': np.mean(off_scores) if off_scores else None,
                'off_target_median': np.median(off_scores) if off_scores else None,
                'selectivity_mean': np.mean(selectivities) if selectivities else None,
                'selectivity_median': np.median(selectivities) if selectivities else None,
                'selectivity_std': np.std(selectivities) if selectivities else None,
                'qed_mean': np.mean(qed_scores) if qed_scores else None,
                'qed_median': np.median(qed_scores) if qed_scores else None,
                'sa_mean': np.mean(sa_scores) if sa_scores else None,
                'sa_median': np.median(sa_scores) if sa_scores else None,
                'vina_score_on_mean': np.mean(vina_scores['on_target']) if vina_scores['on_target'] else None,
                'vina_score_on_median': np.median(vina_scores['on_target']) if vina_scores['on_target'] else None,
                'vina_score_off_mean': np.mean(vina_scores['off_target']) if vina_scores['off_target'] else None,
                'vina_score_off_median': np.median(vina_scores['off_target']) if vina_scores['off_target'] else None,
                'vina_score_diff': vina_score_diff,
                'vina_score_diff_median': vina_score_diff_median,
                'vina_min_on_mean': np.mean(vina_mins['on_target']) if vina_mins['on_target'] else None,
                'vina_min_on_median': np.median(vina_mins['on_target']) if vina_mins['on_target'] else None,
                'vina_min_off_mean': np.mean(vina_mins['off_target']) if vina_mins['off_target'] else None,
                'vina_min_off_median': np.median(vina_mins['off_target']) if vina_mins['off_target'] else None,
                'vina_min_diff': vina_min_diff,
                'vina_min_diff_median': vina_min_diff_median,
                'off_target_weak_pct': off_target_weak_binding_pct,
                'off_target_weak_pct_total': off_target_weak_binding_pct_total,
                'specific_ligands_pct': specific_ligands_pct,
                'specific_ligands_pct_total': specific_ligands_pct_total,
                'strict_selective_pct': strict_selective_pct,
                'strict_selective_pct_total': strict_selective_pct_total,
                'high_selectivity_pct': high_selectivity_pct,
                'high_selectivity_pct_total': high_selectivity_pct_total,
                'both_conditions_pct': both_conditions_pct,
                'both_conditions_pct_total': both_conditions_pct_total,
                'ref_and_selective_pct': ref_and_selective_pct,
                'ref_and_selective_pct_total': ref_and_selective_pct_total,
                'better_selectivity_pct': better_selectivity_pct,
                'better_selectivity_pct_total': better_selectivity_pct_total,
                'better_sel_and_on_pct': better_sel_and_on_pct,
                'better_sel_and_on_pct_total': better_sel_and_on_pct_total,
                'sel_gt1_and_on_lt_ref_pct': sel_gt1_and_on_lt_ref_pct,
                'sel_gt1_and_on_lt_ref_pct_total': sel_gt1_and_on_lt_ref_pct_total,
                'on_lt_ref_and_off_pct': on_lt_ref_and_off_pct,
                'on_lt_ref_and_off_pct_total': on_lt_ref_and_off_pct_total,
                'on_lt_ref_off_sel_gt_ref_pct': on_lt_ref_off_sel_gt_ref_pct,
                'on_lt_ref_off_sel_gt_ref_pct_total': on_lt_ref_off_sel_gt_ref_pct_total,
                'sel_gt0_pct': sel_gt0_pct,
                'sel_gt0_pct_total': sel_gt0_pct_total,
                'sel_gt1_pct': sel_gt1_pct,
                'sel_gt1_pct_total': sel_gt1_pct_total,
                'sel_gt2_pct': sel_gt2_pct,
                'sel_gt2_pct_total': sel_gt2_pct_total,
                'sel_gt3_pct': sel_gt3_pct,
                'sel_gt3_pct_total': sel_gt3_pct_total,
                'sel_gt4_pct': sel_gt4_pct,
                'sel_gt4_pct_total': sel_gt4_pct_total,
                'num_molecules': len(on_scores),
                'docking_failures': docking_failures,  # Number of molecules filtered out due to score >= 0
                'on_target_scores': on_scores,  # Store individual scores for molecule-level comparison
                'off_target_scores': off_scores,  # Store off-target scores for both conditions check
                'file_path': str(file_path)
            }
        else:
            pair_results['models'][model_name] = {
                'error': f'File not found or parsing failed: {file_path}'
            }

    return pair_results


def aggregate_results(all_pairs_results):
    """Aggregate results across all pairs."""
    aggregated = defaultdict(lambda: defaultdict(list))
    total_molecules = defaultdict(int)  # Track total molecules per model
    better_than_reference = defaultdict(lambda: {'count': 0, 'total': 0})  # Track pairs better than Reference
    better_than_reference_total = defaultdict(int)  # Track total molecules better than Reference (out of 400)

    for pair_result in all_pairs_results:
        # Get Reference on-target affinity for this pair (individual molecule score)
        reference_on_target = pair_result.get('reference_on_target')

        for model_name, model_data in pair_result['models'].items():
            if model_data and 'error' not in model_data:
                # Track total molecules
                num_mols = model_data.get('num_molecules', 0)
                if num_mols > 0:
                    total_molecules[model_name] += num_mols

                # Compare with Reference at pair level (using mean)
                if model_name != 'Reference' and model_data.get('on_target_mean') is not None:
                    ref_mean = None
                    if 'Reference' in pair_result['models']:
                        ref_data = pair_result['models']['Reference']
                        if ref_data and 'error' not in ref_data:
                            ref_mean = ref_data.get('on_target_mean')

                    if ref_mean is not None:
                        model_on_target = model_data.get('on_target_mean')
                        better_than_reference[model_name]['total'] += 1
                        if model_on_target < ref_mean:  # Lower affinity is better
                            better_than_reference[model_name]['count'] += 1

                # Compare with Reference at molecule level (for total 400 calculation)
                if model_name != 'Reference' and reference_on_target is not None:
                    on_scores = model_data.get('on_target_scores', [])
                    off_scores = model_data.get('off_target_scores', [])
                    for i, on_score in enumerate(on_scores):
                        if on_score < reference_on_target:  # Lower affinity is better
                            better_than_reference_total[model_name] += 1
                            # Also check if this molecule satisfies both conditions (>Ref AND off>-6)
                            if i < len(off_scores) and off_scores[i] > -6.0:
                                better_than_reference_total[model_name + '_both'] += 1

                for metric in ['on_target_mean', 'on_target_median',
                              'off_target_mean', 'off_target_median',
                              'selectivity_mean', 'selectivity_median', 'selectivity_std',
                              'qed_mean', 'qed_median', 'sa_mean', 'sa_median',
                              'vina_score_on_mean', 'vina_score_on_median',
                              'vina_score_off_mean', 'vina_score_off_median', 'vina_score_diff', 'vina_score_diff_median',
                              'vina_min_on_mean', 'vina_min_on_median',
                              'vina_min_off_mean', 'vina_min_off_median', 'vina_min_diff', 'vina_min_diff_median',
                              'strict_selective_pct', 'strict_selective_pct_total',
                              'both_conditions_pct', 'both_conditions_pct_total',
                              'ref_and_selective_pct', 'ref_and_selective_pct_total',
                              'better_selectivity_pct', 'better_selectivity_pct_total',
                              'better_sel_and_on_pct', 'better_sel_and_on_pct_total']:
                    if model_data.get(metric) is not None:
                        aggregated[model_name][metric].append(model_data[metric])

    # Calculate overall statistics
    final_stats = {}

    # Track total counts for percentage calculations
    total_strict_selective_count = defaultdict(int)
    total_both_conditions_count = defaultdict(int)
    total_ref_and_selective_count = defaultdict(int)
    total_better_selectivity_count = defaultdict(int)
    total_better_sel_and_on_count = defaultdict(int)
    total_sel_gt1_and_on_lt_ref_count = defaultdict(int)
    total_on_lt_ref_and_off_count = defaultdict(int)
    total_on_lt_ref_off_sel_gt_ref_count = defaultdict(int)
    total_sel_gt0_count = defaultdict(int)
    total_sel_gt1_count = defaultdict(int)
    total_sel_gt2_count = defaultdict(int)
    total_sel_gt3_count = defaultdict(int)
    total_sel_gt4_count = defaultdict(int)
    total_pairs_count = len(all_pairs_results)

    # Count total molecules across all pairs
    for pair_result in all_pairs_results:
        for model_name in pair_result['models']:
            model_data = pair_result['models'][model_name]
            if 'error' not in model_data:
                num_mols = model_data.get('num_molecules', 0)

                # Count strict selective ligands (on < -7 AND off > -6)
                strict_pct = model_data.get('strict_selective_pct')
                if strict_pct is not None and num_mols > 0:
                    strict_count = int(round((strict_pct / 100.0) * num_mols))
                    total_strict_selective_count[model_name] += strict_count

                # Count both conditions molecules (on>ref AND off>-6)
                both_pct = model_data.get('both_conditions_pct')
                if both_pct is not None and num_mols > 0:
                    both_count = int(round((both_pct / 100.0) * num_mols))
                    total_both_conditions_count[model_name] += both_count

                # Count ref_and_selective molecules (on>ref AND on<off)
                ref_sel_pct = model_data.get('ref_and_selective_pct')
                if ref_sel_pct is not None and num_mols > 0:
                    ref_sel_count = int(round((ref_sel_pct / 100.0) * num_mols))
                    total_ref_and_selective_count[model_name] += ref_sel_count

                # Count better_selectivity molecules (selectivity > ref_selectivity)
                better_sel_pct = model_data.get('better_selectivity_pct')
                if better_sel_pct is not None and num_mols > 0:
                    better_sel_count = int(round((better_sel_pct / 100.0) * num_mols))
                    total_better_selectivity_count[model_name] += better_sel_count

                # Count better_sel_and_on molecules (selectivity > ref AND on < ref_on)
                better_sel_and_on_pct = model_data.get('better_sel_and_on_pct')
                if better_sel_and_on_pct is not None and num_mols > 0:
                    better_sel_and_on_count = int(round((better_sel_and_on_pct / 100.0) * num_mols))
                    total_better_sel_and_on_count[model_name] += better_sel_and_on_count

                # Count sel_gt1_and_on_lt_ref molecules (selectivity > 1 AND on < ref_on)
                sel_gt1_pct = model_data.get('sel_gt1_and_on_lt_ref_pct')
                if sel_gt1_pct is not None and num_mols > 0:
                    sel_gt1_count = int(round((sel_gt1_pct / 100.0) * num_mols))
                    total_sel_gt1_and_on_lt_ref_count[model_name] += sel_gt1_count

                # Count on_lt_ref_and_off molecules (on < ref AND on < off)
                on_lt_ref_off_pct = model_data.get('on_lt_ref_and_off_pct')
                if on_lt_ref_off_pct is not None and num_mols > 0:
                    on_lt_ref_off_count = int(round((on_lt_ref_off_pct / 100.0) * num_mols))
                    total_on_lt_ref_and_off_count[model_name] += on_lt_ref_off_count

                # Count on_lt_ref_off_sel_gt_ref molecules (on < ref AND on < off AND sel > ref_sel)
                on_lt_ref_off_sel_pct = model_data.get('on_lt_ref_off_sel_gt_ref_pct')
                if on_lt_ref_off_sel_pct is not None and num_mols > 0:
                    on_lt_ref_off_sel_count = int(round((on_lt_ref_off_sel_pct / 100.0) * num_mols))
                    total_on_lt_ref_off_sel_gt_ref_count[model_name] += on_lt_ref_off_sel_count

                # Count sel_gt0 molecules (selectivity > 0)
                sel_gt0_pct = model_data.get('sel_gt0_pct')
                if sel_gt0_pct is not None and num_mols > 0:
                    sel_gt0_count = int(round((sel_gt0_pct / 100.0) * num_mols))
                    total_sel_gt0_count[model_name] += sel_gt0_count

                # Count sel_gt1 molecules (selectivity > 1)
                sel_gt1_pct = model_data.get('sel_gt1_pct')
                if sel_gt1_pct is not None and num_mols > 0:
                    sel_gt1_count = int(round((sel_gt1_pct / 100.0) * num_mols))
                    total_sel_gt1_count[model_name] += sel_gt1_count

                # Count sel_gt2 molecules (selectivity > 2)
                sel_gt2_pct = model_data.get('sel_gt2_pct')
                if sel_gt2_pct is not None and num_mols > 0:
                    sel_gt2_count = int(round((sel_gt2_pct / 100.0) * num_mols))
                    total_sel_gt2_count[model_name] += sel_gt2_count

                # Count sel_gt3 molecules (selectivity > 3)
                sel_gt3_pct = model_data.get('sel_gt3_pct')
                if sel_gt3_pct is not None and num_mols > 0:
                    sel_gt3_count = int(round((sel_gt3_pct / 100.0) * num_mols))
                    total_sel_gt3_count[model_name] += sel_gt3_count

                # Count sel_gt4 molecules (selectivity > 4)
                sel_gt4_pct = model_data.get('sel_gt4_pct')
                if sel_gt4_pct is not None and num_mols > 0:
                    sel_gt4_count = int(round((sel_gt4_pct / 100.0) * num_mols))
                    total_sel_gt4_count[model_name] += sel_gt4_count

    for model_name, metrics in aggregated.items():
        final_stats[model_name] = {}
        for metric_name, values in metrics.items():
            if values:
                # Skip _pct_total metrics as we'll calculate them differently
                if metric_name in ['strict_selective_pct_total', 'both_conditions_pct_total', 'ref_and_selective_pct_total', 'better_selectivity_pct_total', 'better_sel_and_on_pct_total', 'sel_gt1_and_on_lt_ref_pct_total', 'on_lt_ref_and_off_pct_total', 'on_lt_ref_off_sel_gt_ref_pct_total', 'sel_gt0_pct_total', 'sel_gt1_pct_total', 'sel_gt2_pct_total', 'sel_gt3_pct_total', 'sel_gt4_pct_total']:
                    continue
                final_stats[model_name][f'{metric_name}_overall_mean'] = np.mean(values)
                final_stats[model_name][f'{metric_name}_overall_std'] = np.std(values)
                # For key metrics, also calculate median of per-pair means
                if metric_name in ['selectivity_mean', 'on_target_mean', 'off_target_mean', 'qed_mean', 'sa_mean',
                                   'vina_score_on_mean', 'vina_score_off_mean', 'vina_min_on_mean', 'vina_min_off_mean',
                                   'vina_score_diff', 'vina_score_diff_median', 'vina_min_diff', 'vina_min_diff_median']:
                    final_stats[model_name][f'{metric_name}_overall_median'] = np.median(values)
                final_stats[model_name][f'num_pairs'] = len(values)

        # Add total molecules count
        final_stats[model_name]['total_molecules'] = total_molecules.get(model_name, 0)

        # Get total molecules for this model for percentage calculations
        model_total_mols = total_molecules.get(model_name, 0)

        # Calculate strict_selective_pct_total based on total strict selective count / total_molecules
        if model_name in total_strict_selective_count and model_total_mols > 0:
            final_stats[model_name]['strict_selective_pct_total_overall_mean'] = (total_strict_selective_count[model_name] / model_total_mols) * 100
        else:
            final_stats[model_name]['strict_selective_pct_total_overall_mean'] = None

        # Calculate both_conditions_pct_total based on total both conditions count / total_molecules
        if model_name in total_both_conditions_count and model_total_mols > 0:
            final_stats[model_name]['both_conditions_pct_total_overall_mean'] = (total_both_conditions_count[model_name] / model_total_mols) * 100
        else:
            final_stats[model_name]['both_conditions_pct_total_overall_mean'] = None

        # Calculate ref_and_selective_pct_total based on total ref_and_selective count / total_molecules
        if model_name in total_ref_and_selective_count and model_total_mols > 0:
            final_stats[model_name]['ref_and_selective_pct_total_overall_mean'] = (total_ref_and_selective_count[model_name] / model_total_mols) * 100
        else:
            final_stats[model_name]['ref_and_selective_pct_total_overall_mean'] = None

        # Calculate better_selectivity_pct_total based on total better_selectivity count / total_molecules
        if model_name in total_better_selectivity_count and model_total_mols > 0:
            final_stats[model_name]['better_selectivity_pct_total_overall_mean'] = (total_better_selectivity_count[model_name] / model_total_mols) * 100
        else:
            final_stats[model_name]['better_selectivity_pct_total_overall_mean'] = None

        # Calculate better_sel_and_on_pct_total based on total better_sel_and_on count / total_molecules
        if model_name in total_better_sel_and_on_count and model_total_mols > 0:
            final_stats[model_name]['better_sel_and_on_pct_total_overall_mean'] = (total_better_sel_and_on_count[model_name] / model_total_mols) * 100
        else:
            final_stats[model_name]['better_sel_and_on_pct_total_overall_mean'] = None

        # Calculate sel_gt1_and_on_lt_ref_pct_total based on total sel_gt1 count / total_molecules
        if model_name in total_sel_gt1_and_on_lt_ref_count and model_total_mols > 0:
            final_stats[model_name]['sel_gt1_and_on_lt_ref_pct_total_overall_mean'] = (total_sel_gt1_and_on_lt_ref_count[model_name] / model_total_mols) * 100
        else:
            final_stats[model_name]['sel_gt1_and_on_lt_ref_pct_total_overall_mean'] = None

        # Calculate on_lt_ref_and_off_pct_total based on total on<ref&off count / total_molecules
        if model_name in total_on_lt_ref_and_off_count and model_total_mols > 0:
            final_stats[model_name]['on_lt_ref_and_off_pct_total_overall_mean'] = (total_on_lt_ref_and_off_count[model_name] / model_total_mols) * 100
        else:
            final_stats[model_name]['on_lt_ref_and_off_pct_total_overall_mean'] = None

        # Calculate on_lt_ref_off_sel_gt_ref_pct_total based on total on<ref&off&sel>ref count / total_molecules
        if model_name in total_on_lt_ref_off_sel_gt_ref_count and model_total_mols > 0:
            final_stats[model_name]['on_lt_ref_off_sel_gt_ref_pct_total_overall_mean'] = (total_on_lt_ref_off_sel_gt_ref_count[model_name] / model_total_mols) * 100
        else:
            final_stats[model_name]['on_lt_ref_off_sel_gt_ref_pct_total_overall_mean'] = None

        # Calculate sel_gt0_pct_total based on total sel>0 count / total_molecules
        if model_name in total_sel_gt0_count and model_total_mols > 0:
            final_stats[model_name]['sel_gt0_pct_total_overall_mean'] = (total_sel_gt0_count[model_name] / model_total_mols) * 100
        else:
            final_stats[model_name]['sel_gt0_pct_total_overall_mean'] = None

        # Calculate sel_gt1_pct_total based on total sel>1 count / total_molecules
        if model_name in total_sel_gt1_count and model_total_mols > 0:
            final_stats[model_name]['sel_gt1_pct_total_overall_mean'] = (total_sel_gt1_count[model_name] / model_total_mols) * 100
        else:
            final_stats[model_name]['sel_gt1_pct_total_overall_mean'] = None

        # Calculate sel_gt2_pct_total based on total sel>2 count / total_molecules
        if model_name in total_sel_gt2_count and model_total_mols > 0:
            final_stats[model_name]['sel_gt2_pct_total_overall_mean'] = (total_sel_gt2_count[model_name] / model_total_mols) * 100
        else:
            final_stats[model_name]['sel_gt2_pct_total_overall_mean'] = None

        # Calculate sel_gt3_pct_total based on total sel>3 count / total_molecules
        if model_name in total_sel_gt3_count and model_total_mols > 0:
            final_stats[model_name]['sel_gt3_pct_total_overall_mean'] = (total_sel_gt3_count[model_name] / model_total_mols) * 100
        else:
            final_stats[model_name]['sel_gt3_pct_total_overall_mean'] = None

        # Calculate sel_gt4_pct_total based on total sel>4 count / total_molecules
        if model_name in total_sel_gt4_count and model_total_mols > 0:
            final_stats[model_name]['sel_gt4_pct_total_overall_mean'] = (total_sel_gt4_count[model_name] / model_total_mols) * 100
        else:
            final_stats[model_name]['sel_gt4_pct_total_overall_mean'] = None

        # Add better than Reference percentage (pair-level)
        if model_name in better_than_reference:
            ref_stats = better_than_reference[model_name]
            if ref_stats['total'] > 0:
                final_stats[model_name]['better_than_ref_pct'] = (ref_stats['count'] / ref_stats['total']) * 100
            else:
                final_stats[model_name]['better_than_ref_pct'] = None
        else:
            final_stats[model_name]['better_than_ref_pct'] = None

        # Add better than Reference percentage (molecule-level, out of total_molecules)
        if model_name in better_than_reference_total and model_total_mols > 0:
            final_stats[model_name]['better_than_ref_pct_total'] = (better_than_reference_total[model_name] / model_total_mols) * 100
        else:
            final_stats[model_name]['better_than_ref_pct_total'] = None

        # Add both conditions percentage (>Ref AND Off>-7, out of total_molecules)
        both_key = model_name + '_both'
        if both_key in better_than_reference_total and model_total_mols > 0:
            final_stats[model_name]['both_conditions_pct_total_overall_mean'] = (better_than_reference_total[both_key] / model_total_mols) * 100
        else:
            final_stats[model_name]['both_conditions_pct_total_overall_mean'] = None

    return final_stats


def print_detailed_report(all_pairs_results, final_stats, filter_failed_docking=True):
    """Print detailed report."""
    print("=" * 160)
    if filter_failed_docking:
        print("DETAILED DOCKING RESULTS ANALYSIS FOR LOW TM-SCORE PAIRS (FILTERED)")
        print("=" * 160)
        print("\n*** DOCKING FAILURE FILTER APPLIED: Molecules with on-target OR off-target score >= 0 are EXCLUDED ***\n")
    else:
        print("DETAILED DOCKING RESULTS ANALYSIS FOR LOW TM-SCORE PAIRS (UNFILTERED)")
        print("=" * 160)
        print("\n*** NO DOCKING FAILURE FILTER: All molecules including docking failures (score >= 0) are INCLUDED ***\n")
    print(f"Total pairs analyzed: {len(all_pairs_results)}")
    print("\nLOW TM-score pairs from tmscore_extreme_pairs.txt:")
    for pair_result in all_pairs_results:
        print(f"  ({pair_result['on_target_id']}, {pair_result['off_target_id']}) - TM-score: {pair_result['tmscore']:.6f}")
    print("\n")

    # Per-pair detailed table
    print("=" * 160)
    print("PER-PAIR RESULTS (LOW TM-SCORE)")
    print("=" * 160)

    for pair_result in all_pairs_results:
        on_id = pair_result['on_target_id']
        off_id = pair_result['off_target_id']
        tmscore = pair_result['tmscore']

        print(f"\n### Pair ({on_id}, {off_id}) - TM-score: {tmscore:.6f} ###")
        print(f"{'Model':<25} {'On-Tgt(Mean)':<14} {'On-Tgt(Med)':<14} {'Off-Tgt(Mean)':<14} {'Off-Tgt(Med)':<14} {'Select(Mean)':<14} {'Select(Med)':<14} {'QED(Mean)':<12} {'QED(Med)':<12} {'SA(Mean)':<11} {'SA(Med)':<11} {'#Mols':<8}")
        print("-" * 200)

        for model_name in ['Reference', 'TargetDiff', 'KGDiff', 'KGDiff2', 'BInD', 'Unified_BiType', 'Unified_BiPos', 'Unified_BiPos_500', 'Unified_BiPos_500_v2', 'Unified_H1H2_BiNG', 'Unified_H1_BiOn', 'Unified_H2_BiOff', 'Unified_ProType', 'Unified_ProPos', 'Unified_LigType', 'Unified_LigPos', 'TheSelective']:
            model_data = pair_result['models'].get(model_name, {})

            if 'error' in model_data:
                print(f"{model_name:<25} NOT FOUND")
            elif model_data:
                on_mean_str = f"{model_data.get('on_target_mean', 0):.3f}" if model_data.get('on_target_mean') else 'N/A'
                on_med_str = f"{model_data.get('on_target_median', 0):.3f}" if model_data.get('on_target_median') else 'N/A'
                off_mean_str = f"{model_data.get('off_target_mean', 0):.3f}" if model_data.get('off_target_mean') else 'N/A'
                off_med_str = f"{model_data.get('off_target_median', 0):.3f}" if model_data.get('off_target_median') else 'N/A'
                sel_mean_str = f"{model_data.get('selectivity_mean', 0):.3f}" if model_data.get('selectivity_mean') else 'N/A'
                sel_med_str = f"{model_data.get('selectivity_median', 0):.3f}" if model_data.get('selectivity_median') else 'N/A'
                qed_mean_str = f"{model_data.get('qed_mean', 0):.3f}" if model_data.get('qed_mean') else 'N/A'
                qed_med_str = f"{model_data.get('qed_median', 0):.3f}" if model_data.get('qed_median') else 'N/A'
                sa_mean_str = f"{model_data.get('sa_mean', 0):.3f}" if model_data.get('sa_mean') else 'N/A'
                sa_med_str = f"{model_data.get('sa_median', 0):.3f}" if model_data.get('sa_median') else 'N/A'
                num_mols = model_data.get('num_molecules', 0)

                print(f"{model_name:<25} {on_mean_str:<14} {on_med_str:<14} {off_mean_str:<14} {off_med_str:<14} {sel_mean_str:<14} {sel_med_str:<14} {qed_mean_str:<12} {qed_med_str:<12} {sa_mean_str:<11} {sa_med_str:<11} {num_mols:<8}")

    # Aggregated results - Table 1: Basic metrics
    print("\n" + "=" * 200)
    print("AGGREGATED RESULTS - TABLE 1: BASIC METRICS (AVERAGE ACROSS ALL LOW TM-SCORE PAIRS)")
    print("=" * 200)
    print()
    print(f"{'Model':<25} {'On-Tgt(Mean)':<14} {'On-Tgt(Med)':<14} {'Off-Tgt(Mean)':<14} {'Off-Tgt(Med)':<14} {'Select(Mean)':<14} {'Select(Med)':<14} {'QED(Mean)':<12} {'QED(Med)':<12} {'SA(Mean)':<11} {'SA(Med)':<11} {'Success%':<12}")
    print("-" * 200)

    for model_name in ['Reference', 'TargetDiff', 'KGDiff', 'KGDiff2', 'BInD', 'Unified_BiType', 'Unified_BiPos', 'Unified_BiPos_500', 'Unified_BiPos_500_v2', 'Unified_H1H2_BiNG', 'Unified_H1_BiOn', 'Unified_H2_BiOff', 'Unified_ProType', 'Unified_ProPos', 'Unified_LigType', 'Unified_LigPos', 'TheSelective']:
        if model_name in final_stats:
            stats = final_stats[model_name]

            on_mean = stats.get('on_target_mean_overall_mean', None)
            off_mean = stats.get('off_target_mean_overall_mean', None)
            sel_mean = stats.get('selectivity_mean_overall_mean', None)
            qed_mean = stats.get('qed_mean_overall_mean', None)
            sa_mean = stats.get('sa_mean_overall_mean', None)
            total_mols = stats.get('total_molecules', 0)

            # Calculate success rate: Reference /100*100%, others /800*100%
            if model_name == 'Reference':
                success_rate = (total_mols / 100) * 100
            else:
                success_rate = (total_mols / 800) * 100

            # Reference: median of all values (since 1 ligand per pair)
            # Other models: mean of per-pair medians
            if model_name == 'Reference':
                on_median = stats.get('on_target_mean_overall_median', None)
                off_median = stats.get('off_target_mean_overall_median', None)
                sel_median = stats.get('selectivity_mean_overall_median', None)
                qed_median = stats.get('qed_mean_overall_median', None)
                sa_median = stats.get('sa_mean_overall_median', None)
            else:
                on_median = stats.get('on_target_median_overall_mean', None)
                off_median = stats.get('off_target_median_overall_mean', None)
                sel_median = stats.get('selectivity_median_overall_mean', None)
                qed_median = stats.get('qed_median_overall_mean', None)
                sa_median = stats.get('sa_median_overall_mean', None)

            print(f"{model_name:<25} "
                  f"{f'{on_mean:.3f}' if on_mean else 'N/A':<14} "
                  f"{f'{on_median:.3f}' if on_median else 'N/A':<14} "
                  f"{f'{off_mean:.3f}' if off_mean else 'N/A':<14} "
                  f"{f'{off_median:.3f}' if off_median else 'N/A':<14} "
                  f"{f'{sel_mean:.3f}' if sel_mean else 'N/A':<14} "
                  f"{f'{sel_median:.3f}' if sel_median else 'N/A':<14} "
                  f"{f'{qed_mean:.3f}' if qed_mean else 'N/A':<12} "
                  f"{f'{qed_median:.3f}' if qed_median else 'N/A':<12} "
                  f"{f'{sa_mean:.3f}' if sa_mean else 'N/A':<11} "
                  f"{f'{sa_median:.3f}' if sa_median else 'N/A':<11} "
                  f"{f'{success_rate:.1f}%':<12}")

    # ========== VINA SCORE/MIN COMPARISON TABLE ==========
    print("\n" + "=" * 250)
    print("VINA SCORE & MINIMIZE COMPARISON (LOW TM-SCORE PAIRS)")
    print("=" * 250)
    print()
    print(f"{'Model':<25} {'ScOn(Mean)':<12} {'ScOn(Med)':<12} {'ScOff(Mean)':<12} {'ScOff(Med)':<12} {'ScDf(Mean)':<12} {'ScDf(Med)':<12} {'MinOn(Mean)':<12} {'MinOn(Med)':<12} {'MinOff(Mean)':<13} {'MinOff(Med)':<12} {'MinDf(Mean)':<12} {'MinDf(Med)':<12}")
    print("-" * 250)

    for model_name in ['Reference', 'TargetDiff', 'KGDiff', 'KGDiff2', 'BInD', 'Unified_BiType', 'Unified_BiPos', 'Unified_BiPos_500', 'Unified_BiPos_500_v2', 'Unified_H1H2_BiNG', 'Unified_H1_BiOn', 'Unified_H2_BiOff', 'Unified_ProType', 'Unified_ProPos', 'Unified_LigType', 'Unified_LigPos', 'TheSelective']:
        if model_name in final_stats:
            stats = final_stats[model_name]

            vina_score_on_mean = stats.get('vina_score_on_mean_overall_mean', None)
            vina_score_off_mean = stats.get('vina_score_off_mean_overall_mean', None)
            vina_score_diff = stats.get('vina_score_diff_overall_mean', None)
            vina_min_on_mean = stats.get('vina_min_on_mean_overall_mean', None)
            vina_min_off_mean = stats.get('vina_min_off_mean_overall_mean', None)
            vina_min_diff = stats.get('vina_min_diff_overall_mean', None)

            # Reference: median of all values (since 1 ligand per pair)
            # Other models: mean of per-pair medians
            if model_name == 'Reference':
                vina_score_on_median = stats.get('vina_score_on_mean_overall_median', None)
                vina_score_off_median = stats.get('vina_score_off_mean_overall_median', None)
                vina_score_diff_median = stats.get('vina_score_diff_overall_median', None)
                vina_min_on_median = stats.get('vina_min_on_mean_overall_median', None)
                vina_min_off_median = stats.get('vina_min_off_mean_overall_median', None)
                vina_min_diff_median = stats.get('vina_min_diff_overall_median', None)
            else:
                vina_score_on_median = stats.get('vina_score_on_median_overall_mean', None)
                vina_score_off_median = stats.get('vina_score_off_median_overall_mean', None)
                vina_score_diff_median = stats.get('vina_score_diff_median_overall_mean', None)
                vina_min_on_median = stats.get('vina_min_on_median_overall_mean', None)
                vina_min_off_median = stats.get('vina_min_off_median_overall_mean', None)
                vina_min_diff_median = stats.get('vina_min_diff_median_overall_mean', None)

            print(f"{model_name:<25} "
                  f"{f'{vina_score_on_mean:.3f}' if vina_score_on_mean else 'N/A':<12} "
                  f"{f'{vina_score_on_median:.3f}' if vina_score_on_median else 'N/A':<12} "
                  f"{f'{vina_score_off_mean:.3f}' if vina_score_off_mean else 'N/A':<12} "
                  f"{f'{vina_score_off_median:.3f}' if vina_score_off_median else 'N/A':<12} "
                  f"{f'{vina_score_diff:.3f}' if vina_score_diff is not None else 'N/A':<12} "
                  f"{f'{vina_score_diff_median:.3f}' if vina_score_diff_median is not None else 'N/A':<12} "
                  f"{f'{vina_min_on_mean:.3f}' if vina_min_on_mean else 'N/A':<12} "
                  f"{f'{vina_min_on_median:.3f}' if vina_min_on_median else 'N/A':<12} "
                  f"{f'{vina_min_off_mean:.3f}' if vina_min_off_mean else 'N/A':<13} "
                  f"{f'{vina_min_off_median:.3f}' if vina_min_off_median else 'N/A':<12} "
                  f"{f'{vina_min_diff:.3f}' if vina_min_diff is not None else 'N/A':<12} "
                  f"{f'{vina_min_diff_median:.3f}' if vina_min_diff_median is not None else 'N/A':<12}")

    # Aggregated results - Table 2: Selectivity metrics
    print("\n" + "=" * 200)
    print("AGGREGATED RESULTS - TABLE 2: SELECTIVITY METRICS (% OUT OF 400 MOLECULES)")
    print("=" * 200)
    print()
    print(f"{'Model':<25} {'>Ref/400':<12} {'On>Ref&On<Off':<16} {'Sel>Ref':<12} {'Sel>Ref&On<Ref':<16} {'sel>1&On<Ref':<16} {'On<Ref&Off':<14} {'On<Rf,Of&Sel>Rf':<18} {'Sel>0':<12} {'Sel>1':<12} {'Sel>2':<12} {'Sel>3':<12} {'Sel>4':<12}")
    print("-" * 200)

    for model_name in ['Reference', 'TargetDiff', 'KGDiff', 'KGDiff2', 'BInD', 'Unified_BiType', 'Unified_BiPos', 'Unified_BiPos_500', 'Unified_BiPos_500_v2', 'Unified_H1H2_BiNG', 'Unified_H1_BiOn', 'Unified_H2_BiOff', 'Unified_ProType', 'Unified_ProPos', 'Unified_LigType', 'Unified_LigPos', 'TheSelective']:
        if model_name in final_stats:
            stats = final_stats[model_name]

            better_ref_pct_total = stats.get('better_than_ref_pct_total', None)
            ref_and_selective_total = stats.get('ref_and_selective_pct_total_overall_mean', None)
            better_selectivity_total = stats.get('better_selectivity_pct_total_overall_mean', None)
            better_sel_and_on_total = stats.get('better_sel_and_on_pct_total_overall_mean', None)
            sel_gt1_and_on_lt_ref_total = stats.get('sel_gt1_and_on_lt_ref_pct_total_overall_mean', None)
            on_lt_ref_and_off_total = stats.get('on_lt_ref_and_off_pct_total_overall_mean', None)
            on_lt_ref_off_sel_gt_ref_total = stats.get('on_lt_ref_off_sel_gt_ref_pct_total_overall_mean', None)
            sel_gt0_total = stats.get('sel_gt0_pct_total_overall_mean', None)
            sel_gt1_total = stats.get('sel_gt1_pct_total_overall_mean', None)
            sel_gt2_total = stats.get('sel_gt2_pct_total_overall_mean', None)
            sel_gt3_total = stats.get('sel_gt3_pct_total_overall_mean', None)
            sel_gt4_total = stats.get('sel_gt4_pct_total_overall_mean', None)

            print(f"{model_name:<25} "
                  f"{f'{better_ref_pct_total:.1f}%' if better_ref_pct_total is not None else 'N/A':<12} "
                  f"{f'{ref_and_selective_total:.1f}%' if ref_and_selective_total is not None else 'N/A':<16} "
                  f"{f'{better_selectivity_total:.1f}%' if better_selectivity_total is not None else 'N/A':<12} "
                  f"{f'{better_sel_and_on_total:.1f}%' if better_sel_and_on_total is not None else 'N/A':<16} "
                  f"{f'{sel_gt1_and_on_lt_ref_total:.1f}%' if sel_gt1_and_on_lt_ref_total is not None else 'N/A':<16} "
                  f"{f'{on_lt_ref_and_off_total:.1f}%' if on_lt_ref_and_off_total is not None else 'N/A':<14} "
                  f"{f'{on_lt_ref_off_sel_gt_ref_total:.1f}%' if on_lt_ref_off_sel_gt_ref_total is not None else 'N/A':<18} "
                  f"{f'{sel_gt0_total:.1f}%' if sel_gt0_total is not None else 'N/A':<12} "
                  f"{f'{sel_gt1_total:.1f}%' if sel_gt1_total is not None else 'N/A':<12} "
                  f"{f'{sel_gt2_total:.1f}%' if sel_gt2_total is not None else 'N/A':<12} "
                  f"{f'{sel_gt3_total:.1f}%' if sel_gt3_total is not None else 'N/A':<12} "
                  f"{f'{sel_gt4_total:.1f}%' if sel_gt4_total is not None else 'N/A':<12}")

    print("\n" + "=" * 200)
    print("\nINTERPRETATION:")
    print("- On-Target: More negative = stronger binding (better)")
    print("- Off-Target: Less negative = weaker binding (better for selectivity)")
    print("- Selectivity: More positive = more selective for on-target (better)")
    print("- QED: Higher is better (0-1 scale, drug-likeness)")
    print("- SA: Higher is better (0-1 scale, normalized synthetic accessibility - original 1~10 converted to 1~0)")
    print("- TM-score: Higher = more structurally similar proteins (LOW TM-score analysis)")
    print("\nVINA SCORES:")
    print("- Score(On/Off): Vina score_only mode (no docking, just scoring)")
    print("- Min(On/Off): Vina minimize mode (local minimization)")
    print("- Diff: Off-target - On-target (positive = better selectivity)")
    print("- NOTE: These are NOT the same as selectivity or delta score")
    print("- Score/Min Diff measures absolute difference, not selectivity")
    print("\nNOTE:")
    print("- v2 models (dual_head2_atom_v2, dual_head2_1p_all_v2) use gradient weights head1=100,25 / head2=100,0")
    print("- type200 models use gradient weights head1=100,25 / head2=200,0")
    print("- h1h2 models use gradient weights head1=100,25 / head2=100,25")
    print("- head2_ligand_query_787k: Model 8 with ligand query mechanism (checkpoint at 787k)")
    print("- head2_bidirectional_675k: Model 9 with bidirectional cross-attention (checkpoint at 675k)")
    print("=" * 200)


def main():
    parser = argparse.ArgumentParser(description='Analyze docking results for LOW TM-score pairs')
    parser.add_argument('--no-filter', action='store_true',
                        help='Disable docking failure filter (include molecules with vina score >= 0)')
    args = parser.parse_args()

    filter_failed_docking = not args.no_filter

    tmscore_file = './data/tmscore_extreme_pairs.txt'

    print(f"Loading LOW TM-score pairs from {tmscore_file}...")
    print(f"Docking failure filter: {'ENABLED' if filter_failed_docking else 'DISABLED'}")
    pairs = load_low_tmscore_pairs(tmscore_file)
    print(f"Loaded {len(pairs)} LOW TM-score pairs")
    print()

    all_pairs_results = []
    for on_id, off_id, tmscore in pairs:
        print(f"Analyzing pair ({on_id}, {off_id}) - LOW TM-score: {tmscore:.6f}...")
        pair_result = analyze_pair(on_id, off_id, tmscore, filter_failed_docking=filter_failed_docking)
        all_pairs_results.append(pair_result)

    print("\nAggregating results...")
    final_stats = aggregate_results(all_pairs_results)

    # Print to console
    print_detailed_report(all_pairs_results, final_stats, filter_failed_docking=filter_failed_docking)

    # Save to file
    if filter_failed_docking:
        output_file = 'tmscore_pairs_analysis_low_filtered.txt'
    else:
        output_file = 'tmscore_pairs_analysis_low_unfiltered.txt'

    with open(output_file, 'w') as f:
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        print_detailed_report(all_pairs_results, final_stats, filter_failed_docking=filter_failed_docking)
        sys.stdout = old_stdout

    print(f"\n\nReport saved to: {output_file}")


if __name__ == '__main__':
    main()
