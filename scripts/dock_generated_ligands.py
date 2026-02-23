#!/usr/bin/env python3
"""
Unified Docking Script for Generated Ligands
Supports both ID-specific and random off-target selection modes
Based on evaluate_diffusion.py structure
"""

import argparse
import os
import sys
import json
import random
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

# Add current directory to system path for importing internal modules
sys.path.append(os.path.abspath('./'))

from utils.evaluation.docking_vina import VinaDockingTask
from utils.evaluation import analyze, scoring_func, eval_atom_type, eval_bond_length
from utils import misc, reconstruct, transforms

# Additional imports for visualization
import pickle
import shutil

# LMDB imports
import lmdb
from torch_geometric.transforms import Compose
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH, ProteinLigandData, torchify_dict

# Visualization functions
def save_ligand_sdf(ligand_mol, filepath, ligand_name, properties=None):
    """Save ligand molecule as SDF file with properties"""
    try:
        if ligand_mol is None:
            return False

        # Set molecule name
        ligand_mol.SetProp("_Name", ligand_name)

        # Add properties if provided
        if properties:
            for key, value in properties.items():
                if value is not None:
                    ligand_mol.SetProp(str(key), str(value))

        # Write SDF file
        writer = Chem.SDWriter(filepath)
        writer.write(ligand_mol)
        writer.close()

        return True
    except Exception as e:
        print(f"Warning: Failed to save SDF file {filepath}: {e}")
        return False

def load_docked_pose_from_pdb(pose_pdb_path):
    """Load docked ligand pose from PDB file and convert to RDKit molecule"""
    try:
        import tempfile
        import subprocess

        if not os.path.exists(pose_pdb_path):
            return None

        # Convert PDB to SDF using obabel for better molecule handling
        with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp_sdf:
            try:
                subprocess.run(
                    f'obabel {pose_pdb_path} -O {tmp_sdf.name}',
                    shell=True, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )

                # Load molecule from SDF
                supplier = Chem.SDMolSupplier(tmp_sdf.name, removeHs=False)
                mol = next(iter(supplier), None)

                # Clean up temporary file
                os.unlink(tmp_sdf.name)

                return mol
            except Exception:
                # Clean up temporary file on error
                if os.path.exists(tmp_sdf.name):
                    os.unlink(tmp_sdf.name)
                return None

    except Exception as e:
        print(f"Warning: Failed to load docked pose from {pose_pdb_path}: {e}")
        return None

def save_protein_pdb(protein_info, output_dir, target_type):
    """Copy protein PDB file to visualization directory (once per protein)"""
    try:
        source_pdb = protein_info.get('protein_pdb', '')
        if not source_pdb or not os.path.exists(source_pdb):
            print(f"Warning: Source PDB not found: {source_pdb}")
            return None

        # Create target filename with protein name only
        protein_name = protein_info.get('protein_name', 'unknown')
        target_pdb = os.path.join(output_dir, f"{target_type}_{protein_name}.pdb")

        # Only copy if file doesn't exist already
        if not os.path.exists(target_pdb):
            shutil.copy2(source_pdb, target_pdb)
            return target_pdb
        else:
            # File already exists, return existing path
            return target_pdb
    except Exception as e:
        print(f"Warning: Failed to copy PDB file: {e}")
        return None


# Import json_to_txt_converter functions
def get_protein_names_from_data(data):
    """Extract protein names from the data (adapted from json_to_txt_converter.py)"""
    if not data:
        return {}
    
    first_entry = data[0]
    proteins = {}
    
    # On-target protein
    proteins['on_target'] = first_entry['docking_results']['on_target']['protein_info']['protein_name']
    
    # Off-target proteins - dynamically detect available off-targets
    off_targets = {}
    for key, value in first_entry['docking_results'].items():
        if key.startswith('off_target_'):
            off_targets[key] = value['protein_info']['protein_name']
    
    proteins['off_targets'] = off_targets
    proteins['off_target_keys'] = sorted(off_targets.keys())  # Store sorted keys for consistent ordering
    return proteins

def generate_txt_summary(json_path, output_path=None, experiment_name=None):
    """Generate TXT summary from JSON docking results (adapted from json_to_txt_converter.py)"""
    import subprocess
    import sys
    
    if output_path is None:
        output_path = json_path.replace('.json', '.txt')
    
    if experiment_name is None:
        # Extract experiment name from path
        experiment_name = os.path.basename(os.path.dirname(json_path)).upper()
    
    try:
        # Use subprocess to call json_to_txt_converter.py script
        cmd = [
            sys.executable, 
            'json_to_txt_converter.py', 
            json_path, 
            '-o', output_path,
            '-n', experiment_name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            return output_path
        else:
            print(f"Warning: TXT generation failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Warning: Could not generate TXT summary: {e}")
        return None

# Configuration - can be overridden via environment variables
MULTIPRO_VALIDATION_TEST_SET = os.environ.get(
    "MULTIPRO_VALIDATION_TEST_SET", "./data/multipro_validation_test_set")
CROSSDOCK_LMDB_PATH = os.environ.get(
    "CROSSDOCK_LMDB_PATH", "./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb")
CROSSDOCK_SPLIT_FILE = os.environ.get(
    "CROSSDOCK_SPLIT_FILE", "./data/crossdocked_pocket10_pose_split.pt")

# Global cache for CrossDock dataset
_CROSSDOCK_TEST_SET_CACHE = None

# Helper functions from evaluate_diffusion.py
def print_dict(d, logger):
    """Print dictionary contents in a clean format"""
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')

def print_ring_ratio(all_ring_sizes, logger):
    """Print ring size distribution for generated molecules"""
    ring_info = {}
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        ratio = n_mol / len(all_ring_sizes)
        logger.info(f'ring size: {ring_size} ratio: {ratio:.3f}')
        ring_info[ring_size] = f'{ratio:.3f}'
    return ring_info

def load_multipro_validation_info():
    """Load multi-protein validation test set info"""
    info_file = os.environ.get("MULTIPRO_VALIDATION_INFO", "./data/multipro_validation_info.json")
    if not os.path.exists(info_file):
        raise FileNotFoundError(f"Multi-protein validation info not found: {info_file}")

    with open(info_file, 'r') as f:
        validation_info = json.load(f)

    return validation_info

def load_crossdock_test_set(transform=None):
    """
    Load CrossDock test set once and cache it

    Args:
        transform: Data transformation to apply

    Returns:
        test_set: CrossDock test set
    """
    global _CROSSDOCK_TEST_SET_CACHE

    # Check if already loaded with same transform
    if _CROSSDOCK_TEST_SET_CACHE is not None:
        return _CROSSDOCK_TEST_SET_CACHE

    # Use get_dataset to load the test set properly (like sample_diffusion.py)
    from easydict import EasyDict

    config_dict = {
        'name': 'pl',
        'path': CROSSDOCK_LMDB_PATH,
        'split': CROSSDOCK_SPLIT_FILE
    }
    config = EasyDict(config_dict)

    print(f"Loading CrossDock test set from LMDB (this may take a moment)...")
    # Load dataset with transform
    dataset, subsets = get_dataset(config=config, transform=transform)
    test_set = subsets['test']
    print(f"CrossDock test set loaded: {len(test_set)} samples")

    # Cache the test set
    _CROSSDOCK_TEST_SET_CACHE = test_set

    return test_set

def load_crossdock_test_data_from_lmdb(data_id, transform=None):
    """
    Load CrossDock test set data from LMDB using data_id (0-100)

    Args:
        data_id (int): Test set index (0-100)
        transform: Data transformation to apply

    Returns:
        ProteinLigandData: Loaded data or None if failed
    """
    try:
        # Load test set (cached)
        test_set = load_crossdock_test_set(transform=transform)

        if data_id >= len(test_set):
            print(f"Error: data_id {data_id} is out of range. Test set has {len(test_set)} samples.")
            return None

        # Get data from test set
        data = test_set[data_id]

        print(f"Loaded CrossDock test data_id {data_id}")
        if hasattr(data, 'ligand_filename'):
            print(f"  Ligand filename: {data.ligand_filename}")
        if hasattr(data, 'protein_filename'):
            print(f"  Protein filename: {data.protein_filename}")

        return data

    except Exception as e:
        print(f"Error loading CrossDock LMDB data: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_center_of_mass(pos, atomic_numbers):
    """Calculate center of mass for a molecule"""
    if len(pos) == 0:
        return np.array([0.0, 0.0, 0.0])
    
    masses = np.array(atomic_numbers, dtype=float)
    total_mass = np.sum(masses)
    if total_mass == 0:
        return np.mean(pos, axis=0)
    
    com = np.sum(pos * masses.reshape(-1, 1), axis=0) / total_mass
    return com

def load_reference_ligand_center(sdf_path):
    """Load reference ligand center from SDF file"""
    if not os.path.exists(sdf_path):
        return None
    
    try:
        mol = Chem.SDMolSupplier(sdf_path)[0]
        if mol is None:
            return None
        
        conf = mol.GetConformer()
        pos = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        
        return calculate_center_of_mass(pos, atomic_numbers)
    except Exception as e:
        print(f"  Error loading reference ligand: {e}")
        return None

def get_protein_info_by_ids(validation_ids, use_lmdb_only=False, transform=None):
    """Get protein information for specific validation IDs using multipro_validation_info.json like sample_diffusion.py

    Args:
        validation_ids: List of validation IDs
        use_lmdb_only: If True, load from CrossDock LMDB instead of multipro validation set
        transform: Data transformation to apply when loading from LMDB
    """
    if use_lmdb_only:
        # Load from CrossDock LMDB (test set indices 0-100)
        return get_protein_info_from_crossdock_lmdb(validation_ids, transform=transform)

    # Original multipro validation set approach
    # Load validation info to use SAME mapping as sample_diffusion.py
    validation_info = load_multipro_validation_info()

    protein_info_list = []
    for val_id in validation_ids:
        if val_id >= len(validation_info):
            raise ValueError(f"Validation ID {val_id} is out of range. Available: 0-{len(validation_info)-1}")

        # Get protein info from validation_info (SAME as sampling)
        entry = validation_info[val_id]
        protein_dir_name = entry['protein_dir']
        protein_dir = os.path.join(MULTIPRO_VALIDATION_TEST_SET, protein_dir_name)

        print(f"Loading validation_id {val_id}: {protein_dir_name} (data_id: {entry['idx']})")

        # Find actual protein and ligand files
        # Use glob to filter by file extension
        pdb_files = glob(os.path.join(protein_dir, "*.pdb"))
        if not pdb_files:
            print(f"Error: No PDB files found in {protein_dir}")
            continue
        protein_pdb = pdb_files[0]  # Use first PDB file found

        sdf_files = glob(os.path.join(protein_dir, "*.sdf"))
        ref_ligand_sdf = sdf_files[0] if sdf_files else None

        # Load reference ligand center and molecule for automatic translation and box sizing
        ref_ligand_center = None
        ref_ligand_mol = None
        if ref_ligand_sdf:
            ref_ligand_center = load_reference_ligand_center(ref_ligand_sdf)
            # Also load the full molecule for box size calculation
            try:
                suppl = Chem.SDMolSupplier(ref_ligand_sdf, removeHs=False)
                ref_ligand_mol = next((m for m in suppl if m is not None), None)
            except Exception as e:
                print(f"    Warning: Could not load reference ligand molecule: {e}")
                ref_ligand_mol = None

        # Convert .pdb to .pdbqt for docking
        protein_pdbqt = protein_pdb.replace('.pdb', '.pdbqt')
        if not os.path.exists(protein_pdbqt):
            protein_pdbqt = protein_pdb

        protein_info = {
            'validation_id': val_id,
            'protein_name': protein_dir_name,
            'protein_dir': protein_dir,
            'protein_pdb': protein_pdb,
            'protein_pdbqt': protein_pdbqt,
            'ref_ligand': ref_ligand_sdf,
            'ref_ligand_center': ref_ligand_center,
            'ref_ligand_mol': ref_ligand_mol,  # Reference ligand molecule for box size calculation
            'ref_affinity': None,  # Will be loaded from SDF if needed
            'data_idx': None  # Not needed for this mapping
        }

        protein_info_list.append(protein_info)
        print(f"  Validation ID {val_id}: {protein_dir_name}")
        if ref_ligand_center is not None:
            print(f"    Reference ligand center: [{ref_ligand_center[0]:.3f}, {ref_ligand_center[1]:.3f}, {ref_ligand_center[2]:.3f}]")

    return protein_info_list

def get_protein_info_from_crossdock_lmdb(data_ids, transform=None):
    """Get protein information from CrossDock LMDB for specific data IDs (0-100)"""
    protein_info_list = []

    for data_id in data_ids:
        # Load data from CrossDock LMDB
        data = load_crossdock_test_data_from_lmdb(data_id, transform=transform)

        if data is None:
            print(f"Error: Failed to load data_id {data_id} from CrossDock LMDB")
            continue

        # Extract protein name from ligand_filename
        protein_name = 'unknown'
        if hasattr(data, 'ligand_filename'):
            # CrossDock ligand_filename format: "PDBID_chain_rec/PDBID_chain_rec_PDBID_lig_csd_0.sdf"
            protein_name = data.ligand_filename.split('/')[0] if '/' in data.ligand_filename else data.ligand_filename[:10]

        # Get reference ligand center and positions from data
        ref_ligand_center = None
        ref_ligand_mol = None
        ref_ligand_pos = None
        if hasattr(data, 'ligand_pos'):
            ligand_pos = data.ligand_pos.cpu().numpy() if hasattr(data.ligand_pos, 'cpu') else data.ligand_pos
            ligand_element = data.ligand_element.cpu().numpy() if hasattr(data.ligand_element, 'cpu') else data.ligand_element
            ref_ligand_center = calculate_center_of_mass(ligand_pos, ligand_element)
            ref_ligand_pos = ligand_pos  # Store positions for box size calculation

            # Try to load reference ligand from SDF file for box size calculation
            if hasattr(data, 'ligand_filename'):
                ref_sdf_path = os.path.join('./data/crossdocked_pocket10', data.ligand_filename)
                if os.path.exists(ref_sdf_path):
                    try:
                        suppl = Chem.SDMolSupplier(ref_sdf_path, removeHs=False)
                        ref_ligand_mol = next((m for m in suppl if m is not None), None)
                    except Exception as e:
                        print(f"    Warning: Could not load reference ligand from SDF: {e}")

        protein_info = {
            'data_id': data_id,
            'protein_name': protein_name,
            'protein_filename': data.protein_filename if hasattr(data, 'protein_filename') else None,
            'ligand_filename': data.ligand_filename if hasattr(data, 'ligand_filename') else None,
            'protein_pdb': None,  # LMDB mode doesn't use PDB files directly
            'protein_pdbqt': None,
            'ref_ligand': None,
            'ref_ligand_center': ref_ligand_center,
            'ref_ligand_mol': ref_ligand_mol,  # Reference ligand molecule for box size calculation
            'ref_ligand_pos': ref_ligand_pos,  # Reference ligand positions (fallback for box size)
            'ref_affinity': None,
            'lmdb_data': data  # Store the actual data for docking
        }

        protein_info_list.append(protein_info)
        print(f"  Data ID {data_id}: {protein_name}")
        if ref_ligand_center is not None:
            print(f"    Reference ligand center: [{ref_ligand_center[0]:.3f}, {ref_ligand_center[1]:.3f}, {ref_ligand_center[2]:.3f}]")

    return protein_info_list

def get_random_protein_info(exclude_validation_id, num_off_targets, use_lmdb_only=False, transform=None, max_range=100):
    """Get random protein information excluding specified validation ID

    Args:
        exclude_validation_id: ID to exclude (on-target)
        num_off_targets: Number of random off-targets to select
        use_lmdb_only: If True, use CrossDock LMDB (0-100)
        transform: Data transformation for LMDB loading
        max_range: Maximum ID range for LMDB mode
    """
    if use_lmdb_only:
        # CrossDock LMDB mode: randomly select from test set (0-100)
        available_ids = [i for i in range(max_range) if i != exclude_validation_id]

        if len(available_ids) < num_off_targets:
            print(f"Warning: Not enough test entries ({len(available_ids)} available) for {num_off_targets} off-targets")
            num_off_targets = len(available_ids)

        if num_off_targets == 0:
            print("Error: No off-target test entries available")
            return []

        # Randomly select off-target IDs
        selected_ids = random.sample(available_ids, num_off_targets)
        print(f"Randomly selected off-target data_ids: {selected_ids}")

        return get_protein_info_from_crossdock_lmdb(selected_ids, transform=transform)

    # Original multipro validation set approach
    # Load validation info to use SAME mapping as sample_diffusion.py
    validation_info = load_multipro_validation_info()

    # Get available validation IDs (exclude on-target)
    available_validation_ids = []
    for i in range(len(validation_info)):
        if i != exclude_validation_id:
            available_validation_ids.append(i)

    if len(available_validation_ids) < num_off_targets:
        print(f"Warning: Not enough validation entries ({len(available_validation_ids)} available) for {num_off_targets} off-targets")
        num_off_targets = len(available_validation_ids)

    if num_off_targets == 0:
        print("Error: No off-target validation entries available")
        return []

    # Randomly select off-target validation IDs
    selected_validation_ids = random.sample(available_validation_ids, num_off_targets)
    print(f"Randomly selected off-target validation_ids: {selected_validation_ids}")

    return get_protein_info_by_ids(selected_validation_ids, use_lmdb_only=False, transform=transform)



def calculate_box_size_from_ref_ligand(ref_ligand_mol=None, ref_ligand_pos=None, buffer=10.0, round_to=10):
    """
    Calculate docking box size from reference ligand (like dock_hwanhee.py)

    Args:
        ref_ligand_mol: RDKit molecule of reference ligand (preferred)
        ref_ligand_pos: numpy array of reference ligand positions (fallback)
        buffer: Buffer to add to box size (default: 10.0)
        round_to: Round up to nearest multiple (default: 10)

    Returns:
        box_size: (size_x, size_y, size_z) tuple
    """
    if ref_ligand_mol is not None:
        conf = ref_ligand_mol.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(ref_ligand_mol.GetNumAtoms())])
    elif ref_ligand_pos is not None:
        coords = np.array(ref_ligand_pos)
    else:
        raise ValueError("Either ref_ligand_mol or ref_ligand_pos must be provided")

    # Calculate size like dock_hwanhee.py
    size = np.max(coords, axis=0) - np.min(coords, axis=0)
    size = np.ceil(size / round_to) * round_to + buffer

    return tuple(size)


def dock_to_targets(original_ligand_pos, ligand_atom_types, ligand_aromatic, ligand_exp_atom, atom_enc_mode, protein_info_list, target_types, docking_mode='vina_dock', exhaustiveness=16, result_file_path=None):
    """Dock a ligand to target proteins with individual translation for each target

    For off-targets:
    - Translate ligand to reference ligand center
    - Set docking box center to reference ligand center (like dock_hwanhee.py)
    - Set docking box size based on reference ligand size
    """
    results = {}
    
    for protein_info, target_type in zip(protein_info_list, target_types):
        protein_name = protein_info['protein_name']
        print(f"  Docking to {target_type}: {protein_name}")
        
        try:
            # DIFFERENT APPROACH: on-target uses original coordinates (like evaluate_diffusion.py), off-targets use translation
            if target_type == 'on_target':
                # On-target: Use original coordinates (like evaluate_diffusion.py)
                translated_pos = original_ligand_pos
                print(f"    {target_type}: Using original generated coordinates (no translation)")
            else:
                # Off-targets: Apply reference ligand alignment for better comparison
                # Also use reference ligand for docking box (like dock_hwanhee.py)
                target_ref_center = protein_info.get('ref_ligand_center')
                ref_ligand_mol = protein_info.get('ref_ligand_mol')  # Reference ligand molecule for box size
                ref_ligand_pos = protein_info.get('ref_ligand_pos')  # Fallback: reference ligand positions

                if target_ref_center is not None:
                    # Calculate original ligand center
                    original_center = calculate_center_of_mass(original_ligand_pos, ligand_atom_types)

                    # Calculate translation vector for this specific target
                    translation_vector = target_ref_center - original_center

                    # Apply translation for this target
                    translated_pos = original_ligand_pos + translation_vector

                    # Verify translation
                    new_center = calculate_center_of_mass(translated_pos, ligand_atom_types)
                    distance_to_ref = np.linalg.norm(new_center - target_ref_center)

                    print(f"    {target_type}: Applying reference ligand alignment")
                    print(f"    Target reference center: [{target_ref_center[0]:.3f}, {target_ref_center[1]:.3f}, {target_ref_center[2]:.3f}]")
                    print(f"    Translation vector: [{translation_vector[0]:.3f}, {translation_vector[1]:.3f}, {translation_vector[2]:.3f}]")
                    print(f"    Distance to target reference: {distance_to_ref:.3f} Å")

                    # Calculate box size from reference ligand (like dock_hwanhee.py)
                    # Try ref_ligand_mol first, fallback to ref_ligand_pos
                    if ref_ligand_mol is not None:
                        ref_box_size = calculate_box_size_from_ref_ligand(ref_ligand_mol=ref_ligand_mol)
                        print(f"    Reference-based box size (from mol): [{ref_box_size[0]:.1f}, {ref_box_size[1]:.1f}, {ref_box_size[2]:.1f}]")
                    elif ref_ligand_pos is not None:
                        ref_box_size = calculate_box_size_from_ref_ligand(ref_ligand_pos=ref_ligand_pos)
                        print(f"    Reference-based box size (from pos): [{ref_box_size[0]:.1f}, {ref_box_size[1]:.1f}, {ref_box_size[2]:.1f}]")
                    else:
                        ref_box_size = None
                        print(f"    Warning: No reference ligand available for box size calculation")
                else:
                    print(f"    Warning: No reference center found for {target_type} {protein_name}, using original coordinates")
                    translated_pos = original_ligand_pos
                    ref_box_size = None
            
            # Reconstruct molecule with translated coordinates (following evaluate_diffusion.py pattern)
            try:
                ligand_mol = reconstruct.reconstruct_from_generated(translated_pos, ligand_atom_types, ligand_aromatic, ligand_exp_atom)
                smiles = Chem.MolToSmiles(ligand_mol)
            except reconstruct.MolReconsError:
                print(f"    Failed to reconstruct molecule for {protein_name}")
                results[target_type] = {
                    'affinity': None,
                    'protein_info': protein_info,
                    'success': False,
                    'mol': None,
                    'smiles': None,
                    'chem_results': None
                }
                continue
            
            # Check if molecule is fragmented (following evaluate_diffusion.py)
            if '.' in smiles:
                print(f"    Fragmented molecule detected for {protein_name}")
                results[target_type] = {
                    'affinity': None,
                    'protein_info': protein_info,
                    'success': False,
                    'mol': ligand_mol,
                    'smiles': smiles,
                    'chem_results': None
                }
                continue
            
            # Calculate chemical properties (following evaluate_diffusion.py)
            try:
                chem_results = scoring_func.get_chem(ligand_mol)
            except Exception as e:
                print(f"    Failed to calculate chemical properties: {e}")
                chem_results = None
            

            # Create docking task using evaluate_diffusion.py approach
            # Check if LMDB mode (protein_info has 'lmdb_data')
            if 'lmdb_data' in protein_info and protein_info['lmdb_data'] is not None:
                # LMDB mode: Use ligand_filename from LMDB data
                lmdb_data = protein_info['lmdb_data']
                if hasattr(lmdb_data, 'ligand_filename'):
                    ligand_filename = lmdb_data.ligand_filename
                    protein_filename = os.path.basename(ligand_filename)[:10] + '.pdb'
                    protein_root_dir = os.path.join('./data/test_set/', os.path.dirname(ligand_filename))
                    print(f"    LMDB mode - Using protein from ligand_filename: {protein_filename}")
                    print(f"    Protein root: {protein_root_dir}")
                else:
                    print(f"    Error: LMDB data missing ligand_filename")
                    results[target_type] = {
                        'affinity': None,
                        'protein_info': protein_info,
                        'success': False,
                        'mol': ligand_mol,
                        'smiles': smiles,
                        'chem_results': chem_results
                    }
                    continue
            ############ On-target Docking ##############
            elif target_type == 'on_target' and result_file_path:
                # For on-target: Use the SAME protein file approach as evaluate_diffusion.py
                try:
                    # 1. Load result.pt to access original data object
                    # Load the result file to get original data
                    result_data = torch.load(result_file_path)
                    if 'data' in result_data and hasattr(result_data['data'], 'ligand_filename'):
                        ligand_filename = result_data['data'].ligand_filename
                        # Use evaluate_diffusion.py logic: ligand_filename[:10] + '.pdb'

                        # 2. Determine protein file path (same as evaluate_diffusion.py)
                        protein_filename = os.path.basename(ligand_filename)[:10] + '.pdb'
                        protein_root_dir = os.path.join('./data/test_set/', os.path.dirname(ligand_filename))
                        print(f"    On-target using evaluate_diffusion.py protein path: {protein_filename}")
                        print(f"    Protein root: {protein_root_dir}")
                    else:
                        raise Exception("No ligand_filename in data")
                except Exception as e:
                    print(f"    Warning: Could not get original protein path, using fallback: {e}")
                    # Fallback to current approach
                    protein_full_path = protein_info['protein_pdb']
                    protein_filename = os.path.basename(protein_full_path)
                    protein_root_dir = os.path.dirname(protein_full_path)
            else:
                # For off-targets or when no result_file_path: Use multipro validation approach
                protein_full_path = protein_info['protein_pdb']
                protein_filename = os.path.basename(protein_full_path)
                protein_root_dir = os.path.dirname(protein_full_path)

            # Create VinaDockingTask with appropriate center and box size
            # For off-targets: use reference ligand center and box size (like dock_hwanhee.py)
            if target_type != 'on_target' and target_ref_center is not None:
                # Off-target: Use reference ligand center as docking box center
                # and reference ligand-based box size
                protein_path = os.path.join(protein_root_dir, protein_filename)

                # Convert numpy array to list for Vina compatibility
                center_list = [float(x) for x in target_ref_center]

                # Create VinaDockingTask with explicit center
                vina_task = VinaDockingTask(
                    protein_path=protein_path,
                    ligand_rdmol=ligand_mol,
                    center=center_list,  # Use reference ligand center as box center (as list)
                    size_factor=None if ref_box_size else 1.0,  # Use fixed size if ref_box_size available
                    buffer=5.0
                )

                # Override box size with reference ligand-based size if available
                if ref_box_size is not None:
                    # Convert to float for Vina compatibility
                    vina_task.size_x = float(ref_box_size[0])
                    vina_task.size_y = float(ref_box_size[1])
                    vina_task.size_z = float(ref_box_size[2])
                    print(f"    Off-target docking box: center={center_list}, size=[{vina_task.size_x}, {vina_task.size_y}, {vina_task.size_z}]")
            else:
                # On-target: Use ligand center as docking box center (original behavior)
                vina_task = VinaDockingTask.from_generated_mol(
                    ligand_mol, protein_filename, protein_root=protein_root_dir
                )
            
            # Run docking based on mode (exactly following evaluate_diffusion.py pattern)
            vina_results = None
            docked_pose_mol = None  # Store docked pose molecule for visualization

            if docking_mode == 'vina_score':
                # Exactly like evaluate_diffusion.py lines 203-208
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=exhaustiveness)
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=exhaustiveness)
                vina_results = {
                    'score_only': score_only_results,
                    'minimize': minimize_results
                }
            elif docking_mode == 'vina_dock':
                # Exactly like evaluate_diffusion.py lines 203-218
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=exhaustiveness)
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=exhaustiveness)
                vina_results = {
                    'score_only': score_only_results,
                    'minimize': minimize_results
                }

                # Save docked pose for visualization
                pose_save_dir = os.path.join(os.getcwd(), 'temp_docked_poses')
                os.makedirs(pose_save_dir, exist_ok=True)
                pose_save_path = os.path.join(pose_save_dir, f"docked_pose_{protein_name}_{target_type}.pdb")

                dock_results = vina_task.run(mode='dock', exhaustiveness=exhaustiveness, save_path=pose_save_path)
                vina_results['dock'] = dock_results

                # Load docked pose molecule for visualization
                if os.path.exists(pose_save_path):
                    docked_pose_mol = load_docked_pose_from_pdb(pose_save_path)
                    print(f"    Docked pose saved and loaded: {os.path.basename(pose_save_path)}")
                    vina_results['pose_path'] = pose_save_path
            else:
                raise ValueError(f"Unknown docking mode: {docking_mode}")
            
            # Extract primary affinity score for selectivity calculation
            primary_affinity = None
            if vina_results:
                if docking_mode == 'vina_score' and vina_results.get('minimize'):
                    primary_affinity = vina_results['minimize'][0]['affinity'] if vina_results['minimize'] else None
                elif docking_mode == 'vina_dock' and vina_results.get('dock'):
                    primary_affinity = vina_results['dock'][0]['affinity'] if vina_results['dock'] else None
            
            if primary_affinity is not None:
                results[target_type] = {
                    'affinity': primary_affinity,
                    'protein_info': protein_info,
                    'success': True,
                    'mol': ligand_mol,  # Original generated molecule
                    'docked_mol': docked_pose_mol,  # Docked pose molecule
                    'smiles': smiles,
                    'chem_results': chem_results,
                    'vina_results': vina_results
                }
                print(f"    Result: {primary_affinity:.3f}")
            else:
                print(f"    Docking failed")
                results[target_type] = {
                    'affinity': None,
                    'protein_info': protein_info,
                    'success': False,
                    'mol': ligand_mol,  # Original generated molecule
                    'docked_mol': None,  # No docked pose available
                    'smiles': smiles,
                    'chem_results': chem_results,
                    'vina_results': vina_results
                }
                
        except Exception as e:
            print(f"    Error: {e}")
            results[target_type] = {
                'affinity': None,
                'protein_info': protein_info,
                'success': False,
                'mol': None,
                'smiles': None,
                'chem_results': None,
                'vina_results': None
            }
    
    return results

def calculate_selectivity_score(on_target_affinity, off_target_affinities):
    """Calculate selectivity score using best (lowest) off-target value"""
    if on_target_affinity is None:
        return None
    
    valid_off_targets = [a for a in off_target_affinities if a is not None]
    if not valid_off_targets:
        return None
    
    best_off_target = min(valid_off_targets)  # Best (lowest/most negative) off-target affinity
    # Selectivity = on_target_affinity - best_off_target_affinity (more negative = better selectivity)
    return on_target_affinity - best_off_target

def calculate_selectivity_scores_by_vina_type(on_target_results, off_target_results_list):
    """Calculate selectivity scores for each vina result type (score, min, dock)"""
    selectivity_scores = {
        'score': None,
        'min': None, 
        'dock': None
    }
    
    # Extract on-target values for each type
    on_target_values = {}
    if on_target_results and 'vina_results' in on_target_results:
        vina_results = on_target_results['vina_results']
        
        # Score only
        if 'score_only' in vina_results and vina_results['score_only']:
            on_target_values['score'] = vina_results['score_only'][0]['affinity']
        
        # Minimize
        if 'minimize' in vina_results and vina_results['minimize']:
            on_target_values['min'] = vina_results['minimize'][0]['affinity']
        
        # Dock 
        if 'dock' in vina_results and vina_results['dock']:
            on_target_values['dock'] = vina_results['dock'][0]['affinity']
    
    # Extract off-target values for each type
    for vina_type in ['score', 'min', 'dock']:
        if vina_type not in on_target_values:
            continue
            
        off_target_values = []
        for off_target_results in off_target_results_list:
            if off_target_results and 'vina_results' in off_target_results:
                vina_results = off_target_results['vina_results']
                
                if vina_type == 'score' and 'score_only' in vina_results and vina_results['score_only']:
                    off_target_values.append(vina_results['score_only'][0]['affinity'])
                elif vina_type == 'min' and 'minimize' in vina_results and vina_results['minimize']:
                    off_target_values.append(vina_results['minimize'][0]['affinity'])
                elif vina_type == 'dock' and 'dock' in vina_results and vina_results['dock']:
                    off_target_values.append(vina_results['dock'][0]['affinity'])
        
        # Calculate selectivity for this vina type
        selectivity_scores[vina_type] = calculate_selectivity_score(
            on_target_values[vina_type], 
            off_target_values
        )
    
    return selectivity_scores

def get_sample_protein_info(result_filename, validation_info):
    """Get protein info for the sample based on its filename"""
    filename = os.path.basename(result_filename)[:-3]  # Remove .pt extension

    if validation_info is None:
        return None

    if filename.startswith('result_') and not filename[7:].isdigit():
        # Format: result_BRD4_HUMAN_42_168_0.pt - extract protein_dir
        sample_protein_dir = '_'.join(filename.split('_')[1:])
        
        # Find matching validation entry
        for i, entry in enumerate(validation_info):
            if entry['protein_dir'] == sample_protein_dir:
                # Create protein info for this sample
                protein_dir = os.path.join(MULTIPRO_VALIDATION_TEST_SET, entry['protein_dir'])
                ref_ligand_sdf = os.path.join(protein_dir, entry['ligand_file'])
                ref_ligand_center = None
                
                if os.path.exists(ref_ligand_sdf):
                    ref_ligand_center = load_reference_ligand_center(ref_ligand_sdf)
                
                protein_info = {
                    'validation_id': i,
                    'protein_name': entry['protein_dir'],
                    'protein_dir': protein_dir,
                    'protein_pdb': os.path.join(protein_dir, entry['protein_file']),
                    'protein_pdbqt': os.path.join(protein_dir, entry['protein_file']),
                    'ref_ligand': ref_ligand_sdf,
                    'ref_ligand_center': ref_ligand_center,
                    'ref_affinity': entry['affinity'],
                    'data_idx': entry['idx']
                }
                
                print(f"  Sample corresponds to protein: {sample_protein_dir}")
                if ref_ligand_center is not None:
                    print(f"  Sample reference ligand center: [{ref_ligand_center[0]:.3f}, {ref_ligand_center[1]:.3f}, {ref_ligand_center[2]:.3f}]")
                
                return protein_info
    
    return None

def main():
    # --- 1. Setup and Configuration ---
    parser = argparse.ArgumentParser(description="Unified docking script for generated ligands")
    
    # Input/Output
    parser.add_argument('--sample_path', type=str, required=True,
                        help='Path to sample directory containing result files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for docking results')
    
    # Docking mode selection
    parser.add_argument('--mode', type=str, choices=['id_specific', 'random'], default='id_specific',
                        help='Docking mode: id_specific or random')
    
    # ID-specific mode parameters (using multipro_validation_info.json like sampling)
    parser.add_argument('--on_target_id', type=int, default=0,
                        help='On-target validation ID from multipro_validation_info.json (for id_specific mode)')
    parser.add_argument('--off_target_ids', type=int, nargs='*', default=[1, 2, 3],
                        help='Off-target validation IDs from multipro_validation_info.json (for id_specific mode)')
    
    # Random mode parameters
    parser.add_argument('--on_target_random_id', type=int, default=0,
                        help='On-target validation ID (for random mode)')
    parser.add_argument('--num_off_targets', type=int, default=3,
                        help='Number of random off-targets (for random mode)')
    
    # Docking parameters
    parser.add_argument('--docking_mode', type=str, default='vina_dock',
                        choices=['vina_score', 'vina_dock'],
                        help='Docking mode')
    parser.add_argument('--exhaustiveness', type=int, default=16,
                        help='Vina exhaustiveness parameter')
    
    # Evaluation parameters
    parser.add_argument('--eval_step', type=int, default=-1,
                        help='Evaluation step to use (-1 for last step)')
    parser.add_argument('--eval_num_examples', type=int, default=None,
                        help='Number of examples to evaluate')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic',
                        help='Atom encoding mode for molecule reconstruction')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbose output')
    parser.add_argument('--save', type=int, default=1,
                        help='Save results')

    # Visualization parameters
    parser.add_argument('--save_visualization', action='store_true',
                        help='Save SDF files and PyMOL scripts for visualization')
    parser.add_argument('--visualization_dir', type=str, default=None,
                        help='Directory for visualization files (default: output_dir/visualize_ligand_pocket)')

    # LMDB mode parameter
    parser.add_argument('--use_lmdb_only', action='store_true', default=False,
                        help='Use CrossDock LMDB (test set 0-100) instead of multipro validation set')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        exit()
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup visualization directory if requested
    if args.save_visualization:
        if args.visualization_dir is None:
            args.visualization_dir = os.path.join(args.output_dir, 'visualize_ligand_pocket')
        os.makedirs(args.visualization_dir, exist_ok=True)
        print(f"Visualization files will be saved to: {args.visualization_dir}")

    # Setup logger
    logger = misc.get_logger('docking', log_dir=args.output_dir)
    
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')
    
    # --- 2. Load Protein Information ---
    print(f"=== Unified Docking Script ===")
    print(f"Mode: {args.mode}")
    print(f"LMDB Mode: {'CrossDock LMDB (test set 0-100)' if args.use_lmdb_only else 'Multipro validation set'}")

    # Create transform for LMDB loading if needed
    transform = None
    if args.use_lmdb_only:
        protein_featurizer = trans.FeaturizeProteinAtom()
        ligand_featurizer = trans.FeaturizeLigandAtom('add_aromatic')  # Use same mode as default
        transform = Compose([
            protein_featurizer,
            ligand_featurizer,
            trans.FeaturizeLigandBond(),
        ])

    if args.mode == 'id_specific':
        print(f"Original on-target {'data' if args.use_lmdb_only else 'validation'} ID: {args.on_target_id}")
        print(f"Original off-target {'data' if args.use_lmdb_only else 'validation'} IDs: {args.off_target_ids}")

        # PROTEIN ID VALIDATION AND SAFETY CHECK (skip for LMDB mode)
        if not args.use_lmdb_only:
            try:
                from utils.protein_id_manager import validate_protein_ids, get_safe_protein_ids, get_protein_name_by_id

                # Validate ID selection for duplicates
                is_valid, errors = validate_protein_ids(args.on_target_id, args.off_target_ids)

                if not is_valid:
                    print("=== PROTEIN DUPLICATION DETECTED ===")
                    for error in errors:
                        print(f"  WARNING: {error}")

                    # Get safe IDs automatically
                    safe_on_target, safe_off_targets = get_safe_protein_ids(args.on_target_id, args.off_target_ids)
                    print(f"=== AUTOMATIC CORRECTION ===")
                    print(f"Original: on_target={args.on_target_id}, off_targets={args.off_target_ids}")
                    print(f"Corrected: on_target={safe_on_target}, off_targets={safe_off_targets}")

                    # Display protein names for verification
                    print(f"On-target protein: ID {safe_on_target} -> {get_protein_name_by_id(safe_on_target)}")
                    for i, off_id in enumerate(safe_off_targets):
                        print(f"Off-target {i}: ID {off_id} -> {get_protein_name_by_id(off_id)}")

                    # Use safe IDs
                    args.on_target_id = safe_on_target
                    args.off_target_ids = safe_off_targets
                else:
                    print("✓ No protein duplications detected")
                    print(f"On-target protein: ID {args.on_target_id} -> {get_protein_name_by_id(args.on_target_id)}")
                    for i, off_id in enumerate(args.off_target_ids):
                        print(f"Off-target {i}: ID {off_id} -> {get_protein_name_by_id(off_id)}")

            except ImportError as e:
                print(f"WARNING: Could not import protein ID manager: {e}")
                print("Proceeding without duplication check - results may be inconsistent!")

        print(f"Final on-target {'data' if args.use_lmdb_only else 'validation'} ID: {args.on_target_id}")
        print(f"Final off-target {'data' if args.use_lmdb_only else 'validation'} IDs: {args.off_target_ids}")

        # Load specific protein information
        all_validation_ids = [args.on_target_id] + args.off_target_ids
        try:
            protein_info_list = get_protein_info_by_ids(all_validation_ids, use_lmdb_only=args.use_lmdb_only, transform=transform)
            target_types = ['on_target'] + [f'off_target_{i+1}' for i in range(len(args.off_target_ids))]
            logger.info(f"Loaded {len(protein_info_list)} target proteins for ID-specific evaluation")
        except Exception as e:
            logger.error(f"Error loading protein info: {e}")
            import traceback
            traceback.print_exc()
            return

    elif args.mode == 'random':
        print(f"On-target {'data' if args.use_lmdb_only else 'validation'} ID: {args.on_target_random_id}")
        print(f"Number of random off-targets: {args.num_off_targets}")

        # Load on-target and random off-targets
        try:
            on_target_info = get_protein_info_by_ids([args.on_target_random_id], use_lmdb_only=args.use_lmdb_only, transform=transform)
            off_target_info = get_random_protein_info(
                args.on_target_random_id,
                args.num_off_targets,
                use_lmdb_only=args.use_lmdb_only,
                transform=transform,
                max_range=100 if args.use_lmdb_only else None
            )

            protein_info_list = on_target_info + off_target_info
            target_types = ['on_target'] + [f'off_target_{i+1}' for i in range(len(off_target_info))]
            logger.info(f"Loaded 1 on-target + {len(off_target_info)} random off-target proteins")
        except Exception as e:
            logger.error(f"Error loading protein info: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # --- 3. Load Generated Data ---
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    if len(results_fn_list) == 0:
        logger.error(f'No result files found in {args.sample_path}')
        return
    
    # Sort files by data_id (following evaluate_diffusion.py pattern)
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    
    # Limit number of examples if specified
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    
    logger.info(f'Found {len(results_fn_list)} result files to process')
    
    # --- 4. Process Each Generated Ligand ---
    all_results = []
    total_ligands = 0

    # Initialize evaluation metrics (following evaluate_diffusion.py)
    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()

    # Atom type mapping function (mode will be passed to the function)
    atom_enc_mode = args.atom_enc_mode

    # Load validation info once for all samples (only needed for multipro mode)
    validation_info = None
    if not args.use_lmdb_only:
        validation_info = load_multipro_validation_info()

    # Track copied proteins to avoid duplicates
    copied_proteins = set()

    for result_file in tqdm(results_fn_list, desc="Processing ligands"):
        try:
            # Get sample-specific protein info for correct reference ligand
            sample_protein_info = get_sample_protein_info(result_file, validation_info)
            
            # Load generated data (following evaluate_diffusion.py pattern)
            result = torch.load(result_file)

            # Support both formats:
            # 1. Original format: pred_ligand_pos_traj [num_samples, num_steps, num_atoms, 3]
            # 2. Unified format: pos (final) and pos_traj (trajectory for single sample)
            if 'pred_ligand_pos_traj' in result:
                # Original format
                all_pred_ligand_pos = result['pred_ligand_pos_traj']
                all_pred_ligand_v = result['pred_ligand_v_traj']
                all_pred_exp_traj = result.get('pred_exp_traj', [])
                all_pred_exp_atom_traj = result.get('pred_exp_atom_traj', [])
            elif 'pos' in result:
                # Unified format (single sample per file)
                pos = result['pos']  # Final position [num_atoms, 3]
                v = result['v']      # Final types [num_atoms]
                pos_traj = result.get('pos_traj', None)  # [num_steps, num_atoms, 3]
                v_traj = result.get('v_traj', None)      # [num_steps, num_atoms, type]

                # Wrap in list to match expected format [num_samples=1, ...]
                if pos_traj is not None:
                    all_pred_ligand_pos = [pos_traj]
                    all_pred_ligand_v = [v_traj]
                else:
                    # No trajectory, just final position
                    all_pred_ligand_pos = [np.expand_dims(pos, axis=0)]  # [1, num_atoms, 3]
                    all_pred_ligand_v = [np.expand_dims(v, axis=0)]      # [1, num_atoms]
                all_pred_exp_traj = []
                all_pred_exp_atom_traj = []
            else:
                raise KeyError(f"Unknown result format. Expected 'pred_ligand_pos_traj' or 'pos'. Got: {list(result.keys())}")
            
            # Handle missing or malformed exp_atom_traj like in evaluate_diffusion.py
            if not all_pred_exp_atom_traj or len(all_pred_exp_atom_traj) != len(all_pred_ligand_pos):
                print(f"    Debug - Creating default exp_atom_traj for {len(all_pred_ligand_pos)} samples")
                all_pred_exp_atom_traj = [np.ones_like(all_pred_ligand_v[i]) for i in range(len(all_pred_ligand_pos))]
            else:
                # Check if any arrays in exp_atom_traj are empty (from 'wo' mode)
                has_empty_arrays = any(len(exp) == 0 if isinstance(exp, np.ndarray) else False for exp in all_pred_exp_atom_traj)
                if has_empty_arrays:
                    print(f"    Debug - Found empty exp_atom_traj arrays (likely from 'wo' mode), creating defaults")
                    all_pred_exp_atom_traj = [np.ones_like(all_pred_ligand_v[i]) for i in range(len(all_pred_ligand_pos))]

            if args.eval_step == -1:
                # Use final step
                pred_pos = [pos[-1] for pos in all_pred_ligand_pos]  # Final step for each sample
                pred_v = [v[-1] for v in all_pred_ligand_v]          # Final step for each sample
                if all_pred_exp_atom_traj:
                    # Check if individual arrays have elements before accessing
                    pred_exp_atom = []
                    for exp in all_pred_exp_atom_traj:
                        if isinstance(exp, np.ndarray) and len(exp) > 0:
                            pred_exp_atom.append(exp[-1])
                        else:
                            pred_exp_atom.append(None)
                else:
                    pred_exp_atom = [None] * len(pred_pos)
            else:
                # Use specific step
                pred_pos = [pos[args.eval_step] for pos in all_pred_ligand_pos]
                pred_v = [v[args.eval_step] for v in all_pred_ligand_v]
                if all_pred_exp_atom_traj:
                    # Check if individual arrays have elements before accessing
                    pred_exp_atom = []
                    for exp in all_pred_exp_atom_traj:
                        if isinstance(exp, np.ndarray) and len(exp) > args.eval_step:
                            pred_exp_atom.append(exp[args.eval_step])
                        else:
                            pred_exp_atom.append(None)
                else:
                    pred_exp_atom = [None] * len(pred_pos)
            
            # Process each ligand in the batch
            batch_size = len(pred_pos)

            for sample_idx in range(batch_size):
                num_samples += 1  # Count total samples for validity statistics
                total_ligands += 1
                ligand_name = f"ligand_{total_ligands}_from_{os.path.basename(result_file)}"

                print(f"Processing {ligand_name}...")
                
                # Extract ligand data
                ligand_pos = pred_pos[sample_idx]
                ligand_v = pred_v[sample_idx]
                ligand_exp_atom = pred_exp_atom[sample_idx] if pred_exp_atom[sample_idx] is not None else None
                
                # Convert to numpy if needed
                if hasattr(ligand_pos, 'cpu'):
                    ligand_pos = ligand_pos.cpu().numpy()
                elif hasattr(ligand_pos, 'numpy'):
                    ligand_pos = ligand_pos.numpy()
                    
                if hasattr(ligand_v, 'cpu'):
                    ligand_v = ligand_v.cpu().numpy() 
                elif hasattr(ligand_v, 'numpy'):
                    ligand_v = ligand_v.numpy()
                
                # Convert exp_atom to numpy if needed
                if ligand_exp_atom is not None:
                    if hasattr(ligand_exp_atom, 'cpu'):
                        ligand_exp_atom = ligand_exp_atom.cpu().numpy()
                    elif hasattr(ligand_exp_atom, 'numpy'):
                        ligand_exp_atom = ligand_exp_atom.numpy()
                
                # Process molecule data for docking
                try:
                    # Check for valid atom positions
                    if np.any(np.isnan(ligand_pos)) or np.any(np.isinf(ligand_pos)):
                        print(f"  Invalid atom positions detected (NaN or Inf)")
                        continue
                    
                    ligand_atom_types = transforms.get_atomic_number_from_index(ligand_v, mode=atom_enc_mode)
                    
                    # Check for valid atom types
                    if len(ligand_atom_types) == 0:
                        print(f"  No valid atom types found")
                        continue
                    
                    # Check for invalid atomic numbers
                    if np.any(np.array(ligand_atom_types) <= 0):
                        print(f"  Invalid atomic numbers detected: {ligand_atom_types}")
                        continue
                    
                    # Calculate original ligand center (for reference)
                    original_center = calculate_center_of_mass(ligand_pos, ligand_atom_types)
                    print(f"    Original ligand center: [{original_center[0]:.3f}, {original_center[1]:.3f}, {original_center[2]:.3f}]")
                    
                    ligand_aromatic = transforms.is_aromatic_from_index(ligand_v, mode=atom_enc_mode)
                    
                    # Handle missing or mismatched exp_atom data gracefully
                    if ligand_exp_atom is None or len(ligand_exp_atom) == 0:
                        ligand_exp_atom = [1.0] * len(ligand_atom_types)  # Default values
                    else:
                        # Handle shape mismatch - flatten if needed and adjust length
                        if ligand_exp_atom.ndim > 1:
                            ligand_exp_atom = ligand_exp_atom.flatten()
                        
                        # If exp_atom length doesn't match atom count, use defaults
                        if len(ligand_exp_atom) != len(ligand_atom_types):
                            ligand_exp_atom = [1.0] * len(ligand_atom_types)
                        else:
                            # Convert to list if it's numpy array
                            ligand_exp_atom = ligand_exp_atom.tolist()
                    
                    # Add stability check (following evaluate_diffusion.py)
                    stability_results = analyze.check_stability(ligand_pos, ligand_atom_types)
                    is_mol_stable, n_stable_atoms, total_atoms = stability_results

                    # Accumulate stability statistics (following evaluate_diffusion.py)
                    all_mol_stable += int(is_mol_stable)
                    all_atom_stable += n_stable_atoms
                    all_n_atom += total_atoms
                    all_atom_types += Counter(ligand_atom_types)

                    if not is_mol_stable:
                        print(f"    Unstable molecule detected ({n_stable_atoms}/{total_atoms} stable atoms)")
                        # Still proceed but with warning

                except reconstruct.MolReconsError as e:
                    print(f"  Molecule reconstruction failed: {e}")
                    continue
                except Exception as e:
                    print(f"  Unexpected error during reconstruction: {e}")
                    continue
                
                # Dock to all targets with individual translation for each target
                print(f"    Starting multi-target docking with individual translation...")
                docking_results = dock_to_targets(
                    ligand_pos, ligand_atom_types, ligand_aromatic, ligand_exp_atom,
                    atom_enc_mode, protein_info_list, target_types,
                    docking_mode=args.docking_mode,
                    exhaustiveness=args.exhaustiveness,
                    result_file_path=result_file
                )
                
                # Calculate selectivity scores for all vina types
                on_target_results = docking_results['on_target']
                off_target_results_list = [
                    docking_results[target_type] 
                    for target_type in target_types[1:]  # Skip on_target
                ]
                
                # Calculate selectivity for each vina type (score, min, dock)
                selectivity_scores = calculate_selectivity_scores_by_vina_type(on_target_results, off_target_results_list)
                
                # Keep original dock selectivity for backward compatibility
                on_target_affinity = docking_results['on_target']['affinity']
                off_target_affinities = [
                    docking_results[target_type]['affinity'] 
                    for target_type in target_types[1:]  # Skip on_target
                ]
                selectivity_score = calculate_selectivity_score(on_target_affinity, off_target_affinities)
                
                # Extract chemical properties from on-target result (following evaluate_diffusion.py)
                on_target_result = docking_results['on_target']
                chem_results = on_target_result.get('chem_results', {})
                vina_results = on_target_result.get('vina_results', {})
                smiles = on_target_result.get('smiles', None)

                # Update validity statistics (following evaluate_diffusion.py)
                # Check if reconstruction was successful (mol exists)
                if on_target_result.get('mol') is not None:
                    n_recon_success += 1

                    # Check if molecule is complete (not fragmented)
                    if smiles is not None and '.' not in smiles:
                        n_complete += 1

                        # Check if evaluation was successful (docking worked)
                        if on_target_result.get('success', False):
                            n_eval_success += 1

                # Create result summary
                result_summary = {
                    'ligand_name': ligand_name,
                    'source_file': os.path.basename(result_file),
                    'sample_index': sample_idx,
                    'mode': args.mode,
                    'smiles': smiles,
                    'on_target_affinity': on_target_affinity,
                    'off_target_affinities': off_target_affinities,
                    'selectivity_score': selectivity_score,  # Original dock selectivity for backward compatibility
                    'selectivity_scores_by_type': selectivity_scores,  # New selectivity scores for all vina types
                    'docking_results': docking_results,
                    # Chemical properties (following evaluate_diffusion.py)
                    'chem_results': chem_results,
                    'vina_results': vina_results,
                    'translation_info': {
                        'original_center': original_center.tolist() if 'original_center' in locals() else None,
                        'coordinate_handling': 'Unified approach: reference ligand alignment applied to ALL targets',
                        'on_target_translation': 'Reference ligand alignment applied',
                        'off_target_translation': 'Reference ligand alignment applied'
                    },
                    'parameters': {
                        'docking_mode': args.docking_mode,
                        'exhaustiveness': args.exhaustiveness,
                        'mode': args.mode,
                        'auto_translate': True
                    }
                }
                
                if args.mode == 'id_specific':
                    result_summary['on_target_id'] = args.on_target_id
                    result_summary['off_target_ids'] = args.off_target_ids
                elif args.mode == 'random':
                    result_summary['on_target_random_id'] = args.on_target_random_id
                    result_summary['num_off_targets'] = args.num_off_targets
                
                # Print summary
                if on_target_affinity is not None:
                    print(f"  On-target affinity: {on_target_affinity:.3f}")
                    if selectivity_score is not None:
                        print(f"  Selectivity score: {selectivity_score:.3f}")
                else:
                    print(f"  On-target docking failed")
                
                all_results.append(result_summary)

                # Save individual result
                if args.save:
                    result_file_path = os.path.join(args.output_dir, f"{ligand_name}_results.json")
                    with open(result_file_path, 'w') as f:
                        json.dump(result_summary, f, indent=2, default=str)

                # Save visualization files if requested
                if args.save_visualization:
                    print(f"    Saving visualization files for {ligand_name}...")

                    # Copy protein PDB files (only once per protein)
                    for target_type in target_types:
                        if target_type in docking_results and docking_results[target_type]['success']:
                            target_result = docking_results[target_type]
                            protein_name = target_result['protein_info']['protein_name']
                            protein_key = f"{target_type}_{protein_name}"

                            # Copy protein PDB only if not already copied
                            if protein_key not in copied_proteins:
                                protein_pdb_path = save_protein_pdb(
                                    target_result['protein_info'], args.visualization_dir, target_type
                                )
                                if protein_pdb_path:
                                    print(f"      Saved PDB: {os.path.basename(protein_pdb_path)}")
                                    copied_proteins.add(protein_key)

                    # Save SDF files for each target
                    for target_type in target_types:
                        if target_type in docking_results and docking_results[target_type]['success']:
                            target_result = docking_results[target_type]
                            ligand_mol = target_result.get('mol')
                            docked_mol = target_result.get('docked_mol')

                            # Prepare properties for SDF
                            sdf_properties = {
                                'Target_Type': target_type,
                                'Protein_Name': target_result['protein_info']['protein_name'],
                                'Affinity': target_result.get('affinity', 'N/A'),
                                'SMILES': target_result.get('smiles', 'N/A'),
                                'Ligand_Name': ligand_name
                            }

                            # Save original generated coordinates
                            if ligand_mol is not None:
                                sdf_path = os.path.join(args.visualization_dir, f"{ligand_name}_{target_type}_generated.sdf")
                                sdf_properties_gen = dict(sdf_properties)
                                sdf_properties_gen['Pose_Type'] = 'Generated'
                                if save_ligand_sdf(ligand_mol, sdf_path, f"{ligand_name}_{target_type}_generated", sdf_properties_gen):
                                    print(f"      Saved Generated SDF: {os.path.basename(sdf_path)}")

                            # Save docked pose coordinates (if available)
                            if docked_mol is not None:
                                sdf_path_docked = os.path.join(args.visualization_dir, f"{ligand_name}_{target_type}_docked.sdf")
                                sdf_properties_dock = dict(sdf_properties)
                                sdf_properties_dock['Pose_Type'] = 'Docked'
                                if save_ligand_sdf(docked_mol, sdf_path_docked, f"{ligand_name}_{target_type}_docked", sdf_properties_dock):
                                    print(f"      Saved Docked SDF: {os.path.basename(sdf_path_docked)}")
                            else:
                                print(f"      No docked pose available for {target_type} (docking mode: {args.docking_mode})")
                        
        except Exception as e:
            logger.error(f"Error processing {result_file}: {e}")
            continue
    
    # --- 5. Save All Results and Summary ---
    if args.save:
        # Save all results
        results_path = os.path.join(args.output_dir, "docking_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save protein information
        protein_info_path = os.path.join(args.output_dir, "protein_information.json")
        with open(protein_info_path, 'w') as f:
            json.dump({
                'mode': args.mode,
                'proteins': [info for info in protein_info_list],
                'target_types': target_types
            }, f, indent=2, default=str)
        
        logger.info(f"All results saved to: {results_path}")
        
        # --- 5.1. Generate TXT Summary Automatically ---
        if all_results:  # Only generate if we have results
            logger.info("Generating TXT summary...")
            experiment_name = os.path.basename(args.output_dir).upper()
            txt_output_path = os.path.join(args.output_dir, "docking_results.txt")
            
            try:
                txt_file = generate_txt_summary(results_path, txt_output_path, experiment_name)
                if txt_file:
                    logger.info(f"TXT summary saved to: {txt_file}")
                else:
                    logger.warning("TXT summary generation failed")
            except Exception as e:
                logger.error(f"Error generating TXT summary: {e}")
    
    # --- 6. Print Summary ---
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total ligands processed: {total_ligands}")
    logger.info(f"Mode: {args.mode}")

    # --- 6a. Validity Statistics (following evaluate_diffusion.py) ---
    logger.info(f"\n=== Validity Statistics ===")
    if num_samples > 0:
        fraction_mol_stable = all_mol_stable / num_samples
        fraction_atm_stable = all_atom_stable / all_n_atom if all_n_atom > 0 else 0.0
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

        # Print validity dict (following evaluate_diffusion.py format)
        for k, v in validity_dict.items():
            logger.info(f'{k}:\t{v:.4f}')

        logger.info(f'\nNumber of samples: {num_samples}')
        logger.info(f'Number of reconstructed mols: {n_recon_success}, complete mols: {n_complete}, evaluated mols: {n_eval_success}')
    else:
        logger.info("No samples processed for validity statistics")
        validity_dict = {}

    # --- 6b. Docking Statistics ---
    logger.info(f"\n=== Docking Statistics ===")

    # Calculate statistics
    successful_ligands = [r for r in all_results if r.get('on_target_affinity') is not None]
    selectivity_scores = [r['selectivity_score'] for r in successful_ligands if r.get('selectivity_score') is not None]
    
    if successful_ligands:
        logger.info(f"Successful docking: {len(successful_ligands)}/{total_ligands}")
        
        # On-target affinity statistics
        on_target_affinities = [r['on_target_affinity'] for r in successful_ligands]
        logger.info(f"On-target affinity: mean={np.mean(on_target_affinities):.3f}, std={np.std(on_target_affinities):.3f}")
        
        # Selectivity statistics
        if selectivity_scores:
            logger.info(f"Selectivity scores: mean={np.mean(selectivity_scores):.3f}, std={np.std(selectivity_scores):.3f}")
        
        # Chemical properties statistics (following evaluate_diffusion.py)
        qed_scores = [r['chem_results']['qed'] for r in successful_ligands if r.get('chem_results') and 'qed' in r['chem_results']]
        sa_scores = [r['chem_results']['sa'] for r in successful_ligands if r.get('chem_results') and 'sa' in r['chem_results']]
        
        if qed_scores:
            logger.info(f"QED:   Mean: {np.mean(qed_scores):.3f} Median: {np.median(qed_scores):.3f}")
        if sa_scores:
            logger.info(f"SA:    Mean: {np.mean(sa_scores):.3f} Median: {np.median(sa_scores):.3f}")
        
        # Vina scoring statistics (following evaluate_diffusion.py pattern)
        if args.docking_mode == 'vina_score':
            vina_score_only = [r['vina_results']['score_only'][0]['affinity'] for r in successful_ligands 
                             if r.get('vina_results') and r['vina_results'].get('score_only')]
            vina_min = [r['vina_results']['minimize'][0]['affinity'] for r in successful_ligands 
                       if r.get('vina_results') and r['vina_results'].get('minimize')]
            
            if vina_score_only:
                logger.info(f"Vina Score:  Mean: {np.mean(vina_score_only):.3f} Median: {np.median(vina_score_only):.3f}")
            if vina_min:
                logger.info(f"Vina Min  :  Mean: {np.mean(vina_min):.3f} Median: {np.median(vina_min):.3f}")
                
        elif args.docking_mode == 'vina_dock':
            vina_score_only = [r['vina_results']['score_only'][0]['affinity'] for r in successful_ligands 
                             if r.get('vina_results') and r['vina_results'].get('score_only')]
            vina_min = [r['vina_results']['minimize'][0]['affinity'] for r in successful_ligands 
                       if r.get('vina_results') and r['vina_results'].get('minimize')]
            vina_dock = [r['vina_results']['dock'][0]['affinity'] for r in successful_ligands 
                        if r.get('vina_results') and r['vina_results'].get('dock')]
            
            if vina_score_only:
                logger.info(f"Vina Score:  Mean: {np.mean(vina_score_only):.3f} Median: {np.median(vina_score_only):.3f}")
            if vina_min:
                logger.info(f"Vina Min  :  Mean: {np.mean(vina_min):.3f} Median: {np.median(vina_min):.3f}")
            if vina_dock:
                logger.info(f"Vina Dock :  Mean: {np.mean(vina_dock):.3f} Median: {np.median(vina_dock):.3f}")
    else:
        logger.info("No successful docking results")

    # Summary for visualization files
    if args.save_visualization:
        logger.info(f"\n=== Visualization Files ===")
        logger.info(f"Visualization directory: {args.visualization_dir}")

        # Count generated files
        if os.path.exists(args.visualization_dir):
            sdf_files = glob(os.path.join(args.visualization_dir, "*.sdf"))
            pdb_files = glob(os.path.join(args.visualization_dir, "*.pdb"))

            logger.info(f"Generated files:")
            logger.info(f"  SDF files: {len(sdf_files)}")
            logger.info(f"  PDB files: {len(pdb_files)}")

            if pdb_files and sdf_files:
                logger.info(f"\nTo visualize in PyMOL:")
                logger.info(f"1. Load protein PDB files: {', '.join([os.path.basename(f) for f in pdb_files])}")
                logger.info(f"2. Load ligand SDF files from the same directory")
                logger.info(f"3. Use PyMOL commands to show binding sites and interactions")

    # --- 7. Save Validity Metrics (following evaluate_diffusion.py) ---
    if args.save and validity_dict:
        metrics_path = os.path.join(args.output_dir, f'validity_metrics_{args.eval_step}.pt')
        torch.save({
            'info': f'Number of samples: {num_samples}, reconstructed mols: {n_recon_success}, complete mols: {n_complete}, evaluated mols: {n_eval_success}',
            'validity': validity_dict,
            'num_samples': num_samples,
            'n_recon_success': n_recon_success,
            'n_complete': n_complete,
            'n_eval_success': n_eval_success,
            'all_mol_stable': all_mol_stable,
            'all_atom_stable': all_atom_stable,
            'all_n_atom': all_n_atom,
        }, metrics_path)
        logger.info(f"\nValidity metrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()