

import os
import sys
sys.path.append(os.path.abspath('./'))

import pickle
import lmdb
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import logging

from utils.data import PDBProtein, parse_sdf_file
from datasets.pl_data import ProteinLigandData, torchify_dict
from scripts.data_preparation.clean_crossdocked import TYPES_FILENAME


class MultiProteinPairedDataset(Dataset):

    def __init__(self, lmdb_path, transform=None):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.db = None
        self.keys = None

    def _connect_db(self):
        assert self.db is None, 'A connection has already been opened.'
        # Auto-detect if LMDB is a directory or file
        import os
        is_dir = os.path.isdir(self.lmdb_path)

        self.db = lmdb.open(
            self.lmdb_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=is_dir,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        if self.db is not None:
            self.db.close()
            self.db = None
            self.keys = None

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))

        # Combine on-target and off-target proteins into a single graph
        on_target_protein = data['on_target_protein']
        off_target_proteins = data['off_target_proteins']

        # Initialize combined protein attributes from the on-target protein
        combined_protein_pos = on_target_protein['protein_pos']
        combined_protein_element = on_target_protein['protein_element']
        combined_protein_is_backbone = on_target_protein['protein_is_backbone']
        combined_protein_atom_to_aa_type = on_target_protein['protein_atom_to_aa_type']
        combined_protein_id = torch.zeros_like(on_target_protein['protein_element'], dtype=torch.long)

        # Append attributes from off-target proteins
        for i, protein in enumerate(off_target_proteins, 1):
            combined_protein_pos = torch.cat([combined_protein_pos, protein['protein_pos']], dim=0)
            combined_protein_element = torch.cat([combined_protein_element, protein['protein_element']], dim=0)
            combined_protein_is_backbone = torch.cat([combined_protein_is_backbone, protein['protein_is_backbone']], dim=0)
            combined_protein_atom_to_aa_type = torch.cat([combined_protein_atom_to_aa_type, protein['protein_atom_to_aa_type']], dim=0)
            protein_id_tensor = torch.full_like(protein['protein_element'], fill_value=i, dtype=torch.long)
            combined_protein_id = torch.cat([combined_protein_id, protein_id_tensor], dim=0)

        # Center the complex based on selected centering mode
        ligand_pos = data['ligand_pos']
        
        # No centering at dataset level - will be done at model level
        centered_ligand_pos = ligand_pos
        centered_protein_pos = combined_protein_pos

        # Create the final data object
        final_data = {
            # Ligand attributes
            'ligand_pos': centered_ligand_pos,
            'ligand_element': data['ligand_element'],
            'ligand_bond_index': data['ligand_bond_index'],
            'ligand_bond_type': data['ligand_bond_type'],
            'ligand_atom_feature': data['ligand_atom_feature'],

            # Combined protein attributes
            'protein_pos': centered_protein_pos,
            'protein_element': combined_protein_element,
            'protein_is_backbone': combined_protein_is_backbone,
            'protein_atom_to_aa_type': combined_protein_atom_to_aa_type,
            'protein_id': combined_protein_id,

            # Affinities and other metadata
            'on_target_affinity': data['on_target_affinity'],
            'off_target_affinities': torch.tensor(data['off_target_affinities'], dtype=torch.float32),
            'affinity': data['on_target_affinity'],  # Add affinity attribute for transforms
            'id': idx
        }

        # Add any other keys from the original data
        for k, v in data.items():
            if k not in final_data and (k.startswith('ligand_') or k.startswith('protein_')):
                 final_data[k] = v

        # Validate that critical tensors are not empty
        critical_fields = ['ligand_element', 'protein_element', 'ligand_pos', 'protein_pos']
        for field in critical_fields:
            if field in final_data and torch.is_tensor(final_data[field]):
                if final_data[field].numel() == 0:
                    raise ValueError(f"Critical field '{field}' is empty in sample {idx}")
        
        # Validate bond indices if present
        if 'ligand_bond_index' in final_data and torch.is_tensor(final_data['ligand_bond_index']):
            if final_data['ligand_bond_index'].numel() > 0:
                # Ensure bond indices are within valid range
                max_bond_idx = final_data['ligand_bond_index'].max().item()
                num_ligand_atoms = final_data['ligand_element'].size(0)
                if max_bond_idx >= num_ligand_atoms:
                    raise ValueError(f"Bond index {max_bond_idx} >= number of ligand atoms {num_ligand_atoms} in sample {idx}")

        try:
            graph_data = ProteinLigandData(**final_data)
        except Exception as e:
            raise

        if self.transform is not None:
            try:
                graph_data = self.transform(graph_data)
            except Exception as e:
                raise

        return graph_data


class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')

        # Check if raw_path is already a processed LMDB file
        if self.raw_path.endswith('.lmdb') and os.path.exists(self.raw_path):
            # Use the provided LMDB directly
            self.processed_path = self.raw_path
        else:
            # Generate processed path from raw path
            self.index_path = os.path.join(self.raw_path, 'index.pkl')
            self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                               os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')

        self.raw_affinity_path = os.path.join('/data/qianhao', TYPES_FILENAME)
        self.affinity_path = os.path.join('data', 'affinity_info_complete.pkl')
        self.transform = transform
        self.db = None
        self.keys = None
        self.affinity_info = None
        self._warned_missing = set()  # Track warned missing samples to avoid spam

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
            
    def _load_affinity_info(self):
        if self.affinity_info is not None:
            return
        if os.path.exists(self.affinity_path):
            with open(self.affinity_path, 'rb') as f:
                affinity_info = pickle.load(f)
        else:
            affinity_info = {}
            with open(self.raw_affinity_path, 'r') as f:
                for ln in tqdm(f.readlines()):
                    # <label> <pK> <RMSD to crystal> <Receptor> <Ligand> # <Autodock Vina score>
                    label, pk, rmsd, protein_fn, ligand_fn, vina = ln.split()
                    ligand_raw_fn = ligand_fn[:ligand_fn.rfind('.')]
                    affinity_info[ligand_raw_fn] = {
                        'label': float(label),
                        'rmsd': float(rmsd),
                        'pk': float(pk),
                        'vina': float(vina[1:])
                    }
            # save affinity info
            with open(self.affinity_path, 'wb') as f:
                pickle.dump(affinity_info, f)
        
        self.affinity_info = affinity_info

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

        # Build filename to key mapping for tuple-based indexing (with caching)
        cache_path = self.processed_path + '_filename_mapping.pkl'
        if os.path.exists(cache_path):
            # Load cached mapping
            with open(cache_path, 'rb') as f:
                self._filename_to_key = pickle.load(f)
        else:
            # Build mapping and cache it
            self._filename_to_key = {}
            with self.db.begin() as txn:
                for key in tqdm(self.keys, desc='Building filename mapping'):
                    try:
                        data = pickle.loads(txn.get(key))
                        if 'protein_filename' in data and 'ligand_filename' in data:
                            filename_tuple = (data['protein_filename'], data['ligand_filename'])
                            self._filename_to_key[filename_tuple] = key
                    except:
                        continue
            # Save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(self._filename_to_key, f)

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    data_prefix = self.raw_path
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except Exception as e:
                    num_skipped += 1
                    print('Skipping (%d) %s - Error: %s' % (num_skipped, ligand_fn, str(e)))
                    if num_skipped < 5:  # Only print full traceback for first few errors
                        import traceback
                        traceback.print_exc()
                    continue
        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def _update(self, sid, affinity):
        if self.db is None:
            self._connect_db()
        txn = self.db.begin(write=True)
        data = pickle.loads(txn.get(sid))
        data.update({
            'affinity': affinity['vina'],
            'rmsd': affinity['rmsd'],
            'pk': affinity['pk'],
            'rmsd<2': affinity['label']
        })
        txn.put(
            key=sid,
            value=pickle.dumps(data)
        )
        txn.commit()

    def _inject_affinity(self, sid, ligand_path):
        if ligand_path[:-4] in self.affinity_info:
            affinity = self.affinity_info[ligand_path[:-4]]
            self._update(sid, affinity)
        else:
            raise AttributeError(f'affinity_info has no {ligand_path[:-4]}')

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()

        # Handle case where idx is a tuple of (protein_filename, ligand_filename)
        if isinstance(idx, tuple) and len(idx) == 2 and isinstance(idx[0], str):
            # This is a filename tuple from split file
            if idx in self._filename_to_key:
                key = self._filename_to_key[idx]
                original_idx = self.keys.index(key) if key in self.keys else 0
            else:
                # Sample not found in LMDB - warn once and use first sample as fallback
                if idx not in self._warned_missing:
                    import warnings
                    warnings.warn(f"Sample {idx[0][:50]}... not found in LMDB. Using first available sample as fallback.")
                    self._warned_missing.add(idx)
                # Use first available sample
                key = self.keys[0]
                original_idx = 0
        # Handle case where idx is a tuple (extract first element)
        elif isinstance(idx, tuple):
            idx = idx[0]
            # Recursively handle the extracted element
            return self.get_ori_data(idx)
        # Handle case where idx is a string - try to convert to int first
        elif isinstance(idx, str):
            try:
                idx = int(idx)
            except ValueError:
                # If conversion fails, treat it as a key
                key = idx.encode()
                original_idx = self.keys.index(key) if key in self.keys else 0
        # Handle case where idx is bytes (direct key)
        elif isinstance(idx, bytes):
            key = idx
            original_idx = self.keys.index(key) if key in self.keys else 0
        # Handle case where idx is an integer
        elif isinstance(idx, int):
            if idx >= len(self.keys):
                raise IndexError(f"Index {idx} is out of range for dataset with {len(self.keys)} samples")
            key = self.keys[idx]
            original_idx = idx
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

        data = pickle.loads(self.db.begin().get(key))
        if 'affinity' not in data:
            self._load_affinity_info()
            self._inject_affinity(key, data['ligand_filename'])
            data = pickle.loads(self.db.begin().get(key))

        data = ProteinLigandData(**data)
        data.id = original_idx
        assert data.protein_pos.size(0) > 0
        return data


class PDBPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.transform = transform
        self.db = None
        self.keys = None
        
        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
            
    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))



    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, protein_fn, (pka, year, resl), ligand_fn, pdbid) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    data_prefix = self.raw_path
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data.affinity = pka
                    data = data.to_dict()  # avoid torch_geometric version issue
                    assert data['protein_pos'].size(0) > 0
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except Exception as e:
                    num_skipped += 1
                    print('Skipping (%d) %s - Error: %s' % (num_skipped, ligand_fn, str(e)))
                    if num_skipped < 5:  # Only print full traceback for first few errors
                        import traceback
                        traceback.print_exc()
                    continue
        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        # Handle case where idx is a tuple (extract first element)
        if isinstance(idx, tuple):
            idx = idx[0]

        # Handle case where idx is a string - try to convert to int first
        if isinstance(idx, str):
            try:
                idx = int(idx)
            except ValueError:
                # If conversion fails, treat it as a key
                key = idx.encode()
                original_idx = self.keys.index(key) if key in self.keys else 0

        # Handle case where idx is bytes (direct key)
        if isinstance(idx, bytes):
            key = idx
            original_idx = self.keys.index(key) if key in self.keys else 0
        elif isinstance(idx, int):
            # idx is an integer index
            if idx >= len(self.keys):
                raise IndexError(f"Index {idx} is out of range for dataset with {len(self.keys)} samples")
            key = self.keys[idx]
            original_idx = idx

        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = original_idx
        assert data.protein_pos.size(0) > 0
        return data

if __name__ == '__main__':

    dataset = PDBPairDataset('./data/pdbbind2020/')
    print(len(dataset), dataset[0])
