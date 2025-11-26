# TheSelective: Dual-Head Architecture for Selective Molecule Generation

Official implementation of **TheSelective**, a dual-head diffusion model for generating molecules with high selectivity towards target proteins while minimizing off-target binding.

We provide two variants:
- **TheSelective-A**: Atom-centric cross-attention architecture
- **TheSelective-G**: Global cross-attention architecture (1p_all)

## Key Innovation

Unlike existing target-aware molecule generation methods that only consider on-target binding affinity, **TheSelective** introduces a **dual-head architecture** that simultaneously optimizes:

1. **Head 1 (On-Target)**: Maximize binding affinity with protein-ligand interactions
2. **Head 2 (Off-Target)**: Minimize binding affinity without protein-ligand interactions

This enables **selectivity-guided generation**, producing molecules with high specificity.

### TheSelective-A vs TheSelective-G

| Feature | TheSelective-A | TheSelective-G |
|---------|----------------|----------------|
| Cross-Attention | Atom-centric | Global (1p_all) |
| Config Flag | `use_dual_head_sam_pl: True` | `use_dual_head_sam_pl: True` + `use_atom_level_cross_attn: True` |
| Focus | Individual atom interactions | Global protein-ligand patterns |

## Installation

### Requirements
- Python 3.9
- CUDA 11.3+
- 40GB+ GPU memory for training

### Setup Environment

```bash
conda create -n theselective python=3.9
conda activate theselective

# Install PyTorch
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch

# Install PyTorch Geometric
conda install pytorch-scatter pytorch-cluster pytorch-sparse==0.6.13 pyg==2.0.4 -c pyg

# Install dependencies
pip install pyyaml easydict lmdb numpy==1.21.6 pandas==1.4.1
pip install tensorboard==2.9.0 Pillow==9.0.1 scipy==1.7.3

# Install molecular processing tools
conda install -c conda-forge openbabel
pip install meeko==0.1.dev3 vina==1.2.2 pdb2pqr rdkit
```

## Data Preparation
[https://drive.google.com/file/d/1YlPio7GMjS95Ca827rHEy0GXkVuvhSBd/view?usp=drive_link]
Download datasets and place them in `./data/`:

```
data/
├── crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb
├── crossdocked_pocket10_pose_split.pt
└── tmscore_extreme_pairs.txt  # For TM-score pair evaluation
```

## Training

### Train TheSelective-A (Atom-centric)

```bash
python scripts/train_diffusion.py --config configs/training_theselective_a.yml
```

### Train TheSelective-G (Global)

```bash
python scripts/train_diffusion.py --config configs/training_theselective_g.yml
```

## Generation

### Basic Selectivity-Guided Generation

Generate molecules with on-target/off-target selectivity:

```bash
# TheSelective-A
python scripts/sample_diffusion.py \
    --config configs/sampling_theselective_a.yml \
    --use_lmdb_only \
    --data_id 0 \
    --off_target_ids 50 \
    --guide_mode joint_on_off_no_interaction_sequential \
    --w_on 2.0 \
    --w_off 1.0 \
    --head1_type_grad_weight 100 \
    --head1_pos_grad_weight 25 \
    --head2_type_grad_weight 100 \
    --head2_pos_grad_weight 0.0 \
    --batch_size 4 \
    --result_path ./results/theselective_a_id0_50

# TheSelective-G
python scripts/sample_diffusion.py \
    --config configs/sampling_theselective_g.yml \
    --use_lmdb_only \
    --data_id 0 \
    --off_target_ids 50 \
    --guide_mode joint_on_off_no_interaction_sequential \
    --w_on 2.0 \
    --w_off 1.0 \
    --head1_type_grad_weight 100 \
    --head1_pos_grad_weight 25 \
    --head2_type_grad_weight 100 \
    --head2_pos_grad_weight 0.0 \
    --batch_size 4 \
    --result_path ./results/theselective_g_id0_50
```

### Key Generation Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--guide_mode` | Selectivity guidance strategy | `joint_on_off_no_interaction_sequential` |
| `--w_on` | On-target weight (higher = stronger binding) | 2.0 |
| `--w_off` | Off-target weight (higher = weaker binding) | 1.0 |
| `--head1_type_grad_weight` | Head1 atom type gradient | 100 |
| `--head1_pos_grad_weight` | Head1 position gradient | 25 |
| `--head2_type_grad_weight` | Head2 atom type gradient | 100 |
| `--head2_pos_grad_weight` | Head2 position gradient | 0.0 |

## Evaluation

### Docking Evaluation

Evaluate generated molecules with AutoDock Vina:

```bash
python scripts/dock_generated_ligands.py \
    --use_lmdb_only \
    --mode id_specific \
    --sample_path ./results/theselective_a_id0_50 \
    --output_dir ./results/theselective_a_id0_50/docking_results \
    --on_target_id 0 \
    --off_target_ids 50 \
    --docking_mode vina_dock \
    --exhaustiveness 8 \
    --save_visualization
```

### TM-Score Pair Evaluation

Run evaluation on all TM-score pairs (high/low structural similarity):

```bash
# TheSelective-A
bash scripts/run_theselective_a.sh

# TheSelective-G
bash scripts/run_theselective_g.sh
```

### Result Analysis
[https://drive.google.com/file/d/13p3URTI3nps-TdV3aALRGPFHYIYoeaEj/view?usp=drive_link]
```bash
# Analyze HIGH TM-score pairs (structurally similar proteins)
python analysis/analyze_tmscore_pairs_high_fixed.py

# Analyze LOW TM-score pairs (structurally different proteins)
python analysis/analyze_tmscore_pairs_low_fixed.py

# Generate scatter plots for noise robustness
python analysis/evaluate_noise_robustness.py
```

## Model Checkpoints

Pre-trained models (will be available upon acceptance):

| Model | Checkpoint | Description |
|-------|------------|-------------|
| TheSelective-A | `checkpoints/theselective_a.pt` | Atom-centric (844k iterations) |
| TheSelective-G | `checkpoints/theselective_g.pt` | Global (810k iterations) |

## Project Structure

```
TheSelective/
├── configs/
│   ├── training_theselective_a.yml    # Training config for TheSelective-A
│   ├── training_theselective_g.yml    # Training config for TheSelective-G
│   ├── sampling_theselective_a.yml    # Sampling config for TheSelective-A
│   └── sampling_theselective_g.yml    # Sampling config for TheSelective-G
├── models/
│   ├── molopt_score_model.py          # Main dual-head model
│   ├── uni_transformer.py             # UniTransformer backbone
│   └── common.py                      # Shared components
├── scripts/
│   ├── train_diffusion.py             # Training script
│   ├── sample_diffusion.py            # Generation script
│   ├── dock_generated_ligands.py      # Docking evaluation
│   ├── run_theselective_a.sh          # Automation for TheSelective-A
│   └── run_theselective_g.sh          # Automation for TheSelective-G
├── analysis/
│   ├── evaluate_noise_robustness.py   # Scatter plot generation
│   ├── analyze_tmscore_pairs_high_fixed.py  # High TM-score analysis
│   └── analyze_tmscore_pairs_low_fixed.py   # Low TM-score analysis
├── datasets/
│   ├── pl_data.py                     # Protein-ligand data processing
│   └── pl_pair_dataset.py             # Multi-protein pair dataset
└── utils/
    ├── evaluation/                    # Evaluation metrics
    └── ...                            # Utility functions
```

## Acknowledgments

This work builds upon:
- [KGDiff](https://github.com/CMACH508/KGDiff)
- [TargetDiff](https://github.com/guanjq/targetdiff)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
