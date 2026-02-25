# TheSelective: Dual Affinity-Guided Diffusion for Selective Molecule Generation

Official implementation of **TheSelective**, a dual-head diffusion model for generating molecules with high selectivity towards target proteins.

## Installation

### Requirements
- Python 3.9
- CUDA 11.3+
- 40GB+ GPU memory for training

### Setup Environment

```bash
conda create -n theselective python=3.9
conda activate theselective
```

#### Step 1: Install PyTorch (GPU-dependent)

PyTorch must match your CUDA version. Check your CUDA version first:
```bash
nvcc --version
```

Then visit **[PyTorch Get Started](https://pytorch.org/get-started/locally/)** and select the correct configuration.

Example for CUDA 11.3:
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

#### Step 2: Install PyTorch Geometric (GPU-dependent)

PyG must match your PyTorch + CUDA combination. Visit **[PyG Installation](https://data.pyg.org/whl/)** for compatible wheels.

Example for PyTorch 1.11.0 + CUDA 11.3:
```bash
conda install pytorch-scatter pytorch-cluster pytorch-sparse==0.6.13 pyg==2.0.4 -c pyg
```

#### Step 3: Install remaining dependencies

```bash
# Core dependencies
pip install pyyaml easydict lmdb numpy==1.21.6 pandas==1.4.1
pip install tensorboard==2.9.0 Pillow==9.0.1 scipy==1.7.3

# Molecular processing tools
conda install -c conda-forge openbabel
pip install meeko==0.1.dev3 vina==1.2.2 pdb2pqr rdkit

# AutoDockTools
# Linux:
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
# Windows:
python.exe -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

# Other dependencies
pip install -r requirements.txt
```

## Data Preparation

Download the following and extract to `./data/`:

| File | Description | Link |
|------|-------------|------|
| data.zip | LMDB dataset + split file + CrossDocked pocket structures + test_set.zip | [Google Drive](https://drive.google.com/file/d/1YlPio7GMjS95Ca827rHEy0GXkVuvhSBd/view?usp=drive_link) |
| tmscore_extreme_pairs.txt | TM-score pair list for evaluation | [Google Drive](https://drive.google.com/file/d/1nFYCJDvTAhA1EwTc2ZyexX47V1x4HBMX/view?usp=sharing) |

> **Note:** `data.zip` includes `test_set.zip` inside it. After extracting `data.zip`, also extract `test_set.zip` into `./data/test_set/`. This directory contains the original full receptor PDB files (e.g., `4xli_B_rec.pdb`) and corresponding ligand files needed by the docking pipeline.
## Overall Project Structure

```
TheSelective/
├── analysis/
│   ├── analyze_tmscore_high_filtered.py  # High TM-score analysis
│   └── analyze_tmscore_low_filtered.py   # Low TM-score analysis
├── checkpoints/
│   └──theselective.pt
├── configs/
│   ├── training.yml              # Training config (bidirectional_query_atom)
│   └── sampling.yml              # Sampling config
├── data/
│   ├── test_set/                 # Original receptor PDB + ligand files (for docking)
│   ├── crossdocked_pocket10_pose_split.pt           # Train/val/test split
│   ├── crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb   # Training/Validation/Test LMDB
│   └── tmscore_extreme_pairs.txt        # TM-score pair evaluation list
├── datasets/
│   ├── __init__.py
│   ├── pl_data.py
│   └── pl_pair_dataset.py
├── models/
│   ├── __init__.py
│   ├── molopt_score_model.py     # Main dual-head diffusion model
│   ├── uni_transformer.py        # SE(3)-equivariant transformer with selective edges
│   └── common.py                 # Shared components (selective graph utils)
├── results/
├── scripts/
│   ├── data_preparation/
│   ├── __init__.py
│   ├── train_diffusion.py        # Training script
│   ├── sample_diffusion.py       # Generation with guidance
│   ├── dock_generated_ligands.py # Docking evaluation (with QED, SA, Validity)
│   ├── train.sh                  # Training wrapper
│   └── run_theselective.sh       # Full pipeline (gen + dock)
├── utils/
│   ├── evaluation/               # Evaluation metrics
│   └── ...                       # Utility functions
├── README.md
├── setup.py
├── json_to_txt_converter.py
├── requirements.txt
└── .gitignore
```
## Training

```bash
python scripts/train_diffusion.py --config configs/training.yml
```

Or use the wrapper script:

```bash
bash scripts/train.sh
```
## Model Checkpoints

Download the pre-trained model and place it in the `checkpoints/` directory:

```bash
mkdir -p checkpoints
# Download theselective.pt from the link below and place in checkpoints/
```

| Model | Checkpoint | Download | Description |
|-------|------------|----------|-------------|
| TheSelective | `checkpoints/theselective.pt` | [Google Drive](https://drive.google.com/file/d/1Fr2nK1Yky-LWzJ2o_05D0nUgXuXtQxo7/view?usp=drive_link) | Bidirectional cross-attention (675k iterations) |

> **Note:** Update the checkpoint path in `configs/sampling.yml` if you use a different location.


## Generation

### Selectivity-Guided Generation

Generate molecules with on-target/off-target selectivity:

```bash
python scripts/sample_diffusion.py \
    --ckpt ./checkpoints/theselective.pt \
    --data_path ./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb \
    --split_path ./data/crossdocked_pocket10_pose_split.pt \
    --data_id 0 \
    --off_target_id 96 \
    --guide_mode head1_head2_staged \
    --w_on 2.0 \
    --w_off 1.0 \
    --head1_type_grad_weight 100 \
    --head1_pos_grad_weight 25 \
    --head2_type_grad_weight 100 \
    --head2_pos_grad_weight 25 \
    --batch_size 4 \
    --result_path ./results/theselective/id0_96_high
```

### Key Generation Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--data_path` | Path to LMDB dataset (overrides checkpoint config) | `./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb` |
| `--split_path` | Path to train/val/test split file (overrides checkpoint config) | `./data/crossdocked_pocket10_pose_split.pt` |
| `--guide_mode` | Selectivity guidance strategy (Scheduled) | `head1_head2_staged` |
| `--w_on` | On-target weight (higher = stronger binding) | 2.0 |
| `--w_off` | Off-target weight (higher = weaker binding) | 1.0 |
| `--head1_type_grad_weight` | Head1 atom type gradient | 100 |
| `--head1_pos_grad_weight` | Head1 position gradient | 25 |
| `--head2_type_grad_weight` | Head2 atom type gradient | 100 |
| `--head2_pos_grad_weight` | Head2 position gradient | 25 |

## Evaluation

### Docking Evaluation

Evaluate generated molecules with AutoDock Vina:

```bash
python scripts/dock_generated_ligands.py \
    --use_lmdb_only \
    --mode id_specific \
    --sample_path ./results/theselective/id0_96_high \
    --output_dir ./results/theselective/id0_96_high/docking_results \
    --on_target_id 0 \
    --off_target_ids 96 \
    --docking_mode vina_dock \
    --exhaustiveness 8 \
    --save_visualization
```

### TM-Score Pair Evaluation

Run the full evaluation pipeline on all TM-score pairs (high/low structural similarity):

```bash
bash scripts/run_theselective.sh
```

### Result Analysis

Pre-computed results from the paper: [Google Drive](https://drive.google.com/file/d/1HCO8HG1zq16hwNar54F3CEbUaEj0cqr7/view?usp=sharing)

The Main Table & Ablation Table Results are in:[TM-High](https://drive.google.com/file/d/1BULQcQktdzPSpjc1zo7dCqEWuh7V3hfL/view?usp=drive_link)
[TM-Low](https://drive.google.com/file/d/1yO8s-ldOVwGTgXyX4w_TgtPihhlsmgbT/view?usp=drive_link)
```bash
# Analyze HIGH TM-score pairs (structurally similar proteins)
python analysis/analyze_tmscore_high_filtered.py

# Analyze LOW TM-score pairs (structurally different proteins)
python analysis/analyze_tmscore_low_filtered.py
```


## Acknowledgments

This work builds upon:
- [KGDiff](https://github.com/CMACH508/KGDiff)
- [TargetDiff](https://github.com/guanjq/targetdiff)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
