# TheSelective: Dual-Head Diffusion for Selective Molecule Generation

Official implementation of **TheSelective**, a dual-head diffusion model for generating molecules with high selectivity towards target proteins while minimizing off-target binding. The model uses a bidirectional cross-attention mechanism (`bidirectional_query_atom`) to capture both protein-to-ligand and ligand-to-protein interaction patterns.

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

data.zip :
https://drive.google.com/file/d/1YlPio7GMjS95Ca827rHEy0GXkVuvhSBd/view?usp=drive_link

tmscore_extreme_pairs.txt :
https://drive.google.com/file/d/1nFYCJDvTAhA1EwTc2ZyexX47V1x4HBMX/view?usp=sharing

Download datasets and place them in `./data/`:

```
data/
├── crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb
├── crossdocked_pocket10_pose_split.pt
└── tmscore_extreme_pairs.txt  # For TM-score pair evaluation
```

## Training

```bash
python scripts/train_diffusion.py --config configs/training.yml --tag my_experiment
```

Or use the wrapper script:

```bash
bash scripts/train.sh
```
## Model Checkpoints

Pre-trained model - Download file and replace pt file directory in sampling.yml :
https://drive.google.com/file/d/1Fr2nK1Yky-LWzJ2o_05D0nUgXuXtQxo7/view?usp=drive_link
| Model | Checkpoint | Description |
|-------|------------|-------------|
| TheSelective | `checkpoints/theselective.pt` | Bidirectional cross-attention (675k iterations) |


## Generation

### Selectivity-Guided Generation

Generate molecules with on-target/off-target selectivity:

```bash
python scripts/sample_diffusion.py \
    --ckpt ./checkpoints/theselective.pt \
    --data_id 0 \
    --off_target_id 50 \
    --guide_mode head1_head2_staged \
    --w_on 2.0 \
    --w_off 1.0 \
    --head1_type_grad_weight 100 \
    --head1_pos_grad_weight 25 \
    --head2_type_grad_weight 100 \
    --head2_pos_grad_weight 25 \
    --batch_size 4 \
    --result_path ./results/theselective_id0_50
```

### Key Generation Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
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
    --sample_path ./results/theselective_id0_50 \
    --output_dir ./results/theselective_id0_50/docking_results \
    --on_target_id 0 \
    --off_target_ids 50 \
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

results in paper : [https://drive.google.com/file/d/13p3URTI3nps-TdV3aALRGPFHYIYoeaEj/view?usp=drive_link]

```bash
# Analyze HIGH TM-score pairs (structurally similar proteins)
python analysis/analyze_tmscore_high_filtered.py

# Analyze LOW TM-score pairs (structurally different proteins)
python analysis/analyze_tmscore_low_filtered.py
```



## Project Structure

```
TheSelective/
├── configs/
│   ├── training.yml              # Training config (bidirectional_query_atom)
│   └── sampling.yml              # Sampling config
├── models/
│   ├── __init__.py
│   ├── molopt_score_model.py     # Main dual-head diffusion model
│   ├── uni_transformer.py        # SE(3)-equivariant transformer with selective edges
│   └── common.py                 # Shared components (selective graph utils)
├── scripts/
│   ├── __init__.py
│   ├── train_diffusion.py        # Training script
│   ├── sample_diffusion.py       # Generation with guidance
│   ├── evaluate_diffusion.py     # Evaluation metrics
│   ├── sample_for_pocket.py      # Single pocket generation
│   ├── dock_generated_ligands.py # Docking evaluation
│   ├── train.sh                  # Training wrapper
│   └── run_theselective.sh       # Full pipeline (gen + dock)
├── analysis/
│   ├── analyze_tmscore_high_filtered.py  # High TM-score analysis
│   └── analyze_tmscore_low_filtered.py   # Low TM-score analysis
├── datasets/
│   ├── __init__.py
│   ├── pl_data.py
│   └── pl_pair_dataset.py
├── utils/
│   ├── evaluation/               # Evaluation metrics
│   └── ...                       # Utility functions
├── README.md
├── setup.py
├── requirements.txt
└── .gitignore
```

## Acknowledgments

This work builds upon:
- [KGDiff](https://github.com/CMACH508/KGDiff)
- [TargetDiff](https://github.com/guanjq/targetdiff)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
