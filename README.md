# TheSelective: Dual Affinity-Guided Diffusion for Selective Molecular Generation

[![Paper](https://img.shields.io/badge/PAKDD%202026-Paper-1d4ed8.svg)](https://doi.org/10.1007/978-981-92-1462-4_2)
[![License](https://img.shields.io/badge/Code%20License-MIT-047857.svg)](LICENSE)

Official implementation of **TheSelective** (PAKDD 2026), a diffusion framework that generates 3D ligands optimized for *selectivity* — maximizing the binding affinity gap between an intended target and a potential off-target.
![image](https://drive.google.com/file/d/1H0RMWOYBvJTVHDk7TshVZCmNZPtuVqy0/view?usp=drive_link)

---

## Results

Evaluated on **CrossDocked2020** (~100k training pairs, 100 test proteins; RMSD < 1Å, sequence identity < 30% between splits). Each test pocket is paired with a structurally similar protein (**TM-High**) or a geometrically distinct one (**TM-Low**).

`Selectivity = Off-Dock − On-Dock` (kcal/mol). Higher is better.
**Avg.** = mean of per-pair average scores, **Med.** = mean of per-pair median scores.

### TM-High — structurally similar off-targets

| Model | On-Dock ↓ | Off-Dock ↑ | Selectivity ↑ | QED ↑ | SA ↑ | Success |
|---|---|---|---|---|---|---|
| Reference | -7.400 / -7.179 | -7.168 / -7.056 | 0.231 / 0.083 | 0.482 / 0.469 | 0.736 / 0.745 | 98% |
| BInD | -7.510 / -7.569 | **-7.391 / -7.422** | 0.120 / 0.062 | 0.505 / 0.503 | 0.658 / 0.661 | 88.1% |
| TargetDiff | -7.583 / -7.577 | -7.405 / -7.391 | 0.178 / 0.165 | 0.469 / 0.466 | 0.585 / 0.591 | 91.0% |
| KGDiff | -9.290 / -9.309 | -8.446 / -8.466 | 0.844 / 0.727 | **0.527 / 0.537** | 0.548 / 0.550 | 85.6% |
| **TheSelective** | **-9.969 / -9.958** | -8.994 / -9.001 | **0.975 / 0.923** | 0.495 / 0.500 | 0.534 / 0.535 | 50.2% |

### TM-Low — structurally dissimilar off-targets

| Model | On-Dock ↓ | Off-Dock ↑ | Selectivity ↑ | QED ↑ | SA ↑ | Success |
|---|---|---|---|---|---|---|
| Reference | -7.451 / -7.293 | -5.414 / -5.424 | 2.037 / 1.869 | 0.475 / 0.469 | 0.728 / 0.740 | 97% |
| BInD | -7.536 / -7.589 | -5.608 / -5.643 | 1.928 / 1.934 | 0.502 / 0.500 | 0.654 / 0.656 | 89.1% |
| TargetDiff | -7.566 / -7.552 | -5.567 / -5.564 | 1.999 / 1.993 | 0.467 / 0.464 | 0.583 / 0.591 | 91.9% |
| KGDiff | -9.343 / -9.366 | -6.345 / -6.393 | 2.998 / 2.980 | **0.528 / 0.538** | 0.546 / 0.549 | 85.6% |
| **TheSelective** | **-9.954 / -9.909** | **-6.591 / -6.588** | **3.363 / 3.259** | 0.511 / 0.514 | 0.557 / 0.561 | 56.9% |

### Ablation — guidance configuration

| Configuration | TM-High Selectivity ↑ | TM-Low Selectivity ↑ |
|---|---|---|
| No Guide | 0.165 / 0.112 | 2.121 / 2.064 |
| On-Target Guide only | 0.636 / 0.536 | 2.843 / 2.756 |
| Off-Target Guide only | 0.036 / 0.047 | 2.251 / 2.216 |
| Dual Guide (unscheduled) | 0.628 / 0.588 | 2.953 / 2.890 |
| **Dual Guide, Scheduled** | **0.975 / 0.923** | **3.363 / 3.259** |

---

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

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.11.0+cu113.html --no-index

pip install torch-geometric
```

#### Step 3: Install remaining dependencies

```bash
# Core dependencies
# (Note: Versions are strictly pinned to prevent C-API, TensorBoard, and Pillow compatibility issues)
pip install pyyaml easydict lmdb pandas==1.4.1 tensorboard==2.9.0
pip install "numpy>=1.23.0,<1.24.0" "protobuf<=3.20.3" "Pillow>=9.1.0"

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
| data.zip | LMDB dataset + split file + CrossDocked pocket structures + test_set.zip | [Google Drive](https://drive.google.com/file/d/1TbEoMAgx4oOuiTIMgEtSv-MSqJcZEcuV/view?usp=sharing) |
| tmscore_extreme_pairs.txt | TM-score pair list for evaluation | [Google Drive](https://drive.google.com/file/d/1o8LRdtvf9RZcaiRPP85JYkiUclrKLqDR/view?usp=sharing) |

> **Note:** `data.zip` includes `test_set.zip` inside it. After extracting `data.zip`, also extract `test_set.zip` into `./data/test_set/`. This directory contains the original full receptor PDB files (e.g., `4xli_B_rec.pdb`) and corresponding ligand files needed by the docking pipeline.

## Overall Project Structure

```
TheSelective/
├── analysis/
│   ├── analyze_tmscore_high_filtered.py  # High TM-score analysis
│   └── analyze_tmscore_low_filtered.py   # Low TM-score analysis
├── checkpoints/
│   └── theselective.pt
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
| TheSelective | `checkpoints/theselective.pt` | [Google Drive](https://drive.google.com/file/d/1yIPyyngMQChx4ZveNCxJM11JFYp4ktdz/view?usp=sharing) | Bidirectional cross-attention (675k iterations) |

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

Generated molecules and docking results from the paper: [Google Drive](https://drive.google.com/file/d/1nbUjIS_I1HQzeJPkHPvZqEr6SAoSG-co/view?usp=sharing)

Full main-table and ablation-table results (with baselines):
[TM-High](https://drive.google.com/file/d/1oV4pkHLvI8BgNEZeNnnSi-gu3RMTXXAo/view?usp=sharing) ·
[TM-Low](https://drive.google.com/file/d/1uhDwfm75fAH8WY2oNJKUiRYCP0GyoFtl/view?usp=sharing)

```bash
# Analyze HIGH TM-score pairs (structurally similar proteins)
python analysis/analyze_tmscore_high_filtered.py

# Analyze LOW TM-score pairs (structurally different proteins)
python analysis/analyze_tmscore_low_filtered.py
```

## Troubleshooting

If you encounter a `WeightsUnpickler` error (e.g., `Unsupported global: GLOBAL easydict.EasyDict` or numpy arrays) when loading checkpoints or `.pt` result files, add `weights_only=False` to the `torch.load()` calls in your scripts (e.g., `sample_diffusion.py`, `dock_generated_ligands.py`):

```python
ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
```

---

## License and Patent Notice

### Patent

This work was supported by the National Research Foundation of Korea (NRF), funded by the Ministry of Science and ICT (No. RS-2023-00229822).

| | |
|---|---|
| **Application No.** | KR 10-2026-0132757 (filed 2026-07-20) |
| **Priority** | KR 10-2025-0188859 (filed 2025-12-03) |
| **Title** | Method and system for generating selective ligand |
| **Applicant / Assignee** | Yonsei University Industry–Academic Cooperation Foundation |
| **Inventors** | Sanghyun Park, Hyoungjoon Park |

### Source code — MIT License

The source code in this repository is released under the **MIT License**. See [`LICENSE`](LICENSE) for the full text.

### Third-party code

This repository incorporates code from projects released under the MIT License. Their original copyright notices are retained in the corresponding source files:

- KGDiff — Copyright (c) 2023 CMACH508
- TargetDiff — Copyright (c) 2023 Jiaqi Guan

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{park2026theselective,
  title     = {TheSelective: Dual Affinity-Guided Diffusion for Selective Molecular Generation},
  author    = {Park, Hyoungjoon and Kim, Hwanhee and Choi, Seungyeon and
               Lee, Seungyong and Kim, Yoonju and Park, Sanghyun},
  booktitle = {Advances in Knowledge Discovery and Data Mining (PAKDD 2026)},
  series    = {Lecture Notes in Artificial Intelligence},
  volume    = {16598},
  pages     = {16--27},
  year      = {2026},
  publisher = {Springer},
  doi       = {10.1007/978-981-92-1462-4_2}
}
```

## Acknowledgments

This work builds upon:
- [KGDiff](https://github.com/CMACH508/KGDiff) (MIT License)
- [TargetDiff](https://github.com/guanjq/targetdiff) (MIT License)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) (MIT License)

## Contact

**Hyoungjoon Park** — ktori1361@yonsei.ac.kr
[ORCID: 0009-0003-0721-1716](https://orcid.org/0009-0003-0721-1716) ·
[Homepage](https://dannyjpark.github.io/) ·
[Google Scholar](https://scholar.google.com/citations?user=TJPGDTUAAAAJ)

Data Engineering LAB, Department of Computer Science, Yonsei University
Advisor: Prof. Sanghyun Park

For licensing inquiries, please contact the Yonsei University Industry–Academic Cooperation Foundation.
