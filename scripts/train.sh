#!/bin/bash
# Training wrapper for TheSelective
CONFIG=${1:-"./configs/training.yml"}
python scripts/train_diffusion.py --config "$CONFIG" --tag my_experiment
