#!/bin/bash
# TheSelective: Full pipeline (generation + docking) for TM-score pair evaluation
#
# Usage:
#   bash scripts/run_theselective.sh
#
# Prerequisites:
#   - Trained checkpoint
#   - ./data/tmscore_extreme_pairs.txt
#   - Data files in ./data/ (see README for download links)

set -e

# GPU assignment
if [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# Model configuration
CKPT_PATH="${CKPT_PATH:-./checkpoints/theselective.pt}"
GUIDE_MODE="head1_head2_staged"

# Gradient weights
HEAD1_TYPE_WEIGHT=100
HEAD1_POS_WEIGHT=25
HEAD2_TYPE_WEIGHT=100
HEAD2_POS_WEIGHT=25
W_ON=2.0
W_OFF=1.0

# Sampling settings
BATCH_SIZE=4
NUM_SAMPLES=8

# Result path
BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/theselective}"

echo "========================================================================"
echo "TheSelective: TM-score Pair Evaluation"
echo "========================================================================"
echo "Checkpoint: $CKPT_PATH"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Guide Mode: $GUIDE_MODE"
echo "Head1 Weights: type=$HEAD1_TYPE_WEIGHT, pos=$HEAD1_POS_WEIGHT"
echo "Head2 Weights: type=$HEAD2_TYPE_WEIGHT, pos=$HEAD2_POS_WEIGHT"
echo "W_ON=$W_ON, W_OFF=$W_OFF"
echo "Results: $BASE_RESULT_PATH"
echo "Start time: $(date)"
echo ""

# Check TM-score pairs file
PAIRS_FILE="${PAIRS_FILE:-./data/tmscore_extreme_pairs.txt}"
if [ ! -f "$PAIRS_FILE" ]; then
    echo "ERROR: $PAIRS_FILE not found!"
    exit 1
fi

# Create log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./run_theselective_log_${TIMESTAMP}.txt"

{
    echo "========================================================================"
    echo "TheSelective TM-SCORE PAIRS EVALUATION"
    echo "========================================================================"
    echo "Timestamp: $(date)"
    echo "Checkpoint: $CKPT_PATH"
    echo "Gradient Weights: head1_type=$HEAD1_TYPE_WEIGHT, head1_pos=$HEAD1_POS_WEIGHT"
    echo "                  head2_type=$HEAD2_TYPE_WEIGHT, head2_pos=$HEAD2_POS_WEIGHT"
    echo "Selectivity Weights: w_on=$W_ON, w_off=$W_OFF"
    echo "Results: $BASE_RESULT_PATH"
    echo "========================================================================"
    echo ""
} > "$LOG_FILE"

# =======================================================================
# PHASE 1: GENERATION (GPU)
# =======================================================================
echo ""
echo "========================================================================"
echo "PHASE 1: MOLECULE GENERATION (GPU)"
echo "========================================================================"
echo ""

TOTAL_IDS=100
PROCESSED=0

while IFS=',' read -r target_id high_off_id high_score low_off_id low_score; do

    PROCESSED=$((PROCESSED + 1))

    echo ""
    echo "# Generating for ID $target_id ($PROCESSED/$TOTAL_IDS)"
    echo "  HIGH TM-score: ($target_id, $high_off_id) - Score: $high_score"
    echo "  LOW TM-score:  ($target_id, $low_off_id) - Score: $low_score"

    # ===== Generate for HIGHEST TM-score pair =====
    RESULT_PATH_HIGH="${BASE_RESULT_PATH}/id${target_id}_${high_off_id}_high"

    if [ -d "$RESULT_PATH_HIGH" ]; then
        rm -rf "$RESULT_PATH_HIGH"
    fi

    echo "  [HIGH] Generating molecules for ($target_id, $high_off_id)..."
    if python scripts/sample_diffusion.py \
        --ckpt "$CKPT_PATH" \
        --data_id $target_id \
        --off_target_id $high_off_id \
        --guide_mode "$GUIDE_MODE" \
        --head1_type_grad_weight $HEAD1_TYPE_WEIGHT \
        --head1_pos_grad_weight $HEAD1_POS_WEIGHT \
        --head2_type_grad_weight $HEAD2_TYPE_WEIGHT \
        --head2_pos_grad_weight $HEAD2_POS_WEIGHT \
        --w_on $W_ON \
        --w_off $W_OFF \
        --batch_size $BATCH_SIZE \
        --num_samples $NUM_SAMPLES \
        --result_path "$RESULT_PATH_HIGH" 2>&1 | tee -a "$LOG_FILE"; then
        echo "    HIGH Generation: SUCCESS" >> "$LOG_FILE"
    else
        echo "    HIGH Generation: FAILED" >> "$LOG_FILE"
    fi

    # ===== Generate for LOWEST TM-score pair =====
    RESULT_PATH_LOW="${BASE_RESULT_PATH}/id${target_id}_${low_off_id}_low"

    if [ -d "$RESULT_PATH_LOW" ]; then
        rm -rf "$RESULT_PATH_LOW"
    fi

    echo "  [LOW] Generating molecules for ($target_id, $low_off_id)..."
    if python scripts/sample_diffusion.py \
        --ckpt "$CKPT_PATH" \
        --data_id $target_id \
        --off_target_id $low_off_id \
        --guide_mode "$GUIDE_MODE" \
        --head1_type_grad_weight $HEAD1_TYPE_WEIGHT \
        --head1_pos_grad_weight $HEAD1_POS_WEIGHT \
        --head2_type_grad_weight $HEAD2_TYPE_WEIGHT \
        --head2_pos_grad_weight $HEAD2_POS_WEIGHT \
        --w_on $W_ON \
        --w_off $W_OFF \
        --batch_size $BATCH_SIZE \
        --num_samples $NUM_SAMPLES \
        --result_path "$RESULT_PATH_LOW" 2>&1 | tee -a "$LOG_FILE"; then
        echo "    LOW Generation: SUCCESS" >> "$LOG_FILE"
    else
        echo "    LOW Generation: FAILED" >> "$LOG_FILE"
    fi

done < "$PAIRS_FILE"

echo ""
echo "========================================================================"
echo "PHASE 1 COMPLETED: All molecules generated!"
echo "========================================================================"

# =======================================================================
# PHASE 2: DOCKING (CPU)
# =======================================================================
echo ""
echo "========================================================================"
echo "PHASE 2: MOLECULE DOCKING (CPU)"
echo "========================================================================"
echo ""

PROCESSED=0

while IFS=',' read -r target_id high_off_id high_score low_off_id low_score; do

    PROCESSED=$((PROCESSED + 1))

    echo ""
    echo "# Docking for ID $target_id ($PROCESSED/$TOTAL_IDS)"

    # ===== Dock HIGHEST TM-score pair =====
    RESULT_PATH_HIGH="${BASE_RESULT_PATH}/id${target_id}_${high_off_id}_high"
    DOCKING_DIR_HIGH="${RESULT_PATH_HIGH}/docking_results"

    if [ -d "$RESULT_PATH_HIGH" ]; then
        if [ -d "$DOCKING_DIR_HIGH" ]; then
            rm -rf "$DOCKING_DIR_HIGH"
        fi

        echo "  [HIGH] Docking molecules for ($target_id, $high_off_id)..."
        if python scripts/dock_generated_ligands.py \
            --use_lmdb_only \
            --mode id_specific \
            --sample_path "$RESULT_PATH_HIGH" \
            --output_dir "$DOCKING_DIR_HIGH" \
            --on_target_id $target_id \
            --off_target_ids $high_off_id \
            --docking_mode vina_dock \
            --exhaustiveness 8 \
            --save_visualization 2>&1 | tee -a "$LOG_FILE"; then
            echo "    HIGH Docking: SUCCESS" >> "$LOG_FILE"
        else
            echo "    HIGH Docking: FAILED" >> "$LOG_FILE"
        fi
    else
        echo "  [HIGH] WARNING: Generation results not found"
    fi

    # ===== Dock LOWEST TM-score pair =====
    RESULT_PATH_LOW="${BASE_RESULT_PATH}/id${target_id}_${low_off_id}_low"
    DOCKING_DIR_LOW="${RESULT_PATH_LOW}/docking_results"

    if [ -d "$RESULT_PATH_LOW" ]; then
        if [ -d "$DOCKING_DIR_LOW" ]; then
            rm -rf "$DOCKING_DIR_LOW"
        fi

        echo "  [LOW] Docking molecules for ($target_id, $low_off_id)..."
        if python scripts/dock_generated_ligands.py \
            --use_lmdb_only \
            --mode id_specific \
            --sample_path "$RESULT_PATH_LOW" \
            --output_dir "$DOCKING_DIR_LOW" \
            --on_target_id $target_id \
            --off_target_ids $low_off_id \
            --docking_mode vina_dock \
            --exhaustiveness 8 \
            --save_visualization 2>&1 | tee -a "$LOG_FILE"; then
            echo "    LOW Docking: SUCCESS" >> "$LOG_FILE"
        else
            echo "    LOW Docking: FAILED" >> "$LOG_FILE"
        fi
    else
        echo "  [LOW] WARNING: Generation results not found"
    fi

done < "$PAIRS_FILE"

echo ""
echo "========================================================================"
echo "ALL COMPLETED"
echo "========================================================================"
echo "Results directory: $BASE_RESULT_PATH"
echo "Log file: $LOG_FILE"
echo "End time: $(date)"
echo "========================================================================"
