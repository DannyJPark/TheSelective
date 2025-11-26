#!/bin/bash
#
# STEP 2 - Model 1: head2_atom_attention
# Phase 1: Generate all molecules (GPU)
# Phase 2: Dock all molecules (CPU)
# Run on GPU 0
#

set -e  # Exit on error

# GPU assignment
# Note: When running with SLURM (sbatch), CUDA_VISIBLE_DEVICES is automatically set
# Only set it manually when running directly (not via SLURM)
if [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate theselective

MODEL_NAME="head2_atom_attention"
CONFIG_FILE="./configs/sampling_head2_atom_attention.yml"
GUIDE_MODE="head2_only_sequential"

echo "========================================================================"
echo "STEP 2 - Model 1: $MODEL_NAME"
echo "========================================================================"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG_FILE"
echo "Guide Mode: $GUIDE_MODE"
echo "Start time: $(date)"
echo ""

# Generate TM-score pairs if not exists
if [ ! -f "tmscore_extreme_pairs.txt" ]; then
    echo "Generating TM-score extreme pairs..."
    python parse_tmscore_pairs.py
    echo ""
fi

# Create log
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./step2_${MODEL_NAME}_log_${TIMESTAMP}.txt"

{
    echo "========================================================================"
    echo "STEP 2 - MODEL 1: $MODEL_NAME"
    echo "========================================================================"
    echo "Timestamp: $(date)"
    echo "GPU: $CUDA_VISIBLE_DEVICES"
    echo "Processing IDs: 0-99"
    echo "========================================================================"
    echo ""
} > "$LOG_FILE"

# =======================================================================
# PHASE 1: GENERATION (GPU) - All IDs
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
    echo "###################################################################"
    echo "# Generating for ID $target_id ($PROCESSED/$TOTAL_IDS)"
    echo "###################################################################"
    echo "  HIGH TM-score: ($target_id, $high_off_id) - Score: $high_score"
    echo "  LOW TM-score:  ($target_id, $low_off_id) - Score: $low_score"
    echo ""

    {
        echo "ID $target_id Generation:"
        echo "  HIGH: ($target_id, $high_off_id) - $high_score"
        echo "  LOW:  ($target_id, $low_off_id) - $low_score"
    } >> "$LOG_FILE"

    # Generate for HIGHEST TM-score pair
    RESULT_PATH_HIGH="./scratch2/results/cd2020_test_${MODEL_NAME}_id${target_id}_${high_off_id}_w2_1_100_0_lmdb_batch4_10"

    if [ -f "${RESULT_PATH_HIGH}/result_0.pt" ]; then
        echo "  [HIGH] Generation exists. Skipping..."
        echo "    HIGH Generation: EXISTS" >> "$LOG_FILE"
    else
        echo "  [HIGH] Generating molecules for ($target_id, $high_off_id)..."
        if python scripts/sample_diffusion.py \
            --config "$CONFIG_FILE" \
            --use_lmdb_only \
            --data_id $target_id \
            --off_target_ids $high_off_id \
            --guide_mode "$GUIDE_MODE" \
            --w_on 2.0 \
            --w_off 1.0 \
            --type_grad_weight 100.0 \
            --pos_grad_weight 0.0 \
            --batch_size 4 \
            --result_path "$RESULT_PATH_HIGH" 2>&1 | tee -a "$LOG_FILE"; then
            echo "    HIGH Generation: SUCCESS" >> "$LOG_FILE"
        else
            echo "    HIGH Generation: FAILED" >> "$LOG_FILE"
        fi
    fi

    # Generate for LOWEST TM-score pair
    RESULT_PATH_LOW="./scratch2/results/cd2020_test_${MODEL_NAME}_id${target_id}_${low_off_id}_w2_1_100_0_lmdb_batch4_10"

    if [ -f "${RESULT_PATH_LOW}/result_0.pt" ]; then
        echo "  [LOW] Generation exists. Skipping..."
        echo "    LOW Generation: EXISTS" >> "$LOG_FILE"
    else
        echo "  [LOW] Generating molecules for ($target_id, $low_off_id)..."
        if python scripts/sample_diffusion.py \
            --config "$CONFIG_FILE" \
            --use_lmdb_only \
            --data_id $target_id \
            --off_target_ids $low_off_id \
            --guide_mode "$GUIDE_MODE" \
            --w_on 2.0 \
            --w_off 1.0 \
            --type_grad_weight 100.0 \
            --pos_grad_weight 0.0 \
            --batch_size 4 \
            --result_path "$RESULT_PATH_LOW" 2>&1 | tee -a "$LOG_FILE"; then
            echo "    LOW Generation: SUCCESS" >> "$LOG_FILE"
        else
            echo "    LOW Generation: FAILED" >> "$LOG_FILE"
        fi
    fi

    {
        echo "  Generation completed!"
        echo ""
    } >> "$LOG_FILE"

done < tmscore_extreme_pairs.txt

echo ""
echo "========================================================================"
echo "PHASE 1 COMPLETED: All molecules generated!"
echo "========================================================================"
echo "End time: $(date)"
echo ""

{
    echo ""
    echo "========================================================================"
    echo "PHASE 1 COMPLETED"
    echo "========================================================================"
    echo "End time: $(date)"
    echo ""
} >> "$LOG_FILE"

# =======================================================================
# PHASE 2: DOCKING (CPU) - All IDs
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
    echo "###################################################################"
    echo "# Docking for ID $target_id ($PROCESSED/$TOTAL_IDS)"
    echo "###################################################################"
    echo "  HIGH TM-score: ($target_id, $high_off_id)"
    echo "  LOW TM-score:  ($target_id, $low_off_id)"
    echo ""

    {
        echo "ID $target_id Docking:"
    } >> "$LOG_FILE"

    # Dock HIGHEST TM-score pair
    RESULT_PATH_HIGH="./scratch2/results/cd2020_test_${MODEL_NAME}_id${target_id}_${high_off_id}_w2_1_100_0_lmdb_batch4_10"
    DOCKING_FILE_HIGH="${RESULT_PATH_HIGH}/docking_results/docking_results.json"

    if [ -f "$DOCKING_FILE_HIGH" ]; then
        echo "  [HIGH] Docking results exist. Skipping..."
        echo "    HIGH Docking: EXISTS" >> "$LOG_FILE"
    elif [ -d "$RESULT_PATH_HIGH" ]; then
        echo "  [HIGH] Docking molecules for ($target_id, $high_off_id)..."
        if python dock_generated_ligands_unified.py \
            --use_lmdb_only \
            --mode id_specific \
            --sample_path "$RESULT_PATH_HIGH" \
            --output_dir "${RESULT_PATH_HIGH}/docking_results" \
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
        echo "    HIGH Docking: NOT FOUND" >> "$LOG_FILE"
    fi

    # Dock LOWEST TM-score pair
    RESULT_PATH_LOW="./scratch2/results/cd2020_test_${MODEL_NAME}_id${target_id}_${low_off_id}_w2_1_100_0_lmdb_batch4_10"
    DOCKING_FILE_LOW="${RESULT_PATH_LOW}/docking_results/docking_results.json"

    if [ -f "$DOCKING_FILE_LOW" ]; then
        echo "  [LOW] Docking results exist. Skipping..."
        echo "    LOW Docking: EXISTS" >> "$LOG_FILE"
    elif [ -d "$RESULT_PATH_LOW" ]; then
        echo "  [LOW] Docking molecules for ($target_id, $low_off_id)..."
        if python dock_generated_ligands_unified.py \
            --use_lmdb_only \
            --mode id_specific \
            --sample_path "$RESULT_PATH_LOW" \
            --output_dir "${RESULT_PATH_LOW}/docking_results" \
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
        echo "    LOW Docking: NOT FOUND" >> "$LOG_FILE"
    fi

    {
        echo "  Docking completed!"
        echo ""
    } >> "$LOG_FILE"

done < tmscore_extreme_pairs.txt

echo ""
echo "========================================================================"
echo "PHASE 2 COMPLETED: All molecules docked!"
echo "========================================================================"
echo "End time: $(date)"
echo ""

{
    echo ""
    echo "========================================================================"
    echo "PHASE 2 COMPLETED"
    echo "========================================================================"
    echo "End time: $(date)"
    echo ""
} >> "$LOG_FILE"

echo ""
echo "========================================================================"
echo "Model 1 ($MODEL_NAME) ALL COMPLETED!"
echo "========================================================================"
echo "Total processing time: Start $(head -5 $LOG_FILE | tail -1) to End $(date)"
echo "Log file: $LOG_FILE"
echo "========================================================================"

{
    echo ""
    echo "========================================================================"
    echo "MODEL 1 ALL COMPLETED"
    echo "========================================================================"
    echo "Total end time: $(date)"
    echo "========================================================================"
} >> "$LOG_FILE"
