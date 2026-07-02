#!/bin/bash
# Resume docking for folders that don't have docking_results yet
# CPU limited to 20 cores (0-19) using taskset

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate theselective

# Model configuration
BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/theselective}"
PAIRS_FILE="${PAIRS_FILE:-./data/tmscore_extreme_pairs.txt}"

if [ ! -f "$PAIRS_FILE" ]; then
    echo "ERROR: $PAIRS_FILE not found!"
    exit 1
fi

# Create log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./run_resume_docking_log_${TIMESTAMP}.txt"

echo "========================================================================"
echo "RESUME DOCKING (CPU limited to cores 0-19, 20 cores)"
echo "========================================================================"
echo "Results: $BASE_RESULT_PATH"
echo "Start time: $(date)"
echo ""

{
    echo "========================================================================"
    echo "RESUME DOCKING - $(date)"
    echo "CPU limited to cores 0-19 (20 cores)"
    echo "========================================================================"
    echo ""
} > "$LOG_FILE"

TOTAL_IDS=100
PROCESSED=0
SKIPPED=0
DOCKED=0

while IFS=',' read -r target_id high_off_id high_score low_off_id low_score; do

    PROCESSED=$((PROCESSED + 1))

    # ===== Dock HIGHEST TM-score pair =====
    RESULT_PATH_HIGH="${BASE_RESULT_PATH}/id${target_id}_${high_off_id}_high"
    DOCKING_DIR_HIGH="${RESULT_PATH_HIGH}/docking_results"

    if [ -d "$RESULT_PATH_HIGH" ]; then
        if [ -d "$DOCKING_DIR_HIGH" ] && [ -f "$DOCKING_DIR_HIGH/docking_results.json" ]; then
            echo "  [HIGH] SKIP ($target_id, $high_off_id) - already docked"
            SKIPPED=$((SKIPPED + 1))
        else
            # Remove incomplete docking results if any
            if [ -d "$DOCKING_DIR_HIGH" ]; then
                rm -rf "$DOCKING_DIR_HIGH"
            fi

            echo ""
            echo "# Docking ID $target_id ($PROCESSED/$TOTAL_IDS)"
            echo "  [HIGH] Docking molecules for ($target_id, $high_off_id)..."
            if taskset -c 0-19 python scripts/dock_generated_ligands.py \
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
                DOCKED=$((DOCKED + 1))
            else
                echo "    HIGH Docking: FAILED" >> "$LOG_FILE"
            fi
        fi
    else
        echo "  [HIGH] WARNING: Generation results not found for ($target_id, $high_off_id)"
    fi

    # ===== Dock LOWEST TM-score pair =====
    RESULT_PATH_LOW="${BASE_RESULT_PATH}/id${target_id}_${low_off_id}_low"
    DOCKING_DIR_LOW="${RESULT_PATH_LOW}/docking_results"

    if [ -d "$RESULT_PATH_LOW" ]; then
        if [ -d "$DOCKING_DIR_LOW" ] && [ -f "$DOCKING_DIR_LOW/docking_results.json" ]; then
            echo "  [LOW] SKIP ($target_id, $low_off_id) - already docked"
            SKIPPED=$((SKIPPED + 1))
        else
            # Remove incomplete docking results if any
            if [ -d "$DOCKING_DIR_LOW" ]; then
                rm -rf "$DOCKING_DIR_LOW"
            fi

            echo ""
            echo "# Docking ID $target_id ($PROCESSED/$TOTAL_IDS)"
            echo "  [LOW] Docking molecules for ($target_id, $low_off_id)..."
            if taskset -c 0-19 python scripts/dock_generated_ligands.py \
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
                DOCKED=$((DOCKED + 1))
            else
                echo "    LOW Docking: FAILED" >> "$LOG_FILE"
            fi
        fi
    else
        echo "  [LOW] WARNING: Generation results not found for ($target_id, $low_off_id)"
    fi

done < "$PAIRS_FILE"

echo ""
echo "========================================================================"
echo "RESUME DOCKING COMPLETED"
echo "========================================================================"
echo "Skipped (already done): $SKIPPED"
echo "Newly docked: $DOCKED"
echo "Results directory: $BASE_RESULT_PATH"
echo "Log file: $LOG_FILE"
echo "End time: $(date)"
echo "========================================================================"
