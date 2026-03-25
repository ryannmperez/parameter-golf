#!/usr/bin/env bash
# run_experiment.sh — 3-point scaling law experiment runner
#
# Runs train_gpt.py at 3 compute budgets (1min, 3min, 5min),
# scales warmdown proportionally for each, captures val_bpb,
# fits a power law curve, and extrapolates to H100 budget.
#
# Usage:
#   bash run_experiment.sh <experiment_name>
#   bash run_experiment.sh exp_002_qat_int6
#
# Requirements: python3, scipy (pip install scipy)

set -euo pipefail

EXP_NAME="${1:-exp_unknown}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/${EXP_NAME}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "======================================================"
echo "  Parameter Golf — Scaling Experiment Runner"
echo "  Experiment: $EXP_NAME"
echo "  Logs: $LOG_DIR"
echo "======================================================"

# GPU setup
export CUDA_VISIBLE_DEVICES=0

# Training budgets (seconds) and proportional warmdown iters
# Warmdown = ~40% of expected steps at each budget
# Rough A6000 step estimate: ~300 steps/min at batch=524288
declare -a BUDGETS=(60 180 300)
declare -a WARMDOWNS=(50 200 400)   # scaled warmdown iters per budget
declare -a BPB_RESULTS=()
declare -a COMPUTE_RESULTS=()

for i in "${!BUDGETS[@]}"; do
    BUDGET="${BUDGETS[$i]}"
    WARMDOWN="${WARMDOWNS[$i]}"
    RUN_ID="${EXP_NAME}_${BUDGET}s"
    LOG_FILE="${LOG_DIR}/run_${BUDGET}s.log"

    echo ""
    echo "--- Run $((i+1))/3 | Budget: ${BUDGET}s | Warmdown: ${WARMDOWN} iters ---"

    CUDA_VISIBLE_DEVICES=0 \
    MAX_WALLCLOCK_SECONDS=$BUDGET \
    WARMDOWN_ITERS=$WARMDOWN \
    VAL_LOSS_EVERY=9999 \
    RUN_ID=$RUN_ID \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOG_FILE"

    # Extract final val_bpb from log
    BPB=$(grep -oP 'val_bpb:\K[0-9.]+' "$LOG_FILE" | tail -1)
    if [[ -z "$BPB" ]]; then
        echo "[ERROR] Could not extract val_bpb from run ${BUDGET}s"
        exit 1
    fi

    BPB_RESULTS+=("$BPB")
    COMPUTE_RESULTS+=("$BUDGET")
    echo "  → val_bpb: $BPB"
done

echo ""
echo "======================================================"
echo "  Results Summary"
echo "======================================================"
echo "  1 min → bpb: ${BPB_RESULTS[0]}"
echo "  3 min → bpb: ${BPB_RESULTS[1]}"
echo "  5 min → bpb: ${BPB_RESULTS[2]}"

# Write results JSON for scaling.py
RESULTS_JSON="${LOG_DIR}/scaling_results.json"
cat > "$RESULTS_JSON" << EOF
{
  "experiment": "$EXP_NAME",
  "timestamp": "$TIMESTAMP",
  "hardware": "A6000",
  "runs": [
    {"budget_seconds": ${COMPUTE_RESULTS[0]}, "val_bpb": ${BPB_RESULTS[0]}},
    {"budget_seconds": ${COMPUTE_RESULTS[1]}, "val_bpb": ${BPB_RESULTS[1]}},
    {"budget_seconds": ${COMPUTE_RESULTS[2]}, "val_bpb": ${BPB_RESULTS[2]}}
  ]
}
EOF

echo ""
echo "--- Fitting power law + extrapolating to H100 ---"
python3 scaling.py --results "$RESULTS_JSON" | tee "${LOG_DIR}/scaling_fit.txt"

echo ""
echo "--- Validating 5-min artifact ---"
FINAL_MODEL=$(ls logs/${EXP_NAME}_${TIMESTAMP}/ | grep "final_model" | head -1 || echo "")
if [[ -n "$FINAL_MODEL" ]]; then
    python3 validate.py \
        --script train_gpt.py \
        --model "${LOG_DIR}/${FINAL_MODEL}" \
        --bpb "${BPB_RESULTS[2]}"
else
    echo "[WARN] No model file found for size validation — run validate.py manually"
fi

echo ""
echo "======================================================"
echo "  Done. Full logs in: $LOG_DIR"
echo "======================================================"
