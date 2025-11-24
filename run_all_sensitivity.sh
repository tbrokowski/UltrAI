#!/bin/bash

# Run TB deep model, Pathology ML model, and summarize ULTR-AI MAX across all folds
# Uses the sensitivity labels created earlier.
#
# Usage:
#   chmod +x run_all_sensitivity.sh
#   ./run_all_sensitivity.sh
#
# Adjust paths below if needed.

set -euo pipefail

WORKDIR="/users/${USER}/ULTR-AI-Vid"
ROOT="$WORKDIR/ULTR-AI/UltrAI"
BENIN="$WORKDIR/ULTR-AI/benin_trust_workingfiles"
REQS_FILE="$WORKDIR/ULTR-AI/requirements.txt"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EDF_ENV_FILE="${EDF_ENV_FILE:-/users/${USER}/.edf/ultrai.toml}"
AUTO_SUBMIT_VIA_SBATCH="${AUTO_SUBMIT_VIA_SBATCH:-1}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-a127}"
SBATCH_TIME="${SBATCH_TIME:-7:00:00}"
SBATCH_GPUS="${SBATCH_GPUS:-4}"
DEFAULT_VISIBLE_GPUS="${DEFAULT_VISIBLE_GPUS:-0,1,2,3}"
SBATCH_CPUS="${SBATCH_CPUS:-32}"

if [[ -z "${SLURM_JOB_ID:-}" && "${RUN_ALL_SENSITIVITY_IN_SLURM:-0}" != "1" ]]; then
  if [[ "$AUTO_SUBMIT_VIA_SBATCH" == "1" && -x "$(command -v sbatch)" && -f "$EDF_ENV_FILE" ]]; then
    SCRIPT_PATH="$(readlink -f "$0")"
    LOG_DIR="$WORKDIR/logs/run_all_sensitivity"
    mkdir -p "$LOG_DIR"
    echo "[INFO] Submitting sensitivity sweep via SLURM (container: $EDF_ENV_FILE)"
    sbatch \
      --export=ALL,RUN_ALL_SENSITIVITY_IN_SLURM=1 \
      --job-name=tb_sensitivity \
      --nodes=1 \
      --ntasks-per-node=1 \
      --gres=gpu:${SBATCH_GPUS} \
      --cpus-per-task=${SBATCH_CPUS} \
      --time="${SBATCH_TIME}" \
      --environment "${EDF_ENV_FILE}" \
      -A "${SBATCH_ACCOUNT}" \
      --output="${LOG_DIR}/R-%x.%j.out" \
      --error="${LOG_DIR}/R-%x.%j.err" \
      --wrap="bash ${SCRIPT_PATH}"
    exit 0
  else
    cat <<'MSG'
[WARN] This script expects to run inside the ULTR-AI SLURM container.
       Either:
       • submit via sbatch (AUTO_SUBMIT_VIA_SBATCH=1, default), or
       • set RUN_ALL_SENSITIVITY_IN_SLURM=1 and provide a Python >=3.8 interpreter via PYTHON_BIN.
MSG
    exit 1
  fi
fi

mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "python: $(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" -V

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU diagnostics (nvidia-smi) ==="
  nvidia-smi || echo "[WARN] nvidia-smi reported an error."
else
  echo "[WARN] nvidia-smi not found; cannot display GPU status."
fi

echo "=== Verifying CUDA availability in Python ==="
"$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
except ImportError as exc:
    print(f"[ERROR] torch not found: {exc}")
    sys.exit(1)

if not torch.cuda.is_available():
    print("[ERROR] torch.cuda.is_available() returned False. Ensure the job runs on a GPU node.")
    sys.exit(2)

print(f"[OK] torch {torch.__version__} | CUDA {torch.version.cuda} | GPUs: {torch.cuda.device_count()}")
for idx in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(idx)
    print(f"  GPU{idx}: {props.name} ({props.total_memory/1024**3:.1f} GiB)")
PY

ensure_pip() {
  if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  echo "[INFO] pip missing for ${PYTHON_BIN}; attempting ensurepip..."
  if "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install --upgrade pip
    return 0
  fi
  echo "[WARN] Could not bootstrap pip for ${PYTHON_BIN}. Please point PYTHON_BIN to an environment with pip."
  return 1
}

run_gpu_python() {
  local visible="${CUDA_VISIBLE_DEVICES:-$DEFAULT_VISIBLE_GPUS}"
  # Use -u flag for unbuffered output so we see all training epochs in real-time
  CUDA_VISIBLE_DEVICES="$visible" "$PYTHON_BIN" -u "$@"
}

if ensure_pip; then
  if [[ -f "$REQS_FILE" ]]; then
    echo "Installing/validating Python requirements..."
    "$PYTHON_BIN" -m pip install --user -r "$REQS_FILE"
  else
    echo "[WARN] Requirements file not found at $REQS_FILE"
  fi
else
  echo "[WARN] Skipping requirements install because pip is unavailable."
fi

export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export NCCL_NET_PLUGIN=none
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
unset LD_PRELOAD
export NCCL_PLUGIN_P2P=0
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((10000 + RANDOM % 50000))}"

LABELS_CSV="$ROOT/data/labels/sensitivity_analysis_labels.csv"
SPLITS_DIR="$ROOT/data/Splits"

TB_SAVE_DIR="$ROOT/results/models_sensitivity"
TB_PRED_DIR="$ROOT/results/TBPredictions_Sensitivity"

PATH_FEATURES_CSV="$BENIN/CXR_Comp_Files/aiml_traincombined.csv"
PATH_PRED_DIR="$BENIN/predictions_path_sens_analysis"

FOLDS=(0 1 2 3 4)

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "=== Running TB deep model across folds with sensitivity labels ==="
mkdir -p "$TB_SAVE_DIR" "$TB_PRED_DIR"
for F in "${FOLDS[@]}"; do
  echo "=========================================="
  echo "--- Starting Fold_${F} ---"
  echo "=========================================="
  echo "Labels: $LABELS_CSV"
  echo "Test indices: $SPLITS_DIR/Fold_${F}.csv"
  echo "Save dir: $TB_SAVE_DIR"
  echo "Pred dir: $TB_PRED_DIR"
  echo ""
  run_gpu_python "$ROOT/trainTB.py" \
    --labels_file "$LABELS_CSV" \
    --test_indices_file "$SPLITS_DIR/Fold_${F}.csv" \
    --pred_save_dir "$TB_PRED_DIR" \
    --save_dir "$TB_SAVE_DIR" \
    --feature "Fold_${F}" \
    --run_name "sensitivity" \
    --run_id "1"
  echo ""
  echo "--- Completed Fold_${F} ---"
  echo ""
done

echo
echo "=== Running Pathology ML model across folds ==="
mkdir -p "$PATH_PRED_DIR"
run_gpu_python "$ROOT/train_pathology_ml.py" \
  --features_csv "$PATH_FEATURES_CSV" \
  --labels_csv "$LABELS_CSV" \
  --splits_dir "$SPLITS_DIR" \
  --folds "0,1,2,3,4" \
  --output_dir "$PATH_PRED_DIR"

echo
echo "=== Summarizing and computing ULTR-AI MAX across folds ==="
run_gpu_python "$ROOT/summarize_results.py" \
  --tb_preds_dir "$TB_PRED_DIR" \
  --path_preds_dir "$PATH_PRED_DIR" \
  --folds "0,1,2,3,4" \
  --output_dir "$PATH_PRED_DIR"

echo

echo "All done."
echo "TB predictions:          $TB_PRED_DIR"
echo "Pathology predictions:   $PATH_PRED_DIR"
echo "ULTR-AI MAX summary:     $PATH_PRED_DIR/ULTR_AI_MAX_summary.csv"


#bash ULTR-AI/UltrAI/run_all_sensitivity.sh