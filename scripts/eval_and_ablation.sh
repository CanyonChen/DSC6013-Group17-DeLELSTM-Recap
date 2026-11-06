#!/usr/bin/env bash
# DeLELSTM: isolated Conda setup -> GPU checks -> smoke -> experiments -> report.
# Linux/SSH entry point. Run from any directory; no changes to SuperMimic's env.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE=all
OUT="$ROOT/outputs/ablation"
ENV_PREFIX="$ROOT/.conda/delelstm-ablation"
PYTHON_BIN=""
SKIP_SETUP=0
EXTRA=()

usage() {
  cat <<'EOF'
Usage: bash scripts/eval_and_ablation.sh [options]

  --stage all|setup|smoke|run|report|plan    Default: all
  --gpus 0,1                              Logical indices among visible GPUs
  --output-dir PATH                       Default: outputs/ablation
  --env-prefix PATH                       Isolated Conda prefix under .conda/
  --python PATH                           Use an existing Python; implies no setup
  --skip-setup                            Use the already installed project environment
  --resume                                Skip successes, rerun failed/incomplete jobs
  --datasets Electricity PM Exchange      Dataset subset
  --models Delelstm IMV_full IMV_tensor Retain LSTM
  --seeds 555 556 557 558 559               PM adds 1, as in the course campaign
  --temporal-folds 1|2|3                   Default: 3 expanding temporal partitions
  --ridge-values 1e-6 1e-4 1e-2            Default ridge grid
  --cpu-control-runs 1                     Matched CPU training seeds per dataset; 0 disables
  --cpu-threads 4                         Threads per experiment process
  --diagnostic-windows 64                 Sampled test windows for matrix diagnostics
  --epochs N --max-windows N              Debug only; report labelled non-formal
  --help

Examples:
  bash scripts/eval_and_ablation.sh --stage plan
  bash scripts/eval_and_ablation.sh --gpus 0,1
  bash scripts/eval_and_ablation.sh --stage run --gpus 0,1 --resume
  bash scripts/eval_and_ablation.sh --stage report

all = setup + a separate one-epoch smoke campaign + full experiments + report.
Outputs: <output-dir>/report/report.{md,html}; weights and logs remain in jobs/.
Failures are retained and reported; a campaign with failures exits nonzero.
EOF
}

while (($#)); do
  case "$1" in
    --stage) STAGE="${2:?missing stage}"; shift 2 ;;
    --output-dir) OUT="${2:?missing output path}"; shift 2 ;;
    --env-prefix) ENV_PREFIX="${2:?missing environment path}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?missing Python path}"; SKIP_SETUP=1; shift 2 ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
case "$STAGE" in all|setup|smoke|run|report|plan) ;; *) echo "Unknown stage: $STAGE" >&2; exit 2 ;; esac
cd -- "$ROOT"
# Relative paths are always repository-relative, including paths passed from another directory.
[[ "$OUT" = /* ]] || OUT="$ROOT/$OUT"
[[ "$ENV_PREFIX" = /* ]] || ENV_PREFIX="$ROOT/$ENV_PREFIX"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
unset PYTHONPATH

setup_environment() {
  if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "Automatic GPU setup targets Linux x86_64. Use --python for an existing environment." >&2
    exit 2
  fi
  command -v git >/dev/null || { echo "Git is required." >&2; exit 2; }
  local conda_bin bootstrap installer url
  if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    conda_bin="$CONDA_EXE"
  elif command -v conda >/dev/null 2>&1; then
    conda_bin="$(command -v conda)"
  elif [[ -x "$ROOT/.conda/miniforge3/bin/conda" ]]; then
    conda_bin="$ROOT/.conda/miniforge3/bin/conda"
  else
    command -v curl >/dev/null || { echo "curl is required to install Miniforge." >&2; exit 2; }
    bootstrap="$ROOT/.conda/bootstrap"
    mkdir -p -- "$bootstrap"
    installer=Miniforge3-Linux-x86_64.sh
    url="https://github.com/conda-forge/miniforge/releases/latest/download/$installer"
    curl --fail --location --retry 3 "$url" -o "$bootstrap/$installer"
    curl --fail --location --retry 3 "$url.sha256" -o "$bootstrap/$installer.sha256"
    (cd -- "$bootstrap" && sha256sum --check "$installer.sha256")
    bash "$bootstrap/$installer" -b -p "$ROOT/.conda/miniforge3"
    conda_bin="$ROOT/.conda/miniforge3/bin/conda"
  fi
  if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    "$conda_bin" create --yes --prefix "$ENV_PREFIX" --override-channels -c conda-forge python=3.13 pip
  fi
  PYTHON_BIN="$ENV_PREFIX/bin/python"
  "$PYTHON_BIN" -m pip install -r "$ROOT/environment/requirements-ablation.txt"
  "$PYTHON_BIN" -m pip check
}

pick_python() {
  if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$ENV_PREFIX/bin/python" ]]; then
      PYTHON_BIN="$ENV_PREFIX/bin/python"
    elif [[ "$STAGE" == plan ]]; then
      PYTHON_BIN="$(command -v python3 || command -v python)"
    else
      echo "Project environment missing. Run --stage setup or specify --python." >&2
      exit 2
    fi
  fi
}

run_suite() {
  local stage="$1" destination="$2"
  "$PYTHON_BIN" "$ROOT/scripts/ablation_suite.py" --stage "$stage" --output-dir "$destination" "${EXTRA[@]}"
}

if [[ "$STAGE" == plan ]]; then
  pick_python
  run_suite plan "$OUT"
  exit 0
fi
if [[ "$STAGE" == setup || "$STAGE" == all ]] && ((SKIP_SETUP == 0)); then
  setup_environment
fi
pick_python
if [[ "$STAGE" != report ]]; then
  "$PYTHON_BIN" "$ROOT/scripts/setup_upstream.py"
fi
case "$STAGE" in
  setup) run_suite preflight "$OUT-environment-check" ;;
  smoke) run_suite smoke "$OUT-smoke" ;;
  run) run_suite run "$OUT" ;;
  report) run_suite report "$OUT" ;;
  all)
    run_suite smoke "$OUT-smoke"
    run_suite run "$OUT"
    ;;
esac
