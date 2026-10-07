#!/usr/bin/env bash
# Full VCBench run (put vcbench_final_public.csv and vcbench_final_private.csv under data/).
#   scripts/run_vcbench.sh                      one run with the config's seed
#   scripts/run_vcbench.sh --seed 1             another seed (explore/select split, screen sample, CV folds)
set -euo pipefail
cd "$(dirname "$0")/.."
for f in data/vcbench_final_public.csv data/vcbench_final_private.csv; do
  [[ -f $f ]] || { echo "missing $f"; exit 1; }
done
SEED=""
for ((i = 1; i <= $#; i++)); do [[ "${!i}" == "--seed" ]] && { j=$((i + 1)); SEED="_seed${!j}"; }; done
NAME="vcbench${SEED}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "runs/$NAME"
PYTHONUNBUFFERED=1 .venv/bin/python -m featexp run -c configs/vcbench.yaml --name "$NAME" "$@" 2>&1 | tee "runs/$NAME/run.log"
