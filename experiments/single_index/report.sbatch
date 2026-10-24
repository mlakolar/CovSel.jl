#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
OUT=${1:?output directory required}
PLAN=${2:?plan path required}
python3 -m venv .report-env
.report-env/bin/pip --disable-pip-version-check install matplotlib
.report-env/bin/pip freeze > report-environment.txt
.report-env/bin/python experiments/single_index/summarize.py "$PLAN" "$OUT"
.report-env/bin/python experiments/single_index/report.py "$OUT"
