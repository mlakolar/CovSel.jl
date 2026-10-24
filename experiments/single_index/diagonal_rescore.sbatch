#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:10:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=/project2/mkolar_1314/SingleIndexGraph/20261024-sig-finalvalidation/.report-env/bin/python
"$PYTHON" experiments/single_index/rescore_diagonal.py results results_log
"$PYTHON" experiments/single_index/summarize.py experiments/single_index/diagonal_development.toml results_log
"$PYTHON" experiments/single_index/diagonal_assess.py results_log development-log results
