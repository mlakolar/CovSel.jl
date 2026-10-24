#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:15:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=/project2/mkolar_1314/SingleIndexGraph/20261024-sig-finalvalidation/.report-env/bin/python
RUN=/project2/mkolar_1314/SingleIndexGraph/20261024-sig-paper-confirmation
"$PYTHON" experiments/single_index/summarize.py "$RUN/experiments/single_index/paper_confirmation.toml" "$RUN/results"
"$PYTHON" experiments/single_index/report.py "$RUN/results"
"$PYTHON" experiments/single_index/paper_assess.py "$RUN/results"
