#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=/project2/mkolar_1314/SingleIndexGraph/20261024-sig-finalvalidation/.report-env/bin/python
ROOT=/project2/mkolar_1314/SingleIndexGraph
for stage in development confirmation refinement; do
  "$PYTHON" experiments/single_index/summarize.py "$ROOT/20261024-sig-$stage/experiments/single_index/$stage.toml" "$ROOT/20261024-sig-$stage/results"
  "$PYTHON" experiments/single_index/report.py "$ROOT/20261024-sig-$stage/results"
done
"$PYTHON" experiments/single_index/assess.py "$ROOT" results
