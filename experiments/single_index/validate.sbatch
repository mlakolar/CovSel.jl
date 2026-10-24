#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_DEPOT_PATH="$SLURM_SUBMIT_DIR/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
JULIA=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia
"$JULIA" --project -e 'using Pkg; Pkg.test()'

"$JULIA" --project experiments/single_index/validate_dgp.jl
