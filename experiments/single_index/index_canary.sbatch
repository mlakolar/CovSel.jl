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
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --project -e 'using Test,CovarianceSelection; include("test/single_index.jl"); include("test/single_index_worker.jl")'
