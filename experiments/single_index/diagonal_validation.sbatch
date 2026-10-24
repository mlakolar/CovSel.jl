#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --project -e 'using Pkg; Pkg.test()'
/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia --project experiments/single_index/validate_dgp.jl
mkdir -p env110
cp /project2/mkolar_1314/SingleIndexGraph/20261024-sig-refinement/env110/Project.toml env110/
cp /project2/mkolar_1314/SingleIndexGraph/20261024-sig-refinement/env110/Manifest.toml env110/
/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia --project=env110 -e 'using Pkg; Pkg.develop(path=pwd()); using Test,CovarianceSelection; include("test/single_index.jl"); include("experiments/single_index/validate_dgp.jl")'
