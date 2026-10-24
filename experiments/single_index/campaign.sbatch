#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load ver/2506 gcc/14.3.0 parallel/20250522
export JULIA_DEPOT_PATH="$SLURM_SUBMIT_DIR/depot:/home1/mkolar/juliaPkg/depot:"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
JULIA=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia
PLAN=${1:?plan file required}
OUT=${2:?output directory required}
mkdir -p "$OUT"
# Tasks are bounded replication shards, not one scheduler submission per seed.
"$JULIA" --project -e 'using TOML; p=TOML.parsefile(ARGS[1]); println(sum(cld(c["replicates"],p["shard_size"]) for c in p["scenario"])); open(joinpath(ARGS[2],"expected-pointwise-count.txt"),"w") do io; print(io,sum(c["replicates"]*length(c["index_modes"])*length(c["h_factors"])*length(c["pilot_factors"])*length(c["penalties"])*length(c["methods"])*3*c["grid_n"] for c in p["scenario"])); end' "$PLAN" "$OUT" > "$OUT/task-count.txt"
seq 1 "$(cat "$OUT/task-count.txt")" | parallel --jobs "$SLURM_NTASKS" --joblog "$OUT/parallel.tsv" --halt soon,fail=1 \
  srun --exclusive --exact -N1 -n1 -c1 "$JULIA" --project experiments/single_index/worker.jl "$PLAN" {} "$OUT"
python3 experiments/single_index/summarize.py "$PLAN" "$OUT"

python3 experiments/single_index/report.py "$OUT"
