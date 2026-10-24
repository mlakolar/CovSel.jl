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
WORKER_MEM_MB=${SIG_WORKER_MEM_MB:-2048}
PLAN=${1:?plan file required}
OUT=${2:?output directory required}
mkdir -p "$OUT"
# This small source/config fingerprint prevents reuse after scientific changes.
sha256sum Project.toml Manifest.toml src/*.jl src/SingleIndexGraph/*.jl \
  experiments/single_index/worker.jl experiments/single_index/dgp.jl "$PLAN" \
  > "$OUT/scientific-inputs-$SLURM_JOB_ID.sha256"
if test -f "$OUT/scientific-inputs.sha256"; then
  cmp "$OUT/scientific-inputs.sha256" "$OUT/scientific-inputs-$SLURM_JOB_ID.sha256"
else
  cp "$OUT/scientific-inputs-$SLURM_JOB_ID.sha256" "$OUT/scientific-inputs.sha256"
fi
# Tasks are bounded replication shards, not one scheduler submission per seed.
"$JULIA" --project -e 'using TOML; p=TOML.parsefile(ARGS[1]); println(sum(cld(c["replicates"],p["shard_size"]) for c in p["scenario"])); open(joinpath(ARGS[2],"expected-pointwise-count.txt"),"w") do io; print(io,sum(c["replicates"]*length(c["index_modes"])*length(c["h_factors"])*length(c["pilot_factors"])*length(c["penalties"])*length(c["methods"])*3*c["grid_n"] for c in p["scenario"])); end' "$PLAN" "$OUT" > "$OUT/task-count.txt"
# Final shards are atomically renamed by the worker and are safe to reuse.
# A recovery submission must first preserve any interrupted .partial files.
for task in $(seq 1 "$(cat "$OUT/task-count.txt")"); do
  printf -v shard '%s/shard-%05d.csv' "$OUT" "$task"
  test -f "$shard" || echo "$task"
done > "$OUT/tasks-$SLURM_JOB_ID.txt"
parallel --jobs "$SLURM_NTASKS" --joblog "$OUT/parallel-$SLURM_JOB_ID.tsv" --halt soon,fail=1 \
  srun --exclusive --exact -N1 -n1 -c1 --mem="$WORKER_MEM_MB" "$JULIA" --project experiments/single_index/worker.jl "$PLAN" {} "$OUT" :::: "$OUT/tasks-$SLURM_JOB_ID.txt"
python3 experiments/single_index/summarize.py "$PLAN" "$OUT"

if test -f experiments/single_index/report.py; then
  python3 experiments/single_index/report.py "$OUT"
fi
