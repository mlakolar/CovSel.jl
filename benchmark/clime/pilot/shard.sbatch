#!/bin/bash
# ACLIME pilot study, p = 200: one replication per task (pilot_shard.jl replays
# the random stream, so the draws are those of pilot.jl), GNU Parallel, 1 CPU
# each. Step 2 only for the pilots of the README tables, δ grid from 0.1. First
# reproduces replication 2 of er, p = 50, from pilot.jl's output (full grid).
# RUN holds a clone of CovSel.jl and a copy of this directory as RUN/pilot:
#   sbatch --export=ALL,RUN=$RUN shard.sbatch
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=00:55:00
set -uo pipefail
: "${RUN:?}"
export PROJ=$RUN/CovSel.jl/benchmark/clime
export JULIA=$HOME/juliaPkg/julia-1.13.2/bin/julia
export JULIA_DEPOT_PATH=$HOME/juliaPkg/depot:
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export STEP2="paper 2.0;paper 0.5;std 2.0;std 2.0 refit;1/σ;oracle;standardized ACLIME"
module load parallel/20240522
OUT=$RUN/results/p200_shards
mkdir -p $OUT/logs
echo "start $(date -u +%FT%TZ) on $(hostname)"
S="srun --exclusive --exact -N1 -n1 -c1 --mem=1800M $JULIA -t 1 --project=$PROJ"
# check: the rows with step 2 must equal pilot.jl's, the others only lack the errors
DMIN=1 $S $RUN/pilot/pilot_shard.jl er 50 2 $OUT/check_er_p50_r2.csv || exit 1
grep '^er,50,2,' $RUN/results/er_p50_raw.csv > $OUT/check_ref.csv
nfull=$(grep -vc NaN $OUT/check_er_p50_r2.csv)
nbad=$(grep -v NaN $OUT/check_er_p50_r2.csv | grep -vxFf $OUT/check_ref.csv | wc -l)
nstat=$(cut -d, -f1-9 $OUT/check_er_p50_r2.csv | grep -vxFf <(cut -d, -f1-9 $OUT/check_ref.csv) | wc -l)
echo "check: $nfull rows with step 2, $nbad differ; pilot statistics differ in $nstat rows"
[ "$nfull" = 7 ] && [ "$nbad" = 0 ] && [ "$nstat" = 0 ] || { echo "check failed"; exit 1; }
for r in $(seq 1 20); do
  for m in er ar1 band; do
    [ -s $OUT/${m}_r$r.csv ] || echo "DMIN=5 $S $RUN/pilot/pilot_shard.jl $m 200 $r $OUT/${m}_r$r.csv > $OUT/logs/${m}_r$r.log 2>&1"
  done
done > $OUT/tasks.txt
parallel --jobs 8 --joblog $OUT/logs/joblog.tsv < $OUT/tasks.txt
echo "done $(date -u +%FT%TZ)"
