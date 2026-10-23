#!/bin/bash
# ACLIME step-1 pilot study (CovSel.jl branch clime): pilot accuracy and its
# effect on step 2 in the design of Cai, Liu and Zhou (2016), the bias of the
# pilot as n grows, and invariance checks. One CPU per task, GNU Parallel.
# RUN holds a clone of CovSel.jl and a copy of this directory as RUN/pilot:
#   sbatch --export=ALL,RUN=$RUN pilot.sbatch
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=01:00:00
set -uo pipefail
: "${RUN:?}"
export PROJ=$RUN/CovSel.jl/benchmark/clime
export JULIA=$HOME/juliaPkg/julia-1.13.2/bin/julia
export JULIA_DEPOT_PATH=$HOME/juliaPkg/depot:
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
module load parallel/20240522
OUT=$RUN/results
mkdir -p $OUT/logs
echo "start $(date -u +%FT%TZ) on $(hostname)"
$JULIA --project=$PROJ -e 'using Pkg; Pkg.precompile(); using CovSel' || exit 1
S="srun --exclusive --exact -N1 -n1 -c1 --mem=1800M $JULIA -t 1 --project=$PROJ"
{
  for p in 200 100 50; do
    reps=50; [ $p = 200 ] && reps=20
    for m in er ar1 band; do
      echo "$S $RUN/pilot/pilot.jl $m $p $reps $OUT/${m}_p$p.csv > $OUT/logs/${m}_p$p.log 2>&1"
    done
  done
  echo "$S $RUN/pilot/scaling.jl > $OUT/scaling.txt 2> $OUT/logs/scaling.log"
  echo "$S $RUN/pilot/invariance.jl > $OUT/invariance.txt 2> $OUT/logs/invariance.log"
} > $OUT/tasks.txt
parallel --jobs 8 --joblog $OUT/logs/joblog.tsv < $OUT/tasks.txt
echo "done $(date -u +%FT%TZ)"
