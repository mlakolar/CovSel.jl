#!/bin/bash
# The CLIME benchmark campaign on one node (see README.md):
#
#   RUN=<run directory with a clone of CovSel.jl in $RUN/CovSel.jl> sbatch campaign.sbatch
#
# 1. precompile, write the problems and the task list;
# 2. run the single-threaded tasks (Julia solvers, R packages, ACLIME table) with
#    GNU Parallel, one CPU each;
# 3. time ParametricSimplex with 2, 4 and 8 threads on the largest problems.
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=01:50:00

set -uo pipefail
: "${RUN:?set RUN to the run directory}"
export BENCH=$RUN/CovSel.jl/benchmark/clime
export JULIA=$HOME/juliaPkg/julia-1.13.2/bin/julia
export JULIA_DEPOT_PATH=$HOME/juliaPkg/depot:
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
module load r/4.5.3 parallel/20240522
export RSCRIPT=$(command -v Rscript)
export RLIB=$HOME/juliaPkg/Rlib-r4.5.3
OUT=$RUN/results
mkdir -p "$OUT"/{julia,r,aclime,logs,threads}
echo "start $(date -u +%FT%TZ) on $(hostname)"

# 1. compile for this CPU, problems, tasks
$JULIA --project=$BENCH -e 'using Pkg; Pkg.precompile(); using CovSel, HiGHS' || exit 1
[ -d $RUN/problems ] || $JULIA --project=$BENCH $BENCH/prepare.jl $RUN/problems > $RUN/problems.txt || exit 1
$JULIA --project=$BENCH $BENCH/make_tasks.jl $RUN/problems $OUT > $RUN/tasks.txt || exit 1
echo "$(wc -l < $RUN/tasks.txt) tasks, $(date -u +%FT%TZ)"

# 2. single-threaded tasks; --resume skips tasks that finished in an earlier attempt
parallel --jobs 8 --joblog $OUT/logs/joblog.tsv --resume < $RUN/tasks.txt
echo "single-threaded tasks done $(date -u +%FT%TZ)"

# 3. thread scaling of ParametricSimplex
for name in ar1_p4000_n2000 ar1_p4000_n8000 ar1_p2000_n1000 ar1_p2000_n4000 band_p2000_n1000 er_p2000_n1000; do
  dir=$(ls -d $RUN/problems/${name}_s* 2>/dev/null | head -1)
  [ -n "$dir" ] || continue
  for t in 2 4 8; do
    srun --exclusive --exact -N1 -n1 -c8 --mem=12G env OPENBLAS_NUM_THREADS=1 \
      $JULIA -t $t --project=$BENCH $BENCH/run_julia.jl $dir $OUT/threads ParametricSimplex \
      > $OUT/logs/threads_${name}_T$t.log 2>&1
  done
done
echo "done $(date -u +%FT%TZ)"
