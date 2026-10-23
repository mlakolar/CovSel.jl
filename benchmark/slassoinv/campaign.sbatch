#!/bin/bash
# Benchmark of slassoinv (Sun and Zhang, 2013): speed against the R package
# scalreg and CoordinateDescent's scaled lasso, and accuracy against CLIME and
# ACLIME. One CPU per task under GNU Parallel, then the threaded runs on 8 CPUs.
# QUICK=1 runs a canary: every script on small settings, plus one replication at
# the largest settings to measure the time per replication. RERUN=1 skips the
# accuracy cells that one task computes alone (p ≤ 100 and p = 1000 of sz, p ≤ 100
# of clz, hadamard). Every task writes its own files: appends from concurrent
# tasks to one file interleave on the cluster file system.
#
# VERIFY=DIR adds a task that recomputes the estimates stored in DIR (the speed/
# directory of another run) with this CovSel (verify.jl).
#
#   RUN=... sbatch --export=ALL,RUN=$RUN[,QUICK=1][,RERUN=1][,OUTDIR=...][,VERIFY=...] campaign.sbatch
#
# RUN holds a clone of CovSel.jl; R packages in $HOME/juliaPkg/Rlib-r4.5.3.
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=01:30:00
set -uo pipefail
: "${RUN:?}"
QUICK=${QUICK:-0}
RERUN=${RERUN:-0}
export BENCH=$RUN/CovSel.jl/benchmark/slassoinv
export JULIA=$HOME/juliaPkg/julia-1.13.2/bin/julia
export JULIA_DEPOT_PATH=$HOME/juliaPkg/depot:
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
module load parallel/20240522 apptainer
OUT=${OUTDIR:-$RUN/results$([ $QUICK = 1 ] && echo _quick)}
mkdir -p $OUT/{accuracy,speed,logs}
echo "start $(date -u +%FT%TZ) on $(hostname), QUICK=$QUICK, RERUN=$RERUN, OUT=$OUT"
$JULIA --project=$BENCH -e 'using Pkg; Pkg.precompile(); using CovSel, CoordinateDescent' || exit 1
J="srun --exclusive --exact -N1 -n1 -c1 --mem=1800M $JULIA -t 1 --project=$BENCH"
R="srun --exclusive --exact -N1 -n1 -c1 --mem=1800M timeout 1500 apptainer exec --cleanenv --env OPENBLAS_NUM_THREADS=1 /apps/containers/r/4.5/r-4.5.3.sif Rscript"
acc() {  # design model p first last
  echo "$J $BENCH/accuracy.jl $1 $2 $3 $4 $5 $OUT/accuracy/$1_$2_p$3_r$4.csv > $OUT/logs/acc_$1_$2_p$3_r$4.log 2>&1"
}
{
  if [ $QUICK = 1 ]; then
    acc sz 2 300 1 1; acc clz er 200 1 1; acc sz 1 1000 1 2; acc hadamard 32 256 1 1
    acc sz 3 30 1 2; acc clz band 50 1 2; acc hadamard 4 256 1 2
    echo "CDPMAX=1000 RPMAX=1000 $J $BENCH/speed.jl ar1 1000 2000 $OUT/speed > $OUT/logs/speed_ar1_1000_2000.log 2>&1"
    echo "$J $BENCH/speed.jl er 500 1000 $OUT/speed > $OUT/logs/speed_er_500_1000.log 2>&1"
    echo "$J $BENCH/speed.jl band 100 50 $OUT/speed > $OUT/logs/speed_band_100_50.log 2>&1"
    echo "$J $BENCH/speed.jl band 200 400 $OUT/speed > $OUT/logs/speed_band_200_400.log 2>&1"
  else
    # accuracy: replications split into tasks of similar length
    for m in 1 2 3; do
      [ $RERUN = 1 ] || for p in 30 60 90; do acc sz $m $p 1 100; done
      acc sz $m 150 1 50; acc sz $m 150 51 100
      for r in 1 6 11 16; do acc sz $m 300 $r $((r + 4)); done
      [ $RERUN = 1 ] || acc sz $m 1000 1 100
    done
    for m in band ar1 er; do
      [ $RERUN = 1 ] || { acc clz $m 50 1 50; acc clz $m 100 1 50; }
      for r in 1 6 11 16; do acc clz $m 200 $r $((r + 4)); done
    done
    [ $RERUN = 1 ] || for k in 2 4 8 16 32; do acc hadamard $k 256 1 20; done
    # speed, one thread
    for m in ar1 band er; do
      for p in 4000 2000 1000 500 200 100; do
        for n in $((p / 2)) $((2 * p)); do
          echo "$J $BENCH/speed.jl $m $p $n $OUT/speed > $OUT/logs/speed_${m}_${p}_${n}.log 2>&1"
        done
      done
    done
  fi
  [ -n "${VERIFY:-}" ] && echo "$J $BENCH/verify.jl $VERIFY > $OUT/verify.txt 2> $OUT/logs/verify.log"
} > $OUT/tasks.txt
parallel --jobs 8 --joblog $OUT/logs/joblog.tsv < $OUT/tasks.txt
echo "julia phase done $(date -u +%FT%TZ)"
# scalreg on the data written by speed.jl
for d in $(ls -d $OUT/speed/*_s*/ 2>/dev/null); do
  name=$(basename $d)
  p=$(echo $name | sed 's/.*_p\([0-9]*\)_.*/\1/'); n=$(echo $name | sed 's/.*_n\([0-9]*\)_.*/\1/')
  [ $p -le ${RPMAX_R:-$([ $QUICK = 1 ] && echo 200 || echo 500)} ] && echo "$R $BENCH/bench_r.R $d $n $p $OUT/speed/r_$name.csv lib=$HOME/juliaPkg/Rlib-r4.5.3 > $OUT/logs/r_$name.log 2>&1"
done > $OUT/tasks_r.txt
parallel --jobs 8 --joblog $OUT/logs/joblog_r.tsv < $OUT/tasks_r.txt
echo "R phase done $(date -u +%FT%TZ)"
# threads
for p in $([ $QUICK = 1 ] && echo 1000 || echo 4000 2000 1000); do
  for m in $([ $QUICK = 1 ] && echo ar1 || echo ar1 band er); do
    for n in $((p / 2)) $((2 * p)); do
      srun --exclusive --exact -N1 -n1 -c8 --mem=12G $JULIA -t 8 --project=$BENCH $BENCH/speed.jl $m $p $n $OUT/speed threads > $OUT/logs/threads_${m}_${p}_${n}.log 2>&1
    done
  done
done
echo "done $(date -u +%FT%TZ)"
