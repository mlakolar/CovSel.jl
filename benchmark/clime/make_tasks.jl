# Write the task list of the CLIME benchmark campaign, one shell command per line,
# longest tasks first, for GNU Parallel inside a Slurm allocation (see campaign.sbatch):
#
#   julia benchmark/clime/make_tasks.jl PROBLEMS_DIR OUT_DIR > tasks.txt
#
# Uses the environment variables JULIA, BENCH (this directory), RSCRIPT and RLIB of
# the job script, quoted so that they are expanded when the task runs.

include(joinpath(@__DIR__, "problems.jl"))

const STEP = get(ENV, "STEP", raw"srun --exclusive --exact -N1 -n1 -c1")   # STEP="" runs locally
const JL = raw"$JULIA -t 1 --project=$BENCH"

problems_dir, out = ARGS[1], ARGS[2]
tasks = Tuple{Float64,String}[]   # (estimated seconds, command)
task(cost, mem, cmd, log) =
  push!(tasks, (cost, (isempty(STEP) ? "" : "$STEP --mem=$(mem)M timeout 2400 ") * "$cmd > $log 2>&1"))

for name in sort(readdir(problems_dir))
  dir = joinpath(problems_dir, name)
  isdir(dir) || continue
  meta = read_meta(dir)
  model, p, n = meta["model"], parse(Int, meta["p"]), parse(Int, meta["n"])
  # Julia: the simplex solvers always, HiGHS and PDHG up to p = 1000 (run_julia.jl
  # also limits the HiGHS path to p ≤ 500)
  solvers = ["ParametricSimplex", "DualSimplex"]
  p <= 1000 && append!(solvers, ["HiGHS", "PDHG"])
  cost = p <= 200 ? 30.0 : p == 500 ? 300.0 : p == 1000 ? 600.0 : p == 2000 ? 200.0 : 900.0
  mem = p >= 4000 ? 3800 : 1900
  task(cost, mem, "$JL \$BENCH/run_julia.jl $dir $out/julia $(join(solvers, ' '))", "$out/logs/julia_$name.log")
  # R: fastclime up to p = 500, flare up to p = 200, clime's simplex for p = 100,
  # its interior point method for p = 100 and n > p (it fails on a singular S)
  rcmd(method, opts) = "\$RSCRIPT \$BENCH/bench_r.R $dir/S.bin $p $dir/lambda.txt $method $out/r/$(name)__$method lib=\$RLIB write=FALSE $opts"
  if p <= 500
    nl = p <= 200 ? 2p : 400
    task(p <= 200 ? 20.0 : 400.0, p <= 200 ? 1900 : 3800,
      rcmd("fastclime", "fc_direct=TRUE fc_conly=FALSE fc_nlambda=$nl"), "$out/logs/fastclime_$name.log")
  end
  p <= 200 && task(p == 200 ? 300.0 : 30.0, 1900, rcmd("flare", ""), "$out/logs/flare_$name.log")
  p == 100 && task(60.0, 1900, rcmd("clime_simplex", "clime_mode=internal"), "$out/logs/clime_simplex_$name.log")
  (p == 100 && n > p) && task(120.0, 1900, rcmd("clime", "clime_mode=internal"), "$out/logs/clime_$name.log")
end

# the replication of Table 1 of Cai, Liu and Zhou (2016); QUICK=1 for a test run
const ACLIME = get(ENV, "QUICK", "0") == "1" ? ((50, 2, 10.0),) : ((50, 50, 150.0), (100, 50, 600.0), (200, 20, 1200.0))
for model in ("band", "ar1", "er"), (p, reps, cost) in ACLIME
  task(cost, 1900, "$JL \$BENCH/aclime_table.jl $model $p $reps $out/aclime/$(model)_p$p.csv",
    "$out/logs/aclime_$(model)_p$p.log")
end

for (_, cmd) in sort(tasks; by=first, rev=true)
  println(cmd)
end
