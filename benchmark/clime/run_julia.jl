# Time the CLIME solvers of CovSel on one problem written by prepare.jl:
#
#   julia --project=benchmark/clime -t T benchmark/clime/run_julia.jl PROBLEM_DIR OUT_DIR SOLVER...
#
# SOLVER is one of ParametricSimplex, DualSimplex, HiGHS, PDHG (tol 1e-6) and
# PDHG4 (tol 1e-4). Each is timed on the smallest penalty of the grid alone
# ("single") and on the whole grid ("path"), using all the threads of the process.
# For each solver it writes OUT_DIR/<problem>__<solver>__T<threads>.csv, a summary,
# and, for the path, ...__l1.bin: the ℓ1 norms of the column solutions (before
# symmetrization), p × nλ Float64, which summarize.jl compares across solvers.

using CovSel, LinearAlgebra, SparseArrays, Printf
using HiGHS
include(joinpath(@__DIR__, "problems.jl"))

const SOLVERS = Dict(
  "ParametricSimplex" => () -> ParametricSimplex(),
  "DualSimplex" => () -> DualSimplex(),
  "HiGHS" => () -> LPSolver(HiGHS.Optimizer),
  "PDHG" => () -> PDHG(; tol=1e-6),
  "PDHG4" => () -> PDHG(; tol=1e-4))

# the largest p for which each mode is run (the others take too long)
const MAXP = Dict(("HiGHS", "single") => 1000, ("HiGHS", "path") => 500,
                  ("PDHG", "single") => 1000, ("PDHG", "path") => 1000,
                  ("PDHG4", "single") => 1000, ("PDHG4", "path") => 1000)

l1cols(R) = [norm(view(R, :, i), 1) for i in 1:size(R, 2)]

# largest violation of the constraints |(SΩ - I)ᵢⱼ| ≤ λ, relative to λ
violation(S, R, λ) = maximum(abs.(S * R - I)) / λ - 1

function run(dir, outdir, names)
  prob = read_problem(dir)
  name = basename(rstrip(dir, '/'))
  S, λs, p = prob.S, prob.λ, prob.p
  nt = Threads.nthreads()
  nt == 1 && BLAS.set_num_threads(1)
  mkpath(outdir)
  # compile everything on a small problem first
  small = read_problem(dir).S[1:min(p, 30), 1:min(p, 30)]
  for nm in names
    redirect_stderr(devnull) do
      clime(small, [0.5, 0.3], SOLVERS[nm](); ntasks=nt)
    end
  end
  for nm in names
    solver = SOLVERS[nm]()
    rows = String[]
    for mode in ("single", "path")
      p > get(MAXP, (nm, mode), typemax(Int)) && continue
      grid = mode == "single" ? λs[end:end] : λs
      GC.gc()
      local path
      stats = @timed path = clime(S, grid, solver; symmetrize=false, ntasks=nt)
      t = stats.time - stats.gctime
      nsolved = length(path.λ)
      viol = maximum(violation(S, path[k], path.λ[k]) for k in 1:nsolved; init=-Inf)
      nnzmin = nsolved == length(grid) ? nnz(path[end]) : -1
      push!(rows, @sprintf("%s,%s,%d,%d,%s,%d,%s,%.6f,%.6f,%d,%d,%.3e,%d", name, prob.model, p, prob.n, nm, nt,
        mode, t, stats.gctime, stats.bytes, nsolved, viol, nnzmin))
      if mode == "path"
        L = fill(NaN, p, length(λs))
        for k in 1:nsolved
          L[:, findfirst(==(path.λ[k]), λs)] = l1cols(path[k])
        end
        writebin(joinpath(outdir, "$(name)__$(nm)__l1.bin"), L)
      end
      @printf("%-28s %-18s T%-2d %-6s %9.3fs  viol %.1e\n", name, nm, nt, mode, t, viol)
      flush(stdout)
    end
    open(joinpath(outdir, "$(name)__$(nm)__T$(nt).csv"), "w") do io
      foreach(r -> println(io, r), rows)
    end
  end
end

run(ARGS[1], ARGS[2], ARGS[3:end])
