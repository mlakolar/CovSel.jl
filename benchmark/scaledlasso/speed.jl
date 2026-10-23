# Time the scaled lasso estimator on one problem of ../clime/problems.jl.
#
#   julia -t T --project=benchmark/scaledlasso benchmark/scaledlasso/speed.jl MODEL P N OUTDIR [threads]
#
# The data X (n × p, mean zero) are drawn with the seed 7 + p + n, and S = X'X/n.
# Times scaledlasso (λ₀ = :quantile, with and without the least squares refit) on
# one thread, the best of up to 5 runs; with `threads`, on all of Julia's threads
# instead. Without `threads`, also writes X.bin and the estimate est.bin for the
# R package scalreg (bench_r.R), when p ≤ RPMAX (default 1000), and times the
# scaled lasso of CoordinateDescent column by column, when p ≤ CDPMAX (default
# 1000). Appends a line to OUTDIR/julia.csv:
# model,p,n,method,threads,seconds,nnz,maxdiff
# where maxdiff is the largest difference to scaledlasso, relative to its largest
# entry.

include(joinpath(@__DIR__, "common.jl"))
using CoordinateDescent

const RPMAX = parse(Int, get(ENV, "RPMAX", "1000"))
const CDPMAX = parse(Int, get(ENV, "CDPMAX", "1000"))

function best_time(f; runs=5, budget=2.0)
  best, total = Inf, 0.0
  v = nothing
  for _ in 1:runs
    t = @elapsed v = f()
    best = min(best, t)
    total += t
    total > budget && break
  end
  v, best
end

# column j of the precision matrix from CoordinateDescent's scaled lasso of x_j on
# the other columns, with the penalty weights √Sₖₖ of scaledlasso
function cd_scaledlasso(X, λ₀)
  n, p = size(X)
  w = vec(sqrt.(sum(abs2, X; dims=1) / n))
  opts = IterLassoOptions(maxIter=100, optTol=1e-6, optionsCD=CDOptions(optTol=1e-7))
  cols = Vector{SparseVector{Float64,Int}}(undef, p)
  for j in 1:p
    others = [1:j-1; j+1:p]
    sol = scaledLasso(X[:, others], X[:, j], λ₀, w[others], opts)
    c = zeros(p)
    c[others] = -Vector(sol.x) ./ sol.σ^2
    c[j] = 1 / sol.σ^2
    cols[j] = sparse(c)
  end
  CovSel.symmetrize_minabs(sparse(reduce(hcat, cols)))
end

function main(model, p, n, outdir, mode)
  rng = Xoshiro(7 + p + n)
  Ω = precision_matrix(model, p; rng=rng)
  X = sample(Ω, n; rng=rng)
  S = X' * X / n
  λ₀ = CovSel._penalty_level(:quantile, p, n)
  scaledlasso(S[1:10, 1:10], n)                      # compile
  mkpath(outdir)
  io = open(joinpath(outdir, "julia.csv"), "a")
  line(method, nt, t, nz, d) = (@printf(io, "%s,%d,%d,%s,%d,%.6g,%d,%.3g\n", model, p, n, method, nt, t, nz, d); flush(io))
  if mode == "threads"
    nt = Threads.nthreads()
    Ω1, t = best_time(() -> scaledlasso(S, n; ntasks=nt))
    line("scaledlasso", nt, t, nnz(Ω1), 0.0)
  else
    Ω1, t = best_time(() -> scaledlasso(S, n; ntasks=1))
    line("scaledlasso", 1, t, nnz(Ω1), 0.0)
    Ωr, t = best_time(() -> scaledlasso(S, n; refit=true, ntasks=1))
    line("scaledlasso, refit", 1, t, nnz(Ωr), 0.0)
    if p <= RPMAX
      dir = joinpath(outdir, problem_name(model, p, n, 7 + p + n))
      mkpath(dir)
      writebin(joinpath(dir, "X.bin"), X)
      writebin(joinpath(dir, "est.bin"), Matrix(Ω1))
      open(joinpath(dir, "lambda0.txt"), "w") do f
        println(f, repr(λ₀))
      end
    end
    if p <= CDPMAX
      cd_scaledlasso(X[1:20, 1:10], 0.3)                # compile
      Ωc, t = best_time(() -> cd_scaledlasso(X, λ₀); runs=1)
      line("CoordinateDescent", 1, t, nnz(Ωc), maximum(abs, Ωc - Ω1) / maximum(abs, Ω1))
    end
  end
  close(io)
end

main(ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]), ARGS[4], get(ARGS, 5, "single"))
