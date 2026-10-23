# The solvers of CovSel before and after the move onto ProximalSolvers, run by the
# version in the active environment: old/ (CovSel master with its own
# coordinate descent loss and ADMM loop, on CoordinateDescent 0.6.0 and
# ProximalBase 0.4.1) or new/ (this checkout, on ProximalSolvers 0.1 and
# ProximalBase 0.5). The data come from Random and LinearAlgebra only, so that
# both sides solve the same problems. Each case runs once to compile, then
# until it has at least 5 runs and half a second of them, each from the same
# seed. Writes the median times, the passes (the iterations of ADMM and IHT)
# and the solutions to results/<side>_<stamp>.jls; compare.jl compares the
# two sides.
#
#     julia --project=benchmark/compare/old benchmark/compare/run.jl old [regex]
#     julia --project=benchmark/compare/new benchmark/compare/run.jl new [regex]

using CovSel, ProximalBase, LinearAlgebra, Printf, Random, Serialization, SparseArrays, Statistics
BLAS.set_num_threads(1)

const SIDE = ARGS[1]
const OLD = SIDE == "old"
const ONLY = Regex(length(ARGS) > 1 ? ARGS[2] : ".")
const SEED = 20261023

# the coordinate descent of differencePrecisionActiveShooting, with its passes
if OLD
  using CoordinateDescent
  function dtrace(Sx, Sy, λ, order)
    x = SymmetricSparseIterate(size(Sx, 1))
    info = CDInfo()
    coordinateDescent!(x, CDDirectDifferenceLoss(Sx, Sy), ProxL1(λ), CDOptions(; order), info)
    x, info.iterations
  end
else
  using ProximalSolvers
  function dtrace(Sx, Sy, λ, order)
    x = SymmetricSparseIterate(size(Sx, 1))
    r = solve!(x, directDifferenceLoss(Sx, Sy), NormL1(λ), CDSolver(; order))
    x, r.passes
  end
end

# Two samples of n observations, X from N(0, I) and Y from N(0, Σy) with the
# AR(1) covariance Σy, ρ = 0.7, as in benchmark/benchmarks.jl. The difference
# of the precision matrices, Δ = I - inv(Σy), is tridiagonal.
function difference_problem(p; n = 1000)
  rng = Xoshiro(p)
  Σy = [0.7^abs(i - j) for i=1:p, j=1:p]
  X = randn(rng, n, p)
  Y = randn(rng, n, p) * cholesky(Σy).U
  Δ = [abs(i - j) <= 1 ? (I - inv(Σy))[i, j] : 0.0 for i=1:p, j=1:p]
  (Sx = Symmetric(cov(X)), Sy = Symmetric(cov(Y)), Δ = (Δ + Δ') / 2)
end

# the smallest penalty level with a zero direct difference estimate
dtrace_λmax(Sx, Sy) = maximum(abs, Sy - Sx)

const results = Dict{String, Any}()

function run_case(name, f)
  occursin(ONLY, name) || return
  Random.seed!(SEED)
  f()
  times = Float64[]
  local sol
  while length(times) < 5 || sum(times) < 0.5
    Random.seed!(SEED)
    t = @elapsed sol = f()
    push!(times, t)
  end
  results[name] = (order = length(results), time = median(times), passes = sol[2], solution = sol[1])
  @printf("%-46s %10.3f ms %7d\n", name, 1e3 * median(times), sol[2])
end

function dtrace_cases()
  for p in (20, 50, 100)
    P = difference_problem(p)
    λmax = dtrace_λmax(P.Sx, P.Sy)
    for frac in (0.5, 0.2)
      run_case("dtrace p=$p λ=$(frac)λmax random",
               () -> ((x, passes) = dtrace(P.Sx, P.Sy, frac * λmax, :random); (vec(Matrix(x)), passes)))
    end
  end
  P = difference_problem(50)
  λ = 0.2 * dtrace_λmax(P.Sx, P.Sy)
  run_case("dtrace p=50 λ=0.2λmax cyclic", () -> ((x, passes) = dtrace(P.Sx, P.Sy, λ, :cyclic); (vec(Matrix(x)), passes)))
end

# the refit on the tridiagonal support of Δ, by coordinate descent with the
# entries off the support penalized by 1e10 (old) or exactly (new)
function refit_cases()
  for p in (50, 100)
    P = difference_problem(p)
    S = [LinearIndices((p, p))[r, c] for c=1:p for r=1:p if abs(r - c) <= 1]
    run_case("differencePrecisionRefit p=$p", () -> (vec(Matrix(differencePrecisionRefit(P.Sx, P.Sy, S))), 0))
  end
end

function admm_cases()
  for p in (50, 100)
    S = Matrix(difference_problem(p).Sy)
    run_case("covsel! p=$p λ=0.1", () -> begin
      X, Z, U, info = zeros(p, p), zeros(p, p), zeros(p, p), ADMMInfo()
      covsel!(X, Z, U, S, 0.1; info)
      (vec(Z), info.iterations)
    end)
  end
  P = difference_problem(50)
  S = Matrix(P.Sy)
  run_case("covsel! p=50 λ=0.1 penalize_diag=false", () -> begin
    X, Z, U, info = zeros(50, 50), zeros(50, 50), zeros(50, 50), ADMMInfo()
    covsel!(X, Z, U, S, 0.1; penalize_diag=false, info)
    (vec(Z), info.iterations)
  end)
  run_case("covselpath p=50 4λ", () -> (reduce(vcat, vec.(covselpath(S, [0.5, 0.2, 0.1, 0.05]))), 0))
  run_case("fusedGraphicalLasso p=50", () -> begin
    info = ADMMInfo()
    Zx, Zy = fusedGraphicalLasso(Matrix(P.Sx), 1000, S, 1000, 0.05, 0.05; info)
    (vcat(vec(Zx), vec(Zy)), info.iterations)
  end)
  rng = Xoshiro(3)
  n, p = 200, 100
  A1, A2 = randn(rng, n, p), randn(rng, n, p)
  β1 = [i <= 5 ? 1.0 : 0.0 for i=1:p]
  β2 = [2 <= i <= 5 ? 1.0 : 0.0 for i=1:p]
  b1, b2 = A1 * β1 + randn(rng, n), A2 * β2 + randn(rng, n)
  run_case("fusedNeighborhoodSelection n=200 p=100", () -> begin
    info = ADMMInfo()
    z1, z2 = fusedNeighborhoodSelection(A1, b1, A2, b2, 0.1, 0.1; info)
    (vcat(z1, z2), info.iterations)
  end)
end

function iht_cases()
  p = 50
  P = difference_problem(p)
  Sx, Sy = P.Sx, P.Sy
  s = 2p - 1
  η = 1 / (opnorm(Sx) * opnorm(Sy))
  run_case("differencePrecisionIHT p=50 s=$s", () -> ((Δ, fvals) = differencePrecisionIHT(Sx, Sy, η, s); (vec(Matrix(Δ)), length(fvals))))
  ηU = η / (4 * opnorm(CovSel.differenceLatentPrecisionIHT_init(Sx, Sy, s, 1)[2]))
  run_case("differenceLatentPrecisionIHT p=50 s=$s r=1", () -> begin
    Δ, L, U, fvals, iter = differenceLatentPrecisionIHT(Sx, Sy, η, ηU, s, 1; options=IHTOptions(; epsTol=1e-8, maxIter=500))
    (vcat(vec(Matrix(Δ)), vec(L)), iter)
  end)
end

function loss_cases()
  p = 100
  P = difference_problem(p)
  Δ = SymmetricSparseIterate(sparse(P.Δ))
  run_case("diffLoss p=100 tridiagonal Δ", () -> ([diffLoss(P.Sx, Δ, P.Sy, 2), diffLoss(P.Sx, Δ, P.Sy, Inf)], 0))
end

dtrace_cases()
refit_cases()
admm_cases()
iht_cases()
loss_cases()

mkpath(joinpath(@__DIR__, "results"))
file = joinpath(@__DIR__, "results", "$(SIDE)_$(round(Int, time())).jls")
serialize(file, results)
println("wrote ", file)
