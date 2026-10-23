# Benchmark suite for CovarianceSelection. Defines `SUITE`, a BenchmarkTools.BenchmarkGroup,
# in the format used by PkgBenchmark.jl. run.jl runs it and prints a table; see
# README.md.
#
# Groups:
#   "direct difference": the ℓ1 penalized direct difference estimator solved by
#     differencePrecisionActiveShooting, directly and along a path of λ values
#     from λmax (ProximalSolvers' continuation), and by the naive reference solver
#     in test/reference.jl
#     (the comparison of the removed example1.jl); iterative hard
#     thresholding, sparse and sparse plus rank one
#   "ADMM": the graphical lasso and the fused graphical lasso
#   "CLIME": clime along a path of 20 penalties and at one penalty, aclime, and
#     slassoinv (the comparison of the CLIME solvers is in clime/, the
#     benchmark of slassoinv in slassoinv/)
#   "generateData": drawing samples, from Ω and from its Cholesky factor

using BenchmarkTools
using CovarianceSelection, ProximalBase, ProximalSolvers
using LinearAlgebra, Random, Statistics

include(joinpath(@__DIR__, "..", "test", "reference.jl"))

const QUICK = get(ENV, "COVSEL_BENCHMARK_QUICK", "0") == "1"
const DIMENSIONS = QUICK ? (10,) : (20, 50, 100)
const LAMBDAS = QUICK ? (0.2,) : (0.1, 0.2)

# Sample covariances of n observations from two groups with Σx = I and Σy the
# AR(1) covariance with ρ = 0.7, as in example1.jl. The difference of the
# precision matrices, I - inv(Σy), is tridiagonal.
function differenceProblem(p; n=1000)
  Random.seed!(p)
  Σy = [0.7^abs(i - j) for i=1:p, j=1:p]
  Symmetric(cov(randn(n, p))), Symmetric(cov(randn(n, p) * cholesky(Σy).U))
end

const PROBLEMS = Dict(p => differenceProblem(p) for p in DIMENSIONS)

const SUITE = BenchmarkGroup()

SUITE["direct difference"] = BenchmarkGroup()
for p in DIMENSIONS
  Sx, Sy = PROBLEMS[p]
  Sxd, Syd = Matrix(Sx), Matrix(Sy)
  ω = ones(p, p)
  for λ in LAMBDAS
    g = SUITE["direct difference"]["p = $p, λ = $λ"] = BenchmarkGroup()
    g["Reference.differencePrecisionNaive"] =
      @benchmarkable Reference.differencePrecisionNaive($Sxd, $Syd, $λ, $ω)
    g["differencePrecisionActiveShooting"] =
      @benchmarkable differencePrecisionActiveShooting($Sx, $Sy, NormL1($λ))
    g["path of 50 λ from λmax"] =
      @benchmarkable solve!(SymmetricSparseIterate($p), directDifferenceLoss($Sx, $Sy), NormL1($λ), CDSolver(); continuation=50)
  end
  # at most 2p - 1 nonzero entries in the lower triangle, the size of the true
  # support; step sizes as in examples/latent_difference.jl
  s = 2p - 1
  η = 1 / (opnorm(Sx) * opnorm(Sy))
  ηU = η / (4 * opnorm(CovarianceSelection.differenceLatentPrecisionIHT_init(Sx, Sy, s, 1)[2]))
  g = SUITE["direct difference"]["p = $p, IHT"] = BenchmarkGroup()
  g["differencePrecisionIHT"] = @benchmarkable differencePrecisionIHT($Sx, $Sy, $η, $s)
  g["differenceLatentPrecisionIHT, r = 1"] = @benchmarkable differenceLatentPrecisionIHT($Sx, $Sy, $η, $ηU, $s, 1;
    options=IHTOptions(; epsTol=1e-8, maxIter=500))
end

SUITE["ADMM"] = BenchmarkGroup()
for p in DIMENSIONS
  Sx, Sy = Matrix.(PROBLEMS[p])
  g = SUITE["ADMM"]["p = $p"] = BenchmarkGroup()
  # each sample starts from zero iterates, since covsel! warm starts from them
  g["covsel!"] = @benchmarkable covsel!(X, Z, U, $Sy, 0.1) setup=(X = zeros($p, $p); Z = zeros($p, $p); U = zeros($p, $p)) evals=1
  g["fusedGraphicalLasso"] = @benchmarkable fusedGraphicalLasso($Sx, 1000, $Sy, 1000, 0.05, 0.05)
end

SUITE["CLIME"] = BenchmarkGroup()
for p in DIMENSIONS
  Sx, Sy = PROBLEMS[p]
  R = Matrix(Sy)
  R ./= sqrt.(diag(R)) * sqrt.(diag(R))'
  λs = exp.(range(log(0.6), log(0.1); length=20))
  g = SUITE["CLIME"]["p = $p"] = BenchmarkGroup()
  g["clime, path of 20"] = @benchmarkable clime($R, $λs; ntasks=1)
  g["clime, λ = 0.1"] = @benchmarkable clime($R, 0.1; ntasks=1)
  g["aclime"] = @benchmarkable aclime($(Matrix(Sy)), 1000; ntasks=1)
  g["slassoinv"] = @benchmarkable slassoinv($(Matrix(Sy)), 1000; ntasks=1)
end

SUITE["generateData"] = BenchmarkGroup()
for p in DIMENSIONS
  Ω = Matrix(SymTridiagonal(fill(1.0, p), fill(0.3, p - 1)))
  F = cholesky(Symmetric(Ω))
  g = SUITE["generateData"]["p = $p, n = 1000"] = BenchmarkGroup()
  g["from Ω"] = @benchmarkable CovarianceSelection.generateData($Ω, 1000)
  g["from the Cholesky factor"] = @benchmarkable CovarianceSelection.generateData($F, 1000)
end
