# Benchmark suite for CovSel. Defines `SUITE`, a BenchmarkTools.BenchmarkGroup,
# in the format used by PkgBenchmark.jl. run.jl runs it and prints a table; see
# README.md.
#
# Groups:
#   "direct difference": the three solvers of the ℓ1 penalized direct difference
#     estimator (the comparison of the removed example1.jl), with and without
#     warm starts, and iterative hard thresholding
#   "ADMM": the graphical lasso and the fused graphical lasso

using BenchmarkTools
using CovSel, CoordinateDescent, ProximalBase
using LinearAlgebra, Random, Statistics

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
    g["Alt.differencePrecisionNaive"] =
      @benchmarkable CovSel.Alt.differencePrecisionNaive($Sxd, $Syd, $λ, $ω)
    g["Alt.differencePrecisionActiveShooting"] =
      @benchmarkable CovSel.Alt.differencePrecisionActiveShooting($Sxd, $Syd, $λ, $ω)
    g["differencePrecisionActiveShooting"] =
      @benchmarkable differencePrecisionActiveShooting($Sx, $Sy, ProxL1($λ))
    g["differencePrecisionActiveShooting, warmStart=false"] =
      @benchmarkable differencePrecisionActiveShooting($Sx, $Sy, ProxL1($λ), CDOptions(; warmStart=false))
  end
  # at most 2p - 1 nonzero entries in the lower triangle, the size of the true support
  η = 1 / (opnorm(Sx) * opnorm(Sy))
  SUITE["direct difference"]["p = $p, IHT"] = BenchmarkGroup()
  SUITE["direct difference"]["p = $p, IHT"]["differencePrecisionIHT"] =
    @benchmarkable differencePrecisionIHT($Sx, $Sy, $η, $(2p - 1))
end

SUITE["ADMM"] = BenchmarkGroup()
for p in DIMENSIONS
  Sx, Sy = Matrix.(PROBLEMS[p])
  g = SUITE["ADMM"]["p = $p"] = BenchmarkGroup()
  # each sample starts from zero iterates, since covsel! warm starts from them
  g["covsel!"] = @benchmarkable covsel!(X, Z, U, $Sy, 0.1) setup=(X = zeros($p, $p); Z = zeros($p, $p); U = zeros($p, $p)) evals=1
  g["fusedGraphicalLasso"] = @benchmarkable fusedGraphicalLasso($Sx, 1000, $Sy, 1000, 0.05, 0.05)
end
