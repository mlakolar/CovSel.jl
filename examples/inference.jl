# Inference for one entry of the difference of two precision matrices when
# p > n, checked by simulation: under the null hypothesis the standardized
# difference should be approximately N(0, 1).
#
# This ports the Gaussian case of ROCKET (Barber and Kolar, 2018, Annals of
# Statistics) from the unmaintained mlakolar/ROCKET.jl. For an edge (a, b) and
# one sample, regress the variables a and b on the others, select their
# neighborhoods with the square-root lasso, and refit by least squares on the
# union of the two neighborhoods. The residuals ε have covariance close to
# inv(Ω[[a, b], [a, b]]), so
#
#   Ω̂[[a, b], [a, b]] = inv(ε'ε / n),   Var(Ω̂[a, b]) ≈ (Ω[a, a] Ω[b, b] + Ω[a, b]²) / n.
#
# (Least squares on all p - 2 other variables, as in the original script, is
# not possible with p > n.)
#
# Run from the repository root (see examples/README.md for the setup):
#
#   julia --project=examples examples/inference.jl
#
# The default p = 500, n = 400 and 500 replications takes a few seconds;
# COVSEL_EXAMPLES_QUICK=1 selects a smaller setting.

using CovarianceSelection, CoordinateDescent
using LinearAlgebra, Printf, Random, Statistics
using UnicodePlots

const QUICK = get(ENV, "COVSEL_EXAMPLES_QUICK", "0") == "1"

Random.seed!(4)

# Banded precision matrix with random scales: Ω = D Ω⋆ D, with Ω⋆ equal to 1 on
# the diagonal, 0.6 on the first and 0.3 on the second off-diagonals.
function generatePrecision(p)
  D = Diagonal(sqrt.(0.5 .+ 2 .* rand(p)))
  Ωstar = diagm(0 => ones(p), 1 => fill(0.6, p - 1), -1 => fill(0.6, p - 1),
                2 => fill(0.3, p - 2), -2 => fill(0.3, p - 2))
  D * Ωstar * D
end

# Estimate Ω[a, b] and its variance from the rows of Z. sqrtLasso minimizes
# ‖y - Xβ‖ + λ ∑ⱼ ωⱼ|βⱼ|, with ωⱼ the root mean square of column j. Belloni,
# Chernozhukov and Wang (2011) take λ = 1.1 Φ⁻¹(1 - α / (2p)) on this scale;
# sqrt(2 log(2p / α)) is a slightly larger bound on the normal quantile.
function edgeInference(Z, a, b; λ = 1.1 * sqrt(2 * log(2 * size(Z, 2) / 0.05)))
  n, p = size(Z)
  others = setdiff(1:p, (a, b))
  X = Z[:, others]
  S = Int[]
  for j in (a, b)
    sol = sqrtLasso(X, Z[:, j], λ)
    append!(S, others[findall(x -> abs(x) > 1e-4, sol.x)])
  end
  unique!(S)
  Y = Z[:, [a, b]]
  ε = Y - Z[:, S] * (Z[:, S] \ Y)
  Ω̂ = inv(ε' * ε / n)
  (estimate = Ω̂[1, 2], variance = (Ω̂[1, 1] * Ω̂[2, 2] + Ω̂[1, 2]^2) / n, selected = length(S))
end

p, n, numRep = QUICK ? (50, 200, 100) : (500, 400, 500)
a, b = 1, 2
Ω = generatePrecision(p)
println("p = $p, n = $n per group, $numRep replications, edge ($a, $b) with Ω[a, b] = ",
  round(Ω[a, b], digits=3))

# Both groups are drawn from N(0, Ω⁻¹), so Ωx[a, b] - Ωy[a, b] = 0. Ω is
# factored once; with the factor, generateData only solves a triangular system.
F = cholesky(Symmetric(Ω))
T = zeros(numRep)
covered = 0
selected = 0
elapsed = @elapsed for rep=1:numRep
  data = CovarianceSelection.generateData(F, 2n)
  x = edgeInference(data[1:n, :], a, b)
  y = edgeInference(data[n+1:end, :], a, b)
  T[rep] = (x.estimate - y.estimate) / sqrt(x.variance + y.variance)
  global covered += abs(x.estimate - Ω[a, b]) <= 1.96 * sqrt(x.variance)
  global selected += x.selected
end

@printf("\n%.1f seconds; on average %.1f variables selected (the true union has 2)\n", elapsed, selected / numRep)
@printf("standardized difference T: mean %.3f, standard deviation %.3f\n", mean(T), std(T))
@printf("size of the 5%% test that rejects when |T| > 1.96: %.3f (± %.3f)\n",
  mean(abs.(T) .> 1.96), 2 * sqrt(0.05 * 0.95 / numRep))
@printf("coverage of the 95%% interval for Ωx[a, b]: %.3f\n\n", covered / numRep)

# histogram of T against the N(0, 1) density
edges = range(-4, 4, length=33)
density = [count(t -> edges[k] <= t < edges[k+1], T) for k=1:32] ./ (numRep * step(edges))
plt = stairs(collect(edges[1:end-1]), density; style=:post, name="T", xlim=(-4, 4),
  title="Standardized difference under the null", width=60, height=14)
xs = collect(range(-4, 4, length=200))
lineplot!(plt, xs, exp.(-xs .^ 2 ./ 2) ./ sqrt(2π); name="N(0, 1)")
println(plt)
