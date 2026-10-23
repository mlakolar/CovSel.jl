# Graphs that vary over time: observations at times t ∈ [0, 1] from a chain
# graph whose edges weaken, with one more edge from t = 0.5 on. tvCovSel
# estimates the graphs at a few time points jointly, from kernel-weighted
# covariances; compare it with the graphical lasso of each time point alone.
#
# Run from the repository root (see examples/README.md for the setup):
#
#   julia --project=examples examples/time_varying_graphs.jl
#
# COVSEL_EXAMPLES_QUICK=1 selects a smaller setting.

using CovarianceSelection
using LinearAlgebra, Printf, Random, Statistics

const QUICK = get(ENV, "COVSEL_EXAMPLES_QUICK", "0") == "1"
Random.seed!(3)

# the precision matrix at time t: a chain whose edges go from 0.4 to 0.2, and
# the edge (1, p) that appears at t = 0.5; diagonally dominant, so positive
# definite
p, n = QUICK ? (10, 1000) : (30, 4000)
function Ω(t)
  W = Matrix(1.0I, p, p)
  for i in 1:p-1
    W[i, i+1] = W[i+1, i] = 0.4 - 0.2t
  end
  t >= 0.5 && (W[1, p] = W[p, 1] = 0.25)
  W
end

t = sort(rand(n))
X = zeros(n, p)
for i in 1:n
  X[i, :] = cholesky(Ω(t[i])).U \ randn(p)      # a draw from N(0, Ω(tᵢ)⁻¹)
end

# kernel smoothing at τ with the Epanechnikov kernel and bandwidth h: each τ
# sees about 2h n observations
τ = [0.1, 0.3, 0.5, 0.7, 0.9]
h = 0.15
S = kernelCovariances(X, t, τ, h)
λ = 2 * sqrt(log(p) / (2h * n))

# jointly, with the group penalty on each entry across the K time points,
# against the graphical lasso of each time point alone. The joint objective
# weights each time point by 1/K, and an entry equal to a at the K time points
# has the penalty λ √K |a|, √K times its average over the separate fits; so
# the joint fit takes λ/√K, which penalizes such an entry as they do
K = length(τ)
Z = tvCovSel(X, t, τ, h, λ / sqrt(K))
Zsep = similar(Z)
for k in eachindex(τ)
  Zk = zeros(p, p)
  covsel!(zeros(p, p), Zk, zeros(p, p), S[:, :, k], λ)
  Zsep[:, :, k] = Zk
end

println("p = $p, n = $n, h = $h, λ = ", round(λ; digits = 3), "\n")
println("   τ   true edges   joint: edges  TPR   FPR   ‖Ẑ - Ω‖   separate: edges  TPR   FPR   ‖Ẑ - Ω‖")
for k in eachindex(τ)
  truth = Ω(τ[k])
  cj = CovarianceSelection.getConfusionMatrix(truth, Z[:, :, k]; diagonal = false)
  cs = CovarianceSelection.getConfusionMatrix(truth, Zsep[:, :, k]; diagonal = false)
  @printf("%4.1f   %10d   %12d  %.2f  %.3f  %7.3f   %15d  %.2f  %.3f  %7.3f\n", τ[k], count(!iszero, tril(truth, -1)),
    count(x -> abs(x) > 1e-4, tril(Z[:, :, k], -1)), CovarianceSelection.tpr(cj), CovarianceSelection.fpr(cj),
    norm(Z[:, :, k] - truth),
    count(x -> abs(x) > 1e-4, tril(Zsep[:, :, k], -1)), CovarianceSelection.tpr(cs), CovarianceSelection.fpr(cs),
    norm(Zsep[:, :, k] - truth))
end
@printf("\nthe edge (1, %d), which appears at t = 0.5, in the joint estimates: %s\n", p,
  join((@sprintf("%.3f", Z[1, p, k]) for k in eachindex(τ)), ", "))
println("""
The group penalty keeps an entry at all the time points or at none, so the
joint estimates have the same edges at every time point: the chain and a few
false edges. The separate fits select their edges independently, with more
false edges at some time points and fewer at others. An edge that exists only
part of the time, like (1, $p), is kept at all the time points, with values
near zero where it is absent: the support of the joint estimate is the union
of the supports over time, and kernel smoothing blurs the change at t = 0.5.
""")
