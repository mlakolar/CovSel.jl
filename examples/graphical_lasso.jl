# The graphical lasso: estimate a sparse precision matrix from data and follow
# how well its support is recovered along a path of penalties.
#
# Run from the repository root (see examples/README.md for the setup):
#
#   julia --project=examples examples/graphical_lasso.jl

using CovarianceSelection
using LinearAlgebra, Printf, Random, Statistics
using UnicodePlots

Random.seed!(1)

# A random sparse precision matrix Ω, rescaled so that every variable has
# variance one (the scale on which λ ≍ sqrt(log p / n) is the right size),
# and n observations from N(0, Ω⁻¹).
p, n = 50, 400
Ω = CovarianceSelection.generateSparseSPD(p)
d = sqrt.(diag(inv(Ω)))
Ω = d .* Ω .* d'
data = CovarianceSelection.generateData(Ω, n)
S = cov(data)
println("p = $p, n = $n, edges in the true graph: ", count(!iszero, tril(Ω, -1)))

# One fit. covsel! solves
#
#   minimize  tr(SΩ) - log det Ω + λ‖Ω‖₁
#
# by ADMM. X, Z and U are the ADMM iterates; the estimate is the sparse
# iterate Z. Pass the iterates of a previous fit to warm start. The rate
# λ = sqrt(log p / n) is a starting point, not a tuned value; it selects more
# edges than there are (the path below shows the trade-off).
λ = sqrt(log(p) / n)
X, Z, U = zeros(p, p), zeros(p, p), zeros(p, p)
covsel!(X, Z, U, S, λ)

# Edges are the nonzero off-diagonal entries, hence diagonal=false.
t = CovarianceSelection.getConfusionMatrix(Ω, Z; diagonal=false)
@printf("\nλ = %.3f: %d estimated edges, TPR = %.2f, FPR = %.3f, precision = %.2f\n",
  λ, t.tp + t.fp, CovarianceSelection.tpr(t), CovarianceSelection.fpr(t), CovarianceSelection.precision(t))
@printf("relative error ‖Ẑ - Ω‖_F / ‖Ω‖_F = %.3f\n", norm(Z - Ω) / norm(Ω))

# The path. covselpath warm starts each fit from the previous one, so run it
# from the largest λ down. With λ above the largest off-diagonal |S[i,j]| the
# estimate is diagonal.
λmax = maximum(abs(S[i, j]) for j=1:p for i=1:p if i != j)
λarr = λmax .* exp.(range(0, log(0.02), length=25))
path = covselpath(S, λarr)

println("\n     λ   edges   TPR    FPR   precision")
tprs, fprs = Float64[], Float64[]
for (λ, Z) in zip(λarr, path)
  c = CovarianceSelection.getConfusionMatrix(Ω, Z; diagonal=false)
  push!(tprs, CovarianceSelection.tpr(c)); push!(fprs, CovarianceSelection.fpr(c))
  @printf("%6.3f  %5d   %.2f   %.3f   %.2f\n", λ, c.tp + c.fp, CovarianceSelection.tpr(c), CovarianceSelection.fpr(c), CovarianceSelection.precision(c))
end

println()
println(lineplot([0; fprs; 1], [0; tprs; 1]; xlim=(0, 1), ylim=(0, 1), xlabel="FPR", ylabel="TPR",
  title="Edge recovery along the path", width=50, height=12))

# penalize_diag=false leaves the diagonal unpenalized, so that the estimate is
# not shrunk towards zero on the diagonal.
Z = zeros(p, p)
covsel!(zeros(p, p), Z, zeros(p, p), S, λ; penalize_diag=false)
@printf("\nunpenalized diagonal: relative error %.3f\n", norm(Z - Ω) / norm(Ω))
