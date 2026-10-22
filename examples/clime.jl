# CLIME and ACLIME: sparse precision matrices by constrained ℓ1 minimization, a
# path of penalties, cross validation, and the choice of solver.
#
# Run from the repository root (see examples/README.md for the setup):
#
#   julia --project=examples examples/clime.jl
#
# For the column problems to be solved in parallel, start Julia with threads:
# julia -t auto --project=examples examples/clime.jl

using CovSel
using LinearAlgebra, Printf, Random, SparseArrays, Statistics
using UnicodePlots

Random.seed!(1)

# A band precision matrix (as in Cai, Liu and Zhou, 2016) with unequal
# variances, and n observations from N(0, Ω⁻¹).
p, n = 100, 200
Ω1 = Matrix(1.0I, p, p)
for i in 1:p-1; Ω1[i, i+1] = Ω1[i+1, i] = 0.6; end
for i in 1:p-2; Ω1[i, i+2] = Ω1[i+2, i] = 0.3; end
d = 1 .+ rand(p)                       # standard deviations between 1 and 2
Ω = Ω1 ./ (d * d')                     # Σ = D Ω1⁻¹ D
X = CovSel.generateData(Ω, n)
println("p = $p, n = $n, edges in the true graph: ", count(!iszero, tril(Ω, -1)))

edges(A) = count(!iszero, tril(A, -1))
function report(name, Ωhat)
  c = CovSel.getConfusionMatrix(Ω, Ωhat; diagonal=false)
  @printf("%-28s %5d edges  TPR %.2f  FPR %.3f  spectral error %.3f\n", name, edges(Ωhat),
    CovSel.tpr(c), CovSel.fpr(c), opnorm(Matrix(Ωhat - Ω)))
end

# CLIME. Column i of the estimate solves
#
#   minimize ‖β‖₁ subject to ‖Sβ - eᵢ‖∞ ≤ λ,
#
# and the columns are symmetrized. The penalty is in the units of SΩ, so CLIME is
# run on the correlation matrix R = D⁻¹SD⁻¹ and the estimate rescaled back.
S = cov(X)
s = sqrt.(diag(S))
R = S ./ (s * s')
rescale(Θ) = Θ ./ (s * s')
λ = sqrt(log(p) / n)
report("CLIME, λ = $(round(λ; digits=3))", rescale(clime(R, λ)))

# A path. The parametric simplex method follows the exact solution path of each
# column, so the whole path costs about as much as its smallest penalty alone.
λs = exp.(range(log(0.6), log(0.05); length=30))
elapsed = @elapsed path = clime(R, λs)
@printf("\npath of %d penalties in %.2f s:\n", length(λs), elapsed)
println(lineplot(path.λ, edges.(path); xscale=:log10, xlabel="λ", ylabel="edges", width=50, height=10))

# Cross validation of λ on the data. The estimates of the training folds are
# scored by the Gaussian likelihood of the held out fold. The data are
# standardized first, so that λ is on the scale of the correlation matrix.
Z = (X .- mean(X; dims=1)) ./ std(X; dims=1)
cv = climecv(Z, λs; folds=5)
λbest = cv.λ[cv.best]
report("CLIME, λ = $(round(λbest; digits=3)) by CV", rescale(bestestimate(cv)))

# ACLIME (Cai, Liu and Zhou, 2016) scales the constraint of each entry by its
# standard deviation, estimated in a first step, so it adapts to the scales of
# the variables. It takes the unbiased sample covariance and n; the theory's
# universal constant δ = 2 gives a sparse, conservative estimate, and Cai, Liu
# and Zhou choose δ by validation in their simulations.
report("ACLIME, δ = 2", aclime(S, n))
cva = aclimecv(X, (1:40) ./ 20)
report("ACLIME, δ = $(cva.λ[cva.best]) by CV", bestestimate(cva))

# The solvers all give the same estimate, the exact solution of the linear
# programs, except PDHG, a first order method accurate to its tolerance (most of
# its columns are exact, after polishing). It is much slower, and slower still
# for small penalties. LPSolver needs a MathOptInterface solver, for example
# `using HiGHS; clime(R, λ, LPSolver(HiGHS.Optimizer))`.
println()
λs10 = λs[λs .>= 0.1]
reference = clime(R, λs10)
for solver in (ParametricSimplex(), DualSimplex(), PDHG())
  secs = @elapsed Θ = clime(R, λs10, solver)
  @printf("%-20s %6.3f s  largest difference from ParametricSimplex %.1e\n",
    nameof(typeof(solver)), secs, maximum(maximum(abs, Θ[k] - reference[k]) for k in eachindex(λs10)))
end
