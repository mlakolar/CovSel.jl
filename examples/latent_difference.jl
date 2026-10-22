# A difference of two precision matrices that is sparse plus low rank, as when
# one group has a latent factor that the other lacks. It is estimated by
# iterative hard thresholding (differenceLatentPrecisionIHT) and compared with
# the sparse direct difference estimator.
#
# Run from the repository root (see examples/README.md for the setup):
#
#   julia --project=examples examples/latent_difference.jl

using CovSel, ProximalBase
using LinearAlgebra, Printf, Random, Statistics
using UnicodePlots

Random.seed!(3)

# Ωx is a chain graph. In group y two edges change and a latent variable is
# marginalized out, which subtracts a positive semidefinite rank one matrix L
# from the precision matrix. So Δ = Ωx - Ωy = S + L with S sparse.
p = 30
Ωx = Matrix(1.0I, p, p)
for i=1:p-1
  Ωx[i, i+1] = Ωx[i+1, i] = 0.3
end
S = zeros(p, p)
S[4, 3] = S[3, 4] = 0.3         # edge (3, 4) is absent in group y
S[20, 10] = S[10, 20] = -0.3    # edge (10, 20) is present only in group y
L = Matrix(CovSel.genLatentModel(Ωx, 1, 0.2)[2])   # random, rank one, ‖L‖ = 0.2 λmax(Ωx)
Ωy = Ωx - S - L
@assert isposdef(Symmetric(Ωy))

n = 5000
Sx = Symmetric(cov(CovSel.generateData(Ωx, n)))
Sy = Symmetric(cov(CovSel.generateData(Ωy, n)))
@printf("p = %d, n = %d per group, ‖S‖_F = %.3f, ‖L‖_F = %.3f, largest |L[i,j]| = %.3f\n\n",
  p, n, norm(S), norm(L), maximum(abs, L))

# differenceLatentPrecisionIHT minimizes the direct difference loss
#
#   ℓ(Θ) = tr(Sx Θ Sy Θ) / 2 - tr(Θ (Sy - Sx)),   Θ = Δ + UU',
#
# by gradient steps on Δ, each followed by keeping the s largest entries, and
# on U. The gradient of ℓ is Lipschitz with constant ‖Sx‖‖Sy‖, which gives the
# step size for Δ. As a function of U the constant is about 4‖Sx‖‖Sy‖‖U‖², so
# the step for U is scaled by 1 / (4‖L₀‖), with L₀ = U₀U₀' the starting value.
# The sparsity s and the rank r are tuning parameters; here they are the truth.
s, r = 2, 1
ηΔ = 1 / (opnorm(Sx) * opnorm(Sy))

# The starting values hard threshold inv(Sx) - inv(Sy) and take the top r
# singular vectors of what is left; the in-place solver continues from them.
Δ, L̂, U, gD, gU = CovSel.differenceLatentPrecisionIHT_init(Sx, Sy, s, r)
ηU = ηΔ / (4 * opnorm(L̂))

# The callback is called after every iteration with the current Δ and L; here
# it records the size of each step, ‖Θₖ - Θₖ₋₁‖_F, to monitor convergence.
steps = Float64[]
Θprev = Ref(Matrix(Δ) + L̂)
function recordStep(Δ, L)
  Θ = Matrix(Δ) + L
  push!(steps, norm(Θ - Θprev[]))
  Θprev[] = Θ
end
Δ, L̂, U, fvals, iter = differenceLatentPrecisionIHT!(Δ, L̂, U, gD, gU, Sx, Sy, ηΔ, ηU, s, r;
  options=IHTOptions(; epsTol=1e-10, maxIter=10_000, checkEvery=10), callback=recordStep)
# (differenceLatentPrecisionIHT(Sx, Sy, ηΔ, ηU, s, r) does both steps in one call)

support = sort([ProximalBase.ind2subLowerTriangular(p, Δ.nzval2ind[k]) for k=1:nnz(Δ)])
u, û = eigvecs(Symmetric(L))[:, end], U[:, 1] / norm(U)
println("sparse plus low rank, ", iter, " iterations")
println("  support of Δ̂: ", support)
@printf("  ‖Δ̂ - S‖_F = %.3f, ‖L̂ - L‖_F = %.3f, |cos ∠(û, u)| = %.3f\n", norm(Matrix(Δ) - S), norm(L̂ - L), abs(dot(û, u)))
@printf("  ‖Δ̂ + L̂ - (S + L)‖_F = %.3f\n\n", norm(Matrix(Δ) + L̂ - S - L))

println(lineplot(1:length(steps), log10.(steps); xlabel="iteration", ylabel="log10 step",
  title="Convergence", width=50, height=10))

# The sparse estimator has no low rank part, so it either leaves L out or
# spreads it over many small entries.
λ = 2 * sqrt(log(p) / n)
Δsparse = differencePrecisionActiveShooting(Sx, Sy, ProxL1(λ))
@printf("\nsparse only (λ = %.3f): %d nonzero entries in the lower triangle, ‖Δ̂ - (S + L)‖_F = %.3f\n",
  λ, nnz(Δsparse), norm(Matrix(Δsparse) - S - L))
