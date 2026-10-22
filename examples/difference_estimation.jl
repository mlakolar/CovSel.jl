# Direct estimation of the difference of two precision matrices that differ in
# one edge (Zhao, Cai and Li, 2014), compared with taking the difference of
# two graphical lasso estimates and with the fused graphical lasso.
#
# Run from the repository root (see examples/README.md for the setup):
#
#   julia --project=examples examples/difference_estimation.jl

using CovSel, ProximalBase
using LinearAlgebra, Printf, Random, Statistics

Random.seed!(123)

p, n = 20, 1000
Ωx = Matrix(1.0I, p, p)
Ωy = copy(Ωx)
Ωy[1, 2] = Ωy[2, 1] = 0.4
Δ = Ωx - Ωy

X = CovSel.generateData(Ωx, n)
Y = CovSel.generateData(Ωy, n)
Sx, Sy = Symmetric(cov(X)), Symmetric(cov(Y))
println("p = $p, n = $n per group, Δ = Ωx - Ωy has one nonzero entry, Δ[2, 1] = ", Δ[2, 1])

# differencePrecisionActiveShooting minimizes the direct difference loss
#
#   tr(Sx Δ Sy Δ) / 2 - tr(Δ (Sy - Sx)) + λ‖Δ‖₁
#
# by active shooting coordinate descent. With the population covariances the
# loss is minimized at Ωx - Ωy, so Δ is estimated without estimating Ωx or Ωy,
# which need not be sparse. The estimate is a SymmetricSparseIterate.
λ = 0.1
hΔ = differencePrecisionActiveShooting(Sx, Sy, ProxL1(λ))
c = CovSel.getConfusionMatrix(Δ, hΔ)
@printf("\ndirect estimate (λ = %.2f): %d nonzero entries in the lower triangle, TPR = %.2f, FPR = %.3f\n",
  λ, nnz(hΔ), CovSel.tpr(c), CovSel.fpr(c))
@printf("  Δ̂[2, 1] = %.3f, shrunk towards zero by the penalty\n", hΔ[2, 1])

# Refitting minimizes the loss over Δ supported on the estimated support,
# without the penalty. The loss is quadratic, so the refit solves the linear
# system
#
#   ((Sy ⊗ Sx + Sx ⊗ Sy) / 2)[J, J] vec(Δ)[J] = vec(Sy - Sx)[J]
#
# where J holds the entries of the support in both triangles.
support = [CartesianIndex(ProximalBase.ind2subLowerTriangular(p, hΔ.nzval2ind[k])) for k=1:nnz(hΔ)]
rΔ = differencePrecisionRefit(Sx, Sy, support)

J = LinearIndices((p, p))[unique([support; [CartesianIndex(k[2], k[1]) for k in support]])]
x = zeros(p, p)
x[J] = ((kron(Sy, Sx) + kron(Sx, Sy))[J, J] / 2) \ (Sy - Sx)[J]
@printf("refit on that support: Δ̂[2, 1] = %.3f; largest difference from the linear system: %.1e\n",
  rΔ[2, 1], maximum(abs, Matrix(rΔ) - x))

# Iterative hard thresholding minimizes the same loss subject to at most s
# nonzero entries in the lower triangle, with step size η = 1 / (‖Sx‖‖Sy‖),
# the reciprocal of the Lipschitz constant of the gradient.
η = 1 / (opnorm(Sx) * opnorm(Sy))
iΔ, fvals = differencePrecisionIHT(Sx, Sy, η, 1)
@printf("\niterative hard thresholding with s = 1: Δ̂[2, 1] = %.3f after %d iterations\n",
  iΔ[2, 1], length(fvals) - 1)

# The indirect approach estimates each precision matrix and takes the
# difference; errors in the two estimates do not cancel. The fused graphical
# lasso penalizes the difference, which removes most of them.
Zx, Zy = (covselpath(Matrix(S), [0.1])[1] for S in (Sx, Sy))
@printf("\nnonzero entries in the lower triangle of the difference of two graphical lasso estimates: %d\n",
  count(x -> abs(x) > 1e-4, tril(Zx - Zy)))
Fx, Fy = fusedGraphicalLasso(Matrix(Sx), n, Matrix(Sy), n, 0.05, 0.05)
@printf("nonzero entries in the lower triangle of Ω̂x - Ω̂y from the fused graphical lasso: %d\n",
  count(x -> abs(x) > 1e-4, tril(Fx - Fy)))
