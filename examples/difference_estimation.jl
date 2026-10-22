# Estimate the difference of two precision matrices that differ in one edge,
# directly and through the graphical lasso.
#
# Run from an environment with CovSel, ProximalBase and CoordinateDescent:
#
#   julia examples/difference_estimation.jl

using CovSel, ProximalBase, CoordinateDescent
using LinearAlgebra, Random, SparseArrays, Statistics

Random.seed!(123)

p, n = 20, 1000
Ωx = Matrix(1.0I, p, p)
Ωy = copy(Ωx)
Ωy[1, 2] = Ωy[2, 1] = 0.4
Δ = Ωx - Ωy

X = CovSel.generateData(Ωx, n)
Y = CovSel.generateData(Ωy, n)
Sx, Sy = Symmetric(cov(X)), Symmetric(cov(Y))

# direct estimation with an ℓ1 penalty
hΔ = differencePrecisionActiveShooting(Sx, Sy, ProxL1(0.1))
println("direct estimate, nonzeros in the lower triangle: ", nnz(hΔ))
t = CovSel.getConfusionMatrix(Δ, hΔ)
println("  TPR = ", CovSel.tpr(t), ", FPR = ", round(CovSel.fpr(t), digits=3))

# refit on the estimated support
support = [CartesianIndex(ProximalBase.ind2subLowerTriangular(p, hΔ.nzval2ind[k])) for k=1:nnz(hΔ)]
rΔ = differencePrecisionRefit(Sx, Sy, support)
println("refitted Δ[2, 1] = ", round(rΔ[2, 1], digits=3), " (truth ", Δ[2, 1], ")")

# iterative hard thresholding with the true sparsity
η = 1 / (opnorm(Sx) * opnorm(Sy))
iΔ, fvals = differencePrecisionIHT(Sx, Sy, η, nnz(SymmetricSparseIterate(sparse(Δ))))
println("IHT estimate, Δ[2, 1] = ", round(iΔ[2, 1], digits=3), ", objective = ", round(fvals[end], digits=4))

# difference of two graphical lasso estimates, and the fused graphical lasso
Zx, Zy = (covselpath(Matrix(S), [0.1])[1] for S in (Sx, Sy))
println("graphical lasso, nonzero entries of Ωx - Ωy: ", count(x -> abs(x) > 1e-4, Zx - Zy))
Fx, Fy = fusedGraphicalLasso(Matrix(Sx), n, Matrix(Sy), n, 0.05, 0.05)
println("fused graphical lasso, nonzero entries of Ωx - Ωy: ", count(x -> abs(x) > 1e-4, Fx - Fy))
