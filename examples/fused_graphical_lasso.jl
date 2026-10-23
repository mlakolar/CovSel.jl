# The fused graphical lasso for two groups whose graphs differ in a few edges,
# and what the fusion penalty λ2 does. Then fused neighborhood selection, the
# regression analogue for a single node.
#
# Run from the repository root (see examples/README.md for the setup):
#
#   julia --project=examples examples/fused_graphical_lasso.jl

using CovarianceSelection
using LinearAlgebra, Printf, Random, Statistics

Random.seed!(2)

# Two chain graphs on p nodes that differ in four edges. Every off-diagonal
# entry is ±0.3 and no node has more than three neighbors, so both matrices
# are diagonally dominant, hence positive definite.
p = 30
Ωx = Matrix(1.0I, p, p)
for i=1:p-1
  Ωx[i, i+1] = Ωx[i+1, i] = 0.3
end
Ωy = copy(Ωx)
Ωy[6, 5] = Ωy[5, 6] = 0.0       # edge removed
Ωy[21, 20] = Ωy[20, 21] = 0.0   # edge removed
Ωy[11, 10] = Ωy[10, 11] = -0.3  # sign flipped
Ωy[15, 1] = Ωy[1, 15] = 0.3     # edge added
Δ = Ωx - Ωy

nx, ny = 300, 300
X = CovarianceSelection.generateData(Ωx, nx)
Y = CovarianceSelection.generateData(Ωy, ny)
Sx, Sy = cov(X), cov(Y)

# fusedGraphicalLasso solves
#
#   minimize  Σₖ nₖ/n (tr(SₖΩₖ) - log det Ωₖ) + λ1 (‖Ωx‖₁ + ‖Ωy‖₁) + λ2 ‖Ωx - Ωy‖₁
#
# with n = nx + ny. λ1 makes both estimates sparse and λ2 makes them agree
# except in a few entries. With nx = ny the likelihoods are weighted 1/2, so
# λ1 corresponds to a graphical lasso penalty of 2λ1 in each group.
λ1 = sqrt(log(p) / (nx + ny))
println("p = $p, nx = ny = $nx, λ1 = ", round(λ1, digits=3))
println("the graphs differ in ", count(!iszero, tril(Δ, -1)), " edges\n")

println("   λ2   edges x   edges y   entries differing   TPR    FPR   ‖Ẑx - Ωx‖ + ‖Ẑy - Ωy‖")
for λ2 in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
  Zx, Zy = fusedGraphicalLasso(Sx, nx, Sy, ny, λ1, λ2)
  # the estimated difference is Zx - Zy; compare its support with that of Δ
  c = CovarianceSelection.getConfusionMatrix(Δ, Zx - Zy; diagonal=false)
  @printf("%5.2f   %7d   %7d   %17d   %.2f   %.3f   %.3f\n", λ2,
    count(x -> abs(x) > 1e-4, tril(Zx, -1)), count(x -> abs(x) > 1e-4, tril(Zy, -1)),
    c.tp + c.fp, CovarianceSelection.tpr(c), CovarianceSelection.fpr(c), norm(Zx - Ωx) + norm(Zy - Ωy))
end
println("""
λ2 = 0 fits the groups separately, and their estimates differ in many
entries. A larger λ2 removes the spurious differences first; when it is
too large the two estimates are equal and the real differences are lost.
""")

# fusedNeighborhoodSelection solves, for one node a,
#
#   minimize  ‖bx - Ax βx‖² / (2nx) + ‖by - Ay βy‖² / (2ny)
#               + λ1 (‖βx‖₁ + ‖βy‖₁) + λ2 ‖βx - βy‖₁
#
# where b is the column of node a and A holds the other columns. The true
# coefficients are β = -Ω[others, a] / Ω[a, a], and the noise variance is
# 1 / Ω[a, a] = 1, so λ1 = sqrt(2 log p / n) is the usual lasso choice.
# Node 6 has neighbors 5 and 7 in group x but only 7 in group y.
a = 6
others = setdiff(1:p, a)          # others[5] == 5 and others[6] == 7
rest = [1:4; 7:p-1]               # the other nodes
λ = sqrt(2 * log(p) / nx)
βx = -Ωx[others, a] / Ωx[a, a] .+ 0.0   # (+ 0.0 turns -0.0 into 0.0)
βy = -Ωy[others, a] / Ωy[a, a] .+ 0.0
nonzeros(β) = count(x -> abs(x) > 1e-4, β)

@printf("node %d, λ1 = %.3f\n\n", a, λ)
println("           node 5 in x, y    node 7 in x, y    other nonzeros in x, y   entries differing")
@printf("truth      %6.3f %6.3f     %6.3f %6.3f     %8d %8d          %8d\n",
  βx[5], βy[5], βx[6], βy[6], nonzeros(βx[rest]), nonzeros(βy[rest]), nonzeros(βx - βy))
for λ2 in (0.0, 0.05, 0.1, 0.5)
  zx, zy = fusedNeighborhoodSelection(X[:, others], X[:, a], Y[:, others], Y[:, a], λ, λ2)
  @printf("λ2 = %.2f  %6.3f %6.3f     %6.3f %6.3f     %8d %8d          %8d\n", λ2,
    zx[5], zy[5], zx[6], zy[6], nonzeros(zx[rest]), nonzeros(zy[rest]), nonzeros(zx - zy))
end
println("""
Fitting each group on its own (λ2 = 0) finds spurious neighbors and
differences. As λ2 grows the spurious differences go first, and at λ2 = 0.1
only the real one is left; a large λ2 forces the two neighborhoods to be
equal.""")
