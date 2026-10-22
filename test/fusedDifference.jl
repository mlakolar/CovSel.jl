module FusedDifferenceTest

using Test, Random, LinearAlgebra, Statistics
import CovSel, ProximalBase

include("optimality.jl")

Random.seed!(123)

function twoGroups(p, n)
  ρ = 0.3
  iρ = -ρ / (1. - ρ^2)
  Sigma_y = Matrix(1.0I, p, p)
  Sigma_y[1,2] = iρ
  Sigma_y[2,1] = iρ
  Sigma_y[1,1] = 1. / (1. - ρ^2)
  Sigma_y[2,2] = 1. / (1. - ρ^2)
  randn(n, p), randn(n, p) * sqrt(Sigma_y)
end

# with nx = ny the likelihood weights are 1/2
fglObjective(Ωx, Ωy, Sx, Sy, λ1, λ2) =
  (tr(Ωx*Sx) - logdet(Ωx) + tr(Ωy*Sy) - logdet(Ωy)) / 2. + λ1 * (sum(abs, Ωx) + sum(abs, Ωy)) + λ2 * sum(abs, Ωx - Ωy)

fglResidual(Zx, Zy, Sx, Sy, nx, ny, λ1, λ2; kw...) =
  fusedFixedPointResidual(Zx, Zy, nx / (nx + ny) * (Sx - inv(Zx)), ny / (nx + ny) * (Sy - inv(Zy)), λ1, λ2; kw...)

const tight = CovSel.ADMMOptions(;abstol=1e-12, reltol=1e-12, maxiter=10000)

@testset "fused_graphical_lasso" begin
    # simple model that differes in only one edge
    p = 10
    n = 1000
    X, Y = twoGroups(p, n)
    Sx = cov(X)
    Sy = cov(Y)

    for i=1:10

        λ1 = 0.01 + 0.09 * rand()
        λ2 = 0.01 + 0.09 * rand()

        θx = Matrix(1.0I, p, p)
        θy = Matrix(1.0I, p, p)
        Zx = Matrix(1.0I, p, p)
        Zy = Matrix(1.0I, p, p)
        Ux = zeros(p,p)
        Uy = zeros(p,p)

        CovSel.fusedGraphicalLasso!(θx, θy, Zx, Zy, Ux, Uy, Sx, n, Sy, n, λ1, λ2;
            penalize_diag=true, options=tight)

        @test issymmetric(Zx) && issymmetric(Zy)
        @test isposdef(Zx) && isposdef(Zy)
        @test fglResidual(Zx, Zy, Sx, Sy, n, n, λ1, λ2) < 1e-6
        @test isLocalMin((A, B) -> fglObjective(A, B, Sx, Sy, λ1, λ2), Zx, Zy)

        # the allocating version gives the same answer
        Zx1, Zy1 = CovSel.fusedGraphicalLasso(Sx, n, Sy, n, λ1, λ2; options=tight)
        @test Zx1 ≈ Zx atol=1e-6
        @test Zy1 ≈ Zy atol=1e-6
    end

    # unequal sample sizes and an unpenalized diagonal
    Zx, Zy = CovSel.fusedGraphicalLasso(Sx, 300, Sy, 700, 0.05, 0.05; penalize_diag=false, options=tight)
    @test fglResidual(Zx, Zy, Sx, Sy, 300, 700, 0.05, 0.05; penalize_diag=false) < 1e-6

    # a large fusion penalty makes the two estimates equal, and their common
    # value is the graphical lasso estimate for the pooled covariance
    # (λ1 (|x| + |y|) = 2λ1|x| on x = y, and the likelihoods are weighted 1/2)
    Zx, Zy = CovSel.fusedGraphicalLasso(Sx, n, Sy, n, 0.05, 10.0; options=tight)
    @test Zx ≈ Zy atol=1e-6
    Z = zeros(p, p)
    CovSel.covsel!(zeros(p, p), Z, zeros(p, p), (Sx + Sy) / 2, 0.1; options=CovSel.ADMMOptions(;abstol=1e-12, reltol=1e-12, maxiter=10000))
    @test Zx ≈ Z atol=1e-5

    # no fusion penalty: two separate graphical lasso problems with λ = 2λ1
    Zx, Zy = CovSel.fusedGraphicalLasso(Sx, n, Sy, n, 0.05, 0.0; options=tight)
    for (Zk, Sk) in ((Zx, Sx), (Zy, Sy))
      Z = zeros(p, p)
      CovSel.covsel!(zeros(p, p), Z, zeros(p, p), Sk, 0.1; options=CovSel.ADMMOptions(;abstol=1e-12, reltol=1e-12, maxiter=10000))
      @test Zk ≈ Z atol=1e-5
    end

    # unpenalized diagonal and identity covariances: the solution is I
    Zx, Zy = CovSel.fusedGraphicalLasso(Matrix(1.0I, p, p), n, Matrix(1.0I, p, p), n, 0.5, 0.5;
        penalize_diag=false, options=CovSel.ADMMOptions(;abstol=1e-12,reltol=1e-12))
    @test Zx ≈ I
    @test Zy ≈ I

    # Float32
    Zx32, Zy32 = CovSel.fusedGraphicalLasso(Float32.(Sx), n, Float32.(Sy), n, 0.05, 0.05;
        options=CovSel.ADMMOptions(;abstol=1e-6,reltol=1e-6))
    Zx64, Zy64 = CovSel.fusedGraphicalLasso(Sx, n, Sy, n, 0.05, 0.05;
        options=CovSel.ADMMOptions(;abstol=1e-6,reltol=1e-6))
    @test eltype(Zx32) == Float32
    @test Zx32 ≈ Zx64 atol=1e-3
    @test Zy32 ≈ Zy64 atol=1e-3
end


@testset "fused_neighborhood_selection" begin
  p = 5
  n = 1000
  X, Y = twoGroups(p, n)
  A1, b1, A2, b2 = X[:,2:p], X[:,1], Y[:,2:p], Y[:,1]

  fnsObjective(x1, x2, λ1, λ2) =
    (sum(abs2, b1 - A1*x1) + sum(abs2, b2 - A2*x2)) / (2. * n) + λ1 * (sum(abs, x1) + sum(abs, x2)) + λ2 * sum(abs, x1 - x2)

  for i=1:10
    λ1 = 0.1 * rand()
    λ2 = 0.1 * rand()
    z1, z2 = CovSel.fusedNeighborhoodSelection(A1, b1, A2, b2, λ1, λ2; options=tight)

    g1 = A1' * (A1 * z1 - b1) / n
    g2 = A2' * (A2 * z2 - b2) / n
    @test fusedFixedPointResidual(z1, z2, g1, g2, λ1, λ2; t=0.5) < 1e-8
    @test isLocalMin((x, y) -> fnsObjective(x, y, λ1, λ2), z1, z2)
  end

  # without penalties the solutions are the least squares fits
  z1, z2 = CovSel.fusedNeighborhoodSelection(A1, b1, A2, b2, 0.0, 0.0; options=tight)
  @test z1 ≈ A1 \ b1 atol=1e-6
  @test z2 ≈ A2 \ b2 atol=1e-6

  # a large fusion penalty gives the lasso on the stacked data with penalty 2λ1
  z1, z2 = CovSel.fusedNeighborhoodSelection(A1, b1, A2, b2, 0.02, 10.0; options=tight)
  @test z1 ≈ z2 atol=1e-8
  A, b = [A1; A2], [b1; b2]
  @test fusedFixedPointResidual(z1, z1, A' * (A * z1 - b) / (2n), A' * (A * z1 - b) / (2n), 0.02, 0.0; t=0.5) < 1e-8
end


end
