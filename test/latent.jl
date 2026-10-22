module LatentTest

using Test, Random, LinearAlgebra
import CovSel, ProximalBase
using ProximalBase: SymmetricSparseIterate, nnz

Random.seed!(11)

@testset "latent" begin

  @testset "differenceLatentPrecisionIHT" begin
    p, r = 8, 1
    Σx = [0.5^abs(i-j) for i=1:p, j=1:p]
    S = zeros(p, p)
    S[1, 2] = S[2, 1] = 0.4
    u = randn(p); u ./= norm(u)
    Θ = S + 0.3 * u * u'
    Σy = Matrix(inv(Symmetric(inv(Σx) - Θ)))
    @test isposdef(Symmetric(Σy))

    η = 0.5 / (opnorm(Σx) * opnorm(Σy))
    opt = CovSel.IHTOptions(;epsTol=1e-10, maxIter=20000, checkEvery=10)
    calls = Ref(0)
    Δ, L, U, fvals, iter = CovSel.differenceLatentPrecisionIHT(Symmetric(Σx), Symmetric(Σy), η, η, 1, r;
        options=opt, callback=(Δ, L) -> (calls[] += 1))
    @test calls[] == iter
    @test 1 <= iter <= opt.maxIter
    @test L ≈ U * U'
    @test size(U) == (p, r)
    @test nnz(Δ) <= 1

    # the fit improves on the starting point and the residual of the
    # estimating equation is small
    Δ0, L0, U0, _, _ = CovSel.differenceLatentPrecisionIHT_init(Symmetric(Σx), Symmetric(Σy), 1, r)
    res0 = CovSel.diffLoss(Symmetric(Σx), Δ0, U0, Symmetric(Σy), 2)
    res = CovSel.diffLoss(Symmetric(Σx), Δ, U, Symmetric(Σy), 2)
    @test res < res0
    @test fvals[end] <= fvals[1]
    @test Matrix(Δ) + L ≈ Θ atol=0.05

    # the in-place version continues from the given iterates
    Δ1, L1, U1, _, iter1 = CovSel.differenceLatentPrecisionIHT!(copy(Δ), copy(L), copy(U), zeros(p, p), zeros(p, r),
        Symmetric(Σx), Symmetric(Σy), η, η, 1, r; options=CovSel.IHTOptions(;epsTol=1e-10, maxIter=5, checkEvery=10))
    @test iter1 == 5
    @test CovSel.diffLoss(Symmetric(Σx), Δ1, U1, Symmetric(Σy), 2) <= res + 1e-8
  end

  @testset "covselLatentIHT!" begin
    p, r = 6, 1
    Ω = Matrix(2.0I, p, p)
    Ω[1, 2] = Ω[2, 1] = 0.5
    u = ones(p) / sqrt(p)
    Σ = inv(Ω + 0.5 * u * u')

    calls = Ref(0)
    S, L, U, fvals, iter = CovSel.covselLatentIHT!(Σ, 0.05, 0.05, 1, r;
        epsTol=1e-10, maxIter=200, checkEvery=5, callback=(Σ, S, L) -> (calls[] += 1))
    @test calls[] == iter
    @test L ≈ U * U'
    @test issymmetric(S)
    @test isposdef(Symmetric(S + L))
    @test count(!iszero, tril(S, -1)) <= 1
    @test fvals[end] <= fvals[1]
    @test CovSel.covSel_objective(Σ, S, L) >= CovSel.covSel_objective(Σ, inv(Σ)) - 1e-8
  end
end

end
