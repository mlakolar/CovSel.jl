module ScaledLassoTest

using Test, Random, LinearAlgebra, SparseArrays, Statistics
using CovSel
using CovSel: symmetrize_minabs

Random.seed!(17)

ar1(p, ρ) = [ρ^abs(i - j) for i in 1:p, j in 1:p]
samplecov(Σ, n) = cov(randn(n, size(Σ, 1)) * cholesky(Symmetric(Σ)).U)

# Optimality of column j of the unsymmetrized estimate R for the scaled lasso
#     minimize (β - eⱼ)'S(β - eⱼ)/(2σ) + σ/2 + λ₀ ∑ₖ √Sₖₖ |βₖ|,   βⱼ = 0:
# with σ = Rⱼⱼ^(-1/2) and β = -R[:, j]/Rⱼⱼ, the correlations c = S(eⱼ - β) must be
# cₖ = λ₀σ√Sₖₖ sign(βₖ) where βₖ ≠ 0 and |cₖ| ≤ λ₀σ√Sₖₖ elsewhere (k ≠ j), and
# σ² = (β - eⱼ)'S(β - eⱼ). The problem is jointly convex in (β, σ), so these
# conditions certify the solution. Returns the largest violation, relative to
# λ₀σ√Sₖₖ and σ².
function certificate(S, R, j, λ₀)
  σ = 1 / sqrt(R[j, j])
  β = -Vector(R[:, j]) / R[j, j]
  β[j] = 0
  e = zeros(size(S, 1))
  e[j] = 1
  c = S * (e - β)
  viol = abs(dot(β - e, S * (β - e)) - σ^2) / σ^2
  for k in axes(S, 1)
    k == j && continue
    bound = λ₀ * σ * sqrt(S[k, k])
    viol = max(viol, iszero(β[k]) ? (abs(c[k]) - bound) / bound : abs(c[k] - bound * sign(β[k])) / bound)
  end
  viol
end

penalty(λ₀, p, n) = CovSel._penalty_level(λ₀, p, n)

@testset "scaledlasso" begin

  @testset "normal quantile" begin
    # values of Python's statistics.NormalDist().inv_cdf (also AS 241)
    for (p, q) in ((0.975, 1.9599639845400534), (0.5, 0.0), (0.2, -0.8416212335729142),
                   (0.01, -2.3263478740408408), (1e-10, -6.361340902404057),
                   (1e-100, -21.273453560965322), (1 - 1e-10, 6.361340889697423))
      @test CovSel._qnorm(p) ≈ q rtol=1e-14 atol=1e-300
    end
    @test_throws DomainError CovSel._qnorm(0.0)
    @test_throws DomainError CovSel._qnorm(1.0)
  end

  @testset "penalty levels" begin
    for p in (1, 2, 30, 1000), n in (50, 400)
      λ = penalty(:quantile, p, n)
      L = λ * sqrt(n / 2)              # λ₀ = √2 Lₙ(k/p) = √(2/n) L₁(k/p)
      @test CovSel._qnorm((L^4 + 2L^2) / p) ≈ -L rtol=1e-8
    end
    @test penalty(:universal, 100, 50) ≈ sqrt(2log(100) / 50)
    @test penalty(0.3, 100, 50) == 0.3
    @test_throws ArgumentError penalty(-0.1, 10, 10)
    @test_throws ArgumentError penalty(Inf, 10, 10)
    @test_throws ArgumentError penalty(:theory, 10, 10)
  end

  # (with p > n a small λ₀ makes the scaled lasso fit exactly; see "input")
  @testset "optimality: $name" for (name, S, n, λs) in (
      ("AR(1), n > p", samplecov(ar1(30, 0.6), 120), 120, (:quantile, :universal, 0.05, 1.0)),
      ("AR(1), n < p", samplecov(ar1(60, 0.6), 40), 40, (:quantile, :universal, 0.3, 1.0)),
      ("dense", samplecov(0.5 * ones(20, 20) + 0.5I, 60), 60, (:quantile, :universal, 0.05, 1.0)),
      ("unequal variances", samplecov(Diagonal(1:25) * ar1(25, 0.5) * Diagonal(1:25), 100), 100,
       (:quantile, :universal, 0.05, 1.0)))
    p = size(S, 1)
    for λ₀ in λs
      R = scaledlasso(S, n; λ₀, symmetrize=false)
      λ = penalty(λ₀, p, n)
      @test all(certificate(S, R, j, λ) < 1e-9 for j in 1:p)
    end
  end

  @testset "closed forms" begin
    # uncorrelated variables: β = 0 and σ̂² = Sⱼⱼ
    D = Diagonal([1.0, 2.0, 0.5, 4.0])
    @test scaledlasso(Matrix(D), 10) ≈ inv(D)
    @test scaledlasso(fill(2.0, 1, 1), 5) ≈ fill(0.5, 1, 1)
    # a penalty above all correlations: the inverse marginal variances
    S = samplecov(ar1(10, 0.5), 50)
    @test scaledlasso(S, 50; λ₀=10.0) ≈ Diagonal(1 ./ diag(S))
    # 2 × 2, to a tight tolerance
    S = [2.0 0.9; 0.9 1.0]
    λ₀ = 0.2
    R = scaledlasso(S, 100; λ₀, symmetrize=false)
    @test all(certificate(S, R, j, λ₀) < 1e-12 for j in 1:2)
  end

  @testset "ties" begin
    # population matrices: the two neighbours of a variable of an AR(1) covariance
    # have the same correlation with it, and join at the same λ
    for (Σ, n) in ((ar1(12, 0.5), 100), ([1.0 0.5 0.5; 0.5 1.0 0.0; 0.5 0.0 1.0], 50),
                   (Matrix(SymTridiagonal(fill(2.0, 10), fill(0.7, 9))), 200))
      for λ₀ in (0.05, 0.1, :quantile)
        R = scaledlasso(Σ, n; λ₀, symmetrize=false)
        @test all(certificate(Σ, R, j, penalty(λ₀, size(Σ, 1), n)) < 1e-10 for j in axes(Σ, 1))
      end
    end
  end

  @testset "least squares refit" begin
    S = samplecov(ar1(40, 0.6), 100)
    R = scaledlasso(S, 100; symmetrize=false)
    F = scaledlasso(S, 100; refit=true, symmetrize=false)
    for j in 1:40
      A = filter(!=(j), findall(!iszero, R[:, j]))
      @test sort(filter(!=(j), findall(!iszero, F[:, j]))) == A
      β = S[A, A] \ S[A, j]
      σ2 = S[j, j] - dot(S[j, A], β)
      @test F[j, j] ≈ 1 / σ2 rtol=1e-10
      @test Vector(F[A, j]) ≈ -β / σ2 rtol=1e-8
    end
  end

  @testset "equivariance" begin
    # rescaling the variables by D rescales the estimate by D⁻¹, exactly
    S = samplecov(Diagonal(1 .+ 4rand(30)) * ar1(30, 0.5) * Diagonal(1 .+ 4rand(30)), 80)
    D = Diagonal(0.5 .+ 3rand(30))
    for refit in (false, true)
      Ω = scaledlasso(S, 80; refit)
      Ωd = scaledlasso(D * S * D, 80; refit)
      @test Ωd ≈ inv(D) * Ω * inv(D) rtol=1e-10
      @test findnz(Ωd)[1:2] == findnz(Ω)[1:2]
    end
  end

  @testset "symmetrization, threads, element type" begin
    S = samplecov(ar1(50, 0.6), 100)
    R = scaledlasso(S, 100; symmetrize=false)
    Ω = scaledlasso(S, 100)
    @test issymmetric(Ω)
    @test Ω == symmetrize_minabs(R)
    @test scaledlasso(S, 100; ntasks=1) == scaledlasso(S, 100; ntasks=4) == Ω
    @test scaledlasso(Symmetric(S), 100) == Ω
    Ω32 = scaledlasso(Float32.(S), 100)
    @test eltype(Ω32) == Float32
    @test Ω32 ≈ Ω rtol=1e-3
  end

  @testset "frequent refactorization and many breakpoints" begin
    # a small penalty on a larger problem passes through many events
    S = samplecov(ar1(80, 0.8), 400)
    ws = CovSel._SLWorkspace(S)
    λ = 0.01
    for refactor in (1, 3, 100)
      cols = [CovSel._ColumnPath{Float64}(1) for _ in 1:80]
      for j in 1:80
        CovSel._sl_column!(ws, j, λ, false, cols[j]; refactor)
      end
      R = CovSel._assemble(cols, 1)
      @test all(certificate(S, R, j, λ) < 1e-8 for j in 1:80)
    end
  end

  @testset "estimation" begin
    Ω0 = Matrix(SymTridiagonal(fill(1.5, 100), fill(0.5, 99)))
    S = cov(randn(2000, 100) / cholesky(Symmetric(Ω0)).U')
    @test opnorm(Matrix(scaledlasso(S, 2000)) - Ω0) < 0.4
    @test opnorm(Matrix(scaledlasso(S, 2000; refit=true)) - Ω0) < 0.4
  end

  @testset "input" begin
    S = samplecov(ar1(10, 0.5), 50)
    @test_throws ArgumentError scaledlasso(S, 0)
    @test_throws ArgumentError scaledlasso(S, 50; λ₀=-1)
    @test_throws ArgumentError scaledlasso(S, 50; λ₀=:theory)
    @test_throws DimensionMismatch scaledlasso(randn(3, 4), 10)
    # p > n with a tiny penalty: the path reaches a singular block of S
    X = randn(10, 40)
    @test_throws DomainError scaledlasso(X' * X / 10, 10; λ₀=1e-4)
  end
end

end
