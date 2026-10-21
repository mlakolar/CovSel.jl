module UtilsTest

using Test, Random, LinearAlgebra, SparseArrays
import CovSel, ProximalBase
using ProximalBase: SymmetricSparseIterate, SparseIterate, sub2indLowerTriangular

Random.seed!(1)

@testset "HardThreshold!" begin
  X = [4.0  -1.0  0.5;
      -1.0   3.0 -2.5;
       0.5  -2.5  0.1]

  @testset "dense" begin
    out = similar(X)
    CovSel.HardThreshold!(out, X, 3)
    @test out == [4.0 0 0; 0 3.0 -2.5; 0 -2.5 0]

    # diagonal is kept and excluded from the ranking
    CovSel.HardThreshold!(out, X, 1, false)
    @test out == [4.0 0 0; 0 3.0 -2.5; 0 -2.5 0.1]

    @test CovSel.HardThreshold!(copy(X), 0) == zeros(3, 3)
    @test CovSel.HardThreshold!(copy(X), 0, false) == Diagonal(diag(X))
    # asking for more entries than there are keeps everything
    @test CovSel.HardThreshold!(copy(X), 100) == X
    @test_throws DimensionMismatch CovSel.HardThreshold!(zeros(2, 2), X, 1)
  end

  @testset "sparse iterates" begin
    for diagonal in (true, false), s in (0, 1, 2, 4)
      expected = CovSel.HardThreshold!(similar(X), X, s, diagonal)

      Δ = SymmetricSparseIterate(3)
      CovSel.HardThreshold!(Δ, X, s, diagonal)
      @test Matrix(Δ) == expected
      @test nnz(Δ) == count(!iszero, tril(expected))

      x = SparseIterate(6)
      CovSel.HardThreshold!(x, X, s, diagonal)
      @test [x[sub2indLowerTriangular(3, r, c)] for c=1:3 for r=c:3] == [expected[r, c] for c=1:3 for r=c:3]
    end
    @test_throws DimensionMismatch CovSel.HardThreshold!(SymmetricSparseIterate(2), X, 1)

    Δ = SymmetricSparseIterate(sparse(X))
    CovSel.HardThreshold!(Δ, 3)
    @test Matrix(Δ) == [4.0 0 0; 0 3.0 -2.5; 0 -2.5 0]
    @test nnz(Δ) == 3

    CovSel.HardThreshold!(Δ, 0)
    @test nnz(Δ) == 0

    x = SparseIterate([0.0, -3.0, 1.0, 2.0])
    CovSel.HardThreshold!(x, 2)
    @test Vector(x) == [0.0, -3.0, 0.0, 2.0]
  end
end

@testset "covSel_objective" begin
  A = randn(20, 4)
  Σ = A'A / 20
  Ω = inv(Σ + I)
  L = 0.1 * ones(4, 4)
  @test CovSel.covSel_objective(Σ, Ω) ≈ tr(Σ * Ω) - log(det(Ω))
  @test CovSel.covSel_objective(Σ, Ω, L) ≈ CovSel.covSel_objective(Σ, Ω + L)
end

@testset "confusion matrix" begin
  Δ = [1.0 0 2; 0 0 0; 2 0 3]
  hΔ = [1.0 1 0; 1 0 0; 0 0 3]
  t = CovSel.getConfusionMatrix(Δ, hΔ)
  # lower triangle: (1,1) tp, (2,1) fp, (3,1) fn, (2,2) tn, (3,2) tn, (3,3) tp
  @test (t.tp, t.fp, t.fn, t.tn) == (2, 1, 1, 2)
  @test CovSel.tpr(t) == CovSel.recall(t) == 2 / 3
  @test CovSel.fpr(t) == 1 / 3
  @test CovSel.precision(t) == 2 / 3

  x = SparseIterate(6)
  for c=1:3, r=c:3
    x[sub2indLowerTriangular(3, r, c)] = hΔ[r, c]
  end
  t1 = CovSel.getConfusionMatrix(Δ, x)
  @test (t1.tp, t1.fp, t1.fn, t1.tn) == (t.tp, t.fp, t.fn, t.tn)

  # empty supports
  t0 = CovSel.ConfusionMatrix(0, 0, 0, 0)
  @test CovSel.tpr(t0) == 0.0
  @test CovSel.fpr(t0) == 0.0
  @test CovSel.precision(t0) == 1.0
end

end
