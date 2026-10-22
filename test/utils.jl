module UtilsTest

using Test, Random, LinearAlgebra, SparseArrays
import CovSel, ProximalBase
using ProximalBase: SymmetricSparseIterate

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

    # the support stays symmetric when the upper triangle differs by roundoff
    Y = copy(X)
    Y[2, 3] = nextfloat(Y[2, 3])
    @test CovSel.HardThreshold!(similar(Y), Y, 3) == [4.0 0 0; 0 3.0 Y[2, 3]; 0 -2.5 0]
  end

  @testset "sparse iterates" begin
    for diagonal in (true, false), s in (0, 1, 2, 4)
      expected = CovSel.HardThreshold!(similar(X), X, s, diagonal)

      Δ = SymmetricSparseIterate(3)
      CovSel.HardThreshold!(Δ, X, s, diagonal)
      @test Matrix(Δ) == expected
      @test nnz(Δ) == count(!iszero, tril(expected))
    end
    @test_throws DimensionMismatch CovSel.HardThreshold!(SymmetricSparseIterate(2), X, 1)

    Δ = SymmetricSparseIterate(sparse(X))
    CovSel.HardThreshold!(Δ, 3)
    @test Matrix(Δ) == [4.0 0 0; 0 3.0 -2.5; 0 -2.5 0]
    @test nnz(Δ) == 3

    CovSel.HardThreshold!(Δ, 0)
    @test nnz(Δ) == 0
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
  @test CovSel.tpr(t) == 2 / 3
  @test CovSel.fpr(t) == 1 / 3
  @test CovSel.precision(t) == 2 / 3

  # any matrix type for the estimate
  for h in (SymmetricSparseIterate(sparse(hΔ)), sparse(hΔ))
    t1 = CovSel.getConfusionMatrix(Δ, h)
    @test (t1.tp, t1.fp, t1.fn, t1.tn) == (t.tp, t.fp, t.fn, t.tn)
  end

  # off-diagonal entries only: (2,1) fp, (3,1) fn, (3,2) tn
  for h in (hΔ, SymmetricSparseIterate(sparse(hΔ)))
    t2 = CovSel.getConfusionMatrix(Δ, h; diagonal=false)
    @test (t2.tp, t2.fp, t2.fn, t2.tn) == (0, 1, 1, 1)
  end
  # the threshold
  t3 = CovSel.getConfusionMatrix(Δ, 0.5 * hΔ, 0.6)
  @test (t3.tp, t3.fp, t3.fn, t3.tn) == (1, 0, 2, 3)

  # empty supports
  t0 = CovSel.ConfusionMatrix(0, 0, 0, 0)
  @test CovSel.tpr(t0) == 0.0
  @test CovSel.fpr(t0) == 0.0
  @test CovSel.precision(t0) == 1.0
end

end
