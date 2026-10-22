module GenerateTest

using Test, Random, LinearAlgebra, Statistics
import CovSel

Random.seed!(7)

@testset "generate" begin
  p = 30
  Ω = CovSel.generateSparseSPD(p)
  @test issymmetric(Ω)
  @test isposdef(Ω)
  @test all(diag(Ω) .>= 1)

  X = CovSel.generateData(Ω, 20000)
  @test size(X) == (20000, p)
  @test cov(X) ≈ inv(Ω) rtol=0.1

  # the rows are R⁻¹z for Ω = RᵀR and standard normal z
  F = cholesky(Symmetric(Ω))
  X = CovSel.generateData(MersenneTwister(1), Ω, 10)
  @test X * F.U' ≈ randn(MersenneTwister(1), 10, p)
  # a precomputed factorization gives the same draws
  @test CovSel.generateData(MersenneTwister(1), F, 10) == X
  @test size(CovSel.generateData(F, 4)) == (4, p)
  @test eltype(CovSel.generateData(Float32.(Ω), 3)) == Float32
  @test size(CovSel.generateData(Ω, 0)) == (0, p)
  @test_throws PosDefException CovSel.generateData([1.0 2.0; 2.0 1.0], 3)

  Ω1, L = CovSel.genLatentModel(Ω, 2, 0.5)
  @test Ω1 === Ω
  @test rank(L) == 2
  @test opnorm(L) ≈ 0.5 * eigmax(Ω)
end

end
