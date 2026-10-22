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

  Ωy = CovSel.generateRandomDifference(Ω, 1.0)
  @test diag(Ωy) == diag(Ω)
  @test Ωy - Diagonal(Ωy) == -(Ω - Diagonal(Ω))
  @test CovSel.generateRandomDifference(Ω, 0.0) == Ω

  Ω1, L = CovSel.genLatentModel(Ω, 2, 0.5)
  @test Ω1 === Ω
  @test rank(L) == 2
  @test opnorm(L) ≈ 0.5 * eigmax(Ω)
end

end
