module DirectDiffTest

using Test, Random, LinearAlgebra, SparseArrays
using Distributions
import CovSel, ProximalBase, CoordinateDescent


Random.seed!(123)

function genData(p)
  Sigmax = Matrix(1.0I, p, p)
  Sigmay = zeros(p,p)
  rho = 0.7
  for i=1:p
    for j=1:p
      Sigmay[i,j]=rho^abs(i-j)
    end
  end
  sqmy = sqrt(Sigmay)
  n = 1000
  X = randn(n,p)
  Y = randn(n,p) * sqmy;
  hSx = cov(X)
  hSy = cov(Y)
  hSx, hSy
end


@testset "direct_difference_estimation" begin

  @testset "small" begin
    # simple model that differes in only one edge
    p = 10
    hSx, hSy = genData(p)

    tmp = ones(p,p)
    for i=1:5

      # λ = 0.2
      λ = rand(Uniform(0.05, 0.3))
      g = ProximalBase.ProxL1(λ)

      solShoot = CovSel.Alt.differencePrecisionNaive(hSx, hSy, λ, tmp)
      solShoot1 = CovSel.differencePrecisionActiveShooting(Symmetric(hSx), Symmetric(hSy), g)

      @test convert(Matrix, solShoot1) ≈ convert(Matrix, solShoot) atol=1e-5
    end
  end

  @testset "loss" begin
    for rep=1:5
      p = 10
      hSx, hSy = genData(p)

      A = sprand(p, p, 0.15)
      A = (A + A') / 2.
      As = ProximalBase.SymmetricSparseIterate(A)

      @test CovSel.diffLoss(Symmetric(hSx), As, Symmetric(hSy), 2)  ≈ norm(hSx*A*hSy, 2) atol=1e-12
      @test CovSel.diffLoss(Symmetric(hSx), As, Symmetric(hSy), Inf) ≈ norm(hSx*A*hSy, Inf) atol=1e-12
      @test_throws ArgumentError CovSel.diffLoss(Symmetric(hSx), As, Symmetric(hSy), 3)
    end
  end

  @testset "refit" begin
      for rep=1:50
        p = 3
        hSx, hSy = genData(p)

        S = sprand(p, p, 0.2)
        S = S + S' / 2
        indS = findall(x->x!=0, S)
        ilin = LinearIndices(S)[indS]


        opt = CoordinateDescent.CDOptions(;maxIter=5000, optTol=1e-12, randomize=true)
        x = CovSel.differencePrecisionRefit(Symmetric(hSx), Symmetric(hSy), indS)

        x1 = spzeros(p, p)
        x1[ilin] = (0.5 * (kron(hSy, hSx)[ilin, ilin] + kron(hSx, hSy)[ilin, ilin])) \ (hSy - hSx)[ilin]
        @test convert(Matrix, x1) ≈ convert(Matrix, x) atol = 1e-4
      end
  end


end


end
