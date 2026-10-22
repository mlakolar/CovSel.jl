module DirectDiffTest

using Test, Random, LinearAlgebra, SparseArrays, Statistics
import CovSel, ProximalBase, CoordinateDescent
using ProximalBase: SymmetricSparseIterate

Random.seed!(123)

function genData(p; n = 1000)
  Sigmay = [0.7^abs(i-j) for i=1:p, j=1:p]
  X = randn(n,p)
  Y = randn(n,p) * sqrt(Sigmay)
  cov(X), cov(Y)
end

# direct difference loss evaluated with dense matrices
denseLoss(Sx, Sy, D) = tr(Sx*D*Sy*D) / 2 - tr(D*(Sy - Sx))

@testset "direct_difference_estimation" begin

  @testset "active shooting" begin
    p = 10
    hSx, hSy = genData(p)

    tmp = ones(p,p)
    for i=1:5
      λ = 0.05 + 0.25 * rand()
      g = ProximalBase.ProxL1(λ)

      solShoot = CovSel.Alt.differencePrecisionNaive(hSx, hSy, λ, tmp)
      solShoot1 = CovSel.differencePrecisionActiveShooting(Symmetric(hSx), Symmetric(hSy), g)
      @test Matrix(solShoot1) ≈ solShoot atol=1e-5

      solShoot2 = CovSel.Alt.differencePrecisionActiveShooting(hSx, hSy, λ, tmp)
      @test Matrix(solShoot2) ≈ solShoot atol=1e-5

      # warm start from the solution
      x = copy(solShoot1)
      CovSel.differencePrecisionActiveShooting!(x, Symmetric(hSx), Symmetric(hSy), g)
      @test Matrix(x) ≈ solShoot atol=1e-5
    end

    # KKT conditions: |gradient| ≤ λ, with equality and opposite sign on the support
    λ = 0.1
    f = CovSel.CDDirectDifferenceLoss(Symmetric(hSx), Symmetric(hSy))
    x = CovSel.differencePrecisionActiveShooting(Symmetric(hSx), Symmetric(hSy), ProximalBase.ProxL1(λ),
          CoordinateDescent.CDOptions(;optTol=1e-10))
    CoordinateDescent.initialize!(f, x)
    for j=1:CoordinateDescent.numCoordinates(f)
      gj = CoordinateDescent.gradient(f, x, j)
      if x[j] == 0
        @test abs(gj) <= λ + 1e-6
      else
        @test gj ≈ -λ * sign(x[j]) atol=1e-6
      end
    end
    @test CoordinateDescent.numCoordinates(f) == div(p * (p + 1), 2)

    # the loss keeps A = Σx⋅Δ⋅Σy up to date during the descent
    f = CovSel.CDDirectDifferenceLoss(Symmetric(hSx), Symmetric(hSy))
    x1 = SymmetricSparseIterate(p)
    CoordinateDescent.coordinateDescent!(x1, f, ProximalBase.ProxL1(λ))
    @test f.A ≈ hSx * Matrix(x1) * hSy

    # only the triangle that each Symmetric refers to is read
    Mx, My = copy(hSx), copy(hSy)
    Mx[2, 1] = My[1, 3] = 1e3
    x2 = CovSel.differencePrecisionActiveShooting(Symmetric(Mx, :U), Symmetric(My, :L), ProximalBase.ProxL1(λ))
    @test Matrix(x2) ≈ Matrix(x1) atol=1e-6   # random coordinate order, optTol = 1e-7

    # Float32
    x3 = CovSel.differencePrecisionActiveShooting(Symmetric(Float32.(hSx)), Symmetric(Float32.(hSy)), ProximalBase.ProxL1(Float32(λ)))
    @test eltype(x3) == Float32
    @test Matrix(x3) ≈ Matrix(x1) atol=1e-4
    @test_throws DimensionMismatch CovSel.CDDirectDifferenceLoss(Symmetric(hSx), Symmetric(hSy[1:3, 1:3]))
  end

  @testset "no penalty recovers the difference of precisions" begin
    p = 4
    Σx = [0.5^abs(i-j) for i=1:p, j=1:p]
    Σy = Matrix(1.0I, p, p) + 0.3 * ones(p, p)
    x = CovSel.differencePrecisionActiveShooting(Symmetric(Σx), Symmetric(Σy), ProximalBase.ProxL1(0.0),
          CoordinateDescent.CDOptions(;optTol=1e-12, maxIter=10000))
    @test Matrix(x) ≈ inv(Σx) - inv(Σy) atol=1e-6
  end

  @testset "loss" begin
    for rep=1:5
      p = 10
      hSx, hSy = genData(p)

      A = sprand(p, p, 0.15)
      A = (A + A') / 2.
      As = SymmetricSparseIterate(A)
      R = hSx*A*hSy - hSy + hSx

      @test CovSel.diffLoss(Symmetric(hSx), As, Symmetric(hSy), 2)  ≈ norm(R, 2) atol=1e-12
      @test CovSel.diffLoss(Symmetric(hSx), As, Symmetric(hSy), Inf) ≈ norm(R, Inf) atol=1e-12
      @test_throws ArgumentError CovSel.diffLoss(Symmetric(hSx), As, Symmetric(hSy), 3)

      U = randn(p, 2)
      RL = hSx*(A + U*U')*hSy - hSy + hSx
      @test CovSel.diffLoss(Symmetric(hSx), As, U, Symmetric(hSy), 2)  ≈ norm(RL, 2) atol=1e-10
      @test CovSel.diffLoss(Symmetric(hSx), As, U, Symmetric(hSy), Inf) ≈ norm(RL, Inf) atol=1e-10
      @test_throws ArgumentError CovSel.diffLoss(Symmetric(hSx), As, U, Symmetric(hSy), 1)

      # the residual vanishes at the population solution
      Δ = SymmetricSparseIterate(sparse(inv(hSx) - inv(hSy)))
      @test CovSel.diffLoss(Symmetric(hSx), Δ, Symmetric(hSy), Inf) ≈ 0 atol=1e-10

      # objective
      @test CovSel.differencePrecision_objective(Symmetric(hSx), Symmetric(hSy), As) ≈ denseLoss(hSx, hSy, Matrix(A))
      Ax = hSx * (A + U*U') * hSy
      @test CovSel.differencePrecision_objective(Ax, Symmetric(hSx), Symmetric(hSy), As, U) ≈ denseLoss(hSx, hSy, Matrix(A) + U*U')
    end
  end

  @testset "refit" begin
      for rep=1:50
        p = 3
        hSx, hSy = genData(p)

        S = sprand(p, p, 0.2)
        S = S + S'
        indS = findall(x->x!=0, S)
        ilin = LinearIndices(S)[indS]

        x = CovSel.differencePrecisionRefit(Symmetric(hSx), Symmetric(hSy), indS)

        x1 = spzeros(p, p)
        x1[ilin] = (0.5 * (kron(hSy, hSx)[ilin, ilin] + kron(hSx, hSy)[ilin, ilin])) \ (hSy - hSx)[ilin]
        @test Matrix(x1) ≈ Matrix(x) atol = 1e-4

        # the support can be given by its upper triangle only, as linear indices
        iupper = [i for i in ilin if Tuple(CartesianIndices(S)[i])[1] <= Tuple(CartesianIndices(S)[i])[2]]
        @test Matrix(CovSel.differencePrecisionRefit(Symmetric(hSx), Symmetric(hSy), iupper)) ≈ Matrix(x) atol=1e-6
      end
  end

  @testset "IHT" begin
    p = 6
    Σx = [0.5^abs(i-j) for i=1:p, j=1:p]
    Ωy = inv(Σx)
    Ωy[1, 2] = Ωy[2, 1] = Ωy[1, 2] + 0.5
    Σy = inv(Ωy)
    D = inv(Σx) - inv(Σy)

    η = 1 / (opnorm(Σx) * opnorm(Σy))
    Δ, fvals = CovSel.differencePrecisionIHT(Symmetric(Σx), Symmetric(Σy), η, 1; epsTol=1e-14, maxIter=10000)
    @test Matrix(Δ) ≈ D atol=1e-6
    @test nnz(Δ) == 1
    @test fvals[end] ≈ denseLoss(Σx, Σy, D)
    # projected gradient descent with step 1/L does not increase the objective here
    @test all(diff(fvals) .<= 1e-12)

    # without the sparsity constraint the iterates converge to inv(Σx) - inv(Σy)
    Σy = [0.3^abs(i-j) for i=1:p, j=1:p]
    η = 1 / (opnorm(Σx) * opnorm(Σy))
    Δ, _ = CovSel.differencePrecisionIHT(Symmetric(Σx), Symmetric(Σy), η, div(p * (p + 1), 2); epsTol=1e-16, maxIter=20000)
    @test Matrix(Δ) ≈ inv(Σx) - inv(Σy) atol=1e-5
    @test_throws DimensionMismatch CovSel.differencePrecisionIHT(Symmetric(Σx), Symmetric(Σy[1:2, 1:2]), η, 1)

    # only the triangle that each Symmetric refers to is read
    Mx, My = copy(Σx), copy(Σy)
    Mx[2, 1] = My[1, 3] = 1e3
    Δ1, fvals1 = CovSel.differencePrecisionIHT(Symmetric(Σx), Symmetric(Σy), η, 4; maxIter=50)
    Δ2, fvals2 = CovSel.differencePrecisionIHT(Symmetric(Mx, :U), Symmetric(My, :L), η, 4; maxIter=50)
    @test Matrix(Δ2) ≈ Matrix(Δ1)
    @test fvals2 ≈ fvals1
  end

end


end
