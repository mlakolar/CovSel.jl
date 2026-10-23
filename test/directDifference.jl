module DirectDiffTest

using Test, Random, LinearAlgebra, SparseArrays, Statistics
import CovSel, ProximalBase, ProximalOPT
using CovSel: diffLoss
using ProximalBase: SymmetricSparseIterate, NormL1, coordinates, ncoordinates, tril_cartesian, tril_index
using ProximalOPT: CDSolver

include("reference.jl")

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
      g = NormL1(λ)

      solShoot = Reference.differencePrecisionNaive(hSx, hSy, λ, tmp)
      solShoot1 = CovSel.differencePrecisionActiveShooting(Symmetric(hSx), Symmetric(hSy), g)
      @test Matrix(solShoot1) ≈ solShoot atol=1e-5

      # warm start from the solution
      x = copy(solShoot1)
      CovSel.differencePrecisionActiveShooting!(x, Symmetric(hSx), Symmetric(hSy), g)
      @test Matrix(x) ≈ solShoot atol=1e-5
    end

    # KKT conditions: |gradient| ≤ λ, with equality and opposite sign on the
    # support; the gradient over symmetric matrices is (ΣxΔΣy + ΣyΔΣx)/2 - (Σy - Σx)
    λ = 0.1
    x = CovSel.differencePrecisionActiveShooting(Symmetric(hSx), Symmetric(hSy), NormL1(λ), CDSolver(; tol=1e-10))
    D = Matrix(x)
    G = (hSx * D * hSy + hSy * D * hSx) / 2 - (hSy - hSx)
    for c=1:p, r=c:p
      if D[r, c] == 0
        @test abs(G[r, c]) <= λ + 1e-6
      else
        @test G[r, c] ≈ -λ * sign(D[r, c]) atol=1e-6
      end
    end
    @test ncoordinates(x) == div(p * (p + 1), 2)

    # the loss is ProximalBase's KronQuadratic, whose value is the D-trace loss
    f = CovSel.directDifferenceLoss(Symmetric(hSx), Symmetric(hSy))
    @test f isa ProximalBase.KronQuadratic
    @test f(x) ≈ denseLoss(hSx, hSy, D)
    x1 = SymmetricSparseIterate(p)
    ProximalOPT.solve!(x1, f, NormL1(λ), CDSolver())
    @test Matrix(x1) ≈ D atol=1e-6

    # only the triangle that each Symmetric refers to is read
    Mx, My = copy(hSx), copy(hSy)
    Mx[2, 1] = My[1, 3] = 1e3
    x2 = CovSel.differencePrecisionActiveShooting(Symmetric(Mx, :U), Symmetric(My, :L), NormL1(λ))
    @test Matrix(x2) ≈ Matrix(x1) atol=1e-6   # random coordinate order, tol = 1e-7

    # Float32
    x3 = CovSel.differencePrecisionActiveShooting(Symmetric(Float32.(hSx)), Symmetric(Float32.(hSy)), NormL1(Float32(λ)))
    @test eltype(x3) == Float32
    @test Matrix(x3) ≈ Matrix(x1) atol=1e-4

    # weights on the p × p matrix: an entry with weight zero is not penalized,
    # and weights λ⋅W give the same solution as the level λ
    W = ones(p, p)
    x4 = CovSel.differencePrecisionActiveShooting(Symmetric(hSx), Symmetric(hSy), NormL1(1.0, λ * W), CDSolver(; tol=1e-10))
    @test Matrix(x4) ≈ D atol=1e-6
    W[3, 2] = W[2, 3] = 0
    x5 = CovSel.differencePrecisionActiveShooting(Symmetric(hSx), Symmetric(hSy), NormL1(λ, W), CDSolver(; tol=1e-10))
    G5 = (hSx * Matrix(x5) * hSy + hSy * Matrix(x5) * hSx) / 2 - (hSy - hSx)
    @test G5[3, 2] ≈ 0 atol=1e-6

    # the stopping rule is relative to the scales of the loss: multiplying the
    # covariances and λ by a constant divides the solution by it and takes the
    # same number of passes
    passes, sols = Int[], Matrix{Float64}[]
    for a in (1e-3, 1.0, 1e3)
      x = SymmetricSparseIterate(p)
      r = ProximalOPT.solve!(x, CovSel.directDifferenceLoss(Symmetric(a * hSx), Symmetric(a * hSy)),
        NormL1(a * λ), CDSolver(; order=:cyclic))
      @test r.converged
      push!(passes, r.passes)
      push!(sols, a * Matrix(x))
    end
    @test passes[1] == passes[2] == passes[3]
    @test sols[1] ≈ sols[2] atol=1e-8
    @test sols[3] ≈ sols[2] atol=1e-8
    @test_throws DimensionMismatch CovSel.directDifferenceLoss(Symmetric(hSx), Symmetric(hSy[1:3, 1:3]))
  end

  @testset "no penalty recovers the difference of precisions" begin
    p = 4
    Σx = [0.5^abs(i-j) for i=1:p, j=1:p]
    Σy = Matrix(1.0I, p, p) + 0.3 * ones(p, p)
    x = CovSel.differencePrecisionActiveShooting(Symmetric(Σx), Symmetric(Σy), NormL1(0.0),
          CDSolver(; tol=1e-12, maxpasses=10000))
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

      # dense and sparse matrices give the same loss
      for M in (Matrix(A), A)
        @test diffLoss(Symmetric(hSx), M, Symmetric(hSy), 2) ≈ norm(R, 2)
        @test diffLoss(Symmetric(hSx), M, Symmetric(hSy), Inf) ≈ norm(R, Inf)
        @test diffLoss(Symmetric(hSx), M, U, Symmetric(hSy), 2) ≈ norm(RL, 2)
        @test diffLoss(Symmetric(hSx), M, U, Symmetric(hSy), Inf) ≈ norm(RL, Inf)
      end
      @test_throws ArgumentError diffLoss(Symmetric(hSx), Matrix(A), Symmetric(hSy), 1)
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
        @test Matrix(x1) ≈ Matrix(x) atol = 1e-10

        # the support can be given by its upper triangle only, as linear indices
        iupper = [i for i in ilin if Tuple(CartesianIndices(S)[i])[1] <= Tuple(CartesianIndices(S)[i])[2]]
        @test Matrix(CovSel.differencePrecisionRefit(Symmetric(hSx), Symmetric(hSy), iupper)) ≈ Matrix(x) atol=1e-10
      end
      # an empty support gives zero
      hSx, hSy = genData(3)
      @test iszero(Matrix(CovSel.differencePrecisionRefit(Symmetric(hSx), Symmetric(hSy), Int[])))
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
