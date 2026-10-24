module GroupGraphicalLassoTest

using Test, Random, LinearAlgebra, Statistics
import CovarianceSelection
using CovarianceSelection: ADMMOptions, ADMMInfo, groupGraphicalLasso, groupGraphicalLasso!, kernelCovariances, tvCovSel
using ProximalBase: FiberSum, NormL2, prox

Random.seed!(7)

const tight = ADMMOptions(; abstol = 1e-12, reltol = 1e-12, maxiter = 20_000)

# the residual of the fixed point Ω = prox_{tg}(Ω - t∇f(Ω)), which holds exactly
# at the minimizer: f is the weighted sum of the log-likelihoods, g the group
# penalty on every fiber Ω[i, j, :]
function gglResidual(Z, S, λ, w; t = 0.1, penalize_diag = true)
  G = similar(Z)
  for k in axes(Z, 3)
    G[:, :, k] = w[k] * (S[:, :, k] - inv(Z[:, :, k]))
  end
  g = FiberSum(NormL2(λ); diagonal = penalize_diag ? NormL2(λ) : nothing)
  maximum(abs, prox(g, Z - t * G, t) - Z)
end

# sample covariances of n observations from K tridiagonal precision matrices
function covariances(p, K; n = 200)
  S = zeros(p, p, K)
  for k in 1:K
    Ω = Matrix(SymTridiagonal(fill(1.0, p), fill(0.2 + 0.05k, p - 1)))
    S[:, :, k] = cov(randn(n, p) / cholesky(Ω).U')
  end
  S
end

@testset "group graphical lasso" begin
  p = 12

  # one matrix is the graphical lasso
  S1 = covariances(p, 1)
  for pd in (true, false)
    Z = groupGraphicalLasso(S1, 0.1; penalize_diag = pd, options = tight)
    Zc = zeros(p, p)
    CovarianceSelection.covsel!(zeros(p, p), Zc, zeros(p, p), S1[:, :, 1], 0.1; penalize_diag = pd, options = tight)
    @test Z[:, :, 1] ≈ Zc atol = 1e-6
  end

  # K matrices: the optimality conditions, and one support
  K = 3
  S = covariances(p, K)
  w = [0.2, 0.3, 0.5]
  λ = 0.05
  for pd in (true, false)
    Z = groupGraphicalLasso(S, λ; weights = w, penalize_diag = pd, options = tight)
    @test gglResidual(Z, S, λ, w; penalize_diag = pd) < 1e-6
    for k in 1:K
      @test issymmetric(Z[:, :, k]) && isposdef(Z[:, :, k])
    end
    support = Z[:, :, 1] .!= 0
    @test all((Z[:, :, k] .!= 0) == support for k in 2:K)
    @test 0 < count(support) < p^2
  end
  # the same from a vector of matrices
  Zv = groupGraphicalLasso([S[:, :, k] for k in 1:K], λ; weights = w, options = tight)
  @test Zv ≈ groupGraphicalLasso(S, λ; weights = w, options = tight) atol = 1e-10
  # equal weights by default
  @test gglResidual(groupGraphicalLasso(S, λ; options = tight), S, λ, fill(1 / K, K)) < 1e-6

  # a large penalty leaves the diagonal only
  Z = groupGraphicalLasso(S, 10.0; options = tight)
  @test all(Z[i, j, k] == 0 for i in 1:p, j in 1:p, k in 1:K if i != j)

  # info, and invariance to the scale of the data
  Z1 = groupGraphicalLasso(S, λ; weights = w)
  its = Int[]
  for c in (1e-3, 1.0, 1e3)
    info = ADMMInfo()
    Zc = groupGraphicalLasso(c * S, c * λ; weights = w, info)
    @test info.converged
    push!(its, info.iterations)
    @test c * Zc ≈ Z1 rtol = 1e-6
  end
  @test its[1] == its[2] == its[3]
  @test_logs (:warn, r"groupGraphicalLasso! stopped after 2 iterations") groupGraphicalLasso(S, λ;
      options = ADMMOptions(; maxiter = 2))

  # a warm start from the iterates and the final ρ of a converged run stops at once
  Z, U = zeros(p, p, K), zeros(p, p, K)
  info = ADMMInfo()
  groupGraphicalLasso!(Z, U, S, λ; weights = w, info)
  info2 = ADMMInfo()
  groupGraphicalLasso!(Z, U, S, λ; weights = w, options = ADMMOptions(; ρ = info.ρ), info = info2)
  @test info2.converged && info2.iterations <= 2

  # argument checks
  @test_throws DimensionMismatch groupGraphicalLasso(S, λ; weights = [0.5, 0.5])
  @test_throws ArgumentError groupGraphicalLasso(S, λ; weights = [0.5, 0.5, 0.0])
  @test_throws ArgumentError groupGraphicalLasso(S, -1.0)
  @test_throws DimensionMismatch groupGraphicalLasso!(zeros(p, p, K - 1), zeros(p, p, K - 1), S, λ)

  # Float32
  Z32 = groupGraphicalLasso(Float32.(S), 0.05f0; options = ADMMOptions(; abstol = 1e-6, reltol = 1e-6))
  @test eltype(Z32) == Float32
  @test Z32 ≈ groupGraphicalLasso(S, λ; options = ADMMOptions(; abstol = 1e-6, reltol = 1e-6)) atol = 1e-3
end

@testset "time-varying graphs" begin
  n, p = 400, 5
  t = sort(rand(n))
  X = randn(n, p) .+ t       # the mean drifts with time
  τ, h = [0.25, 0.5, 0.75], 0.2

  # the kernel-weighted covariances, against the formula
  for kernel in (:epanechnikov, :gaussian, u -> exp(-abs(u))), center in (true, false)
    S = kernelCovariances(X, t, τ, h; kernel, center)
    kf = kernel isa Symbol ? CovarianceSelection._kernel(kernel) : kernel
    for (k, τk) in enumerate(τ)
      wk = kf.((t .- τk) ./ h)
      m = center ? vec(sum(wk .* X; dims = 1)) / sum(wk) : zeros(p)
      Xc = X .- m'
      @test S[:, :, k] ≈ Xc' * (wk .* Xc) / sum(wk)
      @test issymmetric(S[:, :, k])
    end
  end
  @test_throws ArgumentError kernelCovariances(X, t, [2.0], 0.1)       # no observation near τ = 2
  @test_throws ArgumentError kernelCovariances(X, t, τ, 0.0)
  @test_throws ArgumentError kernelCovariances(X, t, τ, h; kernel = :box)
  @test_throws DimensionMismatch kernelCovariances(X, t[2:end], τ, h)

  # tvCovSel is the group graphical lasso of the kernel-weighted covariances
  Z = tvCovSel(X, t, τ, h, 0.05; options = tight)
  @test Z ≈ groupGraphicalLasso(kernelCovariances(X, t, τ, h), 0.05; options = tight)

  # a chain graph whose edges weaken over time: the edges are found at every time point
  n, p = 3000, 6
  t = sort(rand(n))
  X = zeros(n, p)
  for i in 1:n
    Ω = Matrix(SymTridiagonal(fill(1.0, p), fill(0.45 - 0.2t[i], p - 1)))
    X[i, :] = cholesky(Ω).U \ randn(p)
  end
  Z = tvCovSel(X, t, τ, h, 0.02)
  for k in eachindex(τ), i in 1:p-1
    @test Z[i, i+1, k] > 0                           # the chain edges, with their sign
    @test Z[i, i+1, k] ≈ 0.45 - 0.2τ[k] atol = 0.1   # near the precision at τₖ, shrunk
  end
  @test sum(Z[i, i+1, 1] for i in 1:p-1) > sum(Z[i, i+1, 3] for i in 1:p-1)   # and they weaken
  @test count(Z[i, j, 1] != 0 for i in 1:p, j in 1:p if abs(i - j) > 1) < p^2 / 2
end

end
