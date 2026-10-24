module IHTSolversTests

using Test, LinearAlgebra, Random
using ProximalBase
using ProximalSolvers
import CovarianceSelection
const CS = CovarianceSelection

const METHODS = (ProximalGradientDescent(), ProximalGradientDescent(; normalized = true),
                 AccProxGradDescent(), MonotoneAccProxGradDescent())

theta(S, U) = Matrix(S) + U * U'
triangle_nnz(S; diagonal = true) = count(!iszero, S[i, j] for j in axes(S, 2) for i in (diagonal ? j : j+1):size(S, 1))
function dtrace(A, B, S, U)
  Θ = theta(S, U)
  dot(Θ, A * Θ * B) / 2 + dot(A - B, Θ)
end
logloss(Σ, S, U) = (Θ = theta(S, U); dot(Σ, Θ) - logdet(Symmetric(Θ)))

function latent_start(::Type{T}) where {T}
  Matrix{T}(I, 4, 4), reshape(T[0.12, -0.08, 0.06, 0.1], 4, 1)
end

@testset "latent automatic-step losses differentiate both blocks" begin
  rng = Xoshiro(2424)
  A = Matrix(SymTridiagonal([1.0, 1.5, 2.0, 1.2], [0.1, -0.1, 0.15]))
  B = Matrix(Diagonal([1.2, 0.8, 1.6, 1.0]))
  Σ = Matrix(SymTridiagonal([4.0, 3.0, 2.5, 3.5], [0.25, -0.2, 0.15]))
  S0, U0 = latent_start(Float64)
  x = (SymmetricSparseIterate(S0), U0)
  D = Matrix(Symmetric(randn(rng, 4, 4)))
  V = randn(rng, 4, 1)
  h = 1e-6
  xp = (SymmetricSparseIterate(S0 + h * D), U0 + h * V)
  xm = (SymmetricSparseIterate(S0 - h * D), U0 - h * V)
  for f in (CS._LatentLogdet(Σ), CS._LatentDTrace(Symmetric(A), Symmetric(B)))
    G = (zeros(4, 4), zeros(4, 1))
    value = value_and_gradient!(G, f, x)
    @test value ≈ f(x)
    @test G[1] ≈ G[1]' atol = 1e-13
    @test norm(G[1]) > 0 && norm(G[2]) > 0
    @test (f(xp) - f(xm)) / (2h) ≈ dot(G[1], D) + dot(G[2], V) rtol = 1e-6 atol = 1e-7
  end
  f = CS._LatentLogdet(Σ)
  invalid = (SymmetricSparseIterate(-S0), zeros(4, 1))
  @test f(invalid) == Inf
  @test value_and_gradient!((zeros(4, 4), zeros(4, 1)), f, invalid) == Inf
end

@testset "automatic D-trace IHT is proximal gradient with an exact projection" begin
  for T in (Float32, Float64)
    A = Matrix{T}(I, 2, 2)
    B = T[2 0.8; 0.8 1.1]
    f, g = KronQuadratic(A, B, A - B), IndBallL0(1)
    # An off-diagonal pair outweighs the largest diagonal in Frobenius norm.
    projection = prox(g, Symmetric(B - A))
    @test projection[1, 1] == projection[2, 2] == 0
    @test projection[1, 2] == projection[2, 1] == T(0.8)
    method = ProximalGradientDescent()
    opt = ProximalOptions(; maxiter = 10, callback = _ -> true, store_trace = true)
    Δ, values = CS.differencePrecisionIHT(Symmetric(A), Symmetric(B), 1; method, options = opt)
    reference = SymmetricSparseIterate(T, 2)
    r = solve!(reference, f, g, method, opt)
    @test Δ isa SymmetricSparseIterate{T}
    @test Matrix(Δ) ≈ Matrix(reference)
    @test triangle_nnz(Δ) == 1
    @test iszero(Δ[1, 1]) && iszero(Δ[2, 2]) && !iszero(Δ[1, 2])
    @test length(values) == 2
    @test values[1] == 0
    @test values[end] ≈ f(Δ)
    @test values[end] ≈ r.minimum

    # A nonzero start is copied, and the wrapper preserves the chosen method.
    init = Matrix(Δ)
    before = copy(init)
    out, _ = CS.differencePrecisionIHT(Symmetric(A), Symmetric(B), 1;
        init, method = MonotoneAccProxGradDescent(), options = opt)
    @test init == before
    @test triangle_nnz(out) <= 1
  end
end

@testset "automatic latent log-det IHT guards the positive definite cone" begin
  Σ = Matrix(SymTridiagonal([4.0, 3.0, 2.5, 3.5], [0.25, -0.2, 0.15]))
  S0, U0 = latent_start(Float64)
  Θ0 = theta(S0, U0)
  G = Σ - inv(Symmetric(Θ0))
  trialS = prox(IndBallL0(1; diagonal = false), Symmetric(S0 - G / 0.9))
  trialU = U0 - 2G * U0 / 0.9
  @test !isposdef(Symmetric(theta(trialS, trialU)))
  for method in METHODS
    saved = Matrix{Float64}[]
    callback_values = Float64[]
    callback = (Σ, S, L) -> begin
      push!(saved, Matrix(S) + L)
      push!(callback_values, dot(Σ, Matrix(S) + L) - logdet(Symmetric(Matrix(S) + L)))
      true                         # The legacy callback does not stop the run.
    end
    S, L, U, values, iter = CS.covselLatentIHT!(Σ, 1, 1; init = (S0, U0), method, callback,
        options = ProximalOptions(; maxiter = 80, ftol = 1e-12, xtol = 1e-12, gradtol = 1e-9))
    @test S isa SymmetricSparseIterate
    @test S0 == Matrix{Float64}(I, 4, 4) && U0 == latent_start(Float64)[2]
    @test L ≈ U * U'
    @test size(U) == (4, 1)
    @test triangle_nnz(S; diagonal = false) <= 1
    @test iter > 1 && length(saved) == iter && length(values) == iter + 1
    @test all(Θ -> isposdef(Symmetric(Θ)), saved)
    @test values[2:end] ≈ callback_values
    @test all(isfinite, values)
    @test all(diff(values) .<= 1e-9 * max(1, abs(values[1])))
    @test values[end] ≈ logloss(Σ, S, U)
    @test values[end] < values[1] - 1e-4
  end

  # ProximalOptions.callback keeps its stopping semantics alongside the legacy callback.
  calls = Ref(0)
  out = CS.covselLatentIHT!(Σ, 1, 1; init = (S0, U0), callback = (_, _, _) -> (calls[] += 1),
      options = ProximalOptions(; maxiter = 20, callback = _ -> true))
  @test out[5] == 1 && calls[] == 1 && length(out[4]) == 2
  default = CS.covselLatentIHT!(Σ, 1, 1;
      options = ProximalOptions(; maxiter = 10, callback = _ -> true))
  @test all(isfinite, default[4]) && isposdef(Symmetric(Matrix(default[1]) + default[2]))
  @test default[2] ≈ default[3] * default[3]'
end

@testset "automatic latent D-trace IHT updates both blocks" begin
  for T in (Float32, Float64)
    A = Matrix(SymTridiagonal(T[1, 1.5, 2, 1.2], T[0.1, -0.1, 0.15]))
    B = Matrix(Diagonal(T[1.2, 0.8, 1.6, 1]))
    S0, U0 = latent_start(T)
    # Count the diagonal in the D-trace sparsity budget, as the original API does.
    s = 5
    for method in (ProximalGradientDescent(), MonotoneAccProxGradDescent())
      saved_values = T[]
      calls = Ref(0)
      callback = (S, L) -> begin
        calls[] += 1
        Θ = Matrix(S) + L
        push!(saved_values, dot(Θ, A * Θ * B) / 2 + dot(A - B, Θ))
      end
      opt = ProximalOptions(; maxiter = 80, ftol = (T === Float32 ? 1e-6 : 1e-12),
                           xtol = (T === Float32 ? 1e-6 : 1e-12), gradtol = (T === Float32 ? 1e-5 : 1e-9))
      S, L, U, values, iter = CS.differenceLatentPrecisionIHT(Symmetric(A), Symmetric(B), s, 1;
          init = (S0, U0), method, options = opt, callback)
      @test S isa SymmetricSparseIterate{T} && eltype(U) === T && eltype(L) === T
      @test S0 == Matrix{T}(I, 4, 4) && U0 == latent_start(T)[2]
      @test triangle_nnz(S) <= s
      @test L ≈ U * U'
      @test calls[] == iter && length(values) == iter + 1
      @test values[2:end] ≈ saved_values
      @test all(isfinite, values)
      @test values[end] ≈ dtrace(A, B, S, U)
      @test values[end] < values[1] - T(1e-5)
      @test all(diff(values) .<= (T === Float32 ? 2e-5 : 1e-9) * max(1, abs(values[1])))

      # The in-place overload shares all three output buffers and refreshes L.
      Sin, Lin, Uin = copy(S), zeros(T, 4, 4), copy(U)
      result = CS.differenceLatentPrecisionIHT!(Sin, Lin, Uin, Symmetric(A), Symmetric(B), s;
          method, options = ProximalOptions(; maxiter = 10, callback = _ -> true))
      @test result[1] === Sin && result[2] === Lin && result[3] === Uin
      @test Lin ≈ Uin * Uin'
      @test triangle_nnz(Sin) <= s
      @test result[4][1] ≈ dtrace(A, B, S, U)
      @test result[4][end] ≈ dtrace(A, B, Sin, Uin)
    end
  end
  A = Symmetric(Matrix{Float64}(I, 4, 4))
  B = Symmetric(Matrix(Diagonal([1.2, 0.8, 1.6, 1.0])))
  default = CS.differenceLatentPrecisionIHT(A, B, 4, 1;
      options = ProximalOptions(; maxiter = 10, callback = _ -> true))
  @test all(isfinite, default[4]) && triangle_nnz(default[1]) <= 4
  @test default[2] ≈ default[3] * default[3]'
end

@testset "automatic IHT argument checks" begin
  A = Matrix{Float64}(I, 4, 4)
  S, U = latent_start(Float64)
  @test_throws DimensionMismatch CS.differencePrecisionIHT(Symmetric(A), Symmetric(A[1:3, 1:3]), 1)
  @test_throws ArgumentError CS.differencePrecisionIHT(Symmetric(A), Symmetric(A), -1)
  @test_throws DimensionMismatch CS.differencePrecisionIHT(Symmetric(A), Symmetric(A), 1; init = zeros(3, 3))
  @test_throws ArgumentError CS.differencePrecisionIHT(Symmetric(A), Symmetric(A), 1; method = CDSolver())
  @test_throws DimensionMismatch CS.covselLatentIHT!(zeros(3, 4), 1, 1)
  @test_throws ArgumentError CS.covselLatentIHT!(A, 1, -1)
  @test_throws ArgumentError CS.covselLatentIHT!(A, 1, 5)
  @test_throws ArgumentError CS.covselLatentIHT!(A, 1, 1; init = (-A, U))
  @test_throws DimensionMismatch CS.covselLatentIHT!(A, 1, 1; init = (S, zeros(4, 2)))
  @test_throws DimensionMismatch CS.differenceLatentPrecisionIHT(Symmetric(A), Symmetric(A), 1, 1;
      init = (zeros(3, 3), U))
  @test_throws DimensionMismatch CS.differenceLatentPrecisionIHT!(SymmetricSparseIterate(4), zeros(3, 3), U,
      Symmetric(A), Symmetric(A), 1)
end

end
