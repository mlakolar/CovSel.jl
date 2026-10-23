module CovarianceSelectionTest

using Test, Random, LinearAlgebra, Statistics
import CovarianceSelection, ProximalBase
using CovarianceSelection: covsel!, ADMMOptions, ADMMInfo

include("optimality.jl")

Random.seed!(123)

objective(Ω, S, λ; penalize_diag=true) =
  tr(Ω*S) - logdet(Ω) + λ * (penalize_diag ? sum(abs, Ω) : sum(abs, Ω) - sum(abs, diag(Ω)))

function ar1Cov(p, ρ)
  [ρ^abs(a - b) for a=1:p, b=1:p]
end

const tight = CovarianceSelection.ADMMOptions(;abstol=1e-10, reltol=1e-10, maxiter=10000)

@testset "covsel" begin

  @testset "random" begin
    for REP=1:5, penalize_diag in (true, false)
      n = 1000
      p = 10
      X = randn(n, p) * sqrt(ar1Cov(p, 0.5))
      S = X' * X / n

      λ = 0.1 + 0.4 * rand()

      X = zeros(Float64, (p,p))
      Z = zeros(Float64, (p,p))
      U = zeros(Float64, (p,p))
      CovarianceSelection.covsel!(X, Z, U, S, λ; penalize_diag=penalize_diag, options=tight)

      @test issymmetric(Z)
      @test isposdef(Z)
      @test covselFixedPointResidual(Z, S, λ; penalize_diag=penalize_diag) < 1e-6
      @test isLocalMin(Ω -> objective(Ω, S, λ; penalize_diag=penalize_diag), Z)

      # the default (loose) tolerances give a nearby solution
      Z1 = zeros(p, p)
      CovarianceSelection.covsel!(zeros(p, p), Z1, zeros(p, p), S, λ; penalize_diag=penalize_diag)
      @test norm(Z1 - Z, Inf) < 2e-2
    end

    # large λ: only the diagonal survives, Ω = Diagonal(1 / (S_ii + λ))
    S = ar1Cov(5, 0.3)
    Z = zeros(5, 5)
    CovarianceSelection.covsel!(zeros(5, 5), Z, zeros(5, 5), S, 1.0; options=tight)
    @test Z ≈ Diagonal(1 ./ (diag(S) .+ 1.0))
  end

  @testset "non random" begin
    p = 10
    S = Matrix(1.0I, p, p)

    λ = 0.5

    X = zeros(Float64, (p,p))
    Z = zeros(Float64, (p,p))
    U = zeros(Float64, (p,p))
    CovarianceSelection.covsel!(X, Z, U, S, λ; penalize_diag=false)
    @test Z ≈ S atol=1e-3

    CovarianceSelection.covsel!(X, Z, U, S, λ; penalize_diag=false, options=CovarianceSelection.ADMMOptions(;abstol=1e-12,reltol=1e-12))
    @test Z ≈ S

    # with the diagonal penalized the solution is 1 / (1 + λ) ⋅ I
    fill!(X, 0.); fill!(Z, 0.); fill!(U, 0.)
    CovarianceSelection.covsel!(X, Z, U, S, λ; options=CovarianceSelection.ADMMOptions(;abstol=1e-12,reltol=1e-12))
    @test Z ≈ S / (1 + λ)
  end

  @testset "path" begin
    n, p = 500, 8
    X = randn(n, p) * sqrt(ar1Cov(p, 0.5))
    S = X' * X / n
    λarr = [0.5, 0.2, 0.1]
    opt = CovarianceSelection.ADMMOptions(;abstol=1e-10, reltol=1e-10, maxiter=5000)

    path = CovarianceSelection.covselpath(S, λarr; options=opt)
    @test length(path) == length(λarr)
    for (i, λ) in enumerate(λarr)
      Z = zeros(p, p)
      CovarianceSelection.covsel!(zeros(p, p), Z, zeros(p, p), S, λ; options=opt)
      @test path[i] ≈ Z atol=1e-5
      @test covselFixedPointResidual(path[i], S, λ) < 1e-6
    end
    # smaller λ, denser estimate
    @test issorted([count(!iszero, Z) for Z in path])

    @test redirect_stdout(devnull) do
      CovarianceSelection.covselpath(S, λarr[1:1]; options=opt, verbose=true)
    end[1] ≈ path[1]
  end

  @testset "ADMM options" begin
    p = 30
    S = cov(randn(300, p) * cholesky(ar1Cov(p, 0.5)).U)
    solve(S, λ, opt; info=ADMMInfo()) = (local W = zeros(p, p); covsel!(zeros(p, p), W, zeros(p, p), S, λ; options=opt, info=info); W)

    # info
    info = ADMMInfo()
    Z = solve(S, 0.1, ADMMOptions(); info=info)
    @test info.converged
    @test 1 <= info.iterations < 10_000
    @test isfinite(info.ρ) && info.ρ > 0
    @test info.r_norm >= 0 && info.s_norm >= 0

    # multiplying the data by a constant rescales the solution and leaves the
    # number of iterations unchanged, with or without adaptive ρ
    for adaptive in (true, false)
      its = Int[]
      Z1 = solve(S, 0.1, ADMMOptions(; adaptive=adaptive))
      for c in (1e-3, 1.0, 1e3)
        info = ADMMInfo()
        Zc = solve(c * S, c * 0.1, ADMMOptions(; adaptive=adaptive); info=info)
        push!(its, info.iterations)
        @test c * Zc ≈ Z1 rtol=1e-6
      end
      @test its[1] == its[2] == its[3]
    end

    # adaptive ρ balances the residuals when the variances are very unequal
    D = Diagonal(exp.(range(log(0.1), log(10), length=p)))
    Sh = D * S * D
    λ = 0.05 * tr(Sh) / p
    fixed, adaptive = ADMMInfo(), ADMMInfo()
    Zf = solve(Sh, λ, ADMMOptions(; adaptive=false); info=fixed)
    Za = solve(Sh, λ, ADMMOptions(); info=adaptive)
    @test fixed.converged && adaptive.converged
    @test adaptive.iterations < fixed.iterations / 5
    @test adaptive.ρ != 1
    @test covselFixedPointResidual(Za, Sh, λ; t=0.1 / maximum(diag(Sh))) < 1e-3 * maximum(abs, Za)

    # warm starting from a solution with its final ρ, which the dual variable is
    # scaled by, stops after one iteration
    X, Z, U = zeros(p, p), zeros(p, p), zeros(p, p)
    info = ADMMInfo()
    covsel!(X, Z, U, Sh, 0.01 * tr(Sh) / p; info=info)
    @test info.converged && info.ρ != 1
    covsel!(X, Z, U, Sh, 0.01 * tr(Sh) / p; options=ADMMOptions(; ρ=info.ρ), info=info)
    @test info.iterations == 1

    # stopping at maxiter gives a warning
    info = ADMMInfo()
    @test_logs (:warn, r"covsel! stopped after 3 iterations") solve(S, 0.1, ADMMOptions(; maxiter=3); info=info)
    @test !info.converged
    @test info.iterations == 3

    # the defaults
    o = ADMMOptions()
    @test (o.ρ, o.α, o.maxiter, o.abstol, o.reltol, o.adaptive) == (1.0, 1.5, 10_000, 1e-5, 1e-3, true)
  end

  @testset "Float32" begin
    p = 5
    S = Float32.(ar1Cov(p, 0.5))
    X, Z, U = zeros(Float32, p, p), zeros(Float32, p, p), zeros(Float32, p, p)
    CovarianceSelection.covsel!(X, Z, U, S, 0.1; options=CovarianceSelection.ADMMOptions(;abstol=1e-7, reltol=1e-7))
    @test eltype(Z) == Float32
    Z64 = zeros(p, p)
    CovarianceSelection.covsel!(zeros(p, p), Z64, zeros(p, p), Float64.(S), 0.1; options=CovarianceSelection.ADMMOptions(;abstol=1e-7, reltol=1e-7))
    @test Z ≈ Z64 atol=1e-3
  end
end

end
