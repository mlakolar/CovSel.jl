module CovSelTest

using Test, Random, LinearAlgebra
import CovSel, ProximalBase

include("optimality.jl")

Random.seed!(123)

objective(Ω, S, λ; penalize_diag=true) =
  tr(Ω*S) - logdet(Ω) + λ * (penalize_diag ? sum(abs, Ω) : sum(abs, Ω) - sum(abs, diag(Ω)))

function ar1Cov(p, ρ)
  [ρ^abs(a - b) for a=1:p, b=1:p]
end

const tight = CovSel.ADMMOptions(;abstol=1e-10, reltol=1e-10, maxiter=10000)

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
      CovSel.covsel!(X, Z, U, S, λ; penalize_diag=penalize_diag, options=tight)

      @test issymmetric(Z)
      @test isposdef(Z)
      @test covselFixedPointResidual(Z, S, λ; penalize_diag=penalize_diag) < 1e-6
      @test isLocalMin(Ω -> objective(Ω, S, λ; penalize_diag=penalize_diag), Z)

      # the default (loose) tolerances give a nearby solution
      Z1 = zeros(p, p)
      CovSel.covsel!(zeros(p, p), Z1, zeros(p, p), S, λ; penalize_diag=penalize_diag)
      @test norm(Z1 - Z, Inf) < 2e-2
    end

    # large λ: only the diagonal survives, Ω = Diagonal(1 / (S_ii + λ))
    S = ar1Cov(5, 0.3)
    Z = zeros(5, 5)
    CovSel.covsel!(zeros(5, 5), Z, zeros(5, 5), S, 1.0; options=tight)
    @test Z ≈ Diagonal(1 ./ (diag(S) .+ 1.0))
  end

  @testset "non random" begin
    p = 10
    S = Matrix(1.0I, p, p)

    λ = 0.5

    X = zeros(Float64, (p,p))
    Z = zeros(Float64, (p,p))
    U = zeros(Float64, (p,p))
    CovSel.covsel!(X, Z, U, S, λ; penalize_diag=false)
    @test Z ≈ S atol=1e-3

    CovSel.covsel!(X, Z, U, S, λ; penalize_diag=false, options=CovSel.ADMMOptions(;abstol=1e-12,reltol=1e-12))
    @test Z ≈ S

    # with the diagonal penalized the solution is 1 / (1 + λ) ⋅ I
    fill!(X, 0.); fill!(Z, 0.); fill!(U, 0.)
    CovSel.covsel!(X, Z, U, S, λ; options=CovSel.ADMMOptions(;abstol=1e-12,reltol=1e-12))
    @test Z ≈ S / (1 + λ)
  end

  @testset "path" begin
    n, p = 500, 8
    X = randn(n, p) * sqrt(ar1Cov(p, 0.5))
    S = X' * X / n
    λarr = [0.5, 0.2, 0.1]
    opt = CovSel.ADMMOptions(;abstol=1e-10, reltol=1e-10, maxiter=5000)

    path = CovSel.covselpath(S, λarr; options=opt)
    @test length(path) == length(λarr)
    for (i, λ) in enumerate(λarr)
      Z = zeros(p, p)
      CovSel.covsel!(zeros(p, p), Z, zeros(p, p), S, λ; options=opt)
      @test path[i] ≈ Z atol=1e-5
      @test covselFixedPointResidual(path[i], S, λ) < 1e-6
    end
    # smaller λ, denser estimate
    @test issorted([count(!iszero, Z) for Z in path])

    @test redirect_stdout(devnull) do
      CovSel.covselpath(S, λarr[1:1]; options=opt, verbose=true)
    end[1] ≈ path[1]
  end

  @testset "Float32" begin
    p = 5
    S = Float32.(ar1Cov(p, 0.5))
    X, Z, U = zeros(Float32, p, p), zeros(Float32, p, p), zeros(Float32, p, p)
    CovSel.covsel!(X, Z, U, S, 0.1; options=CovSel.ADMMOptions(;abstol=1e-7, reltol=1e-7))
    @test eltype(Z) == Float32
    Z64 = zeros(p, p)
    CovSel.covsel!(zeros(p, p), Z64, zeros(p, p), Float64.(S), 0.1; options=CovSel.ADMMOptions(;abstol=1e-7, reltol=1e-7))
    @test Z ≈ Z64 atol=1e-3
  end
end

end
