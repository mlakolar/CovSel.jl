module CovSelTest

using Test, Random, LinearAlgebra
import CovSel, ProximalBase
using JuMP, SCS

Random.seed!(123)

# solves the graphical lasso with SCS
function covselSCS(S, λ)
  p = size(S, 1)
  problem = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0, "eps_abs" => 1e-8, "eps_rel" => 1e-8))
  @variable(problem, Ω[1:p, 1:p], PSD)
  @variable(problem, lg_det)
  @variable(problem, B[1:p, 1:p])

  @constraint(problem, [lg_det; 1; [Ω[r, c] for c=1:p for r=1:c]] in MOI.LogDetConeTriangle(p))
  @constraint(problem, Ω .<= B)
  @constraint(problem, -Ω .<= B)

  @objective(problem, Min, tr(Ω*S) - lg_det + λ * sum(B))
  optimize!(problem)
  value.(Ω)
end

objective(Ω, S, λ) = tr(Ω*S) - logdet(Ω) + λ * sum(abs, Ω)

function ar1Cov(p, ρ)
  [ρ^abs(a - b) for a=1:p, b=1:p]
end

@testset "covsel" begin

  @testset "random" begin
    for REP=1:5
      n = 1000
      p = 10
      X = randn(n, p) * sqrt(ar1Cov(p, 0.5))
      S = X' * X / n

      λ = 0.5

      X = zeros(Float64, (p,p))
      Z = zeros(Float64, (p,p))
      U = zeros(Float64, (p,p))
      CovSel.covsel!(X, Z, U, S, λ; penalize_diag=true)

      Ωv = covselSCS(S, λ)
      @test objective(Z, S, λ) ≈ objective(Ωv, S, λ) atol = 1e-2
      @test Ωv ≈ Z atol = 1e-2

      # with tight tolerances ADMM is at least as good as SCS
      CovSel.covsel!(X, Z, U, S, λ; options=CovSel.ADMMOptions(;abstol=1e-10, reltol=1e-10, maxiter=10000))
      @test objective(Z, S, λ) <= objective(Ωv, S, λ) + 1e-6
      @test Ωv ≈ Z atol = 1e-3
    end
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
