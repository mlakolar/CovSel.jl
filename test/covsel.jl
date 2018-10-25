module CovSelTest

using Test, Random
using Distributions
import CovSel, ProximalBase
import SCS, JuMP
using LinearAlgebra


@testset "covsel" begin

  @testset "random" begin
      Random.seed!(123)
      n = 1000
      p = 10

      ρ = 0.5
      covMat = eye(p)
      for a = 1:p
        for b = a+1:p
          t = ρ^(b - a)
          covMat[a, b] = t
          covMat[b, a] = t
        end
      end
      precM = inv(covMat)

      sqCov = sqrtm(covMat)
      X = randn(n, p) * sqCov

      S = X' * X / n

      λ = 0.5

      X = zeros(Float64, (p,p))
      Z = zeros(Float64, (p,p))
      U = zeros(Float64, (p,p))
      CovSel.covsel!(X, Z, U, S, λ; penalize_diag=true)

      # passing in verbose=0 to hide output from SCS
      problem = JuMP.Model(solver = SCS.SCSSolver(verbose=0))
      JuMP.@variable(problem, Ω[1:p, 1:p], SDP)
      JuMP.@objective(problem, Min, tr(Ω*S) - logdet(Ω) + λ * vecnorm(Ω, 1))
      JuMP.solve(problem)

      @test JuMP.getvalue(Ω) - Z ≈ zeros(p,p) atol = 1e-2
  end

  # @testset "non random" begin
  #
  #   p = 10
  #   S = eye(p)
  #
  #   λ = 0.5
  #
  #   X = zeros(Float64, (p,p))
  #   Z = zeros(Float64, (p,p))
  #   U = zeros(Float64, (p,p))
  #   CovSel.covsel!(X, Z, U, S, λ; penalize_diag=false)
  #   @fact Z --> roughly(eye(p); atol=1e-3)
  #
  #   CovSel.covsel!(X, Z, U, S, λ; penalize_diag=false, options=CovSel.ADMMOptions(;abstol=1e-12,reltol=1e-12))
  #   @fact Z --> roughly(eye(p))
  # end

end


end
