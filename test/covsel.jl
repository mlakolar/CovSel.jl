module CovSelTest

using Test, Random, LinearAlgebra, Distributions
import CovSel, ProximalBase
using MathOptInterface
const MOI = MathOptInterface
using JuMP, SCS

Random.seed!(123)


@testset "random" begin
  for REP=1:10

      n = 1000
      p = 10

      ρ = 0.5
      covMat = Matrix(1.0I, p, p)
      for a = 1:p
        for b = a+1:p
          t = ρ^(b - a)
          covMat[a, b] = t
          covMat[b, a] = t
        end
      end
      precM = inv(covMat)

      sqCov = sqrt(covMat)
      X = randn(n, p) * sqCov

      S = X' * X / n

      λ = 0.5

      X = zeros(Float64, (p,p))
      Z = zeros(Float64, (p,p))
      U = zeros(Float64, (p,p))
      CovSel.covsel!(X, Z, U, S, λ; penalize_diag=true)

      # passing in verbose=0 to hide output from SCS
      problem = Model(JuMP.with_optimizer(SCS.Optimizer, verbose=0))
      @variable(problem, Ω[1:p, 1:p], PSD)
      @variable(problem, lg_det)
      @variable(problem, B[1:p, 1:p])

      indOffDiag = Vector{Int64}(undef, 0)
      ind = 0
      for col = 1:p
          for row = 1:p
              ind += 1
              if row <= col
                  push!(indOffDiag, ind)
              end
          end
      end

      @constraint(problem, [lg_det; 1; Ω[indOffDiag]] in MOI.LogDetConeTriangle(p))
      @constraint(problem, Ω .<= B)
      @constraint(problem, -Ω .<= B)

      @objective(problem, Min, tr(Ω*S) - lg_det + λ * sum(B))
      optimize!(problem)

      @test JuMP.result_value.(Ω) - Z ≈ zeros(p,p) atol = 1e-2
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
end



end
