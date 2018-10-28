module FusedDifferenceTest

using Test, Random, Distributions, LinearAlgebra
import CovSel, ProximalBase
using SCS, JuMP, Ipopt
import MathOptInterface
const MOI = MathOptInterface

Random.seed!(123)


@testset "fused_graphical_lasso" begin
    # simple model that differes in only one edge
    p = 10
    Sigma_x = Matrix(1.0I, p, p)
    ρ = 0.3
    iρ = -ρ / (1. - ρ^2)
    Sigma_y = Matrix(1.0I, p, p)
    Sigma_y[1,2] = iρ
    Sigma_y[2,1] = iρ
    Sigma_y[1,1] = 1. / (1. - ρ^2)
    Sigma_y[2,2] = 1. / (1. - ρ^2);

    n = 1000
    X = randn(n, p)
    Y = randn(n, p) * sqrt(Sigma_y)

    # passing in verbose=0 to hide output from SCS
    problem = Model(with_optimizer(SCS.Optimizer, verbose=0))

    Sx = cov(X)
    Sy = cov(Y)
    @variable(problem, Ωx[1:p, 1:p], PSD)
    @variable(problem, Ωy[1:p, 1:p], PSD)
    @variable(problem, lg_det_x)
    @variable(problem, lg_det_y)
    @variable(problem, Bx[1:p, 1:p])
    @variable(problem, By[1:p, 1:p])
    @variable(problem, Bd[1:p, 1:p])

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
    @constraint(problem, [lg_det_x; Ωx[indOffDiag]] in MOI.LogDetConeTriangle(p))
    @constraint(problem, [lg_det_y; Ωy[indOffDiag]] in MOI.LogDetConeTriangle(p))

    @constraint(problem,  Ωx .<= Bx)
    @constraint(problem, -Ωx .<= Bx)
    @constraint(problem,  Ωy .<= By)
    @constraint(problem, -Ωy .<= By)
    @constraint(problem,   Ωx - Ωy  .<= Bd)
    @constraint(problem, -(Ωx - Ωy) .<= Bd)

    for i=1:20

        λ1 = rand(Uniform(0.01,0.1))
        λ2 = rand(Uniform(0.01,0.1))

        θx = Matrix(1.0I, p, p)
        θy = Matrix(1.0I, p, p)
        Zx = Matrix(1.0I, p, p)
        Zy = Matrix(1.0I, p, p)
        Ux = zeros(p,p)
        Uy = zeros(p,p)

        CovSel.fusedGraphicalLasso!(θx, θy, Zx, Zy, Ux, Uy, Sx, n, Sy, n, λ1, λ2;
            penalize_diag=true,
            options=CovSel.ADMMOptions(;abstol=1e-12,reltol=1e-12))

        @objective(problem, Min,
            (tr(Ωx*Sx) - lg_det_x) / 2.0 +
            (tr(Ωy*Sy) - lg_det_y) / 2.0 +
            λ1 * (sum(Bx) + sum(By)) +
            λ2 * sum(Bd)
            )
        optimize!(problem)

        @test JuMP.result_value.(Ωx) - Zx ≈ zeros(p,p) atol = 1e-2
        @test JuMP.result_value.(Ωy) - Zy ≈ zeros(p,p) atol = 1e-2
    end

end


@testset "fused_neighborhood_selection" begin
  # simple model that differes in only one edge
  p = 5
  Sigma_x = Matrix(1.0I, p, p)
  ρ = 0.3
  iρ = -ρ / (1. - ρ^2)
  Sigma_y = Matrix(1.0I, p, p)
  Sigma_y[1,2] = iρ
  Sigma_y[2,1] = iρ
  Sigma_y[1,1] = 1. / (1. - ρ^2)
  Sigma_y[2,2] = 1. / (1. - ρ^2);

  n = 1000
  X = randn(n, p)
  Y = randn(n, p) * sqrt(Sigma_y)

  problem = Model(with_optimizer(Ipopt.Optimizer, print_level=0))

  # m = JuMP.Model(solver=Gurobi.GurobiSolver(OutputFlag=0))
  @variable(problem, xj1[1:p-1])
  @variable(problem, xj2[1:p-1])
  @variable(problem, t1[1:p-1])
  @variable(problem, t2[1:p-1])
  @variable(problem, t3[1:p-1])

  @constraint(problem, xj1 .<= t1 )
  @constraint(problem, -t1 .<= xj1 )
  @constraint(problem, xj2 .<= t2 )
  @constraint(problem, -t2 .<= xj2 )
  @constraint(problem, -t3 .<= xj1 - xj2 )
  @constraint(problem, xj1 - xj2 .<= t3 )

  for i=1:20
    λ1 = rand(Uniform(0., 0.1))
    λ2 = rand(Uniform(0., 0.1))
    z1, z2 = CovSel.fusedNeighborhoodSelection(X[:,2:p], X[:,1], Y[:,2:p], Y[:,1], λ1, λ2;
                      options=CovSel.ADMMOptions(;abstol=1e-12,reltol=1e-12))

    @objective(problem, Min, (sum((X[:,1] - X[:,2:p]*xj1) .^ 2) + sum((Y[:,1] - Y[:,2:p]*xj2) .^ 2)) / (2. * n) + λ1 * (sum(t1)+sum(t2)) + λ2 * sum(t3))
    optimize!(problem)

    @test JuMP.result_value.(xj1) - z1 ≈ zeros(p-1) atol=3e-5
    @test JuMP.result_value.(xj2) - z2 ≈ zeros(p-1) atol=3e-5
  end
end


end
