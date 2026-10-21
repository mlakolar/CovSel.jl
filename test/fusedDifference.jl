module FusedDifferenceTest

using Test, Random, LinearAlgebra, Statistics
import CovSel, ProximalBase
using SCS, JuMP, Ipopt

Random.seed!(123)

function twoGroups(p, n)
  ρ = 0.3
  iρ = -ρ / (1. - ρ^2)
  Sigma_y = Matrix(1.0I, p, p)
  Sigma_y[1,2] = iρ
  Sigma_y[2,1] = iρ
  Sigma_y[1,1] = 1. / (1. - ρ^2)
  Sigma_y[2,2] = 1. / (1. - ρ^2)
  randn(n, p), randn(n, p) * sqrt(Sigma_y)
end

fglObjective(Ωx, Ωy, Sx, Sy, λ1, λ2) =
  (tr(Ωx*Sx) - logdet(Ωx) + tr(Ωy*Sy) - logdet(Ωy)) / 2. + λ1 * (sum(abs, Ωx) + sum(abs, Ωy)) + λ2 * sum(abs, Ωx - Ωy)

@testset "fused_graphical_lasso" begin
    # simple model that differes in only one edge
    p = 10
    n = 1000
    X, Y = twoGroups(p, n)
    Sx = cov(X)
    Sy = cov(Y)

    # passing in verbose=0 to hide output from SCS
    problem = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
    @variable(problem, Ωx[1:p, 1:p], PSD)
    @variable(problem, Ωy[1:p, 1:p], PSD)
    @variable(problem, lg_det_x)
    @variable(problem, lg_det_y)
    @variable(problem, Bx[1:p, 1:p])
    @variable(problem, By[1:p, 1:p])
    @variable(problem, Bd[1:p, 1:p])

    upper(Ω) = [Ω[r, c] for c=1:p for r=1:c]
    @constraint(problem, [lg_det_x; 1; upper(Ωx)] in MOI.LogDetConeTriangle(p))
    @constraint(problem, [lg_det_y; 1; upper(Ωy)] in MOI.LogDetConeTriangle(p))

    @constraint(problem,  Ωx .<= Bx)
    @constraint(problem, -Ωx .<= Bx)
    @constraint(problem,  Ωy .<= By)
    @constraint(problem, -Ωy .<= By)
    @constraint(problem,   Ωx - Ωy  .<= Bd)
    @constraint(problem, -(Ωx - Ωy) .<= Bd)

    for i=1:10

        λ1 = 0.01 + 0.09 * rand()
        λ2 = 0.01 + 0.09 * rand()

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

        myv = fglObjective(Zx, Zy, Sx, Sy, λ1, λ2)
        scsv = fglObjective(value.(Ωx), value.(Ωy), Sx, Sy, λ1, λ2)
        @test myv - scsv < 1e-6
        @test value.(Ωx) ≈ Zx atol=1e-2
        @test value.(Ωy) ≈ Zy atol=1e-2

        # the allocating version gives the same answer
        Zx1, Zy1 = CovSel.fusedGraphicalLasso(Sx, n, Sy, n, λ1, λ2;
            options=CovSel.ADMMOptions(;abstol=1e-12,reltol=1e-12))
        @test Zx1 ≈ Zx atol=1e-6
        @test Zy1 ≈ Zy atol=1e-6
    end

    # a large fusion penalty makes the two estimates equal
    Zx, Zy = CovSel.fusedGraphicalLasso(Sx, n, Sy, n, 0.05, 10.0;
        options=CovSel.ADMMOptions(;abstol=1e-10,reltol=1e-10))
    @test Zx ≈ Zy atol=1e-6

    # unpenalized diagonal and identity covariances: the solution is I
    Zx, Zy = CovSel.fusedGraphicalLasso(Matrix(1.0I, p, p), n, Matrix(1.0I, p, p), n, 0.5, 0.5;
        penalize_diag=false, options=CovSel.ADMMOptions(;abstol=1e-12,reltol=1e-12))
    @test Zx ≈ I
    @test Zy ≈ I

    # Float32
    Zx32, Zy32 = CovSel.fusedGraphicalLasso(Float32.(Sx), n, Float32.(Sy), n, 0.05, 0.05;
        options=CovSel.ADMMOptions(;abstol=1e-6,reltol=1e-6))
    Zx64, Zy64 = CovSel.fusedGraphicalLasso(Sx, n, Sy, n, 0.05, 0.05;
        options=CovSel.ADMMOptions(;abstol=1e-6,reltol=1e-6))
    @test eltype(Zx32) == Float32
    @test Zx32 ≈ Zx64 atol=1e-3
    @test Zy32 ≈ Zy64 atol=1e-3
end


@testset "fused_neighborhood_selection" begin
  p = 5
  n = 1000
  X, Y = twoGroups(p, n)

  problem = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "sb" => "yes"))

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

  for i=1:10
    λ1 = 0.1 * rand()
    λ2 = 0.1 * rand()
    z1, z2 = CovSel.fusedNeighborhoodSelection(X[:,2:p], X[:,1], Y[:,2:p], Y[:,1], λ1, λ2;
                      options=CovSel.ADMMOptions(;abstol=1e-12,reltol=1e-12))

    @objective(problem, Min, (sum((X[:,1] - X[:,2:p]*xj1) .^ 2) + sum((Y[:,1] - Y[:,2:p]*xj2) .^ 2)) / (2. * n) + λ1 * (sum(t1)+sum(t2)) + λ2 * sum(t3))
    optimize!(problem)

    @test value.(xj1) ≈ z1 atol=3e-5
    @test value.(xj2) ≈ z2 atol=3e-5
  end

  # without penalties the solutions are the least squares fits
  z1, z2 = CovSel.fusedNeighborhoodSelection(X[:,2:p], X[:,1], Y[:,2:p], Y[:,1], 0.0, 0.0;
                    options=CovSel.ADMMOptions(;abstol=1e-12,reltol=1e-12))
  @test z1 ≈ X[:,2:p] \ X[:,1] atol=1e-6
  @test z2 ≈ Y[:,2:p] \ Y[:,1] atol=1e-6
end


end
