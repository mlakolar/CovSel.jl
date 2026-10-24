module Danaher2014ExampleTest

using Test, Random, LinearAlgebra, Statistics
using CovarianceSelection: fusedGraphicalLasso, ADMMOptions
include("../examples/danaher2014/simulation.jl")
using .Danaher2014

@testset "Danaher 2014 simulation example" begin
    model = Danaher2014.population(Xoshiro(2014); p=50)
    again = Danaher2014.population(Xoshiro(2014); p=50)
    @test model.Σ == again.Σ && model.graph == again.graph
    @test [count(model.graph[:, :, k]) ÷ 2 for k in 1:3] == [40, 36, 32]
    @test model.Σ[1:40, 1:40, 1] == model.Σ[1:40, 1:40, 3]
    @test model.Σ[46:50, 46:50, 2] == Matrix{Float64}(I, 5, 5)
    @test model.Σ[41:50, 41:50, 3] == Matrix{Float64}(I, 10, 10)
    for k in 1:3
        @test diag(model.Σ[:, :, k]) ≈ ones(50)
        @test isposdef(Symmetric(model.Σ[:, :, k]))
        @test model.Σ[:, :, k] * model.Θ[:, :, k] ≈ I
    end
    # The printed 0.6 covariance transformation fills precision entries; keep
    # the actual truth and the generating skeleton separate.
    @test count(abs.(tril(model.Θ[:, :, 1], -1)) .> 1e-8) > count(model.graph[:, :, 1]) ÷ 2
    @test_throws ArgumentError Danaher2014.population(Xoshiro(1); p=51)

    data = Danaher2014.sample(Xoshiro(2), model, 40)
    @test data.S[:, :, 1] ≈ cov(data.Y[1]) * 39/40
    truth = Danaher2014.metrics(model.Θ, model)
    @test truth.kl ≈ 0 atol=1e-10
    @test truth.dkl_paper ≈ 75 atol=1e-10
    @test truth.sse == 0 && truth.fp == 0 && truth.fp_diff == 0

    # Independent two-variable metric arithmetic; undirected edges count once,
    # while the paper's off-diagonal SSE and L1 norm count both orientations.
    Ω = [1.0 0.2; 0.2 1.0]
    Σ = inv(Ω)
    tiny = (; Θ=cat(Ω, Ω, Ω; dims=3), Σ=cat(Σ, Σ, Σ; dims=3), graph=trues(2, 2, 3))
    Z = cat(Matrix{Float64}(I, 2, 2), Ω, Ω; dims=3)
    m = Danaher2014.metrics(Z, tiny)
    @test (m.edges, m.tp, m.fp, m.tp_diff, m.fp_diff) == (2, 2, 0, 0, 2)
    @test m.sse ≈ 0.08 && m.l1 ≈ 0.8
    @test m.kl ≈ (tr(Σ) - logdet(Σ) - 2)/2
    @test m.dkl_paper ≈ m.kl + 3

    # Verify the Section 7 penalty convention against the normalized API.
    options = ADMMOptions(; abstol=1e-10, reltol=1e-10)
    S = data.S[1:5, 1:5, :]
    A = fusedGraphicalLasso(S, 0.175, 0.025; weights=ones(3), penalize_diag=false, options)
    B = fusedGraphicalLasso(S, [150, 150, 150], 0.175/3, 0.025/3; penalize_diag=false, options)
    @test A ≈ B atol=1e-8
    @test Danaher2014.fixedpoint(A, S, 0.175, 0.025) < 1e-8

    quick = Danaher2014.configuration(String[]; quick=true)
    paper = Danaher2014.configuration(["--mode=paper"])
    @test (quick.p, quick.n, quick.reps) == (50, 40, 1)
    @test (paper.p, paper.n, paper.reps) == (500, 150, 100)
    @test paper.lambda2 == [0.0, 0.025, 0.05, 0.075, 0.1]
    @test_throws ErrorException Danaher2014.configuration(["--unknown=1"])
    @test_throws ErrorException Danaher2014.configuration(["--reps=0"])
    @test_throws ErrorException Danaher2014.configuration(["--lambda1=NaN"])

    base = (; method="FGL", lambda1=0.175, lambda2=0.025)
    row = merge(base, NamedTuple{Danaher2014.MEASURES}(ntuple(_ -> 1.0, length(Danaher2014.MEASURES))))
    row2 = merge(base, NamedTuple{Danaher2014.MEASURES}(ntuple(_ -> 3.0, length(Danaher2014.MEASURES))))
    summary = only(Danaher2014.summarize([row, row2]))
    @test summary.replicates == 2 && summary.sse_mean == 2.0 && summary.sse_se == 1.0
end

end
