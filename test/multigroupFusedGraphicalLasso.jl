module MultigroupFusedGraphicalLassoTest

using Test, Random, LinearAlgebra
using CovarianceSelection: fusedGraphicalLasso, fusedGraphicalLasso!, covsel!, ADMMOptions, ADMMInfo
using ProximalBase: FiberSum, NormL1Fused, prox

const tight = ADMMOptions(; abstol=1e-12, reltol=1e-12, maxiter=20_000)

function residual(Z, S, w, λ1, λ2; penalize_diag=true)
    G = similar(Z)
    for k in axes(Z, 3)
        G[:, :, k] = w[k] * (S[:, :, k] - inv(Z[:, :, k]))
    end
    g = FiberSum(NormL1Fused(λ1, λ2);
                 diagonal=NormL1Fused(penalize_diag ? λ1 : 0.0, λ2))
    maximum(abs, prox(g, Z - 0.1G, 0.1) - Z)
end

function glasso(S, λ; penalize_diag=true)
    p = size(S, 1)
    covsel!(zeros(p, p), zeros(p, p), zeros(p, p), S, λ;
            penalize_diag, options=tight)
end

@testset "multi-group fused graphical lasso" begin
    rng = Xoshiro(42)
    p = 5
    for K in (3, 4)
        S = cat([let A=randn(rng, p, p); A'A / p + I end for _ in 1:K]...; dims=3)
        ns = collect(1:K) .* 100
        w = ns / sum(ns)
        S0 = copy(S)
        for pd in (true, false)
            info = ADMMInfo()
            Z = fusedGraphicalLasso(S, ns, 0.03, 0.04; penalize_diag=pd, options=tight, info)
            @test info.converged
            @test size(Z) == (p, p, K)
            @test all(issymmetric(Z[:, :, k]) && isposdef(Z[:, :, k]) for k in 1:K)
            @test residual(Z, S, w, 0.03, 0.04; penalize_diag=pd) < 1e-8

            # Without fusion each group solves a graphical lasso with λ1/wₖ.
            separate = fusedGraphicalLasso(S, 0.03, 0.0; weights=w, penalize_diag=pd, options=tight)
            for k in 1:K
                @test separate[:, :, k] ≈ glasso(S[:, :, k], 0.03 / w[k]; penalize_diag=pd) atol=1e-7
            end

            # With complete fusion, all K copies of the ℓ1 penalty remain.
            fused = fusedGraphicalLasso(S, ns, 0.03, 10.0; penalize_diag=pd, options=tight)
            pooled = glasso(sum(w[k] * S[:, :, k] for k in 1:K), K * 0.03; penalize_diag=pd)
            @test all(fused[:, :, k] ≈ pooled for k in 1:K)
        end
        @test S == S0

        Z = fusedGraphicalLasso(S, 0.03, 0.04; weights=w, options=tight)
        @test fusedGraphicalLasso([S[:, :, k] for k in 1:K], ns, 0.03, 0.04; options=tight) ≈ Z
        @test fusedGraphicalLasso(view(S, :, :, :), ns, 0.03, 0.04; options=tight) ≈ Z
        @test residual(fusedGraphicalLasso(S, 0.03, 0.04; options=tight),
                       S, fill(1/K, K), 0.03, 0.04) < 1e-8
        # Explicit weights need not sum to one: scaling the whole objective
        # leaves its minimizer unchanged.
        @test fusedGraphicalLasso(S, 0.09, 0.12; weights=3w, options=tight) ≈ Z
        # Group labels have no order in an all-pairs penalty.
        order = randperm(rng, K)
        @test fusedGraphicalLasso(S[:, :, order], 0.03, 0.04; weights=w[order], options=tight) ≈ Z[:, :, order]
    end

    # Independent closed-form solution with partial fusion: groups 1 and 3
    # fuse, while group 2 remains distinct. This also checks fusion on the
    # diagonal when the diagonal ℓ1 penalty is disabled.
    S = reshape([1.0, 4.0, 1.2], 1, 1, 3)
    w = [0.2, 0.5, 0.3]
    for pd in (true, false)
        λ1, λ2 = 0.01, 0.08
        l1 = pd ? λ1 : 0.0
        a = (w[1] + w[3]) / (w[1]*S[1] + w[3]*S[3] + 2l1 + 2λ2)
        b = w[2] / (w[2]*S[2] + l1 - 2λ2)
        # The subgradient for the fused pair is in [-1, 1], so these stationary
        # values satisfy the optimality conditions of the convex objective.
        @test a > b > 0
        @test abs((w[1]*(S[1] - inv(a)) + l1 + λ2) / λ2) < 1
        Z = fusedGraphicalLasso(S, λ1, λ2; weights=w, penalize_diag=pd, options=tight)
        @test vec(Z) ≈ [a, b, a] atol=1e-9
    end

    # One group has no fusion penalty; two groups agree with the established API.
    A, B = [1.0 0.2; 0.2 1.3], [1.2 -0.1; -0.1 0.8]
    for pd in (true, false)
        Z = fusedGraphicalLasso([A], 0.03, 100.0; weights=[0.4], penalize_diag=pd, options=tight)
        @test Z[:, :, 1] ≈ glasso(A, 0.03/0.4; penalize_diag=pd) atol=1e-8
        X, Y = fusedGraphicalLasso(A, 200, B, 500, 0.03, 0.04; penalize_diag=pd, options=tight)
        Z = fusedGraphicalLasso([A, B], [200, 500], 0.03, 0.04; penalize_diag=pd, options=tight)
        @test Z[:, :, 1] ≈ X atol=1e-8
        @test Z[:, :, 2] ≈ Y atol=1e-8
    end

    S = cat(A, B, (A+B)/2; dims=3)
    ns = [200, 500, 300]
    Z, U = zeros(2, 2, 3), zeros(2, 2, 3)
    info = ADMMInfo()
    @test fusedGraphicalLasso!(Z, U, S, ns, 0.03, 0.04; info) === Z
    @test info.converged && all(isfinite, (info.ρ, info.r_norm, info.s_norm))
    Z0 = copy(Z)
    info2 = ADMMInfo()
    fusedGraphicalLasso!(Z, U, S, ns, 0.03, 0.04; options=ADMMOptions(; ρ=info.ρ), info=info2)
    @test info2.converged && info2.iterations <= 2
    @test Z ≈ Z0 rtol=1e-3

    # Data rescaling preserves the iteration count and rescales Ω inversely.
    iterations = Int[]
    for c in (1e-3, 1.0, 1e3)
        info = ADMMInfo()
        Zc = fusedGraphicalLasso(c*S, ns, c*0.03, c*0.04; info)
        @test info.converged
        @test c*Zc ≈ Z0 rtol=1e-6
        push!(iterations, info.iterations)
    end
    @test all(==(iterations[1]), iterations)
    @test_logs (:warn, r"fusedGraphicalLasso! stopped after 2 iterations") fusedGraphicalLasso(
        S, ns, 0.03, 0.04; options=ADMMOptions(; maxiter=2))

    options = ADMMOptions(; abstol=1e-6, reltol=1e-6)
    Z32 = fusedGraphicalLasso(Float32.(S), ns, 0.03, 0.04; options)
    @test eltype(Z32) == Float32
    @test Z32 ≈ fusedGraphicalLasso(S, ns, 0.03, 0.04; options) atol=1e-4
    # In-place buffers determine the arithmetic precision, including when the
    # sample covariances have a different element type.
    @test fusedGraphicalLasso!(zeros(Float32, 2, 2, 3), zeros(Float32, 2, 2, 3),
                              S, ns, 0.03, 0.04; options) ≈ Z32 atol=1e-6

    @test_throws ArgumentError fusedGraphicalLasso(Matrix{Float64}[], 0.03, 0.04)
    @test_throws ArgumentError fusedGraphicalLasso(zeros(2, 2, 0), 0.03, 0.04)
    @test_throws ArgumentError fusedGraphicalLasso(zeros(0, 0, 3), 0.03, 0.04)
    @test_throws DimensionMismatch fusedGraphicalLasso(zeros(2, 3, 3), 0.03, 0.04)
    @test_throws DimensionMismatch fusedGraphicalLasso([A, zeros(3, 3)], 0.03, 0.04)
    @test_throws DimensionMismatch fusedGraphicalLasso!(zeros(2, 2, 2), U, S, 0.03, 0.04)
    @test_throws ArgumentError fusedGraphicalLasso!(Z, Z, S, 0.03, 0.04)
    @test_throws DimensionMismatch fusedGraphicalLasso(S, [200, 500], 0.03, 0.04)
    @test_throws ArgumentError fusedGraphicalLasso(S, [200, 0, 300], 0.03, 0.04)
    @test_throws ArgumentError fusedGraphicalLasso(S, [200, -1, 300], 0.03, 0.04)
    @test_throws DimensionMismatch fusedGraphicalLasso(S, 0.03, 0.04; weights=[0.5, 0.5])
    for v in (0.0, -1.0, Inf, NaN)
        @test_throws ArgumentError fusedGraphicalLasso(S, 0.03, 0.04; weights=[0.2, v, 0.3])
    end
    for v in (-1.0, Inf, NaN)
        @test_throws ArgumentError fusedGraphicalLasso(S, v, 0.04)
        @test_throws ArgumentError fusedGraphicalLasso(S, 0.03, v)
    end
    # Normalization must not overflow for large integer sample counts.
    @test fusedGraphicalLasso(S, fill(typemax(Int), 3), 0.03, 0.04; options=tight) ≈
          fusedGraphicalLasso(S, 0.03, 0.04; options=tight)
end

end
