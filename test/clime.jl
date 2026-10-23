module CLIMETest

using Test, Random, LinearAlgebra, SparseArrays, Statistics
using CovSel
using CovSel: symmetrize_minabs
using HiGHS

Random.seed!(11)

ar1(p, ρ) = [ρ^abs(i - j) for i in 1:p, j in 1:p]
samplecov(Σ, n) = cov(randn(n, size(Σ, 1)) * cholesky(Symmetric(Σ)).U)

# Optimality certificate of β for column i of the linear program
#     minimize ‖β‖₁ subject to |(Sβ - eᵢ)ⱼ| ≤ μ cⱼ:
# the multipliers ν of the tight constraints solve S[A, T] ν_T = -sign(β_A), and β is
# optimal if ν has the signs of the residuals, |Sν| ≤ 1 and the duality gap
# ‖β‖₁ + νᵢ + μ c'|ν| is zero. Returns the largest violation of these conditions.
function certificate(S, β, i, μ; c=ones(size(S, 1)), tol=1e-9)
  p = size(S, 1)
  r = S * β
  r[i] -= 1
  primal = maximum(abs.(r) .- μ .* c)
  A = findall(!iszero, β)
  T = findall(j -> abs(r[j]) >= μ * c[j] - tol * max(1, μ), 1:p)
  ν = zeros(p)
  ν[T] = S[A, T] \ (-sign.(β[A]))
  dual = maximum(abs, S * ν) - 1
  # at μ = 0 the constraints are equalities and the multipliers have any sign
  signs = μ > 0 ? maximum(-sign(r[j]) * ν[j] for j in T; init=0.0) : 0.0
  gap = abs(norm(β, 1) + ν[i] + μ * dot(c, abs.(ν))) / max(1, norm(β, 1))
  max(primal, dual, signs, gap)
end

l1cols(R) = [norm(R[:, i], 1) for i in 1:size(R, 2)]

@testset "clime" begin

  @testset "closed forms" begin
    # S = D diagonal: column i is (1 - λ)/Dᵢᵢ eᵢ for λ < 1, and zero for λ ≥ 1
    D = Diagonal([1.0, 2.0, 0.5, 4.0])
    for solver in (ParametricSimplex(), DualSimplex())
      @test clime(Matrix(D), 0.3, solver) ≈ 0.7 * inv(D)
      @test clime(Matrix(D), 0.0, solver) ≈ inv(D)
      @test iszero(clime(Matrix(D), 1.0, solver))
      @test iszero(clime(Matrix(D), 2.5, solver))
    end
    # 2 × 2, against the certificate, and the inverse at λ = 0
    S = [1.0 0.4; 0.4 1.0]
    for λ in (0.9, 0.5, 0.3, 0.1, 0.0)
      R = clime(S, λ; symmetrize=false)
      @test all(certificate(S, Vector(R[:, i]), i, λ) < 1e-10 for i in 1:2)
    end
    @test clime(S, 0.0) ≈ inv(S)
  end

  @testset "optimality: $name" for (name, S, λs) in (
      ("AR(1), n > p", samplecov(ar1(30, 0.6), 120), [0.6, 0.3, 0.2, 0.1, 0.05]),
      ("AR(1), n < p", samplecov(ar1(40, 0.6), 30), [0.6, 0.4, 0.3]),
      ("dense", samplecov(0.5 * ones(20, 20) + 0.5I, 60), [0.4, 0.2, 0.1]),
      ("unequal variances", samplecov(Diagonal(1:25) * ar1(25, 0.5) * Diagonal(1:25), 100), [0.5, 0.2]))
    p = size(S, 1)
    for solver in (ParametricSimplex(), DualSimplex())
      path = clime(S, λs, solver; symmetrize=false)
      @test path.λ == λs
      worst = maximum(certificate(S, Vector(path[k][:, i]), i, λs[k]) for k in eachindex(λs), i in 1:p)
      @test worst < 1e-8
    end
  end

  @testset "degenerate and dense problems" begin
    # ties in the ratio tests, singular blocks, and solutions with more nonzero
    # entries than the initial size of the basis block (32)
    X = randn(40, 12)
    for (name, S) in (("compound symmetry", 0.5 * ones(30, 30) + 0.5I),
                      ("identical blocks", kron(Matrix(1.0I, 8, 8), [1 0.6 0.3; 0.6 1 0.6; 0.3 0.6 1])),
                      ("duplicated variables", cov(hcat(X, X[:, 1:2]))),
                      ("rounded", round.(cov(randn(90, 30)); digits=1) + I))
      λs = [0.9, 0.6, 0.4, 0.25, 0.1]
      ref = clime(S, λs, LPSolver(HiGHS.Optimizer); symmetrize=false)
      for solver in (ParametricSimplex(), DualSimplex(), ParametricSimplex(; refactor=1), DualSimplex(; refactor=1))
        path = clime(S, λs, solver; symmetrize=false)
        @test path.λ == ref.λ
        for k in eachindex(path.λ)
          @test l1cols(path[k]) ≈ l1cols(ref[k]) rtol=1e-7
        end
      end
    end
    S = samplecov(ar1(40, 0.7), 400)
    Ω = clime(S, 0.0)                              # all 40 entries of every column
    @test Ω ≈ inv(S) rtol=1e-8
    @test clime(S, [0.05, 0.0], ParametricSimplex(; refactor=7))[2] ≈ Ω rtol=1e-8
    @test clime(S, 0.0, DualSimplex(; refactor=3)) ≈ Ω rtol=1e-8
  end

  @testset "Bland's rule" begin
    # the rule that the solvers switch to when they stall, used for every pivot
    function blandpath(S, i, μs; dual=false)
      ws = CovSel._PSMWorkspace(S)
      CovSel._reset!(ws, i)
      μ, tol, out = 1.0, 1e-9, Float64[]
      for μt in μs
        if dual
          # the dual simplex in Bland's mode from the previous basis
          while true
            kind, idx, s = CovSel._most_infeasible(ws, μt, tol, true)
            kind == 0 && break
            CovSel._pivot!(ws, kind, idx, s, tol, true) || return out
            CovSel._primal!(ws)
          end
        else
          while true
            μn, kind, idx, s = CovSel._next_event(ws, μ, tol, true)
            μn < μt && break
            μ = μn
            CovSel._pivot!(ws, kind, idx, s, tol, true) || return out
            CovSel._primal!(ws)
          end
        end
        push!(out, sum(q -> abs(ws.a[q] + μt * ws.b[q]), 1:ws.k; init=0.0))
      end
      out
    end
    for S in (samplecov(ar1(20, 0.6), 60), kron(Matrix(1.0I, 5, 5), [1 0.6 0.3; 0.6 1 0.6; 0.3 0.6 1]))
      μs = [0.6, 0.3, 0.15]
      ref = clime(S, μs; symmetrize=false)
      for i in axes(S, 1), dual in (false, true)
        @test blandpath(S, i, μs; dual=dual) ≈ [norm(ref[k][:, i], 1) for k in 1:3] rtol=1e-9
      end
    end
  end

  @testset "errors in the column problems" begin
    S = samplecov(ar1(30, 0.6), 100)
    for ntasks in (1, 2)   # an error in a task is rethrown as it is
      @test_throws ErrorException clime(S, 0.1, ParametricSimplex(; maxpivots=1); ntasks=ntasks)
      @test_throws ErrorException clime(S, 0.1, DualSimplex(; maxpivots=1); ntasks=ntasks)
    end
  end

  @testset "LPSolver" begin
    S = samplecov(ar1(25, 0.5), 60)
    λs = [0.5, 0.25, 0.1]
    exact = clime(S, λs, ParametricSimplex(); symmetrize=false)
    lp = clime(S, λs, LPSolver(HiGHS.Optimizer); symmetrize=false)
    for k in eachindex(λs)
      @test l1cols(lp[k]) ≈ l1cols(exact[k]) rtol=1e-7
      @test Matrix(lp[k]) ≈ Matrix(exact[k]) atol=1e-7
    end
    @test clime(S, 0.25, LPSolver(HiGHS.Optimizer; attributes=("presolve" => "off",))) ≈
      clime(S, 0.25) atol=1e-7
    # penalties of 1 and more give zero, without calling the solver
    @test iszero(clime(S, [1.5, 1.0], LPSolver(HiGHS.Optimizer))[1])
    # a status other than optimal or infeasible is an error
    @test_throws ErrorException clime(S, 0.1, LPSolver(HiGHS.Optimizer; attributes=("time_limit" => 0.0,)))
    # without MathOptInterface, LPSolver explains what to load (the extension
    # replaces these methods for the arguments that clime passes)
    @test_throws ErrorException CovSel._solve_columns(LPSolver(nothing), nothing)
    @test_throws ErrorException CovSel._solve_diagonal(LPSolver(nothing), nothing)
    # rank one: column 1 needs 1 - λ ≤ β₁ + β₂ ≤ λ, infeasible for λ < 1/2
    S = [1.0 1.0; 1.0 1.0]
    for solver in (LPSolver(HiGHS.Optimizer), ParametricSimplex(), DualSimplex())
      a = @test_logs (:warn,) clime(S, [0.75, 0.5, 0.25], solver)
      @test a.λ == [0.75, 0.5]
      @test_throws DomainError clime(S, 0.25, solver)
    end
  end

  @testset "PDHG" begin
    S = samplecov(ar1(30, 0.6), 120)
    λs = [0.5, 0.3, 0.15]
    exact = clime(S, λs; symmetrize=false)
    for (tol, ntasks) in ((1e-5, 1), (1e-8, 2))
      fo = clime(S, λs, PDHG(; tol=tol); symmetrize=false, ntasks=ntasks)
      @test fo.λ == λs
      for k in eachindex(λs)
        @test l1cols(fo[k]) ≈ l1cols(exact[k]) rtol=20tol
        @test maximum(abs.(S * fo[k] - I)) <= λs[k] * (1 + 20tol)
      end
    end
    @test clime(S, 0.3, PDHG(; tol=1e-9)) ≈ clime(S, 0.3) atol=1e-6
    # ACLIME: step 1 by the parametric simplex, step 2 by PDHG
    @test aclime(S, 120, PDHG(; tol=1e-9)) ≈ aclime(S, 120) atol=1e-6
    @test_logs (:warn, r"did not converge") clime(S, 0.15, PDHG(; maxiter=10))
  end

  @testset "paths" begin
    S = samplecov(ar1(30, 0.6), 100)
    λs = [0.1, 0.4, 0.2, 0.3]    # any order
    for solver in (ParametricSimplex(), DualSimplex())
      path = clime(S, λs, solver)
      @test path isa CLIMEPath{Float64}
      @test path.λ == λs
      @test length(path) == 4
      for (k, λ) in enumerate(λs)
        @test path[k] ≈ clime(S, λ, solver)
        @test issymmetric(path[k])
      end
      # smaller penalty, denser estimate
      @test issorted(nnz.(path[sortperm(λs; rev=true)]))
    end
    # the same estimates from any number of tasks
    @test clime(S, λs; ntasks=3) == clime(S, λs; ntasks=1)
    @test sprint(show, MIME"text/plain"(), clime(S, [0.4, 0.2])) isa String
    @test sprint(show, clime(S, [0.4, 0.2])) == "CLIMEPath(2 estimates, p = 30)"
  end

  @testset "symmetrization" begin
    R = sparse([1.0 0.2 -0.5; 0.3 2.0 0.0; -0.1 1.0 3.0])
    @test symmetrize_minabs(R) == [1.0 0.2 -0.1; 0.2 2.0 0.0; -0.1 0.0 3.0]
    @test symmetrize_minabs(Matrix(R)) == symmetrize_minabs(R)
    S = samplecov(ar1(20, 0.5), 50)
    raw = clime(S, 0.15; symmetrize=false)
    @test !issymmetric(raw)
    @test clime(S, 0.15) == symmetrize_minabs(raw)
  end

  @testset "infeasible penalties" begin
    S = samplecov(ar1(40, 0.5), 20)    # singular
    path = @test_logs (:warn, r"infeasible") clime(S, [0.5, 0.3, 1e-3, 1e-4])
    @test path.λ == [0.5, 0.3]
    @test_throws DomainError clime(S, 1e-3)
    @test_throws DomainError clime(S, 1e-3, DualSimplex())
  end

  @testset "input" begin
    S = samplecov(ar1(10, 0.5), 50)
    @test clime(Symmetric(S), 0.2) == clime(S, 0.2)
    # tiny asymmetry from roundoff is accepted, a real one is not
    S2 = copy(S)
    S2[1, 2] += 1e-14
    @test clime(S2, 0.2) ≈ clime(S, 0.2)
    S2[1, 2] += 0.1
    @test_throws ArgumentError clime(S2, 0.2)
    @test_throws DimensionMismatch clime(randn(3, 4), 0.2)
    @test_throws ArgumentError clime(S, -0.1)
    @test_throws ArgumentError clime(S, [0.2, NaN])
    @test_throws ArgumentError clime(S, Float64[])
    S3 = copy(S)
    S3[3, :] .= 0
    S3[:, 3] .= 0
    @test_throws ArgumentError clime(S3, 0.2)
    # Float32 in, Float32 out
    Ω32 = clime(Float32.(S), 0.2f0)
    @test eltype(Ω32) == Float32
    @test Ω32 ≈ clime(S, 0.2) rtol=1e-4
  end

  @testset "aclime" begin
    n, p = 80, 30
    d = 1 .+ 4 .* rand(p)                       # unequal variances, as in the paper
    Σ = Diagonal(d) * ar1(p, 0.5) * Diagonal(d)
    X = randn(n, p) * cholesky(Symmetric(Σ)).U
    S = cov(X)
    Ω = aclime(S, n)
    @test issymmetric(Ω)
    @test all(>(0), diag(Ω))
    # both steps against HiGHS
    @test aclime(S, n, LPSolver(HiGHS.Optimizer)) ≈ Ω atol=1e-7
    @test aclime(S, n, ParametricSimplex()) ≈ aclime(S, n, DualSimplex()) atol=1e-10
    # the column problems of step 2 with the diagonal of step 1
    σ = diag(S)
    rate = sqrt(log(p) / n)
    Ŝ = S + I / n
    spec = CovSel._CLIMESpec(σ, true, 2rate)
    ω = CovSel._solve_diagonal(ParametricSimplex(), Ŝ, spec, 1)
    raw = aclime(S, n; symmetrize=false)
    @test all(certificate(Ŝ, Vector(raw[:, i]), i, 2rate * sqrt(ω[i]); c=sqrt.(σ)) < 1e-8 for i in 1:p)
    # with the rule of step 1 for large variances
    @test any(σ .> 1 / rate)
    ωt = ifelse.(σ .> 1 / rate, rate, ω)
    rawt = aclime(S, n; truncate=true, symmetrize=false)
    @test all(certificate(Ŝ, Vector(rawt[:, i]), i, 2rate * sqrt(ωt[i]); c=sqrt.(σ)) < 1e-8 for i in 1:p)
    @test aclime(S, n, LPSolver(HiGHS.Optimizer); truncate=true) ≈ aclime(S, n; truncate=true) atol=1e-7
    # a path over δ
    path = aclime(S, n; δ=[2.0, 1.0, 0.5])
    @test path.λ == [2.0, 1.0, 0.5]
    @test path[1] ≈ Ω
    @test issorted(nnz.(path))
    @test_throws ArgumentError aclime(S, 0)
    @test_throws ArgumentError aclime(S, n; δ=-1)
    # large variances: the rule of step 1 can zero whole columns, with a warning;
    # without it (the default) the estimate scales with the data
    Ωbig = @test_logs (:warn, r"diagonal entries") aclime(1e4 * S, n; truncate=true)
    @test any(iszero, diag(Ωbig))
    Ωnt = @test_logs aclime(1e4 * S, n)
    @test all(>(0), diag(Ωnt))
    @test Ω ≈ 1e4 * Ωnt rtol=0.05
  end

  @testset "cross validation" begin
    n, p = 120, 20
    Ω0 = Matrix(SymTridiagonal(ones(p), fill(0.4, p - 1)))
    X = randn(n, p) / cholesky(Symmetric(Ω0)).U'
    λs = [0.5, 0.3, 0.2, 0.1]
    cv = climecv(X, λs; folds=4, rng=MersenneTwister(3))
    @test cv isa CLIMECV{Float64}
    @test cv.λ == λs && cv.folds == 4
    @test length(cv.meanloss) == length(cv.stdloss) == 4
    @test all(isfinite, cv.stdloss[isfinite.(cv.meanloss)])
    @test cv.best == argmin(cv.meanloss)
    @test bestestimate(cv) ≈ clime(cov(X), λs[cv.best])
    # the same folds from the same random number generator
    @test climecv(X, λs; folds=4, rng=MersenneTwister(3)).meanloss == cv.meanloss
    @test climecv(X, λs; folds=4, loss=:trace).meanloss != cv.meanloss
    @test_throws ArgumentError climecv(X, λs; loss=:other)
    @test_throws ArgumentError climecv(X, λs; folds=1)
    cva = aclimecv(X, [2.0, 1.0, 0.5]; folds=3)
    @test cva.λ == [2.0, 1.0, 0.5]
    @test bestestimate(cva) ≈ aclime(cov(X), n; δ=cva.λ[cva.best])
    @test sprint(show, MIME"text/plain"(), cv) isa String
    @test startswith(sprint(show, cv), "CLIMECV(4 penalties")
  end
end

end
