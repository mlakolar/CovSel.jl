# The pilot of ACLIME step 1: how well does ω̂¹ estimate diag(Ω), for which δ₁,
# and how much does it matter for step 2? Design of Cai, Liu and Zhou (2016),
# same random draws as benchmark/clime/aclime_table.jl.
#
#   julia -t 1 --project=benchmark/clime benchmark/clime/pilot/pilot_shard.jl MODEL P REP OUT_raw.csv
#
# Replication REP alone (the data of replications 1, ..., REP-1 are drawn and
# discarded, so the draws are those of pilot.jl); writes its rows to OUT_raw.csv.
# Environment: STEP2 = pilots to run step 2 for, separated by ';' (default all;
# the others get NaN errors), DMIN = smallest δ of the grid, in units of 1/50
# (default 1). The path at larger δ does not depend on DMIN.

using CovSel, LinearAlgebra, SparseArrays, Random, Statistics, Printf
include(joinpath(@__DIR__, "omega1.jl"))

const PS = ParametricSimplex()
const NT = Threads.nthreads()
const STEP2 = filter(!isempty, split(get(ENV, "STEP2", ""), ';'))
const DMIN = parse(Int, get(ENV, "DMIN", "1"))
const D1 = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)

function pilots(S, n, Ω)
  p = size(S, 1)
  σ = diag(S)
  rate = sqrt(log(p) / n)
  Ŝ = S + I / n
  P = Pair{String,Vector{Float64}}[]
  for d in D1   # the paper's step 1, constraint scale σᵢᵢ ∨ σⱼⱼ
    push!(P, "paper $d" => CovSel._solve_diagonal(PS, Ŝ, CovSel._CLIMESpec(σ, true, d * rate), NT))
  end
  h = 1 ./ sqrt.(σ)
  R̂ = h .* Ŝ .* h'   # step 1 on the standardized data: scale √(σᵢᵢ σⱼⱼ)
  for d in D1
    push!(P, "std $d" => CovSel._solve_diagonal(PS, R̂, CovSel._CLIMESpec(ones(p), true, d * rate), NT) ./ σ)
  end
  # refit: residual variance of the least squares regression on the support of
  # the step-1 column, with the degrees of freedom of the support
  for d in (1.0, 2.0)
    push!(P, "paper $d refit" => refit(S, n, supports(Ŝ, CovSel._CLIMESpec(σ, true, d * rate))))
    push!(P, "std $d refit" => refit(S, n, supports(R̂, CovSel._CLIMESpec(ones(p), true, d * rate))))
  end
  push!(P, "inv(Ŝ)" => diag(inv(Symmetric(Ŝ))))   # δ₁ = 0
  p < n - 3 && push!(P, "inv(S) unbiased" => (n - p - 2) / (n - 1) * diag(inv(Symmetric(S))))
  push!(P, "1/σ" => 1 ./ σ)
  push!(P, "oracle" => diag(Ω))
  P
end

function supports(Ŝ, spec)
  p = size(Ŝ, 1)
  ws = CovSel._PSMWorkspace(Ŝ)
  map(1:p) do i
    CovSel._weights!(ws.c, spec, i)
    ws.γ = spec.γ
    col = CovSel._ColumnPath{Float64}(1)
    CovSel._column_path!(ws, i, [0.0], col, PS)
    col.rows[col.ptr[1]:col.ptr[2]-1]
  end
end

function refit(S, n, sup)
  map(axes(S, 1)) do i
    B = setdiff(sup[i], i)
    v = S[i, i] - (isempty(B) ? 0.0 : dot(S[B, i], Symmetric(S[B, B]) \ S[B, i]))
    (n - 1 - length(B)) / ((n - 1) * v)
  end
end

norms(E) = (opnorm(Matrix(E)), opnorm(Matrix(E), 1), norm(E))

function step2(S, n, ω, Stest, δs)
  p = size(S, 1)
  σ = diag(S)
  rate = sqrt(log(p) / n)
  spec = CovSel._CLIMESpec(sqrt.(σ), false, 0.0)
  path = CovSel._estimate(PS, S + I / n, spec, δs .* rate, sqrt.(ω), true, NT, false)
  lik = [CovSel._cvloss(Stest, Ωh, :likelihood) for Ωh in path]
  k = argmin(lik)
  path.Ω[k], δs[k], path.Ω[findfirst(==(2.0), δs)]
end

function draw(model, p, rng; n=200, ntest=200)
  Ω1 = omega1(model, p, rng)
  D = Diagonal(1 .+ 4 .* rand(rng, p))
  Σ = Symmetric(D * inv(Ω1) * D)
  Ω = inv(Σ)
  U = cholesky(Σ).U
  X = randn(rng, n, p) * U
  Xtest = randn(rng, ntest, p) * U
  CovSel._folds(n, 5, rng)          # as aclime_table.jl, to keep the draws in step
  Ω, cov(X), cov(Xtest), n
end

function replication(Ω, S, Stest, n)
  p = size(S, 1)
  δs = collect((DMIN:100) ./ 50)
  rows = []
  for (name, ω) in pilots(S, n, Ω)
    r = ω ./ diag(Ω)
    if !isempty(STEP2) && !(name in STEP2)
      push!(rows, (name, median(r), quantile(r, 0.1), quantile(r, 0.9), mean(abs.(log.(r))), std(log.(r)),
                   NaN, NaN, NaN, NaN, NaN, NaN, NaN))
      continue
    end
    Ωv, δv, Ω2 = step2(S, n, ω, Stest, δs)
    push!(rows, (name, median(r), quantile(r, 0.1), quantile(r, 0.9), mean(abs.(log.(r))), std(log.(r)),
                 norms(Ωv - Ω)..., δv, norms(Ω2 - Ω)...))
  end
  # ACLIME on the standardized data (both steps), scaled back: invariant to the
  # units of the variables
  h = 1 ./ sqrt.(diag(S))
  R = h .* S .* h'
  ωs = CovSel._solve_diagonal(PS, R + I / n, CovSel._CLIMESpec(ones(p), true, 2sqrt(log(p) / n)), NT) .* h .^ 2
  path = [Diagonal(h) * Θ * Diagonal(h) for Θ in aclime(R, n; δ=δs, ntasks=NT)]
  lik = [CovSel._cvloss(Stest, Ωh, :likelihood) for Ωh in path]
  k = argmin(lik)
  r = ωs ./ diag(Ω)
  push!(rows, ("standardized ACLIME", median(r), quantile(r, 0.1), quantile(r, 0.9), mean(abs.(log.(r))), std(log.(r)),
               norms(path[k] - Ω)..., δs[k], norms(path[findfirst(==(2.0), δs)] - Ω)...))
  rows
end

function main(model, p, rep, out)
  rng = Xoshiro(20261023 + 1000 * p + hash(model) % 1000)
  for _ in 1:rep-1
    draw(model, p, rng)
  end
  rows = replication(draw(model, p, rng)...)
  open(out, "w") do io
    for row in rows
      println(io, "$model,$p,$rep,$(row[1]),", join((@sprintf("%.6g", x) for x in row[2:end]), ","))
    end
  end
end

main(ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]), ARGS[4])
