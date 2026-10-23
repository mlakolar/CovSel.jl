# The pilot of ACLIME step 1: how well does ω̂¹ estimate diag(Ω), for which δ₁,
# and how much does it matter for step 2? Design of Cai, Liu and Zhou (2016),
# same random draws as benchmark/clime/aclime_table.jl.
#
#   julia -t 1 --project=benchmark/clime benchmark/clime/pilot/pilot.jl MODEL P REPS OUT.csv
#
# Writes the means and standard errors to OUT.csv and every replication to
# OUT_raw.csv (for paired comparisons).

using CovSel, LinearAlgebra, SparseArrays, Random, Statistics, Printf
include(joinpath(@__DIR__, "omega1.jl"))

const PS = ParametricSimplex()
const NT = Threads.nthreads()
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

function replication(model, p, rng; n=200, ntest=200)
  Ω1 = omega1(model, p, rng)
  D = Diagonal(1 .+ 4 .* rand(rng, p))
  Σ = Symmetric(D * inv(Ω1) * D)
  Ω = inv(Σ)
  U = cholesky(Σ).U
  X = randn(rng, n, p) * U
  Xtest = randn(rng, ntest, p) * U
  CovSel._folds(n, 5, rng)          # as aclime_table.jl, to keep the draws in step
  S, Stest = cov(X), cov(Xtest)
  δs = collect((1:100) ./ 50)
  rows = []
  for (name, ω) in pilots(S, n, Ω)
    r = ω ./ diag(Ω)
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

function main(model, p, reps, out)
  rng = Xoshiro(20261023 + 1000 * p + hash(model) % 1000)
  R = [replication(model, p, rng) for _ in 1:reps]
  header = "med_r,q10_r,q90_r,mabslog_r,sdlog_r,spec,l1,frob,delta,spec2,l12,frob2"
  open(replace(out, r"\.csv$" => "_raw.csv"), "w") do io
    println(io, "model,p,rep,pilot,", header)
    for r in 1:reps, row in R[r]
      println(io, "$model,$p,$r,$(row[1]),", join((@sprintf("%.6g", x) for x in row[2:end]), ","))
    end
  end
  open(out, "w") do io
    println(io, "model,p,reps,pilot,stat,", header)
    for j in eachindex(R[1])
      M = [R[r][j][k] for r in 1:reps, k in 2:13]
      m, s = vec(mean(M; dims=1)), vec(std(M; dims=1)) ./ sqrt(reps)
      println(io, "$model,$p,$reps,$(R[1][j][1]),mean,", join((@sprintf("%.5f", x) for x in m), ","))
      println(io, "$model,$p,$reps,$(R[1][j][1]),se,", join((@sprintf("%.5f", x) for x in s), ","))
      @printf("%-5s p=%-4d %-16s r: med %.2f [%.2f, %.2f] |log| %.2f sd %.2f  validated: %.2f %.2f %.2f δ=%.2f   δ=2: %.2f\n",
              model, p, R[1][j][1], m[1:10]...)
    end
  end
end

main(ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]), ARGS[4])
