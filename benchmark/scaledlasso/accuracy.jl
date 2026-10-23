# Accuracy of the scaled lasso estimator of Sun and Zhang (2013) against CLIME and
# ACLIME, for a range of replications of one design, model and dimension.
#
#   julia --project=benchmark/scaledlasso benchmark/scaledlasso/accuracy.jl DESIGN MODEL P FIRST LAST OUT.csv
#
# DESIGN is
# - sz: the models 1, 2 and 3 of Sun and Zhang (2013, Section 6), n = 100;
# - clz: the band, ar1 and er models of Cai, Liu and Zhou (2016), n = 200, with
#   the random draws of ../clime/aclime_table.jl (the data of replication r are
#   those of replication r there);
# - hadamard: MODEL = the block size k of hadamard_precision, n = 2000.
# A validation sample of the same size chooses the penalty of CLIME and the δ of
# ACLIME by the likelihood; the scaled lasso uses the training sample only. The
# data of the replications before FIRST are drawn and discarded, so a range of
# replications can run on its own. One row per method and replication is
# appended to OUT.csv as soon as the replication is done:
# design,model,p,rep,method,spectral,l1,frobenius,parameter,seconds
#
# Environment: LPMAX, the largest p for CLIME and ACLIME (default 300).

include(joinpath(@__DIR__, "common.jl"))

const LPMAX = parse(Int, get(ENV, "LPMAX", "300"))

function draw(design, model, p, rng)
  if design == "clz"
    n = 200
    Ω1 = omega1(model, p, rng)
    D = Diagonal(1 .+ 4 .* rand(rng, p))
    Σ = Symmetric(D * inv(Ω1) * D)
    U = cholesky(Σ).U
    X = randn(rng, n, p) * U
    Xval = randn(rng, n, p) * U
    CovSel._folds(n, 5, rng)          # as aclime_table.jl, to keep the draws in step
    return inv(Σ), cov(X), cov(Xval), n
  end
  Θ, n = design == "sz" ? (sz_precision(parse(Int, model), p, rng), 100) :
         design == "hadamard" ? (hadamard_precision(p, parse(Int, model)), 2000) :
         throw(ArgumentError("design $design"))
  U = cholesky(Symmetric(inv(Θ))).U
  X = randn(rng, n, p) * U
  Xval = randn(rng, n, p) * U
  Θ, X' * X / n, Xval' * Xval / n, n
end

rngseed(design, model, p) =
  design == "clz" ? 20261023 + 1000 * p + hash(model) % 1000 :
  design == "sz" ? 20131100 + 10000 * parse(Int, model) + p : 20131200 + parse(Int, model)

function replication(design, Θ, S, Sval, n)
  p = size(S, 1)
  rows = Tuple{String,NTuple{3,Float64},Float64,Float64}[]
  λ₀ = CovSel._penalty_level(:quantile, p, n)
  R, t = timed(() -> scaledlasso(S, n; symmetrize=false, ntasks=1))
  Ω, t2 = timed(() -> CovSel.symmetrize_minabs(R))
  push!(rows, ("SLasso", errors(Ω - Θ), λ₀, t + t2))
  Ω, t = timed(() -> scaledlasso(S, n; refit=true, ntasks=1))
  push!(rows, ("SLasso/LSE", errors(Ω - Θ), λ₀, t))
  p <= LPMAX || return rows
  λs = exp.(range(log(1.0), log(0.03); length=40))
  (Ω, λ), t = timed(() -> validated(clime(S, λs; ntasks=1), Sval))
  push!(rows, ("CLIME", errors(Ω - Θ), λ, t))
  # the grid of the replication; from 0.1 for larger p, where the chosen δ is larger
  δs = collect(((p <= 100 && design == "clz" ? 1 : 5):100) ./ 50)
  (Ω, δ), t = timed(() -> validated(aclime(S, n; δ=δs, ntasks=1), Sval))
  push!(rows, ("ACLIME", errors(Ω - Θ), δ, t))
  if design != "hadamard"
    (Ω, δ), t2 = timed(() -> validated(aclime_step2(S, n, diag(R), δs), Sval))
    push!(rows, ("ACLIME, scaled lasso pilot", errors(Ω - Θ), δ, t2))
  end
  rows
end

function main(design, model, p, first, last, out)
  rng = Xoshiro(rngseed(design, model, p))
  for _ in 1:first-1
    draw(design, model, p, rng)
  end
  open(out, "a") do io
    for r in first:last
      Θ, S, Sval, n = draw(design, model, p, rng)
      for (method, (e2, e1, eF), par, t) in replication(design, Θ, S, Sval, n)
        @printf(io, "%s,%s,%d,%d,%s,%.6g,%.6g,%.6g,%.6g,%.4g\n", design, model, p, r, method, e2, e1, eF, par, t)
      end
      flush(io)
    end
  end
end

main(ARGS[1], ARGS[2], parse(Int, ARGS[3]), parse(Int, ARGS[4]), parse(Int, ARGS[5]), ARGS[6])
