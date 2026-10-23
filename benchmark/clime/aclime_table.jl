# Replicate Table 1 of Cai, Liu and Zhou (2016, arXiv:1212.2882): the errors of
# ACLIME and CLIME in the spectral, matrix ℓ1 and Frobenius norms.
#
#   julia --project=benchmark/clime benchmark/clime/aclime_table.jl MODEL P [REPS] [OUT.csv]
#
# MODEL is band, ar1 or er. As in the paper: Σ = D Ω₁⁻¹ D with D diagonal with
# entries uniform on (1, 5); n = 200 training observations and 200 more for
# validation. ACLIME uses δ = 2 in step 1 and chooses δ of step 2 from j/50,
# j = 1, …, 100, by the likelihood on the validation sample; "ACLIMEu" is the
# same without the rule of step 1 for large variances. The paper chooses
# the penalty of CLIME "by cross validation" as in Cai, Liu and Luo (2011),
# without more detail; here it is chosen from 40 values between 1 and 0.03 by
# 5-fold cross validation on the training sample, with the trace loss
# Σᵢ ((S Ω)ᵢᵢ - 1)² ("CLIME") and with the likelihood loss ("CLIMElik"), both
# computed from the same fold paths.

using CovSel, LinearAlgebra, SparseArrays, Random, Statistics, Printf

function omega1(model, p, rng)
  if model == "band"
    Ω = Matrix(1.0I, p, p)
    for i in 1:p-1; Ω[i, i+1] = Ω[i+1, i] = 0.6; end
    for i in 1:p-2; Ω[i, i+2] = Ω[i+2, i] = 0.3; end
    return Ω
  elseif model == "ar1"
    return [0.6^abs(i - j) for i in 1:p, j in 1:p]
  elseif model == "er"
    Ω2 = zeros(p, p)
    for c in 1:p, r in c+1:p
      if rand(rng) < 0.05
        Ω2[r, c] = Ω2[c, r] = 0.4 + 0.4 * rand(rng)
      end
    end
    return Ω2 + (abs(eigmin(Symmetric(Ω2))) + 0.05) * I
  end
  throw(ArgumentError(model))
end

norms(E) = (opnorm(Matrix(E)), opnorm(Matrix(E), 1), norm(E))

function replication(model, p, rng; n=200, ntest=200)
  Ω1 = omega1(model, p, rng)
  D = Diagonal(1 .+ 4 .* rand(rng, p))
  Σ = Symmetric(D * inv(Ω1) * D)
  Ω = inv(Σ)
  U = cholesky(Σ).U
  X = randn(rng, n, p) * U
  Xtest = randn(rng, ntest, p) * U
  S, Stest = cov(X), cov(Xtest)
  # ACLIME, δ of step 2 by the validation likelihood, with the rule of step 1 for
  # large variances (truncate=true, as in the paper) and without it (the default)
  δs = (1:100) ./ 50
  path = redirect_stderr(devnull) do        # warnings about zero diagonal entries
    aclime(S, n; δ=δs, truncate=true)
  end
  lik = [CovSel._cvloss(Stest, Ωh, :likelihood) for Ωh in path]
  Ωa = path[argmin(lik)]
  pathu = aclime(S, n; δ=δs, truncate=false)
  liku = [CovSel._cvloss(Stest, Ωh, :likelihood) for Ωh in pathu]
  Ωu = pathu[argmin(liku)]
  # CLIME, penalty by 5-fold cross validation with the trace and likelihood losses
  λs = exp.(range(log(1.0), log(0.03); length=40))
  Ltr = zeros(length(λs))
  Llik = zeros(length(λs))
  for test in CovSel._folds(n, 5, rng)
    train = setdiff(1:n, test)
    fold = redirect_stderr(devnull) do      # warnings about infeasible penalties
      clime(cov(X[train, :]), λs)
    end
    St = cov(X[test, :])
    for (k, λ) in enumerate(λs)
      kk = findfirst(==(λ), fold.λ)
      Ltr[k] += kk === nothing ? Inf : CovSel._cvloss(St, fold[kk], :trace)
      Llik[k] += kk === nothing ? Inf : CovSel._cvloss(St, fold[kk], :likelihood)
    end
  end
  full = clime(S, λs[[argmin(Ltr), argmin(Llik)]])
  (norms(Ωa - Ω)..., norms(full[1] - Ω)..., norms(full[2] - Ω)..., norms(Ωu - Ω)...,
   δs[argmin(lik)], λs[argmin(Ltr)], λs[argmin(Llik)], δs[argmin(liku)])
end

function main(model, p, reps, out)
  rng = Xoshiro(20261023 + 1000 * p + hash(model) % 1000)
  R = [replication(model, p, rng) for _ in 1:reps]
  M = reduce(hcat, collect.(R))'
  m, s = vec(mean(M; dims=1)), vec(std(M; dims=1)) ./ sqrt(reps)
  names = ["ACLIME spectral", "ACLIME l1", "ACLIME Frobenius", "CLIME spectral", "CLIME l1",
           "CLIME Frobenius", "CLIMElik spectral", "CLIMElik l1", "CLIMElik Frobenius",
           "ACLIMEu spectral", "ACLIMEu l1", "ACLIMEu Frobenius",
           "ACLIME delta", "CLIME lambda", "CLIMElik lambda", "ACLIMEu delta"]
  for (k, nm) in enumerate(names)
    @printf("%-6s p=%-4d %-18s %.3f (%.3f)\n", model, p, nm, m[k], s[k])
  end
  if out !== nothing
    open(out, "w") do io
      println(io, "model,p,reps,", join(replace.(names, " " => "_"), ","))
      println(io, "$model,$p,$reps,", join((@sprintf("%.6f", x) for x in m), ","))
      println(io, "$model,$p,$reps,", join((@sprintf("%.6f", x) for x in s), ","))
    end
  end
end

main(ARGS[1], parse(Int, ARGS[2]), length(ARGS) > 2 ? parse(Int, ARGS[3]) : 100,
     length(ARGS) > 3 ? ARGS[4] : nothing)
