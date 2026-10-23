# Bias of the step-1 pilot as n grows: median and 10%/90% quantiles of ω̂ᵢᵢ/ωᵢᵢ
using CovSel, LinearAlgebra, Random, Statistics, Printf
include(joinpath(@__DIR__, "omega1.jl"))
p = 100
for model in ("band", "ar1", "er"), n in (200, 800, 3200, 12800)
  rng = Xoshiro(7 + n)
  Ω1 = omega1(model, p, rng)
  D = Diagonal(1 .+ 4 .* rand(rng, p))
  Σ = Symmetric(D * inv(Ω1) * D)
  ω = diag(inv(Σ))
  S = cov(randn(rng, n, p) * cholesky(Σ).U)
  σ = diag(S); rate = sqrt(log(p) / n); Ŝ = S + I / n
  h = 1 ./ sqrt.(σ)
  out = String[]
  for δ₁ in (0.5, 2.0)
    r = CovSel._solve_diagonal(ParametricSimplex(), Ŝ, CovSel._CLIMESpec(σ, true, δ₁ * rate), Threads.nthreads()) ./ ω
    rs = CovSel._solve_diagonal(ParametricSimplex(), h .* Ŝ .* h', CovSel._CLIMESpec(ones(p), true, δ₁ * rate), Threads.nthreads()) ./ σ ./ ω
    push!(out, @sprintf("δ₁=%.1f paper %.2f [%.2f,%.2f] std %.2f [%.2f,%.2f]", δ₁,
                        median(r), quantile(r, 0.1), quantile(r, 0.9), median(rs), quantile(rs, 0.1), quantile(rs, 0.9)))
  end
  @printf("%-5s n=%-6d λ₁(δ₁=2)=%.3f  %s   %s\n", model, n, 2rate, out...)
end
