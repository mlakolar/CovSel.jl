# Is ACLIME invariant to rescaling single variables? Compares the estimates from
# S and from a S a, with a uniform on (1, 5), step by step.
using CovarianceSelection, LinearAlgebra, Random, Statistics, SparseArrays
include(joinpath(@__DIR__, "omega1.jl"))
rng = Xoshiro(1)
p, n = 50, 200
Ω1 = omega1("band", p, rng)
Σ = Symmetric(inv(Ω1))
X = randn(rng, n, p) * cholesky(Σ).U
S = cov(X)
a = 1 .+ 4 .* rand(rng, p)          # per-variable rescaling
Sa = a .* S .* a'
rel(A, B) = norm(Matrix(A - B)) / norm(Matrix(B))
for (name, f) in (("step 1 pilot", (S, n) -> CovarianceSelection._solve_diagonal(ParametricSimplex(), S + I / n,
                     CovarianceSelection._CLIMESpec(diag(S), true, 2sqrt(log(p) / n)), 1)),)
  ω, ωa = f(S, n), f(Sa, n)
  println(name, ": max |a² ω̂(aSa) / ω̂(S) - 1| = ", maximum(abs.(a .^ 2 .* ωa ./ ω .- 1)))
end
Ω, Ωa = aclime(S, n), aclime(Sa, n)
println("aclime, truncate=false: ‖a Ω̂(aSa) a - Ω̂(S)‖/‖Ω̂(S)‖ = ", rel(a .* Ωa .* a', Ω))
Ωc = aclime(2.0 .* S, n)
println("aclime, common factor 2: ", rel(2 .* Ωc, Ω))
# step 2 alone, with the pilot of S carried over exactly
function step2(S, n, ω)
  rate = sqrt(log(p) / n)
  spec = CovarianceSelection._CLIMESpec(sqrt.(diag(S)), false, 0.0)
  only(CovarianceSelection._estimate(ParametricSimplex(), S + I / n, spec, [2rate], sqrt.(ω), true, 1, true))
end
ω = CovarianceSelection._solve_diagonal(ParametricSimplex(), S + I / n, CovarianceSelection._CLIMESpec(diag(S), true, 2sqrt(log(p) / n)), 1)
println("step 2 alone (ℓ1 objective not weighted): ", rel(a .* step2(Sa, n, ω ./ a .^ 2) .* a', step2(S, n, ω)))
# step 1 on the standardized data
h = 1 ./ sqrt.(diag(S)); ha = 1 ./ sqrt.(diag(Sa))
ωs = CovarianceSelection._solve_diagonal(ParametricSimplex(), h .* (S + I / n) .* h', CovarianceSelection._CLIMESpec(ones(p), true, 2sqrt(log(p) / n)), 1) .* h .^ 2
ωsa = CovarianceSelection._solve_diagonal(ParametricSimplex(), ha .* (Sa + I / n) .* ha', CovarianceSelection._CLIMESpec(ones(p), true, 2sqrt(log(p) / n)), 1) .* ha .^ 2
println("standardized step 1: max |a² ω̂(aSa) / ω̂(S) - 1| = ", maximum(abs.(a .^ 2 .* ωsa ./ ωs .- 1)))
