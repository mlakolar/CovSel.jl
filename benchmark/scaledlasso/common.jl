# Designs, estimators and errors shared by the scaled lasso benchmarks.

using CovSel, LinearAlgebra, SparseArrays, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "clime", "problems.jl"))      # precision_matrix, sample, writebin
include(joinpath(@__DIR__, "..", "clime", "pilot", "omega1.jl"))  # omega1: the design of Cai, Liu and Zhou

# The three models of Sun and Zhang (2013, Section 6): 1, Θᵢⱼ = 0.6^|i-j|; 2, a
# random graph, B + δI with off-diagonal entries of B equal to 0.5 with
# probability 0.1, δ such that the condition number is p, scaled to a unit
# diagonal; 3, model 1 with the unequal diagonal D^(1/2) Θ D^(1/2),
# dᵢᵢ = (4i + p - 5)/(5(p - 1)).
function sz_precision(model::Int, p::Int, rng)
  ar = [0.6^abs(i - j) for i in 1:p, j in 1:p]
  model == 1 && return ar
  if model == 2
    B = zeros(p, p)
    for c in 1:p, r in c+1:p
      if rand(rng) < 0.1
        B[r, c] = B[c, r] = 0.5
      end
    end
    ev = eigvals(Symmetric(B))
    δ = (ev[end] - p * ev[1]) / (p - 1)
    return (B + δ * I) / δ
  elseif model == 3
    h = sqrt.([(4i + p - 5) / (5(p - 1)) for i in 1:p])
    return h .* ar .* h'
  end
  throw(ArgumentError("model $model"))
end

# Blocks I + (ρ/√k) H of size k on the diagonal, with H the Sylvester Hadamard
# matrix: eigenvalues 1 ± ρ, while the matrix ℓ1 norm 1 + ρ(k - 1)/√k grows like √k
function hadamard_precision(p::Int, k::Int; ρ=0.6)
  ispow2(k) && p % k == 0 || throw(ArgumentError("k must be a power of 2 that divides p"))
  H = ones(1, 1)
  while size(H, 1) < k
    H = [H H; H -H]
  end
  kron(Matrix(1.0I, p ÷ k, p ÷ k), I + (ρ / sqrt(k)) * H)
end

errors(E) = (opnorm(Matrix(E)), opnorm(Matrix(E), 1), norm(E))
likelihood(Sval, Ω) = CovSel._cvloss(Sval, Ω, :likelihood)

# the estimate of a path with the smallest validation loss, and its parameter
function validated(path, Sval)
  loss = [likelihood(Sval, Ω) for Ω in path]
  k = argmin(loss)
  path[k], path.λ[k]
end

const PS = ParametricSimplex()

# ACLIME's step 2 with the diagonal of a pilot estimate ω̂ in place of step 1
function aclime_step2(S, n, ω, δs)
  p = size(S, 1)
  spec = CovSel._CLIMESpec(sqrt.(diag(S)), false, 0.0)
  CovSel._estimate(PS, S + I / n, spec, δs .* sqrt(log(p) / n), sqrt.(Vector{Float64}(ω)), true, 1, false)
end

# Run f() and return its value and the elapsed seconds, warnings silenced
function timed(f)
  t = time_ns()
  v = redirect_stderr(f, devnull)
  v, (time_ns() - t) / 1e9
end
