# Optimality checks shared by the solver tests. A point is a minimizer of a
# smooth loss f plus a penalty g if and only if it is a fixed point of the
# proximal gradient step x ↦ prox_{tg}(x - t∇f(x)) for any t > 0.

using LinearAlgebra
using ProximalBase: shrink, prox, NormL1Fused

# soft thresholding of all entries (or only the off-diagonal ones)
function softThreshold(X, τ, penalize_diag)
  [(r != c || penalize_diag) ? shrink(X[r, c], τ) : X[r, c] for r in axes(X, 1), c in axes(X, 2)]
end

# graphical lasso: f(Ω) = tr(SΩ) - log det Ω, g(Ω) = λ‖Ω‖₁
function covselFixedPointResidual(Ω, S, λ; t=0.1, penalize_diag=true)
  G = S - inv(Ω)
  norm(softThreshold(Ω - t * G, t * λ, penalize_diag) - Ω, Inf)
end

# fused penalty: g(x, y) = λ1 (|x| + |y|) + λ2 |x - y| entrywise, with the
# λ1 part optionally removed from the diagonal
function fusedFixedPointResidual(X, Y, GX, GY, λ1, λ2; t=0.1, penalize_diag=true)
  r = zero(eltype(X))
  for i in CartesianIndices(X)
    l1 = (penalize_diag || ndims(X) == 1 || i[1] != i[2]) ? λ1 : zero(λ1)
    u, v = prox(NormL1Fused(l1, λ2), (X[i] - t * GX[i], Y[i] - t * GY[i]), t)
    r = max(r, abs(u - X[i]), abs(v - Y[i]))
  end
  r
end

randomSymmetric(p) = (E = randn(p, p); (E + E') / 2)

# objective does not decrease under random small symmetric perturbations
function isLocalMin(obj, X...; ε=1e-4, reps=20)
  f0 = obj(X...)
  all(1:reps) do _
    Xp = map(x -> x + ε * (ndims(x) == 2 ? randomSymmetric(size(x, 1)) : randn(length(x))), X)
    obj(Xp...) >= f0 - 1e-12
  end
end
