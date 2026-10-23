# The precision matrix estimator of Sun and Zhang (2013): the scaled lasso of each
# variable on the others, then the columns symmetrized.
#
# Column j is the scaled lasso
#
#     minimize (β - eⱼ)'S(β - eⱼ) / (2σ) + σ/2 + λ₀ ∑ₖ wₖ |βₖ|  over β with βⱼ = 0 and σ > 0,
#
# with wₖ = √Sₖₖ, and gives Θ̃ⱼⱼ = 1/σ̂² and Θ̃ₖⱼ = -β̂ₖ/σ̂². For fixed σ it is the lasso
# with penalty λ = λ₀σ, and σ̂² is the residual variance RSS(λ) = (β̂ - eⱼ)'S(β̂ - eⱼ)
# of the lasso at λ = λ₀σ̂.
#
# The lasso solution is piecewise linear in λ. On a segment with active set A and
# signs s, the optimality conditions S[A, A] β_A = S[A, j] - λ (w ∘ s)_A give
#
#     β_A = a + λ d,  a = N S[A, j],  d = -N (w ∘ s)_A,  N = S[A, A]⁻¹,
#
# and the correlations of all variables with the residual are c = S[:, j] - Sβ =
# u - λ v, with u = S[:, j] - S[:, A] a and v = S[:, A] d. On the segment the
# residual variance is RSS(λ) = r₀ + r₂ λ², with r₀ = Sⱼⱼ - S[j, A] a, the residual
# variance of least squares on A, and r₂ = (w ∘ s)_A' N (w ∘ s)_A: the terms linear
# in λ cancel. The condition λ² = λ₀² RSS(λ) then has the root
# λ* = √(r₀ / (1/λ₀² - r₂)). The scaled lasso is jointly convex, so
# 1 - λ₀² RSS(λ)/λ² is nondecreasing in λ, and the solution is on the first
# segment, from the top of the path, that contains its root.
#
# The path is followed by the homotopy method (Osborne, Presnell and Turlach,
# 2000; Efron et al., 2004): a variable joins A when its correlation reaches
# ±λwₖ, and leaves when its coefficient reaches zero. N is updated in O(k²)
# operations at each event, and the rest of an event costs O(pk), two products
# of k columns of S with a vector, so a column with k events costs O(p k²).

# The penalty level of Sun and Zhang (2013, Section 6): λ₀ = √2 Lₙ(k/p) with
# Lₙ(t) = Φ⁻¹(1 - t)/√n = √(2/n) erfc⁻¹(2t) and k the root of
# k = L₁(k/p)⁴ + 2 L₁(k/p)², whose left side minus right side increases from -∞
# to p/2 on (0, p/2] (L₁(1/2) = 0)
function _quantile_penalty(p::Int, n::Int)
  L(k) = sqrt(2) * erfcinv(2k / p)
  lo, hi = 0.0, p / 2
  for _ in 1:200
    k = (lo + hi) / 2
    l = L(k)
    k - l^4 - 2l^2 < 0 ? (lo = k) : (hi = k)
    hi - lo <= 4eps(hi) && break
  end
  sqrt(2 / n) * L(hi)
end

function _penalty_level(λ₀, p::Int, n::Int)
  if λ₀ isa Real
    isfinite(λ₀) && λ₀ > 0 || throw(ArgumentError("λ₀ must be positive and finite"))
    float(λ₀)
  elseif λ₀ === :quantile
    _quantile_penalty(p, n)
  elseif λ₀ === :universal
    sqrt(2 * log(max(p, 2)) / n)
  else
    throw(ArgumentError("λ₀ must be a positive number, :quantile or :universal"))
  end
end

mutable struct _SLWorkspace{T<:AbstractFloat}
  S::Matrix{T}
  p::Int
  w::Vector{T}           # √diag(S), the penalty weights
  j::Int                 # current column
  k::Int                 # number of active variables
  cap::Int               # size of N
  A::Vector{Int}         # active variables, in the order of the rows of N
  s::Vector{T}           # their signs
  posA::Vector{Int}      # posA[l]: position of l in A, or 0
  N::Matrix{T}           # N[1:k, 1:k] = inv(S[A, A])
  a::Vector{T}           # β_A = a + λ d
  d::Vector{T}
  x::Vector{T}           # S[A, j]
  y::Vector{T}           # (w ∘ s)_A
  g::Vector{T}           # scratch: a column of S on A, then N times it
  u::Vector{T}           # c = u - λ v, the correlations with the residual
  v::Vector{T}
  nupdates::Int          # updates of N since the last factorization
  drift0::T              # _sl_drift right after the last factorization
end

function _SLWorkspace(S::Matrix{T}) where {T<:AbstractFloat}
  p = size(S, 1)
  cap = min(p, 32)
  _SLWorkspace{T}(S, p, sqrt.(diag(S)), 0, 0, cap, zeros(Int, p), zeros(T, p), zeros(Int, p),
    zeros(T, cap, cap), zeros(T, p), zeros(T, p), zeros(T, p), zeros(T, p), zeros(T, p),
    zeros(T, p), zeros(T, p), 0, zero(T))
end

function _sl_reset!(ws::_SLWorkspace{T}, j::Int) where {T}
  @inbounds for q in 1:ws.k
    ws.posA[ws.A[q]] = 0
  end
  ws.j, ws.k, ws.nupdates, ws.drift0 = j, 0, 0, zero(T)
  ws
end

function _sl_grow!(ws::_SLWorkspace{T}, k::Int) where {T}
  k <= ws.cap && return ws
  cap = min(ws.p, max(k, 2 * ws.cap))
  N = zeros(T, cap, cap)
  @inbounds for t in 1:ws.k, q in 1:ws.k
    N[q, t] = ws.N[q, t]
  end
  ws.N, ws.cap = N, cap
  ws
end

# a, d, u and v for the current A and s
function _sl_direction!(ws::_SLWorkspace{T}) where {T}
  k, S, j = ws.k, ws.S, ws.j
  @inbounds for q in 1:k
    l = ws.A[q]
    ws.x[q] = S[l, j]
    ws.y[q] = ws.w[l] * ws.s[q]
  end
  _mulN!(ws.a, ws.N, ws.x, k)
  _mulN!(ws.d, ws.N, ws.y, k)
  @inbounds for q in 1:k
    ws.d[q] = -ws.d[q]
  end
  _combine!(ws.u, S, ws.A, ws.a, k)
  @inbounds @simd for l in 1:ws.p
    ws.u[l] = S[l, j] - ws.u[l]
  end
  _combine!(ws.v, S, ws.A, ws.d, k)
  ws
end

# How far the active variables are from satisfying the optimality conditions
# c = λ w ∘ s for all λ, that is, u = 0 and v = -w ∘ s on A, which rounding
# errors in N, accumulated over the updates, break
function _sl_drift(ws::_SLWorkspace{T}) where {T}
  wj = ws.w[ws.j]
  d = zero(T)
  @inbounds for q in 1:ws.k
    l = ws.A[q]
    d = max(d, abs(ws.u[l]) / (ws.w[l] * wj), abs(ws.v[l] / ws.w[l] + ws.s[q]))
  end
  d
end

# N = inv(S[A, A]) from scratch, in place, by LAPACK; false if S[A, A] is not
# positive definite
function _sl_factorize!(ws::_SLWorkspace{T}) where {T}
  k, N, S = ws.k, ws.N, ws.S
  ws.nupdates = 0
  k == 0 && return true
  @inbounds for q in 1:k, t in 1:q
    N[t, q] = S[ws.A[t], ws.A[q]]
  end
  M = view(N, 1:k, 1:k)
  _, info = LAPACK.potrf!('U', M)
  info == 0 || return false
  LAPACK.potri!('U', M)
  @inbounds for q in 1:k, t in q+1:k
    N[t, q] = N[q, t]
  end
  true
end

# Refactorize after `refactor` updates, or earlier when the updates have drifted
# by more than tol and by more than 10 times what a factorization achieves
function _sl_refactor_if_needed!(ws::_SLWorkspace{T}, refactor::Int, tol::T) where {T}
  ws.nupdates > 0 || return ws
  if ws.nupdates >= refactor || _sl_drift(ws) > max(tol, 10 * ws.drift0)
    _sl_factorize!(ws) || _sl_singular(ws)
    _sl_direction!(ws)
    ws.drift0 = _sl_drift(ws)
  end
  ws
end

_sl_singular(ws::_SLWorkspace) =
  throw(DomainError(ws.j, "the scaled lasso of column $(ws.j) needs more than $(ws.k) " *
    "variables, and S restricted to them is singular: λ₀ is too small for this S (or n), " *
    "or some variables are exactly collinear"))

# l joins A with sign s: N grows by a row and a column, in O(k²)
function _sl_join!(ws::_SLWorkspace{T}, l::Int, s::T) where {T}
  k, S = ws.k, ws.S
  _sl_grow!(ws, k + 1)
  N, g, b = ws.N, ws.g, ws.x
  @inbounds for q in 1:k
    g[q] = S[ws.A[q], l]
  end
  _mulN!(b, N, g, k)
  γ = S[l, l]
  @inbounds for q in 1:k
    γ -= g[q] * b[q]
  end
  γ > sqrt(eps(T)) * S[l, l] || _sl_singular(ws)
  @inbounds for t in 1:k
    bt = b[t] / γ
    @simd for q in 1:k
      N[q, t] += b[q] * bt
    end
    N[k+1, t] = N[t, k+1] = -bt
  end
  @inbounds N[k+1, k+1] = 1 / γ
  ws.k = k + 1
  ws.A[k+1], ws.s[k+1], ws.posA[l] = l, s, k + 1
  ws.nupdates += 1
  ws
end

# the variable in position q leaves A: it is moved to the last position, and
# N shrinks by a row and a column, in O(k²)
function _sl_leave!(ws::_SLWorkspace{T}, q::Int) where {T}
  k, N = ws.k, ws.N
  if q != k
    lq, lk = ws.A[q], ws.A[k]
    ws.A[q], ws.A[k] = lk, lq
    ws.s[q], ws.s[k] = ws.s[k], ws.s[q]
    ws.posA[lk], ws.posA[lq] = q, k
    @inbounds for t in 1:k
      N[t, q], N[t, k] = N[t, k], N[t, q]
    end
    @inbounds for t in 1:k
      N[q, t], N[k, t] = N[k, t], N[q, t]
    end
  end
  ws.posA[ws.A[k]] = 0
  @inbounds begin
    nkk = N[k, k]
    for t in 1:k-1
      c = N[k, t] / nkk
      @simd for r in 1:k-1
        N[r, t] -= N[r, k] * c
      end
    end
  end
  ws.k = k - 1
  ws.nupdates += 1
  ws
end

# The largest λ' ≤ λ at which an active coefficient reaches zero (kind 1, its
# position in A) or the correlation of an inactive variable reaches ±λ'w (kind 2,
# the variable, and the sign it joins with). A coefficient must reach zero
# strictly below λ, while a variable can join at λ itself, up to rounding: with
# ties, as for the two neighbours of a variable of an AR(1) covariance, several
# variables reach the bound at the same λ, and join one at a time. The variable
# that just joined cannot leave, nor the one that just left join, at the same λ.
function _sl_next_event(ws::_SLWorkspace{T}, λ::T, joined::Int, left::Int) where {T}
  below, above = λ * (1 - 16eps(T)), λ * (1 + 16eps(T))
  best, kind, idx, sgn = zero(T), 0, 0, zero(T)
  @inbounds for q in 1:ws.k
    ws.A[q] == joined && continue
    dq = ws.d[q]
    iszero(dq) && continue
    t = -ws.a[q] / dq
    if best < t < below
      best, kind, idx = t, 1, q
    end
  end
  @inbounds for l in 1:ws.p
    (l == ws.j || ws.posA[l] != 0 || l == left) && continue
    ul, vl, wl = ws.u[l], ws.v[l], ws.w[l]
    t = ul / (vl + wl)              # c = +λ'w
    if 0 < t < above && min(t, λ) > best
      best, kind, idx, sgn = min(t, λ), 2, l, one(T)
    end
    t = ul / (vl - wl)              # c = -λ'w
    if 0 < t < above && min(t, λ) > best
      best, kind, idx, sgn = min(t, λ), 2, l, -one(T)
    end
  end
  best, kind, idx, sgn
end

# Solve column j and record column j of Θ̃ in col. With refit, the coefficients
# and the noise level are those of least squares on the support of the scaled
# lasso (Sun and Zhang, 2013, Section 5).
function _sl_column!(ws::_SLWorkspace{T}, j::Int, λ₀::T, refit::Bool, col::_ColumnPath{T};
                     refactor::Int=100, tol::T=T(1e-9)) where {T}
  _sl_reset!(ws, j)
  S, w, p = ws.S, ws.w, ws.p
  # β = 0 for λ ≥ λmax, where RSS = Sⱼⱼ
  λ, l0, s0 = zero(T), 0, zero(T)
  @inbounds for l in 1:p
    l == j && continue
    t = abs(S[l, j]) / w[l]
    if t > λ
      λ, l0, s0 = t, l, sign(S[l, j])
    end
  end
  if λ₀ * w[j] >= λ
    return _sl_record!(col, ws, zero(T), S[j, j], refit)
  end
  _sl_join!(ws, l0, s0)
  joined, left = l0, 0
  for _ in 1:(1000 + 50p)
    _sl_direction!(ws)
    _sl_refactor_if_needed!(ws, refactor, tol)
    r0, r2 = S[j, j], zero(T)
    @inbounds for q in 1:ws.k
      r0 -= ws.x[q] * ws.a[q]
      r2 -= ws.y[q] * ws.d[q]
    end
    r0 = max(r0, zero(T))
    den = 1 / λ₀^2 - r2
    root = den > 0 ? min(sqrt(r0 / den), λ) : λ
    λnext, kind, idx, sgn = _sl_next_event(ws, λ, joined, left)
    if root >= λnext
      root > 0 || throw(DomainError(j, "the scaled lasso of column $j fits variable $j " *
        "exactly (σ̂ = 0): λ₀ is too small for this S (or n)"))
      return _sl_record!(col, ws, root, refit ? r0 : (root / λ₀)^2, refit)
    end
    if kind == 1
      left, joined = ws.A[idx], 0
      _sl_leave!(ws, idx)
    else
      _sl_join!(ws, idx, sgn)
      joined, left = idx, 0
    end
    λ = λnext
  end
  error("the scaled lasso of column $j did not reach its solution in $(1000 + 50p) steps")
end

# column j of Θ̃: Θ̃ⱼⱼ = 1/σ², Θ̃ₖⱼ = -βₖ/σ², with β_A = a + λ d (or a, refitted)
function _sl_record!(col::_ColumnPath{T}, ws::_SLWorkspace{T}, λ::T, σ2::T, refit::Bool) where {T}
  k, j = ws.k, ws.j
  rows, vals = ws.A, ws.g
  @inbounds for q in 1:k
    vals[q] = -(refit ? ws.a[q] : ws.a[q] + λ * ws.d[q]) / σ2
  end
  # j after the active variables (A has room: j is never active)
  @inbounds rows[k+1], vals[k+1] = j, 1 / σ2
  _pushsorted!(col, rows, vals, k + 1)
  col
end

"""
    scaledlasso(S, n; λ₀=:quantile, refit=false, symmetrize=true, ntasks=Threads.nthreads())

Estimate of the precision matrix `Ω = Σ⁻¹` by the scaled lasso of Sun and Zhang
(2013), from the sample covariance matrix `S` of `n` observations. Column `j` is
the scaled lasso of variable `j` on the others, which estimates the regression
coefficients `β` and the residual standard deviation `σ` jointly:

    minimize (β - eⱼ)'S(β - eⱼ) / (2σ) + σ/2 + λ₀ ∑ₖ √Sₖₖ |βₖ|  over βⱼ = 0 and σ > 0,

and gives `Ωⱼⱼ = 1/σ̂²` and `Ωₖⱼ = -β̂ₖ/σ̂²`. The columns are then symmetrized as in
[`clime`](@ref). The penalty of each column scales with its estimated `σ̂`, so
the method needs no tuning: `λ₀` is a universal constant, and the estimate is
equivariant to rescaling the variables (the estimate from `D S D` is
`D⁻¹ Ω̂ D⁻¹` for any positive diagonal `D`).

- `λ₀`: `:quantile`, the penalty level of the simulations of Sun and Zhang
  (2013), `√2 Lₙ(k/p)` with `Lₙ(t) = Φ⁻¹(1 - t)/√n` and `k` the root of
  `k = L₁(k/p)⁴ + 2 L₁(k/p)²`, about `√(2 log(p/k) / n)`; `:universal`,
  `√(2 log p / n)`; or a number.
- `refit`: if true, the coefficients and `σ̂` of each column are those of least
  squares on the variables selected by the scaled lasso (Sun and Zhang, 2013,
  Section 5).
- `symmetrize`: if false, returns the column estimates as they are.
- `ntasks`: number of tasks that solve the columns in parallel.

Each column follows the lasso path by the homotopy method until the penalty
`λ₀σ̂` of the scaled lasso, which it finds exactly; a column with `k` events on
its path costs `O(p k²)` operations. When `p > n` and `λ₀` is small, the path
can reach a singular block of `S` before the solution, and `scaledlasso` throws a
`DomainError`.

Returns a `SparseMatrixCSC`. Sun and Zhang (2013) prove that the error in the
spectral norm is of order `d √(log p / n)`, for `d` nonzero entries per column,
when the spectral norm of the inverse correlation matrix and the diagonal of `Ω`
are bounded; the bounds for CLIME and ACLIME need the matrix ℓ1 norm of `Ω`
bounded instead.

T. Sun and C.-H. Zhang (2013). Sparse matrix inversion with scaled lasso.
Journal of Machine Learning Research 14, 3385–3418.
"""
function scaledlasso(S::AbstractMatrix, n::Integer; λ₀::Union{Real,Symbol}=:quantile,
                     refit::Bool=false, symmetrize::Bool=true,
                     ntasks::Integer=Threads.nthreads())
  n > 0 || throw(ArgumentError("the number of observations n must be positive"))
  Σ = _covariance(S)
  T = eltype(Σ)
  p = size(Σ, 1)
  λ = T(_penalty_level(λ₀, p, n))
  cols = [_ColumnPath{T}(1) for _ in 1:p]
  _foreach_column(() -> _SLWorkspace(Σ), p, ntasks) do ws, j
    _sl_column!(ws, j, λ, refit, cols[j])
  end
  symmetrize ? _assemble_symmetric(cols, 1) : _assemble(cols, 1)
end
