# Parametric simplex method for the column problems of CLIME and ACLIME.
#
# For column i, the linear program is
#
#     minimize ‖β‖₁  subject to  |(Sβ - eᵢ)ⱼ| ≤ cⱼ (μ + γ βᵢ),  j = 1, …, p,          (Pμ)
#
# with constraint weights c > 0 and γ ≥ 0. CLIME has c = 1 and γ = 0, and step 2
# of ACLIME has cⱼ = √σⱼⱼ and γ = 0. Step 1 of ACLIME, whose bounds grow with βᵢ,
# is (P₀) with γ = λₙ and cⱼ = σⱼⱼ ∨ σᵢᵢ.
#
# μ only enters the right-hand side, so the solution is piecewise linear in μ. The
# parametric (dual) simplex method follows it from μ = 1/cᵢ, where β = 0, down to
# the smallest μ asked for: the basis stays optimal until a basic variable reaches a
# bound, and a dual simplex pivot then replaces that variable (Vanderbei, 2020,
# Chapter 7; Pang, Liu and Vanderbei, 2014). Each pivot is one breakpoint of the
# path.
#
# In the standard form, with β = u - v and a slack for each of the 2p one-sided
# constraints, the basis is a 2p × 2p matrix, but all its columns except those of
# the active variables A are unit vectors. Everything the method needs comes from
# the k × k block M = G[R, A], where R are the tight constraints (|R| = |A| = k) and
# row (j, s) of G, for the side s = ±1 of constraint j, is
#
#     G[(j, s), l] = Sⱼₗ - s γ cⱼ [l == i].
#
# The workspace keeps N = M⁻¹, updated after each pivot in O(k²) operations. The
# other work of a pivot is two products of k columns of S with a vector, so a pivot
# costs O(pk) and a column with s nonzero entries O(p s²) in total.
#
# Notation: β_A = a + μ b on A; the residual r = Sβ - eᵢ = r0 + μ r1; the dual
# variable η of the tight constraints solves M'η = τ (τ the signs of β_A), and
# σ = G[R, :]'η, with |σ| ≤ 1 and σ_A = τ at an optimal basis. The slack of side s of
# constraint j is z = cⱼ (μ + γ βᵢ) - s rⱼ.

mutable struct _PSMWorkspace{T<:AbstractFloat}
  S::Matrix{T}
  p::Int
  i::Int                 # current column
  γ::T
  c::Vector{T}           # constraint weights of the current column
  k::Int                 # size of the basis block
  cap::Int               # size of N
  A::Vector{Int}         # active variables, in the order of the rows of N
  τ::Vector{T}           # their signs
  posA::Vector{Int}      # posA[l]: position of l in A, or 0
  R::Vector{Int}         # tight constraints, in the order of the columns of N
  sd::Vector{T}          # their sides
  posR::Vector{Int}      # posR[j]: position of j in R, or 0
  N::Matrix{T}           # N[1:k, 1:k] = inv(G[R, A])
  a::Vector{T}           # β_A = a + μ b
  b::Vector{T}
  η::Vector{T}           # dual variables of the tight constraints
  r0::Vector{T}          # r = r0 + μ r1
  r1::Vector{T}
  σ::Vector{T}           # G[R, :]'η
  g::Vector{T}           # pivot row
  ρ::Vector{T}           # row of the basis inverse, on R
  x::Vector{T}           # scratch, length p
  y::Vector{T}           # scratch, length p
  z::Vector{T}           # scratch, length p
  npivots::Int           # pivots since the last factorization
  drift0::T              # _drift right after the last factorization
  θ::T                   # dual step of the last pivot
end

function _PSMWorkspace(S::Matrix{T}) where {T<:AbstractFloat}
  p = size(S, 1)
  cap = min(p, 32)
  _PSMWorkspace{T}(S, p, 0, zero(T), ones(T, p), 0, cap,
    zeros(Int, p), zeros(T, p), zeros(Int, p),
    zeros(Int, p), zeros(T, p), zeros(Int, p),
    zeros(T, cap, cap),
    zeros(T, p), zeros(T, p), zeros(T, p),
    zeros(T, p), zeros(T, p), zeros(T, p), zeros(T, p), zeros(T, p),
    zeros(T, p), zeros(T, p), zeros(T, p), 0, zero(T), zero(T))
end

function _grow!(ws::_PSMWorkspace{T}, k::Int) where {T}
  k <= ws.cap && return ws
  cap = min(ws.p, max(k, 2 * ws.cap))
  N = zeros(T, cap, cap)
  kk = ws.k
  @inbounds for t in 1:kk, q in 1:kk
    N[q, t] = ws.N[q, t]
  end
  ws.N, ws.cap = N, cap
  ws
end

# start column i at μ = 1/cᵢ, where β = 0 is optimal
function _reset!(ws::_PSMWorkspace{T}, i::Int) where {T}
  @inbounds for q in 1:ws.k
    ws.posA[ws.A[q]] = 0
    ws.posR[ws.R[q]] = 0
  end
  ws.k, ws.i, ws.npivots, ws.drift0 = 0, i, 0, zero(T)
  fill!(ws.σ, zero(T))
  fill!(ws.r0, zero(T))
  fill!(ws.r1, zero(T))
  ws.r0[i] = -one(T)
  ws
end

# G[(j, s), l]
@inline _G(ws::_PSMWorkspace, j, s, l) =
  @inbounds l == ws.i ? ws.S[j, l] - s * ws.γ * ws.c[j] : ws.S[j, l]

# out = S[:, idx[1:k]] * coef[1:k], four columns at a time so that out is loaded
# and stored once for every four columns of S
function _combine!(out::Vector{T}, S::Matrix{T}, idx::Vector{Int}, coef::Vector{T}, k::Int) where {T}
  fill!(out, zero(T))
  p = length(out)
  t = 1
  @inbounds while t + 3 <= k
    c1, c2, c3, c4 = idx[t], idx[t+1], idx[t+2], idx[t+3]
    v1, v2, v3, v4 = coef[t], coef[t+1], coef[t+2], coef[t+3]
    @simd for j in 1:p
      out[j] += v1 * S[j, c1] + v2 * S[j, c2] + v3 * S[j, c3] + v4 * S[j, c4]
    end
    t += 4
  end
  @inbounds while t <= k
    c1, v1 = idx[t], coef[t]
    @simd for j in 1:p
      out[j] += v1 * S[j, c1]
    end
    t += 1
  end
  out
end

# out[1:k] = N[1:k, 1:k] v[1:k], column by column
function _mulN!(out::Vector{T}, N::Matrix{T}, v::Vector{T}, k::Int) where {T}
  @inbounds for q in 1:k
    out[q] = zero(T)
  end
  @inbounds for t in 1:k
    vt = v[t]
    @simd for q in 1:k
      out[q] += N[q, t] * vt
    end
  end
  out
end

# out[1:k] = N[1:k, 1:k]' v[1:k]
function _mulNt!(out::Vector{T}, N::Matrix{T}, v::Vector{T}, k::Int) where {T}
  @inbounds for t in 1:k
    acc = zero(T)
    @simd for q in 1:k
      acc += N[q, t] * v[q]
    end
    out[t] = acc
  end
  out
end

# β_A = a + μ b solves G[R, A] β_A = e_R + μ (s ∘ c)_R; then r0 and r1
function _primal!(ws::_PSMWorkspace{T}) where {T}
  k, N, S, p, i = ws.k, ws.N, ws.S, ws.p, ws.i
  a, b, r0, r1 = ws.a, ws.b, ws.r0, ws.r1
  ri = ws.posR[i]
  @inbounds for q in 1:k
    a[q] = ri == 0 ? zero(T) : N[q, ri]
    b[q] = zero(T)
  end
  @inbounds for t in 1:k
    st = ws.sd[t] * ws.c[ws.R[t]]
    @simd for q in 1:k
      b[q] += N[q, t] * st
    end
  end
  fill!(r0, zero(T))
  fill!(r1, zero(T))
  A = ws.A
  q = 1
  @inbounds while q + 3 <= k
    c1, c2, c3, c4 = A[q], A[q+1], A[q+2], A[q+3]
    a1, a2, a3, a4 = a[q], a[q+1], a[q+2], a[q+3]
    b1, b2, b3, b4 = b[q], b[q+1], b[q+2], b[q+3]
    @simd for j in 1:p
      s1, s2, s3, s4 = S[j, c1], S[j, c2], S[j, c3], S[j, c4]
      r0[j] += (a1 * s1 + a2 * s2) + (a3 * s3 + a4 * s4)
      r1[j] += (b1 * s1 + b2 * s2) + (b3 * s3 + b4 * s4)
    end
    q += 4
  end
  @inbounds while q <= k
    c1, a1, b1 = A[q], a[q], b[q]
    @simd for j in 1:p
      r0[j] += a1 * S[j, c1]
      r1[j] += b1 * S[j, c1]
    end
    q += 1
  end
  @inbounds r0[i] -= one(T)
  ws
end

# How far the tight constraints are from holding with the computed β_A: the
# residual of row t of R should be sₜ cₜ (μ + γβᵢ), for all μ, which rounding
# errors in N, accumulated over the updates, break
function _drift(ws::_PSMWorkspace{T}) where {T}
  qi = ws.posA[ws.i]
  ai = qi == 0 ? zero(T) : ws.a[qi]
  bi = qi == 0 ? zero(T) : ws.b[qi]
  d = zero(T)
  @inbounds for t in 1:ws.k
    j, sc = ws.R[t], ws.sd[t] * ws.c[ws.R[t]]
    d = max(d, abs(ws.r0[j] - sc * ws.γ * ai), abs(ws.r1[j] - sc * (one(T) + ws.γ * bi)))
  end
  d
end

# Refactorize after `refactor` pivots, or earlier when the updates have drifted
# by more than tol and by more than 10 times what a factorization achieves (which
# depends on the conditioning of the basis)
function _refactor_if_needed!(ws::_PSMWorkspace{T}, refactor::Int, tol::T) where {T}
  ws.npivots > 0 || return ws
  if ws.npivots >= refactor || _drift(ws) > max(tol, 10 * ws.drift0)
    _factorize!(ws) || error("singular basis in column $(ws.i)")
    _dual!(ws)
    _primal!(ws)
    ws.drift0 = _drift(ws)
  end
  ws
end

# η solves M'η = τ, then σ = G[R, :]'η; done from scratch after a factorization
function _dual!(ws::_PSMWorkspace{T}) where {T}
  k, N = ws.k, ws.N
  _mulNt!(ws.η, N, ws.τ, k)
  _combine!(ws.σ, ws.S, ws.R, ws.η, k)
  corr = zero(T)
  @inbounds for t in 1:k
    corr += ws.η[t] * ws.sd[t] * ws.c[ws.R[t]]
  end
  @inbounds ws.σ[ws.i] -= ws.γ * corr
  ws
end

# N = inv(G[R, A]) from scratch, in place, by LAPACK
function _factorize!(ws::_PSMWorkspace{T}) where {T}
  k, N = ws.k, ws.N
  ws.npivots = 0
  k == 0 && return true
  @inbounds for q in 1:k, t in 1:k
    N[t, q] = _G(ws, ws.R[t], ws.sd[t], ws.A[q])
  end
  M = view(N, 1:k, 1:k)
  _, ipiv, info = LAPACK.getrf!(M; check=false)
  info == 0 || return false
  LAPACK.getri!(M, ipiv)
  true
end

# The largest μ' ≤ μ at which a basic variable reaches its bound: an active
# variable reaching zero (kind 1, position in A) or the slack of side s of a
# constraint reaching zero (kind 2, constraint j). Returns μ' = -Inf if the basis
# stays feasible for all smaller μ. Rates of the slacks that are zero up to
# rounding are ignored: a slack that stays at zero along the segment, as that of a
# constraint identical to a tight one, would otherwise trigger pivots that swap
# the two forever. Ties are broken by the larger rate, or with `bland`, by the
# smaller index of the variable (see _index).
function _next_event(ws::_PSMWorkspace{T}, μ::T, tol::T, bland::Bool=false) where {T}
  best, kind, idx, side, key = T(-Inf), 0, 0, zero(T), T(-Inf)
  k, a, b = ws.k, ws.a, ws.b
  @inbounds for q in 1:k
    tb = ws.τ[q] * b[q]
    tb > 0 || continue
    μx = min(-a[q] / b[q], μ)
    kx = bland ? -T(ws.A[q]) : tb
    if μx > best || (μx == best && kx > key)
      best, kind, idx, side, key = μx, 1, q, zero(T), kx
    end
  end
  qi = ws.posA[ws.i]
  ai = qi == 0 ? zero(T) : a[qi]
  bi = qi == 0 ? zero(T) : b[qi]
  u1, u0 = one(T) + ws.γ * bi, ws.γ * ai
  # z = z0 + μ z1 reaches zero at μ' = -z0/z1 when z1 > 0; the division is only
  # done when μ' can beat the best so far, -z0 ≥ best⋅z1
  @inbounds for j in 1:ws.p
    cj = ws.c[j]
    tj = ws.posR[j]
    w1, w0 = cj * u1, cj * u0
    if tj != 0
      # the other side of a tight constraint: its slack is 2cⱼ(μ + γβᵢ) exactly,
      # which reaches zero at μ = -γaᵢ/(1 + γbᵢ) ≤ 0; from r0 and r1 the rounding
      # could put that point slightly above zero
      if u1 > 0
        μx = min(-u0 / u1, μ)
        s = -ws.sd[tj]
        kx = bland ? -T(_index(ws, j, s)) : 2w1
        if μx > best || (μx == best && kx > key)
          best, kind, idx, side, key = μx, 2, j, s, kx
        end
      end
      continue
    end
    r1j, r0j = ws.r1[j], ws.r0[j]
    ztol = tol * w1
    z1 = w1 - r1j
    if z1 > ztol
      z0 = w0 - r0j
      if -z0 >= best * z1
        μx = min(-z0 / z1, μ)
        kx = bland ? -T(_index(ws, j, one(T))) : z1
        if μx > best || (μx == best && kx > key)
          best, kind, idx, side, key = μx, 2, j, one(T), kx
        end
      end
    end
    z1 = w1 + r1j
    if z1 > ztol
      z0 = w0 + r0j
      if -z0 >= best * z1
        μx = min(-z0 / z1, μ)
        kx = bland ? -T(_index(ws, j, -one(T))) : z1
        if μx > best || (μx == best && kx > key)
          best, kind, idx, side, key = μx, 2, j, -one(T), kx
        end
      end
    end
  end
  best, kind, idx, side
end

# The order of the variables for Bland's rule: the entries of β first, then the
# slacks of the constraints, side +1 before side -1
@inline _index(ws::_PSMWorkspace, j::Int, s) = ws.p + 2j - (s > 0 ? 1 : 0)

# One dual simplex pivot at a breakpoint, for the leaving variable given by
# _next_event. Returns false if no variable can enter, in which case (Pμ) is
# infeasible below the breakpoint.
function _pivot!(ws::_PSMWorkspace{T}, kind::Int, idx::Int, s::T, tol::T, bland::Bool=false) where {T}
  k, N, S, p, i, γ = ws.k, ws.N, ws.S, ws.p, ws.i, ws.γ
  ρ, g, x = ws.ρ, ws.g, ws.x

  # row of the basis inverse for the leaving variable, ρ (on R), and the pivot
  # row g = G[R, :]'ρ (+ s G[(j, s), :]) over all variables
  if kind == 1
    q = idx
    τq = ws.τ[q]
    @inbounds for t in 1:k
      ρ[t] = τq * N[q, t]
    end
    _combine!(g, S, ws.R, ρ, k)
    corr = zero(T)
    @inbounds for t in 1:k
      corr += ρ[t] * ws.sd[t] * ws.c[ws.R[t]]
    end
    @inbounds g[i] -= γ * corr
  else
    j = idx
    # x = G[(j, s), A]
    @inbounds for q in 1:k
      x[q] = _G(ws, j, s, ws.A[q])
    end
    _mulNt!(ρ, N, x, k)
    @inbounds for t in 1:k
      ρ[t] *= -s
    end
    _combine!(g, S, ws.R, ρ, k)
    @inbounds @simd for l in 1:p
      g[l] += s * S[l, j]
    end
    corr = ws.c[j]
    @inbounds for t in 1:k
      corr += ρ[t] * ws.sd[t] * ws.c[ws.R[t]]
    end
    @inbounds g[i] -= γ * corr
  end

  # dual ratio test, with the tolerance of Harris (1973): the variable that
  # minimizes (d + tol)/|α|, d its reduced cost and α its entry of the pivot row,
  # which among nearly tied variables prefers the largest pivot; with `bland`, the
  # variable of smallest index among those of minimal ratio
  gmax = zero(T)
  @inbounds for l in 1:p
    gmax = max(gmax, abs(g[l]))
  end
  @inbounds for t in 1:k
    gmax = max(gmax, abs(ρ[t]))
  end
  ptol = tol * max(gmax, one(T))
  leaving = kind == 1 ? ws.A[idx] : 0
  if bland
    found, ekind, eidx, esign, epiv, ed = _ratio_bland(ws, g, ρ, leaving, ptol, tol)
    found || return false
    @goto update
  end
  # the variable l enters with the sign -sign(gₗ), and its reduced cost is
  # d = 1 + sign(gₗ) σₗ; divisions are only done for improvements
  θmax = T(Inf)
  ekind, eidx = 0, 0
  @inbounds for l in 1:p
    gl = g[l]
    ag = abs(gl)
    ag > ptol || continue
    (ws.posA[l] != 0 && l != leaving) && continue
    d = one(T) + (gl > 0 ? ws.σ[l] : -ws.σ[l]) + tol
    if d < θmax * ag
      θmax, ekind, eidx = d / ag, 1, l
    end
  end
  @inbounds for t in 1:k
    α = ws.sd[t] * ρ[t]
    α < -ptol || continue
    d = -ws.sd[t] * ws.η[t] + tol
    if d < -θmax * α
      θmax, ekind, eidx = d / -α, 2, t
    end
  end
  isfinite(θmax) || return false
  if ekind == 1
    gl = g[eidx]
    esign, epiv, ed = gl > 0 ? -one(T) : one(T), abs(gl), one(T) + (gl > 0 ? ws.σ[eidx] : -ws.σ[eidx])
  else
    esign, epiv, ed = zero(T), -ws.sd[eidx] * ρ[eidx], -ws.sd[eidx] * ws.η[eidx]
  end

  @label update
  θ = max(ed, zero(T)) / epiv
  ws.θ = θ

  # dual update: η ← η - θρ, σ ← σ - θg
  @inbounds for t in 1:k
    ws.η[t] -= θ * ρ[t]
  end
  @inbounds @simd for l in 1:p
    ws.σ[l] -= θ * g[l]
  end

  # basis update
  if kind == 1
    q = idx
    if ekind == 1 && eidx == leaving
      # βₗ crosses zero: only its sign changes
      ws.τ[q] = esign
    elseif ekind == 1
      _replace_column!(ws, q, eidx, esign)
    else
      _delete!(ws, q, eidx)
    end
  else
    if ekind == 1
      _border!(ws, idx, s, eidx, esign, -s * θ)
    else
      _replace_row!(ws, eidx, idx, s, -s * θ)
    end
  end
  @inbounds for q in 1:ws.k
    ws.σ[ws.A[q]] = ws.τ[q]
  end
  ws.npivots += 1
  true
end

# Bland's rule for the dual ratio test: among the candidates whose ratio is
# within rounding of the smallest, the one of smallest index (the entries of β
# come before the slacks). Returns (found, kind, idx, sign, |pivot|, reduced cost).
function _ratio_bland(ws::_PSMWorkspace{T}, g::Vector{T}, ρ::Vector{T}, leaving::Int, ptol::T, tol::T) where {T}
  p, k = ws.p, ws.k
  θmin = T(Inf)
  @inbounds for l in 1:p
    ag = abs(g[l])
    ag > ptol || continue
    (ws.posA[l] != 0 && l != leaving) && continue
    d = one(T) + (g[l] > 0 ? ws.σ[l] : -ws.σ[l])
    θmin = min(θmin, max(d, zero(T)) / ag)
  end
  @inbounds for t in 1:k
    α = ws.sd[t] * ρ[t]
    α < -ptol || continue
    θmin = min(θmin, max(-ws.sd[t] * ws.η[t], zero(T)) / -α)
  end
  isfinite(θmin) || return (false, 0, 0, zero(T), zero(T), zero(T))
  window = θmin + tol * max(one(T), θmin)
  @inbounds for l in 1:p
    gl = g[l]
    ag = abs(gl)
    ag > ptol || continue
    (ws.posA[l] != 0 && l != leaving) && continue
    d = one(T) + (gl > 0 ? ws.σ[l] : -ws.σ[l])
    max(d, zero(T)) / ag <= window && return (true, 1, l, gl > 0 ? -one(T) : one(T), ag, d)
  end
  best, bidx = typemax(Int), 0
  @inbounds for t in 1:k
    α = ws.sd[t] * ρ[t]
    α < -ptol || continue
    max(-ws.sd[t] * ws.η[t], zero(T)) / -α <= window || continue
    ix = _index(ws, ws.R[t], ws.sd[t])
    ix < best && ((best, bidx) = (ix, t))
  end
  t = bidx
  (true, 2, t, zero(T), -ws.sd[t] * ρ[t], -ws.sd[t] * ws.η[t])
end

# variable l (sign τl) replaces the active variable at position q:
# column q of M becomes u = G[R, l]
function _replace_column!(ws::_PSMWorkspace{T}, q::Int, l::Int, τl::T) where {T}
  k, N, w = ws.k, ws.N, ws.x
  @inbounds for t in 1:k
    ws.y[t] = _G(ws, ws.R[t], ws.sd[t], l)
  end
  _mulN!(w, N, ws.y, k)
  piv = w[q]
  @inbounds for t in 1:k
    N[q, t] /= piv
  end
  @inbounds for t in 1:k
    nqt = N[q, t]
    @simd for qq in 1:k
      N[qq, t] -= ifelse(qq == q, zero(T), w[qq] * nqt)
    end
  end
  @inbounds begin
    ws.posA[ws.A[q]] = 0
    ws.A[q], ws.τ[q], ws.posA[l] = l, τl, q
  end
  ws
end

# the active variable at position q leaves and the tight constraint at position r
# is released: delete row q and column r of N (the inverse of M without row r and
# column q), then move the last row and column into the holes
function _delete!(ws::_PSMWorkspace{T}, q::Int, r::Int) where {T}
  k, N = ws.k, ws.N
  piv = N[q, r]
  @inbounds for t in 1:k
    t == r && continue
    f = N[q, t] / piv
    @simd for qq in 1:k
      N[qq, t] -= ifelse(qq == q, zero(T), N[qq, r] * f)
    end
  end
  @inbounds begin
    ws.posA[ws.A[q]] = 0
    ws.posR[ws.R[r]] = 0
    if q != k
      for t in 1:k
        N[q, t] = N[k, t]
      end
      ws.A[q], ws.τ[q] = ws.A[k], ws.τ[k]
      ws.posA[ws.A[q]] = q
    end
    if r != k
      for qq in 1:k
        N[qq, r] = N[qq, k]
      end
      ws.R[r], ws.sd[r], ws.η[r] = ws.R[k], ws.sd[k], ws.η[k]
      ws.posR[ws.R[r]] = r
    end
  end
  ws.k = k - 1
  ws
end

# constraint j (side s, dual value ηj) becomes tight and variable l (sign τl)
# becomes active: M gains the row G[(j, s), A] and the column G[R, l]
function _border!(ws::_PSMWorkspace{T}, j::Int, s::T, l::Int, τl::T, ηj::T) where {T}
  k = ws.k
  _grow!(ws, k + 1)
  N, xx, yy, u, v = ws.N, ws.x, ws.y, ws.ρ, ws.z
  # u = G[R, l], v = G[(j, s), A], xx = N u (on A), yy = N'v (on R)
  @inbounds for t in 1:k
    u[t] = _G(ws, ws.R[t], ws.sd[t], l)
  end
  @inbounds for q in 1:k
    v[q] = _G(ws, j, s, ws.A[q])
  end
  _mulN!(xx, N, u, k)
  _mulNt!(yy, N, v, k)
  schur = _G(ws, j, s, l)
  @inbounds for q in 1:k
    schur -= v[q] * xx[q]
  end
  @inbounds for t in 1:k
    f = yy[t] / schur
    @simd for q in 1:k
      N[q, t] += xx[q] * f
    end
    N[k+1, t] = -f
  end
  @inbounds for q in 1:k
    N[q, k+1] = -xx[q] / schur
  end
  N[k+1, k+1] = one(T) / schur
  @inbounds begin
    ws.A[k+1], ws.τ[k+1], ws.posA[l] = l, τl, k + 1
    ws.R[k+1], ws.sd[k+1], ws.posR[j], ws.η[k+1] = j, s, k + 1, ηj
  end
  ws.k = k + 1
  ws
end

# constraint j (side s, dual value ηj) becomes tight in place of the tight
# constraint at position r: row r of M becomes G[(j, s), A]
function _replace_row!(ws::_PSMWorkspace{T}, r::Int, j::Int, s::T, ηj::T) where {T}
  k, N, yy = ws.k, ws.N, ws.y
  @inbounds for q in 1:k
    ws.x[q] = _G(ws, j, s, ws.A[q])
  end
  _mulNt!(yy, N, ws.x, k)
  piv = yy[r]
  @inbounds for q in 1:k
    N[q, r] /= piv
  end
  @inbounds for t in 1:k
    t == r && continue
    f = yy[t]
    @simd for q in 1:k
      N[q, t] -= N[q, r] * f
    end
  end
  @inbounds begin
    ws.posR[ws.R[r]] = 0
    ws.R[r], ws.sd[r], ws.posR[j], ws.η[r] = j, s, r, ηj
  end
  ws
end

# The basic variable that is most infeasible at μ, as (kind, idx, side) in the
# convention of _next_event, or kind = 0 if none is infeasible by more than tol.
# The slacks are measured relative to their weight cⱼ, in the units of μ.
function _most_infeasible(ws::_PSMWorkspace{T}, μ::T, tol::T, bland::Bool=false) where {T}
  k, a, b = ws.k, ws.a, ws.b
  scale = one(T)
  @inbounds for q in 1:k
    scale = max(scale, abs(a[q] + μ * b[q]))
  end
  worst, kind, idx, side = tol * scale, 0, 0, zero(T)
  key = typemax(Int)    # with `bland`, the smallest index among the infeasible variables
  @inbounds for q in 1:k
    x = ws.τ[q] * (a[q] + μ * b[q])
    if bland
      if -x > tol * scale && ws.A[q] < key
        kind, idx, side, key = 1, q, zero(T), ws.A[q]
      end
    elseif -x > worst
      worst, kind, idx, side = -x, 1, q, zero(T)
    end
  end
  qi = ws.posA[ws.i]
  βi = qi == 0 ? zero(T) : a[qi] + μ * b[qi]
  ztol = tol * max(μ, one(T))
  bound = μ + ws.γ * βi
  @inbounds for j in 1:ws.p
    cj = ws.c[j]
    tj = ws.posR[j]
    tight = tj == 0 ? zero(T) : ws.sd[tj]
    rj = ws.r0[j] + μ * ws.r1[j]
    for s in (one(T), -one(T))
      s == tight && continue
      # minus the slack, relative to cⱼ; for the other side of a tight
      # constraint the slack is 2cⱼ(μ + γβᵢ) exactly
      v = tight == 0 ? s * rj / cj - bound : -2bound
      if bland
        if v > ztol && kind != 1 && _index(ws, j, s) < key
          kind, idx, side, key = 2, j, s, _index(ws, j, s)
        end
      elseif v > ztol && v > worst
        worst, kind, idx, side = v, 2, j, s
      end
    end
  end
  kind, idx, side
end

# Dual simplex iterations from the current (dual feasible) basis until it is
# primal feasible at μ. Returns the number of pivots, or -1 if the problem is
# infeasible at μ.
function _dual_simplex!(ws::_PSMWorkspace{T}, μ::T, tol::T, refactor::Int, maxpivots::Int) where {T}
  pivots = stall = 0
  while true
    # Bland's rule, which cannot cycle, after many degenerate pivots in a row
    bland = stall > max(50, 2 * ws.k)
    kind, idx, s = _most_infeasible(ws, μ, tol, bland)
    kind == 0 && return pivots
    _pivot!(ws, kind, idx, s, tol, bland) || return -1
    stall = ws.θ > tol ? 0 : stall + 1
    pivots += 1
    pivots > maxpivots && error("DualSimplex: column $(ws.i) needed more than $maxpivots pivots")
    _primal!(ws)
    _refactor_if_needed!(ws, refactor, tol)
  end
end
