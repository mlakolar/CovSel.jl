# A first order method for the column problems of CLIME: the primal-dual hybrid
# gradient method (Chambolle and Pock, 2011) with the adaptive restarts and primal
# weights of PDLP (Applegate et al., 2021), applied to all columns at once.
#
# Column i is the saddle point problem
#
#     min_x max_y  ‖x‖₁ + ⟨Sx, y⟩ - ⟨eᵢ, y⟩ - ⟨bᵢ, |y|⟩,      bᵢ = μᵢ c,
#
# whose dual is: maximize -yᵢ - ⟨bᵢ, |y|⟩ subject to ‖Sy‖∞ ≤ 1. One iteration is
#
#     x⁺ = soft(x - τ S y, τ),    y⁺ = σ soft(y/σ + S(2x⁺ - x) - eᵢ, bᵢ),
#
# with τσ‖S‖² < 1, so it costs two products with S. Every column has its own
# primal weight ω = σ/τ, restarts and stopping rule, and columns that have
# converged leave the products. The iterates are exactly sparse after soft
# thresholding, and the products use only their nonzero entries when they are
# sparse enough.

"""
    PDHG(; tol=1e-6, polish=true, maxiter=20_000, check=40)

The primal-dual hybrid gradient method (Chambolle and Pock, 2011), with the
adaptive restarts, primal weights and step sizes of PDLP (Applegate et al.,
2021). It solves all columns at once, warm started along a sequence of
penalties, and an iteration costs two products of `S` with the current
iterates, which use only their nonzero entries when they are sparse.

A first order method: a column stops when the relative error of its
optimality conditions (primal and dual feasibility, duality gap), checked every
`check` iterations, is below `tol`. With `polish=true`, a column whose error is
below `1e-3` is first *polished*: the linear program restricted to the support
of the primal iterate and the active constraints of the dual one is solved
exactly, and the result is kept, as the exact solution, if it satisfies all
the optimality conditions (it usually does at that point, since the iterates
are exactly sparse).

Step 1 of [`aclime`](@ref) is not of the form PDHG handles and is solved by
[`ParametricSimplex`](@ref).
"""
Base.@kwdef struct PDHG <: CLIMESolver
  tol::Float64 = 1e-6
  polish::Bool = true
  maxiter::Int = 20_000
  check::Int = 40
end

# largest eigenvalue of the positive semidefinite S, from above
function _opnorm_psd(S::Matrix{T}) where {T}
  p = size(S, 1)
  v = ones(T, p) ./ sqrt(T(p))
  w = similar(v)
  λ = zero(T)
  for _ in 1:100
    mul!(w, S, v)
    λnew = dot(v, w)
    nw = norm(w)
    iszero(nw) && return one(T)
    v .= w ./ nw
    abs(λnew - λ) <= 1e-6 * λnew && (λ = λnew; break)
    λ = λnew
  end
  # the Rayleigh quotient is a lower bound; ‖Sv - λv‖ bounds the error
  mul!(w, S, v)
  λ + norm(w .- λ .* v) + eps(T) * p
end

# out[:, 1:m] = S * X[:, 1:m], using the nonzero entries of X when it is sparse
function _mulS!(out::Matrix{T}, S::Matrix{T}, X::Matrix{T}, m::Int, idx::Vector{Vector{Int}},
                vals::Vector{Vector{T}}, threaded::Bool) where {T}
  p = size(S, 1)
  nz = 0
  @inbounds for k in 1:m, j in 1:p
    nz += !iszero(X[j, k])
  end
  if nz > p * m ÷ 8
    mul!(view(out, :, 1:m), S, view(X, :, 1:m))
    return out
  end
  _foreach_maybe_threaded(m, threaded) do k
    tid = Threads.threadid()
    _sparsecol_mul!(out, S, X, k, idx[tid], vals[tid])
  end
  out
end

function _sparsecol_mul!(out::Matrix{T}, S::Matrix{T}, X::Matrix{T}, k::Int, idx::Vector{Int}, vals::Vector{T}) where {T}
  p = size(S, 1)
  nk = 0
  @inbounds for j in 1:p
    x = X[j, k]
    if !iszero(x)
      nk += 1
      idx[nk], vals[nk] = j, x
    end
  end
  @inbounds for j in 1:p
    out[j, k] = zero(T)
  end
  _combine_into!(out, k, S, idx, vals, nk)
end

# out[:, col] += S[:, idx[1:n]] * vals[1:n]
function _combine_into!(out::Matrix{T}, col::Int, S::Matrix{T}, idx::Vector{Int}, vals::Vector{T}, n::Int) where {T}
  p = size(S, 1)
  t = 1
  @inbounds while t + 3 <= n
    c1, c2, c3, c4 = idx[t], idx[t+1], idx[t+2], idx[t+3]
    v1, v2, v3, v4 = vals[t], vals[t+1], vals[t+2], vals[t+3]
    @simd for j in 1:p
      out[j, col] += v1 * S[j, c1] + v2 * S[j, c2] + v3 * S[j, c3] + v4 * S[j, c4]
    end
    t += 4
  end
  @inbounds while t <= n
    c1, v1 = idx[t], vals[t]
    @simd for j in 1:p
      out[j, col] += v1 * S[j, c1]
    end
    t += 1
  end
  out
end

function _foreach_maybe_threaded(f, m::Int, threaded::Bool)
  if threaded && m > 1
    Threads.@threads :static for k in 1:m
      f(k)
    end
  else
    for k in 1:m
      f(k)
    end
  end
  nothing
end

mutable struct _PDHGState{T}
  X::Matrix{T}; Y::Matrix{T}; SX::Matrix{T}; SY::Matrix{T}
  Xn::Matrix{T}; Yn::Matrix{T}; SXn::Matrix{T}; SYn::Matrix{T}
  Xa::Matrix{T}; Ya::Matrix{T}; SXa::Matrix{T}; SYa::Matrix{T}   # averages since the last restart
  Xr::Matrix{T}; Yr::Matrix{T}       # iterates at the last restart
  cols::Vector{Int}                  # problem column stored in each column
  ω::Vector{T}                       # primal weights
  η::Vector{T}                       # step sizes
  wa::Vector{T}                      # total step size since the last restart
  na::Vector{Int}                    # iterations since the last restart
  kktr::Vector{T}                    # KKT error at the last restart
  kktp::Vector{T}                    # KKT error of the restart candidate at the last check
  μ::Vector{T}                       # penalty of each stored column
  idx::Vector{Vector{Int}}
  vals::Vector{Vector{T}}
end

function _PDHGState(S::Matrix{T}) where {T}
  p = size(S, 1)
  z() = zeros(T, p, p)
  nt = Threads.maxthreadid()
  _PDHGState{T}(z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(),
    collect(1:p), ones(T, p), zeros(T, p), zeros(T, p), zeros(Int, p), fill(T(Inf), p), fill(T(Inf), p), zeros(T, p),
    [zeros(Int, p) for _ in 1:nt], [zeros(T, p) for _ in 1:nt])
end

function _swapcols!(st::_PDHGState, a::Int, b::Int)
  a == b && return
  for M in (st.X, st.Y, st.SX, st.SY, st.Xn, st.Yn, st.SXn, st.SYn, st.Xa, st.Ya, st.SXa, st.SYa, st.Xr, st.Yr)
    @inbounds for j in axes(M, 1)
      M[j, a], M[j, b] = M[j, b], M[j, a]
    end
  end
  for v in (st.cols, st.ω, st.η, st.wa, st.na, st.kktr, st.kktp, st.μ)
    v[a], v[b] = v[b], v[a]
  end
end

# relative KKT error of stored column k of the iterate (X, Y) with products
# (SX, SY): primal infeasibility, dual infeasibility and duality gap
function _kkt(X, Y, SX, SY, st::_PDHGState{T}, c::Vector{T}, k::Int) where {T}
  i, μ = st.cols[k], st.μ[k]
  P = rp = rd = zero(T)
  D = -Y[i, k]
  bmax = zero(T)
  @inbounds for j in axes(X, 1)
    b = μ * c[j]
    bmax = max(bmax, b)
    P += abs(X[j, k])
    rp = max(rp, abs(SX[j, k] - (j == i)) - b)
    rd = max(rd, abs(SY[j, k]) - 1)
    D -= b * abs(Y[j, k])
  end
  max(rp / (1 + bmax), rd, abs(P - D) / (1 + abs(P) + abs(D)))
end

_kkt(st::_PDHGState, c, k) = _kkt(st.X, st.Y, st.SX, st.SY, st, c, k)
_kkt_average(st::_PDHGState, c, k) = _kkt(st.Xa, st.Ya, st.SXa, st.SYa, st, c, k)

# copy column k of the arrays in `from` into those in `to`
function _copycol!(to, from, k)
  for (A, B) in zip(to, from)
    @inbounds for j in axes(A, 1)
      A[j, k] = B[j, k]
    end
  end
end

function _solve_columns(solver::PDHG, S::Matrix{T}, spec::_CLIMESpec{T},
                        μgrid::Vector{T}, colscale::Vector{T}, ntasks::Int) where {T}
  (iszero(spec.γ) && !spec.maxweights) || throw(ArgumentError("PDHG: only for γ = 0"))
  p, nμ = size(S, 1), length(μgrid)
  c = spec.w
  st = _PDHGState(S)
  # PDLP starts from the step size 1/max|Sᵢⱼ|, larger than the 1/‖S‖ that
  # guarantees convergence; the adaptive rule adjusts it
  fill!(st.η, one(T) / maximum(abs, S))
  threaded = ntasks > 1
  cols = [_ColumnPath{T}(nμ) for _ in 1:p]
  μbad = fill(-one(T), p)
  tol = T(solver.tol)
  for t in 1:nμ
    # all columns active again, warm started from the previous penalty
    for k in 1:p
      st.μ[k] = μgrid[t] * colscale[st.cols[k]]
      st.kktr[k] = st.kktp[k] = T(Inf)
    end
    copyto!(st.Xr, st.X)
    copyto!(st.Yr, st.Y)
    _reset_averages!(st, 1:p)
    m = p
    iter = 0
    while m > 0 && iter < solver.maxiter
      iter += 1
      _pdhg_step!(st, S, c, iter, m, threaded)
      if iter % solver.check == 0
        k = 1
        while k <= m
          kkt = _kkt(st, c, k)
          kkta = _kkt_average(st, c, k)
          if kkta < kkt
            # the average is the better candidate
            if kkta <= tol || _restart_due(st, k, kkta, iter)
              _copycol!((st.X, st.Y, st.SX, st.SY), (st.Xa, st.Ya, st.SXa, st.SYa), k)
              kkt = kkta
            end
          end
          if (solver.polish && kkt <= 1e-3 && _polish!(st, S, c, k)) || kkt <= tol
            _freeze!(st, k, m)
            m -= 1
            continue
          end
          _restart_due(st, k, kkt, iter) && _restart!(st, k, kkt)
          st.kktp[k] = min(kkt, kkta)
          k += 1
        end
      end
    end
    m > 0 && @warn "PDHG: $m of the column problems did not converge in $(solver.maxiter) iterations " *
                   "at the penalty $(μgrid[t]); they may be infeasible"
    # entries below the tolerance, relative to the largest, are set to zero (an
    # average of iterates has the union of their supports)
    for k in 1:p
      xmax = zero(T)
      @inbounds for j in 1:p
        xmax = max(xmax, abs(st.X[j, k]))
      end
      nk = 0
      @inbounds for j in 1:p
        x = st.X[j, k]
        if abs(x) > tol * xmax
          nk += 1
          st.idx[1][nk], st.vals[1][nk] = j, x
        end
      end
      _pushsorted!(cols[st.cols[k]], st.idx[1], st.vals[1], nk)
    end
  end
  cols, μbad
end

# stored column k has converged: move it to position m, out of the active
# block 1:m-1, and copy it to the other buffers, which then hold it whichever
# buffer is current after the next steps
function _freeze!(st::_PDHGState, k::Int, m::Int)
  _swapcols!(st, k, m)
  @inbounds for j in axes(st.X, 1)
    st.Xn[j, m], st.Yn[j, m] = st.X[j, m], st.Y[j, m]
    st.SXn[j, m], st.SYn[j, m] = st.SX[j, m], st.SY[j, m]
  end
end

# Polish stored column k: with A the support of x, T the nonzero entries of y and
# s their signs, solve S[T, A] β = eᵢ + μ c ∘ s on T and S[A, T] y = -sign(x) on A.
# If β and y have the signs of x and y and satisfy |Sβ - eᵢ| ≤ μc and |Sy| ≤ 1,
# they are optimal: store them and return true.
function _polish!(st::_PDHGState{T}, S::Matrix{T}, c::Vector{T}, k::Int) where {T}
  X, Y = st.X, st.Y
  p = size(S, 1)
  i, μ = st.cols[k], st.μ[k]
  A = findall(!iszero, view(X, :, k))
  R = findall(!iszero, view(Y, :, k))
  length(A) == length(R) || return false
  isempty(A) && return false
  F = lu!(S[R, A]; check=false)
  issuccess(F) || return false
  β = F \ [T(j == i) + μ * c[j] * sign(Y[j, k]) for j in R]
  y = F' \ [-sign(X[j, k]) for j in A]
  all(q -> sign(β[q]) == sign(X[A[q], k]), eachindex(A)) || return false
  all(t -> sign(y[t]) == sign(Y[R[t], k]), eachindex(R)) || return false
  tol = 100 * eps(T) * p
  Sβ = S[:, A] * β
  Sy = S[:, R] * y
  for j in 1:p
    abs(Sβ[j] - (j == i)) <= μ * c[j] * (1 + tol) + tol || return false
    abs(Sy[j]) <= 1 + tol || return false
  end
  @inbounds for j in 1:p
    X[j, k] = Y[j, k] = zero(T)
    st.SX[j, k] = Sβ[j]
    st.SY[j, k] = Sy[j]
  end
  X[A, k] .= β
  Y[R, k] .= y
  true
end

# PDLP's restart criteria on the KKT error of the candidate: a large enough
# decrease since the last restart, a smaller decrease that has stopped, or a long
# time since the last restart
function _restart_due(st::_PDHGState{T}, k::Int, kkt::T, iter::Int) where {T}
  kkt <= T(0.2) * st.kktr[k] || (kkt <= T(0.8) * st.kktr[k] && kkt > st.kktp[k]) || st.na[k] >= 0.36 * iter
end

# restart column k from the current iterate, with PDLP's update of the primal
# weight from the distances moved since the last restart
function _restart!(st::_PDHGState{T}, k::Int, kkt::T) where {T}
  Δx = Δy = zero(T)
  @inbounds for j in axes(st.X, 1)
    Δx += abs2(st.X[j, k] - st.Xr[j, k])
    Δy += abs2(st.Y[j, k] - st.Yr[j, k])
    st.Xr[j, k], st.Yr[j, k] = st.X[j, k], st.Y[j, k]
  end
  if Δx > eps(T) && Δy > eps(T)
    st.ω[k] = exp((log(sqrt(Δy / Δx)) + log(st.ω[k])) / 2)
  end
  st.kktr[k] = kkt
  _reset_averages!(st, k:k)
end

function _reset_averages!(st::_PDHGState, ks)
  for k in ks
    _copycol!((st.Xa, st.Ya, st.SXa, st.SYa), (st.X, st.Y, st.SX, st.SY), k)
    st.na[k] = 0
    st.wa[k] = 0
  end
end

# One step of PDHG on the active columns 1:m, with the adaptive step size of
# PDLP: a step of column k with step size η is accepted if η ≤ η̄ = ‖Δz‖²_ω /
# (2|Δy'SΔx|), with ‖Δz‖²_ω = ω‖Δx‖² + ‖Δy‖²/ω, and otherwise the column stays
# where it is; either way the next step size is
# min((1 - (n+1)^-0.3) η̄, (1 + (n+1)^-0.6) η) after n steps.
function _pdhg_step!(st::_PDHGState{T}, S::Matrix{T}, c::Vector{T}, iter::Int, m::Int, threaded::Bool) where {T}
  X, Y, SX, SY, Xn, Yn = st.X, st.Y, st.SX, st.SY, st.Xn, st.Yn
  p = size(S, 1)
  _foreach_maybe_threaded(m, threaded) do k
    τ = st.η[k] / st.ω[k]
    @inbounds @simd for j in 1:p
      v = X[j, k] - τ * SY[j, k]
      Xn[j, k] = copysign(max(abs(v) - τ, zero(T)), v)
    end
  end
  _mulS!(st.SXn, S, Xn, m, st.idx, st.vals, threaded)
  SXn = st.SXn
  _foreach_maybe_threaded(m, threaded) do k
    σ = st.η[k] * st.ω[k]
    i, μ = st.cols[k], st.μ[k]
    @inbounds @simd for j in 1:p
      w = Y[j, k] / σ + 2 * SXn[j, k] - SX[j, k] - (j == i)
      Yn[j, k] = σ * copysign(max(abs(w) - μ * c[j], zero(T)), w)
    end
  end
  _mulS!(st.SYn, S, Yn, m, st.idx, st.vals, threaded)
  SYn = st.SYn
  _foreach_maybe_threaded(m, threaded) do k
    dx = dy = inner = zero(T)
    @inbounds @simd for j in 1:p
      ΔX, ΔY = Xn[j, k] - X[j, k], Yn[j, k] - Y[j, k]
      dx += ΔX * ΔX
      dy += ΔY * ΔY
      inner += ΔY * (SXn[j, k] - SX[j, k])
    end
    η, ω = st.η[k], st.ω[k]
    η̄ = iszero(inner) ? T(Inf) : (ω * dx + dy / ω) / (2 * abs(inner))
    st.η[k] = min((1 - T(iter + 1)^T(-0.3)) * η̄, (1 + T(iter + 1)^T(-0.6)) * η)
    if η <= η̄
      # accepted: update the running averages, weighted by the step size
      st.na[k] += 1
      w = η / (st.wa[k] += η)
      Xa, Ya, SXa, SYa = st.Xa, st.Ya, st.SXa, st.SYa
      @inbounds @simd for j in 1:p
        Xa[j, k] += w * (Xn[j, k] - Xa[j, k])
        Ya[j, k] += w * (Yn[j, k] - Ya[j, k])
        SXa[j, k] += w * (SXn[j, k] - SXa[j, k])
        SYa[j, k] += w * (SYn[j, k] - SYa[j, k])
      end
    else
      # rejected: the column stays where it is
      @inbounds @simd for j in 1:p
        Xn[j, k], Yn[j, k], SXn[j, k], SYn[j, k] = X[j, k], Y[j, k], SX[j, k], SY[j, k]
      end
    end
  end
  # the new iterates become the current ones; frozen columns are in both buffers
  st.X, st.Xn = Xn, X
  st.Y, st.Yn = Yn, Y
  st.SX, st.SXn = SXn, SX
  st.SY, st.SYn = SYn, SY
  st
end

_solve_diagonal(::PDHG, S, spec, ntasks) = _solve_diagonal(ParametricSimplex(), S, spec, ntasks)
