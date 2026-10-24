####################################
#
#  K precision matrices with a shared sparsity pattern: the group graphical
#  lasso, and graphs that vary over time
#
####################################

"""
    groupGraphicalLasso(S, λ; weights=fill(1/K, K), penalize_diag=true, options=ADMMOptions(), info=ADMMInfo())
    groupGraphicalLasso!(Z, U, S, λ; weights=fill(1/K, K), penalize_diag=true, options=ADMMOptions(), info=ADMMInfo())

K precision matrices that share their sparsity pattern, by ADMM:

    minimize  Σₖ wₖ (tr(Sₖ Ωₖ) - log det Ωₖ) + λ Σᵢⱼ ‖(Ω₁[i,j], …, Ω_K[i,j])‖₂

for the covariance matrices `S₁, …, S_K` of K classes or time points, given
as a `p × p × K` array or a vector of matrices, and positive weights `wₖ`,
such as the shares `nₖ/n` of the observations. The ℓ2 norm of each entry
across the K matrices, the group penalty of the group graphical lasso
(Danaher, Wang and Witten, 2014), sets an entry to zero in all the matrices
at once. With `penalize_diag=false` the diagonal is not penalized. With K = 1
this is the graphical lasso of [`covsel!`](@ref). With equal weights `1/K`, an
entry equal to `a` in all K matrices has the penalty `λ √K |a|`, √K times its
average over K separate graphical lassos with the same `λ`; `λ/√K` penalizes
it as they do.

Returns the `p × p × K` array of the estimates `Ωₖ = Z[:, :, k]`. In the in-place
version `Z` and `U` are the ADMM iterates (the estimates and the scaled dual
variables), `p × p × K` arrays; pass zeros, or the iterates of a previous call
together with its final `info.ρ` to warm start. See [`ADMMInfo`](@ref) for
`info`.

This is ProximalSolvers' `ADMMSolver` on ProximalBase's `SliceSum` of the K
weighted `LogDetLoss`es and the penalty `FiberSum(NormL2(λ))`. The options are
scaled by the data as in [`covsel!`](@ref), with `s` the average of the
`tr(Sₖ)/p`, and by the average weight `w̄`: the penalty parameter is
`ρ s² w̄`, so that each slice sees the penalty parameter of the graphical lasso
when the weights are equal, and the absolute tolerance of the dual residual is
`abstol s w̄`.

See [`tvCovSel`](@ref) for graphs that vary over time.
"""
function groupGraphicalLasso!(
  Z::Array{T, 3},
  U::Array{T, 3},
  S::Union{AbstractArray{<:Real, 3}, AbstractVector{<:AbstractMatrix}},
  λ::Real;
  weights::AbstractVector{<:Real} = fill(one(T) / _nslices(S), _nslices(S)),
  penalize_diag::Bool = true,
  options::ADMMOptions = ADMMOptions(),
  info::ADMMInfo = ADMMInfo()
  ) where {T<:AbstractFloat}

  K = _nslices(S)
  p = size(_covslice(S, 1), 1)
  size(Z) == size(U) == (p, p, K) || throw(DimensionMismatch("Z and U must be p × p × K arrays, here $p × $p × $K"))
  all(k -> size(_covslice(S, k)) == (p, p), 1:K) || throw(DimensionMismatch("the covariance matrices must be p × p"))
  length(weights) == K || throw(DimensionMismatch("$(length(weights)) weights for $K covariance matrices"))
  all(w -> w > 0, weights) || throw(ArgumentError("the weights must be positive"))
  λ >= 0 || throw(ArgumentError("λ must be nonnegative, got $λ"))

  f = SliceSum([LogDetLoss(_covslice(S, k), T(weights[k])) for k in 1:K])
  g = FiberSum(NormL2(T(λ)); diagonal = penalize_diag ? NormL2(T(λ)) : nothing, symmetric = true)

  # ρ has the units of the weighted Σ², the primal residual those of Ω and the
  # dual one those of the weighted Σ
  s = _dataScale(sum(k -> tr(_covslice(S, k)), 1:K) / (p * K))
  w̄ = sum(weights) / K
  method = ADMMSolver(; ρ = T(options.ρ * s^2 * w̄), α = T(options.α), adaptive = options.adaptive,
                      abstol = T(options.abstol / s), abstol_dual = T(options.abstol * s * w̄), reltol = T(options.reltol))
  r = solve!(Z, f, g, method, ProximalOptions(; maxiter = options.maxiter); u = U)
  _finish!(info, "groupGraphicalLasso!", r.iterations, r.converged, r.ρ, s^2 * w̄, r.r_norm, r.s_norm)
  Z
end

function groupGraphicalLasso(S::Union{AbstractArray{<:Real, 3}, AbstractVector{<:AbstractMatrix}}, λ::Real; kwargs...)
  K = _nslices(S)
  p = size(_covslice(S, 1), 1)
  T = float(eltype(_covslice(S, 1)))
  groupGraphicalLasso!(zeros(T, p, p, K), zeros(T, p, p, K), S, λ; kwargs...)
end

_nslices(S::AbstractArray{<:Real, 3}) = size(S, 3)
_nslices(S::AbstractVector{<:AbstractMatrix}) = length(S)
_covslice(S::AbstractArray{<:Real, 3}, k) = view(S, :, :, k)
_covslice(S::AbstractVector{<:AbstractMatrix}, k) = S[k]


"""
    kernelCovariances(X, t, τ, h; kernel=:epanechnikov, center=true)

The covariance matrices at the time points `τ₁, …, τ_K` of the observations
`xᵢ`, the rows of `X`, made at the times `tᵢ`, by kernel smoothing (Zhou,
Lafferty and Wasserman, 2010):

    Sₖ = Σᵢ wᵢₖ (xᵢ - x̄ₖ)(xᵢ - x̄ₖ)' / Σᵢ wᵢₖ,   wᵢₖ = K((tᵢ - τₖ)/h)

with `x̄ₖ` the weighted mean of the observations, or zero with
`center=false`. The kernel `K` is `:epanechnikov`, `3(1 - u²)/4` for `|u| ≤
1`, `:gaussian`, or a function. Returns the `p × p × K` array of the `Sₖ`.
"""
function kernelCovariances(X::AbstractMatrix{<:Real}, t::AbstractVector{<:Real}, τ::AbstractVector{<:Real}, h::Real;
                           kernel = :epanechnikov, center::Bool = true)
  n, p = size(X)
  length(t) == n || throw(DimensionMismatch("X has $n rows and t has length $(length(t))"))
  h > 0 || throw(ArgumentError("the bandwidth h must be positive, got $h"))
  kf = _kernel(kernel)
  T = float(eltype(X))
  S = zeros(T, p, p, length(τ))
  w = zeros(T, n)
  Xc = zeros(T, n, p)
  xbar = zeros(T, p)
  for (k, τk) in enumerate(τ)
    w .= kf.((t .- τk) ./ h)
    W = sum(w)
    W > 0 || throw(ArgumentError("no observation has a positive kernel weight at τ = $τk; increase the bandwidth h"))
    center ? mul!(xbar, transpose(X), w, inv(W), false) : fill!(xbar, zero(T))
    Xc .= (X .- transpose(xbar)) .* sqrt.(w)
    Sk = view(S, :, :, k)
    mul!(Sk, transpose(Xc), Xc, inv(W), false)
    LinearAlgebra.copytri!(Sk, 'U')        # exactly symmetric
  end
  S
end

_kernel(k::Function) = k
function _kernel(k::Symbol)
  k === :epanechnikov && return u -> abs(u) <= 1 ? 3 * (1 - u^2) / 4 : zero(u)
  k === :gaussian && return u -> exp(-u^2 / 2) / sqrt(2π)
  throw(ArgumentError("unknown kernel :$k; use :epanechnikov, :gaussian or a function"))
end


"""
    tvCovSel(X, t, τ, h, λ; kernel=:epanechnikov, center=true, weights=fill(1/K, K), penalize_diag=true,
             options=ADMMOptions(), info=ADMMInfo())

Time-varying graphical models: the precision matrices at the time points
`τ₁, …, τ_K` of the observations `xᵢ` (the rows of `X`) made at the times
`tᵢ`. It is the [`groupGraphicalLasso`](@ref) of the covariance matrices that
[`kernelCovariances`](@ref) estimates with the bandwidth `h`: each time point
has the kernel-smoothed likelihood of Zhou, Lafferty and Wasserman (2010), and
the group penalty makes the K graphs share their edges. Returns the `p × p × K`
array of the estimates.
"""
function tvCovSel(X::AbstractMatrix{<:Real}, t::AbstractVector{<:Real}, τ::AbstractVector{<:Real}, h::Real, λ::Real;
                  kernel = :epanechnikov, center::Bool = true, kwargs...)
  groupGraphicalLasso(kernelCovariances(X, t, τ, h; kernel, center), λ; kwargs...)
end
