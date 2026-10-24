# Automatic-step IHT uses a single product-space step; the explicit-η methods
# retain their alternating updates and legacy thresholding unchanged.
const _IHTGradientMethod = Union{ProximalSolvers.ProximalGradientDescent,
  ProximalSolvers.AccProxGradDescent, ProximalSolvers.MonotoneAccProxGradDescent}

function _iht_covariance(Σ::AbstractMatrix)
  p = size(Σ, 1)
  size(Σ, 2) == p || throw(DimensionMismatch("the covariance must be square"))
  p > 0 && issymmetric(Σ) && all(isfinite, Σ) ||
    throw(ArgumentError("the covariance must be nonempty, finite and symmetric"))
  p
end

function _iht_start(::Type{T}, p, s, r, init; covariance = nothing) where {T}
  s >= 0 || throw(ArgumentError("s must be nonnegative"))
  0 <= r <= p || throw(ArgumentError("r must be between zero and p"))
  if init === nothing
    S, U = SymmetricSparseIterate(T, p), zeros(T, p, r)
    for i in 1:p
      if covariance !== nothing
        covariance[i, i] > 0 || throw(ArgumentError("automatic initialization needs positive covariance diagonal entries; supply init=(S,U)"))
        S[i, i] = inv(covariance[i, i])
      end
      i <= r && (U[i, i] = sqrt(covariance === nothing ? one(T) : S[i, i]) / T(10))
    end
  else
    length(init) == 2 || throw(ArgumentError("init must be (S,U)"))
    S0, U0 = init
    size(S0) == (p, p) && size(U0) == (p, r) || throw(DimensionMismatch("init must have shapes (p,p) and (p,r)"))
    issymmetric(S0) || throw(ArgumentError("the initial sparse matrix must be symmetric"))
    S, U = SymmetricSparseIterate(Matrix{T}(S0)), Matrix{T}(U0)
  end
  all(isfinite, S) && all(isfinite, U) || throw(ArgumentError("the initial blocks must be finite"))
  S, U
end

# Dense matrix buffers are reused across value, gradient and line-search calls.
struct _LatentLogdet{T} <: AbstractFunction
  Σ::Matrix{T}
  Θ::Matrix{T}
  factor::Matrix{T}
  inverse::Matrix{T}
end
function _LatentLogdet(Σ::AbstractMatrix{T}) where {T<:AbstractFloat}
  p = _iht_covariance(Σ)
  _LatentLogdet(Matrix(Σ), zeros(T, p, p), zeros(T, p, p), zeros(T, p, p))
end
struct _LatentDTrace{T,F} <: AbstractFunction
  loss::F
  Θ::Matrix{T}
  raw::Matrix{T}
end
function _LatentDTrace(Σx::Symmetric{T}, Σy::Symmetric{T}) where {T<:AbstractFloat}
  p = _iht_covariance(Σx)
  _iht_covariance(Σy) == p || throw(DimensionMismatch())
  _LatentDTrace(directDifferenceLoss(Σx, Σy), zeros(T, p, p), zeros(T, p, p))
end
ProximalBase.issmooth(::Type{<:_LatentLogdet}) = true
ProximalBase.issmooth(::Type{<:_LatentDTrace}) = true

function _iht_theta!(Θ, x)
  _densify!(Θ, x[1])
  mul!(Θ, x[2], transpose(x[2]), one(eltype(Θ)), one(eltype(Θ)))
end
function _iht_factor!(f::_LatentLogdet, x)
  _iht_theta!(f.Θ, x)
  all(isfinite, f.Θ) || return nothing
  F = cholesky!(Symmetric(copyto!(f.factor, f.Θ)); check = false)
  issuccess(F) ? F : nothing
end
function (f::_LatentLogdet{T})(x::Tuple) where {T}
  F = _iht_factor!(f, x)
  F === nothing ? T(Inf) : dot(f.Σ, f.Θ) - logdet(F)
end
function ProximalBase.value_and_gradient!(G::Tuple, f::_LatentLogdet{T}, x::Tuple) where {T}
  F = _iht_factor!(f, x)
  F === nothing && return T(Inf) # no inverse or gradient outside the cone
  fill!(f.inverse, zero(T))
  for i in axes(f.inverse, 1)
    f.inverse[i, i] = one(T)
  end
  ldiv!(F, f.inverse)
  G[1] .= f.Σ .- f.inverse
  mul!(G[2], G[1], x[2], T(2), zero(T))
  dot(f.Σ, f.Θ) - logdet(F)
end
(f::_LatentDTrace)(x::Tuple) = f.loss(_iht_theta!(f.Θ, x))
function ProximalBase.value_and_gradient!(G::Tuple, f::_LatentDTrace{T}, x::Tuple) where {T}
  value = value_and_gradient!(f.raw, f.loss, _iht_theta!(f.Θ, x))
  G[1] .= (f.raw .+ transpose(f.raw)) ./ T(2)
  mul!(G[2], G[1], x[2], T(2), zero(T))
  value
end

# Collect every accepted outer iterate, so trace display runs at every iteration.
# A legacy callback's return value is ignored; options.callback can stop solve!.
function _automatic_iht!(x, f, g, method, options::ProximalOptions, accepted)
  method isa _IHTGradientMethod || throw(ArgumentError("automatic IHT needs ProximalGradientDescent, AccProxGradDescent or MonotoneAccProxGradDescent"))
  isfinite(g(x)) || throw(ArgumentError("the initial sparse matrix exceeds the sparsity budget"))
  initial = f(x)
  isfinite(initial) || throw(ArgumentError("the initial loss must be finite; for latent log-det use S + U*U' positive definite"))
  fvals = typeof(initial)[initial]
  cb = function (record)
    state = options.store_trace ? last(record) : record
    push!(fvals, state.fg_x)
    accepted === nothing || accepted()
    options.callback === nothing || state.iteration % options.printEvery != 0 || return options.callback(record)
    false
  end
  settings = (; (key => getfield(options, key) for key in fieldnames(typeof(options)))...)
  result = solve!(x, f, g, method, ProximalOptions(; settings..., callback = cb, printEvery = 1))
  fvals, result.iterations
end

"""
    differencePrecisionIHT(Σx, Σy, s; method=ProximalGradientDescent(), options=ProximalOptions(), init=nothing)

IHT with an automatically backtracked step and ProximalBase's Frobenius
projection onto at most `s` lower-triangle coordinates, including the diagonal.
`init` optionally supplies a symmetric starting matrix. Returns `(Δ, fvals)`,
with the initial objective and every accepted iteration in `fvals`.
The overload with positional `η` retains its explicit step and legacy projection.
"""
function differencePrecisionIHT(Σx::Symmetric{T}, Σy::Symmetric{T}, s::Int;
  method = ProximalSolvers.ProximalGradientDescent(), options::ProximalOptions = ProximalOptions(), init = nothing) where {T<:AbstractFloat}
  p = _iht_covariance(Σx)
  _iht_covariance(Σy) == p || throw(DimensionMismatch())
  Δ, _ = _iht_start(T, p, s, 0, init === nothing ? nothing : (init, zeros(T, p, 0)))
  fvals, _ = _automatic_iht!(Δ, directDifferenceLoss(Σx, Σy), IndBallL0(s), method, options, nothing)
  Δ, fvals
end

"""
    covselLatentIHT!(Σ, s, r; method=ProximalGradientDescent(), options=ProximalOptions(), init=nothing, callback=nothing)

Automatic-step IHT for `tr(ΣΘ)-logdet(Θ)`, `Θ=S+UU'`, with at most `s`
off-diagonal pairs in `S`. Returns `(S,L,U,fvals,iter)`, with `S` a
`SymmetricSparseIterate`, `L=UU'` and an objective for the initial point and each
accepted iteration. `callback(Σ,S,L)` observes only accepted iterates; its
return value is ignored. `options.callback` keeps ProximalSolvers' convention.
`init=(S,U)` is copied and must be feasible. By default `S` is positive diagonal
with inverse covariance variances and `U` has small deterministic orthogonal
columns. This nonconvex initialization need not yield a global minimum.
The overload with positional `ηS,ηU` keeps its legacy alternating updates.
"""
function covselLatentIHT!(Σ::AbstractMatrix{T}, s::Int, r::Int;
  method = ProximalSolvers.ProximalGradientDescent(), options::ProximalOptions = ProximalOptions(), init = nothing, callback = nothing) where {T<:AbstractFloat}
  p = _iht_covariance(Σ)
  S, U = _iht_start(T, p, s, r, init; covariance = Σ)
  L = zeros(T, p, p)
  accepted = () -> (mul!(L, U, transpose(U)); callback === nothing || callback(Σ, S, L); nothing)
  fvals, iter = _automatic_iht!((S, U), _LatentLogdet(Σ), SeparableSum(IndBallL0(s; diagonal = false), ZeroFunction()), method, options, accepted)
  mul!(L, U, transpose(U))
  S, L, U, fvals, iter
end

"""
    differenceLatentPrecisionIHT!(Δ, L, U, Σx, Σy, s; method=ProximalGradientDescent(), options=ProximalOptions(), callback=nothing)
    differenceLatentPrecisionIHT(Σx, Σy, s, r; init=nothing, kwargs...)

Automatic-step IHT for the direct difference loss at `Δ+UU'`. The in-place
method starts from and updates `Δ,U`, and overwrites `L=UU'`. Its sparsity budget
counts diagonal coordinates. Returns `(Δ,L,U,fvals,iter)` with initial and every
accepted objective. `callback(Δ,L)` has its legacy ignored return value; stopping
callbacks belong in `options`. The convenience method copies `init=(Δ,U)`, or
starts from zero `Δ` and a small deterministic orthogonal `U`. The explicit-η
overloads retain their original alternating updates and initialization.
"""
function differenceLatentPrecisionIHT!(Δ::SymmetricSparseIterate{T}, L::StridedMatrix{T}, U::StridedMatrix{T},
  Σx::Symmetric{T}, Σy::Symmetric{T}, s::Int;
  method = ProximalSolvers.ProximalGradientDescent(), options::ProximalOptions = ProximalOptions(), callback = nothing) where {T<:AbstractFloat}
  p = _iht_covariance(Σx)
  _iht_covariance(Σy) == p && size(Δ) == size(L) == (p, p) && size(U, 1) == p || throw(DimensionMismatch())
  s >= 0 && size(U, 2) <= p || throw(ArgumentError("s must be nonnegative and the factor rank at most p"))
  all(isfinite, Δ) && all(isfinite, U) || throw(ArgumentError("the initial blocks must be finite"))
  accepted = () -> (mul!(L, U, transpose(U)); callback === nothing || callback(Δ, L); nothing)
  fvals, iter = _automatic_iht!((Δ, U), _LatentDTrace(Σx, Σy), SeparableSum(IndBallL0(s), ZeroFunction()), method, options, accepted)
  mul!(L, U, transpose(U))
  Δ, L, U, fvals, iter
end
function differenceLatentPrecisionIHT(Σx::Symmetric{T}, Σy::Symmetric{T}, s::Int, r::Int;
  init = nothing, kwargs...) where {T<:AbstractFloat}
  p = _iht_covariance(Σx)
  Δ, U = _iht_start(T, p, s, r, init)
  differenceLatentPrecisionIHT!(Δ, zeros(T, p, p), U, Σx, Σy, s; kwargs...)
end
