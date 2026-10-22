"""
    ADMMOptions(; ρ=1.0, α=1.5, maxiter=10_000, abstol=1e-5, reltol=1e-3, adaptive=true)

Options for the ADMM solvers (Boyd et al., 2011):

- `ρ`: the penalty parameter, or its starting value when `adaptive` is true
- `α`: over-relaxation parameter (Section 3.4.3); values between 1.5 and 1.8
  usually speed up convergence, and `α = 1` gives plain ADMM
- `maxiter`: maximum number of iterations; reaching it without meeting the
  tolerances gives a warning
- `abstol`, `reltol`: absolute and relative tolerances of the primal and dual
  residuals (Section 3.3.1). The defaults aim at estimates within about 1% of
  the solution, relative to its largest entry; for more accuracy use, for
  example, `abstol=1e-6, reltol=1e-4`
- `adaptive`: adjust `ρ` during the iterations to balance the primal and dual
  residuals (Section 3.4.1)

`ρ` and `abstol` are relative to the scale of the data, the average variance
s = tr(S)/p for the graphical lasso: the solvers use the penalty parameter
`ρ s²` and the absolute tolerances `abstol / s` for the primal residual and
`abstol s` for the dual residual, which have the units of the precision and the
covariance matrices. Multiplying the data by a constant therefore rescales all
the iterates and leaves the number of iterations unchanged. For data with unit
variances these are the usual parameters.
"""
struct ADMMOptions
  ρ::Float64
  α::Float64
  maxiter::Int64
  abstol::Float64
  reltol::Float64
  adaptive::Bool
end

ADMMOptions(;ρ::Real=1.,
           α::Real=1.5,
           maxiter::Integer=10_000,
           abstol::Real=1e-5,
           reltol::Real=1e-3,
           adaptive::Bool=true) = ADMMOptions(ρ, α, maxiter, abstol, reltol, adaptive)

_withρ(o::ADMMOptions, ρ) = ADMMOptions(ρ, o.α, o.maxiter, o.abstol, o.reltol, o.adaptive)

"""
    ADMMInfo()

Filled in by the ADMM solvers when passed as the `info` keyword: the number of
`iterations`, whether the tolerances were met (`converged`), the final penalty
parameter `ρ`, relative to the scale of the data as in [`ADMMOptions`](@ref),
and the final primal and dual residuals `r_norm` and `s_norm`.

The scaled dual variables are scaled by `ρ`, so to warm start from the iterates
of a previous call, also pass its final `ρ`, as in `ADMMOptions(; ρ=info.ρ)`.
"""
mutable struct ADMMInfo
  iterations::Int
  converged::Bool
  ρ::Float64
  r_norm::Float64
  s_norm::Float64
end

ADMMInfo() = ADMMInfo(0, false, NaN, NaN, NaN)

# Residual balancing (Boyd et al., 2011, Section 3.4.1), with each residual
# measured relative to its tolerance: the factor by which to multiply ρ when one
# of them is more than ten times the other. The caller divides its scaled dual
# variables by the same factor. ρ is left fixed after _maxPenaltyChanges
# changes, which keeps the convergence guarantee of ADMM with a fixed ρ.
const _maxPenaltyChanges = 50

function _penaltyFactor(r_norm, eps_pri, s_norm, eps_dual, changes)
  changes >= _maxPenaltyChanges && return 1
  r, s = r_norm / eps_pri, s_norm / eps_dual
  r > 10 * s ? 2 : s > 10 * r ? 1 // 2 : 1
end

# the scale of the data: the average variance, or 1 if that is not positive
_dataScale(v) = v > 0 && isfinite(v) ? v : one(v)

# record the final state in info, with ρ relative to ρscale, and warn if the
# tolerances were not met
function _finish!(info::ADMMInfo, solver, iter, converged, ρ, ρscale, r_norm, s_norm)
  info.iterations, info.converged, info.ρ = iter, converged, ρ / ρscale
  info.r_norm, info.s_norm = r_norm, s_norm
  converged || @warn "$solver stopped after $iter iterations without meeting the tolerances" r_norm s_norm
  info
end


"""
    covsel!(X, Z, U, Σ, λ; options=ADMMOptions(), penalize_diag=true, info=ADMMInfo())

Graphical lasso by ADMM:

    minimize  tr(Σ Ω) - log det Ω + λ ‖Ω‖₁

where the ℓ1 norm is over all entries, or only the off-diagonal ones when
`penalize_diag=false`. `X`, `Z` and `U` are the ADMM iterates (the primal
iterate, its sparse copy, and the scaled dual variable); pass zeros, or the
iterates of a previous call together with its final `info.ρ` to warm start.
Returns `Z`, the sparse estimate; see [`ADMMInfo`](@ref) for `info`.
"""
function covsel!(
  X::StridedMatrix{T},
  Z::StridedMatrix{T},
  U::StridedMatrix{T},
  Σ::StridedMatrix{T},
  λ::Real;
  options::ADMMOptions = ADMMOptions(),
  penalize_diag::Bool=true,
  info::ADMMInfo=ADMMInfo()
  ) where {T<:AbstractFloat}

  maxiter = options.maxiter
  α = T(options.α)
  reltol = options.reltol

  p = size(Σ, 1)
  g = ProxGaussLikelihood(Symmetric(Σ))

  # ρ has the units of Σ², the primal residual those of Ω and the dual one those of Σ
  s = _dataScale(tr(Σ) / p)
  ρ = T(options.ρ * s^2)
  abstol_pri, abstol_dual = options.abstol / s, options.abstol * s

  tmpStorage = zeros(T, (p, p))
  Zold = copy(Z)

  converged = false
  iter = changes = 0
  r_norm = s_norm = T(NaN)
  while iter < maxiter
    iter += 1
    γ = one(T) / ρ
    λρ = T(λ) * γ

    # x-update
    @. tmpStorage = Z - U
    prox!(g, X, tmpStorage, γ)

    # z-update with relaxation
    copyto!(Zold, Z)
    @. tmpStorage = α*X + (one(T)-α)*Z + U
    @inbounds for c=1:p
      for r=1:c-1
        t = shrink(tmpStorage[r, c], λρ)
        Z[c, r] = t
        Z[r, c] = t
      end
      Z[c, c] = penalize_diag ? shrink(tmpStorage[c, c], λρ) : tmpStorage[c, c]
    end

    # u-update
    @. U = tmpStorage - Z

    # check convergence
    r_norm = norm_diff(X, Z, 2)
    s_norm = ρ * norm_diff(Z, Zold, 2)
    eps_pri = p*abstol_pri + reltol * max( norm(X), norm(Z) )
    eps_dual = p*abstol_dual + reltol * ρ * norm(U)
    if r_norm < eps_pri && s_norm < eps_dual
      converged = true
      break
    end

    # update ρ, and rescale the scaled dual variable
    if options.adaptive
      f = _penaltyFactor(r_norm, eps_pri, s_norm, eps_dual, changes)
      if f != 1
        ρ *= f
        U ./= f
        changes += 1
      end
    end
  end
  _finish!(info, "covsel!", iter, converged, ρ, s^2, r_norm, s_norm)
  Z
end


"""
    covselpath(S, λarr; options=ADMMOptions(), penalize_diag=true, verbose=false)

Solve [`covsel!`](@ref) for each `λ` in `λarr`, warm starting from the
previous solution and its penalty parameter. Returns a vector of the estimated
precision matrices.
"""
function covselpath(S::StridedMatrix{T},
                    λarr;
                    options::ADMMOptions = ADMMOptions(),
                    penalize_diag::Bool=true,
                    verbose::Bool=false) where {T<:AbstractFloat}
  p = size(S, 1)

  solutionpath = Vector{Matrix{T}}(undef, length(λarr))
  X = zeros(T, p, p)
  Z = zeros(T, p, p)
  U = zeros(T, p, p)

  info = ADMMInfo()
  for i=1:length(λarr)
    if verbose
      println("lambda = $i/$(length(λarr))")
    end
    opt = i == 1 ? options : _withρ(options, info.ρ)
    covsel!(X, Z, U, S, λarr[i]; options=opt, penalize_diag=penalize_diag, info=info)
    solutionpath[i] = copy(Z)
  end
  solutionpath
end
