"""
    ADMMOptions(; ρ=1.0, α=1.0, maxiter=1000, abstol=1e-4, reltol=1e-2)

Options for the ADMM solvers: penalty parameter `ρ`, over-relaxation parameter
`α` (between 1 and 1.8 speeds things up), maximum number of iterations, and the
absolute and relative tolerances of the primal and dual residuals (Boyd et al.,
2011, Section 3.3.1).
"""
struct ADMMOptions
  ρ::Float64
  α::Float64
  maxiter::Int64
  abstol::Float64
  reltol::Float64
end

ADMMOptions(;ρ::Real=1.,
           α::Real=1.,
           maxiter::Integer=1000,
           abstol::Real=1e-4,
           reltol::Real=1e-2) = ADMMOptions(ρ, α, maxiter, abstol, reltol)



"""
    covsel!(X, Z, U, Σ, λ; options=ADMMOptions(), penalize_diag=true)

Graphical lasso by ADMM:

    minimize  tr(Σ Ω) - log det Ω + λ ‖Ω‖₁

where the ℓ1 norm is over all entries, or only the off-diagonal ones when
`penalize_diag=false`. `X`, `Z` and `U` are the ADMM iterates (the primal
iterate, its sparse copy, and the scaled dual variable); pass zeros, or the
iterates of a previous call to warm start. Returns `Z`, the sparse estimate.
"""
function covsel!(
  X::StridedMatrix{T},
  Z::StridedMatrix{T},
  U::StridedMatrix{T},
  Σ::StridedMatrix{T},
  λ::Real;
  options::ADMMOptions = ADMMOptions(),
  penalize_diag::Bool=true
  ) where {T<:AbstractFloat}

  maxiter = options.maxiter
  ρ = options.ρ
  α = options.α
  abstol = options.abstol
  reltol = options.reltol
  γ = T(1 / ρ)
  λρ = T(λ * γ)

  p = size(Σ, 1)
  g = ProxGaussLikelihood(Symmetric(Σ))

  tmpStorage = zeros(T, (p, p))
  Zold = copy(Z)

  for iter=1:maxiter
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
    s_norm = norm_diff(Z, Zold, 2) * sqrt(ρ)
    eps_pri = p*abstol + reltol * max( norm(X), norm(Z) )
    eps_dual = p*abstol + reltol * ρ * norm(U)
    if r_norm < eps_pri && s_norm < eps_dual
      break
    end
  end
  Z
end


"""
    covselpath(S, λarr; options=ADMMOptions(), penalize_diag=true, verbose=false)

Solve [`covsel!`](@ref) for each `λ` in `λarr`, warm starting from the
previous solution. Returns a vector of the estimated precision matrices.
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

  for i=1:length(λarr)
    if verbose
      @printf("lambda = %d/%d\n", i, length(λarr))
    end
    covsel!(X, Z, U, S, λarr[i]; options=options, penalize_diag=penalize_diag)
    solutionpath[i] = copy(Z)
  end
  solutionpath
end
