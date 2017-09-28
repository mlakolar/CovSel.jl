struct ADMMOptions
  ρ::Float64
  α::Float64
  maxiter::Int64
  abstol::Float64
  reltol::Float64
end

ADMMOptions(;ρ::Float64=1.,
           α::Float64=1.,
           maxiter::Int64=1000,
           abstol::Float64=1e-4,
           reltol::Float64=1e-2) = ADMMOptions(ρ, α, maxiter, abstol, reltol)



#########################################################
# minimize  trace(S*X) - log det X + lambda*||X||_1
#########################################################

function covsel!{T<:AbstractFloat}(
  X::StridedMatrix{T},
  Z::StridedMatrix{T},
  U::StridedMatrix{T},
  Σ::StridedMatrix{T},
  λ::T;
  options::ADMMOptions = ADMMOptions(),
  penalize_diag::Bool=true
  )

  maxiter = options.maxiter
  ρ = options.ρ
  α = options.α
  abstol = options.abstol
  reltol = options.reltol
  γ = one(T) / ρ
  λρ = λ * γ

  p = size(Σ, 1)
  g = ProxGaussLikelihood(Symmetric(Σ))

  tmpStorage = zeros(T, (p, p))
  Zold = copy(Z)

  for iter=1:maxiter
    # x-update
    @. tmpStorage = Z - U
    prox!(g, X, tmpStorage, γ)

    # z-update with relaxation
    copy!(Zold, Z)
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
    r_norm = norm_diff(X, Z)
    s_norm = norm_diff(Z, Zold) * sqrt(ρ)
    eps_pri = p*abstol + reltol * max( vecnorm(X), vecnorm(Z) )
    eps_dual = p*abstol + reltol * ρ * vecnorm(U)
    if r_norm < eps_pri && s_norm < eps_dual
      break
    end
  end
  Z
end


function covselpath{T<:AbstractFloat}(S::StridedMatrix{T},
                    λarr;
                    options::ADMMOptions = ADMMOptions(),
                    penalize_diag::Bool=true,
                    verbose::Bool=false)
  p = size(S, 1)

  solutionpath = Array(Array{Float64, 2}, length(λarr))
  X = zeros(p, p)
  Z = zeros(p, p)
  U = zeros(p, p)

  for i=1:length(λarr)
    if verbose
      @printf("lambda = %d/%d\n", i, length(λarr))
    end
    covsel!(X, Z, U, S, λarr[i]; options=options, penalize_diag=penalize_diag)
    solutionpath[i] = copy(Z)
  end
  solutionpath
end
