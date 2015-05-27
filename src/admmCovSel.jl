abstract CovSelSolver
immutable ADMMSolver <: CovSelSolver
  ρ::Float64
  α::Float64
  maxiter::Int64
  abstol::Float64
  reltol::Float64
end

ADMMSolver(;ρ::Float64=1.,
           α::Float64=1.,
           maxiter::Int64=1000,
           abstol::Float64=1e-4,
           reltol::Float64=1e-2) = ADMMSolver(ρ, α, maxiter, abstol, reltol)


# inimize  trace(S*X) - log det X + lambda*||X||_1
function solve!{T<:FloatingPoint}(
    solver::ADMMSolver,
    X::StridedMatrix{T},
    Z::StridedMatrix{T},
    U::StridedMatrix{T},
    S::StridedMatrix{T}, λ::T)
  maxiter = solver.maxiter
  ρ = solver.ρ
  α = solver.α
  abstol = solver.abstol
  reltol = solver.reltol
  λρ = λ/ρ

  n = size(S, 1)
  tmpStorage = zeros(T, (n, n))
  Zold = copy(Z)

  for iter=1:maxiter
    # x-update
    for c=1:n, r=1:n
      tmpStorage[c, r] = ρ * (Z[c, r] - U[c, r]) - S[c, r]
    end
    ef = eigfact(Symmetric(tmpStorage))
    efVectors = ef[:vectors]::Array{T, 2}
    efValues = ef[:values]::Array{T, 1}
    for i=1:n
      t = efValues[i]
      efValues[i] = (t + sqrt(t^2. + 4.*ρ)) / (2.*ρ)
    end
    for c=1:n, r=1:n
      X[c, r] = zero(T)
      for i=1:n
        X[c, r] += efVectors[r,i] * efValues[i] * efVectors[c, i]
      end
    end

    # z-update with relaxation
    copy!(Zold, Z)
    for c=1:n, r=1:n
      tmpStorage[r, c] = α*X[r, c] + (one(T)-α)*Z[r, c]
      Z[r, c] = shrink(tmpStorage[r, c] + U[r, c], λρ)
    end

    for c=1:n, r=1:n
      U[r,c] = U[r,c] + tmpStorage[r, c] - Z[r, c];
    end

    # check convergence
    r_norm = _normdiff(X, Z)
    s_norm = _normdiff(Z, Zold) * sqrt(ρ)
    eps_pri = n*abstol + reltol * max( vecnorm(X), vecnorm(Z) )
    eps_dual = n*abstol + reltol * ρ * vecnorm(U)
    if r_norm < eps_pri && s_norm < eps_dual
      break
    end
  end
  Z
end
