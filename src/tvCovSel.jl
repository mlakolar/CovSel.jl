#
# # inimize  trace(S*X) - log det X   subject to support(X) ⊆ non_zero_set
# function covsel_refit!{T<:AbstractFloat}(
#     X::StridedMatrix{T},
#     Z::StridedMatrix{T},
#     U::StridedMatrix{T},
#     S::StridedMatrix{T},
#     non_zero_set::Vector{Int64};
#     options::ADMMOptions = ADMMOptions()
#     )
#
#   maxiter = options.maxiter
#   ρ = options.ρ
#   α = options.α
#   abstol = options.abstol
#   reltol = options.reltol
#
#   p = size(S, 1)
#   tmpStorage = zeros(T, (p, p))
#   Zold = copy(Z)
#
#   for iter=1:maxiter
#     # x-update
#     @simd for i=1:length(tmpStorage)
#       @inbounds tmpStorage[i] = ρ * (Z[i] - U[i]) - S[i]
#     end
#     ef = eigfact(Symmetric(tmpStorage))
#     efVectors = ef[:vectors]::Array{T, 2}
#     efValues = ef[:values]::Array{T, 1}
#     @simd for i=1:p
#       @inbounds t = efValues[i]
#       @inbounds efValues[i] = (t + sqrt(t^2. + 4.*ρ)) / (2.*ρ)
#     end
#     @inbounds for c=1:p, r=1:p
#       X[r, c] = zero(T)
#       for i=1:p
#         X[r, c] = X[r, c] + efVectors[r,i] * efValues[i] * efVectors[c, i]
#       end
#     end
#
#     # z-update with relaxation
#     copy!(Zold, Z)
#     fill!(Z, zero(T))
#     for i in non_zero_set
#       @inbounds Z[i] = α*X[i] + (one(T)-α)*Z[i] + U[i]
#     end
#
#     # u-update
#     @simd for i in eachindex(X)
#       @inbounds U[i] = U[i] + X[i] - Z[i]
#     end
#
#     # check convergence
#     r_norm = _normdiff(X, Z)
#     s_norm = _normdiff(Z, Zold) * sqrt(ρ)
#     eps_pri = p*abstol + reltol * max( vecnorm(X), vecnorm(Z) )
#     eps_dual = p*abstol + reltol * ρ * vecnorm(U)
#     if r_norm < eps_pri && s_norm < eps_dual
#       break
#     end
#   end
#   Z
# end

# # inimize  sum_i trace(S_i*X_i) - log det X_i + lambda * sum_ab (sum_i X_i,ab ^2)
# function solve!{T<:FloatingPoint}(
#     solver::ADMMSolver,
#     X::Array{T, 3},
#     Z::Array{T, 3},
#     U::Array{T, 3},
#     S::Array{T, 3},
#     λ::T;
#     penalize_diag::Bool=true
#     )
#   maxiter = solver.maxiter
#   ρ = solver.ρ
#   α = solver.α
#   abstol = solver.abstol
#   reltol = solver.reltol
#   λρ = λ/ρ

#   K = size(S, 3)
#   p = size(S, 1)

#   tmpStorage = zeros(T, (p,p,K))
#   Zold = copy(Z)

#   for iter=1:maxiter

#     # x-update
#     @simd for i in eachindex(X)
#       tmpStorage[i] = ρ * (Z[i] - U[i]) - S[i]
#     end
#     for k=1:K
#       ef = eigfact(Symmetric(sub(tmpStorage, :, :, k)))
#       efVectors = ef[:vectors]::Array{T, 2}
#       efValues = ef[:values]::Array{T, 1}
#       for i=1:p
#         t = efValues[i]
#         efValues[i] = (t + sqrt(t^2. + 4.*ρ)) / (2.*ρ)
#       end
#       for c=1:p, r=1:p
#         X[r, c, k] = zero(T)
#         @simd for i=1:p
#           X[r, c, k] = X[r, c, k] + efVectors[r,i] * efValues[i] * efVectors[c, i]
#         end
#       end
#     end

#     # z-update with relaxation
#     copy!(Zold, Z)
#     @simd @inbounds for i in eachindex(tmpStorage)
#       tmpStorage[i] = α*X[i] + (one(T)-α)*Z[i] + U[i]
#     end
#     for c=1:p
#       for r=1:c-1
#         tmpSlice = slice(tmpStorage, r, c, :)
#         t = max(one(T) - λρ / vecnorm(tmpSlice), zero(T))
#         if t > zero(T)
#           @simd @inbounds for k=1:K
#             Z[r,c,k] = t * tmpSlice[k]
#             Z[c,r,k] = t * tmpSlice[k]
#           end
#         else
#           @simd @inbounds for k=1:K
#             Z[r,c,k] = zero(T)
#             Z[c,r,k] = zero(T)
#           end
#         end
#       end
#       if penalize_diag
#         tmpSlice = slice(tmpStorage, c, c, :)
#         t = max(one(T) - λρ / vecnorm(tmpSlice), zero(T))
#         if t > zero(T)
#           @simd @inbounds for k=1:K
#             Z[c,c,k] = t * tmpSlice[k]
#           end
#         else
#           @simd @inbounds for k=1:K
#             Z[c,r,k] = zero(T)
#           end
#         end
#       else
#         @simd @inbounds for k=1:K
#           Z[c, c, k] = tmpStorage[c, c, k]
#         end
#       end
#     end

#     # u-update
#     @simd @inbounds for i in eachindex
#       U[i] = tmpStorage[i] - Z[i]
#     end

#     # check convergence
#     r_norm = _normdiff(X, Z)
#     s_norm = _normdiff(Z, Zold) * sqrt(ρ)
#     eps_pri = p*abstol + reltol * max( vecnorm(X), vecnorm(Z) )
#     eps_dual = p*abstol + reltol * ρ * vecnorm(U)
#     if r_norm < eps_pri && s_norm < eps_dual
#       break
#     end
#   end
#   Z
# end
