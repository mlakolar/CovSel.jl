
###############################
#
# FGL from Danaher et al.
#
# all pairs of groups are fused
###############################

"""
    fusedGraphicalLasso(Σx, nx, Σy, ny, λ1, λ2; options=ADMMOptions(), penalize_diag=true, info=ADMMInfo())
    fusedGraphicalLasso!(Θx, Θy, Zx, Zy, Ux, Uy, Σx, nx, Σy, ny, λ1, λ2; options=ADMMOptions(), penalize_diag=true, info=ADMMInfo())

Fused graphical lasso (Danaher, Wang and Witten, 2014) for two groups by ADMM:

    minimize  Σₖ nₖ/n⋅(tr(Σₖ Ωₖ) - log det Ωₖ) + λ1 (‖Ωx‖₁ + ‖Ωy‖₁) + λ2 ‖Ωx - Ωy‖₁

where `Σx`, `Σy` are the sample covariances from `nx` and `ny` observations
and `n = nx + ny`. With `penalize_diag=false` the diagonal is excluded from
the λ1 penalty, and only fused. In the in-place version `Z` and `U` are the ADMM
iterates (the sparse estimates and the scaled dual variables); pass zeros, or
the iterates of a previous call together with its final `info.ρ` to warm start.
`Θx` and `Θy` are overwritten with copies of the estimates. Returns `(Zx, Zy)`;
see [`ADMMInfo`](@ref) for `info`.

This is ProximalSolvers' `ADMMSolver` on the stacked `p × p × 2` array
`(Ωx, Ωy)`, with ProximalBase's `SliceSum` of the two weighted `LogDetLoss`es
and the penalty `FiberSum(NormL1Fused(λ1, λ2))` on the pairs of entries. The
options are scaled by the data as in [`covsel!`](@ref), with `s` the average of
the two average variances, and the residuals are those of the two matrices
added.
"""
function fusedGraphicalLasso!(
    Θx::StridedMatrix{T}, Θy::StridedMatrix{T},
    Zx::StridedMatrix{T}, Zy::StridedMatrix{T},
    Ux::StridedMatrix{T}, Uy::StridedMatrix{T},
    Σx::StridedMatrix{T}, nx::Int,
    Σy::StridedMatrix{T}, ny::Int,
    λ1::Real, λ2::Real;
    options::ADMMOptions = ADMMOptions(),
    penalize_diag::Bool=true,
    info::ADMMInfo=ADMMInfo()) where {T<:AbstractFloat}

    p = size(Σx, 1)
    all(A -> size(A) == (p, p), (Θx, Θy, Zx, Zy, Ux, Uy, Σx, Σy)) ||
      throw(DimensionMismatch("Θ, Z, U and Σ must be p × p matrices"))
    n = nx + ny
    l1, l2 = T(λ1), T(λ2)
    f = SliceSum(LogDetLoss(Σx, T(nx / n)), LogDetLoss(Σy, T(ny / n)))
    # the λ1 penalty is left off the diagonal when penalize_diag is false
    g = FiberSum(NormL1Fused(l1, l2); diagonal = NormL1Fused(penalize_diag ? l1 : zero(T), l2), symmetric = true)

    # the scale of the data, as in covsel!. The residuals and the norms of the
    # iterates are those of the two matrices added, the stopping test
    # ‖r_x‖ + ‖r_y‖ ≤ p abstol + reltol (…), as ADMMSolver's √n abstol is
    # √2 p abstol for the 2p² entries
    s = _dataScale((tr(Σx) + tr(Σy)) / (2p))
    method = ADMMSolver(; ρ = T(options.ρ * s^2), α = T(options.α), adaptive = options.adaptive,
                        abstol = T(options.abstol / (sqrt(2) * s)), abstol_dual = T(options.abstol * s / sqrt(2)),
                        reltol = T(options.reltol), norm = _pairnorm)
    Z, U = cat(Zx, Zy; dims = 3), cat(Ux, Uy; dims = 3)
    r = solve!(Z, f, g, method, ProximalOptions(; maxiter = options.maxiter); u = U)
    copyto!(Zx, view(Z, :, :, 1)); copyto!(Zy, view(Z, :, :, 2))
    copyto!(Ux, view(U, :, :, 1)); copyto!(Uy, view(U, :, :, 2))
    copyto!(Θx, Zx); copyto!(Θy, Zy)
    _finish!(info, "fusedGraphicalLasso!", r.iterations, r.converged, r.ρ, s^2, r.r_norm, r.s_norm)
    (Zx, Zy)
end


# the norm of a stacked pair of matrices, the sum of their Frobenius norms, and
# that of the difference of two such pairs, each in one pass
function _pairnorm(x::Array{T, 3}) where {T}
  n = size(x, 1) * size(x, 2)
  s1 = s2 = zero(T)
  @inbounds @simd for i in 1:n
    s1 += abs2(x[i])
    s2 += abs2(x[n+i])
  end
  sqrt(s1) + sqrt(s2)
end

function _pairnorm(x::Array{T, 3}, y::Array{T, 3}) where {T}
  n = size(x, 1) * size(x, 2)
  s1 = s2 = zero(T)
  @inbounds @simd for i in 1:n
    s1 += abs2(x[i] - y[i])
    s2 += abs2(x[n+i] - y[n+i])
  end
  sqrt(s1) + sqrt(s2)
end

function fusedGraphicalLasso(
    Σx::StridedMatrix{T}, nx::Int,
    Σy::StridedMatrix{T}, ny::Int,
    λ1::Real, λ2::Real;
    options::ADMMOptions = ADMMOptions(),
    penalize_diag::Bool=true,
    info::ADMMInfo=ADMMInfo()) where {T<:AbstractFloat}

    p = size(Σx, 1)

    Θx = Matrix{T}(I, p, p)
    Θy = Matrix{T}(I, p, p)
    Zx = zeros(T, p, p)
    Zy = zeros(T, p, p)
    Ux = zeros(T, p, p)
    Uy = zeros(T, p, p)

    fusedGraphicalLasso!(
    Θx, Θy,
    Zx, Zy,
    Ux, Uy,
    Σx, nx, Σy, ny,
    λ1, λ2; options=options, penalize_diag=penalize_diag, info=info)
end



"""
    fusedGraphicalLasso(S, λ1, λ2; weights=fill(1/K, K), penalize_diag=true, options=ADMMOptions(), info=ADMMInfo())
    fusedGraphicalLasso(S, ns, λ1, λ2; kwargs...)
    fusedGraphicalLasso!(Z, U, S, λ1, λ2; kwargs...)
    fusedGraphicalLasso!(Z, U, S, ns, λ1, λ2; kwargs...)

Fused graphical lasso of K groups (Danaher, Wang and Witten, 2014), by ADMM:

    minimize  Σₖ wₖ (tr(Sₖ Ωₖ) - log det Ωₖ)
                + λ1 Σₖ ‖Ωₖ‖₁ + λ2 Σₖ<ₗ ‖Ωₖ - Ωₗ‖₁

`S` is a `p × p × K` array or a vector of K covariance matrices of the same
size. Every pair of groups is fused. The positive likelihood `weights` default
to `1/K`; passing the sample sizes `ns` instead sets them to `ns / sum(ns)`.
Weights supplied explicitly are used as given, without normalization. With
these normalized weights, divide the penalties in Danaher et al.'s
sample-size-weighted objective by `sum(ns)`. With `penalize_diag=false`, only
the λ1 term is omitted from the diagonal; the diagonal entries are still fused.

Returns a `p × p × K` array with the estimates in `Z[:, :, k]`. The in-place
method updates `Z` and the scaled dual variables `U`, dense arrays of that
size. Start them at zero, or reuse both together with
`options=ADMMOptions(; ρ=info.ρ)` from a previous fit to warm start. `Z` and
`U` must not alias. See [`ADMMInfo`](@ref) for convergence information.

With K = 1 the fusion term vanishes, giving [`covsel!`](@ref) with penalty
`λ1 / weights[1]`. With K = 2 the objective agrees with the two-matrix method
when `weights = [nx, ny] / (nx + ny)`. The two-matrix method returns a tuple;
these collection methods always return a stacked array.

The solver uses ProximalBase's `SliceSum` of weighted `LogDetLoss`es and
`FiberSum(NormL1Fused(λ1, λ2); symmetric=true)`. `NormL1Fused` computes the
all-pairs proximal operator by sorting and pooling, in O(K log K) per fiber.
As in [`groupGraphicalLasso`](@ref), the ADMM penalty parameter is
`ρ s² w̄`, where `s` is the average variance across the groups and `w̄` the
average likelihood weight. See [`groupGraphicalLasso`](@ref) for joint
estimation with a group penalty that encourages shared sparsity.
"""
function fusedGraphicalLasso!(
    Z::Array{T, 3}, U::Array{T, 3},
    S::Union{AbstractArray{<:Real, 3}, AbstractVector{<:AbstractMatrix}},
    λ1::Real, λ2::Real;
    weights::AbstractVector{<:Real} = fill(one(T) / _nslices(S), _nslices(S)),
    penalize_diag::Bool = true,
    options::ADMMOptions = ADMMOptions(),
    info::ADMMInfo = ADMMInfo()) where {T<:AbstractFloat}

    p, K = _fglsize(S)
    size(Z) == size(U) == (p, p, K) ||
      throw(DimensionMismatch("Z and U must be p × p × K arrays, here $p × $p × $K"))
    Base.mightalias(Z, U) && throw(ArgumentError("Z and U must not alias"))
    length(weights) == K || throw(DimensionMismatch("$(length(weights)) weights for $K covariance matrices"))
    w = T.(weights)
    all(v -> isfinite(v) && v > 0, w) || throw(ArgumentError("the weights must be finite and positive"))
    l1, l2 = T(λ1), T(λ2)
    isfinite(l1) && l1 >= 0 || throw(ArgumentError("λ1 must be finite and nonnegative, got $λ1"))
    isfinite(l2) && l2 >= 0 || throw(ArgumentError("λ2 must be finite and nonnegative, got $λ2"))

    f = SliceSum([LogDetLoss(Matrix{T}(_covslice(S, k)), w[k]) for k in 1:K])
    h = NormL1Fused(l1, l2)
    g = FiberSum(h; diagonal = penalize_diag ? h : NormL1Fused(zero(T), l2), symmetric = true)

    s = _dataScale(sum(k -> tr(_covslice(S, k)), 1:K) / (p * K))
    w̄ = sum(w) / K
    method = ADMMSolver(; ρ = T(options.ρ * s^2 * w̄), α = T(options.α), adaptive = options.adaptive,
                        abstol = T(options.abstol / s), abstol_dual = T(options.abstol * s * w̄),
                        reltol = T(options.reltol))
    r = solve!(Z, f, g, method, ProximalOptions(; maxiter = options.maxiter); u = U)
    _finish!(info, "fusedGraphicalLasso!", r.iterations, r.converged, r.ρ, s^2 * w̄, r.r_norm, r.s_norm)
    Z
end

function fusedGraphicalLasso(
    S::Union{AbstractArray{<:Real, 3}, AbstractVector{<:AbstractMatrix}},
    λ1::Real, λ2::Real; kwargs...)
    p, K = _fglsize(S)
    T = float(eltype(_covslice(S, 1)))
    fusedGraphicalLasso!(zeros(T, p, p, K), zeros(T, p, p, K), S, λ1, λ2; kwargs...)
end

function _fglsize(S)
    K = _nslices(S)
    K > 0 || throw(ArgumentError("at least one covariance matrix is required"))
    p = size(_covslice(S, 1), 1)
    p > 0 || throw(ArgumentError("the covariance matrices must be nonempty"))
    all(k -> size(_covslice(S, k)) == (p, p), 1:K) ||
      throw(DimensionMismatch("the covariance matrices must all be p × p"))
    p, K
end

function _fglweights(S, ns)
    length(ns) == _nslices(S) || throw(DimensionMismatch("$(length(ns)) sample sizes for $(_nslices(S)) covariance matrices"))
    all(n -> n > 0, ns) || throw(ArgumentError("the sample sizes must be positive"))
    # Convert before summing so that integer sample sizes cannot overflow.
    n = Float64.(ns)
    n ./ sum(n)
end

function fusedGraphicalLasso(
    S::Union{AbstractArray{<:Real, 3}, AbstractVector{<:AbstractMatrix}},
    ns::AbstractVector{<:Integer}, λ1::Real, λ2::Real; kwargs...)
    fusedGraphicalLasso(S, λ1, λ2; weights = _fglweights(S, ns), kwargs...)
end

function fusedGraphicalLasso!(
    Z::Array{T, 3}, U::Array{T, 3},
    S::Union{AbstractArray{<:Real, 3}, AbstractVector{<:AbstractMatrix}},
    ns::AbstractVector{<:Integer}, λ1::Real, λ2::Real; kwargs...) where {T<:AbstractFloat}
    fusedGraphicalLasso!(Z, U, S, λ1, λ2; weights = _fglweights(S, ns), kwargs...)
end


####################################
#
# Fused neighborhood selection
#
# ADMM implementation of NIPS paper
####################################

"""
    fusedNeighborhoodSelection(A1, b1, A2, b2, λ1, λ2; options=ADMMOptions(), info=ADMMInfo())

Fused neighborhood selection for one node in two groups by ADMM:

    minimize  ‖b1 - A1 x1‖² / (2 n1) + ‖b2 - A2 x2‖² / (2 n2)
                + λ1 (‖x1‖₁ + ‖x2‖₁) + λ2 ‖x1 - x2‖₁

Returns `(x1, x2)`; see [`ADMMInfo`](@ref) for `info`.
"""
function fusedNeighborhoodSelection(
    A1, b1,
    A2, b2,
    λ1, λ2;
    options::ADMMOptions = ADMMOptions(),
    info::ADMMInfo=ADMMInfo()
    )

    @assert size(A1, 2) == size(A2, 2)
    @assert size(A1, 1) == length(b1)
    @assert size(A2, 1) == length(b2)

    maxiter = options.maxiter
    α = options.α
    reltol = options.reltol

    n1 = size(A1, 1)
    n2 = size(A2, 1)
    p = size(A1, 2)

    # the x-updates solve (AᵀA / n + ρI) x = v; with AᵀA / n = V D Vᵀ this is
    # x = V (D + ρI)⁻¹ Vᵀ v, so that ρ can change without a new factorization
    D1, V1 = eigen(Symmetric(A1'A1 / n1))
    D2, V2 = eigen(Symmetric(A2'A2 / n2))
    eigsolve!(x, V, D, v, w, ρ) = (mul!(w, V', v); w ./= D .+ ρ; mul!(x, V, w))

    Atb1 = A1'*b1 / n1
    Atb2 = A2'*b2 / n2

    # scales of the columns of A and of b: ρ has the units of A², the primal
    # residual those of the coefficients and the dual one those of Aᵀb / n
    sa = sqrt(_dataScale((sum(D1) + sum(D2)) / (2p)))
    sb = sqrt(_dataScale((sum(abs2, b1) / n1 + sum(abs2, b2) / n2) / 2))
    ρ = options.ρ * sa^2
    abstol_pri, abstol_dual = options.abstol * sb / sa, options.abstol * sa * sb

    h = NormL1Fused(float(λ1), float(λ2))

    x1 = zeros(p)
    x2 = zeros(p)
    z1 = zeros(p)
    z2 = zeros(p)
    u1 = zeros(p)
    u2 = zeros(p)

    T1 = zeros(p)  # temp storage
    T2 = zeros(p)
    w = zeros(p)

    z1old = copy(z1)
    z2old = copy(z2)

    # ADMM
    converged = false
    iter = changes = 0
    r_norm = s_norm = NaN
    while iter < maxiter
        iter += 1
        γ = 1 / ρ

        # update x1 and x2
        @. T1 = Atb1 + (z1 - u1) * ρ
        eigsolve!(x1, V1, D1, T1, w, ρ)
        @. T2 = Atb2 + (z2 - u2) * ρ
        eigsolve!(x2, V2, D2, T2, w, ρ)

        copyto!(z1old, z1)
        copyto!(z2old, z2)
        # update z1 and z2
        @. T1 = α*x1 + (1 - α)*z1 + u1
        @. T2 = α*x2 + (1 - α)*z2 + u2
        prox!((z1, z2), h, (T1, T2), γ)

        # update U
        @. u1 = T1 - z1
        @. u2 = T2 - z2

        # check convergence
        r_norm = norm_diff(x1, z1, 2) + norm_diff(x2, z2, 2)
        s_norm = ρ * (norm_diff(z1, z1old, 2) + norm_diff(z2, z2old, 2))
        eps_pri = sqrt(p)*abstol_pri + reltol * max( norm(x1) + norm(x2), norm(z1) + norm(z2))
        eps_dual = sqrt(p)*abstol_dual + reltol * ρ *( norm(u1) + norm(u2) )
        if r_norm < eps_pri && s_norm < eps_dual
            converged = true
            break
        end

        # update ρ, and rescale the scaled dual variables
        if options.adaptive
            f = _penaltyFactor(r_norm, eps_pri, s_norm, eps_dual, changes)
            if f != 1
                ρ *= f
                u1 ./= f
                u2 ./= f
                changes += 1
            end
        end
    end
    _finish!(info, "fusedNeighborhoodSelection", iter, converged, ρ, sa^2, r_norm, s_norm)
    (z1, z2)
end
