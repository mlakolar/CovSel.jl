
###############################
#
# FGL from Danaher et al.
#
# currently implements only two groups
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
