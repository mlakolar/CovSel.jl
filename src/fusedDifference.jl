
###############################
#
# FGL from Danaher et al.
#
# currently implements only two groups
###############################

"""
    fusedGraphicalLasso(Σx, nx, Σy, ny, λ1, λ2; options=ADMMOptions(), penalize_diag=true)
    fusedGraphicalLasso!(Θx, Θy, Zx, Zy, Ux, Uy, Σx, nx, Σy, ny, λ1, λ2; options=ADMMOptions(), penalize_diag=true)

Fused graphical lasso (Danaher, Wang and Witten, 2014) for two groups by ADMM:

    minimize  Σₖ nₖ/n⋅(tr(Σₖ Ωₖ) - log det Ωₖ) + λ1 (‖Ωx‖₁ + ‖Ωy‖₁) + λ2 ‖Ωx - Ωy‖₁

where `Σx`, `Σy` are the sample covariances from `nx` and `ny` observations
and `n = nx + ny`. With `penalize_diag=false` the diagonal is excluded from
the λ1 penalty. In the in-place version `Θ`, `Z` and `U` are the ADMM iterates.
Returns `(Zx, Zy)`, the sparse estimates of the precision matrices.
"""
function fusedGraphicalLasso!(
    Θx::StridedMatrix{T}, Θy::StridedMatrix{T},
    Zx::StridedMatrix{T}, Zy::StridedMatrix{T},
    Ux::StridedMatrix{T}, Uy::StridedMatrix{T},
    Σx::StridedMatrix{T}, nx::Int,
    Σy::StridedMatrix{T}, ny::Int,
    λ1::Real, λ2::Real;
    options::ADMMOptions = ADMMOptions(),
    penalize_diag::Bool=true) where {T<:AbstractFloat}

    maxiter = options.maxiter
    ρ = options.ρ
    α = options.α
    abstol = options.abstol
    reltol = options.reltol

    n = nx + ny
    γ = T(1 / ρ)
    γx = nx * γ / n
    γy = ny * γ / n
    l1, l2 = T(λ1), T(λ2)

    p = size(Σx, 1)

    gx = ProxGaussLikelihood(Symmetric(Σx))
    gy = ProxGaussLikelihood(Symmetric(Σy))

    Tx = zeros(T, p, p)  # temp storage
    Ty = zeros(T, p, p)  # temp storage

    Zxold = copy(Zx)
    Zyold = copy(Zy)

    # ADMM
    for it=1:maxiter
        # update Θx and Θy
        @. Tx = Zx - Ux
        prox!(gx, Θx, Tx, γx)
        @. Ty = Zy - Uy
        prox!(gy, Θy, Ty, γy)

        copyto!(Zxold, Zx)
        copyto!(Zyold, Zy)
        # update Zx and Zy with relaxation
        # α⋅X + (1-α)⋅Z
        @. Tx = α * Θx + (1 - α)*Zx + Ux
        @. Ty = α * Θy + (1 - α)*Zy + Uy
        for i=1:p
            for j=i:p
                if i==j
                    z1, z2 = proxL1Fused(Tx[i,i], Ty[i,i], penalize_diag ? l1 : zero(T), l2, γ)
                    Zx[i,i] = z1
                    Zy[i,i] = z2
                else
                    z1, z2 = proxL1Fused(Tx[i,j], Ty[i,j], l1, l2, γ)
                    Zx[i,j] = Zx[j,i] = z1
                    Zy[i,j] = Zy[j,i] = z2
                end
            end
        end

        # update U
        @. Ux = Tx - Zx
        @. Uy = Ty - Zy

        # check convergence
        r_norm = norm_diff(Θx, Zx, 2) + norm_diff(Θy, Zy, 2)
        s_norm = (norm_diff(Zx, Zxold, 2) + norm_diff(Zy, Zyold, 2)) * sqrt(ρ)
        eps_pri = p*abstol + reltol * max( norm(Θx) + norm(Θy), norm(Zx) + norm(Zy))
        eps_dual = p*abstol + reltol * ρ * ( norm(Ux) + norm(Uy) )
        if r_norm < eps_pri && s_norm < eps_dual
            break
        end
    end
    (Zx, Zy)
end


function fusedGraphicalLasso(
    Σx::StridedMatrix{T}, nx::Int,
    Σy::StridedMatrix{T}, ny::Int,
    λ1::Real, λ2::Real;
    options::ADMMOptions = ADMMOptions(),
    penalize_diag::Bool=true) where {T<:AbstractFloat}

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
    λ1, λ2; options=options, penalize_diag=penalize_diag)
end



####################################
#
# Fused neighborhood selection
#
# ADMM implementation of NIPS paper
####################################

"""
    fusedNeighborhoodSelection(A1, b1, A2, b2, λ1, λ2; options=ADMMOptions())

Fused neighborhood selection for one node in two groups by ADMM:

    minimize  ‖b1 - A1 x1‖² / (2 n1) + ‖b2 - A2 x2‖² / (2 n2)
                + λ1 (‖x1‖₁ + ‖x2‖₁) + λ2 ‖x1 - x2‖₁

Returns `(x1, x2)`.
"""
function fusedNeighborhoodSelection(
    A1, b1,
    A2, b2,
    λ1, λ2;
    options::ADMMOptions = ADMMOptions()
    )

    @assert size(A1, 2) == size(A2, 2)
    @assert size(A1, 1) == length(b1)
    @assert size(A2, 1) == length(b2)

    maxiter = options.maxiter
    ρ = options.ρ
    α = options.α
    abstol = options.abstol
    reltol = options.reltol
    γ = 1 / ρ

    n1 = size(A1, 1)
    n2 = size(A2, 1)
    p = size(A1, 2)

    AtA1 = A1'A1 / n1
    AtA2 = A2'A2 / n2
    for i=1:p
        AtA1[i,i] += ρ
        AtA2[i,i] += ρ
    end
    cAtA1 = cholesky!(Symmetric(AtA1))
    cAtA2 = cholesky!(Symmetric(AtA2))

    Atb1 = A1'*b1 / n1
    Atb2 = A2'*b2 / n2

    x1 = zeros(p)
    x2 = zeros(p)
    z1 = zeros(p)
    z2 = zeros(p)
    u1 = zeros(p)
    u2 = zeros(p)

    T1 = zeros(p)  # temp storage
    T2 = zeros(p)

    z1old = copy(z1)
    z2old = copy(z2)

    # ADMM
    for it=1:maxiter
        # update x1 and x2
        @. T1 = Atb1 + (z1 - u1) * ρ
        x1 = cAtA1 \ T1
        @. T2 = Atb2 + (z2 - u2) * ρ
        x2 = cAtA2 \ T2

        copyto!(z1old, z1)
        copyto!(z2old, z2)
        # update z1 and z2
        @. T1 = α*x1 + (1 - α)*z1 + u1
        @. T2 = α*x2 + (1 - α)*z2 + u2
        for i=1:p
            z1[i], z2[i] = proxL1Fused(T1[i], T2[i], float(λ1), float(λ2), γ)
        end

        # update U
        @. u1 = T1 - z1
        @. u2 = T2 - z2

        # check convergence
        r_norm = norm_diff(x1, z1, 2) + norm_diff(x2, z2, 2)
        s_norm = (norm_diff(z1, z1old, 2) + norm_diff(z2, z2old, 2)) * sqrt(ρ)
        eps_pri = sqrt(p)*abstol + reltol * max( norm(x1) + norm(x2), norm(z1) + norm(z2))
        eps_dual = sqrt(p)*abstol + reltol * ρ *( norm(u1) + norm(u2) )
        if r_norm < eps_pri && s_norm < eps_dual
            break
        end
    end
    (z1, z2)
end
