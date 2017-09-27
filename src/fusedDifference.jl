
###############################
#
# FGL from Danaher et al.
#
# currently implements only two groups
###############################

function fusedGraphicalLasso!{T<:AbstractFloat}(
    Θx::StridedMatrix{T}, Θy::StridedMatrix{T},
    Zx::StridedMatrix{T}, Zy::StridedMatrix{T},
    Ux::StridedMatrix{T}, Uy::StridedMatrix{T},
    Σx::StridedMatrix{T}, nx::Int,
    Σy::StridedMatrix{T}, ny::Int,
    λ1::Real, λ2::Real;
    options::ADMMOptions = ADMMOptions(),
    penalize_diag::Bool=true)

    maxiter = options.maxiter
    ρ = options.ρ
    α = options.α
    abstol = options.abstol
    reltol = options.reltol

    n = nx + ny
    γ = 1./ρ
    γx = nx * γ / n
    γy = ny * γ / n

    p = size(Σx, 1)

    gx = ProxGaussLikelihood(Symmetric(Σx))
    gy = ProxGaussLikelihood(Symmetric(Σy))

    Tx = zeros(p, p)  # temp storage
    Ty = zeros(p, p)  # temp storage

    Zxold = copy(Zx)
    Zyold = copy(Zy)

    # ADMM
    for it=1:maxiter
        # update Θx and Θy
        @. Tx = Zx - Ux
        prox!(gx, Θx, Tx, γx)
        @. Ty = Zy - Uy
        prox!(gy, Θy, Ty, γy)

        copy!(Zxold, Zx)
        copy!(Zyold, Zy)
        # update Zx and Zy with relaxation
        # α⋅X + (1-α)⋅Z
        @. Tx = α*Θx + (1.-α)*Zx + Ux
        @. Ty = α*Θy + (1.-α)*Zy + Uy
        for i=1:p
            for j=i:p
                if i==j
                    z1, z2 = proxL1Fused(Tx[i,i], Ty[i,i], penalize_diag ? λ1 : 0., λ2, γ)
                    Zx[i,i] = z1
                    Zy[i,i] = z2
                else
                    z1, z2 = proxL1Fused(Tx[i,j], Ty[i,j], λ1, λ2, γ)
                    Zx[i,j] = Zx[j,i] = z1
                    Zy[i,j] = Zy[j,i] = z2
                end
            end
        end

        # update U
        @. Ux = Tx - Zx
        @. Uy = Ty - Zy

        # check convergence
        r_norm = _normdiff(Θx, Zx) + _normdiff(Θy, Zy)
        s_norm = (_normdiff(Zx, Zxold) + _normdiff(Zy, Zyold)) * sqrt(ρ)
        eps_pri = p*abstol + reltol * max( vecnorm(Θx) + vecnorm(Θy), vecnorm(Zx) + vecnorm(Zy))
        eps_dual = p*abstol + reltol * ρ * ( vecnorm(Ux) + vecnorm(Uy) )
        if r_norm < eps_pri && s_norm < eps_dual
            break
        end
    end
    (Zx, Zy)
end


function fusedGraphicalLasso{T<:AbstractFloat}(
    Σx::StridedMatrix{T}, nx::Int,
    Σy::StridedMatrix{T}, ny::Int,
    λ1::Real, λ2::Real;
    options::ADMMOptions = ADMMOptions(),
    penalize_diag::Bool=true)

    p = size(Σx, 1)

    Θx = eye(p)
    Θy = eye(p)
    Zx = zeros(p, p)
    Zy = zeros(p, p)
    Ux = zeros(p, p)
    Uy = zeros(p, p)

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

function fusedNeighborhoodSelection(
    A1, b1,
    A2, b2,
    λ1, λ2;
    options::ADMMOptions = ADMMOptions()
    )

    assert(size(A1, 2) == size(A2, 2))
    assert(size(A1, 1) == length(b1))
    assert(size(A2, 1) == length(b2))

    maxiter = options.maxiter
    ρ = options.ρ
    α = options.α
    abstol = options.abstol
    reltol = options.reltol
    γ = 1./ρ

    n1 = size(A1, 1)
    n2 = size(A2, 1)
    p = size(A1, 2)

    AtA1 = A1'A1 / n1
    AtA2 = A2'A2 / n2
    for i=1:p
        AtA1[i,i] += ρ
        AtA2[i,i] += ρ
    end
    cAtA1 = cholfact!(AtA1)
    cAtA2 = cholfact!(AtA2)

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

        copy!(z1old, z1)
        copy!(z2old, z2)
        # update z1 and z2
        @. T1 = α*x1 + (1.-α)*z1 + u1
        @. T2 = α*x2 + (1.-α)*z2 + u2
        for i=1:p
            z1[i], z2[i] = proxL1Fused(T1[i], T2[i], λ1, λ2, γ)
        end

        # update U
        @. u1 = T1 - z1
        @. u2 = T2 - z2

        # check convergence
        r_norm = _normdiff(x1, z1) + _normdiff(x2, z2)
        s_norm = (_normdiff(z1, z1old) + _normdiff(z2, z2old)) * sqrt(ρ)
        eps_pri = sqrt(p)*abstol + reltol * max( vecnorm(x1) + vecnorm(x2), vecnorm(z1) + vecnorm(z2))
        eps_dual = sqrt(p)*abstol + reltol * ρ *( vecnorm(u1) + vecnorm(u2) )
        if r_norm < eps_pri && s_norm < eps_dual
            break
        end
    end
    (z1, z2)
end
