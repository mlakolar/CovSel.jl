
###############################
#
# FGL from Danaher et al.
#
# currently implements only two groups
###############################


function fusedGraphicalLasso(X, Y, λ1, λ2;
        options::ADMMOptions = ADMMOptions(),
        penalize_diag::Bool=true
        )

    maxiter = options.maxiter
    ρ = options.ρ
    α = options.α
    abstol = options.abstol
    reltol = options.reltol

    assert(size(X, 2) == size(Y, 2))

    nx = size(X, 1)
    ny = size(Y, 1)
    n = nx + ny
    γ = 1./ρ
    γx = nx * γ / n
    γy = ny * γ / n

    p = size(X, 2)

    Σx = cov(X)
    Σy = cov(Y)
    gx = ProxGaussLikelihood(Σx)
    gy = ProxGaussLikelihood(Σy)

    Θx = eye(p)
    Θy = eye(p)
    Zx = zeros(p, p)
    Zy = zeros(p, p)
    Ux = zeros(p, p)
    Uy = zeros(p, p)

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


####################################
#
# Direct difference estimation
#
####################################

struct ActiveShootingOptions <: CovSelSolver
  maxIter::Int64
  maxInnerIter::Int64
  maxChangeTol::Float64
  kktTol::Float64
end

ActiveShootingOptions(
  ;maxIter::Int64=1000,
  maxInnerIter::Int64=2000,
  maxChangeTol::Float64=1e-7,
  kktTol::Float64=1e-7) = ActiveShootingOptions(maxIter, maxInnerIter, maxChangeTol, kktTol)


function updateDelta!{T<:AbstractFloat}(
  Δ::SparseMatrixCSC{T},
  A::StridedMatrix{T}, Σx::StridedMatrix{T}, Σy::StridedMatrix{T},
  λ::T, Ups::StridedMatrix{T};
  options::ActiveShootingOptions = ActiveShootingOptions())
  # Delta is a sparse matrix stored in CSC format
  # only diagonal and lower triangular elements are used

  # extract fields from sparse matrix Delta
  rows = rowvals(Δ)
  vals = nonzeros(Δ)

  maxInnerIter = options.maxInnerIter
  maxChangeTol = options.maxChangeTol

  p = size(Δ, 1)

  for iter=1:maxInnerIter
    fDone = true
    for colInd=1:p
      for j in nzrange(Δ, colInd)
        rowInd = rows[j]

        if rowInd==colInd
          # diagonal elements
          ix0 = one(T) / (Σx[rowInd,rowInd] * Σy[rowInd,rowInd])
          x1 = A[rowInd,rowInd] - Σy[rowInd,rowInd] + Σx[rowInd,rowInd]
          tmp = shrink(-x1*ix0 + vals[j], λ*Ups[rowInd,colInd]*ix0)
        else
          # off-diagonal elements
          ix0 = one(T) / ((Σx[rowInd,rowInd]*Σy[colInd,colInd] + Σx[colInd,colInd]*Σy[rowInd,rowInd])/2.
                            + Σx[rowInd,colInd]*Σy[rowInd,colInd])
          x1 = A[rowInd,colInd] + A[colInd,rowInd] - 2.*(Σy[rowInd,colInd] - Σx[rowInd,colInd])
          tmp = shrink(-x1*ix0/2. + vals[j], λ*Ups[rowInd,colInd] * ix0)
        end

        # size of update
        h = tmp - vals[j]
        # update is too big -- not done
        if abs(h) > maxChangeTol
          fDone = false
        end
        # update Δ
        vals[j] = tmp

        # update A -- only active set
        for ci=1:p
          for k in nzrange(Δ, ci)
            ri = rows[k]

            if rowInd == colInd
              A[ri,ci] = A[ri,ci] + h * Σx[ri,rowInd]*Σy[rowInd,ci]
              if ri != ci
                A[ci,ri] = A[ci,ri] + h * Σx[ci,rowInd]*Σy[rowInd,ri]
              end
            else
              A[ri,ci] = A[ri,ci] + h * (Σx[ri,colInd]*Σy[rowInd,ci] + Σx[ri,rowInd]*Σy[colInd,ci])
              if ri != ci
                A[ci,ri] = A[ci,ri] + h * (Σx[ci,colInd]*Σy[rowInd,ri] + Σx[ci,rowInd]*Σy[colInd,ri])
              end
            end
          end
        end   # end update A
      end
    end

    if fDone
      break
    end
  end  # while

  sparse(Δ)
end

function findViolator!{T<:AbstractFloat}(
  Δ::SparseMatrixCSC{T},
  A::StridedMatrix{T}, Σx::StridedMatrix{T}, Σy::StridedMatrix{T},
  λ::T, Ups::StridedMatrix{T};
  options::ActiveShootingOptions = ActiveShootingOptions())

  p = size(Σx, 1)

  vmax = zero(T)
  im = 0
  for c=1:p
    for r=c:p
      t = abs((A[r,c] + A[c,r]) / 2. - (Σy[r,c] - Σx[r,c])) - λ * Ups[r,c]
      if t > vmax
        vmax = t
        im = (c-1)*p+r
      end
    end
  end
  if vmax > options.kktTol
    Δ[im] = eps(T)
  else
    im = 0
  end
  im
end


function updateA!{T<:AbstractFloat}(
  A::StridedMatrix{T},
  Δ::SparseMatrixCSC{T},
  Σx::StridedMatrix{T}, Σy::StridedMatrix{T})

  p = size(A, 1)

  rows = rowvals(Δ)
  vals = nonzeros(Δ)

  for ac=1:p
    for ar=1:p
      v = zero(T)
      #
      for ci=1:p
        for j in nzrange(Δ, ci)
          ri = rows[j]
          if ri == ci
            v += Σx[ri, ar] * Σy[ci, ac] * vals[j]
          else
            v += (Σx[ri, ar] * Σy[ci, ac] + Σx[ci, ar] * Σy[ri, ac]) * vals[j]
          end
        end
      end
      #
      A[ar,ac] = v
    end
  end
  A
end



differencePrecisionActiveShooting{T<:AbstractFloat}(
  Σx::StridedMatrix{T}, Σy::StridedMatrix{T},
  λ::T, Ups::StridedMatrix{T};
  options::ActiveShootingOptions = ActiveShootingOptions()) =
    differencePrecisionActiveShooting!(spzeros(size(Σx)...), Σx, Σy, λ, Ups; options=options)

function differencePrecisionActiveShooting!{T<:AbstractFloat}(
  Δ::SparseMatrixCSC{T},
  Σx::StridedMatrix{T}, Σy::StridedMatrix{T},
  λ::T, Ups::StridedMatrix{T};
  options::ActiveShootingOptions = ActiveShootingOptions())

  maxIter = options.maxIter
  p = size(Σx, 1)
  A = zeros(p,p)
  tril!(Δ)

  # lDelta = tril(Δ, -1)
  # dDelta = spdiagm(diag(Δ))
  # A = Σx * (lDelta + lDelta' + dDelta) * Σy
  updateA!(A, Δ, Σx, Σy)
  if iszero(Δ)
    ind = findViolator!(Δ, A, Σx, Σy, λ, Ups; options=options)
  end

  for iter=1:maxIter
    updateDelta!(Δ, A, Σx, Σy, λ, Ups; options=options)

    # update A
    # lDelta = tril(Δ, -1)
    # dDelta = spdiagm(diag(Δ))
    # A = Σx * (lDelta + lDelta' + dDelta) * Σy
    updateA!(A, Δ, Σx, Σy)

    if findViolator!(Δ, A, Σx, Σy, λ, Ups; options=options) == 0
      break
    end
  end

  lDelta = tril(Δ, -1)
  dDelta = spdiagm(diag(Δ))
  Δ = (lDelta + lDelta') + dDelta
end


function differencePrecisionNaive(Σx, Σy, λ, Ups; options::ActiveShootingOptions = ActiveShootingOptions())
  maxIter = options.maxIter
  maxChangeTol = options.maxChangeTol
  p = size(Σx, 1)

  Δ = zeros(p, p)
  A = zeros(p, p)

  iter = 1;
  while iter < maxIter
    fDone = true
    for a=1:p
      for b=a:p
        if a==b
          # diagonal elements
          x0 = Σx[a,a]*Σy[a,a] / 2.
          x1 = A[a,a] - Σy[a,a] + Σx[a,a]
          tmp = shrink(-x1/x0/2. + Δ[a,b], λ*Ups[a,b] / 2. / x0)
        else
          # off-diagonal elements
          x0 = (Σx[a,a]*Σy[b,b] + Σx[b,b]*Σy[a,a])/2. + Σx[a,b]*Σy[a,b]
          x1 = A[a,b] + A[b,a] - 2.*(Σy[a,b] - Σx[a,b])
          tmp = shrink(-x1/x0/2. + Δ[a,b], λ*Ups[a,b] / x0)
        end

        h = tmp - Δ[a,b]
        Δ[a,b] = tmp
        Δ[b,a] = tmp
        if abs(h) > maxChangeTol
          fDone = false
        end
        for j=1:p
          for k=1:p
            if a == b
              A[j,k] = A[j,k] + h * Σx[j,a]*Σy[a,k]
            else
              A[j,k] = A[j,k] + h * (Σx[j,a]*Σy[b,k] + Σx[j,b]*Σy[a,k])
            end
          end
        end
      end
    end

    iter = iter + 1;
    if fDone
      break
    end
  end
  Δ
end



####################################
#
# Direct difference estimation using iterative hard thresholding
#
####################################



function differencePrecisionIHT_objective{T<:AbstractFloat}(
  Σx::StridedMatrix{T}, Σy::StridedMatrix{T},
  Δ::StridedMatrix{T}, L::StridedMatrix{T})

  A = Δ + L
  trace((Σx*A*Σy / 2 - (Σy - Σx)) * A)
end

differencePrecisionIHT_objective{T<:AbstractFloat}(
  Σx::StridedMatrix{T}, Σy::StridedMatrix{T},
  Δ::StridedMatrix{T}) =
    trace((Σx*Δ*Σy / 2 - (Σy - Σx)) * Δ)


function differencePrecisionIHT_grad_delta!{T<:AbstractFloat}(
  grad_out::StridedMatrix{T},
  Σx::StridedMatrix{T}, Σy::StridedMatrix{T},
  Δ::StridedMatrix{T}, L::StridedMatrix{T})
  p = size(Σx, 1)

  # (a, b) indices of updated grad_out[a,b] element
  for a=1:p
    for b=a:p
      grad_out[a,b] = zero(T)

      for j=1:p, k=1:p
        grad_out[a,b] += (Σx[j, a] * Σy[k, b] + Σy[j, a] * Σx[k, b]) * ( Δ[j, k] + L[j,k] ) / 2.
      end
      grad_out[a,b] += Σx[a,b] - Σy[a,b]

      # update the other symmetric element
      if a != b
        grad_out[b,a] = grad_out[a,b]
      end
    end
  end

  grad_out
end

function differencePrecisionIHT_grad_delta!{T<:AbstractFloat}(
  grad_out::StridedMatrix{T},
  Σx::StridedMatrix{T}, Σy::StridedMatrix{T},
  Δ::StridedMatrix{T})
  p = size(Σx, 1)

  # (a, b) indices of updated grad_out[a,b] element
  for a=1:p
    for b=a:p
      grad_out[a,b] = zero(T)

      for j=1:p, k=1:p
        grad_out[a,b] += (Σx[j, a] * Σy[k, b] + Σy[j, a] * Σx[k, b]) * Δ[j, k] / 2.
      end
      grad_out[a,b] += Σx[a,b] - Σy[a,b]

      # update the other symmetric element
      if a != b
        grad_out[b,a] = grad_out[a,b]
      end
    end
  end

  grad_out
end


function differencePrecisionIHT_grad_u!(grad_out_u, grad_out_delta, Σx, Σy, Delta, L, U)
  differencePrecisionIHT_grad_delta!(grad_out_delta, Σx, Σy, Delta, L)
  scale!(grad_out_delta, 2.)
  grad_out_u = grad_out_delta * U
end


# η is the step size
# s is the target sparsity
function differencePrecisionIHT{T<:AbstractFloat}(
  Σx::StridedMatrix{T}, Σy::StridedMatrix{T},
  η::T, s::Int64;
  epsTol=1e-5, maxIter=1000)

  assert(size(Σx,1) == size(Σx,2) == size(Σy,1) == size(Σy,2))

  p = size(Σx, 1)

  gD = zeros(p, p)
  Δ = zeros(p, p)
  L = zeros(p, p)

  fvals = []
  # gradient descent
  #
  fv = differencePrecisionIHT_objective(Σx, Σy, Δ, L)
  push!(fvals, fv)
  for iter=1:maxIter
    # update Δ
    differencePrecisionIHT_grad_delta!(gD, Σx, Σy, Δ, L)
    Δ .-= η .* gD
    HardThreshold!(Δ, s)

    # check for convergence
    fv_new = differencePrecisionIHT_objective(Σx, Σy, Δ, L)
    push!(fvals, fv_new)
    if abs(fv_new - fv) <= epsTol
      break
    end
    fv = fv_new
  end

  (Δ, fvals)
end


####################################
#
# Direct difference estimation using iterative hard thresholding
#
#  now with low rank
#
####################################



# η is the step size
# s is the target sparsity
# r is the target rank
function differenceLatentPrecisionIHT{T<:AbstractFloat}(
  Σx::StridedMatrix{T}, Σy::StridedMatrix{T},
  η::T, s::Int64, r::Int64;
  epsTol=1e-5, maxIter=1000)

  assert(size(Σx,1) == size(Σx,2) == size(Σy,1) == size(Σy,2))

  p = size(Σx, 1)

  gD = zeros(p, p)
  gU = zeros(p, r)
  Δ = zeros(p, p)
  L = zeros(p, p)

  # initialize Δ, L, U
  #
  tmp = inv(Σx) - inv(Σy)
  HardThreshold!(Δ, tmp, s)
  tmp .= tmp - Δ
  U, d, V = svd(tmp)
  U = U[:,1:r] .* sqrt.(d[1:r])
  L = U * U'

  fvals = []
  # gradient descent
  #
  fv = differencePrecisionIHT_objective(Σx, Σy, Δ, L)
  push!(fvals, fv)
  for iter=1:maxIter
    # update Δ
    differencePrecisionIHT_grad_delta!(gD, Σx, Σy, Δ, L)
    @. Δ -= η * gD
    HardThreshold!(Δ, s)

    # update U and L
    differencePrecisionIHT_grad_u!(gU, gD, Σx, Σy, Δ, L, U)
    @. U -= η * gU
    L = U * U'

    # check for convergence
    fv_new = differencePrecisionIHT_objective(Σx, Σy, Δ, L)
    push!(fvals, fv_new)
    if abs(fv_new - fv) <= epsTol
      break
    end
    fv = fv_new
  end

  (Δ, L, U, fvals)
end


















##
