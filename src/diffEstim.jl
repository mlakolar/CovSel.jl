
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


function updateDelta!(Delta, hSx, hSy, A, lambda; maxIter=2000, optTol=1e-7)
  # Delta is a sparse matrix stored in CSC format
  # only diagonal and lower triangular elements are used

  # extract fields from sparse matrix Delta
  colptr = Delta.colptr
  nzval = Delta.nzval
  rowval = Delta.rowval

  p = size(Delta, 1)

  iter = 1
  while iter <= maxIter
    fDone = true
    for colInd=1:p
      for j=colptr[colInd]:colptr[colInd+1]-1
        rowInd = rowval[j]

        # a sanity check in
        if rowInd < colInd
          continue
        end

        if rowInd==colInd
          # diagonal elements
          x0 = hSx[rowInd,rowInd]*hSy[rowInd,rowInd]/2
          x1 = A[rowInd,rowInd] - hSy[rowInd,rowInd] + hSx[rowInd,rowInd]
          tmp = shrink(-x1/x0/2 + nzval[j], lambda[rowInd,colInd] / 2 / x0)
        else
          # off-diagonal elements
          x0 = (hSx[rowInd,rowInd]*hSy[colInd,colInd] + hSx[colInd,colInd]*hSy[rowInd,rowInd])/2
          + hSx[rowInd,colInd]*hSy[rowInd,colInd]
          x1 = A[rowInd,colInd] + A[colInd,rowInd] - 2*(hSy[rowInd,colInd] - hSx[rowInd,colInd])
          tmp = shrink(-x1/x0/2 + nzval[j], lambda[rowInd,colInd] / x0)
        end

        # size of update
        h = tmp - nzval[j]
        # update is too big -- not done
        if abs(h) > optTol
          fDone = false
        end
        # update Delta
        nzval[j] = tmp

        # update A -- only active set
        for ci=1:p
          for k=colptr[ci]:colptr[ci+1]-1
            ri = rowval[k]

            if rowInd == colInd
              A[ri,ci] = A[ri,ci] + h * hSx[ri,rowInd]*hSy[rowInd,ci]
              if ri != ci
                A[ci,ri] = A[ci,ri] + h * hSx[ci,rowInd]*hSy[rowInd,ri]
              end
            else
              A[ri,ci] = A[ri,ci] + h * (hSx[ri,colInd]*hSy[rowInd,ci] + hSx[ri,rowInd]*hSy[colInd,ci])
              if ri != ci
                A[ci,ri] = A[ci,ri] + h * (hSx[ci,colInd]*hSy[rowInd,ri] + hSx[ci,rowInd]*hSy[colInd,ri])
              end
            end
          end
        end   # end update A
      end
    end

    iter = iter + 1;
    if fDone
      break
    end

  end  # while

  sparse(Delta)
end

function findViolator!(active_set, Delta, A, hSx, hSy, lambda; kktTol=1e-7)
  p = size(hSx, 1)

  tmp = abs((A + A') / 2 - (hSy - hSx)) - lambda
  ind = indmax(tmp)
  if tmp[ind] > kktTol
    push!(active_set, ind)
    Delta[ind] = eps()
  else
    ind = 0
  end

  return ind
end

function differencePrecisionActiveShooting(hSx, hSy, lambda; maxIter=1000, maxInnerIter=1000, optTol=1e-7, Delta = [])
  p = size(hSx, 1)

  if isempty(Delta)
    Delta = spzeros(p, p)
    A = zeros(p, p)

    active_set = Array(Integer, 0)
    indAdd = findViolator!(active_set, Delta, A, hSx, hSy, lambda)
    if indAdd == 0
      return Delta
    end
  else
    lDelta = tril(Delta, -1)
    dDelta = spdiagm(diag(Delta))
    A = hSx * (lDelta + lDelta' + dDelta) * hSy

    # make sure Delta is lower triangular
    Delta = lDelta + dDelta
    active_set = find(Delta)
  end

  iter = 1;
  while iter < maxIter
    old_active_set = copy(active_set)
    updateDelta!(Delta, hSx, hSy, A, lambda; maxIter=maxInnerIter, optTol=optTol)
    active_set = find(Delta)

    # update A
    lDelta = tril(Delta, -1)
    dDelta = spdiagm(diag(Delta))
    A = hSx * (lDelta + lDelta' + dDelta) * hSy
    # add violating element into active set
    indAdd = findViolator!(active_set, Delta, A, hSx, hSy, lambda)

    if old_active_set == active_set
      break
    else
    end

    iter = iter + 1;

  end
  lDelta = tril(Delta, -1)
  dDelta = spdiagm(diag(Delta))
  (lDelta + lDelta') + dDelta
end


function differencePrecisionNaive(hSx, hSy, lambda, Ups; maxIter=2000, optTol=1e-7)
  p = size(hSx, 1)

  Delta = zeros(p, p)
  A = zeros(p, p)

  iter = 1;
  while iter < maxIter
    fDone = true
    for a=1:p
      for b=a:p
        if a==b
          # diagonal elements
          x0 = hSx[a,a]*hSy[a,a] / 2.
          x1 = A[a,a] - hSy[a,a] + hSx[a,a]
          tmp = shrink(-x1/x0/2. + Delta[a,b], lambda*Ups[a,b] / 2. / x0)
        else
          # off-diagonal elements
          x0 = (hSx[a,a]*hSy[b,b] + hSx[b,b]*hSy[a,a])/2. + hSx[a,b]*hSy[a,b]
          x1 = A[a,b] + A[b,a] - 2.*(hSy[a,b] - hSx[a,b])
          tmp = shrink(-x1/x0/2. + Delta[a,b], lambda*Ups[a,b] / x0)
        end

        h = tmp - Delta[a,b]
        Delta[a,b] = tmp
        Delta[b,a] = tmp
        if abs(h) > optTol
          fDone = false
        end
        for j=1:p
          for k=1:p
            if a == b
              A[j,k] = A[j,k] + h * hSx[j,a]*hSy[a,k]
            else
              A[j,k] = A[j,k] + h * (hSx[j,a]*hSy[b,k] + hSx[j,b]*hSy[a,k])
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
  Delta
end
