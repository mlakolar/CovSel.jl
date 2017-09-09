module Alt

using ProximalBase: shrink
using CoordinateDescent: CDOptions

function differencePrecisionNaive(Σx, Σy, λ, Ups, options = CDOptions())
  maxIter = options.maxIter
  optTol = options.optTol
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
          tmp = shrink(-x1/(2.*x0) + Δ[a,b], λ*Ups[a,b] / x0)
        end

        h = tmp - Δ[a,b]
        Δ[a,b] = tmp
        Δ[b,a] = tmp
        if abs(h) > optTol
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



function updateDelta!{T<:AbstractFloat}(
  Δ::SparseMatrixCSC{T},
  A::StridedMatrix{T}, Σx::StridedMatrix{T}, Σy::StridedMatrix{T},
  λ::T, Ups::StridedMatrix{T},
  options::CDOptions = CDOptions())
  # Delta is a sparse matrix stored in CSC format
  # only diagonal and lower triangular elements are used

  # extract fields from sparse matrix Delta
  rows = rowvals(Δ)
  vals = nonzeros(Δ)

  maxInnerIter = options.maxIter
  maxChangeTol = options.optTol

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
  λ::T, Ups::StridedMatrix{T},
  options::CDOptions = CDOptions())

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
  if vmax > options.optTol
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
  λ::T, Ups::StridedMatrix{T},
  options::CDOptions = CDOptions()) =
    differencePrecisionActiveShooting!(spzeros(size(Σx)...), Σx, Σy, λ, Ups, options)

function differencePrecisionActiveShooting!{T<:AbstractFloat}(
  Δ::SparseMatrixCSC{T},
  Σx::StridedMatrix{T}, Σy::StridedMatrix{T},
  λ::T, Ups::StridedMatrix{T},
  options::CDOptions = CDOptions())

  maxIter = options.maxIter
  p = size(Σx, 1)
  A = zeros(p,p)
  tril!(Δ)

  # lDelta = tril(Δ, -1)
  # dDelta = spdiagm(diag(Δ))
  # A = Σx * (lDelta + lDelta' + dDelta) * Σy
  updateA!(A, Δ, Σx, Σy)
  if iszero(Δ)
    ind = findViolator!(Δ, A, Σx, Σy, λ, Ups, options)
  end

  for iter=1:maxIter
    updateDelta!(Δ, A, Σx, Σy, λ, Ups, options)
    updateA!(A, Δ, Σx, Σy)

    if findViolator!(Δ, A, Σx, Σy, λ, Ups, options) == 0
      break
    end
  end

  lDelta = tril(Δ, -1)
  dDelta = spdiagm(diag(Δ))
  Δ = (lDelta + lDelta') + dDelta
end



end
