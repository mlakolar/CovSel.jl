# this file implements direct difference estimation for Δ = Ωx - Ωy

####################################
#
# utils
#
####################################

function _mul_Σx_Δ_Σy{T<:AbstractFloat, I<:Integer}(
  Σx::StridedMatrix{T},
  x::SparseVector{T},
  Σy::StridedMatrix{T},
  ar::I,
  ac::I
  )

  p = size(Σx, 1)

  nzval = SparseArrays.nonzeros(x)
  rowval = SparseArrays.nonzeroinds(x)

  v = zero(T)
  for j=1:length(nzval)
    ri, ci = ind2subLowerTriangular(p, rowval[j])
    if ri == ci
      @inbounds v += Σx[ri, ar] * Σy[ci, ac] * nzval[j]
    else
      @inbounds v += (Σx[ri, ar] * Σy[ci, ac] + Σx[ci, ar] * Σy[ri, ac]) * nzval[j]
    end
  end
  v
end

####################################
#
# loss tr(Σx⋅Δ⋅Σy⋅Δ)/2 + tr(Δ(Σy-Σx))
#
####################################
struct CDDirectDifferenceLoss{T<:AbstractFloat, S} <: HD.CoordinateDifferentiableFunction{T}
  Σx::S
  Σy::S
  A::Matrix{T}    #Σx⋅Δ⋅Σy
  p::Int64

  CDDirectDifferenceLoss{T, S}(Σx::AbstractMatrix{T}, Σy::AbstractMatrix{T}, A::Matrix{T}, p) where {T,S} =
    new(Σx, Σy, A, p)
end

function CDDirectDifferenceLoss{T<:AbstractFloat}(Σx::AbstractMatrix{T}, Σy::AbstractMatrix{T})
  (issymmetric(Σx) && issymmetric(Σy)) || throw(DimensionMismatch())
  (p = size(Σx, 1)) == size(Σy, 1) || throw(DimensionMismatch())
  CDDirectDifferenceLoss{T, typeof(Σx)}(Σx, Σy, zeros(T, p, p), p)
end

HD.numCoordinates(f::CDDirectDifferenceLoss) = div(f.p * (f.p + 1), 2)

function HD.initialize!{T<:AbstractFloat}(f::CDDirectDifferenceLoss{T}, x::SparseVector{T})
  # compute residuals for the loss

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  p = f.p

  for ac=1:p, ar=1:p
      @inbounds A[ar,ac] = _mul_Σx_Δ_Σy(Σx, x, Σy, ar, ac)
  end

  nothing
end

function HD.gradient{T <: AbstractFloat}(
  f::CDDirectDifferenceLoss{T},
  x::SparseVector{T},
  j::Int64)

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  p = f.p

  ri, ci = ind2subLowerTriangular(p, j)
  @inbounds return (A[ri,ci] + A[ci,ri]) / 2. - (Σy[ri, ci] - Σx[ri, ci])
end

function HD.quadraticApprox{T<:AbstractFloat}(
  f::CDDirectDifferenceLoss{T},
  x::SparseVector{T},
  j::Int64)

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  p = f.p

  ri, ci = ind2subLowerTriangular(p, j)

  a = zero(T)
  b = zero(T)
  if ri == ci
    @inbounds a = Σx[ri,ri] * Σy[ri,ri]
    @inbounds b = A[ri,ri] + Σx[ri,ri] - Σy[ri,ri]
    b = b / a
  else
    @inbounds a = (Σx[ri,ri]*Σy[ci,ci] + Σx[ci,ci]*Σy[ri,ri]) / 2. + Σx[ri,ci]*Σy[ri,ci]
    @inbounds b = A[ri,ci] + A[ci,ri] - 2.*(Σy[ri,ci] - Σx[ri,ci])
    b = b / (2. * a)
  end
  (a, b)
end

function HD.updateSingle!{T<:AbstractFloat}(
  f::CDDirectDifferenceLoss{T},
  x::SparseVector{T},
  h::T,
  j::Int64)

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  p = f.p

  ri, ci = ind2subLowerTriangular(p, j)

  if ri == ci
    for ac=1:p, ar=1:p
      @inbounds A[ar, ac] += h * Σx[ar, ri] * Σy[ac, ri]
    end
  else
    for ac=1:p, ar=1:p
      @inbounds A[ar, ac] += h * (Σx[ar, ri] * Σy[ci, ac] + Σx[ar, ci] * Σy[ri, ac])
    end
  end
  nothing
end

HD.updateAfterActive!{T<:AbstractFloat}(
  f::CDDirectDifferenceLoss{T},
  x::SparseVector{T}) = nothing


#####################################################
#
# Direct difference estimation using Active Shooting
#
#####################################################

differencePrecisionActiveShooting!(
  x::SparseVector,
  Σx::StridedMatrix,
  Σy::StridedMatrix,
  λ::StridedVector,
  options::HD.CDOptions=HD.CDOptions()) =
  HD.coordinateDescent!(x, CDDirectDifferenceLoss(Σx, Σy), λ, options)


function differencePrecisionActiveShooting(
  Σx::StridedMatrix,
  Σy::StridedMatrix,
  λ::StridedVector,
  options::HD.CDOptions=HD.CDOptions())

  f = CDDirectDifferenceLoss(Σx, Σy)
  HD.coordinateDescent!(spzeros(HD.numCoordinates(f)), f, λ, options)
end
