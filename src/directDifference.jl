# this file implements direct difference estimation for Δ = Ωx - Ωy

####################################
#
# loss function evaluation
#
####################################

"""
  Computes vecnorm(Σx⋅Δ⋅Σy - Σy + Σx, p)
"""
function diffLoss{T<:AbstractFloat}(
  Σx::StridedMatrix{T},
  x::SparseIterate{T},
  Σy::StridedMatrix{T},
  p::Real)

  d = size(Σx, 1)
  if p == 2.
    v = zero(T)
    for c=1:d, r=1:d
      v += abs2( _mul_Σx_Δ_Σy(Σx, Δ, Σy, r, c) - Σy[r, c] + Σy[r, c] )
    end
    return sqrt(v)
  elseif p==Inf
    v = zero(T)
    for c=1:d, r=1:d
      t = abs( _mul_Σx_Δ_Σy(Σx, Δ, Σy, r, c) - Σy[r, c] + Σy[r, c] )
      if t > v
        v = t
      end
    end
  else
    throw(ArgumentError("p should be 2 or Inf"))
  end
end

"""
  Computes vecnorm(Σx⋅(Δ+UU')⋅Σy - Σy + Σx, p)
"""
function diffLoss{T<:AbstractFloat}(
  Σx::StridedMatrix{T},
  x::SparseIterate{T},
  U::StridedMatrix{T},
  Σy::StridedMatrix{T},
  p::Real)

  d = size(Σx, 1)
  if p == 2.
    v = zero(T)
    for c=1:d, r=1:d
      v += abs2( _mul_Σx_Δ_Σy(Σx, Δ, Σy, r, c) - Σy[r, c] + Σy[r, c] )
    end
    return sqrt(v)
  elseif p==Inf
    v = zero(T)
    for c=1:d, r=1:d
      t = abs( _mul_Σx_Δ_Σy(Σx, Δ, Σy, r, c) - Σy[r, c] + Σy[r, c] )
      if t > v
        v = t
      end
    end
  else
    throw(ArgumentError("p should be 2 or Inf"))
  end
end




####################################
#
# loss tr(Σx⋅Δ⋅Σy⋅Δ)/2 + tr(Δ(Σy-Σx))
#
####################################
struct CDDirectDifferenceLoss{T<:AbstractFloat, S} <: CoordinateDifferentiableFunction
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

function HD.initialize!{T<:AbstractFloat}(f::CDDirectDifferenceLoss{T}, x::SparseIterate{T})
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
  x::SparseIterate{T},
  j::Int64)

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  p = f.p

  ri, ci = ind2subLowerTriangular(p, j)
  @inbounds return (A[ri,ci] + A[ci,ri]) / 2. - (Σy[ri, ci] - Σx[ri, ci])
end

# function HD.quadraticApprox{T<:AbstractFloat}(
#   f::CDDirectDifferenceLoss{T},
#   x::SparseIterate{T},
#   j::Int64)
#
#   Σx = f.Σx
#   Σy = f.Σy
#   A = f.A
#   p = f.p
#
#   ri, ci = ind2subLowerTriangular(p, j)
#
#   a = zero(T)
#   b = zero(T)
#   if ri == ci
#     @inbounds a = Σx[ri,ri] * Σy[ri,ri]
#     @inbounds b = A[ri,ri] + Σx[ri,ri] - Σy[ri,ri]
#     b = b / a
#   else
#     @inbounds a = (Σx[ri,ri]*Σy[ci,ci] + Σx[ci,ci]*Σy[ri,ri]) / 2. + Σx[ri,ci]*Σy[ri,ci]
#     @inbounds b = A[ri,ci] + A[ci,ri] - 2.*(Σy[ri,ci] - Σx[ri,ci])
#     b = b / (2. * a)
#   end
#   (a, b)
# end
#
# function HD.updateSingle!{T<:AbstractFloat}(
#   f::CDDirectDifferenceLoss{T},
#   x::SparseIterate{T},
#   h::T,
#   j::Int64)
#
#   Σx = f.Σx
#   Σy = f.Σy
#   A = f.A
#   p = f.p
#
#   ri, ci = ind2subLowerTriangular(p, j)
#
#
#   for ac=1:p, ar=1:p
#     @inbounds A[ar, ac] += (ri == ci) ? (h * Σx[ar, ri] * Σy[ac, ri]) :
#           (h * (Σx[ar, ri] * Σy[ci, ac] + Σx[ar, ci] * Σy[ri, ac]))
#   end
#   nothing
# end



#####################################################
#
# Direct difference estimation using Active Shooting
#
#####################################################

differencePrecisionActiveShooting!(
  x::SparseIterate,
  Σx::StridedMatrix,
  Σy::StridedMatrix,
  λ::StridedVector,
  options=CDOptions()) =
  coordinateDescent!(x, CDDirectDifferenceLoss(Σx, Σy), λ, options)


@inline function differencePrecisionActiveShooting(
  Σx::StridedMatrix,
  Σy::StridedMatrix,
  λ::StridedVector,
  options=CDOptions())

  f = CDDirectDifferenceLoss(Σx, Σy)
  HD.coordinateDescent!(SparseIterate(HD.numCoordinates(f)), f, λ, options)
end
