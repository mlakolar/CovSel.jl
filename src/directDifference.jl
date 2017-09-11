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
  Σx::Symmetric{T},
  Δ::SymmetricSparseIterate{T},
  Σy::Symmetric{T},
  p::Real)

  d = size(Σx, 1)
  if p == 2.
    v = zero(T)
    for c=1:d, r=1:d
      v += abs2( A_mul_X_mul_B_rc(Σx, Δ, Σy, r, c) - Σy[r, c] + Σy[r, c] )
    end
    return sqrt(v)
  elseif p==Inf
    v = zero(T)
    for c=1:d, r=1:d
      t = abs( A_mul_X_mul_B_rc(Σx, Δ, Σy, r, c) - Σy[r, c] + Σy[r, c] )
      if t > v
        v = t
      end
    end
    return v
  else
    throw(ArgumentError("p should be 2 or Inf"))
  end
end

"""
  Computes vecnorm(Σx⋅(Δ+UU')⋅Σy - Σy + Σx, p)
"""
function diffLoss{T<:AbstractFloat}(
  Σx::Symmetric{T},
  Δ::SymmetricSparseIterate{T},
  U::StridedMatrix{T},
  Σy::Symmetric{T},
  p::Real)

  d = size(Σx, 1)
  if p == 2.
    v = zero(T)
    for c=1:d, r=1:d
      v += abs2( A_mul_X_mul_B_rc(Σx, Δ, Σy, r, c) + A_mul_UUt_mul_B_rc(Σx, U, Σy, r, c) - Σy[r, c] + Σy[r, c] )
    end
    return sqrt(v)
  elseif p==Inf
    v = zero(T)
    for c=1:d, r=1:d
      t = abs( A_mul_X_mul_B_rc(Σx, Δ, Σy, r, c) + A_mul_UUt_mul_B_rc(Σx, U, Σy, r, c) - Σy[r, c] + Σy[r, c] )
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
# loss tr(Σx⋅Δ⋅Σy⋅Δ)/2 - tr(Δ(Σy-Σx))
#
####################################
struct CDDirectDifferenceLoss{T<:AbstractFloat, S} <: CoordinateDifferentiableFunction
  Σx::Symmetric{T, S}
  Σy::Symmetric{T, S}
  A::Matrix{T}    # stores Σx⋅Δ⋅Σy
  p::Int64
end

function CDDirectDifferenceLoss(Σx::Symmetric{T,S}, Σy::Symmetric{T,S}) where {T<:AbstractFloat} where S
  (issymmetric(Σx) && issymmetric(Σy)) || throw(DimensionMismatch())
  (p = size(Σx, 1)) == size(Σy, 1) || throw(DimensionMismatch())
  CDDirectDifferenceLoss{T,S}(Σx, Σy, zeros(T, p, p), p)
end

CoordinateDescent.numCoordinates(f::CDDirectDifferenceLoss) = div(f.p * (f.p + 1), 2)

function CoordinateDescent.initialize!{T<:AbstractFloat}(f::CDDirectDifferenceLoss{T}, x::SymmetricSparseIterate{T})
  # compute residuals for the loss

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  p = f.p

  for ac=1:p, ar=1:p
      @inbounds A[ar,ac] = A_mul_X_mul_B_rc(Σx, x, Σy, ar, ac)
  end

  nothing
end

function CoordinateDescent.gradient{T <: AbstractFloat}(
  f::CDDirectDifferenceLoss{T},
  x::SymmetricSparseIterate{T},
  j::Int64)

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  p = f.p

  ri, ci = ind2subLowerTriangular(p, j)
  @inbounds return (A[ri,ci] + A[ci,ri]) / 2. - (Σy[ri, ci] - Σx[ri, ci])
end


function CoordinateDescent.descendCoordinate!{T <: AbstractFloat}(
  f::CDDirectDifferenceLoss{T},
  g::ProxL1{T},
  x::SymmetricSparseIterate{T},
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
    @inbounds b = Σy[ri,ri] - Σx[ri,ri] - A[ri,ri]
  else
    @inbounds a = (Σx[ri,ri]*Σy[ci,ci] + Σx[ci,ci]*Σy[ri,ri]) + 2.*Σx[ri,ci]*Σy[ri,ci]
    @inbounds b = 2.*(Σy[ri,ci] - Σx[ri,ci]) - A[ri,ci] - A[ci,ri]
  end
  oldVal = x[ri, ci]
  x[ri, ci] += b / a
  newVal = cdprox!(g, x, j, ri == ci ? 1. / a : 2. / a)
  h = newVal - oldVal

  # update internals
  for ac=1:p, ar=1:p
    @inbounds A[ar, ac] += (ri == ci) ? (h * Σx[ar, ri] * Σy[ac, ri]) :
          (h * (Σx[ar, ri] * Σy[ci, ac] + Σx[ar, ci] * Σy[ri, ac]))
  end
  h
end


#####################################################
#
# Direct difference estimation using Active Shooting
#
#####################################################

differencePrecisionActiveShooting!(
  x::SymmetricSparseIterate,
  Σx::Symmetric,
  Σy::Symmetric,
  g::ProxL1,
  options=CDOptions()) =
  coordinateDescent!(x, CDDirectDifferenceLoss(Σx, Σy), g, options)


@inline function differencePrecisionActiveShooting(
  Σx::Symmetric,
  Σy::Symmetric,
  g::ProxL1,
  options=CDOptions())

  f = CDDirectDifferenceLoss(Σx, Σy)
  coordinateDescent!(SymmetricSparseIterate(f.p), f, g, options)
end

function differencePrecisionRefit(
  Σx::Symmetric{T},
  Σy::Symmetric{T},
  S::Vector{Int64},
  options=CDOptions()
  ) where T

  f = CDDirectDifferenceLoss(Σx, Σy)
  t = CoordinateDescent.numCoordinates(f)
  ω = ones(T, t) * 1e10
  for i=S
    r,c = ind2sub(Σx, i)
    if r >= c
      ind = sub2indLowerTriangular(f.p, r, c)
      ω[ind] = 0.
    end
  end
  g = ProxL1(one(T), ω)
  x = SymmetricSparseIterate(f.p)
  coordinateDescent!(x, f, g, options)
end



####################################
#
# loss tr(Σx⋅θ⋅Σy⋅θ)/2 - tr(Θ⋅E_ab)    ---> computes inverse of \
#
####################################
struct CDInverseKroneckerLoss{T<:AbstractFloat, S} <: CoordinateDifferentiableFunction
  Σx::Symmetric{T, S}
  Σy::Symmetric{T, S}
  A::Matrix{T}    # stores Σx⋅Θ⋅Σy
  a::Int
  b::Int
  p::Int64
end

function CDInverseKroneckerLoss(Σx::Symmetric{T,S}, Σy::Symmetric{T,S}, a::Int, b::Int) where {T<:AbstractFloat} where S
  (issymmetric(Σx) && issymmetric(Σy)) || throw(DimensionMismatch())
  (p = size(Σx, 1)) == size(Σy, 1) || throw(DimensionMismatch())
  CDInverseKroneckerLoss{T,S}(Σx, Σy, zeros(T, p, p), a, b, p)
end

CoordinateDescent.numCoordinates(f::CDInverseKroneckerLoss) = f.p*f.p
function CoordinateDescent.initialize!(f::CDInverseKroneckerLoss, x::SparseIterate)
# compute residuals for the loss

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  p = f.p

  for ac=1:p, ar=1:p
      @inbounds A[ar,ac] = A_mul_X_mul_B_rc(Σx, x, Σy, ar, ac)
  end

  nothing
end

function CoordinateDescent.gradient(
  f::CDInverseKroneckerLoss{T},
  x::SparseIterate{T},
  j::Int64) where {T <: AbstractFloat}

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  ri, ci = ind2sub(Σx, j)
  @inbounds v = A[ri,ci]
  return ri == f.a && ci == f.b ? v - 1. : v
end


function CoordinateDescent.descendCoordinate!(
  f::CDInverseKroneckerLoss{T},
  g::ProxL1{T},
  x::SparseIterate{T},
  j::Int64) where {T <: AbstractFloat}

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  p = size(Σx, 1)

  ri, ci = ind2sub(Σx, j)

  a = zero(T)
  b = zero(T)
  @inbounds a  = Σx[ri,ri] * Σy[ci,ci]
  @inbounds b = A[ri,ci]
  b = ri == f.a && ci == f.b ? b - 1. : b

  @inbounds oldVal = x[j]
  a = one(T) / a
  @inbounds x[j] -= b * a
  newVal = cdprox!(g, x, j, a)
  h = newVal - oldVal

  # update internals
  for ac=1:p, ar=1:p
    @inbounds A[ar, ac] += h * Σx[ar, ri] * Σy[ci, ac]
  end
  h
end
