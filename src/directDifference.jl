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
# loss tr(Σx⋅θ⋅Σy⋅θ)/2 - tr(Θ⋅E_ab)    ---> computes inverse of (Σy ⊗ Σx)
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




########################################################################
#
# Direct difference estimation using iterative hard thresholding
#
########################################################################

# A = Σx⋅(Δ + (UV' + V'U) / 2.)⋅Σy                 --- this is ensured by the algorithm
function differencePrecision_objective(
  A::StridedMatrix{T},
  Σx::Symmetric{T}, Σy::Symmetric{T},
  Δ::SymmetricSparseIterate{T}, U::StridedMatrix{T}, V::StridedMatrix{T}) where {T<:AbstractFloat}

  p, r = size(U)
  v = zero(T)
  # add sparse part
  for c=1:p
    for j=1:p
      t = Δ[j, c]
      v += (A[j,c]/2. + Σx[j,c] - Σy[j,c])*t
    end
  end

  # add low rank part
  for c=1:r
    for a=1:p, b=1:p
      v += (A[a,b] / 2. + Σx[a,b] - Σy[a,b])*U[a,c]*V[b,c] / 2.
      v += (A[a,b] / 2. + Σx[a,b] - Σy[a,b])*U[b,c]*V[a,c] / 2.
    end
  end

  v
end

differencePrecision_objective{T<:AbstractFloat}(
  Σx::Symmetric{T}, Σy::Symmetric{T},
  Δ::SymmetricSparseIterate{T}) =
    trace((A_mul_X_mul_B(Σx, Δ, Σy) / 2 - (Σy - Σx)) * Δ)


function diffPrecision_grad_delta!{T<:AbstractFloat}(
  grad_out::StridedMatrix{T},
  A::StridedMatrix{T},
  Σx::Symmetric{T}, Σy::Symmetric{T})

  p = size(Σx, 1)

  # (a, b) indices of updated grad_out[a,b] element
  for a=1:p, b=a:p
    grad_out[a,b] = (A[a,b] + A[b,a])/2. + Σx[a,b] - Σy[a,b]
    # update the other symmetric element
    if a != b
      grad_out[b,a] = grad_out[a,b]
    end
  end

  grad_out
end


function differencePrecisionIHT_grad_u!(grad_U, grad_Δ, A, Σx, Σy, U)
  diffPrecision_grad_delta!(grad_Δ, A, Σx, Σy)
  scale!(grad_Δ, 2.)
  A_mul_B!(grad_U, grad_Δ, U)
end


# η is the step size
# s is the target sparsity
function differencePrecisionIHT{T<:AbstractFloat}(
  Σx::Symmetric{T}, Σy::Symmetric{T},
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
  fv = differencePrecision_objective(Σx, Σy, Δ)
  push!(fvals, fv)
  for iter=1:maxIter
    # update Δ
    diffPrecision_grad_delta!(gD, Σx, Σy, Δ)
    Δ .-= η .* gD
    HardThreshold!(Δ, s)

    # check for convergence
    fv_new = differencePrecision_objective(Σx, Σy, Δ)
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
function differenceLatentPrecisionIHT!{T<:AbstractFloat}(
  Δ::SymmetricSparseIterate{T}, L::StridedMatrix{T},                 # these are pre-allocated
  U::StridedMatrix{T}, gD::StridedMatrix{T}, gU::StridedMatrix{T},   # these are pre-allocated
  Σx::Symmetric{T}, Σy::Symmetric{T},
  ηΔ, ηU, s, r;
  options::IHTOptions=IHTOptions(), callback=nothing)

  epsTol, maxIter, checkEvery = options.epsTol, options.maxIter, options.checkEvery

  p = size(Σx, 1)
  A = A_mul_UUt_mul_B(Σx, U, Σy)
  for c=1:p, r=1:p
    A[r,c] += A_mul_X_mul_B_rc(Σx, Δ, Σy, r, c)
  end

  fvals = []
  # gradient descent
  #
  fv = differencePrecision_objective(A, Σx, Σy, Δ, U)
  push!(fvals, fv)
  for iter=1:maxIter
    # update Δ
    diffPrecision_grad_delta!(gD, A, Σx, Σy)
    max_gD = maximum( abs.(gD) )
    for c=1:p, r=c:p
      Δ[r,c] = Δ[r,c]  - ηΔ * gD[r,c]
    end
    HardThreshold!(Δ, s)
    # update A
    for c=1:p, r=1:p
      A[r,c] = A_mul_X_mul_B_rc(Σx, Δ, Σy, r, c) + A_mul_UUt_mul_B_rc(Σx, U, Σy, r, c)
    end

    # update U and L
    differencePrecisionIHT_grad_u!(gU, gD, A, Σx, Σy, U)
    @. U -= ηU * gU
    A_mul_Bt!(L, U, U)
    max_gU = maximum( abs.(gU) )
    # update A
    for c=1:p, r=1:p
      A[r,c] = A_mul_X_mul_B_rc(Σx, Δ, Σy, r, c) + A_mul_UUt_mul_B_rc(Σx, U, Σy, r, c)
    end

    if callback != nothing
      callback(Δ, L)
    end

    done = max(max_gD, max_gU) < epsTol
    # check for convergence
    if mod(iter, checkEvery) == 0
      fv_new = differencePrecision_objective(A, Σx, Σy, Δ, U)
      push!(fvals, fv_new)
      done = abs(fv_new - fv) <= epsTol
      fv = fv_new
    end
    done && break
  end

  (Δ, L, U, fvals, iter)
end



function differenceLatentPrecisionIHT_init(
  Σx::Symmetric{T}, Σy::Symmetric{T},
  s::Int64, r::Int64) where{T}

  size(Σx) == size(Σy) || throw(DimensionMismatch())
  p = size(Σx, 1)

  gD = zeros(p, p)
  gU = zeros(p, r)
  Δ = SymmetricSparseIterate(T, p)
  L = zeros(p, p)

  # initialize Δ, L, U
  #
  tmp = inv(Σx) - inv(Σy)
  HardThreshold!(Δ, tmp, s)
  for colInd=1:p, rowInd=colInd:p
    tmp[rowInd,colInd] = tmp[rowInd,colInd] - Δ[rowInd, colInd]
    if rowInd != colInd
      tmp[colInd,rowInd] = tmp[rowInd,colInd]
    end
  end
  U, d, V = svd(tmp)
  U = U[:,1:r] .* sqrt.(d[1:r])'
  L = U * U'

  (Δ, L, U, gD, gU)
end






##
