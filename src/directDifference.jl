# this file implements direct difference estimation for Δ = Ωx - Ωy

####################################
#
# loss function evaluation
#
####################################

"""
    diffLoss(Σx, Δ, Σy, p)
    diffLoss(Σx, Δ, U, Σy, p)

Entrywise `p`-norm (`p` is 2 or `Inf`) of the residual `Σx⋅Δ⋅Σy - Σy + Σx` of
the direct difference estimating equation, or of `Σx⋅(Δ + UU')⋅Σy - Σy + Σx`
when the low rank factor `U` is given.
"""
diffLoss(Σx::Symmetric{T}, Δ::SymmetricSparseIterate{T}, Σy::Symmetric{T}, p::Real) where {T<:AbstractFloat} =
  _diffLoss((r, c) -> A_mul_X_mul_B_rc(Σx, Δ, Σy, r, c) - Σy[r, c] + Σx[r, c], size(Σx, 1), p)

diffLoss(Σx::Symmetric{T}, Δ::SymmetricSparseIterate{T}, U::StridedMatrix{T}, Σy::Symmetric{T}, p::Real) where {T<:AbstractFloat} =
  _diffLoss((r, c) -> A_mul_X_mul_B_rc(Σx, Δ, Σy, r, c) + A_mul_UUt_mul_B_rc(Σx, U, Σy, r, c) - Σy[r, c] + Σx[r, c], size(Σx, 1), p)

function _diffLoss(residual, d::Int, p::Real)
  if p == 2
    return sqrt(sum(abs2(residual(r, c)) for c=1:d for r=1:d))
  elseif p == Inf
    return maximum(abs(residual(r, c)) for c=1:d for r=1:d)
  else
    throw(ArgumentError("p should be 2 or Inf"))
  end
end




####################################
#
# loss tr(Σx⋅Δ⋅Σy⋅Δ)/2 - tr(Δ(Σy-Σx))
#
####################################
"""
    CDDirectDifferenceLoss(Σx::Symmetric, Σy::Symmetric)

The direct difference loss `tr(Σx⋅Δ⋅Σy⋅Δ)/2 - tr(Δ⋅(Σy - Σx))` as a
`CoordinateDifferentiableFunction` of a `SymmetricSparseIterate` `Δ`. Its
minimizer with population covariances is `Δ = Ωx - Ωy`, the difference of the
precision matrices (Zhao, Cai and Li, 2014). The coordinates are the entries of
the lower triangle of `Δ`.
"""
struct CDDirectDifferenceLoss{T<:AbstractFloat, S} <: CoordinateDifferentiableFunction
  Σx::Symmetric{T, S}
  Σy::Symmetric{T, S}
  X::Matrix{T}    # Σx and Σy as dense matrices; indexing a Symmetric branches on
  Y::Matrix{T}    # every access, which dominates the updates below
  A::Matrix{T}    # stores Σx⋅Δ⋅Σy
  p::Int64
end

function CDDirectDifferenceLoss(Σx::Symmetric{T,S}, Σy::Symmetric{T,S}) where {T<:AbstractFloat} where S
  (issymmetric(Σx) && issymmetric(Σy)) || throw(DimensionMismatch())
  (p = size(Σx, 1)) == size(Σy, 1) || throw(DimensionMismatch())
  CDDirectDifferenceLoss{T,S}(Σx, Σy, Matrix(Σx), Matrix(Σy), zeros(T, p, p), p)
end

CoordinateDescent.numCoordinates(f::CDDirectDifferenceLoss) = div(f.p * (f.p + 1), 2)

# The loss as a function of the coordinate for Δ[r, c] (and Δ[c, r]) is
# quadratic: this is its second derivative, and (Σy - Σx)[r, c], doubled off the
# diagonal, is minus its derivative at Δ = 0.
@inline function _curvature(X, Y, r, c)
  @inbounds r == c ? X[r, r] * Y[r, r] : (X[r, r] * Y[c, c] + X[c, c] * Y[r, r]) + 2 * X[r, c] * Y[r, c]
end

# Scales for the stopping rule of CoordinateDescent 0.5 and later (relativeTol),
# as for its quadratic loss: the change of coordinate k is measured as
# |Δxₖ| √aₖ, with aₖ the curvature, and compared with optTol ⋅ maxₖ |bₖ| / √aₖ,
# with -bₖ the derivative at zero. Neither depends on the units of Σx and Σy.
if isdefined(CoordinateDescent, :coordinateScale)
  CoordinateDescent.coordinateScale(f::CDDirectDifferenceLoss, k::Int) =
    sqrt(_curvature(f.X, f.Y, ind2subLowerTriangular(f.p, k)...))

  function CoordinateDescent.responseScale(f::CDDirectDifferenceLoss{T}) where {T<:AbstractFloat}
    X, Y, p = f.X, f.Y, f.p
    s = zero(T)
    for c=1:p, r=c:p
      b = (r == c ? 1 : 2) * (Y[r, c] - X[r, c])
      s = max(s, abs(b) / sqrt(_curvature(X, Y, r, c)))
    end
    s
  end
end

function CoordinateDescent.initialize!(f::CDDirectDifferenceLoss{T}, x::SymmetricSparseIterate{T}) where {T<:AbstractFloat}
  copyto!(f.A, A_mul_X_mul_B(f.Σx, x, f.Σy))
  nothing
end

function CoordinateDescent.gradient(
  f::CDDirectDifferenceLoss{T},
  x::SymmetricSparseIterate{T},
  j::Int64) where {T<:AbstractFloat}

  X, Y, A = f.X, f.Y, f.A
  ri, ci = ind2subLowerTriangular(f.p, j)
  @inbounds return (A[ri,ci] + A[ci,ri]) / 2 - (Y[ri, ci] - X[ri, ci])
end


function CoordinateDescent.descendCoordinate!(
  f::CDDirectDifferenceLoss{T},
  g::ProxL1{T},
  x::SymmetricSparseIterate{T},
  j::Int64) where {T<:AbstractFloat}

  X, Y, A, p = f.X, f.Y, f.A, f.p
  ri, ci = ind2subLowerTriangular(p, j)

  a = _curvature(X, Y, ri, ci)
  if ri == ci
    @inbounds b = Y[ri,ri] - X[ri,ri] - A[ri,ri]
  else
    @inbounds b = 2 * (Y[ri,ci] - X[ri,ci]) - A[ri,ci] - A[ci,ri]
  end
  oldVal = x[ri, ci]
  x[ri, ci] += b / a
  newVal = cdprox!(g, x, j, ri == ci ? one(T) / a : 2 / a)
  h = newVal - oldVal

  # nothing to update when the coordinate did not move, as for most
  # coordinates that stay at zero in a full pass
  iszero(h) && return h

  # update A = Σx⋅Δ⋅Σy for the change h in Δ[ri, ci] and Δ[ci, ri]:
  # A += h⋅(Σx[:, ri]⋅Σy[ci, :] + Σx[:, ci]⋅Σy[ri, :]), or h⋅Σx[:, ri]⋅Σy[ri, :] on the diagonal
  if ri == ci
    @inbounds for ac=1:p
      t = h * Y[ac, ri]
      @simd for ar=1:p
        A[ar, ac] += X[ar, ri] * t
      end
    end
  else
    @inbounds for ac=1:p
      t1 = h * Y[ac, ci]
      t2 = h * Y[ac, ri]
      @simd for ar=1:p
        A[ar, ac] += X[ar, ri] * t1 + X[ar, ci] * t2
      end
    end
  end
  h
end


#####################################################
#
# Direct difference estimation using Active Shooting
#
#####################################################

"""
    differencePrecisionActiveShooting(Σx, Σy, g::ProxL1, options=CDOptions())
    differencePrecisionActiveShooting!(x, Σx, Σy, g::ProxL1, options=CDOptions())

Direct estimation of the difference of precision matrices `Δ = Ωx - Ωy`
(Zhao, Cai and Li, 2014) by active shooting coordinate descent:

    minimize  tr(Σx⋅Δ⋅Σy⋅Δ)/2 - tr(Δ⋅(Σy - Σx)) + g(Δ)

where `g` is a (weighted) ℓ1 penalty on the lower triangle of `Δ`. The
in-place version warm starts from the `SymmetricSparseIterate` `x`.
Returns the `SymmetricSparseIterate` estimate.
"""
differencePrecisionActiveShooting!(
  x::SymmetricSparseIterate,
  Σx::Symmetric,
  Σy::Symmetric,
  g::ProxL1,
  options=CDOptions()) =
  coordinateDescent!(x, CDDirectDifferenceLoss(Σx, Σy), g, options)


function differencePrecisionActiveShooting(
  Σx::Symmetric,
  Σy::Symmetric,
  g::ProxL1,
  options=CDOptions())

  f = CDDirectDifferenceLoss(Σx, Σy)
  coordinateDescent!(SymmetricSparseIterate(eltype(Σx), f.p), f, g, options)
end

"""
    differencePrecisionRefit(Σx, Σy, S, options=CDOptions())

Minimize the direct difference loss `tr(Σx⋅Δ⋅Σy⋅Δ)/2 - tr(Δ⋅(Σy - Σx))` over
symmetric `Δ` supported on `S`, a collection of linear or Cartesian indices into
a `p × p` matrix (an entry and its transpose are the same coordinate).
"""
function differencePrecisionRefit(
  Σx::Symmetric{T},
  Σy::Symmetric{T},
  S,
  options=CDOptions()
  ) where T

  f = CDDirectDifferenceLoss(Σx, Σy)
  t = CoordinateDescent.numCoordinates(f)
  # entries outside of S get a huge penalty
  ω = fill(T(1e10), t)
  for i=S
    r, c = Tuple(CartesianIndices(Σx)[i])
    ω[r >= c ? sub2indLowerTriangular(f.p, r, c) : sub2indLowerTriangular(f.p, c, r)] = zero(T)
  end
  g = ProxL1(one(T), ω)
  x = SymmetricSparseIterate(T, f.p)
  coordinateDescent!(x, f, g, options)
end








########################################################################
#
# Direct difference estimation using iterative hard thresholding
#
########################################################################

# A = Σx⋅(Δ + (UV' + VU') / 2.)⋅Σy                 --- this is ensured by the algorithm
# returns tr(A⋅Θ)/2 - tr(Θ⋅(Σy - Σx)) with Θ = Δ + (UV' + VU') / 2
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
      v += (A[j,c]/2 + Σx[j,c] - Σy[j,c])*t
    end
  end

  # add low rank part
  for c=1:r
    for a=1:p, b=1:p
      v += (A[a,b] / 2 + Σx[a,b] - Σy[a,b])*U[a,c]*V[b,c] / 2
      v += (A[a,b] / 2 + Σx[a,b] - Σy[a,b])*U[b,c]*V[a,c] / 2
    end
  end

  v
end

differencePrecision_objective(A::StridedMatrix{T}, Σx::Symmetric{T}, Σy::Symmetric{T},
  Δ::SymmetricSparseIterate{T}, U::StridedMatrix{T}) where {T<:AbstractFloat} =
  differencePrecision_objective(A, Σx, Σy, Δ, U, U)

differencePrecision_objective(A::StridedMatrix{T}, Σx::Symmetric{T}, Σy::Symmetric{T},
  Δ::SymmetricSparseIterate{T}) where {T<:AbstractFloat} =
  differencePrecision_objective(A, Σx, Σy, Δ, zeros(T, size(A, 1), 0))

differencePrecision_objective(Σx::Symmetric{T}, Σy::Symmetric{T}, Δ::SymmetricSparseIterate{T}) where {T<:AbstractFloat} =
  differencePrecision_objective(A_mul_X_mul_B(Σx, Δ, Σy), Σx, Σy, Δ)


# gradient (A + A')/2 + Σx - Σy of the loss with respect to Θ; the IHT solvers
# pass Σx and Σy as dense matrices, which are faster to index than a Symmetric
function diffPrecision_grad_delta!(
  grad_out::StridedMatrix{T},
  A::StridedMatrix{T},
  Σx::AbstractMatrix{T}, Σy::AbstractMatrix{T}) where {T<:AbstractFloat}

  p = size(Σx, 1)

  # (a, b) indices of updated grad_out[a,b] element
  for a=1:p, b=a:p
    grad_out[a,b] = (A[a,b] + A[b,a])/2 + Σx[a,b] - Σy[a,b]
    # update the other symmetric element
    if a != b
      grad_out[b,a] = grad_out[a,b]
    end
  end

  grad_out
end


# gradient with respect to U, where Θ = Δ + UU'
function differencePrecisionIHT_grad_u!(grad_U, grad_Δ, A, Σx, Σy, U)
  diffPrecision_grad_delta!(grad_Δ, A, Σx, Σy)
  rmul!(grad_Δ, 2)
  mul!(grad_U, grad_Δ, U)
end

# Δ ← Δ - η⋅G on the lower triangle
function _gradientStep!(Δ::SymmetricSparseIterate{T}, η, G::StridedMatrix{T}) where {T<:AbstractFloat}
  p = size(G, 1)
  for c=1:p, r=c:p
    Δ[r, c] = Δ[r, c] - T(η) * G[r, c]
  end
  Δ
end

# D = Δ as a dense symmetric matrix
function _densify!(D::StridedMatrix{T}, Δ::SymmetricSparseIterate{T}) where {T<:AbstractFloat}
  fill!(D, zero(T))
  for k=1:nnz(Δ)
    r, c = ind2subLowerTriangular(Δ.p, Δ.nzval2ind[k])
    D[r, c] = D[c, r] = Δ.nzval[k]
  end
  D
end

# A = Σx⋅D⋅Σy without allocating, with tmp as storage
function _sandwich!(A::StridedMatrix{T}, tmp::StridedMatrix{T}, Σx::Symmetric{T}, D::StridedMatrix{T}, Σy::Symmetric{T}) where {T<:AbstractFloat}
  mul!(tmp, Σx, D)
  mul!(A, tmp, Σy)
end

# Θ = Δ + L as a dense matrix and A = Σx⋅Θ⋅Σy
function _latentSandwich!(A, tmp, Θ, Σx, Δ, L, Σy)
  _densify!(Θ, Δ)
  Θ .+= L
  _sandwich!(A, tmp, Σx, Θ, Σy)
end

# the loss tr(A⋅Θ)/2 - tr(Θ⋅(Σy - Σx)) for a dense symmetric Θ with A = Σx⋅Θ⋅Σy
function _objective(A::StridedMatrix{T}, X::StridedMatrix{T}, Y::StridedMatrix{T}, Θ::StridedMatrix{T}) where {T<:AbstractFloat}
  v = zero(T)
  @inbounds for c in axes(Θ, 2), r in axes(Θ, 1)
    v += (A[r, c] / 2 + X[r, c] - Y[r, c]) * Θ[r, c]
  end
  v
end


"""
    differencePrecisionIHT(Σx, Σy, η, s; epsTol=1e-5, maxIter=1000)

Direct estimation of the difference of precision matrices `Δ = Ωx - Ωy` with
at most `s` nonzero entries in its lower triangle, by projected gradient
descent on `tr(Σx⋅Δ⋅Σy⋅Δ)/2 - tr(Δ⋅(Σy - Σx))` with step size `η` and hard
thresholding. Stops when the objective changes by at most `epsTol`.
Returns `(Δ, fvals)` with `Δ` a `SymmetricSparseIterate` and `fvals` the
objective values.
"""
function differencePrecisionIHT(
  Σx::Symmetric{T}, Σy::Symmetric{T},
  η::Real, s::Int;
  epsTol=1e-5, maxIter=1000) where {T<:AbstractFloat}

  size(Σx) == size(Σy) || throw(DimensionMismatch())

  p = size(Σx, 1)

  X, Y = Matrix(Σx), Matrix(Σy)
  gD = zeros(T, p, p)
  A = zeros(T, p, p)      # Σx⋅Δ⋅Σy
  D = zeros(T, p, p)      # Δ as a dense matrix
  tmp = zeros(T, p, p)
  Δ = SymmetricSparseIterate(T, p)

  fvals = T[]
  # gradient descent
  #
  fv = _objective(A, X, Y, D)
  push!(fvals, fv)
  for iter=1:maxIter
    # update Δ
    diffPrecision_grad_delta!(gD, A, X, Y)
    _gradientStep!(Δ, η, gD)
    HardThreshold!(Δ, s)
    _sandwich!(A, tmp, Σx, _densify!(D, Δ), Σy)

    # check for convergence
    fv_new = _objective(A, X, Y, D)
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



"""
    differenceLatentPrecisionIHT(Σx, Σy, ηΔ, ηU, s, r; options=IHTOptions(), callback=nothing)
    differenceLatentPrecisionIHT!(Δ, L, U, gD, gU, Σx, Σy, ηΔ, ηU, s, r; options=IHTOptions(), callback=nothing)

Estimate the difference of precision matrices as `Δ + L`, with `Δ` sparse
(at most `s` nonzero entries in its lower triangle) and `L = UU'` of rank `r`,
by gradient descent on `tr(Σx⋅Θ⋅Σy⋅Θ)/2 - tr(Θ⋅(Σy - Σx))`, `Θ = Δ + UU'`,
with step sizes `ηΔ` and `ηU` and hard thresholding of `Δ`. The in-place
version starts from `Δ` and `U` and uses `gD` (`p × p`) and `gU` (`p × r`) as
gradient storage. `callback(Δ, L)`, if given, is called after every iteration.

Returns `(Δ, L, U, fvals, iter)`: the estimates, the objective values checked
every `options.checkEvery` iterations, and the number of iterations.
"""
function differenceLatentPrecisionIHT!(
  Δ::SymmetricSparseIterate{T}, L::StridedMatrix{T},                 # these are pre-allocated
  U::StridedMatrix{T}, gD::StridedMatrix{T}, gU::StridedMatrix{T},   # these are pre-allocated
  Σx::Symmetric{T}, Σy::Symmetric{T},
  ηΔ, ηU, s::Int, r::Int;
  options::IHTOptions=IHTOptions(), callback=nothing) where {T<:AbstractFloat}

  epsTol, maxIter, checkEvery = options.epsTol, options.maxIter, options.checkEvery

  p = size(Σx, 1)
  X, Y = Matrix(Σx), Matrix(Σy)
  A = zeros(T, p, p)      # Σx⋅Θ⋅Σy
  Θ = zeros(T, p, p)      # Δ + L as a dense matrix
  tmp = zeros(T, p, p)
  mul!(L, U, U')
  _latentSandwich!(A, tmp, Θ, Σx, Δ, L, Σy)

  fvals = T[]
  # gradient descent
  #
  fv = _objective(A, X, Y, Θ)
  push!(fvals, fv)
  iter = 0
  while iter < maxIter
    iter += 1
    # update Δ
    diffPrecision_grad_delta!(gD, A, X, Y)
    max_gD = maximum(abs, gD)
    _gradientStep!(Δ, ηΔ, gD)
    HardThreshold!(Δ, s)

    # update A
    _latentSandwich!(A, tmp, Θ, Σx, Δ, L, Σy)

    # update U and L
    differencePrecisionIHT_grad_u!(gU, gD, A, X, Y, U)
    @. U -= ηU * gU
    mul!(L, U, U')
    max_gU = maximum(abs, gU)

    # update A
    _latentSandwich!(A, tmp, Θ, Σx, Δ, L, Σy)

    if callback !== nothing
      callback(Δ, L)
    end

    done = max(max_gD, max_gU) < epsTol
    # check for convergence
    if mod(iter, checkEvery) == 0
      fv_new = _objective(A, X, Y, Θ)
      push!(fvals, fv_new)
      done = abs(fv_new - fv) <= epsTol
      fv = fv_new
    end
    done && break
  end

  (Δ, L, U, fvals, iter)
end



# initialize Δ by hard thresholding inv(Σx) - inv(Σy) and U from the top r
# singular vectors of the remainder
function differenceLatentPrecisionIHT_init(
  Σx::Symmetric{T}, Σy::Symmetric{T},
  s::Int, r::Int) where {T<:AbstractFloat}

  size(Σx) == size(Σy) || throw(DimensionMismatch())
  p = size(Σx, 1)

  gD = zeros(T, p, p)
  gU = zeros(T, p, r)
  Δ = SymmetricSparseIterate(T, p)

  # initialize Δ, L, U
  #
  tmp = Matrix(inv(Σx) - inv(Σy))
  HardThreshold!(Δ, tmp, s)
  for colInd=1:p, rowInd=colInd:p
    tmp[rowInd,colInd] = tmp[rowInd,colInd] - Δ[rowInd, colInd]
    if rowInd != colInd
      tmp[colInd,rowInd] = tmp[rowInd,colInd]
    end
  end
  F = svd(tmp)
  U = F.U[:,1:r] .* sqrt.(F.S[1:r])'
  L = U * U'

  (Δ, L, U, gD, gU)
end

function differenceLatentPrecisionIHT(
  Σx::Symmetric{T}, Σy::Symmetric{T},
  ηΔ, ηU, s::Int, r::Int;
  options::IHTOptions=IHTOptions(), callback=nothing) where {T<:AbstractFloat}

  Δ, L, U, gD, gU = differenceLatentPrecisionIHT_init(Σx, Σy, s, r)
  differenceLatentPrecisionIHT!(Δ, L, U, gD, gU, Σx, Σy, ηΔ, ηU, s, r; options=options, callback=callback)
end






##
