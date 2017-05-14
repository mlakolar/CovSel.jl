function _normdiff{T<:AbstractFloat}(A::StridedMatrix{T}, B::StridedMatrix{T})
  v = zero(T)
  n = size(A, 1)
  for c=1:n, r=1:n
    v += abs2( A[r, c] - B[r, c] )
  end
  sqrt(v)
end

function _normdiff{T<:AbstractFloat}(A::StridedVector{T}, B::StridedVector{T})
  v = zero(T)
  @inbounds @simd for i in eachindex(A)
    v += abs2( A[i] - B[i] )
  end
  sqrt(v)
end


"""
Removes all but largest s elements in absolute value.
Matrix X is symmetric, so we only look for the elements on or below
the main diagonal to decide value of the s-th largest element.
"""
function HardThreshold!{T<:AbstractFloat}(out::StridedMatrix{T}, X::StridedMatrix{T}, s::Int64)
  assert(size(out) == size(X))
  if s <= 0
    fill!(out, 0.)
    return out
  else
    p = size(X, 1)

    # find value of s-th largest element in abs(X)
    h = binary_maxheap(T)
    for c=1:p
      for r=c:p
        push!(h, abs(X[r,c]))
      end
    end

    val = zero(T)
    for k=1:s
      val = pop!(h)
    end

    for it in eachindex(X)
      out[it] = abs(X[it]) >= val ? X[it] : 0.
    end
  end

  out
end

HardThreshold!(X, s) = HardThreshold!(X, X, s)

#####
#
# functions to operate with sparse lower triangular
#

function ind2subLowerTriangular{T<:Integer}(p::T, ind::T)
  rvLinear = div(p*(p+1), 2) - ind
  k = trunc(T, (sqrt(1+8*rvLinear)-1.)/2. )
  j = rvLinear - div(k*(k+1), 2)
  (p-j, p-k)
end

function vec2tril(x::SparseVector, p::Int64)
  nx = nnz(x)
  nzval = SparseArrays.nonzeros(x)
  nzind = SparseArrays.nonzeroinds(x)

  I = zeros(Int64, nx)
  J = zeros(Int64, nx)
  for i=1:nx
    I[i], J[i] = ind2subLowerTriangular(p, nzind[i])
  end

  sparse(I,J,nzval)
end

function tril2symmetric(Δ::SparseMatrixCSC)
  lDelta = tril(Δ, -1)
  dDelta = spdiagm(diag(Δ))
  (lDelta + lDelta') + dDelta
end
