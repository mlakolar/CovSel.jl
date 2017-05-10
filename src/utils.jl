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
