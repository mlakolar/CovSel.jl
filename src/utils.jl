function _normdiff{T<:FloatingPoint}(A::StridedMatrix{T}, B::StridedMatrix{T})
  v = zero(T)
  n = size(A, 1)
  for c=1:n, r=1:n
    v += abs2( A[r, c] - B[r, c] )
  end
  sqrt(v)
end
