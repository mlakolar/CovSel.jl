

"""
Removes all but largest s elements in absolute value.
Matrix X is symmetric, so we only look for the elements on or below
the main diagonal to decide value of the s-th largest element.
"""
function HardThreshold!{T<:AbstractFloat}(out::StridedMatrix{T}, X::StridedMatrix{T}, s::Int64, diagonal::Bool=true)
  assert(size(out) == size(X))
  if s <= 0
    fill!(out, 0.)
    return out
  else
    p = size(X, 1)

    # find value of s-th largest element in abs(X)
    h = binary_maxheap(T)
    for c=1:p, r=(diagonal ? c : c + 1):p
        push!(h, abs(X[r,c]))
    end

    val = zero(T)
    for k=1:s
      val = pop!(h)
    end

    for c=1:p, r=1:p
      if (!diagonal && r == c)
        out[r, c] = X[r, c]
      else
        out[r, c] = abs(X[r, c]) >= val ? X[r, c] : zero(T)
      end
    end
  end

  out
end

HardThreshold!(X::StridedMatrix, s::Int64, diagonal::Bool=true) = HardThreshold!(X, X, s, diagonal)

function HardThreshold!{T<:AbstractFloat}(out::SparseIterate{T}, X::StridedMatrix{T}, s::Int64, diagonal::Bool=true)

  p = size(X, 1)

  # find value of s-th largest element in abs(X)
  h = binary_maxheap(T)
  for c=1:p, r=(diagonal ? c : c + 1):p
      push!(h, abs(X[r,c]))
  end

  val = zero(T)
  for k=1:s
    val = pop!(h)
  end

  for c=1:p, r=c:p
    if (!diagonal && r == c)
      out[sub2indLowerTriangular(p, r, c)] = X[r, c]
    else
      out[sub2indLowerTriangular(p, r, c)] = abs(X[r, c]) >= val ? X[r, c] : zero(T)
    end
  end

  out
end

function HardThreshold!{T<:AbstractFloat}(X::SparseIterate{T}, s::Int64)

  # find value of s-th largest element in abs(X)
  h = binary_maxheap(T)
  for i=1:nnz(X)
      push!(h, abs(X.nzval[i]))
  end

  val = zero(T)
  for k=1:s
    val = pop!(h)
  end

  for i=1:nnz(X)
    if abs(X.nzval[i]) <= val
      X.nzval[i] = zero(T)
    end
  end
  dropzeros!(X)
  X
end


##################################
#
# Gaussian likelihood computation
#
#################################

covSel_objective(Σ, S) = trace(Σ*S) - log(det(S))

covSel_objective(Σ, S, L) = trace(Σ*(S+L)) - log(det(S+L))
