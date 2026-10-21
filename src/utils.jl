# value of the s-th largest element of v (zero when s exceeds length(v));
# entries with absolute value at least this threshold are kept
function _threshold(v::Vector{T}, s::Int) where {T<:AbstractFloat}
  s > length(v) && return zero(T)
  partialsort!(v, s; rev=true)
end

"""
    HardThreshold!(out, X, s, diagonal=true)
    HardThreshold!(X, s, diagonal=true)

Keep the `s` largest entries of the symmetric matrix `X` in absolute value and
set the rest to zero. Only entries on or below the main diagonal are ranked.
With `diagonal=false` the diagonal is excluded from the ranking and always kept.
Ties at the threshold are all kept. `out` is a dense matrix, a
`SymmetricSparseIterate`, or a `SparseIterate` vector indexed by the lower
triangle (see `sub2indLowerTriangular`).
"""
function HardThreshold!(out::StridedMatrix{T}, X::StridedMatrix{T}, s::Int, diagonal::Bool=true) where {T<:AbstractFloat}
  size(out) == size(X) || throw(DimensionMismatch())
  if s <= 0
    for c in axes(X, 2), r in axes(X, 1)
      out[r, c] = (!diagonal && r == c) ? X[r, c] : zero(T)
    end
    return out
  end

  p = size(X, 1)
  val = _threshold([abs(X[r, c]) for c=1:p for r=(diagonal ? c : c + 1):p], s)
  for c=1:p, r=1:p
    if !diagonal && r == c
      out[r, c] = X[r, c]
    else
      out[r, c] = abs(X[r, c]) >= val ? X[r, c] : zero(T)
    end
  end
  out
end

HardThreshold!(X::StridedMatrix, s::Int, diagonal::Bool=true) = HardThreshold!(X, X, s, diagonal)

_setLower!(out::SymmetricSparseIterate, v, p, r, c) = (out[r, c] = v)
_setLower!(out::SparseIterate{T, 1}, v, p, r, c) where {T} = (out[sub2indLowerTriangular(p, r, c)] = v)

function HardThreshold!(out::Union{SparseIterate{T, 1},SymmetricSparseIterate{T}}, X::StridedMatrix{T}, s::Int, diagonal::Bool=true) where {T<:AbstractFloat}
  p = size(X, 1)
  length(out) == div(p * (p + 1), 2) || throw(DimensionMismatch())

  val = s <= 0 ? T(Inf) : _threshold([abs(X[r, c]) for c=1:p for r=(diagonal ? c : c + 1):p], s)
  for c=1:p, r=c:p
    v = (!diagonal && r == c) || abs(X[r, c]) >= val ? X[r, c] : zero(T)
    _setLower!(out, v, p, r, c)
  end
  dropzeros!(out)
end

"""
    HardThreshold!(X::Union{SparseIterate,SymmetricSparseIterate}, s)

Keep the `s` largest stored entries of `X` in absolute value (ties at the
threshold are all kept) and drop the rest.
"""
function HardThreshold!(X::Union{SparseIterate{T},SymmetricSparseIterate{T}}, s::Int) where {T<:AbstractFloat}
  val = s <= 0 ? T(Inf) : _threshold(abs.(X.nzval[1:nnz(X)]), s)
  for i=1:nnz(X)
    if abs(X.nzval[i]) < val
      X.nzval[i] = zero(T)
    end
  end
  dropzeros!(X)
end


##################################
#
# Gaussian likelihood computation
#
#################################

# tr(Σ Ω) - log det Ω
covSel_objective(Σ, S) = tr(Σ*S) - logdet(S)

covSel_objective(Σ, S, L) = tr(Σ*(S+L)) - logdet(S+L)
