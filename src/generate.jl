"""
    generateData(Ω, n)

Draw `n` observations from `N(0, Ω⁻¹)`, returned as the rows of an `n × p` matrix.
"""
function generateData(Ω, n)
  d, U = eigen(Symmetric(Ω))
  X = randn(n, size(Ω, 1)) * (U * Diagonal(1 ./ sqrt.(d)) * U')
end


############################

"""
    generateSparseSPD(p, α=0.95, smallest_coef=0.1, largest_coef=0.9)

Random sparse positive definite precision matrix `CC'`, where `C` is lower
triangular with unit diagonal and each off-diagonal entry is zero with
probability `α` and otherwise uniform on `[smallest_coef, largest_coef]`.
"""
function generateSparseSPD(p, α=0.95, smallest_coef=.1, largest_coef=.9)

  aux = rand(p, p)
  for c=1:p, r=c+1:p
    aux[r,c] = aux[r,c] < α ? 0. : smallest_coef + (largest_coef - smallest_coef) * rand()
  end
  tril!(aux, -1)

  chol = aux - I
  prec = chol * chol'

  return prec
end

# returns Ω and a random rank r matrix L with operator norm relStrength⋅λmax(Ω)
function genLatentModel(Ω::StridedMatrix, r, relStrength)
  p = size(Ω, 1)
  evmax = eigmax(Symmetric(Ω))

  U = randn(p, r)
  L = Symmetric( U*U' )
  L = L / opnorm(L) * relStrength * evmax

  (Ω, L)
end
