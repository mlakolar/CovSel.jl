
# generate guassian data from a precision matrix
function generateData(Ω, n)
  p = size(Ω, 1)
  d, U = eig(Ω)
  for i=1:p
    d[i] = 1. / sqrt(d[i])
  end
  X = randn(n, p) * (U * diagm(d) * U')
end


############################

function generateSparseSPD(p, α=0.95, smallest_coef=.1, largest_coef=.9)

  aux = rand(p, p)
  d = Uniform(smallest_coef, largest_coef)
  for c=1:p, r=c+1:p
    aux[r,c] = aux[r,c] < α ? 0. : rand(d)
  end
  tril!(aux, -1)

  # permutation = randperm(p)
  # aux = aux[permutation, :]
  # aux = aux[:, permutation]
  # @show aux
  chol = aux - eye(p)
  prec = chol * chol'

  return prec
end

# takes as an input a precision matrix,
# returns a new precision matrix
# off diagonal elements have some probability of having flipped sign
function generateRandomDifference(Ω::StridedMatrix, probability_flip)

  p = size(Ω, 1)
  Ωy = copy(Ω)

  for c=1:p, r=c+1:p
    if rand() < probability_flip
      v = -Ω[r,c]
      Ωy[r,c] = Ωy[c,r] = v
    end
  end

  return Ωy
end


# ρ -- strength of an edge
# generate
function genLatentModel(Ω::StridedMatrix, r, relStrength)
  p = size(Ω, 1)
  evmax = eigmax(Ω)

  U = randn(p, r)
  L = Symmetric( U*U' )
  L = L / norm(L, 2.) * relStrength * evmax

  (Ω, L)
end
