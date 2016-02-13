reload("CovSel")
import CovSel
import ProximalOPT

# generate data
n = 1000
p = 100

ρ = 0.5
covMat = eye(p)
for a = 1:p
  for b = a+1:p
    t = ρ^(abs(a-b))
    covMat[a, b] = t
    covMat[b, a] = t
  end
end
precM = inv(covMat)

sqCov = sqrtm(covMat)
X = randn(n, p) * sqCov

S = X' * X / n

X = zeros(Float64, (p,p))
Z = zeros(Float64, (p,p))
U = zeros(Float64, (p,p))
@time CovSel.covsel!(X, Z, U, S, 0.2; penalize_diag=false)


non_zero_set = find( abs(precM) .> 1e-4 )
@time CovSel.covsel_refit!(X, Z, U, S, non_zero_set; penalize_diag=false)

