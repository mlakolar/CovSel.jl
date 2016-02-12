reload("CovSel")
import CovSel

# generate data
n = 1000
p = 1000

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

solver = CovSel.ADMMSolver();
X = zeros(Float64, (p,p))
Z = zeros(Float64, (p,p))
U = zeros(Float64, (p,p))
CovSel.solve!(solver, X, Z, U, S, 0.2; penalize_diag=false)

