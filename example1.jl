reload("CovSel")
import CovSel

# generate data
n = 1000
p = 100

ρ = 0.5
covMat = eye(p)
for a = 1:p
  for b = a+1:p
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
CovSel.solve!(solver, X, Z, U, S, 0.2; penalize_diag=true)

##

using Convex
using SCS

# passing in verbose=0 to hide output from SCS
solver = SCSSolver(verbose=0)
set_default_solver(solver);

Ω = Variable(p, p)
problem = minimize(trace(Ω*S) - logdet(Ω) + 0.2 * vecnorm(Ω, 1), Ω ⪰ 0)
solve!(problem)

Ω.value
