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
    covMat[a, b] = t
    covMat[b, a] = t
  end
end
precM = inv(covMat)

sqCov = sqrtm(covMat)
X = randn(n, p) * sqCov

S = X' * X / n

λ = 0.5

X = zeros(Float64, (p,p))
Z = zeros(Float64, (p,p))
U = zeros(Float64, (p,p))
<<<<<<< HEAD
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
=======
@time CovSel.covsel!(X, Z, U, S, λ; penalize_diag=false)


non_zero_set = find( abs(precM) .> 1e-4 )
@time CovSel.covsel_refit!(X, Z, U, S, non_zero_set)

# find connected components
import Images
nonZero = zeros(S)
for i in eachindex(S)
  if abs(S[i]) > λ
    nonZero[ i ] = 1
  end
end
labels = Images.label_components(nonZero)

function collect_groups(labels)
    groups = [Int[] for i = 1:maximum(labels)]
    for (i,l) in enumerate(labels)
        if l != 0
            push!(groups[l], i)
        end
    end
    groups
end

groups = collect_groups(labels)
S[ groups[1] ]

S[1:2, 1:2]
>>>>>>> f0201db59be1986070a460fb91758dec306d4d52
