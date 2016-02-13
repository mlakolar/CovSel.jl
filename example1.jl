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

λ = 0.5

X = zeros(Float64, (p,p))
Z = zeros(Float64, (p,p))
U = zeros(Float64, (p,p))
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
