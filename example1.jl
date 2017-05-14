using Distributions
import CovSel
import HD

# simple model that differes in only one edge
srand(123)

p = 50
Sigmax = eye(p,p)
Sigmay = zeros(p,p)
rho = 0.7
for i=1:p
    for j=1:p
        Sigmay[i,j]=rho^abs(i-j)
    end
end
sqmy = sqrtm(Sigmay)
n = 1000
X = randn(n,p)
Y = randn(n,p) * sqmy;
hSx = cov(X)
hSy = cov(Y)

tmp = ones(p,p)

# λ = 0.2
λ = rand(Uniform(0.15, 0.2))
λvec = λ*ones(div(p*(p+1), 2))

reload("HD")
reload("CovSel")

@time o1 = CovSel.Alt.differencePrecisionNaive(hSx, hSy, λ, tmp)
@time o2 = CovSel.Alt.differencePrecisionActiveShooting(hSx, hSy, λ, tmp)

@time o3 = CovSel.differencePrecisionActiveShooting(hSx, hSy, λvec, HD.CDOptions(;kktTol=1e-3))
@time o4 = CovSel.differencePrecisionActiveShooting1(hSx, hSy, λvec)
@show maximum(abs.(CovSel.vec2tril(o2, p) - tril(o1)))
@show maximum(abs.(CovSel.vec2tril(o3, p) - tril(o1)))


# non_zero_set = find( abs(precM) .> 1e-4 )
# @time CovSel.covsel_refit!(X, Z, U, S, non_zero_set)
#
# # find connected components
# import Images
# nonZero = zeros(S)
# for i in eachindex(S)
#   if abs(S[i]) > λ
#     nonZero[ i ] = 1
#   end
# end
# labels = Images.label_components(nonZero)
#
# function collect_groups(labels)
#     groups = [Int[] for i = 1:maximum(labels)]
#     for (i,l) in enumerate(labels)
#         if l != 0
#             push!(groups[l], i)
#         end
#     end
#     groups
# end
#
# groups = collect_groups(labels)
# S[ groups[1] ]
#
# S[1:2, 1:2]
