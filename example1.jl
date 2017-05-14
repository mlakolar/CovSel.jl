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
λ = rand(Uniform(0.05, 0.3))

reload("HD")
reload("CovSel")

solShoot = CovSel.differencePrecisionNaive(hSx, hSy, λ, tmp)
@time solShoot1 = CovSel.differencePrecisionActiveShooting(hSx, hSy, λ, tmp)
@time solShoot2 = CovSel.differencePrecision1(hSx, hSy, λ*tmp)

maximum(abs.(solShoot1 - solShoot))

vs2 = zeros(p, p)
ind = 0
for ci=1:p, ri=ci:p
  ind += 1
  vs2[ri,ci] = solShoot2[ind]
end
maximum(abs.(vs2 - tril(solShoot1)))

##################################
##################################
Δ = spzeros(p,p)
A = zeros(p,p)
ind1 = CovSel.findViolator!(Δ, A, hSx, hSy, λ, tmp)
ind2sub((p,p), ind1)
CovSel.updateDelta!(Δ, A, hSx, hSy, λ, tmp)

##################################
##################################
f = CovSel.CDDirectDifferenceLoss(hSx, hSy)
p = f.p
λnew = ones(zeros(HD.numCoordinates(f))) * λ
x = spzeros(HD.numCoordinates(f))
ind2 = HD.add_violating_index!(x, f, λnew)
CovSel.ind2subLowerTriangular(ind2, p)
HD.minimize_active_set!(x, f, λnew)

@show CovSel.ind2subLowerTriangular(p, 14)
@show CovSel.ind2subLowerTriangular(p,  7)
@show CovSel.ind2subLowerTriangular(p,  6)
@show CovSel.ind2subLowerTriangular(p,  1)
@show CovSel.ind2subLowerTriangular(p,  13)
@show CovSel.ind2subLowerTriangular(p,  10)
@show CovSel.ind2subLowerTriangular(p,  15)




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
