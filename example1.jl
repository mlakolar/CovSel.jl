reload("CovSel")
import CovSel






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
