module CovSel

using ProximalBase: shrink, ProximableFunction, ProxGaussLikelihood, prox!, proxL1Fused
using DataStructures: binary_maxheap

export
  ADMMOptions,
  covsel!, covselpath,
  #covsel_refit!, covselpath_refit

  # difference estimation
  #
  fusedGraphicalLasso,
  fusedNeighborhoodSelection,
  differencePrecisionActiveShooting,
  differencePrecisionIHT,
  differenceLatentPrecisionIHT

include("admmCovSel.jl")
include("diffEstim.jl")
include("utils.jl")

end
