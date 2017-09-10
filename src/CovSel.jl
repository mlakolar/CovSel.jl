module CovSel

using ProximalBase
using DataStructures: binary_maxheap
using CoordinateDescent, Distributions

export
  ADMMOptions,
  covsel!, covselpath,
  #covsel_refit!, covselpath_refit

  # difference estimation
  #
  fusedGraphicalLasso, fusedGraphicalLasso!,   # Danaher et al.
  fusedNeighborhoodSelection,

  # Direct Difference Estimation and Iference
  differencePrecisionActiveShooting, differencePrecisionActiveShooting!, differencePrecisionRefit,
  CDInverseKroneckerLoss, CDDirectDifferenceLoss,
  
  differencePrecisionIHT,
  differenceLatentPrecisionIHT

include("utils.jl")

include("admmCovSel.jl")
include("diffEstim.jl")
include("directDifference.jl")
include("generate.jl")
include("evaluation.jl")


# alternative implementations
include("altImplementation.jl")

end
