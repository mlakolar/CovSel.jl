module CovSel

using ProximalBase
using DataStructures: BinaryMaxHeap
using CoordinateDescent, Distributions
using Printf, SparseArrays, LinearAlgebra

export
  ADMMOptions,
  covsel!, covselpath,
  #covsel_refit!, covselpath_refit

  # difference estimation
  #
  fusedGraphicalLasso, fusedGraphicalLasso!,   # Danaher et al.
  fusedNeighborhoodSelection,

  # Direct Difference Estimation and Inference
  differencePrecisionActiveShooting, differencePrecisionActiveShooting!, differencePrecisionRefit,
  CDDirectDifferenceLoss,

  differencePrecisionIHT,
  differenceLatentPrecisionIHT

include("utils.jl")

include("admmCovSel.jl")
include("tvCovSel.jl")
include("LatentCovSel_IHT.jl")

include("directDifference.jl")
include("fusedDifference.jl")

include("generate.jl")
include("evaluation.jl")


# alternative implementations
include("altImplementation.jl")

end
