"""
    CovSel

Methods for learning high-dimensional Gaussian graphical models:

* the graphical lasso by ADMM ([`covsel!`](@ref), [`covselpath`](@ref));
* the fused graphical lasso for two groups ([`fusedGraphicalLasso`](@ref)) and
  fused neighborhood selection ([`fusedNeighborhoodSelection`](@ref));
* direct estimation of the difference of two precision matrices by coordinate
  descent ([`differencePrecisionActiveShooting`](@ref),
  [`differencePrecisionRefit`](@ref)) or iterative hard thresholding, with an
  optional low rank component ([`differencePrecisionIHT`](@ref),
  [`differenceLatentPrecisionIHT`](@ref)).
"""
module CovSel

using ProximalBase
using CoordinateDescent
using Printf, SparseArrays, LinearAlgebra

export
  ADMMOptions, IHTOptions,
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
  differenceLatentPrecisionIHT, differenceLatentPrecisionIHT!

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
