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
using SparseArrays, LinearAlgebra
using Random: AbstractRNG, default_rng

export
  ADMMOptions, ADMMInfo, IHTOptions,
  covsel!, covselpath,

  # difference estimation
  #
  fusedGraphicalLasso, fusedGraphicalLasso!,   # Danaher et al.
  fusedNeighborhoodSelection,

  # Direct Difference Estimation and Inference
  differencePrecisionActiveShooting, differencePrecisionActiveShooting!, differencePrecisionRefit,
  CDDirectDifferenceLoss, diffLoss,

  differencePrecisionIHT,
  differenceLatentPrecisionIHT, differenceLatentPrecisionIHT!,

  # CLIME and ACLIME
  clime, aclime, CLIMEPath,
  CLIMESolver, DualSimplex, ParametricSimplex, LPSolver

include("utils.jl")

include("admmCovSel.jl")
include("LatentCovSel_IHT.jl")

include("directDifference.jl")
include("fusedDifference.jl")

include("clime_simplex.jl")
include("clime.jl")

include("generate.jl")
include("evaluation.jl")

end
