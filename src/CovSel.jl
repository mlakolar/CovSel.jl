"""
    CovSel

Methods for learning high-dimensional Gaussian graphical models:

* CLIME and ACLIME, sparse precision matrices by constrained ℓ1 minimization
  ([`clime`](@ref), [`aclime`](@ref)), with cross validation
  ([`climecv`](@ref), [`aclimecv`](@ref)) and a choice of solvers
  ([`CLIMESolver`](@ref));
* the graphical lasso by ADMM ([`covsel!`](@ref), [`covselpath`](@ref));
* the fused graphical lasso for two groups ([`fusedGraphicalLasso`](@ref)) and
  fused neighborhood selection ([`fusedNeighborhoodSelection`](@ref));
* direct estimation of the difference of two precision matrices by coordinate
  descent ([`differencePrecisionActiveShooting`](@ref),
  [`differencePrecisionRefit`](@ref)) or iterative hard thresholding, with an
  optional low rank component ([`differencePrecisionIHT`](@ref),
  [`differenceLatentPrecisionIHT`](@ref)).

The losses and penalties are those of ProximalBase, and the graphical lasso and
the direct difference estimator are solved by the ADMM and the coordinate
descent of ProximalSolvers.
"""
module CovSel

using ProximalBase
using ProximalSolvers: ProximalSolvers, ADMMSolver, CDSolver, ProximalOptions, refit!, solve!
using SparseArrays, LinearAlgebra
using Random: AbstractRNG, default_rng, randperm

export
  ADMMOptions, ADMMInfo, IHTOptions,
  covsel!, covselpath,

  # difference estimation
  #
  fusedGraphicalLasso, fusedGraphicalLasso!,   # Danaher et al.
  fusedNeighborhoodSelection,

  # Direct Difference Estimation and Inference
  differencePrecisionActiveShooting, differencePrecisionActiveShooting!, differencePrecisionRefit,
  directDifferenceLoss, diffLoss,

  differencePrecisionIHT,
  differenceLatentPrecisionIHT, differenceLatentPrecisionIHT!,

  # CLIME and ACLIME
  clime, aclime, CLIMEPath, climecv, aclimecv, CLIMECV, bestestimate,
  CLIMESolver, DualSimplex, ParametricSimplex, LPSolver, PDHG

include("utils.jl")

include("admmCovSel.jl")
include("LatentCovSel_IHT.jl")

include("directDifference.jl")
include("fusedDifference.jl")

include("clime_simplex.jl")
include("clime.jl")
include("clime_pdhg.jl")
include("clime_cv.jl")

include("generate.jl")
include("evaluation.jl")

end
