"""
    CovarianceSelection

Methods for learning high-dimensional Gaussian graphical models:

* CLIME and ACLIME, sparse precision matrices by constrained ℓ1 minimization
  ([`clime`](@ref), [`aclime`](@ref)), with cross validation
  ([`climecv`](@ref), [`aclimecv`](@ref)) and a choice of solvers
  ([`CLIMESolver`](@ref));
* the precision matrix by the scaled lasso of each column, which needs no
  tuning ([`slassoinv`](@ref));
* the graphical lasso by ADMM ([`covsel!`](@ref), [`covselpath`](@ref));
* the fused graphical lasso for two groups ([`fusedGraphicalLasso`](@ref)) and
  fused neighborhood selection ([`fusedNeighborhoodSelection`](@ref));
* K precision matrices with a shared sparsity pattern, the group graphical
  lasso ([`groupGraphicalLasso`](@ref)), and graphs that vary over time by
  kernel smoothing ([`tvCovSel`](@ref), [`kernelCovariances`](@ref));
* direct estimation of the difference of two precision matrices by coordinate
  descent ([`differencePrecisionActiveShooting`](@ref),
  [`differencePrecisionRefit`](@ref)) or iterative hard thresholding, with an
  optional low rank component ([`differencePrecisionIHT`](@ref),
  [`differenceLatentPrecisionIHT`](@ref)).

The losses and penalties are those of ProximalBase, and the graphical lasso and
the direct difference estimator are solved by the ADMM and the coordinate
descent of ProximalSolvers. Before its first release the package was called
CovSel.
"""
module CovarianceSelection

using ProximalBase
using ProximalSolvers: ProximalSolvers, ADMMSolver, CDSolver, ProximalOptions, refit!, solve!
using SparseArrays, LinearAlgebra
using Random: AbstractRNG, default_rng, randperm
using SpecialFunctions: erfcinv

export
  ADMMOptions, ADMMInfo, IHTOptions,
  covsel!, covselpath,
  covselLatentIHT!,

  # difference estimation
  #
  fusedGraphicalLasso, fusedGraphicalLasso!,   # Danaher et al.
  fusedNeighborhoodSelection,

  # K precision matrices: shared support, time-varying graphs
  groupGraphicalLasso, groupGraphicalLasso!, kernelCovariances, tvCovSel,

  # Direct Difference Estimation and Inference
  differencePrecisionActiveShooting, differencePrecisionActiveShooting!, differencePrecisionRefit,
  directDifferenceLoss, diffLoss,

  differencePrecisionIHT,
  differenceLatentPrecisionIHT, differenceLatentPrecisionIHT!,

  # CLIME and ACLIME
  clime, aclime, CLIMEPath, climecv, aclimecv, CLIMECV, bestestimate,
  CLIMESolver, DualSimplex, ParametricSimplex, LPSolver, PDHG,

  # scaled lasso of Sun and Zhang
  slassoinv

include("utils.jl")

include("admmCovarianceSelection.jl")
include("LatentCovarianceSelection_IHT.jl")

include("directDifference.jl")
include("automaticIHT.jl")
include("fusedDifference.jl")
include("groupGraphicalLasso.jl")

include("clime_simplex.jl")
include("clime.jl")
include("clime_pdhg.jl")
include("clime_cv.jl")
include("slassoinv.jl")

include("generate.jl")
include("evaluation.jl")

include("SingleIndexGraph/SingleIndexGraph.jl")

include("precompile.jl")

end
