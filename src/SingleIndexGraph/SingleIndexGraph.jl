"""
    CovarianceSelection.SingleIndexGraph

Pointwise precision-entry inference and Gaussian multiplier bands along a scalar
index. See `docs/single_index/README.md` for the estimator, assumptions, tuning,
diagonal extension, and the limits of grid-based simultaneous inference.
"""
module SingleIndexGraph

using LinearAlgebra
using Random: AbstractRNG, default_rng, randn, randperm
using SpecialFunctions: erfcinv
using ..CovarianceSelection: slassoinv, aclime, _SLWorkspace, _sl_column!, _ColumnPath

export IndexEstimate, PrecisionCurve, estimate_index, local_design,
       local_covariance, precision_pilot, fit_precision, confint, confband,
       precision_errors, select_tuning, select_index_tuning, biweight

include("index.jl")
include("smoothing.jl")
include("inference.jl")
include("tuning.jl")

end
