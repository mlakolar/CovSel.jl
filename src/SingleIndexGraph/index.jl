"""A sparse, oriented Stein index estimate, with its moment and threshold."""
struct IndexEstimate
    direction::Vector{Float64}
    moment::Vector{Float64}
    threshold::Vector{Float64}
    blocks::Int
    discarded::Int
end

_median(x) = (v = sort(collect(x)); n = length(v); (v[(n+1)÷2] + v[(n+2)÷2])/2)

"""
    estimate_index(X, score; probe=(p-1,p), blocks=9, threshold=nothing,
                   multiplier=1, anchor=1)

Coordinatewise median-of-means of `X[:,a] .* X[:,b] .* score`, followed by soft
thresholding and unit normalization with positive anchor. Rows are observations.
`score` is the *known negative log-density gradient* of the covariates, supplied
by the caller; for standard Gaussian covariates it is the covariate matrix.
The probe must be fixed before observing the data and have nonzero derivative
signal. A zero vector or zero anchor throws, rather than inventing a direction.
With `threshold=nothing`, use `multiplier * sqrt(2log(2d)/n)` times the empirical
coordinate standard deviation. This practical scale estimate is not a robust
finite-sample calibration theorem. Supply a scalar/vector threshold for exact
control of this choice. Blocks are deterministic equal consecutive blocks.
"""
function estimate_index(X::AbstractMatrix, score::AbstractMatrix;
                        probe=(size(X,2)-1,size(X,2)), blocks::Integer=9,
                        threshold=nothing, multiplier::Real=1, anchor::Integer=1)
    n, p = size(X)
    size(score,1) == n || throw(DimensionMismatch("score rows must match X"))
    d = size(score,2)
    1 <= anchor <= d || throw(ArgumentError("invalid orientation anchor"))
    all(j -> 1 <= j <= p, probe) || throw(ArgumentError("invalid probe"))
    length(probe) == 2 || throw(ArgumentError("probe must contain two coordinates"))
    1 <= blocks <= n || throw(ArgumentError("blocks must lie in 1:n"))
    isfinite(multiplier) && multiplier >= 0 || throw(ArgumentError("invalid multiplier"))
    all(isfinite,X) && all(isfinite,score) || throw(ArgumentError("nonfinite data"))
    Y = Float64.(score) .* (X[:,probe[1]] .* X[:,probe[2]])
    m = n ÷ blocks
    moment = [_median([sum(@view Y[(b-1)*m+1:b*m,j])/m for b in 1:blocks]) for j in 1:d]
    tau = if threshold === nothing
        [multiplier * sqrt(2log(2d)/n) *
         sqrt(sum(abs2, Y[:,j] .- sum(Y[:,j])/n)/max(n-1,1)) for j in 1:d]
    elseif threshold isa Real
        fill(Float64(threshold),d)
    else
        Float64.(threshold)
    end
    length(tau) == d && all(x -> isfinite(x) && x >= 0,tau) ||
        throw(ArgumentError("invalid threshold"))
    g = sign.(moment) .* max.(abs.(moment) .- tau,0)
    norm(g) > 0 && g[anchor] != 0 || throw(DomainError(g,"unidentified or thresholded-away index"))
    g .*= sign(g[anchor])/norm(g)
    IndexEstimate(g,moment,tau,blocks,n-m*blocks)
end
