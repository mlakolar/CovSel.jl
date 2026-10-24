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
                   multiplier=1, anchor=1, unshrink=false)

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
`unshrink=true` keeps raw moments on the selected support before normalization.
"""
function estimate_index(X::AbstractMatrix, score::AbstractMatrix;
                        probe=(size(X,2)-1,size(X,2)), blocks::Integer=9,
                        threshold=nothing, multiplier::Real=1, anchor::Integer=1,
                        unshrink::Bool=false)
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
    unshrink && (g = moment .* (abs.(moment) .> tau))
    norm(g) > 0 && g[anchor] != 0 || throw(DomainError(g,"unidentified or thresholded-away index"))
    g .*= sign(g[anchor])/norm(g)
    IndexEstimate(g,moment,tau,blocks,n-m*blocks)
end

"""
    select_index_tuning(X, score; multipliers=(0.25,0.5,0.75,1,1.25),
                        unshrink_options=(false,true), blocks=5, block_options=(blocks,), folds=3,
                        probe=(p-1,p), anchor=1, rng=default_rng())

Select the Stein threshold by held-out oriented moment alignment. A training
direction is unit-normalized; its validation objective has population value
proportional to its cosine with the true direction when the identifying Stein
constant is nonzero and the training anchor orientation is correct. Its sign is oriented using the training anchor moment.
No true direction or precision coverage enters selection. Optional unshrinking
retains raw moments on the threshold-selected support. Return the full-data
index estimate, selected settings, candidate scores, and fold assignment.
A block candidate of one uses ordinary means; more blocks use their coordinatewise
median. See `docs/single_index/INDEX_TUNING.md` for assumptions and limitations.
This is an empirical selection rule, not an index or coverage guarantee.
"""
function select_index_tuning(X::AbstractMatrix,score::AbstractMatrix;
        multipliers=(0.25,0.5,0.75,1.0,1.25),unshrink_options=(false,true),
        blocks::Integer=5,block_options=(blocks,),folds::Integer=3,probe=(size(X,2)-1,size(X,2)),
        anchor::Integer=1,rng::AbstractRNG=default_rng())
    n,p = size(X)
    size(score,1)==n && size(score,2)>0 || throw(DimensionMismatch("invalid score matrix"))
    2<=folds<=n÷10 || throw(ArgumentError("invalid index fold count"))
    1<=blocks<=n-cld(n,folds) || throw(ArgumentError("invalid index block count"))
    length(probe)==2 && all(j -> j isa Integer && 1<=j<=p,probe) || throw(ArgumentError("invalid probe"))
    1<=anchor<=size(score,2) || throw(ArgumentError("invalid anchor"))
    ms,us,bs = collect(multipliers),collect(unshrink_options),collect(block_options)
    !isempty(bs) && all(b -> b isa Integer && 1<=b<=n-cld(n,folds),bs) ||
        throw(ArgumentError("invalid block candidates"))
    !isempty(ms) && all(m -> isfinite(m) && m>=0,ms) || throw(ArgumentError("invalid index multipliers"))
    !isempty(us) && all(u -> u isa Bool,us) || throw(ArgumentError("invalid unshrink choices"))
    all(isfinite,X) && all(isfinite,score) || throw(ArgumentError("nonfinite index data"))
    perm = randperm(rng,n)
    foldid = zeros(Int,n)
    foldid[perm] = mod.(0:n-1,folds).+1
    scores = NamedTuple[]
    for b in bs,m in ms,u in us
        total,complexity = 0.0,0.0
        failure = ""
        try
            for f in 1:folds
                tr,va = findall(!=(f),foldid),findall(==(f),foldid)
                idx = estimate_index(X[tr,:],score[tr,:];probe,blocks=b,multiplier=m,anchor,unshrink=u)
                v = X[va,probe[1]].*X[va,probe[2]].*(score[va,:]*idx.direction)
                total += sign(idx.moment[anchor])*sum(v)
                complexity += count(!iszero,idx.direction)/folds
            end
        catch err
            err isa InterruptException && rethrow()
            total = -Inf
            failure = sprint(showerror,err)
        end
        push!(scores,(multiplier=Float64(m),unshrink=u,blocks=Int(b),alignment=total/n,
                      complexity=complexity,failure=failure))
    end
    best = sort(scores;by=x -> (-x.alignment,x.complexity,-x.multiplier,!x.unshrink,x.blocks))[1]
    isfinite(best.alignment) || throw(DomainError(scores,"all index tuning candidates failed"))
    estimate = estimate_index(X,score;probe,blocks=best.blocks,multiplier=best.multiplier,anchor,unshrink=best.unshrink)
    (;estimate,multiplier=best.multiplier,unshrink=best.unshrink,blocks=best.blocks,scores,foldid)
end
