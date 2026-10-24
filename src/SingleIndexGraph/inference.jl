"""
Fitted entries on an index grid. Arrays `estimate`, `se`, `failure` have shape
(points, entries). Influence columns use `g + (e-1)*length(points)`. Failed
columns are NaN. Failures are retained and cause `confband` to throw.
"""
struct PrecisionCurve
    points::Vector{Float64}
    entries::Vector{Tuple{Int,Int}}
    estimate::Matrix{Float64}
    se::Matrix{Float64}
    influence::Matrix{Float64}
    failure::Matrix{String}
    diagnostics::Vector{NamedTuple}
    settings::NamedTuple
end

"""
    fit_precision(X, index; points, entries, bandwidth, pilot_bandwidth=bandwidth,
                  pilot=:sqrtlasso, correction=:block, multiplier=1, delta=2,
                  degree=2, refit=false, kernel=biweight, oracle_coefficients=nothing,
                  residual_adjustment=:none, studentization=:hc0, inverse_bias=:none)

Estimate selected precision entries, including diagonals. The default fits
common-complement weighted square-root lasso regressions, smooths their
*target-fixed* residual products by a local quadratic, then inverts a 1x1
(diagonal) or 2x2 block. HC0 scores use observation-specific polynomial fitted
values and the full realized equivalent weights. `pilot=:scaledlasso` or
`:aclime` obtains the regressions from a localized full precision pilot.
`correction=:onestep` instead uses `P + P' - P' S_final P` and its fitted score;
this experimental alternative requires a full precision pilot. Oracle nuisance
coefficients may be supplied by `(point,J,A)->B` for simulation benchmarks.

`refit=true` refits selected regression neighborhoods before the correction.
`residual_adjustment=:df` refits the union support and normalizes each residual
by its WLS residual-maker row norm. `studentization=:hc3` corrects final smoothing
leverage. `inverse_bias=:secondorder` subtracts plug-in inverse curvature bias.
See `docs/single_index/DIAGONAL_TUNING.md` for formulas and limitations.
Every requested point/entry is attempted; numerical failures are recorded, never
silently dropped or repaired. Input errors throw before fitting.
"""
function fit_precision(X::AbstractMatrix,t::AbstractVector;points,entries,
                       bandwidth::Real,pilot_bandwidth::Real=bandwidth,
                       pilot::Symbol=:sqrtlasso,correction::Symbol=:block,
                       multiplier::Real=1,delta::Real=2,degree::Integer=2,refit::Bool=false,
                       kernel=biweight,oracle_coefficients=nothing,
                       residual_adjustment::Symbol=:none,studentization::Symbol=:hc0,
                       inverse_bias::Symbol=:none)
    _data(X,t)
    all(x -> isfinite(x) && x > 0,(bandwidth,pilot_bandwidth,multiplier,delta)) ||
        throw(ArgumentError("bandwidths and tuning constants must be positive and finite"))
    pilot in (:sqrtlasso,:scaledlasso,:aclime) || throw(ArgumentError("unknown pilot"))
    correction in (:block,:onestep) || throw(ArgumentError("unknown correction"))
    correction === :onestep && pilot === :sqrtlasso && throw(ArgumentError("one-step needs a full pilot"))
    correction === :onestep && oracle_coefficients !== nothing &&
        throw(ArgumentError("oracle coefficients apply only to block correction"))
    degree in 0:3 || throw(ArgumentError("invalid degree"))
    residual_adjustment in (:none,:df) || throw(ArgumentError("unknown residual adjustment"))
    studentization in (:hc0,:hc3) || throw(ArgumentError("unknown Studentization"))
    inverse_bias in (:none,:secondorder) || throw(ArgumentError("unknown inverse bias correction"))
    correction === :onestep && (residual_adjustment !== :none || inverse_bias !== :none) &&
        throw(ArgumentError("residual and inverse bias adjustments require block correction"))
    residual_adjustment === :df && oracle_coefficients !== nothing &&
        throw(ArgumentError("oracle coefficients do not have fitted residual degrees of freedom"))
    pts = Float64.(collect(points))
    !isempty(pts) && all(isfinite,pts) || throw(ArgumentError("invalid points"))
    p = size(X,2)
    es = Tuple{Int,Int}[]
    for pair in entries
        length(pair) == 2 || throw(ArgumentError("each entry is a pair"))
        a,b = pair
        a isa Integer && b isa Integer && 1 <= a <= p && 1 <= b <= p ||
            throw(ArgumentError("invalid precision entry"))
        push!(es,minmax(Int(a),Int(b)))
    end
    !isempty(es) && length(unique(es)) == length(es) || throw(ArgumentError("entries must be nonempty and unique"))
    n,G,E = size(X,1),length(pts),length(es)
    est,se = fill(NaN,G,E),fill(NaN,G,E)
    influences = fill(NaN,n,G*E)
    failures = fill("",G,E)
    diagnostics = NamedTuple[]
    for (g,point) in enumerate(pts)
        design = nothing
        P = nothing
        try
            design = local_design(t,point,bandwidth;degree,kernel)
            if pilot !== :sqrtlasso && oracle_coefficients === nothing
                P = precision_pilot(X,t,point,pilot_bandwidth;method=pilot,multiplier,delta,refit,kernel).precision
            end
        catch err
            err isa InterruptException && rethrow()
            failures[g,:] .= sprint(showerror,err)
            push!(diagnostics,(point=point,condition=NaN,local_n=0,equivalent_n=NaN))
            continue
        end
        push!(diagnostics,(point=point,condition=design.condition,local_n=design.local_n,
                           equivalent_n=design.equivalent_n))
        for (e,(a,b)) in enumerate(es)
            try
                if correction === :onestep
                    v = X*P[:,a]
                    z = v .* (X*P[:,b])
                    value = P[a,b]+P[b,a]-dot(design.weights,z)
                else
                    J = a == b ? [a] : [a,b]
                    A = setdiff(1:p,J)
                    B = if oracle_coefficients !== nothing
                        Matrix{Float64}(oracle_coefficients(point,J,A))
                    elseif pilot === :sqrtlasso
                        _regression(X,t,point,pilot_bandwidth,J,A,multiplier,kernel;refit).coefficients
                    else
                        isposdef(Symmetric(P[J,J])) || throw(DomainError(point,"nonpositive pilot target block"))
                        -P[A,J]/P[J,J]
                    end
                    size(B) == (length(A),length(J)) || throw(DimensionMismatch("invalid nuisance coefficients"))
                    factor = ones(n)
                    if residual_adjustment === :df
                        adjusted = _residual_df(X,t,point,pilot_bandwidth,J,A,B,kernel)
                        B,factor = adjusted.coefficients,adjusted.factor
                    end
                    R = X[:,J] - X[:,A]*B
                    R ./= sqrt.(factor)
                    Qinv = R'*(design.weights .* R)
                    isposdef(Symmetric(Qinv)) || throw(DomainError(point,"nonpositive final residual covariance"))
                    Q = inv(Symmetric(Qinv))
                    value = Q[1,end]
                    V = R*Q
                    z = V[:,1].*V[:,end]
                end
                fitted = design.basis*(design.map*z)
                influence = -design.weights .* (z .- fitted)
                if studentization === :hc3
                    all(<(1),design.leverage) || throw(DomainError(point,"unit smoothing leverage"))
                    influence ./= 1 .- design.leverage
                end
                variance = sum(abs2,influence)
                if inverse_bias === :secondorder
                    # Second derivative of matrix inversion, estimated with the
                    # same centered moment scores as the sandwich variance.
                    k = size(R,2)
                    moments = [design.basis*(design.map*(R[:,j].*R[:,l])) for j in 1:k,l in 1:k]
                    bias = zeros(k,k)
                    for i in 1:n
                        design.weights[i] == 0 && continue
                        moment_error = [R[i,j]*R[i,l]-moments[j,l][i] for j in 1:k,l in 1:k]
                        scale = studentization === :hc3 ? 1-design.leverage[i] : 1.0
                        bias += (design.weights[i]/scale)^2 .* (Q*moment_error*Q*moment_error*Q)
                    end
                    value -= bias[1,end]
                end
                isfinite(value) && isfinite(variance) && variance > 0 ||
                    throw(DomainError(variance,"nonfinite center or nonpositive variance"))
                est[g,e],se[g,e] = value,sqrt(variance)
                influences[:,g+(e-1)*G] = influence
            catch err
                err isa InterruptException && rethrow()
                failures[g,e] = sprint(showerror,err)
            end
        end
    end
    PrecisionCurve(pts,es,est,se,influences,failures,diagnostics,
       (;bandwidth=Float64(bandwidth),pilot_bandwidth=Float64(pilot_bandwidth),
         pilot,correction,multiplier=Float64(multiplier),delta=Float64(delta),degree,refit,
         residual_adjustment,studentization,inverse_bias))
end

function _endpoints(f,critical,diagonal_transform)
    diagonal_transform in (:identity,:log) || throw(ArgumentError("unknown diagonal transform"))
    lower = f.estimate .- critical.*f.se
    upper = f.estimate .+ critical.*f.se
    if diagonal_transform === :log
        for (e,(a,b)) in enumerate(f.entries)
            a == b || continue
            for g in eachindex(f.points)
                isempty(f.failure[g,e]) || continue
                value = f.estimate[g,e]
                value>0 || throw(DomainError(value,"log diagonal interval needs a positive estimate"))
                c = critical isa Number ? critical : critical[1,e]
                radius = c*f.se[g,e]/value
                lower[g,e],upper[g,e] = value*exp(-radius),value*exp(radius)
            end
        end
    end
    (lower=lower,upper=upper)
end

"""Pointwise intervals; `diagonal_transform=:log` uses log-Wald diagonals.
Off-diagonal endpoints are unchanged. Failed fits retain NaN endpoints.
"""
function confint(f::PrecisionCurve;level::Real=0.95,diagonal_transform::Symbol=:identity)
    0 < level < 1 || throw(ArgumentError("level must lie in (0,1)"))
    critical = sqrt(2)*erfcinv(1-level)
    endpoints = _endpoints(f,critical,diagonal_transform)
    (;endpoints...,critical,diagonal_transform)
end

"""
    confband(f; level=0.95, draws=1999, joint=true, rng=default_rng(),
             diagonal_transform=:identity)

Gaussian observation-multiplier sup-t band over the requested grid. One n-vector
of multipliers is shared across every point and entry in each draw. `joint=true`
controls the family of all requested entries and points; `joint=false` produces
one curvewise critical value per entry, not a joint family guarantee. Fails if
any requested fit failed. This is a finite-grid procedure, not a numerical
certificate of continuum coverage. Smoothing and index bias remain.
`diagonal_transform=:log` gives multiplicative diagonal endpoints; standardized
multiplier scores and critical values are unchanged.
"""
function confband(f::PrecisionCurve;level::Real=0.95,draws::Integer=1999,
                  joint::Bool=true,rng::AbstractRNG=default_rng(),
                  diagonal_transform::Symbol=:identity)
    0 < level < 1 || throw(ArgumentError("level must lie in (0,1)"))
    draws >= 100 || throw(ArgumentError("draws must be at least 100"))
    all(isempty,f.failure) && all(x -> isfinite(x) && x > 0,f.se) ||
        throw(DomainError(f.failure,"band requires every requested fit to succeed"))
    G,E = size(f.estimate)
    maxima = zeros(draws,joint ? 1 : E)
    scaled = f.influence ./ vec(f.se)'
    for b in 1:draws
        z = reshape(scaled'*randn(rng,size(scaled,1)),G,E)
        if joint
            maxima[b,1] = maximum(abs,z)
        else
            for e in 1:E
                maxima[b,e] = maximum(abs,@view z[:,e])
            end
        end
    end
    # Conservative empirical inverse CDF, with a pointwise-normal floor.
    k = min(draws,ceil(Int,(draws+1)*level))
    c = [max(sort(maxima[:,e])[k],sqrt(2)*erfcinv(1-level)) for e in axes(maxima,2)]
    critical = joint ? fill(c[1],1,E) : reshape(c,1,E)
    endpoints = _endpoints(f,critical,diagonal_transform)
    (;endpoints...,critical=vec(critical),joint,draws,diagonal_transform)
end

"""
    precision_errors(estimate, truth)

Entrywise max and sum, Frobenius, spectral, induced 1/infinity errors and
relative Frobenius error. Julia's matrix `norm(A,1)` is entrywise, so induced
norms are explicitly computed with `opnorm`. No positivity repair is made.
"""
function precision_errors(P::AbstractMatrix,truth::AbstractMatrix)
    size(P) == size(truth) || throw(DimensionMismatch("matrix dimensions must agree"))
    all(isfinite,P) && all(isfinite,truth) || throw(ArgumentError("nonfinite matrix"))
    D = P-truth
    (max_entry=maximum(abs,D),entry_l1=sum(abs,D),frobenius=norm(D),
     spectral=opnorm(D,2),induced_l1=opnorm(D,1),induced_linf=opnorm(D,Inf),
     relative_frobenius=norm(D)/max(norm(truth),eps(Float64)))
end
