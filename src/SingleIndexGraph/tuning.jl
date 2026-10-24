"""
    select_tuning(X, index; points, entries, bandwidths, pilot_bandwidths,
                  multipliers=(0.5,1,1.5), folds=3, undersmooth=1,
                  rng=default_rng(), kernel=biweight)

Select nuisance bandwidth/penalty by held-out weighted prediction loss, then
select the final polynomial bandwidth by held-out residual-product prediction
loss. Regression nuisances for validation exclude that validation fold.
The index is supplied by the caller and is not re-estimated within folds.
No truth or empirical coverage enters selection. Candidate failures are scored
as Inf; ties choose the first candidate. `undersmooth` multiplies the selected
final bandwidth (e.g. n^(-1/20)); it is a sensitivity rule, not a coverage-optimal
theorem. Pilot selection uses common-complement square-root lasso. A selected
pair's two diagonal products and cross-product contribute equally after scaling
by training second moments. Returns all candidate scores and selected values.
Bandwidth grids and evaluation points must be prespecified.
"""
function select_tuning(X,t;points,entries,bandwidths,pilot_bandwidths,
                       multipliers=(0.5,1.0,1.5),folds::Integer=3,
                       undersmooth::Real=1,rng::AbstractRNG=default_rng(),kernel=biweight)
    _data(X,t)
    n,p = size(X)
    2 <= folds <= n÷10 || throw(ArgumentError("invalid fold count"))
    isfinite(undersmooth) && 0 < undersmooth <= 1 || throw(ArgumentError("invalid undersmoothing factor"))
    hs,hbs,ms = collect(bandwidths),collect(pilot_bandwidths),collect(multipliers)
    all(v -> !isempty(v) && all(x -> isfinite(x) && x>0,v),(hs,hbs,ms)) ||
        throw(ArgumentError("candidate grids must be nonempty, positive and finite"))
    pts = collect(points)
    !isempty(pts) && all(isfinite,pts) || throw(ArgumentError("invalid points"))
    Js = unique([sort(unique(collect(pair))) for pair in entries])
    !isempty(Js) && all(J -> length(J) in (1,2) && all(j -> j isa Integer && 1<=j<=p,J),Js) ||
        throw(ArgumentError("invalid entries"))
    perm = randperm(rng,n)
    foldid = zeros(Int,n)
    foldid[perm] = mod.(0:n-1,folds).+1
    # Candidate losses must use the same held-out observations and weights.
    # Changing the validation window with the candidate would change the risk.
    pilot_validation_h, final_validation_h = minimum(hbs),minimum(hs)
    pilot_scores = NamedTuple[]
    for hb in hbs,m in ms
        total,weight = 0.0,0.0
        try
            for f in 1:folds,point in pts,J in Js
                tr,va = findall(!=(f),foldid),findall(==(f),foldid)
                A = setdiff(1:p,J)
                B = _regression(X[tr,:],t[tr],point,hb,J,A,m,kernel).coefficients
                w = kernel.((t[va].-point)./pilot_validation_h)
                sum(w)>0 || throw(DomainError(point,"empty validation window"))
                R = X[va,J]-X[va,A]*B
                scale = max.(vec(sum(abs2,X[tr,J];dims=1))./length(tr),eps(Float64))
                total += sum(w .* vec(sum(R.^2 ./ scale';dims=2)))
                weight += sum(w)*length(J)
            end
        catch err
            err isa InterruptException && rethrow()
            total,weight = Inf,1.0
        end
        push!(pilot_scores,(bandwidth=Float64(hb),multiplier=Float64(m),loss=total/weight))
    end
    best = pilot_scores[argmin([x.loss for x in pilot_scores])]
    isfinite(best.loss) || throw(DomainError(pilot_scores,"all pilot candidates failed"))
    final_scores = NamedTuple[]
    for h in hs
        total,weight = 0.0,0.0
        try
            for f in 1:folds,point in pts,J in Js
                tr,va = findall(!=(f),foldid),findall(==(f),foldid)
                A = setdiff(1:p,J)
                B = _regression(X[tr,:],t[tr],point,best.bandwidth,J,A,best.multiplier,kernel).coefficients
                Rt,Rv = X[tr,J]-X[tr,A]*B,X[va,J]-X[va,A]*B
                D = local_design(t[tr],point,h;kernel)
                w = kernel.((t[va].-point)./final_validation_h)
                sum(w)>0 || throw(DomainError(point,"empty validation window"))
                u = (t[va].-point)./h
                V = hcat(ones(length(va)),u,u.^2)
                for a in eachindex(J),b in a:length(J)
                    zt,zv = Rt[:,a].*Rt[:,b],Rv[:,a].*Rv[:,b]
                    scale = max(sum(abs2,zt)/length(tr),eps(Float64))
                    total += sum(w.*(zv-V*(D.map*zt)).^2)/scale
                    weight += sum(w)
                end
            end
        catch err
            err isa InterruptException && rethrow()
            total,weight = Inf,1.0
        end
        push!(final_scores,(bandwidth=Float64(h),loss=total/weight))
    end
    final = final_scores[argmin([x.loss for x in final_scores])]
    isfinite(final.loss) || throw(DomainError(final_scores,"all final candidates failed"))
    (bandwidth=final.bandwidth*undersmooth,pilot_bandwidth=best.bandwidth,
     multiplier=best.multiplier,pilot_scores=pilot_scores,final_scores=final_scores,
     foldid=foldid,undersmooth=Float64(undersmooth),
     validation_bandwidths=(pilot=pilot_validation_h,final=final_validation_h))
end
