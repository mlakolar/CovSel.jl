"""
    select_tuning(X, index; points, entries, bandwidths, pilot_bandwidths,
                  multipliers=(0.5,1,1.5), folds=3, undersmooth=1,
                  rng=default_rng(), kernel=biweight, rule=:prediction, refit=false,
                  bandwidth_cap=2.5*n^(-1/5), pilot_ratio=1.5, pilot_se=1)

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

`rule=:onese` favors the smallest nuisance support within one fold standard error
of the best prediction loss, then the largest eligible final bandwidth.
`rule=:inference` additionally caps final candidates at `bandwidth_cap` and
requires pilot candidates at least `pilot_ratio` times the largest capped final
candidate. `pilot_se=0` uses minimum pilot loss instead of the one-SE window.
`refit=true` scores support-refitted regressions. These are empirical
selection safeguards, not coverage guarantees; see `DIAGONAL_TUNING.md`.
"""
function select_tuning(X,t;points,entries,bandwidths,pilot_bandwidths,
                       multipliers=(0.5,1.0,1.5),folds::Integer=3,
                       undersmooth::Real=1,rng::AbstractRNG=default_rng(),kernel=biweight,
                       rule::Symbol=:prediction,refit::Bool=false,
                       bandwidth_cap::Real=2.5*size(X,1)^(-0.2),pilot_ratio::Real=1.5,
                       pilot_se::Real=1)
    _data(X,t)
    n,p = size(X)
    rule in (:prediction,:onese,:inference) || throw(ArgumentError("unknown tuning rule"))
    isfinite(bandwidth_cap) && bandwidth_cap>0 && isfinite(pilot_ratio) && pilot_ratio>=1 ||
        throw(ArgumentError("invalid inference bandwidth safeguards"))
    isfinite(pilot_se) && pilot_se>=0 || throw(ArgumentError("invalid pilot SE allowance"))
    2 <= folds <= n÷10 || throw(ArgumentError("invalid fold count"))
    isfinite(undersmooth) && 0 < undersmooth <= 1 || throw(ArgumentError("invalid undersmoothing factor"))
    hs,hbs,ms = collect(bandwidths),collect(pilot_bandwidths),collect(multipliers)
    all(v -> !isempty(v) && all(x -> isfinite(x) && x>0,v),(hs,hbs,ms)) ||
        throw(ArgumentError("candidate grids must be nonempty, positive and finite"))
    pilot_validation_h, final_validation_h = minimum(hbs),minimum(hs)
    if rule === :inference
        hs = sort(unique(min.(hs,bandwidth_cap)))
        hbs = filter(>=(pilot_ratio*maximum(hs)),hbs)
        isempty(hbs) && throw(ArgumentError("no pilot bandwidth satisfies the inference floor"))
    end
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
    pilot_scores = NamedTuple[]
    for hb in hbs,m in ms
        total,weight,complexity,countfits = 0.0,0.0,0.0,0
        foldtotal,foldweight = zeros(folds),zeros(folds)
        try
            for f in 1:folds,point in pts,J in Js
                tr,va = findall(!=(f),foldid),findall(==(f),foldid)
                A = setdiff(1:p,J)
                B = _regression(X[tr,:],t[tr],point,hb,J,A,m,kernel;refit).coefficients
                complexity += count(!iszero,B)/length(J)
                countfits += 1
                w = kernel.((t[va].-point)./pilot_validation_h)
                sum(w)>0 || throw(DomainError(point,"empty validation window"))
                R = X[va,J]-X[va,A]*B
                scale = max.(vec(sum(abs2,X[tr,J];dims=1))./length(tr),eps(Float64))
                loss = sum(w .* vec(sum(R.^2 ./ scale';dims=2)))
                mass = sum(w)*length(J)
                total += loss
                weight += mass
                foldtotal[f] += loss
                foldweight[f] += mass
            end
        catch err
            err isa InterruptException && rethrow()
            total,weight = Inf,1.0
        end
        losses = foldtotal ./ foldweight
        standard_error = all(isfinite,losses) ? sqrt(sum(abs2,losses.-sum(losses)/folds)/(folds*(folds-1))) : Inf
        push!(pilot_scores,(bandwidth=Float64(hb),multiplier=Float64(m),loss=total/weight,
                            standard_error=standard_error,complexity=complexity/max(countfits,1)))
    end
    best = pilot_scores[argmin([x.loss for x in pilot_scores])]
    isfinite(best.loss) || throw(DomainError(pilot_scores,"all pilot candidates failed"))
    if rule !== :prediction
        eligible = filter(x -> isfinite(x.loss) && x.loss <= best.loss+pilot_se*best.standard_error,pilot_scores)
        best = sort(eligible;by=x -> (x.complexity,-x.bandwidth,-x.multiplier))[1]
    end
    final_scores = NamedTuple[]
    for h in hs
        total,weight = 0.0,0.0
        foldtotal,foldweight = zeros(folds),zeros(folds)
        try
            for f in 1:folds,point in pts,J in Js
                tr,va = findall(!=(f),foldid),findall(==(f),foldid)
                A = setdiff(1:p,J)
                B = _regression(X[tr,:],t[tr],point,best.bandwidth,J,A,best.multiplier,kernel;refit).coefficients
                Rt,Rv = X[tr,J]-X[tr,A]*B,X[va,J]-X[va,A]*B
                D = local_design(t[tr],point,h;kernel)
                w = kernel.((t[va].-point)./final_validation_h)
                sum(w)>0 || throw(DomainError(point,"empty validation window"))
                u = (t[va].-point)./h
                V = hcat(ones(length(va)),u,u.^2)
                for a in eachindex(J),b in a:length(J)
                    zt,zv = Rt[:,a].*Rt[:,b],Rv[:,a].*Rv[:,b]
                    scale = max(sum(abs2,zt)/length(tr),eps(Float64))
                    loss = sum(w.*(zv-V*(D.map*zt)).^2)/scale
                    total += loss
                    weight += sum(w)
                    foldtotal[f] += loss
                    foldweight[f] += sum(w)
                end
            end
        catch err
            err isa InterruptException && rethrow()
            total,weight = Inf,1.0
        end
        losses = foldtotal ./ foldweight
        standard_error = all(isfinite,losses) ? sqrt(sum(abs2,losses.-sum(losses)/folds)/(folds*(folds-1))) : Inf
        push!(final_scores,(bandwidth=Float64(h),loss=total/weight,standard_error=standard_error))
    end
    final = final_scores[argmin([x.loss for x in final_scores])]
    isfinite(final.loss) || throw(DomainError(final_scores,"all final candidates failed"))
    if rule !== :prediction
        eligible = filter(x -> isfinite(x.loss) && x.loss <= final.loss+final.standard_error,final_scores)
        final = sort(eligible;by=x -> -x.bandwidth)[1]
    end
    (bandwidth=final.bandwidth*undersmooth,pilot_bandwidth=best.bandwidth,
     multiplier=best.multiplier,pilot_scores=pilot_scores,final_scores=final_scores,
     foldid=foldid,undersmooth=Float64(undersmooth),rule=rule,refit=refit,
     bandwidth_cap=Float64(bandwidth_cap),pilot_ratio=Float64(pilot_ratio),pilot_se=Float64(pilot_se),
     validation_bandwidths=(pilot=pilot_validation_h,final=final_validation_h))
end
