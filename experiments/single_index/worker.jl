using CovarianceSelection, LinearAlgebra, Random, TOML
using CovarianceSelection.SingleIndexGraph
include("dgp.jl")
BLAS.set_num_threads(1)

# CSV strings are quoted; row count and all failures remain explicit.
function csvrow(io,items)
    println(io,join([x isa AbstractString ? "\""*replace(x,'"'=>"\"\"",'\n'=>' ')*"\"" : string(x) for x in items],','))
end
function runshard(planpath,taskid,outdir)
    plan = TOML.parsefile(planpath)
    tasks = [(c,r:min(r+Int(plan["shard_size"])-1,Int(c["replicates"]))) for c in plan["scenario"] for r in 1:Int(plan["shard_size"]):Int(c["replicates"])]
    c,reps = tasks[taskid]
    mkpath(outdir)
    path = joinpath(outdir,"shard-$(lpad(taskid,5,'0')).csv")
    isfile(path) && error("refusing to overwrite completed shard")
    tmp = path*".partial"
    open(tmp,"w") do io
        csvrow(io,["scenario","rep","method","index_mode","h_factor","pilot_factor","penalty","kind","entry_a","entry_b","point","estimate","se","truth","covered","length","failure","metric","value","seed","seconds","selected_h","selected_pilot_h","selected_penalty"])
        for rep in reps
            seed = Int(plan["seed_base"])+Int(get(c,"seed_group",c["id"]))*100000+rep
            D = generate_data(seed,c)
            entries = [(1,1),(1,2),(1,Int(c["p"])-2)]
            grid = collect(range(Float64(c["grid_min"]),Float64(c["grid_max"]);length=Int(c["grid_n"])))
            truths = [inv(Symmetric(D.truth(t)))[a,b] for t in grid,(a,b) in entries]
            for mode in c["index_modes"]
                ti = D.t
                indexerror = 0.0
                indexfailure = ""
                if mode == "estimated"
                    try
                        idx = estimate_index(D.X,D.score;blocks=Int(get(c,"mom_blocks",5)),
                                             multiplier=Float64(get(c,"index_penalty",0.5)))
                        ti = D.Z*idx.direction
                        indexerror = norm(idx.direction-D.gamma)
                    catch err
                        err isa InterruptException && rethrow()
                        indexfailure = sprint(showerror,err)
                        indexerror = NaN
                    end
                end
                for hf in c["h_factors"],pf in c["pilot_factors"],pen in c["penalties"]
                    pen_use = Float64(pen)
                    h = Float64(hf)*2.5*Int(c["n"])^(-0.2)
                    hb = Float64(pf)*2.5*Int(c["n"])^(-0.2)
                    tunefailure = indexfailure
                    if get(c,"select_tuning",false) && isempty(tunefailure)
                        try
                            tune = select_tuning(D.X,ti;points=[-0.5,0.0,0.5],entries,
                                bandwidths=[0.6,0.9,1.2],pilot_bandwidths=[0.6,0.9,1.2],
                                multipliers=[0.4,0.7,1.0],undersmooth=Int(c["n"])^(-0.05),
                                rng=MersenneTwister(seed+700000000))
                            h,hb,pen_use = tune.bandwidth,tune.pilot_bandwidth,tune.multiplier
                        catch err
                            err isa InterruptException && rethrow()
                            tunefailure = sprint(showerror,err)
                        end
                    end
                    for method in c["methods"]
                        start = time()
                        failure = tunefailure
                        fit = nothing
                        try
                            isempty(failure) || error(failure)
                            pil = startswith(method,"aclime") ? :aclime : startswith(method,"scaled") ? :scaledlasso : :sqrtlasso
                            correction = endswith(method,"onestep") ? :onestep : :block
                            oracle = method == "oracle" ? ((point,J,A)->D.truth(point)[A,A]\D.truth(point)[A,J]) : nothing
                            fit = fit_precision(D.X,ti;points=grid,entries,bandwidth=h,pilot_bandwidth=hb,
                                pilot=pil,correction,multiplier=pen_use,delta=Float64(get(c,"aclime_delta",2)),
                                degree=Int(get(c,"smooth_degree",2)),refit=occursin("refit",method),oracle_coefficients=oracle)
                        catch err
                            err isa InterruptException && rethrow()
                            failure = sprint(showerror,err)
                        end
                        elapsed = time()-start
                        ci = fit === nothing ? nothing : confint(fit)
                        for (e,(a,b)) in enumerate(entries),(g,t) in enumerate(grid)
                            fail = fit === nothing ? failure : fit.failure[g,e]
                            valid = isempty(fail)
                            value = valid ? fit.estimate[g,e] : NaN
                            se = valid ? fit.se[g,e] : NaN
                            cover = valid && ci.lower[g,e] <= truths[g,e] <= ci.upper[g,e]
                            csvrow(io,[c["name"],rep,method,mode,hf,pf,pen,"pointwise",a,b,t,value,se,truths[g,e],Int(cover),valid ? ci.upper[g,e]-ci.lower[g,e] : NaN,fail,"index_error",indexerror,seed,elapsed,h,hb,pen_use])
                        end
                        if Int(c["draws"]) >= 100
                            for joint in (false,true)
                                band = nothing
                                fail = fit === nothing ? failure : ""
                                try
                                    fit === nothing && error(failure)
                                    band = confband(fit;draws=Int(c["draws"]),joint,rng=MersenneTwister(seed+800000000))
                                catch err
                                    err isa InterruptException && rethrow()
                                    fail = sprint(showerror,err)
                                end
                                for (e,(a,b)) in enumerate(entries)
                                    cover = band !== nothing && all(band.lower[:,e] .<= truths[:,e] .<= band.upper[:,e])
                                    width = band === nothing ? NaN : sum(band.upper[:,e]-band.lower[:,e])/length(grid)
                                    csvrow(io,[c["name"],rep,method,mode,hf,pf,pen,joint ? "family_component" : "curve",a,b,NaN,NaN,NaN,NaN,Int(cover),width,fail,"critical",band === nothing ? NaN : band.critical[e],seed,elapsed,h,hb,pen_use])
                                end
                                if joint
                                    cover = band !== nothing && all(band.lower .<= truths .<= band.upper)
                                    csvrow(io,[c["name"],rep,method,mode,hf,pf,pen,"family",0,0,NaN,NaN,NaN,NaN,Int(cover),band === nothing ? NaN : sum(band.upper-band.lower)/length(truths),fail,"critical",band === nothing ? NaN : band.critical[1],seed,elapsed,h,hb,pen_use])
                                end
                            end
                        end
                        flush(io)
                    end
                    # Full-matrix norms at zero, for pilots and their one-step corrections.
                    if get(c,"norms",true)
                        for pil in (:scaledlasso,:aclime), refit in get(c,"norm_refits",[false])
                            basename = string(pil)*(refit ? "_refit" : "")
                            failure = tunefailure
                            try
                                isempty(failure) || error(failure)
                                P = precision_pilot(D.X,ti,0.0,hb;method=pil,multiplier=pen_use,refit).precision
                                W = local_design(ti,0.0,h)
                                S = D.X'*(W.weights.*D.X)
                                for (method,Q) in ((basename*"_pilot",P),(basename*"_onestep",P+P'-P'*S*P))
                                    errors = precision_errors(Q,inv(Symmetric(D.truth(0.0))))
                                    for (key,value) in pairs(errors)
                                        csvrow(io,[c["name"],rep,method,mode,hf,pf,pen,"norm",0,0,0.0,NaN,NaN,NaN,0,NaN,"",string(key),value,seed,0,h,hb,pen_use])
                                    end
                                end
                            catch err
                                err isa InterruptException && rethrow()
                                for method in (basename*"_pilot",basename*"_onestep"), metric in
                                    ("max_entry","entry_l1","frobenius","spectral","induced_l1","induced_linf","relative_frobenius")
                                    csvrow(io,[c["name"],rep,method,mode,hf,pf,pen,"norm",0,0,0.0,NaN,NaN,NaN,0,NaN,sprint(showerror,err),metric,NaN,seed,0,h,hb,pen_use])
                                end
                            end
                        end
                    end
                end
            end
            println("scenario=",c["name"]," rep=",rep," seed=",seed)
            flush(stdout)
        end
    end
    mv(tmp,path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    runshard(ARGS[1],parse(Int,ARGS[2]),ARGS[3])
end
