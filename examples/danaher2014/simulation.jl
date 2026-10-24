module Danaher2014

using CovarianceSelection
using LinearAlgebra, Printf, Random, Statistics
using ProximalBase: FiberSum, NormL1Fused, prox
import ProximalBase, ProximalSolvers

const PAPER = "https://pmc.ncbi.nlm.nih.gov/articles/PMC4012833/"
const OPTIONS = ADMMOptions(; abstol=1e-9, reltol=1e-8, maxiter=20_000)

# A preferential-attachment tree: one edge per new vertex, so a block of q
# vertices has q-1 edges. The paper does not supply its graph or random seed.
function powerlaw_tree(rng, q)
    edges = [(1, 2)]
    stubs = [1, 2]
    for v in 3:q
        parent = rand(rng, stubs)
        push!(edges, (parent, v))
        push!(stubs, parent, v)
    end
    edges
end

# Section 7.1.1, literally: draw edge weights, divide each row by 1.5 times
# its off-diagonal absolute row sum, symmetrize, invert, standardize to a
# correlation matrix, and multiply its off-diagonal entries by 0.6.
function population(rng; p=500, blocks=10)
    p % blocks == 0 && p >= 2blocks && blocks >= 3 ||
        throw(ArgumentError("p must be a multiple of blocks, with at least two vertices per block and three blocks"))
    q = p ÷ blocks
    Σ = zeros(p, p, 3)
    graph = falses(p, p, 3)
    rejected = 0
    for block in 1:blocks
        B, edges = Matrix{Float64}(I, q, q), Tuple{Int,Int}[]
        accepted = false
        for attempt in 1:10_000
            edges = powerlaw_tree(rng, q)
            W = zeros(q, q)
            for (i, j) in edges
                W[i, j] = W[j, i] = (rand(rng, Bool) ? 1 : -1) * (0.1 + 0.3rand(rng))
            end
            W ./= 1.5sum(abs, W; dims=2)
            A = Matrix(Symmetric(I + (W + W')/2))
            # Row normalization followed by symmetrization does not guarantee
            # positive definiteness for every tree. Reject the draw rather
            # than silently changing the printed covariance construction.
            if !isposdef(Symmetric(A))
                rejected += 1
                continue
            end
            C = inv(cholesky(Symmetric(A)))
            d = sqrt.(diag(C))
            B = 0.6 .* (C ./ (d*d')) + 0.4I
            LinearAlgebra.copytri!(B, 'U')
            accepted = true
            break
        end
        accepted || error("could not generate a positive definite block")
        ix = (block-1)*q+1:block*q
        for k in 1:3
            present = block <= blocks-k+1
            Σ[ix, ix, k] = present ? B : Matrix{Float64}(I, q, q)
            if present
                for (i, j) in edges
                    graph[ix[i], ix[j], k] = graph[ix[j], ix[i], k] = true
                end
            end
        end
    end
    Θ = cat([inv(cholesky(Symmetric(Σ[:, :, k]))) for k in 1:3]...; dims=3)
    (; Σ, Θ, graph, rejected)
end

function sample(rng, model, n)
    n > 1 || throw(ArgumentError("n must exceed one"))
    Y = [randn(rng, n, size(model.Σ, 1)) * cholesky(Symmetric(model.Σ[:, :, k])).U for k in 1:3]
    # JGL centers each group and uses the ML covariance, divisor n, not n-1.
    S = cat([cov(y; corrected=false) for y in Y]...; dims=3)
    (; Y, S)
end

function fixedpoint(Z, S, λ1, λ2)
    G = similar(Z)
    for k in axes(Z, 3)
        G[:, :, k] = S[:, :, k] - inv(Symmetric(Z[:, :, k]))
    end
    g = FiberSum(NormL1Fused(λ1, λ2); diagonal=NormL1Fused(0.0, λ2))
    maximum(abs, prox(g, Z - 0.1G, 0.1) - Z) / max(1.0, maximum(abs, Z))
end

function metrics(Z, model; method="FGL", edge_tol=1e-5, truth_tol=1e-8, fusion_tol=1e-6)
    p, _, K = size(Z)
    edges = tp = fp = tp_diff = fp_diff = graph_tp = graph_fp = graph_tp_diff = graph_fp_diff = 0
    sse = l1 = kl = 0.0
    for k in 1:K
        A, Ω, Σ = Z[:, :, k], model.Θ[:, :, k], model.Σ[:, :, k]
        F = cholesky(Symmetric(A))
        # The paper's dKL omits -p/2 per class. Save it and the usual KL.
        kl += (dot(A, Σ) - logdet(F) - logdet(cholesky(Symmetric(Σ))) - p)/2
        for j in 1:p, i in j+1:p
            selected = abs(A[i, j]) > edge_tol
            truth = abs(Ω[i, j]) > truth_tol
            skeleton = model.graph[i, j, k]
            edges += selected
            tp += selected && truth
            fp += selected && !truth
            graph_tp += selected && skeleton
            graph_fp += selected && !skeleton
            sse += 2abs2(A[i, j] - Ω[i, j])
            l1 += 2abs(A[i, j])
        end
    end
    threshold = method == "GL" ? 1e-2 : fusion_tol
    for k in 1:K-1, l in k+1:K, j in 1:p, i in j+1:p
        selected = abs(Z[i, j, k] - Z[i, j, l]) > threshold
        truth = abs(model.Θ[i, j, k] - model.Θ[i, j, l]) > truth_tol
        skeleton = model.graph[i, j, k] != model.graph[i, j, l]
        tp_diff += selected && truth
        fp_diff += selected && !truth
        graph_tp_diff += selected && skeleton
        graph_fp_diff += selected && !skeleton
    end
    (; edges, tp, fp, tp_diff, fp_diff, graph_tp, graph_fp, graph_tp_diff, graph_fp_diff,
       sse, l1, kl, dkl_paper=kl+K*p/2)
end

function write_tsv(path, rows)
    isempty(rows) && return
    open(path, "w") do io
        println(io, join(string.(keys(first(rows))), '\t'))
        for row in rows
            println(io, join(values(row), '\t'))
        end
    end
end

const MEASURES = (:edges, :tp, :fp, :tp_diff, :fp_diff, :graph_tp, :graph_fp,
                  :graph_tp_diff, :graph_fp_diff, :sse, :l1, :kl, :dkl_paper,
                  :seconds, :iterations, :fixedpoint)

function summarize(rows)
    out = NamedTuple[]
    for key in unique([(r.method, r.lambda1, r.lambda2) for r in rows])
        group = filter(r -> (r.method, r.lambda1, r.lambda2) == key, rows)
        row = (; method=key[1], lambda1=key[2], lambda2=key[3], replicates=length(group))
        for field in MEASURES
            v = [getproperty(r, field) for r in group]
            se = length(v) > 1 ? std(v)/sqrt(length(v)) : NaN
            row = merge(row, NamedTuple{(Symbol(field, "_mean"), Symbol(field, "_se"))}((mean(v), se)))
        end
        push!(out, row)
    end
    out
end

function configuration(args; quick=get(ENV, "COVSEL_EXAMPLES_QUICK", "0") == "1")
    opts = Dict{String,String}()
    allowed = ("mode", "p", "n", "reps", "seed", "lambda1", "lambda2", "out", "export-data")
    for arg in args
        startswith(arg, "--") && occursin('=', arg) || error("use --name=value; see examples/danaher2014/README.md")
        key, value = split(arg[3:end], '='; limit=2)
        key in allowed || error("unknown option --$key")
        opts[key] = value
    end
    mode = get(opts, "mode", quick ? "quick" : "demo")
    mode in ("quick", "demo", "pilot", "paper") || error("mode must be quick, demo, pilot, or paper")
    defaults = mode == "quick" ? (50, 40, 1, [0.25, 0.175], [0.025, 0.1]) :
               mode == "demo" ? (100, 150, 3, [0.3, 0.225, 0.175, 0.1], [0.025, 0.1, 0.2]) :
               mode == "pilot" ? (500, 150, 1, [0.175], [0.025, 0.1]) :
               (500, 150, 100, collect(0.35:-0.025:0.05), [0.025, 0.05, 0.075, 0.1])
    p, n, reps = [parse(Int, get(opts, name, string(defaults[i]))) for (i, name) in enumerate(("p", "n", "reps"))]
    reps > 0 || error("reps must be positive")
    λ1 = parse.(Float64, split(get(opts, "lambda1", join(defaults[4], ',')), ','))
    λ2 = parse.(Float64, split(get(opts, "lambda2", join(defaults[5], ',')), ','))
    all(x -> isfinite(x) && x > 0, λ1) || error("lambda1 values must be finite and positive")
    all(x -> isfinite(x) && x >= 0, λ2) || error("lambda2 values must be finite and nonnegative")
    seed = parse(Int, get(opts, "seed", "2014"))
    out = abspath(get(opts, "out", joinpath(@__DIR__, "output", mode)))
    export_data = get(opts, "export-data", "false") == "true"
    (; mode, p, n, reps, seed, lambda1=sort(unique(λ1); rev=true),
       lambda2=sort(unique([0.0; λ2])), out, export_data)
end

function run(config)
    BLAS.set_num_threads(1)
    mkpath(config.out)
    # A new output folder prevents mixing or overwriting separate runs.
    rawpath = joinpath(config.out, "replicates.tsv")
    isfile(rawpath) && error("$rawpath already exists; choose a new --out directory")
    model = population(Xoshiro(config.seed); p=config.p)
    edgecounts = [count(model.graph[:, :, k]) ÷ 2 for k in 1:3]
    actualcounts = [count(abs.(tril(model.Θ[:, :, k], -1)) .> 1e-8) for k in 1:3]
    open(joinpath(config.out, "metadata.txt"), "w") do io
        println(io, "paper = ", PAPER)
        for field in keys(config)
            println(io, field, " = ", getproperty(config, field))
        end
        println(io, "julia = ", VERSION, "\nblas = ", BLAS.get_config())
        println(io, "covarianceselection = ", Base.pkgversion(CovarianceSelection),
                "\nproximalbase = ", Base.pkgversion(ProximalBase),
                "\nproximalsolvers = ", Base.pkgversion(ProximalSolvers))
        println(io, "covariance_divisor = n\nlikelihood_weights = [1, 1, 1]\npenalize_diag = false")
        println(io, "population_fixed_across_replicates = true\nrejected_non_spd_blocks = ", model.rejected)
        println(io, "generating_graph_edges = ", edgecounts, "\nprecision_edges_at_1e-8 = ", actualcounts)
        println(io, "edge_tol = 1e-5\ntruth_tol = 1e-8\nfgl_difference_tol = 1e-6\ngl_difference_tol = 1e-2")
        println(io, "options = ", OPTIONS)
        println(io, "scope = regenerated simulation design; original graph, seeds and complete grid unavailable")
    end
    println("Danaher et al. (2014), Section 7.1 / Figure 2: ", config.mode)
    println("p=$(config.p), n=$(config.n) per group, K=3, replicates=$(config.reps)")
    println("Generating graph edges: $edgecounts; actual precision support at 1e-8: $actualcounts")
    rows = NamedTuple[]
    # Compile the solver before timing the fits. Every point starts from zero,
    # so timings are cold fits rather than a mixture of warm and cold starts.
    fusedGraphicalLasso(cat(Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2),
                            Matrix{Float64}(I, 2, 2); dims=3), 0.1, 0.1;
                        weights=ones(3), penalize_diag=false, options=OPTIONS)
    for rep in 1:config.reps
        data = sample(Xoshiro(config.seed + rep), model, config.n)
        if rep == 1 && config.export_data
            for k in 1:3
                open(joinpath(config.out, "data_group_$k.tsv"), "w") do io
                    for i in axes(data.Y[k], 1)
                        println(io, join(data.Y[k][i, :], '\t'))
                    end
                end
            end
        end
        for λ2 in config.lambda2, λ1 in config.lambda1
            info = ADMMInfo()
            method = iszero(λ2) ? "GL" : "FGL"
            # Section 7 multiplies displayed penalties by n in its n-weighted
            # objective. Divide the whole objective by n: weights=ones(3),
            # displayed penalties unchanged. With default weights=1/3, both
            # displayed penalties would instead need to be divided by 3.
            seconds = @elapsed Z = fusedGraphicalLasso(data.S, λ1, λ2;
                weights=ones(3), penalize_diag=false, options=OPTIONS, info)
            info.converged || error("unconverged fit at rep=$rep, λ1=$λ1, λ2=$λ2")
            r = fixedpoint(Z, data.S, λ1, λ2)
            r < 1e-5 || error("optimality residual $r at rep=$rep, λ1=$λ1, λ2=$λ2")
            m = metrics(Z, model; method)
            row = merge((; rep, method, lambda1=λ1, lambda2=λ2), m,
                        (; seconds, iterations=info.iterations, fixedpoint=r))
            push!(rows, row)
            # Save after every fit, so an interrupted long run retains results.
            write_tsv(rawpath, rows)
            if rep == 1 && config.export_data
                for k in 1:3
                    open(joinpath(config.out, "fit_$(λ1)_$(λ2)_group_$k.tsv"), "w") do io
                        for i in axes(Z, 1)
                            println(io, join(Z[i, :, k], '\t'))
                        end
                    end
                end
            end
            @printf("rep=%d %-3s λ1=%.3f λ2=%.3f edges=%5d SSE=%8.3f dKL=%8.3f residual=%.1e time=%.2fs\n",
                    rep, method, λ1, λ2, m.edges, m.sse, m.dkl_paper, r, seconds)
        end
    end
    summary = summarize(rows)
    write_tsv(joinpath(config.out, "summary.tsv"), summary)
    println("Results: ", config.out)
    (; rows, summary, model)
end

end
