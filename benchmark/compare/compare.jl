# Compares the runs of run.jl on the two sides: for every case, the fastest
# median time over the runs of each side (load on the machine only adds time),
# the passes or iterations, and the largest difference of the solutions.
#
#     julia benchmark/compare/compare.jl

using Printf, Serialization

dir = joinpath(@__DIR__, "results")
load(side) = [deserialize(joinpath(dir, f)) for f in readdir(dir) if startswith(f, side * "_")]
olds, news = load("old"), load("new")
fastest(runs, key) = minimum(r[key].time for r in runs if haskey(r, key))

cases = sort!(collect(intersect(keys(first(olds)), keys(first(news)))); by = c -> first(news)[c].order)
println("| case | old (ms) | new (ms) | new / old | passes, old / new | largest difference |")
println("|---|---:|---:|---:|---|---:|")
ratios = Float64[]
for c in cases
  told, tnew = fastest(olds, c), fastest(news, c)
  o, nw = first(olds)[c], first(news)[c]
  d = length(o.solution) == length(nw.solution) ? maximum(abs, o.solution - nw.solution; init = 0.0) : NaN
  push!(ratios, tnew / told)
  @printf("| %s | %.2f | %.2f | %.2f | %d / %d | %.0e |\n", c, 1e3told, 1e3tnew, tnew / told, o.passes, nw.passes, d)
end
@printf("\n%d cases, %d rounds of each side; new / old: geometric mean %.2f, range %.2f to %.2f\n",
        length(cases), min(length(olds), length(news)), exp(sum(log, ratios) / length(ratios)), minimum(ratios), maximum(ratios))
