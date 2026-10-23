# Write the problems of the CLIME benchmark into a directory:
#
#   julia --project=benchmark/clime benchmark/clime/prepare.jl OUT_DIR [quick]
#
# One subdirectory per problem (see problems.jl). The full set has the AR(1),
# band and Erdős–Rényi models with p = 100, 200, 500, 1000, 2000 and n = p/2 and
# 2p, plus AR(1) with p = 4000; `quick` writes a small set for testing.

include(joinpath(@__DIR__, "problems.jl"))

function problem_list(quick::Bool)
  quick && return [("ar1", 50, 100), ("band", 60, 30), ("er", 80, 160)]
  list = Tuple{String,Int,Int}[]
  for model in ("ar1", "band", "er"), p in (100, 200, 500, 1000, 2000), n in (p ÷ 2, 2p)
    push!(list, (model, p, n))
  end
  push!(list, ("ar1", 4000, 2000), ("ar1", 4000, 8000))
  list
end

outdir = ARGS[1]
quick = length(ARGS) > 1 && ARGS[2] == "quick"
for (k, (model, p, n)) in enumerate(problem_list(quick))
  seed = 1000 + k
  dir = joinpath(outdir, problem_name(model, p, n, seed))
  isdir(dir) || write_problem(dir, model, p, n, seed)
  println(dir)
end
