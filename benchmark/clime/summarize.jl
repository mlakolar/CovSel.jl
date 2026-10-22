# Tables of the CLIME benchmark campaign, in Markdown:
#
#   julia --project=benchmark/clime benchmark/clime/summarize.jl RUN_DIR > tables.md
#
# Reads RUN_DIR/problems and RUN_DIR/results (julia/, r/, threads/, aclime/). The
# accuracy of a solver is the largest relative difference between the ℓ1 norms of
# its column solutions and those of ParametricSimplex, over all columns and
# penalties, and its largest constraint violation relative to λ.

using Printf, Statistics
include(joinpath(@__DIR__, "problems.jl"))

run = ARGS[1]
res = joinpath(run, "results")

struct Row
  problem::String; model::String; p::Int; n::Int; solver::String; threads::Int
  mode::String; seconds::Float64; nsolved::Int; viol::Float64
end

function julia_rows(dir)
  rows = Row[]
  isdir(dir) || return rows
  for f in readdir(dir; join=true)
    endswith(f, ".csv") || continue
    for line in eachline(f)
      v = split(line, ',')
      push!(rows, Row(v[1], v[2], parse(Int, v[3]), parse(Int, v[4]), v[5], parse(Int, v[6]), v[7],
        parse(Float64, v[8]), parse(Int, v[11]), parse(Float64, v[12])))
    end
  end
  rows
end

# minimal CSV reader for the files of bench_r.R (quoted strings, no embedded commas
# except in the last column, which is not used)
function readcsv(f)
  lines = readlines(f)
  header = strip.(split(lines[1], ','), '"')
  [Dict(zip(header, strip.(split(l, ','; limit=length(header)), '"'))) for l in lines[2:end]]
end

const RMETHODS = ["fastclime", "flare", "clime_simplex", "clime"]

function r_rows(dir, problems)
  rows = Row[]
  isdir(dir) || return rows
  for f in readdir(dir; join=true)
    endswith(f, "_timing.csv") || continue
    problem, method = split(replace(basename(f), "_timing.csv" => ""), "__")
    meta = problems[problem]
    for d in readcsv(f)
      d["method"] in (method, "fastclime_direct") || continue
      d["status"] in ("ok", "truncated") || continue
      push!(rows, Row(problem, meta.model, meta.p, meta.n, method, 1, "path", parse(Float64, d["elapsed"]),
        parse(Int, d["K"]), NaN))
    end
  end
  rows
end

problems = Dict{String,NamedTuple}()
for name in readdir(joinpath(run, "problems"))
  m = read_meta(joinpath(run, "problems", name))
  problems[name] = (model=m["model"], p=parse(Int, m["p"]), n=parse(Int, m["n"]),
    λ=parse.(Float64, readlines(joinpath(run, "problems", name, "lambda.txt"))))
end

rows = vcat(julia_rows(joinpath(res, "julia")), r_rows(joinpath(res, "r"), problems))
threadrows = julia_rows(joinpath(res, "threads"))

# accuracy relative to ParametricSimplex
function reference(problem)
  f = joinpath(res, "julia", "$(problem)__ParametricSimplex__l1.bin")
  isfile(f) || return nothing
  m = problems[problem]
  readbin(f, m.p, length(m.λ))
end

function accuracy(problem, solver)
  ref = reference(problem)
  ref === nothing && return (NaN, NaN)
  m = problems[problem]
  if solver in RMETHODS
    f = joinpath(res, "r", "$(problem)__$(solver)_diag.csv")
    isfile(f) || return (NaN, NaN)
    gap = viol = 0.0
    for d in readcsv(f)
      l1 = tryparse(Float64, d["l1"])
      (l1 === nothing || isnan(l1)) && continue
      k, col = parse(Int, d["k"]), parse(Int, d["col"])
      r = ref[col, k]
      isnan(r) && continue
      gap = max(gap, abs(l1 - r) / max(r, 1e-12))
      viol = max(viol, parse(Float64, d["viol_rel"]))
    end
    return (gap, viol)
  else
    f = joinpath(res, "julia", "$(problem)__$(solver)__l1.bin")
    isfile(f) || return (NaN, NaN)
    L = readbin(f, m.p, length(m.λ))
    ok = .!isnan.(L) .& .!isnan.(ref)
    gap = any(ok) ? maximum(abs.(L[ok] .- ref[ok]) ./ max.(ref[ok], 1e-12)) : NaN
    vs = [r.viol for r in rows if r.problem == problem && r.solver == solver && r.mode == "path"]
    return (gap, isempty(vs) ? NaN : maximum(vs))
  end
end

fmt(t) = isnan(t) ? "—" : t < 0.01 ? @sprintf("%.4f", t) : t < 10 ? @sprintf("%.3f", t) : t < 100 ? @sprintf("%.1f", t) : @sprintf("%.0f", t)
# a negative violation (strictly feasible) shows as 0
fmte(x) = isnan(x) ? "—" : x <= 0 ? "0" : @sprintf("%.0e", x)

const SOLVERS = ["ParametricSimplex", "DualSimplex", "HiGHS", "PDHG", "fastclime", "flare", "clime_simplex", "clime"]

function timing(problem, solver, mode; threads=1, rows=rows)
  t = [r.seconds for r in rows if r.problem == problem && r.solver == solver && r.mode == mode && r.threads == threads]
  isempty(t) ? NaN : minimum(t)
end

sortedproblems = sort(collect(keys(problems)); by=k -> (problems[k].model, problems[k].p, problems[k].n))

for (mode, title) in (("path", "the whole path of 30 penalties"), ("single", "the smallest penalty alone"))
  println("\n### Seconds for $title, one thread\n")
  solvers = mode == "single" ? SOLVERS[1:4] : SOLVERS
  println("| model | p | n | ", join(solvers, " | "), " |")
  println("|---|---:|---:|", repeat("---:|", length(solvers)))
  for pr in sortedproblems
    m = problems[pr]
    ts = [timing(pr, s, mode) for s in solvers]
    all(isnan, ts) && continue
    println("| $(m.model) | $(m.p) | $(m.n) | ", join(fmt.(ts), " | "), " |")
  end
end

println("\n### Accuracy: largest relative difference of the column ℓ1 norms from ParametricSimplex / largest relative constraint violation\n")
println("| model | p | n | ", join(SOLVERS[2:end], " | "), " |")
println("|---|---:|---:|", repeat("---:|", length(SOLVERS) - 1))
for pr in sortedproblems
  m = problems[pr]
  acc = [accuracy(pr, s) for s in SOLVERS[2:end]]
  all(a -> isnan(a[1]), acc) && continue
  println("| $(m.model) | $(m.p) | $(m.n) | ", join(["$(fmte(a[1])) / $(fmte(a[2]))" for a in acc], " | "), " |")
end

if !isempty(threadrows)
  println("\n### ParametricSimplex with several threads: seconds for the path (speedup over one thread)\n")
  println("| model | p | n | 1 | 2 | 4 | 8 |")
  println("|---|---:|---:|---:|---:|---:|---:|")
  for pr in sortedproblems
    m = problems[pr]
    t1 = timing(pr, "ParametricSimplex", "path")
    ts = [timing(pr, "ParametricSimplex", "path"; threads=t, rows=threadrows) for t in (2, 4, 8)]
    all(isnan, ts) && continue
    println("| $(m.model) | $(m.p) | $(m.n) | $(fmt(t1)) | ",
      join(["$(fmt(t)) ($(isnan(t) ? "—" : @sprintf("%.1f", t1 / t))×)" for t in ts], " | "), " |")
  end
end

adir = joinpath(res, "aclime")
if isdir(adir) && !isempty(readdir(adir))
  println("\n### Replication of Table 1 of Cai, Liu and Zhou (2016): mean error (standard error)\n")
  ncols = length(split(readline(first(filter(f -> endswith(f, ".csv"), readdir(adir; join=true)))), ',')) - 3
  variants = ncols >= 12 ? 4 : 3
  println("| model | p | reps | norm | ACLIME | CLIME (trace CV) | CLIME (likelihood CV) |",
    variants == 4 ? " ACLIME, truncate=false |" : "")
  println("|---|---:|---:|---|---:|---:|---:|", variants == 4 ? "---:|" : "")
  for model in ("band", "ar1", "er"), p in (50, 100, 200)
    f = joinpath(adir, "$(model)_p$p.csv")
    isfile(f) || continue
    lines = readlines(f)
    header = split(lines[1], ',')
    m = parse.(Float64, split(lines[2], ',')[4:end])
    s = parse.(Float64, split(lines[3], ',')[4:end])
    reps = split(lines[2], ',')[3]
    for (j, norm) in enumerate(("spectral", "matrix ℓ1", "Frobenius"))
      cells = [@sprintf("%.2f (%.2f)", m[3(q-1)+j], s[3(q-1)+j]) for q in 1:variants]
      println("| $model | $p | $reps | $norm | ", join(cells, " | "), " |")
    end
  end
end
