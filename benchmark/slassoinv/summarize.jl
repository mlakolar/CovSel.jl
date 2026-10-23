# Tables of the scaled lasso benchmark from the results of campaign.sbatch.
#
#   julia benchmark/slassoinv/summarize.jl RESULTS_DIR
#
# Prints Markdown tables: speed (julia_*.csv and r_*.csv in RESULTS_DIR/speed) and
# accuracy (RESULTS_DIR/accuracy/*.csv): mean errors with standard errors, and
# for the design of Sun and Zhang the means of their Table 1.

using Printf, Statistics

const RES = ARGS[1]

# Table 1 of Sun and Zhang (2013): model, p, then the spectral, matrix ℓ1 and
# Frobenius errors of SLasso, SLasso/LSE, GLasso and CLIME (means of 100
# replications; NA: not computed)
const SZTABLE = """
1 30 2.46 1.77 2.49 2.29 2.95 2.73 3.09 2.92 4.20 3.37 4.24 3.80
1 60 2.68 2.04 2.94 2.68 3.12 3.17 3.55 3.27 6.41 5.35 7.15 6.32
1 90 2.75 2.09 3.07 2.87 3.21 3.49 3.72 3.42 8.09 6.87 9.25 8.42
1 150 2.84 2.18 3.19 3.05 3.29 3.81 3.88 3.55 10.79 9.32 12.55 11.68
1 300 2.93 2.25 3.29 NA 3.39 4.36 4.06 NA 15.83 13.89 18.44 NA
1 1000 3.08 2.38 3.39 NA 3.51 5.13 4.44 NA 30.55 26.68 35.11 NA
2 30 0.72 1.08 0.82 0.81 1.27 1.86 1.49 1.45 1.86 2.40 1.84 1.87
2 60 1.06 1.41 1.15 1.19 1.93 2.68 2.21 2.20 3.27 4.19 3.18 3.42
2 90 1.48 1.73 1.54 1.61 2.58 3.81 2.89 2.90 4.42 6.18 4.40 4.65
2 150 1.96 2.04 2.02 2.06 3.25 5.21 3.60 3.65 5.95 9.17 6.19 6.33
2 300 2.88 2.34 2.89 NA 4.45 6.75 4.92 NA 9.26 13.99 9.79 NA
2 1000 5.46 4.88 5.52 NA 7.09 10.26 7.98 NA 18.85 26.08 20.81 NA
3 30 1.84 1.28 2.08 1.63 2.30 2.00 2.59 2.17 2.69 2.15 2.91 2.37
3 60 2.18 1.58 2.63 2.10 2.63 2.46 3.10 2.65 4.10 3.41 4.84 3.98
3 90 2.34 1.71 2.84 2.38 2.77 2.73 3.30 2.91 5.19 4.40 6.25 5.37
3 150 2.51 1.84 3.06 2.76 2.93 3.04 3.45 3.18 6.93 5.96 8.43 7.75
3 300 2.70 1.99 3.26 NA 3.10 3.39 3.58 NA 10.18 8.89 12.41 NA
3 1000 2.94 2.16 3.47 NA 3.32 4.07 3.73 NA 19.63 17.04 23.55 NA
"""

sztable() = Dict((parse(Int, f[1]), parse(Int, f[2])) => f[3:end]
                 for f in split.(filter(!isempty, split(SZTABLE, '\n'))))

const METHODS = ["SLasso", "SLasso/LSE", "CLIME", "ACLIME", "ACLIME (scaled lasso pilot)"]
const NORMS = ("spectral", "matrix ℓ1", "Frobenius")

# rows of the accuracy files: (design, model, p, rep, method) => (e2, e1, eF, parameter, seconds)
function accuracy()
  d = Dict{NTuple{5,Any},NTuple{5,Float64}}()
  for f in readdir(joinpath(RES, "accuracy"); join=true)
    endswith(f, ".csv") || continue
    for line in eachline(f)
      x = split(line, ',')
      length(x) == 10 || continue
      d[(x[1], x[2], parse(Int, x[3]), parse(Int, x[4]), x[5])] = Tuple(parse.(Float64, x[6:10]))
    end
  end
  d
end

meanse(v) = isempty(v) ? (NaN, NaN, 0) : (mean(v), length(v) > 1 ? std(v) / sqrt(length(v)) : NaN, length(v))
cell(v) = (m = meanse(v); m[3] == 0 ? "—" : @sprintf("%.2f (%.2f)", m[1], m[2]))

# models in numeric order when they are numbers (the block sizes of hadamard)
function cases(d, design)
  sort(unique((k[2], k[3]) for k in keys(d) if k[1] == design);
       by=c -> (something(tryparse(Int, c[1]), typemax(Int)), c[1], c[2]))
end

values(d, design, model, p, method, i) =
  [v[i] for (k, v) in d if k[1] == design && k[2] == model && k[3] == p && k[5] == method]

function accuracy_tables(d)
  names = Dict("band" => "band", "ar1" => "AR(1)", "er" => "Erdős–Rényi")
  for (design, title) in (("sz", "Design of Sun and Zhang (2013): n = 100"),
                          ("clz", "Design of Cai, Liu and Zhou (2016): n = 200"),
                          ("hadamard", "Hadamard blocks of size k: p = 256, n = 2000"))
    cs = cases(d, design)
    isempty(cs) && continue
    ms = filter(m -> any(k -> k[1] == design && k[5] == m, keys(d)), METHODS)
    sz = design == "sz" ? sztable() : nothing
    for (i, norm) in enumerate(NORMS)
      println("\n### $title, $norm norm: mean error (standard error)\n")
      extra = design == "sz" ? " paper: SLasso | paper: SLasso/LSE | paper: CLIME |" : ""
      println("| model | p | reps |", join((" $m |" for m in ms)), extra)
      println("|---|---:|---:|", repeat("---:|", length(ms) + (design == "sz" ? 3 : 0)))
      for (model, p) in cs
        reps = length(values(d, design, model, p, "SLasso", 1))
        label = design == "hadamard" ? "k = $model" : get(names, model, "model $model")
        row = "| $label | $p | $reps |" * join((" $(cell(values(d, design, model, p, m, i))) |" for m in ms))
        if design == "sz"
          t = get(sz, (parse(Int, model), p), fill("NA", 12))
          row *= " $(t[4i-3]) | $(t[4i-2]) | $(t[4i]) |"
        end
        println(row)
      end
    end
    println("\nMean tuning parameter (λ₀ of SLasso, λ of CLIME on the correlation matrix, δ of ACLIME) and median seconds per replication:\n")
    println("| model | p |", join((" $m |" for m in ms)))
    println("|---|---:|", repeat("---:|", length(ms)))
    for (model, p) in cs
      label = design == "hadamard" ? "k = $model" : get(names, model, "model $model")
      println("| $label | $p |", join((begin
        par, sec = values(d, design, model, p, m, 4), values(d, design, model, p, m, 5)
        isempty(par) ? " — |" : @sprintf(" %.3f, %.3g s |", mean(par), median(sec))
      end for m in ms)))
    end
  end
end

# (results computed before the rename label the method scaledlasso)
csvlines(prefix) = [split(replace(l, "scaledlasso" => "slassoinv"), ',')
                    for f in readdir(joinpath(RES, "speed"); join=true)
                    if startswith(basename(f), prefix) && endswith(f, ".csv") for l in eachline(f)]

function speed_tables()
  isdir(joinpath(RES, "speed")) || return
  rows = csvlines("julia_")
  isempty(rows) && return
  key(x) = (x[1], parse(Int, x[2]), parse(Int, x[3]))
  probs = sort(unique(key.(rows)); by=k -> (k[1], k[2], k[3]))
  get1(k, method, nt) = (r = filter(x -> key(x) == k && x[4] == method && (nt == 0 || parse(Int, x[5]) == nt), rows);
                         isempty(r) ? nothing : r[end])
  rrows = csvlines("r_")
  getr(k) = (r = filter(x -> x[1] == "$(k[1])_p$(k[2])_n$(k[3])_s$(7 + k[2] + k[3])", rrows); isempty(r) ? nothing : r[end])
  fmt(t) = t < 0.01 ? @sprintf("%.4f", t) : t < 1 ? @sprintf("%.3f", t) : t < 100 ? @sprintf("%.2f", t) : @sprintf("%.0f", t)
  println("\n### Seconds for the whole estimate (relative to slassoinv on one thread)\n")
  println("| problem | slassoinv | refit | 8 threads | CoordinateDescent | scalreg | nonzero entries |")
  println("|---|---:|---:|---:|---:|---:|---:|")
  for k in probs
    s = get1(k, "slassoinv", 1)
    s === nothing && continue
    t1 = parse(Float64, s[6])
    r = get1(k, "slassoinv refit", 1)
    th = get1(k, "slassoinv", 8)
    c = get1(k, "CoordinateDescent", 1)
    R = getr(k)
    name = Dict("ar1" => "AR(1)", "band" => "band", "er" => "Erdős–Rényi")[k[1]]
    println("| $name, p = $(k[2]), n = $(k[3]) | ", fmt(t1), " | ",
      r === nothing ? "—" : fmt(parse(Float64, r[6])), " | ",
      th === nothing ? "—" : @sprintf("%s (%.1f×)", fmt(parse(Float64, th[6])), t1 / parse(Float64, th[6])), " | ",
      c === nothing ? "—" : @sprintf("%s (%.0f×)", fmt(parse(Float64, c[6])), parse(Float64, c[6]) / t1), " | ",
      R === nothing ? "—" : @sprintf("%s (%.0f×)", fmt(parse(Float64, R[3])), parse(Float64, R[3]) / t1), " | ",
      s[7], " |")
  end
  println("\nLargest difference to slassoinv, relative to its largest entry: CoordinateDescent ",
    join(unique(@sprintf("%.0e", parse(Float64, x[8])) for x in rows if x[4] == "CoordinateDescent"), ", "),
    isempty(rrows) ? "" : "; scalreg " * join((@sprintf("%.0e", parse(Float64, x[5])) for x in rrows), ", "))
end

speed_tables()
accuracy_tables(accuracy())
