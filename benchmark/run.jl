# Run the suite in benchmarks.jl: check that differencePrecisionActiveShooting
# agrees with the naive reference solver, then print the median time, memory
# and number of allocations of each benchmark.
#
#   julia --project=benchmark benchmark/run.jl [results.json]
#
# With a file name the results are also saved, for comparing two runs with
# BenchmarkTools.judge. COVSEL_BENCHMARK_QUICK=1 runs a small version.

include("benchmarks.jl")
using Printf

println("Julia $VERSION, CovarianceSelection $(pkgversion(CovarianceSelection)), ProximalSolvers $(pkgversion(ProximalSolvers)), ",
  "$(Threads.nthreads()) threads, $(BLAS.get_num_threads()) BLAS threads\n")

println("Largest absolute difference between differencePrecisionActiveShooting and the reference solver:")
for p in DIMENSIONS, λ in LAMBDAS
  Sx, Sy = PROBLEMS[p]
  naive = Reference.differencePrecisionNaive(Matrix(Sx), Matrix(Sy), λ, ones(p, p))
  shooting = differencePrecisionActiveShooting(Sx, Sy, NormL1(λ))
  @printf("  p = %3d, λ = %.1f: %.1e\n", p, λ, maximum(abs, Matrix(shooting) - naive))
end

if QUICK
  results = run(SUITE; seconds=0.1, samples=1)
else
  tune!(SUITE)
  results = run(SUITE; verbose=true)
end

rows = sort([(join(path, " / "), trial) for (path, trial) in BenchmarkTools.leaves(median(results))]; by=first)
width = maximum(length(first(r)) for r in rows) + 2
println("\n", rpad("benchmark", width), lpad("median", 11), lpad("memory", 11), lpad("allocs", 9))
for (name, trial) in rows
  println(rpad(name, width),
    lpad(BenchmarkTools.prettytime(time(trial)), 11),
    lpad(BenchmarkTools.prettymemory(memory(trial)), 11),
    lpad(allocs(trial), 9))
end

if !isempty(ARGS)
  BenchmarkTools.save(ARGS[1], results)
  println("\nresults saved to ", ARGS[1])
end
