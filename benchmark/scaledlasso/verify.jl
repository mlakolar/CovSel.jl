# Recompute the scaledlasso estimates stored by speed.jl (est.bin, from the
# benchmark commit) with the current CovSel, and report the largest difference.
#
#   julia --project=benchmark/scaledlasso benchmark/scaledlasso/verify.jl OTHER_RESULTS/speed

using CovSel, LinearAlgebra, Printf

worst = 0.0
for d in sort(filter(isdir, readdir(ARGS[1]; join=true)))
  m = match(r"_p(\d+)_n(\d+)_s", basename(d))
  p, n = parse(Int, m[1]), parse(Int, m[2])
  X = read!(joinpath(d, "X.bin"), Matrix{Float64}(undef, n, p))
  old = read!(joinpath(d, "est.bin"), Matrix{Float64}(undef, p, p))
  new = Matrix(scaledlasso(X' * X / n, n; ntasks=1))
  diff = maximum(abs, new - old) / maximum(abs, old)
  global worst = max(worst, diff)
  @printf("%-28s identical: %-5s largest relative difference %.1e\n", basename(d), new == old, diff)
end
println("worst: ", worst)
