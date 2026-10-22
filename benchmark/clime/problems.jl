# Test problems for the CLIME benchmark: precision matrices, samples, penalty grids,
# and the files shared by the Julia and R runners.

using LinearAlgebra, Random, Statistics, Printf

# precision matrices of the simulations of Cai, Liu and Luo (2011) and Cai, Liu and
# Zhou (2016): AR(1) covariance (tridiagonal precision), a band precision matrix, and
# an Erdős–Rényi graph with about 4 neighbours per node
function precision_matrix(model::AbstractString, p::Integer; rng=Random.default_rng())
  if model == "ar1"
    ρ = 0.6
    Ω = Matrix(SymTridiagonal(fill(1 + ρ^2, p), fill(-ρ, p - 1))) ./ (1 - ρ^2)
    Ω[1, 1] = Ω[p, p] = 1 / (1 - ρ^2)
    return Ω
  elseif model == "band"
    Ω = Matrix(1.0I, p, p)
    for i in 1:p-1
      Ω[i, i+1] = Ω[i+1, i] = 0.6
    end
    for i in 1:p-2
      Ω[i, i+2] = Ω[i+2, i] = 0.3
    end
    return Ω
  elseif model == "er"
    B = zeros(p, p)
    prob = min(1.0, 4 / p)
    for c in 1:p, r in c+1:p
      if rand(rng) < prob
        B[r, c] = B[c, r] = 0.4 + 0.4 * rand(rng)
      end
    end
    return B + (abs(min(eigmin(Symmetric(B)), 0)) + 0.05) * I
  end
  throw(ArgumentError("unknown model $model"))
end

problem_name(model, p, n, seed) = "$(model)_p$(p)_n$(n)_s$(seed)"

# n observations from N(0, Ω⁻¹); returns the data
function sample(Ω::AbstractMatrix, n::Integer; rng=Random.default_rng())
  F = cholesky(Symmetric(Ω))
  randn(rng, n, size(Ω, 1)) / F.U'
end

# 30 penalties from 0.8 down to √(log p / n), equally spaced on the log scale
penalty_grid(p, n; length=30) = exp.(range(log(0.8), log(sqrt(log(p) / n)); length=length))

writebin(path, A::AbstractArray{Float64}) = open(io -> write(io, A), path, "w")
readbin(path, dims...) = read!(path, Array{Float64}(undef, dims...))

# Write the problem files: S.bin (the sample correlation matrix, p×p Float64,
# column-major), C.bin (the sample covariance matrix), Omega.bin (the truth),
# lambda.txt (one penalty per line, decreasing) and meta.txt.
function write_problem(dir, model, p, n, seed)
  rng = Xoshiro(seed)
  Ω = precision_matrix(model, p; rng=rng)
  X = sample(Ω, n; rng=rng)
  C = cov(X)
  S = cor(X)
  mkpath(dir)
  writebin(joinpath(dir, "S.bin"), S)
  writebin(joinpath(dir, "C.bin"), C)
  writebin(joinpath(dir, "Omega.bin"), Ω)
  open(joinpath(dir, "lambda.txt"), "w") do io
    foreach(λ -> println(io, λ), penalty_grid(p, n))
  end
  open(joinpath(dir, "meta.txt"), "w") do io
    println(io, "model=$model\np=$p\nn=$n\nseed=$seed")
  end
  dir
end

function read_meta(dir)
  d = Dict{String,String}()
  for line in eachline(joinpath(dir, "meta.txt"))
    k, v = split(line, "="; limit=2)
    d[k] = v
  end
  d
end

function read_problem(dir)
  meta = read_meta(dir)
  p, n = parse(Int, meta["p"]), parse(Int, meta["n"])
  (S=readbin(joinpath(dir, "S.bin"), p, p), C=readbin(joinpath(dir, "C.bin"), p, p),
   Ω=readbin(joinpath(dir, "Omega.bin"), p, p), λ=parse.(Float64, readlines(joinpath(dir, "lambda.txt"))),
   model=meta["model"], p=p, n=n, seed=parse(Int, meta["seed"]))
end
