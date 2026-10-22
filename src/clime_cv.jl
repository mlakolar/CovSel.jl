# Cross validation of the penalty of CLIME and of the constant δ of ACLIME.

"""
    CLIMECV{T}

Result of [`climecv`](@ref) and [`aclimecv`](@ref):

- `λ`: the penalties (the constants δ for ACLIME), in the order given
- `meanloss`, `stdloss`: the mean of the validation loss over the folds and
  its standard error
- `best`: the index of the smallest mean loss
- `folds`: the number of folds
- `path`: the [`CLIMEPath`](@ref) of the estimates on all the data

The selected estimate is `path[best]`, or `bestestimate(cv)`.
"""
struct CLIMECV{T<:AbstractFloat}
  λ::Vector{T}
  meanloss::Vector{T}
  stdloss::Vector{T}
  best::Int
  folds::Int
  path::CLIMEPath{T}
end

"""
    bestestimate(cv::CLIMECV)

The estimate on all the data for the penalty with the smallest mean
validation loss.
"""
bestestimate(cv::CLIMECV) = cv.path.Ω[findfirst(==(cv.λ[cv.best]), cv.path.λ)]

Base.show(io::IO, cv::CLIMECV) =
  print(io, "CLIMECV(", length(cv.λ), " penalties, best ", round(cv.λ[cv.best]; sigdigits=4), ")")

function Base.show(io::IO, ::MIME"text/plain", cv::CLIMECV)
  println(io, "CLIMECV: $(cv.folds)-fold cross validation of $(length(cv.λ)) penalties, best ",
    round(cv.λ[cv.best]; sigdigits=4))
  print(io, "         λ    mean loss    std error")
  for k in eachindex(cv.λ)
    print(io, "\n  ", k == cv.best ? "*" : " ", lpad(string(round(cv.λ[k]; sigdigits=4)), 8),
      lpad(string(round(cv.meanloss[k]; sigdigits=6)), 13), lpad(string(round(cv.stdloss[k]; sigdigits=3)), 13))
  end
end

# sample covariance matrix of the rows of X, centered, divided by n - 1
function _samplecov(X::AbstractMatrix)
  T = float(eltype(X))
  Xc = Matrix{T}(X)
  Xc .-= sum(Xc; dims=1) ./ size(Xc, 1)
  Symmetric(Xc' * Xc ./ (size(Xc, 1) - 1))
end

# validation losses of the estimate Ω for the covariance S of held out data:
# minus the Gaussian log likelihood up to constants, tr(SΩ) - log det Ω (Inf if
# Ω is not positive definite), or the trace loss of Cai, Liu and Luo (2011),
# Σᵢ ((SΩ)ᵢᵢ - 1)²
function _cvloss(S::AbstractMatrix, Ω::SparseMatrixCSC, loss::Symbol)
  rows, vals = rowvals(Ω), nonzeros(Ω)
  if loss == :likelihood
    F = cholesky(Symmetric(SparseMatrixCSC{Float64,Int}(Ω)); check=false)
    issuccess(F) || return Inf
    tr = 0.0
    for c in axes(Ω, 2), k in nzrange(Ω, c)
      tr += S[rows[k], c] * vals[k]
    end
    return tr - logdet(F)
  elseif loss == :trace
    l = 0.0
    for i in axes(Ω, 2)
      d = -1.0
      for k in nzrange(Ω, i)
        d += S[i, rows[k]] * vals[k]
      end
      l += abs2(d)
    end
    return l
  end
  throw(ArgumentError("loss must be :likelihood or :trace"))
end

function _folds(n::Integer, k::Integer, rng::AbstractRNG)
  2 <= k <= n || throw(ArgumentError("the number of folds must be between 2 and n = $n"))
  perm = randperm(rng, n)
  [perm[f:k:n] for f in 1:k]
end

function _crossvalidate(fit, X::AbstractMatrix, λs::AbstractVector, folds::Integer, loss::Symbol, rng::AbstractRNG)
  n = size(X, 1)
  _checkpenalties(λs, "penalties")
  loss in (:likelihood, :trace) || throw(ArgumentError("loss must be :likelihood or :trace"))
  T = float(eltype(X))
  L = fill(T(Inf), folds, length(λs))
  for (f, test) in enumerate(_folds(n, folds, rng))
    train = setdiff(1:n, test)
    path = fit(X[train, :], λs)
    Stest = _samplecov(X[test, :])
    for (k, λ) in enumerate(path.λ)
      L[f, findfirst(==(λ), λs)] = _cvloss(Stest, path.Ω[k], loss)
    end
  end
  meanloss = vec(sum(L; dims=1)) ./ folds
  stdloss = map(1:length(λs)) do k
    isfinite(meanloss[k]) || return T(Inf)
    sqrt(sum(abs2, L[:, k] .- meanloss[k]) / (folds - 1) / folds)
  end
  best = argmin(meanloss)
  CLIMECV{T}(T.(λs), meanloss, stdloss, best, folds, fit(X, λs))
end

"""
    climecv(X, λ, solver=ParametricSimplex(); folds=5, loss=:likelihood, rng=Random.default_rng(), kwargs...)

Choose the penalty of [`clime`](@ref) by `folds`-fold cross validation on the
`n × p` data matrix `X`, among the penalties `λ`. The estimates from the
sample covariance matrices of the training folds are scored on the sample
covariance matrix `S` of the held out fold, by

- `loss = :likelihood`: `tr(SΩ) - log det Ω`, minus the Gaussian log likelihood
  up to constants, infinite for an estimate that is not positive definite;
- `loss = :trace`: `Σᵢ ((SΩ)ᵢᵢ - 1)²`, the loss of Cai, Liu and Luo (2011).

Other keyword arguments are passed to [`clime`](@ref). Returns a
[`CLIMECV`](@ref); [`bestestimate`](@ref) gives the estimate on all the data
for the best penalty.

# Examples
```julia
cv = climecv(X, exp.(range(log(0.5), log(0.05); length=20)))
Ω = bestestimate(cv)
```
"""
climecv(X::AbstractMatrix, λs::AbstractVector{<:Real}, solver::CLIMESolver=ParametricSimplex();
        folds::Integer=5, loss::Symbol=:likelihood, rng::AbstractRNG=default_rng(), kwargs...) =
  _crossvalidate((Xt, λ) -> clime(_samplecov(Xt), λ, solver; kwargs...), X, λs, folds, loss, rng)

"""
    aclimecv(X, δ, solver=ParametricSimplex(); folds=5, loss=:likelihood, rng=Random.default_rng(), kwargs...)

Choose the constant `δ` of step 2 of [`aclime`](@ref) by cross validation on the
`n × p` data matrix `X`, as [`climecv`](@ref) does for the penalty of CLIME.
Cai, Liu and Zhou (2016) take `δ = (1:100)/50` and the likelihood loss. Other
keyword arguments are passed to [`aclime`](@ref); the field `λ` of the result
holds the values of `δ`.
"""
aclimecv(X::AbstractMatrix, δs::AbstractVector{<:Real}, solver::CLIMESolver=ParametricSimplex();
         folds::Integer=5, loss::Symbol=:likelihood, rng::AbstractRNG=default_rng(), kwargs...) =
  _crossvalidate((Xt, δ) -> aclime(_samplecov(Xt), size(Xt, 1), solver; δ=δ, kwargs...), X, δs, folds, loss, rng)
