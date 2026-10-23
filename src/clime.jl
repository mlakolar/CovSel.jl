# CLIME (Cai, Liu and Luo, 2011) and ACLIME (Cai, Liu and Zhou, 2016): sparse
# precision matrices by constrained ℓ1 minimization, one column at a time.

"""
    CLIMESolver

Supertype of the solvers accepted by [`clime`](@ref) and [`aclime`](@ref):
[`ParametricSimplex`](@ref) (the default), [`DualSimplex`](@ref),
[`LPSolver`](@ref) and [`PDHG`](@ref).
"""
abstract type CLIMESolver end

"""
    DualSimplex(; tol=1e-9, refactor=100, maxpivots=0)

A dual simplex method specialized to the column problems, warm started along a
sequence of penalties: each column starts from zero, the solution for the
largest penalty, and the optimal basis for one penalty is the starting point
for the next. The solutions are exact (vertices of the linear program).

A pivot only touches the k × k block of `S` formed by the tight constraints and
the nonzero entries of the solution, whose inverse is updated in O(k²)
operations, so a pivot costs O(pk) operations, mostly two products of k columns
of `S` with a vector.

- `tol`: tolerance of the feasibility and ratio tests
- `refactor`: largest number of pivots between two factorizations of the basis
  block, which is also refactorized when its updates have drifted
- `maxpivots`: maximum number of pivots for one penalty; `0` means `1000 + 50p`

In the benchmarks of `benchmark/clime`, [`ParametricSimplex`](@ref), which
follows the solution path through all its breakpoints, was as fast for one
penalty and up to twice as fast for a path, so it is the default.
"""
Base.@kwdef struct DualSimplex <: CLIMESolver
  tol::Float64 = 1e-9
  refactor::Int = 100
  maxpivots::Int = 0
end

"""
    ParametricSimplex(; tol=1e-9, refactor=100, maxpivots=0)

The parametric simplex method (Pang, Liu and Vanderbei, 2014): for each column
it follows the piecewise linear solution path of the linear program from the
largest penalty, where the solution is zero, down to the smallest penalty asked
for, with one pivot per breakpoint of the path. The solutions are exact, and
penalties between the largest and the smallest one come for free. It uses the
same pivots as [`DualSimplex`](@ref), which can skip breakpoints, and is faster
when the penalties are close together.

- `tol`: tolerance of the ratio tests
- `refactor`: largest number of pivots between two factorizations of the basis
  block, which is also refactorized when its updates have drifted
- `maxpivots`: maximum number of pivots for one column; `0` means `1000 + 50p`
"""
Base.@kwdef struct ParametricSimplex <: CLIMESolver
  tol::Float64 = 1e-9
  refactor::Int = 100
  maxpivots::Int = 0
end

const _Simplex = Union{DualSimplex,ParametricSimplex}

_maxpivots(solver::_Simplex, p) = solver.maxpivots > 0 ? solver.maxpivots : 1000 + 50 * p

# the tolerance in the precision T: at least 100 eps(T)
_tol(solver::_Simplex, ::Type{T}) where {T} = max(T(solver.tol), 100 * eps(T))

"""
    LPSolver(optimizer; silent=true, attributes=())

Solve each column problem as a linear program with a
[MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) solver,
for example `LPSolver(HiGHS.Optimizer)`. Requires loading MathOptInterface,
which any such solver does (`using HiGHS`). One model is built for each task
and reused for all columns and penalties, so the solver can warm start from the
previous basis. `attributes` are passed to the solver as
`MOI.RawOptimizerAttribute`s, for example `("presolve" => "off",)` for HiGHS.
"""
struct LPSolver{O} <: CLIMESolver
  optimizer::O
  silent::Bool
  attributes::Vector{Pair{String,Any}}
end

LPSolver(optimizer; silent::Bool=true, attributes=()) =
  LPSolver(optimizer, silent, Pair{String,Any}[String(k) => v for (k, v) in attributes])


# Column i of the problem: minimize ‖β‖₁ subject to
# |(Sβ - eᵢ)ⱼ| ≤ cⱼ (μ + γ βᵢ), with c = w, or c = max.(w, wᵢ) when `maxweights`.
struct _CLIMESpec{T<:AbstractFloat}
  w::Vector{T}
  maxweights::Bool
  γ::T
end

function _weights!(c::Vector{T}, spec::_CLIMESpec{T}, i::Int) where {T}
  if spec.maxweights
    wi = spec.w[i]
    @inbounds for j in eachindex(c)
      c[j] = max(spec.w[j], wi)
    end
  else
    copyto!(c, spec.w)
  end
  c
end

# The solution path of one column: for penalty t, the nonzero entries
# rows[ptr[t]:ptr[t+1]-1], sorted, with values vals[...].
struct _ColumnPath{T}
  rows::Vector{Int}
  vals::Vector{T}
  ptr::Vector{Int}
end

_ColumnPath{T}(nμ::Integer) where {T} = (ptr = sizehint!(Int[1], nμ + 1); _ColumnPath{T}(Int[], T[], ptr))

function _pushsorted!(col::_ColumnPath{T}, rows, vals, k::Int) where {T}
  n0 = length(col.rows)
  @inbounds for q in 1:k
    # insertion sort while appending: k is the number of nonzero entries
    r, v = rows[q], vals[q]
    if !iszero(v)
      push!(col.rows, r)
      push!(col.vals, v)
      m = length(col.rows)
      while m > n0 + 1 && col.rows[m-1] > r
        col.rows[m], col.vals[m] = col.rows[m-1], col.vals[m-1]
        m -= 1
      end
      col.rows[m], col.vals[m] = r, v
    end
  end
  push!(col.ptr, length(col.rows) + 1)
  col
end

_pushempty!(col::_ColumnPath) = (push!(col.ptr, length(col.rows) + 1); col)

# entry r of column j of the estimate for penalty t, from the column paths
@inline function _pathentry(cols::Vector{_ColumnPath{T}}, t::Int, r::Int, j::Int) where {T}
  c = cols[j]
  lo, hi = c.ptr[t], c.ptr[t+1] - 1
  @inbounds while lo <= hi
    mid = (lo + hi) >>> 1
    rm = c.rows[mid]
    rm == r && return c.vals[mid]
    rm < r ? (lo = mid + 1) : (hi = mid - 1)
  end
  zero(T)
end

# the symmetrized estimate for penalty t, from the column paths: entry (r, j)
# is the one of raw[r, j] and raw[j, r] with the smaller absolute value
function _assemble_symmetric(cols::Vector{_ColumnPath{T}}, t::Int) where {T}
  p = length(cols)
  colptr = Vector{Int}(undef, p + 1)
  colptr[1] = 1
  @inbounds for j in 1:p
    c = cols[j]
    m = 0
    for q in c.ptr[t]:c.ptr[t+1]-1
      m += !iszero(_pathentry(cols, t, j, c.rows[q]))
    end
    colptr[j+1] = colptr[j] + m
  end
  rowval = Vector{Int}(undef, colptr[end] - 1)
  nzval = Vector{T}(undef, colptr[end] - 1)
  @inbounds for j in 1:p
    c = cols[j]
    k = colptr[j]
    for q in c.ptr[t]:c.ptr[t+1]-1
      r, x = c.rows[q], c.vals[q]
      y = _pathentry(cols, t, j, r)
      iszero(y) && continue
      rowval[k], nzval[k] = r, abs(x) <= abs(y) ? x : y
      k += 1
    end
  end
  SparseMatrixCSC(p, p, colptr, rowval, nzval)
end

# the estimate for penalty t, from the column paths
function _assemble(cols::Vector{_ColumnPath{T}}, t::Int) where {T}
  p = length(cols)
  colptr = Vector{Int}(undef, p + 1)
  colptr[1] = 1
  @inbounds for i in 1:p
    c = cols[i]
    colptr[i+1] = colptr[i] + c.ptr[t+1] - c.ptr[t]
  end
  nz = colptr[end] - 1
  rowval = Vector{Int}(undef, nz)
  nzval = Vector{T}(undef, nz)
  @inbounds for i in 1:p
    c = cols[i]
    lo, hi = c.ptr[t], c.ptr[t+1] - 1
    copyto!(rowval, colptr[i], c.rows, lo, hi - lo + 1)
    copyto!(nzval, colptr[i], c.vals, lo, hi - lo + 1)
  end
  SparseMatrixCSC(p, p, colptr, rowval, nzval)
end

"""
    CovSel.symmetrize_minabs(Ω)

Symmetrize the column estimates of CLIME: entry `(i, j)` of the result is
whichever of `Ω[i, j]` and `Ω[j, i]` has the smaller absolute value, so an entry
is nonzero only if both are.
"""
function symmetrize_minabs(R::SparseMatrixCSC{T}) where {T}
  p = LinearAlgebra.checksquare(R)
  Rt = copy(transpose(R))
  colptr = Vector{Int}(undef, p + 1)
  colptr[1] = 1
  rowval = Int[]
  nzval = T[]
  sizehint!(rowval, nnz(R))
  sizehint!(nzval, nnz(R))
  @inbounds for j in 1:p
    a, ae = R.colptr[j], R.colptr[j+1] - 1
    b, be = Rt.colptr[j], Rt.colptr[j+1] - 1
    while a <= ae && b <= be
      ra, rb = R.rowval[a], Rt.rowval[b]
      if ra == rb
        x, y = R.nzval[a], Rt.nzval[b]
        push!(rowval, ra)
        push!(nzval, abs(x) <= abs(y) ? x : y)
        a += 1
        b += 1
      elseif ra < rb
        a += 1
      else
        b += 1
      end
    end
    colptr[j+1] = length(rowval) + 1
  end
  SparseMatrixCSC(p, p, colptr, rowval, nzval)
end

symmetrize_minabs(R::AbstractMatrix) = symmetrize_minabs(sparse(R))


####################################
#
# parametric simplex driver
#
####################################

# The column drivers solve column i for the decreasing penalties μs and record
# the solutions in col. They return the largest of μs at which the problem is
# infeasible, or -1 if there is none; the solutions for the infeasible penalties
# are recorded as empty.

function _column_path!(ws::_PSMWorkspace{T}, i::Int, μs::AbstractVector{T}, col::_ColumnPath{T},
                       solver::ParametricSimplex) where {T}
  _reset!(ws, i)
  tol = _tol(solver, T)
  maxpivots = _maxpivots(solver, ws.p)
  nμ = length(μs)
  μ = one(T) / ws.c[i]
  t = 1
  while t <= nμ && μs[t] >= μ
    _pushempty!(col)
    t += 1
  end
  pivots = stall = 0
  while t <= nμ
    # Bland's rule, which cannot cycle, after many pivots without progress
    bland = stall > max(50, 2 * ws.k)
    μn, kind, idx, s = _next_event(ws, μ, tol, bland)
    while t <= nμ && μs[t] >= μn
      _record!(col, ws, μs[t])
      t += 1
    end
    t > nμ && break
    stall = μn < μ ? 0 : stall + 1
    μ = μn
    if !_pivot!(ws, kind, idx, s, tol, bland)
      μbad = μs[t]
      while t <= nμ
        _pushempty!(col)
        t += 1
      end
      return μbad
    end
    pivots += 1
    pivots > maxpivots && error("ParametricSimplex: column $i needed more than $maxpivots pivots")
    _primal!(ws)
    _refactor_if_needed!(ws, solver.refactor, tol)
  end
  -one(T)
end

function _column_path!(ws::_PSMWorkspace{T}, i::Int, μs::AbstractVector{T}, col::_ColumnPath{T},
                       solver::DualSimplex) where {T}
  _reset!(ws, i)
  tol = _tol(solver, T)
  maxpivots = _maxpivots(solver, ws.p)
  μmax = one(T) / ws.c[i]
  nμ = length(μs)
  for t in 1:nμ
    μ = μs[t]
    if μ >= μmax
      _pushempty!(col)
    elseif _dual_simplex!(ws, μ, tol, solver.refactor, maxpivots) >= 0
      _record!(col, ws, μ)
    else
      for _ in t:nμ
        _pushempty!(col)
      end
      return μ
    end
  end
  -one(T)
end

# record the solution at μ, with entries of the wrong sign (within the
# tolerance of the dual simplex) set to zero
function _record!(col::_ColumnPath{T}, ws::_PSMWorkspace{T}, μ::T) where {T}
  k = ws.k
  @inbounds for q in 1:k
    v = ws.a[q] + μ * ws.b[q]
    ws.x[q] = ws.τ[q] * v > 0 ? v : zero(T)
  end
  _pushsorted!(col, ws.A, ws.x, k)
end

# βᵢ at μ = 0 (step 1 of ACLIME), or NaN if (P₀) is infeasible
function _column_diag!(ws::_PSMWorkspace{T}, i::Int, solver::_Simplex) where {T}
  col = _ColumnPath{T}(1)
  _column_path!(ws, i, T[0], col, solver) < 0 || return T(NaN)
  lo, hi = col.ptr[1], col.ptr[2] - 1
  q = searchsortedfirst(view(col.rows, lo:hi), i)
  q <= hi - lo + 1 && col.rows[lo+q-1] == i ? col.vals[lo+q-1] : zero(T)
end

# run f(state, i) for i = 1:p on ntasks tasks, each with its own state = init(),
# taking the columns one at a time
function _foreach_column(f, init, p::Int, ntasks::Int)
  next = Threads.Atomic{Int}(1)
  ntasks = clamp(ntasks, 1, p)
  ntasks == 1 && return _column_worker(f, init, next, p)
  tasks = [Threads.@spawn(_column_worker(f, init, next, p)) for _ in 1:ntasks]
  for t in tasks
    try
      wait(t)
    catch e
      e isa TaskFailedException ? throw(t.result) : rethrow()
    end
  end
  nothing
end

function _column_worker(f, init, next::Threads.Atomic{Int}, p::Int)
  state = init()
  while (i = Threads.atomic_add!(next, 1)) <= p
    f(state, i)
  end
  nothing
end

# Solve column i for the penalties μgrid .* colscale[i], for all i. Returns the
# column paths and, for each column, the largest penalty of μgrid at which it is
# infeasible (-1 if none).
function _solve_columns(solver::_Simplex, S::Matrix{T}, spec::_CLIMESpec{T},
                        μgrid::Vector{T}, colscale::Vector{T}, ntasks::Int) where {T}
  p, nμ = size(S, 1), length(μgrid)
  cols = Vector{_ColumnPath{T}}(undef, p)
  μbad = fill(-one(T), p)
  _foreach_column(() -> (_PSMWorkspace(S), similar(μgrid)), p, ntasks) do (ws, μs), i
    _weights!(ws.c, spec, i)
    ws.γ = spec.γ
    μs .= μgrid .* colscale[i]
    cols[i] = _ColumnPath{T}(nμ)
    bad = _column_path!(ws, i, μs, cols[i], solver)
    μbad[i] = bad < 0 ? bad : bad / colscale[i]
  end
  cols, μbad
end

function _solve_diagonal(solver::_Simplex, S::Matrix{T}, spec::_CLIMESpec{T}, ntasks::Int) where {T}
  p = size(S, 1)
  d = zeros(T, p)
  _foreach_column(() -> _PSMWorkspace(S), p, ntasks) do ws, i
    _weights!(ws.c, spec, i)
    ws.γ = spec.γ
    d[i] = _column_diag!(ws, i, solver)
  end
  d
end

_needmoi() = error("LPSolver needs MathOptInterface: load it, or a solver that uses it, e.g. `using HiGHS`")
_solve_columns(::LPSolver, args...) = _needmoi()
_solve_diagonal(::LPSolver, args...) = _needmoi()


####################################
#
# estimators
#
####################################

"""
    CLIMEPath{T} <: AbstractVector{SparseMatrixCSC{T,Int}}

Estimates along a sequence of penalties, returned by [`clime`](@ref) and
[`aclime`](@ref) when given several of them. `path[k]` is the estimate for the
penalty `path.λ[k]` (the constant δ for ACLIME), in the order in which the
penalties were given.
"""
struct CLIMEPath{T<:AbstractFloat} <: AbstractVector{SparseMatrixCSC{T,Int}}
  λ::Vector{T}
  Ω::Vector{SparseMatrixCSC{T,Int}}
end

Base.size(path::CLIMEPath) = size(path.Ω)
Base.getindex(path::CLIMEPath, k::Int) = path.Ω[k]
Base.IndexStyle(::Type{<:CLIMEPath}) = IndexLinear()

function Base.show(io::IO, path::CLIMEPath)
  p = isempty(path) ? 0 : size(path.Ω[1], 1)
  print(io, "CLIMEPath(", length(path), " estimates, p = ", p, ")")
end

function Base.show(io::IO, ::MIME"text/plain", path::CLIMEPath)
  n = length(path)
  p = n > 0 ? size(path.Ω[1], 1) : 0
  print(io, "CLIMEPath with $n estimate", n == 1 ? "" : "s", " of a $p×$p precision matrix")
  n == 0 && return
  println(io, ":")
  print(io, "         λ   nonzero off-diagonal pairs")
  for k in 1:n
    Ω = path.Ω[k]
    print(io, "\n  ", lpad(string(round(path.λ[k]; sigdigits=4)), 8), "   ",
      (nnz(Ω) - count(!iszero, diag(Ω))) ÷ 2)
  end
end

# S as a dense symmetric Matrix of floats; S itself when it already is one
function _covariance(S::AbstractMatrix)
  p = LinearAlgebra.checksquare(S)
  T = float(eltype(S))
  T <: AbstractFloat || throw(ArgumentError("S must be a real matrix"))
  M = if S isa Matrix{T} && issymmetric(S)
    S
  elseif S isa Symmetric{T,Matrix{T}} && issymmetric(parent(S))
    parent(S)
  else
    Matrix{T}(S)
  end
  if !(S isa Union{Symmetric,Hermitian}) && !issymmetric(M)
    atol = sqrt(eps(T)) * maximum(abs, M; init=zero(T))
    all(abs(M[r, c] - M[c, r]) <= atol for c in 1:p for r in c+1:p) ||
      throw(ArgumentError("S must be symmetric; wrap it as Symmetric(S) to use one of its triangles"))
    @inbounds for c in 1:p, r in c+1:p
      M[r, c] = M[c, r] = (M[r, c] + M[c, r]) / 2
    end
  end
  all(isfinite, M) || throw(ArgumentError("S has entries that are not finite"))
  for i in 1:p
    M[i, i] > 0 || throw(ArgumentError("the diagonal of S must be positive; S[$i, $i] = $(M[i, i])"))
  end
  M
end

function _checkpenalties(λ, name)
  isempty(λ) && throw(ArgumentError("no $name given"))
  all(x -> isfinite(x) && x >= 0, λ) || throw(ArgumentError("$name must be finite and nonnegative"))
end

# Solve, symmetrize and drop the infeasible penalties. λ are the penalties as given,
# μgrid = λ sorted in decreasing order.
function _estimate(solver::CLIMESolver, S::Matrix{T}, spec::_CLIMESpec{T}, λ::AbstractVector,
                   colscale::Vector{T}, symmetrize::Bool, ntasks::Int, scalar::Bool) where {T}
  order = sortperm(λ; rev=true)
  μgrid = T.(λ[order])
  cols, μbad = _solve_columns(solver, S, spec, μgrid, colscale, ntasks)
  worst = maximum(μbad)
  nfeas = count(μ -> μ > worst, μgrid)
  if nfeas < length(μgrid)
    msg = "the column problems are infeasible for the penalty $(round(worst; sigdigits=4))" *
          (nfeas < length(μgrid) - 1 ? " and smaller ones" : "") * ", as S is (nearly) singular"
    scalar && throw(DomainError(only(λ), msg))
    @warn msg * "; dropping the $(length(μgrid) - nfeas) smallest penalties"
  end
  Ω = Vector{SparseMatrixCSC{T,Int}}(undef, length(λ))
  keep = falses(length(λ))
  for t in 1:nfeas
    Ω[order[t]] = symmetrize ? _assemble_symmetric(cols, t) : _assemble(cols, t)
    keep[order[t]] = true
  end
  CLIMEPath{T}(T.(λ[keep]), Ω[keep])
end

"""
    clime(S, λ, solver=ParametricSimplex(); symmetrize=true, ntasks=Threads.nthreads())

CLIME estimate of the precision matrix `Ω = Σ⁻¹` from the covariance matrix `S`
(Cai, Liu and Luo, 2011). Column `i` of the estimate solves

    minimize ‖β‖₁  subject to  ‖S β - eᵢ‖∞ ≤ λ,

and the columns are symmetrized by keeping, for each pair of entries `(i, j)` and
`(j, i)`, the one with the smaller absolute value. Returns a `SparseMatrixCSC`.

With a vector of penalties `λ`, returns a [`CLIMEPath`](@ref) of the estimates
for each of them. With the default [`ParametricSimplex`](@ref) solver, a whole
path costs about as much as its smallest penalty alone.

The column problems are independent: they are distributed over `ntasks` tasks
(start Julia with several threads, `julia -t auto`, to solve them in parallel).
With `symmetrize=false` the column solutions are returned as they are.

The penalty is in the units of `S Ω`, so it is not invariant to rescaling the
variables. To treat all variables alike, use the correlation matrix
`R = D⁻¹ S D⁻¹`, with `D` the diagonal matrix of standard deviations: if `Θ̂` is
the estimate from `R`, then `D⁻¹ Θ̂ D⁻¹` estimates `Ω`. When `S` is singular, as
the sample covariance matrix is for `n ≤ p`, the column problems are infeasible
for small `λ`: an infeasible penalty throws a `DomainError`, and is dropped from
a path with a warning.

The solver can be [`ParametricSimplex`](@ref) (the default), [`DualSimplex`](@ref),
any linear programming solver of MathOptInterface ([`LPSolver`](@ref)), or the
first order method [`PDHG`](@ref). All but `PDHG` give the exact solutions of
the linear programs. See also [`aclime`](@ref) and [`climecv`](@ref).

# Examples
```julia
S = cov(X)
Ω = clime(S, 0.1)
path = clime(S, range(0.4, 0.05; length=20))
path.λ, path[end]
```
"""
function clime(S::AbstractMatrix, λ::Union{Real,AbstractVector{<:Real}}, solver::CLIMESolver=ParametricSimplex();
               symmetrize::Bool=true, ntasks::Integer=Threads.nthreads())
  λs = λ isa Real ? [λ] : λ
  _checkpenalties(λs, "penalties λ")
  Σ = _covariance(S)
  T = eltype(Σ)
  p = size(Σ, 1)
  spec = _CLIMESpec(ones(T, p), false, zero(T))
  path = _estimate(solver, Σ, spec, λs, ones(T, p), symmetrize, ntasks, λ isa Real)
  λ isa Real ? only(path) : path
end

"""
    aclime(S, n, solver=ParametricSimplex(); δ=2, δ₁=2, truncate=false, symmetrize=true, ntasks=Threads.nthreads())

Adaptive CLIME (ACLIME) estimate of the precision matrix `Ω = Σ⁻¹` from the
unbiased sample covariance matrix `S` of `n` observations (Cai, Liu and Zhou,
2016). It is CLIME with the constraint of each entry scaled by estimated
standard deviations, which adapts the penalty to the scales of the variables
and leaves only a universal constant `δ`, for which the theory suggests 2. The
estimate is not invariant to rescaling single variables, though: the ℓ1 norm
and the scale `Sⱼⱼ ∨ Sᵢᵢ` of step 1 depend on their units. Passing the
correlation matrix `R = D^(-1/2) S D^(-1/2)`, with `D` the diagonal of `S`, and
scaling the estimate back, `D^(-1/2) Ω̂ D^(-1/2)`, gives an estimate that is.

With `Ŝ = S + I/n`, `λₙ = δ √(log p / n)` and `λ₁ = δ₁ √(log p / n)`:

1. the diagonal of `Ω` is estimated first: `ω̂ᵢᵢ` is entry `i` of the solution of

       minimize ‖b‖₁  subject to  |(Ŝ b - eᵢ)ⱼ| ≤ λ₁ (Sⱼⱼ ∨ Sᵢᵢ) bᵢ  for all j;

2. column `i` of the estimate solves

       minimize ‖b‖₁  subject to  |(Ŝ b - eᵢ)ⱼ| ≤ λₙ √(Sⱼⱼ ω̂ᵢᵢ)  for all j,

   and the columns are symmetrized as in [`clime`](@ref).

Returns a `SparseMatrixCSC`. With a vector `δ`, returns a [`CLIMEPath`](@ref)
of the estimates for each `δ`, whose field `λ` holds the values of `δ`. Cai,
Liu and Zhou (2016) choose `δ` from `(1:100)/50` by cross validation, which
[`aclimecv`](@ref) does. Since `Ŝ` is positive definite, all the column problems
are feasible.

`truncate=true` adds the rule of Cai, Liu and Zhou (2016) for large variances to
step 1: `ω̂ᵢᵢ` is replaced by `√(log p / n)` when `Sᵢᵢ > √(n / log p)`. The rule
is off by default because it depends on the units of the variables: when the
variances are large, whole columns of the estimate can be zero (`aclime` warns),
and in the simulation design of the paper it applies to most variables; see
`benchmark/clime` for its effect.
"""
function aclime(S::AbstractMatrix, n::Integer, solver::CLIMESolver=ParametricSimplex();
                δ=2, δ₁::Real=2, truncate::Bool=false, symmetrize::Bool=true,
                ntasks::Integer=Threads.nthreads())
  n > 0 || throw(ArgumentError("the number of observations n must be positive"))
  δ₁ >= 0 || throw(ArgumentError("δ₁ must be nonnegative"))
  δs = δ isa Real ? [δ] : collect(δ)
  _checkpenalties(δs, "constants δ")
  Σ = _covariance(S)
  T = eltype(Σ)
  p = size(Σ, 1)
  σ = diag(Σ)
  rate = T(sqrt(log(p) / n))
  Ŝ = Σ + I / T(n)

  # step 1: the diagonal
  ω = _solve_diagonal(solver, Ŝ, _CLIMESpec(σ, true, T(δ₁) * rate), ntasks)
  all(isfinite, ω) || error("ACLIME: the problems of step 1 are infeasible")
  if truncate
    for i in 1:p
      if σ[i] > 1 / rate
        ω[i] = rate
      end
    end
  end

  # step 2: CLIME with the constraint of entry j of column i scaled by √(σⱼⱼ ω̂ᵢᵢ);
  # Ŝ is positive definite, so all the penalties are feasible
  spec = _CLIMESpec(sqrt.(σ), false, zero(T))
  path = _estimate(solver, Ŝ, spec, T.(δs) .* rate, sqrt.(ω), symmetrize, ntasks, δ isa Real)
  for Ω in path.Ω
    truncate && any(i -> iszero(Ω[i, i]), 1:p) || continue
    @warn "ACLIME: some diagonal entries of the estimate are zero. With truncate=true, the variables " *
          "with variance above √(n / log p) = $(round(1 / rate; sigdigits=3)) get √(log p / n) as " *
          "their estimated diagonal in step 1; rescale the data, e.g. to unit variances, or use truncate=false."
    break
  end
  δ isa Real ? only(path.Ω) : CLIMEPath{T}(T.(δs), path.Ω)
end
