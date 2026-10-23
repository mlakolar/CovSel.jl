# The column problems of CLIME and ACLIME as linear programs for any
# MathOptInterface solver (LPSolver).
module CovarianceSelectionMathOptInterfaceExt

using CovarianceSelection
using CovarianceSelection: LPSolver, _CLIMESpec, _ColumnPath, _foreach_column, _weights!, _pushsorted!, _pushempty!
import MathOptInterface as MOI

# A model for the column problems with γ = 0: with β = u - v,
#
#     minimize Σ (u + v)  subject to  δᵢⱼ - cⱼ μ ≤ Sⱼ (u - v) ≤ δᵢⱼ + cⱼ μ,  u, v ≥ 0,
#
# as p ranged constraints, whose bounds are the only part that changes with the
# column and the penalty, so the solver can warm start from its last basis.
struct _Model{O}
  opt::O
  u::Vector{MOI.VariableIndex}
  v::Vector{MOI.VariableIndex}
  rows::Vector{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.Interval{Float64}}}
  uval::Vector{Float64}
  vval::Vector{Float64}
  c::Vector{Float64}
  β::Vector{Float64}
  idx::Vector{Int}
end

function _optimizer(solver::LPSolver)
  opt = MOI.instantiate(solver.optimizer; with_cache_type=Float64, with_bridge_type=Float64)
  MOI.set(opt, MOI.Silent(), solver.silent)
  for (name, value) in solver.attributes
    MOI.set(opt, MOI.RawOptimizerAttribute(name), value)
  end
  opt
end

function _Model(solver::LPSolver, S::AbstractMatrix)
  p = size(S, 1)
  opt = _optimizer(solver)
  u = MOI.add_variables(opt, p)
  v = MOI.add_variables(opt, p)
  for x in (u, v), xi in x
    MOI.add_constraint(opt, xi, MOI.GreaterThan(0.0))
  end
  rows = map(1:p) do j
    terms = MOI.ScalarAffineTerm{Float64}[]
    for l in 1:p
      Sjl = Float64(S[j, l])
      iszero(Sjl) && continue
      push!(terms, MOI.ScalarAffineTerm(Sjl, u[l]), MOI.ScalarAffineTerm(-Sjl, v[l]))
    end
    MOI.add_constraint(opt, MOI.ScalarAffineFunction(terms, 0.0), MOI.Interval(-1.0, 1.0))
  end
  obj = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x) for x in vcat(u, v)], 0.0)
  MOI.set(opt, MOI.ObjectiveFunction{typeof(obj)}(), obj)
  MOI.set(opt, MOI.ObjectiveSense(), MOI.MIN_SENSE)
  _Model(opt, u, v, rows, zeros(p), zeros(p), zeros(p), zeros(p), collect(1:p))
end

# solve column i at μ; returns false if infeasible
function _solve!(m::_Model, i::Int, μ::Float64)
  for j in eachindex(m.rows)
    δ = j == i ? 1.0 : 0.0
    MOI.set(m.opt, MOI.ConstraintSet(), m.rows[j], MOI.Interval(δ - m.c[j] * μ, δ + m.c[j] * μ))
  end
  MOI.optimize!(m.opt)
  status = MOI.get(m.opt, MOI.TerminationStatus())
  status == MOI.OPTIMAL && return true
  status in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && return false
  error("LPSolver: the solver stopped with status $status (", MOI.get(m.opt, MOI.RawStatusString()),
    ") on column $i at the penalty $μ. Near the smallest feasible penalty of a singular S, the ",
    "linear programs are badly conditioned; the simplex solvers of CovarianceSelection handle them.")
end

function _getsolution!(m::_Model)
  MOI.get!(m.uval, m.opt, MOI.VariablePrimal(), m.u)
  MOI.get!(m.vval, m.opt, MOI.VariablePrimal(), m.v)
  m.β .= m.uval .- m.vval
end

function CovarianceSelection._solve_columns(solver::LPSolver, S::Matrix{T}, spec::_CLIMESpec{T},
                               μgrid::Vector{T}, colscale::Vector{T}, ntasks::Int) where {T}
  iszero(spec.γ) || throw(ArgumentError("LPSolver: γ must be zero"))
  p, nμ = size(S, 1), length(μgrid)
  cols = Vector{_ColumnPath{T}}(undef, p)
  μbad = fill(-one(T), p)
  _foreach_column(() -> _Model(solver, S), p, ntasks) do m, i
    m.c .= spec.maxweights ? max.(spec.w, spec.w[i]) : spec.w
    col = cols[i] = _ColumnPath{T}(nμ)
    μmax = 1 / m.c[i]
    for t in 1:nμ
      μ = Float64(μgrid[t] * colscale[i])
      if μ >= μmax
        _pushempty!(col)
      elseif _solve!(m, i, μ)
        _getsolution!(m)
        # entries that are zero up to the tolerance of the solver are dropped
        tol = 1e-9 * max(1.0, maximum(abs, m.β))
        k = 0
        for l in 1:p
          if abs(m.β[l]) > tol
            k += 1
            m.idx[k], m.β[k] = l, m.β[l]
          end
        end
        _pushsorted!(col, m.idx, T.(view(m.β, 1:k)), k)
      else
        μbad[i] = μgrid[t]
        for _ in t:nμ
          _pushempty!(col)
        end
        break
      end
    end
  end
  cols, μbad
end

# Step 1 of ACLIME: with t = βᵢ, the constraints
#     Sⱼ (u - v) - γ cⱼ t ≤ δᵢⱼ  and  Sⱼ (u - v) + γ cⱼ t ≥ δᵢⱼ,
# whose coefficients of t, and the row t - uᵢ + vᵢ = 0, change with the column.
function CovarianceSelection._solve_diagonal(solver::LPSolver, S::Matrix{T}, spec::_CLIMESpec{T}, ntasks::Int) where {T}
  p = size(S, 1)
  d = zeros(T, p)
  _foreach_column(() -> _Step1Model(solver, S), p, ntasks) do m, i
    d[i] = _solve_step1!(m, spec, i)
  end
  d
end

mutable struct _Step1Model{O}
  opt::O
  u::Vector{MOI.VariableIndex}
  v::Vector{MOI.VariableIndex}
  t::MOI.VariableIndex
  upper::Vector{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}}
  lower::Vector{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}}
  link::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}
  linked::Int
  c::Vector{Float64}
end

function _Step1Model(solver::LPSolver, S::AbstractMatrix)
  p = size(S, 1)
  opt = _optimizer(solver)
  u = MOI.add_variables(opt, p)
  v = MOI.add_variables(opt, p)
  for x in (u, v), xi in x
    MOI.add_constraint(opt, xi, MOI.GreaterThan(0.0))
  end
  t = MOI.add_variable(opt)
  function row(j)
    terms = [MOI.ScalarAffineTerm(0.0, t)]
    for l in 1:p
      Sjl = Float64(S[j, l])
      iszero(Sjl) && continue
      push!(terms, MOI.ScalarAffineTerm(Sjl, u[l]), MOI.ScalarAffineTerm(-Sjl, v[l]))
    end
    MOI.ScalarAffineFunction(terms, 0.0)
  end
  upper = [MOI.add_constraint(opt, row(j), MOI.LessThan(0.0)) for j in 1:p]
  lower = [MOI.add_constraint(opt, row(j), MOI.GreaterThan(0.0)) for j in 1:p]
  link = MOI.add_constraint(opt, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, t),
    MOI.ScalarAffineTerm(-1.0, u[1]), MOI.ScalarAffineTerm(1.0, v[1])], 0.0), MOI.EqualTo(0.0))
  obj = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x) for x in vcat(u, v)], 0.0)
  MOI.set(opt, MOI.ObjectiveFunction{typeof(obj)}(), obj)
  MOI.set(opt, MOI.ObjectiveSense(), MOI.MIN_SENSE)
  _Step1Model(opt, u, v, t, upper, lower, link, 1, zeros(p))
end

function _solve_step1!(m::_Step1Model, spec::_CLIMESpec{T}, i::Int) where {T}
  _weights!(m.c, _CLIMESpec(Float64.(spec.w), spec.maxweights, Float64(spec.γ)), i)
  γ = Float64(spec.γ)
  for j in eachindex(m.upper)
    δ = j == i ? 1.0 : 0.0
    MOI.modify(m.opt, m.upper[j], MOI.ScalarCoefficientChange(m.t, -γ * m.c[j]))
    MOI.modify(m.opt, m.lower[j], MOI.ScalarCoefficientChange(m.t, γ * m.c[j]))
    MOI.set(m.opt, MOI.ConstraintSet(), m.upper[j], MOI.LessThan(δ))
    MOI.set(m.opt, MOI.ConstraintSet(), m.lower[j], MOI.GreaterThan(δ))
  end
  if m.linked != i
    MOI.modify(m.opt, m.link, MOI.ScalarCoefficientChange(m.u[m.linked], 0.0))
    MOI.modify(m.opt, m.link, MOI.ScalarCoefficientChange(m.v[m.linked], 0.0))
    MOI.modify(m.opt, m.link, MOI.ScalarCoefficientChange(m.u[i], -1.0))
    MOI.modify(m.opt, m.link, MOI.ScalarCoefficientChange(m.v[i], 1.0))
    m.linked = i
  end
  MOI.optimize!(m.opt)
  status = MOI.get(m.opt, MOI.TerminationStatus())
  status == MOI.OPTIMAL || return T(NaN)
  T(MOI.get(m.opt, MOI.VariablePrimal(), m.t))
end

end
