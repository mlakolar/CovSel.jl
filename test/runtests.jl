using FactCheck

import CovSel, ProximalBase
using Distributions

function try_import(name::Symbol)
    try
        @eval import $name
        return true
    catch e
        return false
    end
end

grb = try_import(:Gurobi)
jmp = try_import(:JuMP)
scs = try_import(:SCS)
cvx = try_import(:Convex)


tests = [
  # "test_covsel",
  "directDifference"
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
	include(f)
end

FactCheck.exitstatus()
