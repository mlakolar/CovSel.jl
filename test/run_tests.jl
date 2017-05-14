using FactCheck

import CovSel
using Distributions
using ProximalBase: shrink

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
  "test_diff"
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
	include(f)
end

FactCheck.exitstatus()
