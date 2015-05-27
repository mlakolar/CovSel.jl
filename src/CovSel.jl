module CovSel

using ProximalOPT.shrink

export
  ADMMSolver,
  solve!



include("admmCovSel.jl")
include("utils.jl")

end
