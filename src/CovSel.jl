module CovSel

using ProximalOPT: shrink, ProxL1L2, prox!

export
  ADMMSolver,
  solve!

include("admmCovSel.jl")
include("utils.jl")

end
