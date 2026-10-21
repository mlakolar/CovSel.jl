using CovSel
using Test
using Aqua

tests = [
  "utils",
  "covsel",
  "directDifference",
  "fusedDifference",
  "latent",
  "generate"
]

@testset "CovSel" begin
  for t in tests
    f = "$t.jl"
    println("* running $f ...")
    include(f)
  end

  @testset "Aqua" begin
    Aqua.test_all(CovSel)
  end
end
