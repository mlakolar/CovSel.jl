using Pkg
Pkg.activate(joinpath(@__DIR__,"../.."))
Pkg.add([PackageSpec(url="https://github.com/mlakolar/ProximalBase.jl",rev="master"),
         PackageSpec(url="https://github.com/mlakolar/ProximalSolvers.jl",rev="master")])
Pkg.instantiate()
