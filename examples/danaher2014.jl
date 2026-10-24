# Recreate the three-class simulation design in Figure 2 of Danaher, Wang and
# Witten (2014). Default: a small demonstration. --mode=pilot uses p=500,
# n=150; --mode=paper uses those dimensions and 100 replicates. See the README
# for the scope of the reproduction, plotting, and optional R/JGL comparison.
include(joinpath(@__DIR__, "danaher2014", "simulation.jl"))
Danaher2014.run(Danaher2014.configuration(ARGS))
