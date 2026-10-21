# CovSel.jl

[![CI](https://github.com/mlakolar/CovSel.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/CovSel.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/mlakolar/CovSel.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/CovSel.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Various methods for learning high-dimensional Gaussian graphical models.

* Graphical lasso by ADMM: `covsel!`, `covselpath`
* Fused graphical lasso for two groups (Danaher, Wang and Witten, 2014): `fusedGraphicalLasso`, `fusedGraphicalLasso!`
* Fused neighborhood selection: `fusedNeighborhoodSelection`
* Direct estimation of the difference of two precision matrices `Ωx - Ωy` (Zhao, Cai and Li, 2014)
  * ℓ1 penalized, by active shooting coordinate descent: `differencePrecisionActiveShooting`, `differencePrecisionRefit`
  * sparsity constrained, by iterative hard thresholding: `differencePrecisionIHT`
  * sparse plus low rank: `differenceLatentPrecisionIHT`

```julia
using CovSel, ProximalBase, LinearAlgebra, Statistics

p, n = 20, 1000
Ωy = Matrix(1.0I, p, p)
Ωy[1, 2] = Ωy[2, 1] = 0.4
X = randn(n, p)
Y = CovSel.generateData(Ωy, n)
Sx, Sy = cov(X), cov(Y)

# graphical lasso: minimize tr(SΩ) - log det Ω + λ‖Ω‖₁
Ω = covselpath(Sy, [0.2, 0.1])[end]

# direct difference estimation:
# minimize tr(Sx⋅Δ⋅Sy⋅Δ)/2 - tr(Δ⋅(Sy - Sx)) + λ‖Δ‖₁
Δ = differencePrecisionActiveShooting(Symmetric(Sx), Symmetric(Sy), ProxL1(0.1))
Matrix(Δ)
```

See [`examples/difference_estimation.jl`](examples/difference_estimation.jl) for more.

Package depends on [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl) and [CoordinateDescent.jl](https://github.com/mlakolar/CoordinateDescent.jl).
