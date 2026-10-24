####################################
#
#  Precompilation
#
####################################

# The estimators' first calls, compiled into the package image with the solvers
# of ProximalSolvers that they run. PrecompileTools runs this workload when the
# package is precompiled; the preference `precompile_workload = false` skips it.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
  p, n = 6, 120
  Σy = [0.5^abs(i - j) for i in 1:p, j in 1:p]
  X = [sin(1.0 * i * j) + cos(0.3 * (i + j)) for i in 1:n, j in 1:p]
  Y = X * cholesky(Σy).U
  Sx, Sy = Symmetric(X'X / n), Symmetric(Y'Y / n)
  t = [i / n for i in 1:n]
  @compile_workload begin
    covsel!(zeros(p, p), zeros(p, p), zeros(p, p), Matrix(Sy), 0.1)
    covselpath(Matrix(Sy), [0.2, 0.1])
    fusedGraphicalLasso(Matrix(Sx), n, Matrix(Sy), n, 0.05, 0.05)
    tvCovSel(Y, t, [0.3, 0.7], 0.3, 0.1)
    clime(Matrix(Sy), 0.2)
    aclime(Matrix(Sy), n)
    slassoinv(Matrix(Sy), n)
    Δ = differencePrecisionActiveShooting(Sx, Sy, NormL1(0.1))
    differencePrecisionRefit(Sx, Sy, findall(!iszero, vec(Matrix(Δ))))
  end
end
