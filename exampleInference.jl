using Revise
using PyPlot

using CovSel
using ROCKET

function generatePrecisionXia(p)
    D = diagm(0.5 + 2*rand(p))
    sqrtD = sqrt.(D)

    Ωstar = eye(p) + diagm(2*0.6*ones(p-1),1) + diagm(2*0.3*ones(p-2),2)
    Ωstar = (Ωstar + Ωstar')/2
    Ω = sqrtD*Ωstar*sqrtD

    return (Ω + Ω')/2
end


p = 500
n = 400
numRep = 500
T = zeros(numRep)

Ω = generatePrecisionXia(p)

for rep=1:numRep
  if mod(rep, 10) == 0
    @show rep
  end
  data = CovSel.generateData(Ω, 2*n)
  X = data[1:n, :]
  Y = data[n+1:end, :]

  methodType = 4
  covarianceType = 4
  ex, vx = teInference(X, 1, 2, methodType, covarianceType)
  ey, vy = teInference(Y, 1, 2, methodType, covarianceType)

  eθ = ex - ey
  stdθ = sqrt( vx + vy)
  T[rep] = eθ / stdθ
end


plt[:hist](T, 50)
plt[:axvline](x=0, linewidth=2, color="red")
