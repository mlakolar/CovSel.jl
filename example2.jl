reload("ProximalBase")
reload("CoordinateDescent")
reload("CovSel")

import CovSel

srand(123)

p = 5
Sigmax = eye(p,p)
Sigmay = zeros(p,p)
rho = 0.7
for i=1:p
    for j=1:p
        Sigmay[i,j]=rho^abs(i-j)
    end
end

sqmy = sqrtm(Sigmay)

n = 1000
X = randn(n,p)
Y = randn(n,p) * sqmy;

hSx = cov(X)
hSy = cov(Y);


A = kron(hSy, hSx)
b = reshape(hSy-hSx, p*p)

S = sprand(p, p, 0.1)
S = S + S' / 2
indS = find(S)
# full(S)
# for i=indS
#   @show ind2sub(hSx, i)
# end

x = CovSel.differencePrecisionRefit(Symmetric(hSx), Symmetric(hSy), indS)
@show full(x)

x1 = zeros(p, p)
x1[indS] = A[indS, indS] \ b[indS]

full(x) - x1
