# A naive reference solver for the ℓ1 penalized direct difference estimator,
# used by the tests and the benchmarks to check differencePrecisionActiveShooting.
# It shares no code with the package: full coordinate descent sweeps over a
# dense Δ, updating the whole of A = Σx⋅Δ⋅Σy after every coordinate.
module Reference

soft(v, t) = sign(v) * max(abs(v) - t, zero(v))

"""
    differencePrecisionNaive(Σx, Σy, λ, ω; maxIter=10_000, optTol=1e-7)

Minimize `tr(Σx⋅Δ⋅Σy⋅Δ)/2 - tr(Δ⋅(Σy - Σx)) + λ ∑_{a≤b} ω[a,b] |Δ[a,b]|` over
symmetric `Δ` by cyclic coordinate descent over the lower triangle, until no
coordinate changes by more than `optTol` in a sweep. Returns `Δ` as a dense
matrix.
"""
function differencePrecisionNaive(Σx, Σy, λ, ω; maxIter=10_000, optTol=1e-7)
  p = size(Σx, 1)

  Δ = zeros(p, p)
  A = zeros(p, p)

  iter = 1;
  while iter < maxIter
    fDone = true
    for a=1:p
      for b=a:p
        if a==b
          # diagonal elements
          x0 = Σx[a,a]*Σy[a,a] / 2.
          x1 = A[a,a] - Σy[a,a] + Σx[a,a]
          tmp = soft(-x1/x0/2. + Δ[a,b], λ*ω[a,b] / 2. / x0)
        else
          # off-diagonal elements
          x0 = (Σx[a,a]*Σy[b,b] + Σx[b,b]*Σy[a,a])/2. + Σx[a,b]*Σy[a,b]
          x1 = A[a,b] + A[b,a] - 2. * (Σy[a,b] - Σx[a,b])
          tmp = soft(-x1/(2. * x0) + Δ[a,b], λ*ω[a,b] / x0)
        end

        h = tmp - Δ[a,b]
        Δ[a,b] = tmp
        Δ[b,a] = tmp
        if abs(h) > optTol
          fDone = false
        end
        for j=1:p
          for k=1:p
            if a == b
              A[j,k] = A[j,k] + h * Σx[j,a]*Σy[a,k]
            else
              A[j,k] = A[j,k] + h * (Σx[j,a]*Σy[b,k] + Σx[j,b]*Σy[a,k])
            end
          end
        end
      end
    end

    iter = iter + 1;
    if fDone
      break
    end
  end
  Δ
end

end
