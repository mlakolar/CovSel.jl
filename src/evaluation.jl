"""
    ConfusionMatrix(tp, fp, fn, tn)

Counts of true and false positives and negatives of an estimated support,
computed by [`getConfusionMatrix`](@ref). See also `tpr`, `fpr`, `precision`
and `recall`.
"""
struct ConfusionMatrix
  tp::Int
  fp::Int
  fn::Int
  tn::Int
end

"""
    getConfusionMatrix(Δ, hΔ, thr=1e-4)

Compare the support of the lower triangle (diagonal included) of the true
symmetric matrix `Δ` with that of the estimate `hΔ`, whose entries count as
nonzero when larger than `thr` in absolute value. `hΔ` is a matrix or a
`SparseIterate` vector indexed by the lower triangle.
"""
function getConfusionMatrix(Δ::AbstractMatrix{T}, hΔ::AbstractMatrix{T}, thr=1e-4) where T
  p = size(Δ, 1)

  tp = fp = fn = tn = 0
  for c=1:p
    for r=c:p
      if Δ[r, c] != zero(T)
        if abs(hΔ[r, c]) > thr
          # not zero estimated as not zero
          tp += 1
        else
          # not zero estimated as zero
          fn += 1
        end
      else
        if abs(hΔ[r, c]) > thr
          # zero estimated as not zero
          fp += 1
        else
          # zero estimated as zero
          tn += 1
        end
      end
    end
  end
  ConfusionMatrix(tp, fp, fn, tn)
end

function getConfusionMatrix(Δ::AbstractMatrix{T}, hΔ::SparseIterate{T, 1}, thr=1e-4) where T
  p = size(Δ, 1)

  tp = fp = fn = tn = 0
  for c=1:p
    for r=c:p
      if Δ[r, c] != zero(T)
        if abs(hΔ[sub2indLowerTriangular(p, r, c)]) > thr
          # not zero estimated as not zero
          tp += 1
        else
          # not zero estimated as zero
          fn += 1
        end
      else
        if abs(hΔ[sub2indLowerTriangular(p, r, c)]) > thr
          # zero estimated as not zero
          fp += 1
        else
          # zero estimated as zero
          tn += 1
        end
      end
    end
  end
  ConfusionMatrix(tp, fp, fn, tn)
end

function tpr(t::ConfusionMatrix)
  dn = t.tp + t.fn
  dn > 0 ? t.tp / dn : 0.
end
function fpr(t::ConfusionMatrix)
  dn = t.fp + t.tn
  dn > 0 ? t.fp / dn : 0.
end
function precision(t::ConfusionMatrix)
  dn = t.tp + t.fp
  dn > 0 ? t.tp / dn : 1.
end
recall(t::ConfusionMatrix) = tpr(t)
