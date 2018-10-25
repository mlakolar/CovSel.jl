struct ConfusionMatrix
  tp::Int
  fp::Int
  fn::Int
  tn::Int
end

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

function getConfusionMatrix(Δ::AbstractMatrix{T}, hΔ::SparseIterate{T}, thr=1e-4) where T
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
