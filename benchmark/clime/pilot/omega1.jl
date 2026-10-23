# The precision matrices Ω₁ of the design of Cai, Liu and Zhou (2016), as in
# ../aclime_table.jl.

function omega1(model, p, rng)
  if model == "band"
    Ω = Matrix(1.0I, p, p)
    for i in 1:p-1; Ω[i, i+1] = Ω[i+1, i] = 0.6; end
    for i in 1:p-2; Ω[i, i+2] = Ω[i+2, i] = 0.3; end
    return Ω
  elseif model == "ar1"
    return [0.6^abs(i - j) for i in 1:p, j in 1:p]
  elseif model == "er"
    Ω2 = zeros(p, p)
    for c in 1:p, r in c+1:p
      if rand(rng) < 0.05
        Ω2[r, c] = Ω2[c, r] = 0.4 + 0.4 * rand(rng)
      end
    end
    return Ω2 + (abs(eigmin(Symmetric(Ω2))) + 0.05) * I
  end
  throw(ArgumentError(model))
end
