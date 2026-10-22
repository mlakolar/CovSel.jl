"""
    IHTOptions(; epsTol=1e-2, maxIter=500, checkEvery=10)

Options for the iterative hard thresholding solvers. The iterations stop when
the largest gradient entry falls below `epsTol`, or when the objective, checked
every `checkEvery` iterations, changes by at most `epsTol`.
"""
struct IHTOptions
 epsTol::Float64
 maxIter::Int64
 checkEvery::Int64
end


IHTOptions(;epsTol::Real=1e-2, maxIter::Integer=500, checkEvery::Integer=10) =
  IHTOptions(epsTol, maxIter, checkEvery)



  # gradient of tr(Σ(S + L)) - log det(S + L) with respect to S
  function covselLatentIHT_grad_S!(gS, Σ, S, L)
    gS .= Symmetric(Σ) - inv(Symmetric(S + L))
    gS
  end

  # gradient with respect to U, where L = UU'
  function covselLatentIHT_grad_U!(gU, gS, Σ, S, L, U)
    covselLatentIHT_grad_S!(gS, Σ, S, L)
    rmul!(gS, 2.)
    mul!(gU, gS, U)
  end

  # minimize tr(Σ(S + L)) - log det(S + L) over S with at most s nonzero
  # off-diagonal entries (lower triangle) and L = UU' of rank r, by gradient
  # descent with step sizes ηS and ηU and hard thresholding of S
  function covselLatentIHT!(Σ::StridedMatrix, ηS, ηU, s, r;
    epsTol=1e-2, maxIter=100, checkEvery=5, callback=nothing)

    p = size(Σ, 1)

    gS = zeros(p, p)
    gU = zeros(p, r)

    S = zeros(p, p)

    # init
    tmp = Matrix(inv(Symmetric(Σ)))
    HardThreshold!(S, tmp, s, false)

    tmp .= tmp - S
    F = svd(tmp)
    U = F.U[:,1:r] .* sqrt.(F.S[1:r])'
    L = U * U'


    fvals = Float64[]
    # gradient descent
    #
    fv = covSel_objective(Σ, S, L)
    push!(fvals, fv)
    iter = 0
    while iter < maxIter
      iter += 1
      # update S
      covselLatentIHT_grad_S!(gS, Σ, S, L)
      @. S -= ηS * gS
      HardThreshold!(S, S, s, false)
      max_gS = maximum( abs.(gS) )

      # update U and L
      covselLatentIHT_grad_U!(gU, gS, Σ, S, L, U)
      @. U -= ηU * gU
      mul!(L, U, U')
      max_gU = maximum( abs.(gU) )
      if callback !== nothing
        callback(Σ, S, L)
      end

      done = max(max_gS, max_gU) < epsTol
      # check for convergence
      if mod(iter, checkEvery) == 0
        fv_new = covSel_objective(Σ, S, L)
        push!(fvals, fv_new)
        done = abs(fv_new - fv) <= epsTol
        fv = fv_new
      end
      done && break
    end

    (S, L, U, fvals, iter)
  end
