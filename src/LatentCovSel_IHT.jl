
struct IHTOptions
 epsTol::Float64
 maxIter::Int64
 checkEvery::Int64
end


IHTOptions(;epsTol::Float64=1e-2, maxIter::Int64=500, checkEvery::Int64=10) =
  IHTOptions(epsTol, maxIter, checkEvery)



  function covselLatentIHT_grad_S!(gS, Σ, S, L)
    T = S + L
    T = inv(T)
    gS .= Σ - T
    gS
  end

  function covselLatentIHT_grad_U!(gU, gS, Σ, S, L, U)
    covselLatentIHT_grad_S!(gS, Σ, S, L)
    scale!(gS, 2.)
    A_mul_B!(gU, gS, U)
  end

  function covselLatentIHT!(Σ::StridedMatrix, ηS, ηU, s, r;
    epsTol=1e-2, maxIter=100, checkEvery=5, callback=nothing)

    p = size(Σ, 1)

    gS = zeros(p, p)
    gU = zeros(p, r)

    S = zeros(p, p)
    L = zeros(p, p)

    # init
    tmp = full(inv(Symmetric(Σ)))
    HardThreshold!(S, tmp, s, false)

    tmp .= tmp - S
    U, d, V = svd(tmp)
    U = U[:,1:r] .* sqrt.(d[1:r])'
    L = U * U'


    fvals = []
    # gradient descent
    #
    fv = covSel_objective(Σ, S, L)
    push!(fvals, fv)
    for iter=1:maxIter
      # update S
      covselLatentIHT_grad_S!(gS, Σ, S, L)
      @. S -= ηS * gS
      HardThreshold!(S, S, s, false)
      max_gS = maximum( abs.(gS) )

      # update U and L
      covselLatentIHT_grad_U!(gU, gS, Σ, S, L, U)
      @. U -= ηU * gU
      A_mul_Bt!(L, U, U)
      max_gU = maximum( abs.(gU) )
      if callback != nothing
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
