

facts("direct_difference_estimation") do

  function genData(p)
    srand(123)
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
    hSy = cov(Y)
    hSx, hSy
  end

  context("small") do
    # simple model that differes in only one edge
    p = 10
    hSx, hSy = genData(p)

    tmp = ones(p,p)
    for i=1:5

      # λ = 0.2
      λ = rand(Uniform(0.05, 0.3))
      g = ProximalBase.ProxL1(λ)

      solShoot = CovSel.Alt.differencePrecisionNaive(hSx, hSy, λ, tmp)
      solShoot1 = CovSel.differencePrecisionActiveShooting(Symmetric(hSx), Symmetric(hSy), g)

      @fact maximum(abs.(full(solShoot1) - solShoot)) --> roughly(0.; atol=1e-5)
    end
  end

  context("loss") do
    for rep=1:5
      p = 10
      hSx, hSy = genData(p)

      A = sprand(p, p, 0.15)
      A = (A + A') / 2.
      As = convert(ProximalBase.SymmetricSparseIterate, A)

      @fact CovSel.diffLoss(Symmetric(hSx), As, Symmetric(hSy), 2) - vecnorm(hSx*A*hSy, 2)--> roughly(0.; atol=1e-12)
      @fact CovSel.diffLoss(Symmetric(hSx), As, Symmetric(hSy), Inf) - vecnorm(hSx*A*hSy, Inf) --> roughly(0.; atol=1e-12)
      @fact_throws CovSel.diffLoss(Symmetric(hSx), As, Symmetric(hSy), 3) ArgumentError
    end
  end

  context("invert Kroneker") do
    for rep=1:50
      p = 20
      hSx, hSy = genData(p)
      A = kron(hSy, hSx)

      λ = rand(Uniform(0.05, 0.3))
      g = ProximalBase.ProxL1(λ)

      for i = [(1,1), (3,1), (4,5)]
        ri = i[1]
        ci = i[2]
        x = ProximalBase.SparseIterate(p*p)
        x1 = ProximalBase.SparseIterate(p*p)

        f = CovSel.CDInverseKroneckerLoss(Symmetric(hSx), Symmetric(hSy), ri, ci)
        b = zeros(Float64, p*p)
        b[(ci-1)*p+ri] = -1.

        f1 = CoordinateDescent.CDQuadraticLoss(A, b)
        CoordinateDescent.coordinateDescent!(x, f, g, CoordinateDescent.CDOptions(;maxIter=5000, optTol=1e-12))
        CoordinateDescent.coordinateDescent!(x1, f1, g, CoordinateDescent.CDOptions(;maxIter=5000, optTol=1e-12))

        @fact full(x) - full(x1) --> roughly(zeros(p*p); atol=1e-7)
      end
    end
  end

  context("refit") do
    if cvx && grb
      for rep=1:50
        p = 10
        hSx, hSy = genData(p)

        Convex.set_default_solver(Gurobi.GurobiSolver(OutputFlag=0))
        Delta = Convex.Variable(p,p);

        S = sprand(p, p, 0.1)
        S = S + S' / 2
        indS = find(S)

        opt = CoordinateDescent.CDOptions(;maxIter=5000, optTol=1e-12, randomize=true)
        x = CovSel.differencePrecisionRefit(Symmetric(hSx), Symmetric(hSy), indS)

        prob = Convex.minimize(Convex.quadform(vec(Delta), kron(hSy,hSx)) / 2 - trace((hSy-hSx)*Delta))
        prob.constraints += [Delta == Delta']
        for i=1:p*p
          if S[i] == 0.
            prob.constraints += [Delta[i] == 0.]
          end
        end
        Convex.solve!(prob)

        @fact maximum(abs.(Delta.value - x)) --> roughly(0.; atol=1e-3)
      end
    end
  end


end
