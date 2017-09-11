# facts("fused_graphical_lasso") do
#
#   context("random") do
#     if scs && cvx
#       # simple model that differes in only one edge
#       srand(123)
#       p = 5
#       Sigma_x = eye(p)
#       ρ = 0.3
#       iρ = -ρ / (1. - ρ^2)
#       Sigma_y = eye(p)
#       Sigma_y[1,2] = iρ
#       Sigma_y[2,1] = iρ
#       Sigma_y[1,1] = 1./(1. - ρ^2)
#       Sigma_y[2,2] = 1./(1. - ρ^2);
#
#       n = 1000
#       X = randn(n, p)
#       Y = randn(n, p) * sqrtm(Sigma_y)
#
#       # passing in verbose=0 to hide output from SCS
#       solver = SCS.SCSSolver(verbose=0)
#       Convex.set_default_solver(solver);
#       Sx = cov(X)
#       Sy = cov(Y)
#       Ωx = Convex.Variable(p, p)
#       Ωy = Convex.Variable(p, p)
#
#
#       for i=1:20
#
#         λ1 = rand(Uniform(0.01,0.1))
#         λ2 = rand(Uniform(0.01,0.1))
#         Zx, Zy = CovSel.fusedGraphicalLasso(X, Y, λ1, λ2;
#                           penalize_diag=true,
#                           options=CovSel.ADMMOptions(;abstol=1e-12,reltol=1e-12))
#
#         problem = Convex.minimize(
#               (trace(Ωx*Sx) - logdet(Ωx))/2. +
#               (trace(Ωy*Sy) - logdet(Ωy))/2.
#               + λ1 * vecnorm(Ωx, 1) + λ1 * vecnorm(Ωy, 1)
#               + λ2 * vecnorm(Ωx - Ωy, 1),
#               [Ωx in :SDP, Ωy in :SDP])
#         Convex.solve!(problem)
#
#         @fact Ωx.value - Zx --> roughly(zeros(p,p); atol=1e-2)
#         @fact Ωy.value - Zy --> roughly(zeros(p,p); atol=1e-2)
#
#       end
#
#     end
#   end
#
# end
#
#
# facts("fused_neighborhood_selection") do
#
#   context("random") do
#     if jmp && grb
#       # simple model that differes in only one edge
#       srand(123)
#       p = 5
#       Sigma_x = eye(p)
#       ρ = 0.3
#       iρ = -ρ / (1. - ρ^2)
#       Sigma_y = eye(p)
#       Sigma_y[1,2] = iρ
#       Sigma_y[2,1] = iρ
#       Sigma_y[1,1] = 1./(1. - ρ^2)
#       Sigma_y[2,2] = 1./(1. - ρ^2);
#
#       n = 1000
#       X = randn(n, p)
#       Y = randn(n, p) * sqrtm(Sigma_y)
#
#       m = JuMP.Model(solver=Gurobi.GurobiSolver(OutputFlag=0))
#       JuMP.@variable(m, xj1[1:p-1])
#       JuMP.@variable(m, xj2[1:p-1])
#       JuMP.@variable(m, t1[1:p-1])
#       JuMP.@variable(m, t2[1:p-1])
#       JuMP.@variable(m, t3[1:p-1])
#
#       JuMP.@constraint(m, xj1 .<= t1 )
#       JuMP.@constraint(m, -t1 .<= xj1 )
#       JuMP.@constraint(m, xj2 .<= t2 )
#       JuMP.@constraint(m, -t2 .<= xj2 )
#       JuMP.@constraint(m, -t3 .<= xj1 - xj2 )
#       JuMP.@constraint(m, xj1 - xj2 .<= t3 )
#
#       for i=1:20
#
#         λ1 = rand(Uniform(0., 0.1))
#         λ2 = rand(Uniform(0., 0.1))
#         z1, z2 = CovSel.fusedNeighborhoodSelection(X[:,2:p], X[:,1], Y[:,2:p], Y[:,1], λ1, λ2;
#                           options=CovSel.ADMMOptions(;abstol=1e-12,reltol=1e-12))
#
#         JuMP.@objective(m, Min, (sum((X[:,1] - X[:,2:p]*xj1).^2) + sum((Y[:,1] - Y[:,2:p]*xj2).^2))/(2.*n) + λ1 * (sum(t1)+sum(t2)) + λ2 * sum(t3))
#         JuMP.solve(m)
#
#         @fact JuMP.getvalue(xj1) - z1 --> roughly(zeros(p-1); atol=1e-5)
#         @fact JuMP.getvalue(xj2) - z2 --> roughly(zeros(p-1); atol=1e-5)
#
#       end
#
#     end
#   end
#
# end


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

    #   Convex.set_default_solver(Gurobi.GurobiSolver(OutputFlag=0))
    #   Delta = Convex.Variable(p,p);

    tmp = ones(p,p)
    for i=1:5

      # λ = 0.2
      λ = rand(Uniform(0.05, 0.3))
      g = ProximalBase.ProxL1(λ)

      solShoot = CovSel.Alt.differencePrecisionNaive(hSx, hSy, λ, tmp)
      solShoot1 = CovSel.differencePrecisionActiveShooting(Symmetric(hSx), Symmetric(hSy), g)

      # prob = Convex.minimize(Convex.quadform(vec(Delta), kron(hSy,hSx)) / 2 - trace((hSy-hSx)*Delta) +  λ * norm(vec(Delta), 1))
      # prob.constraints += [Delta == Delta']
      # Convex.solve!(prob)

      # @fact maximum(abs.(tril(Delta.value - solShoot))) --> roughly(0.; atol=2e-3)
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
