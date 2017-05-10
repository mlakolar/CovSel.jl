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

  context("small") do
    if cvx && grb
      # simple model that differes in only one edge
      srand(123)

      p = 10
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

      Convex.set_default_solver(Gurobi.GurobiSolver(OutputFlag=0))
      Delta = Convex.Variable(p,p);

      for i=1:5

        # λ = 0.2
        λ = rand(Uniform(0.05, 0.3))

        solShoot = CovSel.differencePrecisionNaive(hSx, hSy, λ, ones(p,p))
        solShoot1 = CovSel.differencePrecisionActiveShooting(hSx, hSy, λ)

        prob = Convex.minimize(Convex.quadform(vec(Delta), kron(hSy,hSx)) / 2 - trace((hSy-hSx)*Delta) +  λ * norm(vec(Delta), 1))
        prob.constraints += [Delta == Delta']
        Convex.solve!(prob)

        @fact maximum(abs.(Delta.value - solShoot)) --> roughly(0.; atol=1e-3)
        @fact maximum(abs.(solShoot1 - solShoot)) --> roughly(0.; atol=1e-5)
        # @fact Delta.value - solShoot --> roughly(zeros(p,p); atol=1e-5)
      end

    end
  end

end
