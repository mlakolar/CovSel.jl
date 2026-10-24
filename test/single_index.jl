using LinearAlgebra, Random
using CovarianceSelection.SingleIndexGraph
const SIG = CovarianceSelection.SingleIndexGraph

@testset "Single-index precision inference" begin
    rng = MersenneTwister(42017)
    n,p = 360,6
    t = collect(range(-1,1;length=n))
    X = randn(rng,n,p)
    X[:,1] += 0.4X[:,3]
    X[:,2] += 0.3X[:,3]
    @testset "Polynomial weights and HC0 algebra" begin
        D = local_design(t,0.2,0.7)
        @test sum(D.weights) ≈ 1
        @test dot(D.weights,t.-0.2) ≈ 0 atol=1e-14
        @test dot(D.weights,(t.-0.2).^2) ≈ 0 atol=1e-14
        @test minimum(eigvals(Symmetric(local_covariance(X,t,0.2,0.7).covariance))) > 0
        @test_throws DomainError local_design(t,5.0,0.1)
        @test_throws ArgumentError local_design(t,0.0,-1)
        # Independent direct residual-product implementation, including a diagonal.
        B = [0.4 0.3;zeros(3,2)]
        f = fit_precision(X,t;points=[0.2],entries=[(1,2)],bandwidth=0.7,
                          oracle_coefficients=(x,J,A)->B)
        R = X[:,1:2]-X[:,3:6]*B
        Q = inv(R'*(D.weights.*R))
        @test f.estimate[1] ≈ Q[1,2]
        Y = [R[k,:]*R[k,:]' for k in 1:n]
        C = [sum(D.map[j,k]*Y[k] for k in 1:n) for j in 1:3]
        direct = [-D.weights[k]*(Q*(Y[k]-sum(D.basis[k,j]*C[j] for j in 1:3))*Q)[1,2] for k in 1:n]
        @test vec(f.influence) ≈ direct atol=1e-13
        @test f.se[1]^2 ≈ sum(abs2,direct)
        @test abs(sum(f.influence)) < 1e-12
    end
    @testset "Pilots and diagonal inference" begin
        entries = [(1,1),(1,2),(1,6)]
        for pilot in (:sqrtlasso,:scaledlasso,:aclime)
            f = fit_precision(X,t;points=[-0.3,0.3],entries,bandwidth=0.8,
                              pilot_bandwidth=0.9,pilot,multiplier=0.6)
            @test all(isempty,f.failure)
            @test all(>(0),f.se)
            @test vec(sum(f.influence;dims=1)) ≈ zeros(6) atol=1e-11
            @test diag(f.influence'*f.influence) ≈ vec(f.se).^2
            ci = confint(f)
            @test all(ci.lower .< ci.upper)
            band = confband(f;draws=199,rng=MersenneTwister(17))
            @test all(band.critical .>= ci.critical)
            # Reproduce the same observation multipliers manually.
            r = MersenneTwister(17)
            A = f.influence ./ vec(f.se)'
            maxima = [maximum(abs,A'*randn(r,n)) for _ in 1:199]
            c = max(sort(maxima)[190],ci.critical)
            @test band.critical == fill(c,3)
            @test length(confband(f;draws=199,joint=false,rng=MersenneTwister(2)).critical) == 3
        end
        for pilot in (:scaledlasso,:aclime)
            f = fit_precision(X,t;points=[0.0],entries,bandwidth=0.8,pilot,
                              correction=:onestep,multiplier=0.6)
            @test all(isempty,f.failure)
            P = precision_pilot(X,t,0.0,0.8;method=pilot,multiplier=0.6).precision
            S = X'*(local_design(t,0.0,0.8).weights.*X)
            Q = P+P'-P'*S*P
            @test vec(f.estimate) ≈ [Q[a,b] for (a,b) in entries]
        end
        # Support refits use ordinary least squares on the selected neighborhoods.
        Sdiag = Matrix(Diagonal([2.0,3.0,4.0]))
        @test SIG._refit_inverse(Sdiag,Matrix{Float64}(I,3,3)) ≈ inv(Sdiag)
        for pilot in (:sqrtlasso,:scaledlasso,:aclime)
            f = fit_precision(X,t;points=[0.0],entries=[(1,1),(1,2)],bandwidth=0.8,
                              pilot,multiplier=0.7,refit=true)
            @test all(isempty,f.failure)
            @test all(>(0),f.se)
        end
        failed = fit_precision(X,t;points=[5.0],entries,bandwidth=0.2)
        @test all(!isempty,failed.failure)
        @test_throws DomainError confband(failed;draws=100)
        @test_throws ArgumentError fit_precision(X,t;points=[0],entries=[(1,7)],bandwidth=0.5)
        @test_throws ArgumentError fit_precision(X,t;points=[0],entries=[(1,2)],bandwidth=0.5,
            pilot=:scaledlasso,correction=:onestep,oracle_coefficients=(point,J,A)->zeros(length(A),length(J)))
        @test_throws ArgumentError fit_precision(X,t;points=[0],entries=[(1,2),(2,1)],bandwidth=0.5)
    end
    @testset "Stein index and norms" begin
        score = reshape(collect(range(-1,1;length=n)),n,1)
        XP = ones(n,2)
        XP[:,2] = score[:,1]
        q = estimate_index(XP,score;probe=(1,2),blocks=5,threshold=0)
        @test q.direction == [1.0]
        @test q.discarded == 0
        two = hcat(score,score.^3)
        unshrunk = estimate_index(XP,two;probe=(1,2),blocks=5,threshold=0.02,unshrink=true)
        active = unshrunk.moment .* (abs.(unshrunk.moment).>unshrunk.threshold)
        @test unshrunk.direction ≈ active/norm(active)
        tuned = select_index_tuning(XP,two;probe=(1,2),blocks=5,multipliers=[0.0,0.5],
                                    unshrink_options=[false,true],block_options=[1,5],rng=MersenneTwister(19))
        @test length(tuned.scores)==8
        @test tuned.multiplier in (0.0,0.5)
        @test tuned.blocks in (1,5)
        @test norm(tuned.estimate.direction) ≈ 1
        @test all(x -> x.alignment>0,tuned.scores)
        negative = copy(XP)
        negative[:,2] .*= -1
        oriented = select_index_tuning(negative,score;probe=(1,2),blocks=5,multipliers=[0.0],
                                       unshrink_options=[true],rng=MersenneTwister(19))
        @test oriented.estimate.direction == [1.0]
        @test oriented.scores[1].alignment>0
        @test_throws DomainError estimate_index(zeros(n,2),score;probe=(1,2),threshold=0)
        P = [2.0 1;1 4]
        norms = precision_errors(P,zeros(2,2))
        @test norms.induced_l1 == 5
        @test norms.entry_l1 == 8
        @test norms.max_entry == 4
        @test norms.spectral ≈ opnorm(P)
    end
    @testset "Finite sample corrections" begin
        J,A = [1,2],[3,4,5,6]
        B = [0.4 0.3;zeros(3,2)]
        d = SIG._residual_df(X,t,0.0,0.9,J,A,B,biweight)
        w = biweight.(t./0.9)
        U = X[:,[3]]
        H = U*((U'*(w.*U)) \ (U'.*w'))
        M = Matrix{Float64}(I,n,n)-H
        @test d.df == 1
        @test d.factor ≈ vec(sum(abs2,M;dims=2)) atol=1e-12
        @test X[:,A]*d.coefficients ≈ H*X[:,J]
        # A common union support gives the same residual maker for both responses.
        D = local_design(t,0.0,0.8)
        @test D.leverage ≈ diag(D.basis*D.map)
        @test all(0 .<= D.leverage .< 1)
        raw = fit_precision(X,t;points=[0.0],entries=[(1,1)],bandwidth=0.8,
                            oracle_coefficients=(x,J,A)->reshape([0,0.4,0,0,0],5,1),
                            studentization=:hc3)
        bc = fit_precision(X,t;points=[0.0],entries=[(1,1)],bandwidth=0.8,
                           oracle_coefficients=(x,J,A)->reshape([0,0.4,0,0,0],5,1),
                           studentization=:hc3,inverse_bias=:secondorder)
        @test all(isempty,bc.failure)
        @test bc.estimate[1] ≈ raw.estimate[1]-raw.se[1]^2/raw.estimate[1]
        @test bc.influence == raw.influence
        adjusted = fit_precision(X,t;points=[0.0],entries=[(1,1),(1,2)],bandwidth=0.8,
                                 pilot_bandwidth=0.9,multiplier=0.7,refit=true,
                                 residual_adjustment=:df,studentization=:hc3,inverse_bias=:secondorder)
        @test all(isempty,adjusted.failure)
        @test all(isfinite,adjusted.estimate)
        identity = confint(adjusted)
        logs = confint(adjusted;diagonal_transform=:log)
        @test logs.lower[1] ≈ adjusted.estimate[1]*exp(-logs.critical*adjusted.se[1]/adjusted.estimate[1])
        @test logs.upper[1] ≈ adjusted.estimate[1]*exp(logs.critical*adjusted.se[1]/adjusted.estimate[1])
        @test logs.lower[2] == identity.lower[2]
        @test logs.upper[2] == identity.upper[2]
        b0 = confband(adjusted;draws=199,rng=MersenneTwister(27))
        bl = confband(adjusted;draws=199,rng=MersenneTwister(27),diagonal_transform=:log)
        @test bl.critical == b0.critical
        @test all(bl.lower[:,1] .> 0)
        @test bl.lower[:,2] == b0.lower[:,2]
        @test_throws ArgumentError fit_precision(X,t;points=[0.0],entries=[(1,1)],bandwidth=0.8,
            pilot=:scaledlasso,correction=:onestep,residual_adjustment=:df)
    end
    @testset "Truth-free tuning" begin
        s = select_tuning(X,t;points=[0.0],entries=[(1,2)],bandwidths=[0.6,0.9],
                          pilot_bandwidths=[0.8,1.0],multipliers=[0.6,1.0],rng=MersenneTwister(7))
        @test s.validation_bandwidths == (pilot=0.8,final=0.6)
        @test s.bandwidth in (0.6,0.9)
        @test s.pilot_bandwidth in (0.8,1.0)
        @test length(s.pilot_scores) == 4
        @test all(isfinite,[v.loss for v in s.final_scores])
        guarded = select_tuning(X,t;points=[0.0],entries=[(1,2)],bandwidths=[0.6,0.9],
            pilot_bandwidths=[0.8,1.0,1.2],multipliers=[0.6,1.0],rng=MersenneTwister(7),
            rule=:inference,refit=true)
        @test guarded.bandwidth <= guarded.bandwidth_cap
        @test guarded.pilot_bandwidth >= guarded.pilot_ratio*guarded.bandwidth
        @test all(x -> x.complexity>=0 && x.standard_error>=0,guarded.pilot_scores)
        minpilot = select_tuning(X,t;points=[0.0],entries=[(1,2)],bandwidths=[0.6,0.9],
            pilot_bandwidths=[0.8,1.0,1.2],multipliers=[0.6,1.0],rng=MersenneTwister(7),
            rule=:inference,refit=true,pilot_se=0)
        winner = filter(x -> x.bandwidth==minpilot.pilot_bandwidth && x.multiplier==minpilot.multiplier,minpilot.pilot_scores)[1]
        @test winner.loss == minimum(x.loss for x in minpilot.pilot_scores)
    end
end
