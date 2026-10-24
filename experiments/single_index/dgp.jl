# Scientific definitions are independent of inference code.
using LinearAlgebra, Random

function covariance_truth(t,p;graph="block",varying=false,curvature=1.0,signal=0.7,degree=4)
    if graph == "chain"
        rho = 0.25+0.12tanh(curvature*t)
        S = [rho^abs(a-b) for a in 1:p,b in 1:p]
        return S
    elseif graph == "grid"
        side = ceil(Int,sqrt(p))
        O = zeros(p,p)
        edge = 0.12+0.06tanh(curvature*t)
        for a in 1:p,b in a+1:p
            if b == a+side || (b == a+1 && (a-1)÷side == (b-1)÷side)
                O[a,b] = O[b,a] = -edge
            end
        end
        for a in 1:p
            O[a,a] = 1+sum(abs,O[a,:])
        end
        return inv(Symmetric(O))
    end
    r = min(degree-2,p-4)
    r >= 1 || error("block design requires p>=5 and degree>=3")
    S = Matrix{Float64}(I,p,p)
    u = 0.35/sqrt(r)*(varying ? 1+0.5tanh(t) : 1)
    delta = 0.20tanh(2curvature*t)
    # Keep structural residual covariance fixed apart from its smooth target.
    q = 1-0.35^2
    S[1,1] = S[2,2] = q+r*u*u
    S[1,2] = S[2,1] = delta+r*u*u
    for a in 3:r+2
        S[1,a] = S[a,1] = S[2,a] = S[a,2] = u
    end
    S[p-1,p] = S[p,p-1] = signal*tanh(2t)
    S
end

function generate_data(seed,c)
    rng = MersenneTwister(seed)
    n,p,d = Int(c["n"]),Int(c["p"]),Int(c["d"])
    sg = min(Int(get(c,"index_sparsity",2)),d)
    gamma = zeros(d)
    gamma[1] = sg == 1 ? 1.0 : inv(sqrt(2))
    sg > 1 && (gamma[2:sg] .= inv(sqrt(2*(sg-1))))
    Z = randn(rng,n,d)
    score = copy(Z)
    design = get(c,"design","gaussian")
    if design == "bump"
        sig = 0.397635054118470
        for k in eachindex(Z)
            while true
                v = 2rand(rng)-1
                if log(rand(rng)) <= 1-1/(1-v*v)
                    Z[k] = v/sig
                    score[k] = 2sig^2*Z[k]/(1-sig^2*Z[k]^2)^2
                    break
                end
            end
        end
    end
    t = design == "uniform" ? 2rand(rng,n).-1 : Z*gamma
    signal = Float64(get(c,"signal",0.7))
    graph = get(c,"graph","block")
    varying = get(c,"varying",false)
    curvature = Float64(get(c,"curvature",1.0))
    degree = Int(get(c,"graph_degree",4))
    truth(s) = covariance_truth(s,p;graph,varying,curvature,signal,degree)
    X = randn(rng,n,p)
    if graph == "block"
        r = min(degree-2,p-4)
        q = 1-0.35^2
        for k in 1:n
            u = 0.35/sqrt(r)*(varying ? 1+0.5tanh(t[k]) : 1)
            delta = 0.20tanh(2curvature*t[k])
            e1,e2 = X[k,1],X[k,2]
            b = u*sum(@view X[k,3:r+2])
            X[k,1] = b+(sqrt(q+delta)*e1+sqrt(q-delta)*e2)/sqrt(2)
            X[k,2] = b+(sqrt(q+delta)*e1-sqrt(q-delta)*e2)/sqrt(2)
            rho = signal*tanh(2t[k])
            X[k,p] = rho*X[k,p-1]+sqrt(1-rho^2)*X[k,p]
        end
    elseif graph == "chain"
        for k in 1:n
            rho = 0.25+0.12tanh(curvature*t[k])
            for a in 2:p
                X[k,a] = rho*X[k,a-1]+sqrt(1-rho^2)*X[k,a]
            end
        end
    elseif graph == "grid"
        for k in 1:n
            X[k,:] = cholesky(Symmetric(truth(t[k]))).L*X[k,:]
        end
    else
        error("unknown graph")
    end
    noise = get(c,"noise","gaussian")
    if noise == "t8"
        for k in 1:n
            X[k,:] .*= sqrt(6/sum(abs2,randn(rng,8)))
        end
    elseif noise == "mixture"
        # E(scale^2)=1; target covariance is unchanged.
        for k in 1:n
            X[k,:] .*= (rand(rng)<0.05 ? 3.0 : 1.0)/sqrt(1.4)
        end
    elseif noise != "gaussian"
        error("unknown noise")
    end
    (;X,Z,score,t,gamma,truth)
end
