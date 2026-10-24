using Test, LinearAlgebra, Random, CovarianceSelection
using CovarianceSelection.SingleIndexGraph
include("dgp.jl")
@testset "Independent DGP checks" begin
    for graph in ("block","chain","grid"), t in (-1.0,0.0,1.0), varying in (false,true)
        S = covariance_truth(t,12;graph,varying)
        @test isposdef(Symmetric(S))
        O = inv(Symmetric(S))
        @test S*O ≈ I atol=1e-12
        if graph == "block"
            q=1-0.35^2
            delta=0.2tanh(2t)
            @test O[1,2] ≈ -delta/(q*q-delta*delta) atol=1e-12
            @test O[1,1] ≈ q/(q*q-delta*delta) atol=1e-12
            @test abs(O[1,10]) < 1e-12
        elseif graph == "chain"
            @test abs(O[1,4]) < 1e-12
        end
    end
    # Structural generating equations, verified without a generic covariance draw.
    c=Dict{String,Any}("n"=>20000,"p"=>10,"d"=>4,"varying"=>true)
    D=generate_data(6011,c)
    @test D.X == generate_data(6011,c).X
    r=2
    u=0.35/sqrt(r).*(1 .+ 0.5tanh.(D.t))
    b=u.*vec(sum(D.X[:,3:4];dims=2))
    e1,e2=D.X[:,1]-b,D.X[:,2]-b
    @test abs(sum(e1.^2)/length(e1)-(1-0.35^2)) < 0.03
    @test abs(sum(e1.*e2 .- 0.2tanh.(2D.t))/length(e1)) < 0.03
    rho=0.7tanh.(2D.t)
    eprobe=(D.X[:,10]-rho.*D.X[:,9])./sqrt.(1 .- rho.^2)
    @test abs(sum(abs2,eprobe)/length(eprobe)-1) < 0.03
    for design in ("bump","gaussian")
        c["design"]=design
        D=generate_data(4011,c)
        @test abs(sum(abs2,D.Z[:,1])/size(D.Z,1)-1) < 0.04
        @test abs(sum(D.Z[:,1].*D.score[:,1])/size(D.Z,1)-1) < 0.08
    end
end
