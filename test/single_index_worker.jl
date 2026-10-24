using Test, TOML
include("../experiments/single_index/worker.jl")
@testset "Worker correction and log labels" begin
    root = mktempdir()
    config = Dict("seed_base"=>710000000,"shard_size"=>1,"scenario"=>[
        Dict("id"=>1,"name"=>"worker_flags","replicates"=>1,"n"=>240,"p"=>8,
             "d"=>4,"index_sparsity"=>2,"graph_degree"=>4,"grid_min"=>-0.25,
             "grid_max"=>0.25,"grid_n"=>3,"h_factors"=>[1.0],"pilot_factors"=>[1.6],
             "penalties"=>[0.7],"index_modes"=>["known"],"norms"=>false,"draws"=>199,
             "methods"=>["scaled_onestep","scaled_onestep_log"])])
    path = joinpath(root,"plan.toml")
    open(io -> TOML.print(io,config),path,"w")
    runshard(path,1,root)
    rows = [split(line,',') for line in readlines(joinpath(root,"shard-00001.csv"))[2:end]]
    @test length(rows)==32
    @test all(r -> r[17]=="\"\"",rows)
    normal = filter(r -> r[3]=="\"scaled_onestep\"",rows)
    logs = filter(r -> r[3]=="\"scaled_onestep_log\"",rows)
    @test length(normal)==length(logs)==16
    @test all(length(r)==28 for r in rows)
    for (a,b) in zip(normal,logs)
        @test a[8:11]==b[8:11]
        @test a[12:14]==b[12:14]
        if a[8]=="\"pointwise\""
            if a[9]==a[10]
                @test parse(Float64,b[16])>parse(Float64,a[16])
            else
                @test a[16]==b[16]
            end
        else
            @test a[19]==b[19] # identical standardized multiplier critical value
        end
    end
    c = config["scenario"][1]
    c["index_modes"] = ["estimated_cv"]
    c["index_multipliers"] = [0.0]
    c["index_unshrink_options"] = [true]
    c["methods"] = ["sqrt_refit_df_hc3_bc_log"]
    c["draws"] = 0
    open(io -> TOML.print(io,config),path,"w")
    runshard(path,1,joinpath(root,"index"))
    rr = [split(line,',') for line in readlines(joinpath(root,"index","shard-00001.csv"))[2:end]]
    @test length(rr)==9
    @test all(r -> r[17]=="\"\"",rr)
    @test all(r -> parse(Float64,r[25])==0 && parse(Int,r[26])==4 && r[27]=="true" && parse(Int,r[28])==5,rr)
end
