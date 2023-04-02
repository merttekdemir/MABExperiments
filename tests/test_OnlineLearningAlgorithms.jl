module UnitTests_OnlineLearningAlgorithms
using Test, Distributions, Random, DataStructures

include(joinpath("..", "src", "MABStruct.jl")); M = MABStructs;
include(joinpath("..", "src", "OnlineLearningAlgorithms.jl")); O = OnlineLearningAlgorithms;

"""Testing functions"""

function test_degenerate_case(algorithm::Function, argnames::Vector{Symbol}, default_argnames::Vector{Symbol}, default_values_algo::Dict)
    T_degenerate = 100
    A_degenerate = (Distributions.Bernoulli(0), Distributions.Bernoulli(1), Distributions.Bernoulli(0))
    game_degenerate = M.MABStruct(T_degenerate, A_degenerate, Distributions.Categorical(3))
    
    # This test is conditioned on the proper functioning of run!
    M.run!(game_degenerate, algorithm, argnames, default_argnames, default_values_algo)
    @test argmax(game_degenerate.choices_per_arm) == 2
end


"""Start tests"""

@testset verbose = true "OnlineLearningAlgorithms tests" begin
    @testset "ExponentiatedGradient" begin
        algorithm = O.ExponentiatedGradient
        argnames = Symbol[:ξ, :reward_vector]
        default_argnames = Symbol[:η]
        kw_list = [Distributions.Categorical(3), [0.2, 1, 0.3]]
        default_values_algo = Dict{Any, Any}("η" => 0.05)
        
        # Test Argument Error
        @test_throws ArgumentError algorithm(kw_list...; η=-2)
    
        # Test Degenerate Error
        test_degenerate_case(algorithm, argnames, default_argnames, default_values_algo)
        
        # Test Proper Update
        ξ_test = Distributions.Categorical(3)
        probs(ξ_test) .= algorithm(Distributions.Categorical(3), [0.1, 0.8, 0.1])
        @test argmax(probs(ξ_test)) == 2
        for _ in 1:10
            probs(ξ_test) .= algorithm(ξ_test, [0.8, 0.1, 0.1])
        end
        @test argmax(probs(ξ_test)) == 1
        for _ in 1:50
            probs(ξ_test) .= algorithm(ξ_test, [0.1, 0.1, 0.8])
        end
        @test argmax(probs(ξ_test)) == 3
    end

    @testset "FtrlExponentiatedGradient" begin
        algorithm = O.FtrlExponentiatedGradient
        argnames = Symbol[:τ, :cumulative_reward_per_arm]
        kw_list = [10, [100.0, 200.0, 100.0]]
        default_argnames = Symbol[:α]
        default_values_algo = Dict{Any, Any}("α" => 0.05)
        
        # Test Argument Error
        @test_throws ArgumentError algorithm(kw_list...; α=-2.0)
    
        # Test Degenerate Case
        test_degenerate_case(algorithm, argnames, default_argnames, default_values_algo)
        
        # Test Proper Update
        ξ_test = Distributions.Categorical(3)
        probs(ξ_test) .= algorithm(1, [0.1, 0.8, 0.1])
        @test argmax(probs(ξ_test)) == 2
        probs(ξ_test) .= algorithm(2, [0.9, 0.8, 0.1])
        @test argmax(probs(ξ_test)) == 1
        probs(ξ_test) .= algorithm(3, [0.9, 0.8, 1.0])
        @test argmax(probs(ξ_test)) == 3
    end

    @testset "EXP3" begin
        algorithm = O.EXP3
        argnames = Symbol[:ξ, :reward_vector, :γ, :T, :τ]
        kw_list = [Distributions.Categorical(3), [0.2, 1, 0.3], Int64[0 for _ in 1:100], 100, 2]
        default_argnames = Symbol[:η]
        default_values_algo = Dict{Any, Any}("η" => 0.05)
        
        # Test Argument Error
        @test_throws ArgumentError algorithm(kw_list...; η=-2.0)
    
        # Test Degenerate Case
        test_degenerate_case(algorithm, argnames, default_argnames, default_values_algo)
        
        # Test Proper Update
        ξ_test = Distributions.Categorical(3)
        γ = Int64[0 for _ in 1:1001]
        γ[1] = 2
        probs(ξ_test) .= algorithm(Distributions.Categorical(3), [0.1, 0.5, 0.1], γ, 1001, 1)
        @test argmax(probs(ξ_test)) == 2
        for i in 2:50
            γ[i] = 1
            probs(ξ_test) .= algorithm(ξ_test, [0.6, 0.1, 0.1], γ, 1001, i)
        end
        @test argmax(probs(ξ_test)) == 1
        for i in 51:1000
            γ[i] = 3
            probs(ξ_test) .= algorithm(ξ_test, [0.1, 0.1, 0.9], γ, 1001, i)
        end
        @test argmax(probs(ξ_test)) == 3
    end

    @testset "ExploreThenCommit" begin
        algorithm = O.ExploreThenCommit
        argnames = Symbol[:ξ, :τ, :cumulative_reward_per_arm_bandit, :choices_per_arm]
        kw_list = [Distributions.Categorical(3), 40, [100.0, 200.0, 100.0], [2, 2, 2]]
        default_argnames = Symbol[:m]
        default_values_algo = Dict{Any, Any}("m" => 10)

        # Test Argument Error
        @test_throws ArgumentError algorithm(kw_list...; m=-2)

        # Test Degenerate Case
        test_degenerate_case(algorithm, argnames, default_argnames, default_values_algo)
        
        # Test Exploration
        repeated_results = [algorithm(Distributions.Categorical(3), τ, [100.0, 200.0, 100.0], Int64[10, 10, 10]; m=100) for τ in 1:50]
        @test all(any.([repeated_results[j] .≈ 1.0 for j in 1:50]))
        repeated_results_action = map(x -> argmax(x), repeated_results)
        @test abs(counter(repeated_results_action)[1] - counter(repeated_results_action)[2]) < 2

        # Test Exploitation
        kw_list[2] = 100
        @test all(algorithm(kw_list...; m=1) .≈ probs(kw_list[1]))
    end

    @testset "UpperConfidenceBound" begin
        algorithm = O.UpperConfidenceBound
        argnames = Symbol[:ξ, :τ, :choices_per_arm, :average_reward_per_arm_bandit]
        kw_list = [Distributions.Categorical(3), 30, [10, 15, 5], [0.5, 0.7, 0.2]]
        default_argnames = Symbol[:α]
        default_values_algo = Dict{Any, Any}("α" => 3)

        # Test Argument Error
        @test_throws ArgumentError algorithm(kw_list...; α=1)

        # Test Degenerate Case
        test_degenerate_case(algorithm, argnames, default_argnames, default_values_algo)

        # Test Proper Update  # TODO: FIX
        ξ_new = Distributions.Categorical(3)
        lower_confidence_band = -1 .* kw_list[4] .- sqrt(2*default_values_algo["α"]*log(kw_list[2]))./kw_list[3]
        default_kw_dict = Dict{Any, Any}(:α => 3)
        @test argmin(lower_confidence_band) == argmax(algorithm(kw_list...; default_kw_dict...))
    end

    @testset "EpsilonGreedy" begin
        algorithm = O.EpsilonGreedy
        argnames = Symbol[:ξ, :average_reward_per_arm_bandit]
        kw_list = [Distributions.Categorical(3), [0.0, 1.0, 0.0]]
        default_argnames = Symbol[:ϵ]
        default_values_algo = Dict{Any, Any}("ϵ" => 0.25)
        
        # Test Argument Error
        @test_throws ArgumentError algorithm(kw_list...; ϵ=1.2)
        @test_throws ArgumentError algorithm(kw_list...; ϵ=-1.2)
    
        # Test Degenerate Case
        test_degenerate_case(algorithm, argnames, default_argnames, default_values_algo)

        # Test Exploration
        repeated_results = [algorithm(Distributions.Categorical(3), [0.5, 1.0, 0.0]; ϵ=1.0) for _ in 1:100]
        @test all(any.([repeated_results[j] .≈ 1.0 for j in 1:100]))
        repeated_results_action = map(x -> argmax(x), repeated_results)
        @test abs(counter(repeated_results_action)[1] - counter(repeated_results_action)[2]) < 20

        # Test Exploitation 
        @test algorithm(Distributions.Categorical(3), [0.0, 1.0, 0.0]; ϵ=0)[2] ≈ 1.0  
    end

    @testset "LinearDecayedEpsilonGreedy" begin
        algorithm = O.LinearDecayedEpsilonGreedy
        argnames = Symbol[:T, :τ, :ξ, :reward_vector]
        kw_list = [100, 10, Distributions.Categorical(3), [0.0, 1.0, 0.0]]
        default_argnames = Symbol[:ϵ_start, :ϵ_end]
        default_values_algo = Dict{Any, Any}("ϵ_start" => 1.0, "ϵ_end" => 0.0)

        # Test Argument Error
        @test_throws ArgumentError algorithm(kw_list...; ϵ_start=1.5)
        @test_throws ArgumentError algorithm(kw_list...; ϵ_start=-2)
        @test_throws ArgumentError algorithm(kw_list...; ϵ_end=1.5)
        @test_throws ArgumentError algorithm(kw_list...; ϵ_end=-2)
        @test_throws ArgumentError algorithm(kw_list...; ϵ_start=0.5, ϵ_end=1.0)
     
        # Test Degenerate Case
        test_degenerate_case(algorithm, argnames, default_argnames, default_values_algo)

        # Test Exploration
        repeated_results = [algorithm(100, τ, Distributions.Categorical(3), [0.5, 1.0, 0.0]; ϵ_start=1.0, ϵ_end=1.0) for τ in 1:100]
        @test all(any.([repeated_results[j] .≈ 1.0 for j in 1:100]))
        repeated_results_action = map(x -> argmax(x), repeated_results)
        @test any([abs(counter(repeated_results_action)[1] - counter(repeated_results_action)[i]) for i in 2:3] .< 20)

        # Test Exploitation 
        @test algorithm(100, 10, Distributions.Categorical(3), [0.0, 1.0, 0.0]; ϵ_start=0.0, ϵ_end=0.0)[2] ≈ 1.0
    end

    @testset "ExpDecayedEpsilonGreedy" begin
        algorithm = O.ExpDecayedEpsilonGreedy
        argnames = Symbol[:T, :τ, :ξ, :average_reward_per_arm_bandit]
        kw_list = [100, 10, Distributions.Categorical(3), [0.0, 1.0, 0.0]]
        default_argnames = Symbol[:ϵ_start, :ϵ_end]
        default_values_algo = Dict{Any, Any}("ϵ_start" => 1.0, "ϵ_end" => 0.0001)
        
        # Test Argument Error
        @test_throws ArgumentError algorithm(kw_list...; ϵ_start=1.5)
        @test_throws ArgumentError algorithm(kw_list...; ϵ_start=-2)
        @test_throws ArgumentError algorithm(kw_list...; ϵ_end=1.5)
        @test_throws ArgumentError algorithm(kw_list...; ϵ_end=0)
        @test_throws ArgumentError algorithm(kw_list...; ϵ_start=0.5, ϵ_end=1.0)

        # Test Degenerate Case
        test_degenerate_case(algorithm, argnames, default_argnames, default_values_algo)

        # Test Exploration
        repeated_results = [algorithm(100, τ, Distributions.Categorical(3), [0.5, 1.0, 0.0]; ϵ_start=1.0, ϵ_end=1.0) for τ in 1:100]
        @test all(any.([repeated_results[j] .≈ 1.0 for j in 1:100]))
        repeated_results_action = map(x -> argmax(x), repeated_results)
        @test any([abs(counter(repeated_results_action)[1] - counter(repeated_results_action)[i]) for i in 2:3] .< 20)

        # Test Exploitation 
        @test algorithm(100, 100, Distributions.Categorical(3), [0.0, 1.0, 0.0]; ϵ_start=1.0, ϵ_end=0.001)[2] ≈ 1.0
    end

    @testset "Hedge" begin
        algorithm = O.Hedge
        argnames = Symbol[:ξ, :reward_vector]
        kw_list = [Distributions.Categorical(3), [0.1, 0.5, 0.2]]
        default_argnames = Symbol[:η]
        default_values_algo = Dict{Any, Any}("η" => 0.05)
        
        # Test Argument Error
        @test_throws ArgumentError algorithm(kw_list...; η=-2)
    
        # Test Degenerate Case
        test_degenerate_case(algorithm, argnames, default_argnames, default_values_algo)
    
        # Test Proper Update
        ξ_test = Distributions.Categorical(3)
        probs(ξ_test) .= algorithm(Distributions.Categorical(3), [0.1, 0.8, 0.1])
        @test argmax(probs(ξ_test)) == 2
        for _ in 1:10
            probs(ξ_test) .= algorithm(ξ_test, [0.8, 0.1, 0.1])
        end
        @test argmax(probs(ξ_test)) == 1
        for _ in 1:50
            probs(ξ_test) .= algorithm(ξ_test, [0.1, 0.1, 0.8])
        end
        @test argmax(probs(ξ_test)) == 3
    end
end 

end  # module