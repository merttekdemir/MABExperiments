module UnitTests
using Test, Distributions

include(joinpath("..", "src", "MABStruct.jl")); M = MABStructs;


A0 = (Normal(0.15, 0.7), Normal(0.54, 0.2), Normal(0.38, 0.5))
ξ0 = Categorical([1/4, 1/4, 1/4, 1/4])
ξ = Categorical([1/3, 1/3, 1/3])
game = M.MABStruct(10, A0, ξ)


@testset verbose = true "MABStruct tests" begin
    @testset "struct" begin
        # @inferred - this tests if the compiler predicted the same type as the returned one, so we can check for performance, u can check that with @code_warntype
        @test_throws ArgumentError M.MABStruct(10, A0, ξ0, "test_0") 
        @test typeof(game.γ) == Vector{Int8} 
        @test typeof(game.sequence_of_rewards) == Vector{Vector{Float64}} 
        @test typeof(game.cumulative_reward_per_arm) == Vector{Float64} 
        @test typeof(game.average_reward_per_arm) == Vector{Float64} 
        @test typeof(game.best_fixed_choice) == Vector{Int8} 
        @test typeof(game.cumulative_reward_fixed) == Vector{Float64} 
        @test typeof(game.average_reward_fixed) == Vector{Float64} 
        @test typeof(game.best_dynamic_choice) == Vector{Int8} 
        @test typeof(game.cumulative_reward_dynamic) == Vector{Float64} 
        @test typeof(game.average_reward_dynamic) == Vector{Float64} 
        @test true
    end
end
    @testset "update_instance!" begin
        @test 
        @test 
    end
    @testset "pull" begin
        @test 
        @test 
    end
    @testset "run_step!" begin
        @test 
        @test 
    end
    @testset "run!" begin
        @test 
        @test 
    end
    @testset "reset!" begin
        @test 
        @test 
    end
end

tests = ["src MAB tests"]

end # module