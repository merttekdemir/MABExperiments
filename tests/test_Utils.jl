module UnitTests_Utils
using Test

include(joinpath("..", "src", "Utils.jl")); U = Utils;
include(joinpath("..", "src", "OnlineLearningAlgorithms.jl")); O = OnlineLearningAlgorithms;

# MAB Experiments tests
@testset verbose = true "Utils tests" begin
    @testset "Test method_args function" begin
        function testing_args(a::Int64, b::Int64, c::Int64, d::Int64; o=1)
            return a, b, c, d
        end
        @test U.method_args(O.Hedge, false) == (Symbol[:ξ, :reward_vector], Symbol[:η])
        @test U.method_args(O.ExpDecayedEpsilonGreedy, false) == (Symbol[:T, :τ, :ξ, :average_reward_per_arm_bandit], Symbol[:ϵ_start, :ϵ_end])
        @test U.method_args(O.EXP3, false) == (Symbol[:ξ, :reward_vector, :γ, :T, :τ], Symbol[:η])
        @test U.method_args(O.FtrlExponentiatedGradient, true) == (Symbol[:τ, :cumulative_reward_per_arm],  Vector{Symbol}())
        @test U.method_args(O.ExponentiatedGradient, true) == (Symbol[:ξ, :reward_vector],  Vector{Symbol}())
        @test U.method_args(testing_args, false) == (Symbol[:a, :b, :c, :d], Symbol[:o])
    end
end

end  # module