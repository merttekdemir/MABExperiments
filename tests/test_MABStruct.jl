module UnitTests
using Test, Distributions, Random

include(joinpath("..", "src", "MABStruct.jl")); M = MABStructs;


### Initialise values ###

T = 10
A = (Beta(0.15, 0.7), Beta(0.54, 0.2), Beta(0.38, 0.5))
A_wrong_type_1 = [Beta(0.15, 0.7), Beta(0.54, 0.2), Beta(0.38, 0.5)]
A_wrong_type_2 = Dict([("arm1", Beta(0.15, 0.7)), ("arm2", Beta(0.54, 0.2)), ("arm3", Beta(0.38, 0.5))])
A_wrong_content = (10, 11, 14)
# A_wrong_distributions = [Normal(0.15, 0.7), Normal(0.54, 0.2), Normal(0.38, 0.5)]
ξ_wrong_length = Categorical([1/4, 1/4, 1/4, 1/4])
ξ = Categorical([1/3, 1/3, 1/3])
name_wrong = 10
seed = 42


### Define test functions ###

function test_struct_wrong_arguments(T, A, A_wrong_type_1, A_wrong_type_2, A_wrong_content, ξ_wrong_length, ξ, name_wrong)
    # Wrong value of number of iteration 
    @test_throws ArgumentError M.MABStruct(-10, A, ξ, "test_wrong_T")

    # Wrong type for A
    @test_throws MethodError M.MABStruct(T, A_wrong_type_1, ξ, "test_wrong_type_A_1")
    @test_throws MethodError M.MABStruct(T, A_wrong_type_2, ξ, "test_wrong_type_A_2")

    # Wrong content for A 
    @test_throws MethodError M.MABStruct(T, A_wrong_content, ξ, "test_wrong_content_A")

    # Length mismatch A and ξ
    @test_throws ArgumentError M.MABStruct(T, A, ξ_wrong_length, "test_length_mismatch_A_ξ")

    # Invalid input name
    @test_throws MethodError M.MABStruct(T, A, ξ_wrong_length, name_wrong)
end


function test_struct_type(game)
    @test game.γ isa Vector{Int8} 
    @test game.reward_vector isa Vector{Float64}
    @test game.choices_per_arm isa Vector{Int64}
    @test game.algorithm_reward isa Vector{Float64} 
    @test game.algorithm_cumulative_reward isa Vector{Float64}
    @test game.sequence_of_rewards isa Vector{Vector{Float64}} 
    @test eltype(game.sequence_of_rewards) == Vector{Float64}
    @test game.cumulative_reward_per_arm_bandit isa Vector{Float64}
    @test game.cumulative_reward_per_arm isa Vector{Float64} 
    @test game.average_reward_per_arm isa Vector{Float64} 
    @test game.average_reward_per_arm_bandit isa Vector{Float64}
    @test game.best_fixed_choice isa Vector{Int8} 
    @test game.cumulative_reward_fixed isa Vector{Float64} 
    @test game.average_reward_fixed isa Vector{Float64} 
    @test game.regret_fixed isa Vector{Float64}
    @test game.best_dynamic_choice isa Vector{Int8} 
    @test game.cumulative_reward_dynamic isa Vector{Float64} 
    @test game.average_reward_dynamic isa Vector{Float64} 
    @test game.regret_dynamic isa Vector{Float64}
end


function test_struct_length(game)
    n_arms = length(game.A)
    @test length(game.γ) == game.T 
    @test length(game.reward_vector) == n_arms
    @test length(game.choices_per_arm) == n_arms
    @test length(game.algorithm_reward) == game.T 
    @test length(game.algorithm_cumulative_reward) == game.T 
    @test length(game.sequence_of_rewards) == game.T
    @test all(length.(game.sequence_of_rewards) .== n_arms)  # Test that each reward vector inside the sequence has n_arms components
    @test length(game.cumulative_reward_per_arm_bandit) == n_arms
    @test length(game.cumulative_reward_per_arm) == n_arms 
    @test length(game.average_reward_per_arm) == n_arms 
    @test length(game.average_reward_per_arm_bandit) == n_arms
    @test length(game.best_fixed_choice) == game.T
    @test length(game.cumulative_reward_fixed) == game.T
    @test length(game.average_reward_fixed) == game.T
    @test length(game.regret_fixed) == game.T
    @test length(game.best_dynamic_choice) == game.T
    @test length(game.cumulative_reward_dynamic) == game.T
    @test length(game.average_reward_dynamic) == game.T 
    @test length(game.regret_dynamic) == game.T 
end


function test_pull!(game)
    # Save initial reward vector
    rew = copy(game.reward_vector)

    # Pull the arms once
    M.pull!(game)

    # Verify the vector of rewards mantained its length and type
    @test length(game.reward_vector) == length(game.A)
    @test eltype(game.reward_vector) == Float64

    # Verify the values do not exceed one and have been updated
    # @test all(game.reward_vector .- 1.0 .< 0)  # This test is not necessary in general, it must be uncommented for specific applications
    @test all(game.reward_vector .!= rew)    
end


function test_update(game, action)
    # Set reward_vector to 1 in one action and be sure the claculations are correct in all the update_instance
    # Set the action to a wrong one and to a good one to check it works in both cases
end


function test_runstep(game_manual, game_automatic; seed=42)
    # Set the seed to match the one used for the manual iteration
    Random.seed!(seed)

    # Run one step
    M.run_step!(game_automatic)

    # Test that the two fundamental elements, action and rewards, are the same. All the others follow from correctness of the update function
    println(game_manual.γ)
    println(game_automatic.γ)
    @test all(game_manual.γ == game_automatic.γ)
    @test all(abs.(game_manual.reward_vector .- game_automatic.reward_vector) .< 1e-5)
end


function test_run(game)
    # ok
end


function test_reset_content!(game::M.MABStruct)
    # Reset the game
    M.reset!(game)

    # Initialise a new game for comparison
    game_new = M.MABStruct(game.T, game.A, game.ξ)

    # Verify that all the attributes are reset to initialisation
    @test game.τ == game_new.τ
    @test all(game.γ .== game_new.γ)
    @test all(game.reward_vector .== game_new.reward_vector)
    @test all(game.choices_per_arm .== game_new.choices_per_arm)
    @test all(game.algorithm_reward .== game_new.algorithm_reward)
    @test all(game.algorithm_cumulative_reward .== game_new.algorithm_cumulative_reward)
    @test all([all(game.sequence_of_rewards[i] .== game_new.sequence_of_rewards[i]) for i in 1:length(game.sequence_of_rewards)])
    @test all(game.cumulative_reward_per_arm_bandit .== game_new.cumulative_reward_per_arm_bandit)
    @test all(game.cumulative_reward_per_arm .== game_new.cumulative_reward_per_arm)
    @test all(game.average_reward_per_arm .== game_new.average_reward_per_arm)
    @test all(game.average_reward_per_arm_bandit .== game_new.average_reward_per_arm_bandit)
    @test all(game.best_fixed_choice .== game_new.best_fixed_choice)
    @test all(game.cumulative_reward_fixed .== game_new.cumulative_reward_fixed)
    @test all(game.average_reward_fixed .== game_new.average_reward_fixed)
    @test all(game.regret_fixed .== game_new.regret_fixed)
    @test all(game.best_dynamic_choice .== game_new.best_dynamic_choice)
    @test all(game.cumulative_reward_dynamic .== game_new.cumulative_reward_dynamic)
    @test all(game.average_reward_dynamic .== game_new.average_reward_dynamic)
    @test all(game.regret_dynamic .== game_new.regret_dynamic)
end


function test_set_instance(game)
    # ok
end


function test_update_kw_list(game)
    # Set Vector argnames and obtain initial kw_list
    argnames = [:γ, :T, :reward_vector]
    kw_list = [getfield(game, argname) for argname in argnames]
    kw_list_old = [copy(getfield(game, argname)) for argname in argnames]
    println(kw_list_old)
    # Update some game attributes
    M.fill!(game.γ, 1)
    game.T = 1000
    M.fill!(game.reward_vector, 0.5)

    # Retrieve the new kw_list
    kw_list_new = M.update_kw_list(game, argnames)
    println(kw_list_new)
    print(kw_list_old)
    print(kw_list)
    # Test that the new kw_list has been updated correctly
    @test all(kw_list_old .!= kw_list_new)
    @test all(kw_list_new[1] .== 1)
    @test kw_list_new[2] == 1000
    @test all(kw_list_new[3] .== 0.5)
end

### STARTING TESTS ###
@testset verbose = true "MABStruct tests" begin
    @testset "Test object_creation" begin
        # @inferred - this tests if the compiler predicted the same type as the returned one, so we can check for performance, u can check that with @code_warntype
        test_struct_wrong_arguments(T, A, A_wrong_type_1, A_wrong_type_2, A_wrong_content, ξ_wrong_length, ξ, name_wrong)
        game = M.MABStruct(T, A, ξ)
        test_struct_type(game)
        test_struct_length(game)
    end
    @testset "Test run_step!" begin
        game_manual = M.MABStruct(T, A, ξ)
        game_automatic = M.MABStruct(T, A, ξ)
        Random.seed!(seed)
        action = rand(game_manual.ξ)
        @test action <= length(game_manual.A)
        @testset "Test pull!" begin
            test_pull!(game_manual)
        end
        @testset "Test update_instance!"  begin
            game = M.MABStruct(10, A, ξ)
            test_update(game, action)
        end
        test_runstep(game_manual, game_automatic; seed=seed)
    end
    @testset "Test run!" begin
        @test true
        @test true
    end
    @testset "Test reset!" begin
        game = M.MABStruct(10, A, ξ)
        test_reset_content!(game)
        test_struct_type(game)
        test_struct_length(game)
    end
    @testset "Test set_instance!" begin
        game = M.MABStruct(10, A, ξ)
        test_set_instance(game)
    end
    # @testset "Test update_kw_list" begin
    #     game = M.MABStruct(10, A, ξ)
    #     test_update_kw_list(game)
    # end
end

# Online Learning Algorithms tests
@testset verbose = true "OnlineLearningAlgorithms tests" begin

end

# MAB Experiments tests
@testset verbose = true "MABExperiments tests" begin
    
end

tests = ["MABStruct tests", "OnlineLearningAlgorithms tests", "MABExperiments tests"]

end # module