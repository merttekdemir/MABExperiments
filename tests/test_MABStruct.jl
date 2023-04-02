module UnitTests_MABStruct
using Test, Distributions, Random

include(joinpath("..", "src", "MABStruct.jl")); M = MABStructs;
include(joinpath("..", "src", "OnlineLearningAlgorithms.jl")); O = OnlineLearningAlgorithms;


"""Initialise values"""

T = 10
A = (Beta(0.15, 0.7), Beta(0.54, 0.2), Beta(0.38, 0.5))
A_wrong_type_1 = [Beta(0.15, 0.7), Beta(0.54, 0.2), Beta(0.38, 0.5)]
A_wrong_type_2 = Dict([("arm1", Beta(0.15, 0.7)), ("arm2", Beta(0.54, 0.2)), ("arm3", Beta(0.38, 0.5))])
A_wrong_content = (10, 11, 14)
# A_wrong_distributions = [Normal(0.15, 0.7), Normal(0.54, 0.2), Normal(0.38, 0.5)]
ξ_wrong_length = Categorical([1/4, 1/4, 1/4, 1/4])
ξ = Categorical([1/3, 1/3, 1/3])
name_wrong = 10
verb = true
seed = 42

"""Define test functions"""

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
    # Test that the type of all the elements is as expected after the construction
    @test game.ξ_start isa Distributions.Categorical
    @test game.γ isa Vector{Int64} 
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
    @test game.best_fixed_choice isa Vector{Int64} 
    @test game.cumulative_reward_fixed isa Vector{Float64} 
    @test game.average_reward_fixed isa Vector{Float64} 
    @test game.regret_fixed isa Vector{Float64}
    @test game.best_dynamic_choice isa Vector{Int64} 
    @test game.cumulative_reward_dynamic isa Vector{Float64} 
    @test game.average_reward_dynamic isa Vector{Float64} 
    @test game.regret_dynamic isa Vector{Float64}
end


function test_struct_length(game)
    # Test that the length of all the elements is as expected after the construction
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


function test_update!(game_manual, action)
    # Initialise a game with the same configuration as the argument
    game_stored = M.MABStruct(game_manual.T, game_manual.A, game_manual.ξ)

    # Set all the instances to the argument attributes
    game_stored.τ = game_manual.τ
    game_stored.γ = copy(game_manual.γ)
    game_stored.reward_vector = copy(game_manual.reward_vector)
    game_stored.choices_per_arm = copy(game_manual.choices_per_arm)
    game_stored.algorithm_reward = copy(game_manual.algorithm_reward)
    game_stored.algorithm_cumulative_reward = copy(game_manual.algorithm_cumulative_reward)
    game_stored.sequence_of_rewards = copy(game_manual.sequence_of_rewards)
    game_stored.cumulative_reward_per_arm_bandit = copy(game_manual.cumulative_reward_per_arm_bandit)
    game_stored.cumulative_reward_per_arm = copy(game_manual.cumulative_reward_per_arm)
    game_stored.average_reward_per_arm = copy(game_manual.average_reward_per_arm)
    game_stored.average_reward_per_arm_bandit = copy(game_manual.average_reward_per_arm_bandit)
    game_stored.best_fixed_choice = copy(game_manual.best_fixed_choice)
    game_stored.cumulative_reward_fixed = copy(game_manual.cumulative_reward_fixed)
    game_stored.average_reward_fixed = copy(game_manual.average_reward_fixed)
    game_stored.regret_fixed = copy(game_manual.regret_fixed)

    # Test the type functioning of update_instance!
    @test action <= length(game_manual.A)
    @test_throws MethodError M.update_instance!(game_stored, 0.1)
    @test_throws MethodError M.update_instance!(game_stored, [1, 2])

    # Perform an automatic update on the instantiated game
    M.update_instance!(game_stored, action)

    # Perform a manual update on the argument game
    game_manual.τ += 1
    i = game_manual.τ
    game_manual.γ[i] = action
    game_manual.choices_per_arm[action] += 1
    game_manual.algorithm_reward[i] = game_manual.reward_vector[action]
    game_manual.algorithm_cumulative_reward[i] = game_manual.algorithm_cumulative_reward[max(i-1, 1)] + game_manual.reward_vector[action]
    game_manual.sequence_of_rewards[i] .= game_manual.reward_vector
    game_manual.cumulative_reward_per_arm_bandit[action] += game_manual.reward_vector[action] 
    game_manual.cumulative_reward_per_arm .+= game_manual.reward_vector
    game_manual.average_reward_per_arm .= game_manual.cumulative_reward_per_arm ./ i
    game_manual.average_reward_per_arm_bandit[action] = game_manual.cumulative_reward_per_arm_bandit[action]/game_manual.choices_per_arm[action] 
    game_manual.best_fixed_choice[i] = argmax(game_manual.cumulative_reward_per_arm)
    game_manual.cumulative_reward_fixed[i] = game_manual.cumulative_reward_per_arm[game_manual.best_fixed_choice[i]]
    game_manual.average_reward_fixed[i] += game_manual.average_reward_per_arm[game_manual.best_fixed_choice[i]]
    game_manual.regret_fixed[i] = game_manual.cumulative_reward_fixed[i] - game_manual.algorithm_cumulative_reward[i]

    # Test for correctness of the automatic update
    @test game_manual.τ == game_stored.τ
    @test all(game_manual.γ .== game_stored.γ)
    @test all(game_manual.choices_per_arm .== game_stored.choices_per_arm)
    @test all(game_manual.algorithm_reward .== game_stored.algorithm_reward)
    @test all(game_manual.algorithm_cumulative_reward .== game_stored.algorithm_cumulative_reward)
    @test all(all.([game_manual.sequence_of_rewards[j] .== game_stored.sequence_of_rewards[j] for j in 1:length(game_manual.sequence_of_rewards)]))
    @test all(game_manual.cumulative_reward_per_arm_bandit .== game_stored.cumulative_reward_per_arm_bandit)
    @test all(game_manual.cumulative_reward_per_arm .== game_stored.cumulative_reward_per_arm)
    @test all(game_manual.average_reward_per_arm .== game_stored.average_reward_per_arm)
    @test all(game_manual.average_reward_per_arm_bandit .== game_stored.average_reward_per_arm_bandit)
    @test all(game_manual.best_fixed_choice .== game_stored.best_fixed_choice)
    @test all(game_manual.cumulative_reward_fixed .== game_stored.cumulative_reward_fixed)
    @test all(game_manual.average_reward_fixed .== game_stored.average_reward_fixed)
    @test all(game_manual.regret_fixed .== game_stored.regret_fixed)
end


function test_runstep(game_manual, game_automatic; seed=42)
    # Set the seed to replicate manual and automatic results
    Random.seed!(seed)

    # Perform a manual step of run_step! function
    action = rand(game_manual.ξ)
    M.pull!(game_manual)
    M.update_instance!(game_manual, action)

    # Set the seed to match the one used for the manual iteration
    Random.seed!(seed)

    # Run one step of run_step! on the automatic game
    M.run_step!(game_automatic)

    # Test that the two fundamental elements, action and rewards, are the same. All the others follow from correctness of the update function
    @test all(game_manual.γ .== game_automatic.γ)
    @test all(abs.(game_manual.reward_vector ≈ game_automatic.reward_vector))
end


function test_get_kw_list(game)
    # Set the parameters
    algorithm = O.EXP3
    default_values = Dict("EXP3" => Dict{Symbol, Any}([(:η, 0.3)]))
    default_values_algo = default_values[string(algorithm)]
    argnames = [:ξ, :reward_vector, :γ, :T, :τ]
    default_argnames = [:η]

    # Check functioning of get_kw_list when default is empty
    kw_list = [game.ξ, game.reward_vector, game.γ, game.T, game.τ]
    default_kw_dict = Dict(:η => 0.3)
    τ_update, τ_position, kw_list_automatic, default_kw_dict_automatic = M.get_kw_list(game, argnames, Symbol[], default_values_algo)

    @test τ_update == true 
    @test τ_position == 5
    @test kw_list == kw_list_automatic
    @test isempty(default_kw_dict_automatic)

    # Check functioning of get_kw_list when default is not empty
    kw_list_no_τ = [game.ξ, game.reward_vector, game.γ, game.T]
    τ_update, τ_position, kw_list_automatic, default_kw_dict_automatic = M.get_kw_list(game, argnames[1:4], default_argnames, default_values_algo)
    
    @test τ_update == false 
    @test τ_position === nothing
    @test all(kw_list_no_τ .== kw_list_automatic)
    @test default_kw_dict[:η] == default_kw_dict_automatic[:η]
end


function test_behavior_update_kw_list(game)
    # Set Vector argnames and obtain initial kw_list
    argnames = [:γ, :reward_vector]
    algorithm = O.EXP3
    default_values = Dict("EXP3" => Dict{Symbol, Any}([(:η, 0.3)]))
    default_values_algo = default_values[string(algorithm)]
    _, _, kw_list, _ = M.get_kw_list(game, argnames, Symbol[], default_values_algo)

    # define a kw_list which does not update with the update in game
    kw_list_unrelated = [copy(getfield(game, argname)) for argname in argnames]  # copy does not point at the attributes

    # Update some game attributes
    M.fill!(game.γ, 1)
    M.fill!(game.reward_vector, 0.5)

    # Test that the new kw_list has been updated correctly
    @test all(kw_list_unrelated .!= kw_list)
    @test all(kw_list[1] .== 1)
    @test all(kw_list[2] .== 0.5)
end


function test_run(game)
    algorithm = O.EXP3
    default_values = Dict("EXP3" => Dict{Symbol, Any}([(:η, 0.3)]))
    default_values_algo = default_values[string(algorithm)]
    argnames = [:ξ, :reward_vector, :γ, :T, :τ]
    default_argnames = [:η]

    # Test the checks on τ
    game.τ = game.T
    @test M.run!(game, algorithm, argnames, default_argnames, default_values_algo) === nothing
    game.τ = 0

    # Run few iterations with a degenerate case
    T_degenerate = 100
    A_degenerate = (Distributions.Bernoulli(0), Distributions.Bernoulli(1), Distributions.Bernoulli(0))
    game_degenerate = M.MABStruct(T_degenerate, A_degenerate, game.ξ)
    M.run!(game_degenerate, algorithm, argnames, Symbol[], default_values_algo)

    game_degenerate_2 = M.MABStruct(T_degenerate, A_degenerate, game.ξ)
    M.run!(game_degenerate_2, algorithm, argnames, default_argnames, default_values_algo)

    # Veirify that the run is updating the vectors and the algorithm is working
    @test argmax(game_degenerate.choices_per_arm) == 2
    @test any(game_degenerate.reward_vector .== 1.0) && any(game_degenerate_2.reward_vector .== 1.0)
    @test all(any.([game_degenerate.sequence_of_rewards[j] .== 1.0 for j in 1:game_degenerate.T])) 
    @test all(any.([game_degenerate_2.sequence_of_rewards[j] .== 1.0 for j in 1:game_degenerate.T]))

end


function test_reset_content!(game::M.MABStruct)
    # Reset the game
    M.reset!(game)

    # Initialise a new game for comparison
    game_new = M.MABStruct(game.T, game.A, game.ξ)

    # Verify that all the attributes are reset to initialisation
    @test all(probs(game.ξ_start) .≈ probs(game.ξ))
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
    # instantiate a new game to test the set_instance
    game_new = M.MABStruct(game.T, game.A, game.ξ)

    # Modify the argument game
    game.τ = 10
    game.ξ_start = Distributions.Categorical([0.0, 1.0, 0.0])
    fill!(game.reward_vector, 1.0)
    fill!(game.choices_per_arm, 2)
    fill!(game.algorithm_reward, 10.0)
    fill!(game.algorithm_cumulative_reward, 1.0)
    game.cumulative_reward_per_arm_bandit[3] = 1.0
    fill!(game.cumulative_reward_per_arm, 1.0)
    fill!(game.average_reward_per_arm, 1.0)
    fill!(game.average_reward_per_arm_bandit, 2.0)
    fill!(game.best_fixed_choice, 2)
    fill!(game.cumulative_reward_fixed, 2.0)
    fill!(game.average_reward_fixed, 3.0)
    fill!(game.regret_fixed, 3.0)
    fill!(game.best_dynamic_choice, 2)
    fill!(game.cumulative_reward_dynamic, 3.0)
    fill!(game.average_reward_dynamic, 3.0)
    fill!(game.regret_dynamic, 3.0)

    # Set the new game to our argument game
    M.set_instance!(game_new, game)

    # Verify that all the attributes correspond after applyin the function
    @test game.τ == game_new.τ
    @test game.ξ_start == game_new.ξ_start
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
end


"""Startign tests"""

@testset verbose = verb "MABStruct tests" begin
    @testset "Test object_creation" begin
        test_struct_wrong_arguments(T, A, A_wrong_type_1, A_wrong_type_2, A_wrong_content, ξ_wrong_length, ξ, name_wrong)
        game = M.MABStruct(T, A, ξ)
        test_struct_type(game)
        test_struct_length(game)
    end
    @testset "Test set_instance!" begin
        game = M.MABStruct(10, A, ξ)
        test_set_instance(game)
    end
    @testset "Test reset!" begin
        game = M.MABStruct(10, A, ξ)
        test_reset_content!(game)
        test_struct_type(game)
        test_struct_length(game)
    end
    @testset "Test run_step!" begin
        game = M.MABStruct(T, A, ξ)
        action = rand(game.ξ)
        @test action <= length(game.A)
        @testset "Test pull!" begin
            test_pull!(game)
        end
        @testset "Test update_instance!"  begin
            test_update!(game, action)
        end
        game_manual = M.MABStruct(T, A, ξ)
        game_automatic = M.MABStruct(T, A, ξ)
        test_runstep(game_manual, game_automatic; seed=seed)
    end
    @testset "Test run!" begin
        game = M.MABStruct(T, A, ξ)
        test_get_kw_list(game)
        test_behavior_update_kw_list(game)
        test_run(game)
    end
end


end # module