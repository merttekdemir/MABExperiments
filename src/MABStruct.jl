module MABStructs

#---------------------------------------------------------------------------------------------------
"""
This file contains the main element of the project, the class of the Multi-Armed game.
The constructor is used to allocate the memory of each experiment before-hand.
The performance benefits greatly by the memory allocation, as the process does not need to be repeated at each iteration.
The functions used outside of the scope of the constructor are: run!, set_instance! and reset!, the others are auxiliary
function to the previous 3.
In the MABStruct object, the complete results of the experiments are store. These results are not all used in the main file
MABExperiment.jl, however they are available for further and deeper analysis. 
Most of the quantities stored are necessary for the well-functioning of the project.
"""
#---------------------------------------------------------------------------------------------------
    using Random, Distributions

"""
INNER CONSTRUCTOR
The following is a parametric mutable struct; the parameter is the distribution of the arms of the game under analysis.
Greek letters follow the notation of "ONLINE LEARNING WITH KNAPSACKS: THE BEST OF BOTH WORLDS" by 
Matteo Castiglioni, Politecnico di Milano
Andrea Celli, Bocconi University
Christian Kroer, Columbia University
...
### Arguments 
- `T::Int64`: Total number of iterations of the experiment.
- `A::DT`: Vector of distributions of the arms.
- `ξ::Discrete.Categorical: # Mixed action, output of the algorithms. Updated at each iteration. Discrete.Categorical is a type in Distributions.
- `name::String: Default to "Multi-Arm Bandit Experiment", can be avoided when constructing the bandit object.

### Attributes
-  ξ_start::Categorical{Float64, Vector{Float64}}: Placeholder for starting value of ξ. 
                                                    Necessary to correctly reset the game in "reset!".
-  γ::Vector{Int64}: Sequence of choices made by the algorithm under analysis. Plotted in "MABExperiments.jl"
-  reward_vector::Vector{Float64}: Last iteration of rewards, pulled in "pull!" from the arms distributions. 
                                    Needed to run the majority of the algorithms.
-  choices_per_arm::Vector{Int64}: Number of choices made for each arm. Counter of γ. Updating at each iteration is more efficient 
                                    than counting γ each time. Needed to run "UpperConfidenceBound" and "ExploreThenCommit"
-  algorithm_reward::Vector{Float64}: Reward of the algorithm iteration by iteration. Useful for plotting.
-  algorithm_cumulative_reward::Vector{Float64}: Cumulative reward of the algorithm up to the current iteration.
                                                    Calculated at each iteration for efficiency. Useful for plotting.
-  sequence_of_rewards::Vector{Vector{Float64}}: Each element stores the rewards of that specific iteration. In combination with
                                                    γ unequivocally defines the algorithm rewards. Useful for plotting.
-  cumulative_reward_per_arm::Vector{Float64}: Cumulative reward of the game, divided by arm.
                                                Simulates "observing" full information feedback.
                                                Useful for plotting and introducing new algorithms.
-  average_reward_per_arm::Vector{Float64}: Average reward of the game, divided by arm.
                                            Simulates "observing" full information feedback.
                                            Useful for plotting and introducing new algorithms.
-  cumulative_reward_per_arm_bandit::Vector{Float64}: Cumulative reward of the algorithm, divided by arm. 
                                                        Simulates "observing" only the bandit feedback.
                                                        Needed for "ExploreThenCommit" algorithm.
-  average_reward_per_arm_bandit::Vector{Float64}: Average reward of the algorithm, divided by arm. 
                                                    Simulates "observing" only the bandit feedback.
                                                    Needed for "EpsilonGreedy" algorithms.
-  best_fixed_choice::Vector{Int64}: Stores the best fixed arm in hindsight at each iteration of the game.
                                        Necessary to calculate the step by step regret and the final one.
-  cumulative_reward_fixed::Vector{Float64}: Stores the cumulative reward of the best fixed arm in hindsight.
                                                Useful for plotting and analysis.
-  average_reward_fixed::Vector{Float64}: Stores the cumulative reward of the best fixed arm in hindsight.
                                            Useful for plotting and analysis.
-  regret_fixed::Vector{Float64}: Regret of the best fixed arm in hindsight. 
                                    Necessary for plotting and analysis.
-  regret_pseudo::Vector{Float64}: Regret compared to the pseudo benchmark, used for the stochastic setting.
                                    Necessary for plotting.
-  τ::Int64: Current iteration of the game. Defaulted to 0 and updated in "update_instance!".
"""
    mutable struct MABStruct{DT<:Tuple{Vararg{Distribution}}}
        name::String
        T::Int64 
        A::DT
        ξ::Categorical{Float64, Vector{Float64}}  
        ξ_start::Categorical{Float64, Vector{Float64}}
        γ::Vector{Int64}
        reward_vector::Vector{Float64}
        choices_per_arm::Vector{Int64}
        algorithm_reward::Vector{Float64}
        algorithm_cumulative_reward::Vector{Float64}
        sequence_of_rewards::Vector{Vector{Float64}}
        cumulative_reward_per_arm_bandit::Vector{Float64}
        cumulative_reward_per_arm::Vector{Float64}
        average_reward_per_arm::Vector{Float64}
        average_reward_per_arm_bandit::Vector{Float64}
        best_fixed_choice::Vector{Int64}
        cumulative_reward_fixed::Vector{Float64}
        average_reward_fixed::Vector{Float64}
        regret_fixed::Vector{Float64}
        regret_pseudo::Vector{Float64}
        τ::Int64

        function MABStruct(T::Int64, A::DT, ξ::Categorical, name::String) where DT <: Tuple{Vararg{Distribution}}
            # Sanity Checks
            n_actions = length(A) 
            n_actions == ncategories(ξ) || throw(ArgumentError("Error in construction"))  # Must have same number of possible actions on both sides, the player and the game.
            
            # Initialised values
            ξ_start = Distributions.Categorical(copy(probs(ξ)))
            γ = zeros(Int64, T)
            reward_vector = zeros(Float64, n_actions)
            choices_per_arm = zeros(Int64, n_actions)
            algorithm_reward = zeros(Float64, T)
            algorithm_cumulative_reward = zeros(Float64, T)
            sequence_of_rewards = [zeros(Float64, n_actions) for _ in 1:T]
            cumulative_reward_per_arm_bandit = zeros(Float64, n_actions)
            cumulative_reward_per_arm = zeros(Float64, n_actions)
            average_reward_per_arm = zeros(Float64, n_actions)
            average_reward_per_arm_bandit = zeros(Float64, n_actions)
            best_fixed_choice = zeros(Int64, T)
            best_fixed_choice = zeros(Int64, T)
            cumulative_reward_fixed = zeros(Float64, T)
            average_reward_fixed = zeros(Float64, T)
            regret_fixed = zeros(Float64, T)
            regret_pseudo = zeros(Float64, T)

            τ = 0

            return new{DT}(name, T, A, ξ, ξ_start, γ, reward_vector, 
                        choices_per_arm, algorithm_reward,
                        algorithm_cumulative_reward, sequence_of_rewards,
                        cumulative_reward_per_arm_bandit, cumulative_reward_per_arm, 
                        average_reward_per_arm, average_reward_per_arm_bandit,
                        best_fixed_choice, cumulative_reward_fixed, 
                        average_reward_fixed, regret_fixed, regret_pseudo, τ
                        )
        end
    end

""" 
OUTER CONSTRUCTOR
Implements default name.
"""
    MABStruct(T::Int64, A::Tuple, ξ::Categorical) = MABStruct(T, A, ξ, "Multi-Arm Bandit Experiment")

#---------------------------------------------------------------------------------------------------

"""
Base.zero(::Type{MABStruct}, A::DT) where DT <: Tuple{Vararg{Distribution}}

Extends the zero method to MABStruct objects.

...
### Arguments 
- `::Type{MABStruct}`: Type of the object to define.
- `A::DT`: DT <: Tuple{Vararg{Distribution}}. Vector of distributions of arm. 
                                              Needed to build an object that can interact with the experiment ones.
"""
    Base.zero(::Type{MABStruct}, A::DT) where DT <: Tuple{Vararg{Distribution}} = MABStruct(0, A, Distributions.Categorical(length(A)))
    Base.zero(::MABStruct, A::DT) where DT <: Tuple{Vararg{Distribution}} = zero(MABStruct, A)

    function set_instance!(bandit::MABStruct, bandit_new::MABStruct)
        bandit.name = bandit_new.name
        bandit.ξ = bandit_new.ξ
        bandit.ξ_start = bandit_new.ξ_start
        bandit.τ = copy.(bandit_new.τ)
        bandit.γ = copy.(bandit_new.γ)
        bandit.reward_vector = copy.(bandit_new.reward_vector)
        bandit.choices_per_arm = copy.(bandit_new.choices_per_arm)
        bandit.algorithm_reward = copy.(bandit_new.algorithm_reward)
        bandit.algorithm_cumulative_reward = copy.(bandit_new.algorithm_cumulative_reward)
        bandit.sequence_of_rewards = copy.(bandit_new.sequence_of_rewards)
        bandit.cumulative_reward_per_arm_bandit = copy.(bandit_new.cumulative_reward_per_arm_bandit)
        bandit.cumulative_reward_per_arm = copy.(bandit_new.cumulative_reward_per_arm)
        bandit.average_reward_per_arm = copy.(bandit_new.average_reward_per_arm)
        bandit.average_reward_per_arm_bandit = copy.(bandit_new.average_reward_per_arm_bandit)
        bandit.best_fixed_choice = copy.(bandit_new.best_fixed_choice)
        bandit.cumulative_reward_fixed = copy.(bandit_new.cumulative_reward_fixed)
        bandit.average_reward_fixed = copy.(bandit_new.average_reward_fixed)
        bandit.regret_fixed = copy.(bandit_new.regret_fixed)
        bandit.regret_pseudo = copy.(bandit_new.regret_pseudo)
        return 
    end

#---------------------------------------------------------------------------------------------------

"""
update_instance!(bandit::MABStruct, action::Integer)

Updates the instance of the game.
Each element in the bandit game is updated efficiently by updating the value at the previous iteration.
No full reassignment is performed, so to save computation effort.

...
### Arguments 
- `bandit::MABStruct`: Game to be updated.
- `action::Integer`: The action drawn from the mixed action ξ.
"""

    function update_instance!(bandit::MABStruct, action::Integer)
        bandit.τ += 1
        i = bandit.τ
        bandit.γ[i] = action
        bandit.choices_per_arm[action] += 1
        bandit.algorithm_reward[i] = bandit.reward_vector[action]
        bandit.algorithm_cumulative_reward[i] = bandit.algorithm_cumulative_reward[max(i-1, 1)] + bandit.reward_vector[action]
        bandit.sequence_of_rewards[i] .= bandit.reward_vector
        bandit.cumulative_reward_per_arm_bandit[action] += bandit.reward_vector[action] 
        bandit.cumulative_reward_per_arm .+= bandit.reward_vector
        bandit.average_reward_per_arm .= bandit.cumulative_reward_per_arm ./ i
        bandit.average_reward_per_arm_bandit[action] = bandit.cumulative_reward_per_arm_bandit[action]/bandit.choices_per_arm[action] 
        bandit.best_fixed_choice[i] = argmax(bandit.cumulative_reward_per_arm)
        bandit.cumulative_reward_fixed[i] = bandit.cumulative_reward_per_arm[bandit.best_fixed_choice[i]]
        bandit.average_reward_fixed[i] += bandit.average_reward_per_arm[bandit.best_fixed_choice[i]]
        bandit.regret_fixed[i] = bandit.cumulative_reward_fixed[i] - bandit.algorithm_cumulative_reward[i]
        bandit.regret_pseudo[i] = (maximum(mean.(bandit.A)) - mean(bandit.A[bandit.γ[i]]))
        if i > 1 
            bandit.regret_pseudo[i] += bandit.regret_pseudo[i-1]
        end
        return 
    end

#---------------------------------------------------------------------------------------------------

"""
pull!(bandit::MABStruct)

Pulls the arms of the game, giving a reward for each arm, drawn from the assigned distribution.
The dot operations make it efficient in assignment. 
However, by profiling the code with "@profile", the computations may be demanding depending on the distributions chosen.

The commented control may be useful in applications with rewards between 0 and 1.
...
### Arguments 
- `bandit::MABStruct`: Game to pull the arms from.
"""

    function pull!(bandit::MABStruct)
        bandit.reward_vector .= rand.(bandit.A)
        # if any(bandit.reward_vector .> 1.0)  # This control is not necessary, unless for specific applications
        #    throw(ArgumentError)
    end

#---------------------------------------------------------------------------------------------------

"""
run_step!(bandit::MABStruct)

Draws the action played, pulls the rewards through "pull!" and runs the update on the game instance through "update_instance!"

...
### Arguments 
- `bandit::MABStruct`: Game to update.
"""
    function run_step!(bandit::MABStruct)
        # Sample an action from the policy distribution
        action = rand(bandit.ξ)

        # Pull the arms, update the reward_vector of the bandit instance
        pull!(bandit)
        update_instance!(bandit, action)
    end


#---------------------------------------------------------------------------------------------------

"""
Base.show(io::IO, bandit::MABStruct)

Prints an instance of the game for full information.

...
### Arguments 
- `bandit::MABStruct`: Game to print.
"""

    function Base.show(io::IO, bandit::MABStruct)
        println(io, "$(bandit.name): Iteration $(bandit.τ) of $(bandit.T)")
        println(io, "Full Information Case")
        println(io, "Policy: $(bandit.ξ)")
        println(io, "Choice vector: $(bandit.γ)")
        println(io, "Choices per arm: $(bandit.choices_per_arm)")
        println(io, "History of rewards: $(bandit.sequence_of_rewards)")
        println(io, "cumulative_reward_per_arm: $(bandit.cumulative_reward_per_arm)")
        println(io, "average_reward_per_arm: $(bandit.average_reward_per_arm)")
        println(io, "best_fixed_choice: $(bandit.best_fixed_choice)")
        println(io, "cumulative_reward_fixed: $(bandit.cumulative_reward_fixed)")
        println(io, "average_reward_fixed: $(bandit.average_reward_fixed)")
        println(io, "regret_fixed: $(bandit.regret_fixed)")
        println(io, "regret_pseudo: $(bandit.regret_pseudo)")
    end

#---------------------------------------------------------------------------------------------------

"""
get_kw_list(bandit::MABStruct, argnames::Vector{Symbol}, default_argnames::Vector{Symbol}, default_values_algo::Dict{Any, Any})

Auxiliary function used to automatically retrieve the kw values to pass to the "run!" function.
All the elements in kw_list are pointing to the current values in game.
This behavior is necessary for efficiency, it cut running time by half when implemented.
Indeed, being pointers, we do not need to recompute kw_list at each iteration of the game.

Checks on τ are therefore necessary, as pointer to integers are not present in Julia.
If τ belongs to the arguments of the algorithm, we flag it and update it manually in "run!"

The function mantains the correct order, so that the result can be splat when calling the algorithm.
E.g.: algorithm(kw_list...; default_kw_list...)
...
### Arguments 
- `bandit::MABStruct`: Game, used for accessing values of arguments.
- `argnames::Vector{Symbol}`: List of arguments of the algorithm.
- `default_argnames::Vector{Symbol}`: List of default arguments.
- `default_values_algo::Dict{Any, Any}`: Dict pairing default argument names to the values assigned by the user in the config, 
                                         for the specific algorithm.
"""

    function get_kw_list(bandit::MABStruct, argnames::Vector{Symbol}, default_argnames::Vector{Symbol}, default_values_algo::Dict{Any, Any})
        if :τ in argnames
            τ_position = findfirst(item -> item==:τ, argnames)  # Store the positionof τ
            τ_update = true  # Flag the presence of τ
        else
            τ_update = false
            τ_position = nothing
        end
        
        if isempty(default_argnames)
            return τ_update, τ_position, [getfield(bandit, argname) for argname in argnames], Dict()  # Get the fields in argnames from bandit
        else  # If default_argnames is not empty we also collect the values in default_values_algo
            return τ_update, τ_position, [getfield(bandit, argname) for argname in argnames], Dict([(default_argname, default_values_algo[string(default_argname)]) for default_argname in default_argnames])
        end
    end

#---------------------------------------------------------------------------------------------------

"""
run!(bandit::MABStruct, optimizer::Function, argnames::Vector{Symbol}, default_argnames::Vector{Symbol}, default_values_algo::Dict{Any, Any}; verbose=false::Bool)

Main function of the project. Automatically runs an experiment of the game instance, given the algorithm to test.

...
### Arguments 
- `bandit::MABStruct`: Game, used for accessing values of arguments.
- `optimizer::Function`: Algorithm used to compute the next mixed action ξ.
- `argnames::Vector{Symbol}`: List of arguments of the algorithm.
- `default_argnames::Vector{Symbol}`: List of default arguments.
- `default_values_algo::Dict{Any, Any}`: Dict pairing default argument names to the values assigned by the user in the config, 
                                         for the specific algorithm.
- `verbose=false::Bool`: Default. Set to True to print the game at each iteration.
"""

    function run!(bandit::MABStruct, optimizer::Function, argnames::Vector{Symbol}, default_argnames::Vector{Symbol}, default_values_algo::Dict{Any, Any}; verbose=false::Bool)
        if bandit.τ >= bandit.T
            println("Maximum iterations reached, reset the game or instantiate a new one")
            return nothing
        end
        
        τ_update, τ_position, kw_list, default_kw_dict = get_kw_list(bandit, argnames, default_argnames, default_values_algo)

        for i in 1:bandit.T
            # Manually update the τ argument in kw_list
            τ_update && (kw_list[τ_position] += 1)

            # Run a single step of the bandit game, to set the rewards and store all the information to run the optimizers
            run_step!(bandit)
            verbose && println(bandit)
            # Run the optimizers
            if isempty(default_kw_dict)
                probs(bandit.ξ) .= optimizer(kw_list...)  # The order is mantained by the construction
            else
                probs(bandit.ξ) .= optimizer(kw_list...; default_kw_dict...)
            end
            @assert sum(probs(bandit.ξ)) ≈ 1.0
        end
    end

#---------------------------------------------------------------------------------------------------

"""
reset!(bandit::MABStruct; name="MAB Experiment"::String)

Initialise the game to constructor conditions.

...
### Arguments 
- `bandit::MABStruct`: Game to reset.
- `name="MAB Experiment"::String`: Default. Name of the game after reset.
"""

    function reset!(bandit::MABStruct; name="MAB Experiment"::String)
        bandit.name = name
        bandit.ξ = Distributions.Categorical(copy(probs(bandit.ξ_start)))
        fill!(bandit.γ, 0)
        fill!(bandit.reward_vector, 0)
        fill!(bandit.choices_per_arm, 0)
        fill!(bandit.algorithm_reward, 0)
        fill!(bandit.algorithm_cumulative_reward, 0)
        foreach(x->fill!(x, 0), bandit.sequence_of_rewards)
        fill!(bandit.cumulative_reward_per_arm_bandit, 0)
        fill!(bandit.cumulative_reward_per_arm, 0)
        fill!(bandit.average_reward_per_arm, 0)
        fill!(bandit.average_reward_per_arm_bandit, 0)
        fill!(bandit.best_fixed_choice, 0)
        fill!(bandit.cumulative_reward_fixed, 0)
        fill!(bandit.average_reward_fixed, 0)
        fill!(bandit.regret_fixed, 0)
        fill!(bandit.regret_pseudo, 0)
        bandit.τ = 0
        return
    end

end  # module