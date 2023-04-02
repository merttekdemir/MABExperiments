module MABStructs

    using Random, Distributions
    mutable struct MABStruct{DT<:Tuple{Vararg{Distribution}}}  # DT is the Type of the vector of distributions of the arms
        name::String  # Name of the algorithm
        T::Int64  # Total number of iterations
        A::DT  # Vector of Distributions per arm
        ξ::Categorical{Float64, Vector{Float64}}  # Mixed action, it is the output of all the algorithms and is updated iteration by iteration
        ξ_start::Categorical{Float64, Vector{Float64}}  # Placeholder for the initial configuration of ξ, needed
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
        best_dynamic_choice::Vector{Int64}
        cumulative_reward_dynamic::Vector{Float64}
        average_reward_dynamic::Vector{Float64}
        regret_dynamic::Vector{Float64}
        τ::Int64

        function MABStruct(T::Int64, A::DT, ξ::Categorical, name::String) where DT <: Tuple{Vararg{Distribution}}
            # Sanity Checks
            n_actions = length(A)
            n_actions == ncategories(ξ) || throw(ArgumentError("Error in construction"))
            
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
            best_dynamic_choice = zeros(Int64, T)
            best_dynamic_choice = zeros(Int64, T)
            cumulative_reward_dynamic = zeros(Float64, T)
            average_reward_dynamic = zeros(Float64, T)
            regret_dynamic = zeros(Float64, T)

            τ = 0

            return new{DT}(name, T, A, ξ, ξ_start, γ, reward_vector, 
                        choices_per_arm, algorithm_reward,
                        algorithm_cumulative_reward, sequence_of_rewards,
                        cumulative_reward_per_arm_bandit, cumulative_reward_per_arm, 
                        average_reward_per_arm, average_reward_per_arm_bandit,
                        best_fixed_choice, cumulative_reward_fixed, 
                        average_reward_fixed, regret_fixed, regret_pseudo, 
                        best_dynamic_choice, cumulative_reward_dynamic, 
                        average_reward_dynamic, regret_dynamic, τ
                        )
        end
    end
    MABStruct(T::Int64, A::Tuple, ξ::Categorical) = MABStruct(T, A, ξ, "Multi-Arm Bandit Experiment")

    Base.zero(::Type{MABStruct}, A::DT) where DT <: Tuple{Vararg{Distribution}} = MABStruct(0, A, Distributions.Categorical(length(A)))
    Base.zero(::MABStruct, A::DT) where DT <: Tuple{Vararg{Distribution}} = zero(MABStruct, A)

    function set_instance!(bandit::MABStruct, bandit_other::MABStruct)
        bandit.name = bandit_other.name
        bandit.ξ = bandit_other.ξ
        bandit.ξ_start = bandit_other.ξ_start
        bandit.τ = copy.(bandit_other.τ)
        bandit.γ = copy.(bandit_other.γ)
        bandit.reward_vector = copy.(bandit_other.reward_vector)
        bandit.choices_per_arm = copy.(bandit_other.choices_per_arm)
        bandit.algorithm_reward = copy.(bandit_other.algorithm_reward)
        bandit.algorithm_cumulative_reward = copy.(bandit_other.algorithm_cumulative_reward)
        bandit.sequence_of_rewards = copy.(bandit_other.sequence_of_rewards)
        bandit.cumulative_reward_per_arm_bandit = copy.(bandit_other.cumulative_reward_per_arm_bandit)
        bandit.cumulative_reward_per_arm = copy.(bandit_other.cumulative_reward_per_arm)
        bandit.average_reward_per_arm = copy.(bandit_other.average_reward_per_arm)
        bandit.average_reward_per_arm_bandit = copy.(bandit_other.average_reward_per_arm_bandit)
        bandit.best_fixed_choice = copy.(bandit_other.best_fixed_choice)
        bandit.cumulative_reward_fixed = copy.(bandit_other.cumulative_reward_fixed)
        bandit.average_reward_fixed = copy.(bandit_other.average_reward_fixed)
        bandit.regret_fixed = copy.(bandit_other.regret_fixed)
        bandit.regret_pseudo = copy.(bandit_other.regret_pseudo)
        #TODO define bdc and see if we update elementwise or vectorwise
        # bandit.best_dynamic_choice .= bdc()
        # bandit.cumulative_reward_dynamic = bandit.cumulative_reward_per_arm[bandit.best_fixed_choice]
        # bandit.average_reward_dynamic = zeros(Float64, T)
        # bandit.regret_dynamic = bandit_other.regret_dynamic
        return 
    end


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
        #TODO define bdc and see if we update elementwise or vectorwise
        # bandit.best_dynamic_choice .= bdc()
        # bandit.cumulative_reward_dynamic = bandit.cumulative_reward_per_arm[bandit.best_fixed_choice]
        # bandit.average_reward_dynamic = zeros(Float64, T)
        # bandit.regret_dynamic = zeros(Float64, T)
        return 
    end

    function pull!(bandit::MABStruct)
        bandit.reward_vector .= rand.(bandit.A)
        # if any(bandit.reward_vector .> 1.0)  # This control is not necessary, unless for specific applications
        #    throw(ArgumentError)
    end

    function run_step!(bandit::MABStruct)
        #Sample an action from the policy distribution
        action = rand(bandit.ξ)
        pull!(bandit)
        update_instance!(bandit, action)
    end

    #TODO fix specific printing for full info case and partial info case
    #TODO figure out if we should use IO
    #TODO figure out print vs println (base.print vs base.show)
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
        println(io, "best_dynamic_choice: $(bandit.best_dynamic_choice)")
        println(io, "cumulative_reward_dynamic: $(bandit.cumulative_reward_dynamic)")
        println(io, "average_reward_dynamic: $(bandit.average_reward_dynamic)")
        println(io, "regret_fixed: $(bandit.regret_fixed)")
        println(io, "regret_pseudo: $(bandit.regret_pseudo)")
        println(io, "regret_dynamic: $(bandit.regret_dynamic)")
    end

    function get_kw_list(bandit::MABStruct, argnames::Vector{Symbol}, default_argnames::Vector{Symbol}, default_values_algo::Dict{Any, Any})
        if :τ in argnames  # While all the others elements in kw_list work with pointers, hence get updated automatically, τ needs manual update
            τ_position = findfirst(item -> item==:τ, argnames)  # Store the position
            τ_update = true  # Flag the presence of τ
        else
            τ_update = false
            τ_position = nothing
        end
        
        if isempty(default_argnames) # TODO: Add check on quality of default_values
            return τ_update, τ_position, [getfield(bandit, argname) for argname in argnames], Dict()  # Get the fields in argnames from bandit
        else  # If default_argnames is not empty we also collect the values in default_values_algo
            return τ_update, τ_position, [getfield(bandit, argname) for argname in argnames], Dict([(default_argname, default_values_algo[string(default_argname)]) for default_argname in default_argnames])
        end
    end

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
            # println("sum")
            # println(sum(probs(bandit.ξ)) - 1.0)
            #@assert abs(sum(probs(bandit.ξ)) - 1.0) < 1e-5
            @assert sum(probs(bandit.ξ)) ≈ 1.0
        end
    end

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
        fill!(bandit.best_dynamic_choice, 0)
        fill!(bandit.cumulative_reward_dynamic, 0)
        fill!(bandit.average_reward_dynamic, 0)
        fill!(bandit.regret_dynamic, 0)
        fill!(bandit.regret_pseudo, 0)
        bandit.τ = 0
        return
    end

end  # module