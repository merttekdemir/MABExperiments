module MABStructs

    using Random, Distributions


    Random.seed!(42)

    mutable struct MABStruct{DT<:Tuple{Vararg{Distribution}}}
        name::String
        T::Int64
        A::DT
        ξ::Categorical
        γ::Vector{Int8}
        reward_vector::Vector{Float64}
        choices_per_arm::Vector{Int8}
        algorithm_reward::Vector{Float64}
        algorithm_cumulative_reward::Vector{Float64}
        sequence_of_rewards::Vector{Vector{Float64}}
        cumulative_reward_per_arm_bandit::Vector{Float64}
        cumulative_reward_per_arm::Vector{Float64}
        average_reward_per_arm::Vector{Float64}
        best_fixed_choice::Vector{Int8}
        cumulative_reward_fixed::Vector{Float64}
        average_reward_fixed::Vector{Float64}
        regret_fixed::Vector{Float64}
        best_dynamic_choice::Vector{Int8}
        cumulative_reward_dynamic::Vector{Float64}
        average_reward_dynamic::Vector{Float64}
        regret_dynamic::Vector{Float64}
        τ::Int64

        function MABStruct(T::Int64, A::DT, ξ::Categorical, name::String) where DT <: Tuple{Vararg{Distribution}}
            # Sanity Checks
            length(A) == ncategories(ξ) || throw(ArgumentError("Error in construction"))
            
            # Initialised values
            γ = zeros(Int8, T)
            reward_vector = zeros(Float64, length(A))
            choices_per_arm = zeros(Int64, length(A))
            algorithm_reward = zeros(Float64, T)
            algorithm_cumulative_reward = zeros(Float64, T)
            sequence_of_rewards = [zeros(Float64, length(A)) for _ in 1:T]
            cumulative_reward_per_arm_bandit = zeros(Float64, length(A))
            cumulative_reward_per_arm = zeros(Float64, length(A))
            average_reward_per_arm = zeros(Float64, length(A))
            best_fixed_choice = zeros(Int8, T)
            cumulative_reward_fixed = zeros(Float64, T)
            average_reward_fixed = zeros(Float64, T)
            regret_fixed = zeros(Float64, T)
            best_dynamic_choice = zeros(Int8, T)
            cumulative_reward_dynamic = zeros(Float64, T)
            average_reward_dynamic = zeros(Float64, T)
            regret_dynamic = zeros(Float64, T)

            τ = 0

            return new{DT}(name, T, A, ξ, γ, reward_vector, 
                        choices_per_arm, algorithm_reward,
                        algorithm_cumulative_reward, sequence_of_rewards,
                        cumulative_reward_per_arm_bandit, cumulative_reward_per_arm, 
                        average_reward_per_arm, best_fixed_choice, 
                        cumulative_reward_fixed, average_reward_fixed, 
                        regret_fixed, best_dynamic_choice, 
                        cumulative_reward_dynamic, average_reward_dynamic, 
                        regret_dynamic, τ
                        )
        end
    end
    MABStruct(T::Int64, A::Tuple, ξ::Categorical) = MABStruct(T, A, ξ, "Multi-Arm Bandit Experiment")

    Base.zero(::Type{MABStruct}, A::DT) where DT <: Tuple{Vararg{Distribution}} = MABStruct(0, A, Distributions.Categorical(length(A)))
    Base.zero(::MABStruct, A::DT) where DT <: Tuple{Vararg{Distribution}} = zero(MABStruct, A)

    function Base.zeros(::Type{MABStruct}, A::DT, dim::Int64) where DT <: Tuple{Vararg{Distribution}}
        return [zero(MABStruct, A) for _ in range(1, dim)]
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
        bandit.best_fixed_choice[i] = argmax(bandit.cumulative_reward_per_arm)
        bandit.cumulative_reward_fixed[i] = bandit.cumulative_reward_per_arm[bandit.best_fixed_choice[i]]
        bandit.average_reward_fixed[i] += bandit.average_reward_per_arm[bandit.best_fixed_choice[i]]
        bandit.regret_fixed[i] = bandit.cumulative_reward_fixed[i] - bandit.algorithm_cumulative_reward[i]
        #TODO define bdc and see if we update elementwise or vectorwise
        # bandit.best_dynamic_choice .= bdc()
        # bandit.cumulative_reward_dynamic = bandit.cumulative_reward_per_arm[bandit.best_fixed_choice]
        # bandit.average_reward_dynamic = zeros(Float64, T)
        # bandit.regret_dynamic = zeros(Float64, T)
        return 
    end

    function set_instance!(bandit::MABStruct, bandit_new::MABStruct) # T <: Tuple{Vararg{Distribution}}
        bandit.name = bandit_new.name
        bandit.ξ = bandit_new.ξ
        bandit.τ = bandit_new.τ
        bandit.γ = bandit_new.γ
        bandit.reward_vector = bandit_new.reward_vector
        bandit.choices_per_arm = bandit_new.choices_per_arm
        bandit.algorithm_reward = bandit_new.algorithm_reward
        bandit.algorithm_cumulative_reward = bandit_new.algorithm_cumulative_reward
        bandit.sequence_of_rewards = bandit_new.sequence_of_rewards
        bandit.cumulative_reward_per_arm_bandit = bandit_new.cumulative_reward_per_arm_bandit
        bandit.cumulative_reward_per_arm = bandit_new.cumulative_reward_per_arm
        bandit.average_reward_per_arm = bandit_new.average_reward_per_arm
        bandit.best_fixed_choice = bandit_new.best_fixed_choice
        bandit.cumulative_reward_fixed = bandit_new.cumulative_reward_fixed
        bandit.average_reward_fixed = bandit_new.average_reward_fixed
        bandit.regret_fixed = bandit_new.regret_fixed
        #TODO define bdc and see if we update elementwise or vectorwise
        # bandit.best_dynamic_choice .= bdc()
        # bandit.cumulative_reward_dynamic = bandit.cumulative_reward_per_arm[bandit.best_fixed_choice]
        # bandit.average_reward_dynamic = zeros(Float64, T)
        # bandit.regret_dynamic = bandit_new.regret_dynamic
        return 
    end

    function pull!(bandit::MABStruct)
        bandit.reward_vector = [rand(distrib) for distrib in bandit.A]
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
        println(io, "regret_dynamic: $(bandit.regret_dynamic)")
    end

    function update_kw_list(bandit::MABStruct, argnames::Vector{Symbol}) # T <: UnionAll{Float64, Vector{Float64}}
        return [getproperty(bandit, argname) for argname in argnames]
    end

    function run!(bandit::MABStruct, optimizer::Function, argnames::Vector{Symbol}, default_argnames::Vector{Symbol}, default_values_algo::Dict{Any, Any}; verbose=false::Bool)
        # println(bandit)
        
        if isempty(default_argnames) # TODO: Add check on quality of default_values
            kw_list, default_kw_dict = [getproperty(bandit, argname) for argname in argnames], Dict()
        else
            kw_list, default_kw_dict = [getproperty(bandit, argname) for argname in argnames], Dict([(default_argname, default_values_algo[default_argname]) for default_argname in default_argnames])
        end

        cache = nothing
        for _ in 1:bandit.T
            run_step!(bandit)
            verbose && println(bandit)
            kw_list = update_kw_list(bandit, argnames)
            # update policy
            if isempty(default_kw_dict)
                probs(bandit.ξ) .= optimizer(kw_list...)  # The order is mantained by the construction
            else
                probs(bandit.ξ) .= optimizer(kw_list...; default_kw_dict...)
            end
            print(sum(probs(bandit.ξ)) - 1.0)
            @assert abs(sum(probs(bandit.ξ)) - 1.0) < 1e-5
        end
        # println("Game Terminated")
    end

    function reset!(bandit::MABStruct, name::String)
        bandit.name = name
        fill!(bandit.γ, 0)
        fill!(bandit.reward_vector, 0)
        fill!(bandit.choices_per_arm, 0)
        fill!(bandit.algorithm_reward, 0)
        fill!(bandit.algorithm_cumulative_reward, 0)
        foreach(x->fill!(x, 0), bandit.sequence_of_rewards)
        fill!(bandit.cumulative_reward_per_arm_bandit, 0)
        fill!(bandit.cumulative_reward_per_arm, 0)
        fill!(bandit.average_reward_per_arm, 0)
        fill!(bandit.best_fixed_choice, 0)
        fill!(bandit.cumulative_reward_fixed, 0)
        fill!(bandit.average_reward_fixed, 0)
        fill!(bandit.regret_fixed, 0)
        fill!(bandit.best_dynamic_choice, 0)
        fill!(bandit.cumulative_reward_dynamic, 0)
        fill!(bandit.average_reward_dynamic, 0)
        fill!(bandit.regret_dynamic, 0)
        bandit.τ = 0
        return
    end

end  # module