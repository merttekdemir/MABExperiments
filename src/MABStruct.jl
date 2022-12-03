module MABStructs

    using Random, Distributions
    mutable struct MABStruct{DT<:Tuple{Vararg{Distribution}}}
        name::String
        T::Int64
        A::DT
        ξ::Categorical
        γ::Vector{Int8}
        sequence_of_rewards::Vector{Vector{Float64}}
        cumulative_reward_per_arm::Vector{Float64}
        average_reward_per_arm::Vector{Float64}
        best_fixed_choice::Vector{Int8}
        cumulative_reward_fixed::Vector{Float64}
        average_reward_fixed::Vector{Float64}
        best_dynamic_choice::Vector{Int8}
        cumulative_reward_dynamic::Vector{Float64}
        average_reward_dynamic::Vector{Float64}
        τ::Int64

        function MABStruct(T::Int64, A::DT, ξ::Categorical, name::String) where DT <: Tuple{Vararg{Distribution}}
            # Sanity Checks
            length(A) == ncategories(ξ) || throw(ArgumentError("Error in construction"))
            
            # Initialised values
            γ = zeros(Int8, T)
            sequence_of_rewards = [zeros(Float64, length(A)) for _ in 1:T]
            cumulative_reward_per_arm = zeros(Float64, length(A))
            average_reward_per_arm = zeros(Float64, length(A))
            best_fixed_choice = zeros(Int8, T)
            cumulative_reward_fixed = zeros(Float64, T)
            average_reward_fixed = zeros(Float64, T)
            best_dynamic_choice = zeros(Int8, T)
            cumulative_reward_dynamic = zeros(Float64, T)
            average_reward_dynamic = zeros(Float64, T)
            τ = 0

            return new{DT}(name, T, A, ξ, γ, sequence_of_rewards,
                        cumulative_reward_per_arm, average_reward_per_arm,
                        best_fixed_choice, cumulative_reward_fixed, average_reward_fixed, 
                        best_dynamic_choice, cumulative_reward_dynamic,
                        average_reward_dynamic, τ)
        end
    end

    MABStruct(T::Int64, A::DT, ξ::Categorical) where DT <: Tuple{Vararg{Distribution}} = MABStruct(T, A, ξ, "Multi-Arm Bandit Experiment") 

    function update_instance!(bandit::MABStruct, action::Integer, reward_vector::Vector)
        bandit.τ += 1
        i = bandit.τ
        bandit.γ[i] = action
        bandit.sequence_of_rewards[i] .= reward_vector
        bandit.cumulative_reward_per_arm .+= reward_vector
        bandit.average_reward_per_arm .= bandit.cumulative_reward_per_arm ./ i
        bandit.best_fixed_choice[i] = argmax(bandit.cumulative_reward_per_arm)
        bandit.cumulative_reward_fixed[i] = bandit.cumulative_reward_per_arm[bandit.best_fixed_choice[i]]
        bandit.average_reward_fixed[i] = bandit.average_reward_per_arm[bandit.best_fixed_choice[i]]
        #TODO define bdc and see if we update elementwise or vectorwise
        # bandit.best_dynamic_choice .= bdc()
        # bandit.cumulative_reward_dynamic = bandit.cumulative_reward_per_arm[bandit.best_fixed_choice]
        # bandit.average_reward_dynamic = zeros(Float64, T)
        return 
    end

    function pull(bandit::MABStruct)
        return [rand(distrib) for distrib in bandit.A]
    end

    function run_step!(bandit::MABStruct)
        #Sample an action from the policy distribution
        action = rand(bandit.ξ)
        reward_vector = pull(bandit)
        update_instance!(bandit, action, reward_vector)
    end

    #TODO fix specific printing for full info case and partial info case
    #TODO figure out if we should use IO
    #TODO figure out print vs println (base.print vs base.show)
    function Base.show(io::IO, bandit::MABStruct)
        println(io, "$(bandit.name): Iteration $(bandit.τ) of $(bandit.T)")
        println(io, "Full Information Case")
        println(io, "Policy: $(bandit.γ)")
        println(io, "History of rewards: $(bandit.sequence_of_rewards)")
        println(io, "cumulative_reward_per_arm: $(bandit.cumulative_reward_per_arm)")
        println(io, "average_reward_per_arm: $(bandit.average_reward_per_arm)")
        println(io, "best_fixed_choice: $(bandit.best_fixed_choice)")
        println(io, "cumulative_reward_fixed: $(bandit.cumulative_reward_fixed)")
        println(io, "average_reward_fixed: $(bandit.average_reward_fixed)")
        println(io, "best_dynamic_choice: $(bandit.best_dynamic_choice)")
        println(io, "cumulative_reward_dynamic: $(bandit.cumulative_reward_dynamic)")
        println(io, "average_reward_dynamic: $(bandit.average_reward_dynamic)")
    end

    #TODO does it make sense to use a kw_dict
    function run!(bandit::MABStruct, optimizer::Function, verbose=false::Bool; kw_dict::Dict)
        # println(bandit)
        for τ in 1:bandit.T
            run_step!(bandit)
            verbose && println(bandit)
            #TODO update policy
            probs(bandit.ξ) .= optimizer(bandit.ξ, bandit.sequence_of_rewards[τ]; kw_dict...)
            println(sum(probs(bandit.ξ)))
        end
        println("Game Terminated")
    end

    function reset!(bandit::MABStruct)
        fill!(bandit.γ, 0)
        foreach(x->fill!(x, 0), bandit.sequence_of_rewards)
        fill!(bandit.cumulative_reward_per_arm, 0)
        fill!(bandit.average_reward_per_arm, 0)
        fill!(bandit.best_fixed_choice, 0)
        fill!(bandit.cumulative_reward_fixed, 0)
        fill!(bandit.average_reward_fixed, 0)
        fill!(bandit.best_dynamic_choice, 0)
        fill!(bandit.cumulative_reward_dynamic, 0)
        fill!(bandit.average_reward_dynamic, 0)
        bandit.τ = 0
        return
    end

end  # module