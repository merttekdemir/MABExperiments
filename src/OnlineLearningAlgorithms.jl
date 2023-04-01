module OnlineLearningAlgorithms
using Random, Distributions, DataStructures


    function ExponentiatedGradient(ξ::Categorical{Float64, Vector{Float64}}, reward_vector::Vector{Float64}; η=0.01::Float64)
        loss = -1 .* reward_vector
        #TODO ask CB if it makes sense to use expm1
        updated_probs = probs(ξ) .* exp.(-η.*loss)
        return updated_probs ./ sum(updated_probs)
        # TODO: Fix whether we should return the probs or the DiscreteCategorical
    end

    function FtrlExponentiatedGradient(τ::Int64, cumulative_reward_per_arm::Vector{Float64}; α=1/sqrt(log(length(cumulative_reward_per_arm)))::Float64)
        # α > 0
        #Regret upper bound of O(1)/η + O(T)/η
        cumulative_reward_per_arm = cumulative_reward_per_arm ./ maximum(cumulative_reward_per_arm)
        loss = -1 .* cumulative_reward_per_arm
        η = 1 / α*sqrt(τ) # L_∞ is 1 because of linear loss
        #TODO ask CB if it makes sense to use expm1
        updated_probs = exp.(-η.*loss)
        return updated_probs ./ sum(updated_probs)
    end

    function EXP3(ξ::Categorical{Float64, Vector{Float64}}, reward_vector::Vector{Float64}, γ::Vector{Int64}, T::Int64, τ::Int64; η=1/sqrt(T)::Float64)
        most_recent_action = γ[τ]
        loss = reward_vector[most_recent_action]/ probs(ξ)[most_recent_action]  # observed loss
        loss_vector = zeros(Float64, length(reward_vector))
        loss_vector[most_recent_action] = loss
        return probs(ξ) .* exp.(-η.*loss_vector) ./ sum(probs(ξ) .* exp.(-η.*loss_vector))
    end

    # function ImplicitlyNormalizedForecaster(ξ, reward_vector, γ; q=0.5, η=1/sqrt(T), eps=1e-5, β = 1, total_number_iterations = 1000)
    #    most_recent_action = last(γ)
    #    loss = reward_vector[most_recent_action]/ probs(ξ[most_recent_action]) #observed loss
    #    loss_vector = zeros(Float64, length(reward_vector))
    #    loss_vector[most_recent_action] = loss
    #    tsallis_coef = ((1-q)/q)

    #    candidate = (tsallis_coef * (β + (1/tsallis_coef .* (probs(ξ).^(q-1))) + (η .* loss_vector))).^(1/(q-1))
    #    error = 1 - sum(candidate)
    #    if error < -eps
    #        while isnothing(top)
    #            bottom = β
    #            β = 2*β
    #            candidate = (tsallis_coef * (β + (1/tsallis_coef .* (probs(ξ).^(q-1))) + (η .* loss_vector))).^(1/(q-1))
    #            error = 1 - sum(candidate)
    #            if error > eps 
    #                top = β
    #            end
    #        end
    #    elseif error > eps
    #        while isnothing(bottom)
    #            top = β
    #            β = β/2
    #            candidate = (tsallis_coef * (β + (1/tsallis_coef .* (probs(ξ).^(q-1))) + (η .* loss_vector))).^(1/(q-1))
    #            error = 1 - sum(candidate)
    #            if error < eps 
    #                bottom = β
    #            end
    #        end
    #    else
    #        return candidate
    #    end
    #    iteration = 0
    #    while true
    #        if iteration >= total_number_iterations
    #            return candidate ./ sum(candidate)
    #        end
    #        β = (bot+top)/2 
    #        candidate = (tsallis_coef * (β + (1/tsallis_coef .* (probs(ξ).^(q-1))) + (η .* loss_vector))).^(1/(q-1))
    #        error = 1 - sum(candidate)
    #        if error > eps
    #            top = β
    #        elseif error < -eps
    #            bottom = β
    #        else
    #            return candidate ./ sum(candidate)
    #        end
    #        iteration += 1
    #        
    #    end
    #end

    function ExploreThenCommit(ξ::Categorical{Float64, Vector{Float64}}, τ::Int64, γ::Vector{Int64}, cumulative_reward_per_arm_bandit::Vector{Float64}, choices_per_arm::Vector{Int64}; m=10::Int64)
        d = length(cumulative_reward_per_arm_bandit)

        if τ <= d*m
            probs(ξ) .*= 0
            probs(ξ)[(τ % d) + 1] = 1
            return probs(ξ)
        elseif τ == (d*m + 1)
            probs(ξ) .*= 0
            probs(ξ)[argmax(cumulative_reward_per_arm_bandit ./ choices_per_arm)] = 1
            return probs(ξ)
        else
            return probs(ξ)
        end 
    end
    
    function UpperConfidenceBound(ξ::Categorical{Float64, Vector{Float64}}, τ::Int64, choices_per_arm::Vector{Int64}, average_reward_per_arm_bandit::Vector{Float64}; α=3::Int64)
        probs(ξ) .*= 0

        if α <= 2
            throw(ArgumentError)
        end
        d = length(choices_per_arm)
        if minimum(choices_per_arm) == 0
            probs(ξ)[argmin(choices_per_arm)] = 1
            return probs(ξ)
        end

        lower_confidence_band = -1 .* average_reward_per_arm_bandit .- sqrt(2*α*log(τ))./choices_per_arm
        probs(ξ)[argmin(lower_confidence_band)] = 1
        return probs(ξ)
    end 

    function EpsilonGreedy(ξ::Categorical{Float64, Vector{Float64}}, average_reward_per_arm_bandit::Vector{Float64};ϵ=0.1::Float64)
        probs(ξ) .*= 0

        if rand(Uniform(0, 1)) < ϵ
            arm = rand(1:length(average_reward_per_arm_bandit))
            probs(ξ)[arm] = 1
            return probs(ξ)
        end
        arm = argmax(average_reward_per_arm_bandit)
        probs(ξ)[arm] = 1
        return probs(ξ)
    end

    function LinearDecayedEpsilonGreedy(T::Int64, τ::Int64, ξ::Categorical{Float64, Vector{Float64}}, average_reward_per_arm_bandit::Vector{Float64};ϵ_start=1.0::Float64, ϵ_end=0.0::Float64)
        probs(ξ) .*= 0
        ϵ = ((ϵ_start - ϵ_end)/T)*τ
        if rand(Uniform(0, 1)) < ϵ
            arm = rand(1:length(average_reward_per_arm_bandit))
            probs(ξ)[arm] = 1
            return probs(ξ)
        end
        arm = argmax(average_reward_per_arm_bandit)
        probs(ξ)[arm] = 1
        return probs(ξ)
    end

    function ExpDecayedEpsilonGreedy(T::Int64, τ::Int64, ξ::Categorical{Float64, Vector{Float64}}, average_reward_per_arm_bandit::Vector{Float64};ϵ_start=1.0::Float64, ϵ_end=0.001::Float64)
        probs(ξ) .*= 0
        ϵ = (ϵ_end/ϵ_start)^(τ/T)
        if rand(Uniform(0, 1)) < ϵ
            arm = rand(1:length(average_reward_per_arm_bandit))
            probs(ξ)[arm] = 1
            return probs(ξ)
        end
        arm = argmax(average_reward_per_arm_bandit)
        probs(ξ)[arm] = 1
        return probs(ξ)
    end

    function Hedge(ξ::Categorical{Float64, Vector{Float64}}, reward_vector::Vector{Float64}; η=0.1::Float64)
        #full info algo
        # η > 0
        # https://people.eecs.berkeley.edu/~jiantao/2902021spring/scribe/EE290_Lecture_09.pdf
        updated_probs = probs(ξ) .* exp.(reward_vector .* η)
        return updated_probs ./ sum(updated_probs)
    end


end #Module