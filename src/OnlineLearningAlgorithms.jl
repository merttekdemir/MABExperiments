module OnlineLearningAlgorithms
export ExponentiatedGradient
using Random, Distributions, DataStructures


    function ExponentiatedGradient(ξ, reward_vector; η=0.01)
        loss = -1 .* reward_vector
        #TODO ask CB if it makes sense to use expm1
        return probs(ξ) .* exp.(-η.*loss) ./ sum(probs(ξ) .* exp.(-η.*loss))  
        # TODO: Fix whether we should return the probs or the DiscreteCategorical
    end

    function FtlrExponentiatedGradient(τ, cumulative_reward_per_arm; α=1/sqrt(log(length(cumulative_reward_per_arm))))
        # α > 0
        #Regret upper bound of O(1)/η + O(T)/η
        loss = -1 .* cumulative_reward_per_arm
        η = 1 / α*sqrt(τ) # L_∞ is 1 because of linear loss
        #TODO ask CB if it makes sense to use expm1
        return exp.(-η.*loss) ./ sum(exp.(-η.*loss))
    end

    function EXP3(ξ, reward_vector, γ; η=1/sqrt(T))
        most_recent_action = last(γ)
        loss = reward_vector[most_recent_action]/ probs(ξ[most_recent_action]) #observed loss
        loss_vector = zeros(Float64, length(reward_vector))
        loss_vector[most_recent_action] = loss
        return probs(ξ) .* exp.(-η.*loss_vector) ./ sum(probs(ξ) .* exp.(-η.*loss_vector))
    end

    # function ImplicityNormalizedForecaster(ξ, reward_vector, γ; q=0.5, η=1/sqrt(T), eps=1e-5, β = 1, total_number_iterations = 1000)
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

    function ExploreThenCommit(τ, γ, cumulative_reward_per_arm_bandit; m=10)
        d = length(cumulative_reward_per_arm_bandit)
        counter_per_arm = counter(γ[1:min(d*m, τ)])
        choices_per_arm = zeros(Int64, d)
        for i in sort(collect(keys(counter_per_arm)))
            if i != 0
                choices_per_arm[i] = choices[i]
            end
        end
        if τ <= d*m
            ξ = zeros(Float64, d)
            ξ[(t % d) + 1] = 1
            return ξ
        else 
            ξ = zeros(Float64, d)
            ξ[argmax(cumulative_reward_per_arm_bandit ./ choices_per_arm)] = 1
            return ξ
        end 
    end
    
    function UpperConfidenceBound(ξ, τ, choices_per_arm, average_reward_per_arm_bandit; alpha = 3)
        probs(ξ) .*= 0

        if alpha <= 2
            throw(ArgumentError)
        end
        d = length(choices_per_arm)
        if min(choices_per_arm) == 0
            probs(ξ)[argmin(choices_per_arm)] = 1
            return probs(ξ)
        end

        lower_confidence_band = -1 .* average_reward_per_arm_bandit .- sqrt(2αlog(τ))./choices_per_arm
        probs(ξ)[argmin(lower_confidence_band)] = 1
        return probs(ξ)
    end 

    function EpsilonGreedy(ξ, average_reward_per_arm_bandit ;ϵ=0.1)
        probs(ξ) .*= 0

        if Uniform{Float64}{0, 1} < ϵ
            arm = rand(1:length(average_reward_per_arm_bandit))
            probs(ξ)[arm] = 1
            return probs(ξ)
        end
        arm = argmax(average_reward_per_arm_bandit)
        probs(ξ)[arm] = 1
        return probs(ξ)
    end

    function LinearDecayedEpsilonGreedy(T, τ, ξ, average_reward_per_arm_bandit;ϵ_start=1.0, ϵ_end=0)
        probs(ξ) .*= 0
        ϵ = ((ϵ_start - ϵ_end)/T)*τ
        if Uniform{Float64}{0, 1} < ϵ
            arm = rand(1:length(average_reward_per_arm_bandit))
            probs(ξ)[arm] = 1
            return probs(ξ)
        end
        arm = argmax(average_reward_per_arm_bandit)
        probs(ξ)[arm] = 1
        return probs(ξ)
    end

    function ExpDecayedEpsilonGreedy(T, τ, ξ, average_reward_per_arm_bandit;ϵ_start=1.0, ϵ_end=0.001)
        probs(ξ) .*= 0
        ϵ = (ϵ_end/ϵ_start)^(τ/T)
        if Uniform{Float64}{0, 1} < ϵ
            arm = rand(1:length(average_reward_per_arm_bandit))
            probs(ξ)[arm] = 1
            return probs(ξ)
        end
        arm = argmax(average_reward_per_arm_bandit)
        probs(ξ)[arm] = 1
        return probs(ξ)
    end

    function Hedge(ξ, reward_vector; η)
        #full info algo
        # η > 0
        # https://people.eecs.berkeley.edu/~jiantao/2902021spring/scribe/EE290_Lecture_09.pdf
        updated_probs = probs(ξ) .* exp(reward_vector .* -η)
        return updated_probs ./ sum(updated_probs)
    end


end #Module