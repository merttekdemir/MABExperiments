module OnlineLearningAlgorithms
export ExponentiatedGradient
using Random, Distributions


    function ExponentiatedGradient(ξ, reward_vector; η=0.01)
        loss = -1 .* reward_vector
        #TODO ask CB if it makes sense to use expm1
        return probs(ξ) .* exp.(-η.*loss) ./ sum(probs(ξ) .* exp.(-η.*loss))
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

    function ImplicityNormalizedForecaster(ξ, reward_vector, γ; q=0.5, η=1/sqrt(T), eps=1e-5, β = 1, total_number_iterations = 1000)
        most_recent_action = last(γ)
        loss = reward_vector[most_recent_action]/ probs(ξ[most_recent_action]) #observed loss
        loss_vector = zeros(Float64, length(reward_vector))
        loss_vector[most_recent_action] = loss
        tsallis_coef = ((1-q)/q)

        candidate = (tsallis_coef * (β + (1/tsallis_coef .* (probs(ξ).^(q-1))) + (η .* loss_vector))).^(1/(q-1))
        error = 1 - sum(candidate)
        if error < -eps
            while isnothing(top)
                bottom = β
                β = 2*β
                candidate = (tsallis_coef * (β + (1/tsallis_coef .* (probs(ξ).^(q-1))) + (η .* loss_vector))).^(1/(q-1))
                error = 1 - sum(candidate)
                if error > eps 
                    top = β
                end
            end
        elseif error > eps
            while isnothing(bottom)
                top = β
                β = β/2
                candidate = (tsallis_coef * (β + (1/tsallis_coef .* (probs(ξ).^(q-1))) + (η .* loss_vector))).^(1/(q-1))
                error = 1 - sum(candidate)
                if error < eps 
                    bottom = β
                end
            end
        else
            return candidate
        end
        iteration = 0
        while true
            if iteration >= total_number_iterations
                return candidate ./ sum(candidate)
            end
            β = (bot+top)/2 
            candidate = (tsallis_coef * (β + (1/tsallis_coef .* (probs(ξ).^(q-1))) + (η .* loss_vector))).^(1/(q-1))
            error = 1 - sum(candidate)
            if error > eps
                top = β
            elseif error < -eps
                bottom = β
            else
                return candidate ./ sum(candidate)
            iteration += 1
        end
    end
    

end #Module