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

    function ImplicityNormalizedForecaster(ξ, reward_vector, γ; q=0.5, η=1/sqrt(T), eps=1e-5, β = 1)
        most_recent_action = last(γ)
        loss = reward_vector[most_recent_action]/ probs(ξ[most_recent_action]) #observed loss
        loss_vector = zeros(Float64, length(reward_vector))
        loss_vector[most_recent_action] = loss
        tsallis_coef = ((1-q)/q)
        while true
            candidate = (tsallis_coef * (β + (1/tsallis_coef .* (probs(ξ).^(q-1))) + (η .* loss_vector))).^(1/(q-1))
            error = 1 - sum(candidate)
            if abs(error) <= eps
                return candidate
            elseif error > eps
                β_previous = β
                β = β - β/4
            elseif error < -eps
            end
        end
        return 
    end

    function EpsilonGreedy(ξ, stochastic_average_reward_per_arm ;ϵ=0.1)
        if Uniform{Float64}{0, 1} < ϵ
            probs(ξ) = probs(ξ).*0
            arm = Uniform{Int64}{1, length(last(stochastic_average_reward_per_arm))}
            probs(ξ)[arm] = 1
            return probs(ξ)
        end
        probs(ξ) = probs(ξ).*0
        arm = argmax(stochastic_average_reward_per_arm)
        probs(ξ)[arm] = 1
        return probs(ξ)
    end

    function Hedge(ξ, reward_vector; η)
        #full info algo
        # η > 0
        # https://people.eecs.berkeley.edu/~jiantao/2902021spring/scribe/EE290_Lecture_09.pdf
        return probs(ξ) .* exp(reward_vector .* -η)
    end


    

end #Module