module OnlineLearningAlgorithms
export ExponentiatedGradient
using Random, Distributions

    function ExponentiatedGradient(ξ, reward_vector; η=0.01)
            loss = -1 .* reward_vector
            #TODO ask CB if it makes sense to use expm1
            return probs(ξ) .* exp.(-η.*loss) ./ sum(probs(ξ) .* exp.(-η.*loss))
    end

end #Module