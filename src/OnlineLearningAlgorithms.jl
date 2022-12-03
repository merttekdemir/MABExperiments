module OnlineLearningAlgorithms
export ExponentiatedGradient
using Random, Distributions

    #TODO xi_{0}
    #v non-empty convex subset of Rd
    #V is the space of all simplexes of cardinality |A|
    function ExponentiatedGradient(ξ, reward_vector; η=0.01)
            loss = -1 .* reward_vector
            #TODO ask CB if it makes sense to use expm1
            println(ξ)
            return probs(ξ) .* exp.(-η.*loss) ./ sum(probs(ξ) .* exp.(-η.*loss))
    end

end #Module