using Random, Distributions
NUMBER_OF_EXPERIMENTS_PER_ALGORITHM = 1000
Random.seed!(42)
seeds = rand(1:10000000, NUMBER_OF_EXPERIMENTS_PER_ALGORITHM)

include("MABStruct.jl"); M = MABStructs;
# include("src\\MABStruct.jl"); M = MABStructs;
include("OnlineLearningAlgorithms.jl"); O=OnlineLearningAlgorithms;

A = (Beta(0.15, 0.7), Beta(0.54, 0.2), Beta(0.38, 0.5))
ξ = Distributions.Categorical([1/3, 1/3, 1/3])

function experiment_1(A, ξ)
    algorithms = [O.ExponentiatedGradient]
    experiments = Dict(algorithm => zeros(M.MABStruct, NUMBER_OF_EXPERIMENTS_PER_ALGORITHM) for algorithm in algorithms)

    for algorithm in algorithms
        for j in 1:NUMBER_OF_EXPERIMENTS_PER_ALGORITHM
            Random.seed!(seeds[j])

    #Correct learning rate OMD: √(2*log(length(game.A))/game.T)
            game = M.MABStruct(1, A, ξ, "MAB_Experiment_$j")
            M.run!(game, algorithm, false; kw_dict=Dict(:η => √(2*log(length(game.A))/game.T)))
            # experiments[algorithm][j] = game
            M.set_instance!(experiments[algorithm][j], game)
        end
    end
end
# M.run!(game, O.ExponentiatedGradient, true; kw_dict=Dict(:η => √(2*log(length(game.A))/game.T)))
