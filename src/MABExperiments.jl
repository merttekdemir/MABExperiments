using Random, Distributions
Random.seed!(42)

include("MABStruct.jl"); M = MABStructs;
include("OnlineLearningAlgorithms.jl"); O=OnlineLearningAlgorithms;

A = (Beta(0.15, 0.7), Beta(0.54, 0.2), Beta(0.38, 0.5))
ξ = Distributions.Categorical([1/3, 1/3, 1/3])
game = M.MABStruct(10, A, ξ)

#Correct learning rate OMD: √(2*log(length(game.A))/game.T)
M.run!(game, O.ExponentiatedGradient, true; kw_dict=Dict(:η => √(2*log(length(game.A))/game.T)))