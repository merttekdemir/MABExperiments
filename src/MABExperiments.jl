using Random, Distributions, Plots, Statistics, StatsBase
NUMBER_OF_EXPERIMENTS_PER_ALGORITHM = 10
NUMBER_OF_ITERATIONS_PER_EXPERIMENT = 10
Random.seed!(42)
seeds = rand(1:10000000, NUMBER_OF_EXPERIMENTS_PER_ALGORITHM)

include("MABStruct.jl"); M = MABStructs;
include("OnlineLearningAlgorithms.jl"); O = OnlineLearningAlgorithms;
include("Utils.jl"); U = Utils;
# include("MABPlots.jl"); P = MABPlots;

A = (Beta(0.54, 0.2), Beta(0.54, 0.2), Beta(0.54, 0.2))
ξ = Distributions.Categorical([1/3, 1/3, 1/3])
default_values = Dict{String, Any}("ExponentiatedGradient" => Dict{Symbol, Any}(),
                      "FtrlExponentiatedGradient" => Dict{Symbol, Any}(),
                      "EXP3" => Dict{Symbol, Any}(),
                      "ImplicityNormalizedForecaster" => Dict{Symbol, Any}(),
                      "ExploreThenCommit" => Dict{Symbol, Any}(),
                      "UpperConfidenceBound" => Dict{Symbol, Any}(),
                      "EpsilonGreedy" => Dict{Symbol, Any}(),
                      "LinearDecayedEpsilonGreedy" => Dict{Symbol, Any}(),
                      "ExpDecayedEpsilonGreedy" => Dict{Symbol, Any}(),
                      "Hedge" => Dict{Symbol, Any}(),
)  # Define it as a Dict of Dict, first key is the algorithm, second set of keys is the parameter per algorithm, getting it from the config would be optimal
# Define a function that extracts the argument names from the algorithms definition
algorithms = [O.ExponentiatedGradient, O.FtrlExponentiatedGradient, O.EXP3, O.ExploreThenCommit,
                       O.UpperConfidenceBound, O.EpsilonGreedy, O.ExpDecayedEpsilonGreedy,
                       O.LinearDecayedEpsilonGreedy, O.Hedge]
experiments = Dict(string(algorithm) => [zero(M.MABStruct, A) for i in 1:NUMBER_OF_EXPERIMENTS_PER_ALGORITHM] for algorithm in algorithms)

function experiment_1(A, ξ, algorithms)
    game = M.MABStruct(NUMBER_OF_ITERATIONS_PER_EXPERIMENT, A, ξ, "MAB_Experiment_1")
    for algorithm in algorithms
        default_values_algo = default_values[string(algorithm)]
        argnames, default_argnames = U.method_args(algorithm, isempty(default_values_algo))
       

        for j in 1:NUMBER_OF_EXPERIMENTS_PER_ALGORITHM
            Random.seed!(seeds[j])

    #Correct learning rate OMD: √(2*log(length(game.A))/game.T)
            M.reset!(game; name="MAB_experiment_$j")
            M.run!(game, algorithm, argnames, default_argnames, default_values_algo; verbose=false)
            M.set_instance!(experiments[string(algorithm)][j], game)
        end
    end
    return experiments
end

# M.run!(game, O.ExponentiatedGradient, true; kw_dict=Dict(:η => √(2*log(length(game.A))/game.T)))

experiments = experiment_1(A, ξ, [O.EXP3]);
# P.PlotSeriesOverTime(experiments, :regret_fixed)