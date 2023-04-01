using Random, Distributions, Plots, Statistics, StatsBase
NUMBER_OF_EXPERIMENTS_PER_ALGORITHM = 100
NUMBER_OF_ITERATIONS_PER_EXPERIMENT = 10000
Random.seed!(42)
seeds = rand(1:10000000, NUMBER_OF_EXPERIMENTS_PER_ALGORITHM)

include("MABStruct.jl"); M = MABStructs;
include("OnlineLearningAlgorithms.jl"); O = OnlineLearningAlgorithms;
include("MABPlots.jl"); P = MABPlots;

# A = (Beta(0.15, 0.7), Beta(0.54, 0.2), Beta(0.38, 0.5))
A = (Normal(0.5, 0.7), Normal(0.5, 0.2), Normal(0.4, 0.5), Normal(0.2, 0.2))
ξ = Distributions.Categorical(length(A))
default_values = Dict("ExponentiatedGradient" => Dict(),
                      "FtrlExponentiatedGradient" => Dict(),
                      "EXP3" => Dict(),
                      "ImplicityNormalizedForecaster" => Dict(),
                      "ExploreThenCommit" => Dict(),
                      "UpperConfidenceBound" => Dict(),
                      "EpsilonGreedy" => Dict(),
                      "LinearDecayedEpsilonGreedy" => Dict(),
                      "ExpDecayedEpsilonGreedy" => Dict(),
                      "Hedge" => Dict(),
)  # Define it as a Dict of Dict, first key is the algorithm, second set of keys is the parameter per algorithm, getting it from the config would be optimal
# Define a function that extracts the argument names from the algorithms definition
algorithms = [O.ExponentiatedGradient, O.FtrlExponentiatedGradient, O.EXP3, O.ExploreThenCommit,
                       O.UpperConfidenceBound, O.EpsilonGreedy, O.ExpDecayedEpsilonGreedy,
                       O.LinearDecayedEpsilonGreedy, O.Hedge]
experiments = Dict(string(algorithm) => [zero(M.MABStruct, A) for i in 1:NUMBER_OF_EXPERIMENTS_PER_ALGORITHM] for algorithm in algorithms)

function method_args(optimizer::Function, default_values_bool::Bool)
     method = methods(optimizer)[1]
     matching = match(r"(\()(.*)(\;\s)(.*)(\))", string(method))
     if (matching[4] == "") | default_values_bool
         return [Symbol(match[1]) for match in eachmatch(r"([\w|_]+)::", matching[2])], Vector{Symbol}()
     else
         return [Symbol(match[1]) for match in eachmatch(r"([\w|_]+)::", matching[2])], Symbol.(split(matching[4], ", "))
     end
 end

function experiment_1(A, ξ, algorithms)
    game = M.MABStruct(NUMBER_OF_ITERATIONS_PER_EXPERIMENT, A, ξ, "MAB_Experiment_1")
    for algorithm in algorithms
        default_values_algo = default_values[string(algorithm)]
        argnames, default_argnames = method_args(algorithm, isempty(default_values_algo))
       

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


experiments = experiment_1(A, ξ, algorithms);
P.PlotSeriesOverTime(experiments, :regret_fixed; filename="ExperimentOutputs/plot_of_regret_over_time")
P.PlotSeriesOverTime(experiments, :γ; filename="ExperimentOutputs/plot_of_action_over_time")
P.PlotSeriesHistogram(experiments, :γ; filename="ExperimentOutputs/plot_of_action_histogram")
