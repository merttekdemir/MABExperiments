using Random, Distributions, Plots, Statistics, StatsBase, YAML
CONF = YAML.load(open("src/configuration.yml"))
NUMBER_OF_EXPERIMENTS_PER_ALGORITHM = CONF["NUMBER_OF_EXPERIMENTS_PER_ALGORITHM"]
NUMBER_OF_ITERATIONS_PER_EXPERIMENT = CONF["NUMBER_OF_ITERATIONS_PER_EXPERIMENT"]

Random.seed!(CONF["SEED"])

seeds = rand(1:10000000, NUMBER_OF_EXPERIMENTS_PER_ALGORITHM)

include("MABStruct.jl"); M = MABStructs;
include("OnlineLearningAlgorithms.jl"); O = OnlineLearningAlgorithms;
include("MABPlots.jl"); P = MABPlots;

A = Tuple(getfield(Distributions, Symbol(i["Dist"]))(i["Params"]...) for i in CONF["A"])
ξ = getfield(Distributions, Symbol(CONF["ξ"][1]["Dist"]))(CONF["ξ"][1]["Params"]...)


default_values = Dict("ExponentiatedGradient" => get(CONF, "ExponentiatedGradientDefaultValues", [Dict()]),
                      "FtrlExponentiatedGradient" => get(CONF, "FtrlExponentiatedGradientDefaultValues", [Dict()]),
                      "EXP3" => get(CONF, "EXP3DefaultValues", [Dict()]),
                      "ImplicityNormalizedForecaster" => get(CONF, "ImplicityNormalizedForecasterDefaultValues", [Dict()]),
                      "ExploreThenCommit" => get(CONF, "ExploreThenCommitDefaultValues", [Dict()]),
                      "UpperConfidenceBound" => get(CONF, "UpperConfidenceBoundDefaultValues", [Dict()]),
                      "EpsilonGreedy" => get(CONF, "EpsilonGreedyDefaultValues", [Dict()]),
                      "LinearDecayedEpsilonGreedy" => get(CONF, "LinearDecayedEpsilonGreedyDefaultValues", [Dict()]),
                      "ExpDecayedEpsilonGreedy" => get(CONF, "ExpDecayedEpsilonGreedyDefaultValues", [Dict()]),
                      "Hedge" => get(CONF, "HedgeDefaultValues", [Dict()]),
)  # Define it as a Dict of Dict, first key is the algorithm, second set of keys is the parameter per algorithm, getting it from the config would be optimal
# Define a function that extracts the argument names from the algorithms definition

for algorithm in keys(default_values)
    if length(default_values[algorithm]) > 1
        default_values[algorithm] = merge([i for i in default_values[algorithm]]...)
    end
end

experiments = Dict(string(algorithm) => [zero(M.MABStruct, A) for _ in 1:NUMBER_OF_EXPERIMENTS_PER_ALGORITHM] for algorithm in algorithms)

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
        default_values_algo = default_values[string(algorithm)][1]
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

function experiment_2(A, ξ, algorithms)

    
    @Threads.threads for algorithm in algorithms
        game = M.MABStruct(NUMBER_OF_ITERATIONS_PER_EXPERIMENT, A, ξ, "$(string(algorithm)) MAB_Experiment_1")
        default_values_algo = default_values[string(algorithm)][1]
        argnames, default_argnames = method_args(algorithm, isempty(default_values_algo))
       

        for j in 1:NUMBER_OF_EXPERIMENTS_PER_ALGORITHM
            Random.seed!(seeds[j])

    #Correct learning rate OMD: √(2*log(length(game.A))/game.T)
            M.reset!(game; name="$(string(algorithm))_MAB_Experiment_$j")
            M.run!(game, algorithm, argnames, default_argnames, default_values_algo; verbose=false)
            M.set_instance!(experiments[string(algorithm)][j], game)
        end
    end
    return experiments
end

experiments = experiment_2(A, ξ, algorithms);


if haskey(CONF, "PlotSeriesOverTime")
    for i in CONF["PlotSeriesOverTime"]
        P.PlotSeriesOverTime(experiments, Symbol(i["MABField"]); filename=i["FileName"])
    end
end

if haskey(CONF, "PlotSeriesHistogram")
    for i in CONF["PlotSeriesHistogram"]
        P.PlotSeriesHistogram(experiments, Symbol(i["MABField"]); filename=i["FileName"])
    end
end

println("Experiment Terminated Successfully")
