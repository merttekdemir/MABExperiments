using Random, Distributions, Plots, Statistics, StatsBase, YAML

#Import configuration file
CONF = YAML.load(open("src/configuration.yml"))

#Global experiment params
NUMBER_OF_EXPERIMENTS_PER_ALGORITHM = CONF["NUMBER_OF_EXPERIMENTS_PER_ALGORITHM"]
NUMBER_OF_ITERATIONS_PER_EXPERIMENT = CONF["NUMBER_OF_ITERATIONS_PER_EXPERIMENT"]
Random.seed!(CONF["SEED"])

#A pre-seeded random list of numbers, one per game-algorithm pair
#In this way experiment repetetions for each algorithm are tested under the same samples from the stochastic actions
seeds = rand(1:10000000, NUMBER_OF_EXPERIMENTS_PER_ALGORITHM)

#Include necessary scripts
include("OnlineLearningAlgorithms.jl"); O = OnlineLearningAlgorithms;
include("MABStruct.jl"); M = MABStructs;
include("MABPlots.jl"); P = MABPlots;
include("Utils.jl"); U = Utils;

#The probability laws governing the distribution of the rewards from each of the possible actions determined from the config
A = Tuple(getfield(Distributions, Symbol(i["Dist"]))(i["Params"]...) for i in CONF["A"])

#The inital policy vector determined from the config
ξ = getfield(Distributions, Symbol(CONF["ξ"][1]["Dist"]))(CONF["ξ"][1]["Params"]...)

#Collect the default arguments for each of the algorithms as given in the config
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
)  

#Format the default values due to the way the config parser reads in the information
for algorithm in keys(default_values)
    #More than 1 default parameter is stored as several dicts so we merge them
    if length(default_values[algorithm]) > 1
        default_values[algorithm] = [merge([i for i in default_values[algorithm]]...)]
    end
end

#Get the list of algorithms to be tested from the config
algorithms = [getfield(O, Symbol(algorithm)) for algorithm in CONF["Algorithms"]]

#Allocate the memory necessary for the total number of games to be run before hand
experiments = Dict(string(algorithm) => [zero(M.MABStruct, A) for _ in 1:NUMBER_OF_EXPERIMENTS_PER_ALGORITHM] for algorithm in algorithms)


"""
    RunExperiment(A, ξ, algorithms)

    The master function for running the experiment in the case of a single thread.
    Runs a MABStruct game for each algorithm provided.
"""
function RunExperiment(A, ξ, algorithms)
    #Create a MABStruct game
    game = M.MABStruct(NUMBER_OF_ITERATIONS_PER_EXPERIMENT, A, ξ, "MAB_Experiment_1")

    #Iterate over the total algorithms
    for algorithm in algorithms
        #Recover the default values for the algorithm
        default_values_algo = default_values[string(algorithm)][1]
        argnames, default_argnames = U.method_args(algorithm, isempty(default_values_algo))
       
        #Iterate over the number of experiments per algorithm
        for j in 1:NUMBER_OF_EXPERIMENTS_PER_ALGORITHM
            Random.seed!(seeds[j])
            #Reset the game (more efficient that creating a new game from scratch)
            M.reset!(game; name="MAB_experiment_$j")
            #Run the game for the number of iterations
            M.run!(game, algorithm, argnames, default_argnames, default_values_algo; verbose=false)
            #Copy the results of the game into the pre-initalized experiments results dict
            M.set_instance!(experiments[string(algorithm)][j], game)
        end
    end
    return experiments
end

"""
    RunExperimentMultiThread(A, ξ, algorithms)

    The master function for running the experiment in the case of a single thread.
    Runs a MABStruct game for each algorithm provided.
"""
function RunExperimentMultiThread(A, ξ, algorithms)
    #Each algorithm can be run independently so we can parallelize the games
    #Iterate over the total algorithms
    @Threads.threads for algorithm in algorithms
        #Create a MABStruct game
        game = M.MABStruct(NUMBER_OF_ITERATIONS_PER_EXPERIMENT, A, ξ, "$(string(algorithm)) MAB_Experiment_1")
        #Recover the default values for the algorithm
        default_values_algo = default_values[string(algorithm)][1]
        argnames, default_argnames = U.method_args(algorithm, isempty(default_values_algo))
       
        #Iterate over the number of experiments per algorithm
        for j in 1:NUMBER_OF_EXPERIMENTS_PER_ALGORITHM
            Random.seed!(seeds[j])
            #Reset the game (more efficient that creating a new game from scratch)
            M.reset!(game; name="$(string(algorithm))_MAB_Experiment_$j")
            #Run the game for the number of iterations
            M.run!(game, algorithm, argnames, default_argnames, default_values_algo; verbose=false)
            #Copy the results of the game into the pre-initalized experiments results dict
            M.set_instance!(experiments[string(algorithm)][j], game)
        end
    end
    return experiments
end

#Check if more than one thread is available and run correpsonding expeirment function
if Threads.nthreads() > 1
    experiments = RunExperimentMultiThread(A, ξ, algorithms);
else
    experiments = RunExperiment(A, ξ, algorithms);
end

#Plot according to config file
if haskey(CONF, "PlotSeriesOverTime")
    for i in CONF["PlotSeriesOverTime"]
        P.PlotSeriesOverTime(experiments, Symbol(i["MABField"]); filename=i["FileName"], display_plot=i["DisplayPlot"])
    end
end

if haskey(CONF, "PlotSeriesHistogram")
    for i in CONF["PlotSeriesHistogram"]
        P.PlotSeriesHistogram(experiments, Symbol(i["MABField"]); filename=i["FileName"], display_plot=i["DisplayPlot"])
    end
end

println("Experiment Terminated Successfully")
