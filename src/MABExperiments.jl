using Random, Distributions, Plots, Statistics
NUMBER_OF_EXPERIMENTS_PER_ALGORITHM = 10
NUMBER_OF_ITERATIONS_PER_EXPERIMENT = 10
Random.seed!(42)
seeds = rand(1:10000000, NUMBER_OF_EXPERIMENTS_PER_ALGORITHM)

include("MABStruct.jl"); M = MABStructs;
include("OnlineLearningAlgorithms.jl"); O=OnlineLearningAlgorithms;
include("MABPlots.jl"); P=MABPlots;

A = (Beta(0.15, 0.7), Beta(0.54, 0.2), Beta(0.38, 0.5))
ξ = Distributions.Categorical([1/3, 1/3, 1/3])
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
experiments = Dict(string(algorithm) => zeros(M.MABStruct, A, NUMBER_OF_EXPERIMENTS_PER_ALGORITHM) for algorithm in algorithms)

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
            #Random.seed!(rand(1:10000))
            println(algorithm)
            println(j)
            println(rand(2,10))

    #Correct learning rate OMD: √(2*log(length(game.A))/game.T)
            M.reset!(game, "MAB_experiment_$j")
            M.run!(game, algorithm, argnames, default_argnames, default_values_algo; verbose=false)
            println(game)
            # experiments[algorithm][j] = game
            M.set_instance!(experiments[string(algorithm)][j], game)
        end
    end
end

function PlotSeriesOverTime(experiments::Dict{String, Vector{M.MABStruct{DT}}}, MABAttribute::Symbol) where DT <: Tuple{Vararg{Distribution}}
    #TODO check if attribute in mabstruct attributes
    fig = plot(layout=length(experiments))
    for (i, algorithm) in enumerate(experiments)
        s = [getfield(experiments[algorithm[1]][j], MABAttribute) for j in 1:length(experiments[algorithm[1]])]
        sample_mean_over_time = Statistics.mean(s)
        sample_std_error_over_time = Statistics.std(s, mean=sample_mean_over_time)./sqrt(length(s))
        plot!(xlabel="Iteration", ylabel="Regret Fixed", legend=:topleft, subplot=i)
        xaxis = [τ for τ in 1:length(s[1])]
        sublinear_regret = sqrt.(xaxis)
        plot!(xaxis, sample_mean_over_time, label="Simulated Regret", subplot=i,
        ribbon=(sample_mean_over_time .- 1.96.*sample_std_error_over_time, sample_mean_over_time .+ 1.96.*sample_std_error_over_time))
        #plot!(xaxis, 100 .* sublinear_regret, label="Sublinear Regret Benchmark", linestlye=:dash)
        # plot!(xaxis, sublinear_regret./sample_mean_over_time, label="Sublinear Regret Benchmark", linestlye=:dash)
        # plot!(xaxis, xaxis, label="45 degree line")
    end
    display(fig)
    return fig
end
# M.run!(game, O.ExponentiatedGradient, true; kw_dict=Dict(:η => √(2*log(length(game.A))/game.T)))

experiment_1(A, ξ, algorithms);
PlotSeriesOverTime(experiments, :regret_fixed)