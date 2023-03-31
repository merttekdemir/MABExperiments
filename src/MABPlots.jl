module MABPlots
using Statistics, Plots, Distributions
include("MABStruct.jl"); M = MABStructs;

    function PlotSeriesOverTime(series::Vector{Vector{Float64}})
        sample_mean_over_time = Statistics.mean(series)
        sample_std_error_over_time = Statistics.std(series, mean=sample_mean_over_time)./sqrt(length(series))
        p = plot(xlabel="Iteration", ylabel="Regret Fixed", legend=:topleft)
        xaxis = [τ for τ in 1:length(series[1])]
        sublinear_regret = sqrt.(xaxis)
        p = plot!(xaxis, sample_mean_over_time, label="Simulated Regret", ribbon=(sample_mean_over_time .- 1.96.*sample_std_error_over_time, sample_mean_over_time .+ 1.96.*sample_std_error_over_time))
        #plot!(xaxis, 100 .* sublinear_regret, label="Sublinear Regret Benchmark", linestlye=:dash)
        # plot!(xaxis, sublinear_regret./sample_mean_over_time, label="Sublinear Regret Benchmark", linestlye=:dash)
        # plot!(xaxis, xaxis, label="45 degree line")
        return p
    end

    function PlotSeriesOverTime(experiment::Dict{Function, Vector{M.MABStruct{DT}}}, MABAttribute::Symbol) where DT <: Tuple{Vararg{Distribution}}
        #TODO check if attribute in mabstruct attributes
        fig = plot(layout=length(experiment))
        for (i, algorithm) in enumerate(experiments)
            s = [experiments[algorithm].MABAttribute[j] for j in 1:length(experiments[algorithm])]
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
    end

    # function PlotDiagnostics(MABExperiments::Vector{MABStruct})


    # end

end #module