module MABPlots
using Statistics, Plots, Distributions, StatsBase
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


    function PlotSeriesOverTime(experiments::Dict, MABField::Symbol)
        MABField in fieldnames(M.MABStructs.MABStruct) || throw(ArgumentError("MABField is not a field of MABStruct"))
    
        plot_size = length(experiments) * 150
        plot_title = "Experiment Diganostics For $(String(MABField))"
        fig = plot(layout=length(experiments), size=(plot_size,plot_size), plot_title=plot_title)
        for (i, algorithm) in enumerate(experiments)
    
            data = [getfield(experiments[algorithm[1]][j], MABField) for j in 1:length(experiments[algorithm[1]])]
    
            if typeof(data) == Vector{Vector{Float64}}
                sample_mean_over_time = Statistics.mean(data)
                sample_std_error_over_time = Statistics.std(data, mean=sample_mean_over_time)./sqrt(length(data))
                CI = 1.96.*sample_std_error_over_time
                subplot_title = "$(algorithm[1])"
                plot!(xlabel="Iteration", ylabel="$(String(MABField))", legend=:topleft, title=subplot_title, subplot=i)
                xaxis = [τ for τ in 1:length(data[1])]
                sublinear_regret = sqrt.(xaxis)
                plot!(xaxis, sample_mean_over_time, label="$(String(MABField))", subplot=i, ribbon=CI)
    
            elseif typeof(data) == Vector{Vector{Int64}}
                subplot_title = "$(algorithm[1])"
                plot!(xlabel="Iteration", ylabel="$(String(MABField))", legend=:topleft, title=subplot_title, subplot=i)
                xaxis = [τ for τ in 1:length(data[1])]
                sublinear_regret = sqrt.(xaxis)
                plot!(xaxis, StatsBase.mode(data), label="Mode across experiments", subplot=i)
            end
        end
        display(fig)
        return fig
    end

    # function PlotDiagnostics(MABExperiments::Vector{MABStruct})


    # end

end #module