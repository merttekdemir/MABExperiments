module MABPlots
using Statistics, Plots, Distributions, StatsBase
include("MABStruct.jl"); M = MABStructs;

    function PlotSeriesOverTime(experiments::Dict, MABField::Symbol; filename=nothing::Union{Nothing, String}, display_plot=false::Bool) where DT <: Tuple{Vararg{Distribution}}
        MABField in fieldnames(M.MABStructs.MABStruct) || throw(ArgumentError("MABField is not a field of MABStruct"))
    
        plot_size = length(experiments) * 300
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
        filename === nothing || savefig(fig, filename)
        display_plot === false || display(fig)
        return fig
    end

    function PlotSeriesHistogram(experiments::Dict, MABField::Symbol; filename=nothing::Union{Nothing, String}, display_plot=false::Bool) where DT <: Tuple{Vararg{Distribution}}
        MABField in fieldnames(M.MABStructs.MABStruct) || throw(ArgumentError("MABField is not a field of MABStruct"))
        plot_size = length(experiments) * 300
        plot_title = "Experiment Diganostics For $(String(MABField))"
        fig = plot(layout=length(experiments), size=(plot_size,plot_size), title=plot_title)
        for (i, algorithm) in enumerate(experiments)
            data = [getfield(experiments[algorithm[1]][j], MABField) for j in 1:length(experiments[algorithm[1]])]
            (i == 1 && (typeof(data) != Vector{Vector{Int64}})) && throw(ArgumentError("MABField is not a suitable field for a histogram"))                
            subplot_title = "$(algorithm[1])"
            plot!(xlabel="Arm", ylabel="$(String(MABField))", legend=:topleft, title=subplot_title, subplot=i)
            cnts = StatsBase.mode(data)
            cntmap = StatsBase.countmap(cnts)
            x = sort(collect(keys(cntmap)))
            y = [cntmap[i] for i in x]
            bar!(x, y, label="Mode At Each Time Step", bar_width=1, subplot=i, xticks=x)
        end
        filename === nothing || savefig(fig, filename)
        display_plot === false || display(fig)
        return fig
    end

end #module