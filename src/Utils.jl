module Utils

function method_args(optimizer::Function, default_values_bool::Bool)
    method = methods(optimizer)[1]
    matching = match(r"(\()(.*)(\;\s)(.*)(\))", string(method))
    if (matching[4] == "") || default_values_bool
        return [Symbol(match[1]) for match in eachmatch(r"([\w|_]+)::", matching[2])], Vector{Symbol}()
    else
        return [Symbol(match[1]) for match in eachmatch(r"([\w|_]+)::", matching[2])], Symbol.(split(matching[4], ", "))
    end
end

end  # module