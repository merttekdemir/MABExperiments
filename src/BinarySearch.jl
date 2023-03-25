using Distributions

function binary_search(ξ=Distributions.Categorical([1/3, 1/3, 1/3]), β = 23, q = 1/2, η=1/sqrt(1000000), total_number_iterations = 100, eps=1e-1)
    loss_vector = zeros(Float64, length(probs(ξ)))
    loss_vector[2] = -1
    tsallis_coef = ((1-q)/q)
    candidate = (tsallis_coef .* (β .+ (1/tsallis_coef .* (probs(ξ).^(q-1))) + (η .* loss_vector))).^(1/(q-1))
    error = 1 - sum(candidate)
    bottom = nothing
    top = nothing
    if error > eps
        println("first loop")
        while isnothing(top)
            println("error = ", error)
            println("β = ", β)
            bottom = β
            β = 2*β
            candidate = (tsallis_coef .* (β .+ (1/tsallis_coef .* (probs(ξ).^(q-1))) + (η .* loss_vector))).^(1/(q-1))
            error = 1 - sum(candidate)
            if error < -eps 
                top = β
            end
        end
    elseif error < -eps
        println("second loop")
        while isnothing(bottom)
            println("error = ", error)
            println("β = ", β)
            top = β
            β = β/2
            candidate = (tsallis_coef .* (β .+ (1/tsallis_coef .* (probs(ξ).^(q-1))) + (η .* loss_vector))).^(1/(q-1))
            error = 1 - sum(candidate)
            if error > eps 
                bottom = β
            end
        end
    else
        return candidate
    end
    iteration = 0
    println("third loop")
    while true
        println("iteration = ", iteration)
        # println("bottom = ", bottom)
        # println("top = ", top)
        println("error = ", error)
        println("β = ", β)
        if iteration >= total_number_iterations
            println("max iterations reached")
            println(error)
            return candidate ./ sum(candidate)
        end
        β = (bottom+top)/2 
        println("β updated = ", β)
        candidate = (tsallis_coef .* (β .+ (1/tsallis_coef .* (probs(ξ).^(q-1))) + (η .* loss_vector))).^(1/(q-1))
        error = 1 - sum(candidate)
        if error > eps
            top = β
        elseif error < -eps
            bottom = β
        else
            println(error)
            return candidate ./ sum(candidate)
        end
        iteration += 1
        # println("bottom = ", bottom)
        # println("top = ", top)
        println("error = ", error)
        println("β = ", β)
    end
end

result = binary_search()
println(result)