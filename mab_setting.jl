using Random, Distributions

Random.seed!(42)

Base.@kwdef mutable struct BanditGame 
    T::Int64
    A::Tuple
    ξ::Categorical
    γ::Vector{Int8} = zeros(Int8, T)
    sequence_of_rewards::Vector{Vector{Float64}} = [zeros(Float64, length(A)) for _ in 1:T]
    cumulative_reward_per_arm::Vector{Float64} = zeros(Float64, length(A))
    average_reward_per_arm::Vector{Float64} = zeros(Float64, length(A))
    best_fixed_choice::Vector{Int8} = zeros(Int8, T)
    cumulative_reward_fixed::Vector{Float64} = zeros(Float64, T)
    average_reward_fixed::Vector{Float64} = zeros(Float64, T)   
    best_dynamic_choice::Vector{Int8} = zeros(Int8, T)
    cumulative_reward_dynamic::Vector{Float64} = zeros(Float64, T)
    average_reward_dynamic::Vector{Float64} = zeros(Float64, T)
    τ::Int64 = 0

    function BanditGame(T::Int64, A::Tuple, ξ::Categorical)
        # Sanity Checks
        (all(typeof(a) <: Distribution for a in A) 
        && length(A)==ncategories(ξ)
        ) || throw(ArgumentError("Error in construction"))
        
        # Initialised values
        γ = zeros(Int8, T)
        sequence_of_rewards = [zeros(Float64, length(A)) for _ in 1:T]
        cumulative_reward_per_arm = zeros(Float64, length(A))
        average_reward_per_arm = zeros(Float64, length(A))
        best_fixed_choice = zeros(Int8, T)
        cumulative_reward_fixed = zeros(Float64, T)
        average_reward_fixed = zeros(Float64, T)
        best_dynamic_choice = zeros(Int8, T)
        cumulative_reward_dynamic = zeros(Float64, T)
        average_reward_dynamic = zeros(Float64, T)
        τ = 0

        return new(T, A, ξ, γ, sequence_of_rewards, cumulative_reward_per_arm, average_reward_per_arm, 
        best_fixed_choice, cumulative_reward_fixed, average_reward_fixed, 
        best_dynamic_choice, cumulative_reward_dynamic, average_reward_dynamic, τ)
    end
end

# Base.getindex(bandit::BanditGame, x::Int64) = getindex(bandit.A, x)

function update_instance(bandit::BanditGame, action::Integer, result::Vector{Float64})
    bandit.τ += 1
    i = bandit.τ
    bandit.γ[i] = action
    bandit.sequence_of_rewards[i] = result
    bandit.cumulative_reward_per_arm .+= rewards
    bandit.average_reward_per_arm .= bandit.cumulative_reward_per_arm ./ i
    bandit.best_fixed_choice = argmax(bandit.cumulative_reward_per_arm)
    bandit.cumulative_reward_fixed = bandit.cumulative_reward_per_arm[bandit.best_fixed_choice]
    bandit.average_reward_fixed = bandit.average_reward_per_arm[bandit.best_fixed_choice]
    bandit.best_dynamic_choice .= bdc()
    bandit.cumulative_reward_dynamic = bandit.cumulative_reward_per_arm[bandit.best_fixed_choice]
    bandit.average_reward_dynamic = zeros(Float64, T)
    τ = 0
    return 
end

function pull(bandit::BanditGame)
    return (rand(bandit.A[i]) for i in 1:length(bandit.A)) 
end

function run_step(bandit::BanditGame)
    action = rand(bandit.ξ)
    reward_vector = pull
    update_instance(bandit, action, result)
end

function run()
end


game = BanditGame(10, (Normal(0.15, 0.7), Normal(0.54, 0.2), Normal(0.38, 0.5)), Distributions.Categorical([1/3, 1/3, 1/3]))
ξ = Distributions.Categorical([1/3, 1/3, 1/3])
# xi_0 = [1 / len(A) for _ in range(len(A))]  # Starting probability distribution
# T = 10

# A = (Normal(2*Random.rand()-1, Random.rand()*1.5),
#     Normal(2*Random.rand()-1, Random.rand()*1.5))
