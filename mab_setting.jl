using Random, Distributions


Random.seed!(42)

Base.@kwdef mutable struct BanditGame
    name::String = "Multi-Arm Bandit Experiment"
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

    function BanditGame(T::Int64, A::Tuple, ξ::Categorical, name::String)
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

        return new(name, T, A, ξ, γ, sequence_of_rewards, cumulative_reward_per_arm, average_reward_per_arm, 
        best_fixed_choice, cumulative_reward_fixed, average_reward_fixed, 
        best_dynamic_choice, cumulative_reward_dynamic, average_reward_dynamic, τ)
    end
end
BanditGame(T::Int64, A::Tuple, ξ::Categorical) = BanditGame(T, A, ξ, "Multi-Arm Bandit Experiment")

# Base.getindex(bandit::BanditGame, x::Int64) = getindex(bandit.A, x)

function update_instance(bandit::BanditGame, action::Integer, reward_vector::Tuple)
    bandit.τ += 1
    i = bandit.τ
    bandit.γ[i] = action
    bandit.sequence_of_rewards[i] .= reward_vector
    bandit.cumulative_reward_per_arm .+= reward_vector
    bandit.average_reward_per_arm .= bandit.cumulative_reward_per_arm ./ i
    bandit.best_fixed_choice[i] = argmax(bandit.cumulative_reward_per_arm)
    bandit.cumulative_reward_fixed[i] = bandit.cumulative_reward_per_arm[bandit.best_fixed_choice[i]]
    bandit.average_reward_fixed[i] = bandit.average_reward_per_arm[bandit.best_fixed_choice[i]]
    #TODO define bdc and see if we update elementwise or vectorwise
    # bandit.best_dynamic_choice .= bdc()
    # bandit.cumulative_reward_dynamic = bandit.cumulative_reward_per_arm[bandit.best_fixed_choice]
    # bandit.average_reward_dynamic = zeros(Float64, T)
    return 
end

function pull(bandit::BanditGame)
    return Tuple(rand(bandit.A[i]) for i in 1:length(bandit.A)) 
end

function run_step(bandit::BanditGame)
    #Sample an action from the policy distribution
    action = rand(bandit.ξ)
    reward_vector = pull(bandit)
    update_instance(bandit, action, reward_vector)
end

#TODO fix specific printing for full info case and partial info case
#TODO figure out if we should use IO
#TODO figure out print vs println (base.print vs base.show)
function Base.show(io::IO, bandit::BanditGame)
    println(io, "$(bandit.name): Iteration $(bandit.τ) of $(bandit.T)")
    println(io, "Full Information Case")
    println(io, "Policy: $(bandit.γ)")
    println(io, "History of rewards: $(bandit.sequence_of_rewards)")
    println(io, "cumulative_reward_per_arm: $(bandit.cumulative_reward_per_arm)")
    println(io, "average_reward_per_arm: $(bandit.average_reward_per_arm)")
    println(io, "best_fixed_choice: $(bandit.best_fixed_choice)")
    println(io, "cumulative_reward_fixed: $(bandit.cumulative_reward_fixed)")
    println(io, "average_reward_fixed: $(bandit.average_reward_fixed)")
    println(io, "best_dynamic_choice: $(bandit.best_dynamic_choice)")
    println(io, "cumulative_reward_dynamic: $(bandit.cumulative_reward_dynamic)")
    println(io, "average_reward_dynamic: $(bandit.average_reward_dynamic)")
end


function run(bandit::BanditGame)
    println(bandit)
    for i in 1:bandit.T
        run_step(bandit)
        println(bandit)
        #TODO update policy
    end

end


game = BanditGame(10, (Normal(0.15, 0.7), Normal(0.54, 0.2), Normal(0.38, 0.5)), Distributions.Categorical([1/3, 1/3, 1/3]))
ξ = Distributions.Categorical([1/3, 1/3, 1/3])
# xi_0 = [1 / len(A) for _ in range(len(A))]  # Starting probability distribution
# T = 10

# A = (Normal(2*Random.rand()-1, Random.rand()*1.5),
#     Normal(2*Random.rand()-1, Random.rand()*1.5))

