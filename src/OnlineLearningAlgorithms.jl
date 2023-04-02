module OnlineLearningAlgorithms
using Random, Distributions, DataStructures

#------------------------------------------------------------------------------------------------
"""
The following is a collection of algorithms for solving online optimization problems.
These algorithms are used to solve classical reinforment learning problems presented 
by the multi-armed bandit problem. The algorithms attempt to trade off exploration and
exploitation in order to suggest (mixed) policy solutions aiming to minimized the long 
run regret. Depending on the choice of algorithm, it may be used in the full-information
stochastic or naive adversarial setting problem. The algorithms are written such as to apply
an update to the strateg ξ at a given time step of the game τ and are build to interact with
the overarching MABStruct. Note that in the below implementations linear losses are assumed.
"""
#------------------------------------------------------------------------------------------------


    """
    ExponentiatedGradient(ξ::Categorical{Float64, Vector{Float64}}, reward_vector::Vector{Float64}; η=0.01::Float64)

    The Exponentiated Gradient Algorithm under the onlilne mirror descent framework implements the
    negative entorpy mirror map. For more details on the algorithm and its regret bound refer to
    arXiv:1912.13213v5 pg.61. This algorithm assumes full information of the unobserved losses.

    ###Arguments

    - `ξ::Distributions.Categorical{Float64, Vector{Float64}}`: A categorical distribution over the n=|A| possible 
    actions in the bandit game.

    - `reward_vector::Vector{Float64, A}`: A vector representing the observed losses from each of the 'A'
    possible actions in the bandit game for a given round. 

    - `η::Float64`: The learning rate set optimally at √(2*log('A')/T) (Refer to the arXiv:1912.13213v5 pg.51).

    """
    function ExponentiatedGradient(ξ::Categorical{Float64, Vector{Float64}}, reward_vector::Vector{Float64}; η=0.01::Float64)
        (η > 0) || throw(ArgumentError("Only positive values can be accepted for η"))
        loss = -1 .* reward_vector
        #TODO ask CB if it makes sense to use expm1
        updated_probs = probs(ξ) .* exp.(-η.*loss)
        return updated_probs ./ sum(updated_probs)
    end

    """
    FtrlExponentiatedGradient(τ::Int64, cumulative_reward_per_arm::Vector{Float64}; α=1/sqrt(log(length(cumulative_reward_per_arm)))::Float64)
    
    The Exponentiated Gradient under the follow the regularized leader framework implements the negative entorpy
    regularization function. Under linear losses this can be equivalent to the results seen from the OMD framework.
    For more details on the algorithm and its regret bound refer to arXiv:1912.13213v5 pg.70. This algorithm
    assumes full information of the unobserved losses.

    ###Arguments

    - `τ::Int64`: Current iteration/time step of the bandit game. 

    - `cumulative_reward_per_arm::Vector{Float64, A}`: A vector representing the cumulative rewards from each of the 'A'
    possible actions in the bandit game for a given round. 

    - `α::Float64`: The learning rate set optimally at √(2*log('A')/T) (Refer to the arXiv:1912.13213v5 pg.62)

    """
    function FtrlExponentiatedGradient(τ::Int64, cumulative_reward_per_arm::Vector{Float64}; α=1/sqrt(log(length(cumulative_reward_per_arm)))::Float64)
        # Check Arguments
        (α > 0) || throw(ArgumentError("Only positive values can be accepted for α"))
        
        #Regret upper bound of O(1)/η + O(T)/η
        cum_rew_per_arm_normalised = cumulative_reward_per_arm ./ maximum(cumulative_reward_per_arm)

        loss = -1 .* cum_rew_per_arm_normalised
        η = 1 / α*sqrt(τ) # L_∞ is 1 because of linear loss
        #TODO ask CB if it makes sense to use expm1
        updated_probs = exp.(-η.*loss)
        return updated_probs ./ sum(updated_probs)
    end

    """
    EXP3(ξ::Categorical{Float64, Vector{Float64}}, reward_vector::Vector{Float64}, γ::Vector{Int64}, T::Int64, τ::Int64; η=1/sqrt(T)::Float64)

    The Exponential-weight algorithm for Exploration and Exploitation (EXP3) applies the Exponentiated
    Gradient Algorithm in the OMD framework for the bandit information case by using stochastic estimates
    for the unobserved losses. For more details on the algorithm and its regret bound refer to arXiv:1912.13213v5 pg.107.

    ###Arguments

    - `ξ::Distributions.Categorical{Float64, Vector{Float64}}`: A categorical distribution over the n=|A| possible 
    actions in the bandit game.

    - `reward_vector::Vector{Float64, A}`: A vector representing the observed losses from each of the 'A'
    possible actions in the bandit game for a given round. 

    - `γ::Int64`: A value between [1,'A'] representing the action taken in the most recent round of the game.

    - `T::Int64`: Terminal iteration/time step of the bandit game.  

    - `τ::Int64`: Current iteration/time step of the bandit game.

    - `η::Float64`: The learning rate set optimally at 1√T (Refer to the arXiv:1912.13213v5 pg.107).
    """
    function EXP3(ξ::Categorical{Float64, Vector{Float64}}, reward_vector::Vector{Float64}, γ::Vector{Int64}, T::Int64, τ::Int64; η=1/sqrt(T)::Float64)
        # Check Arguments
        (η > 0) || throw(ArgumentError("Only positive values can be accepted for η"))

        # Pick the most recent action reward
        most_recent_action = γ[τ]
        loss = -1.0 * reward_vector[most_recent_action]/ probs(ξ)[most_recent_action]  # observed loss
        loss_vector = zeros(Float64, length(reward_vector))
        loss_vector[most_recent_action] = loss
        return probs(ξ) .* exp.(-η.*loss_vector) ./ sum(probs(ξ) .* exp.(-η.*loss_vector))
    end

    """
    ExploreThenCommit(ξ::Categorical{Float64, Vector{Float64}}, τ::Int64, cumulative_reward_per_arm_bandit::Vector{Float64}, choices_per_arm::Vector{Int64}; m=10::Int64)

    The Explore then commit algorithm is the most natural algorithm for the stochastic bandit
    setting. It involves m*A exploration rounds, where A represents the possible actions in
    the game. After the exploitation phase the algorithm takes a greedy approach selecting
    the best observed action. For more details on the algorithm and its regret bound refer to
     arXiv:1912.13213v5 pg.113.

    ###Arguments

    - `τ::Int64`: Current iteration/time step of the bandit game. 

    - `γ::Int64`: A value between [1,'A'] representing the action taken in the most recent round of the game.

    - `cumulative_reward_per_arm_bandit::Vector{Float64, A}`: A vector representing the cumulative observed
    rewards from each of the 'A' possible actions in the bandit game at a given round.
    
    - `m::Int64`: A hpyerparameter determining the number of exploration steps as m*'A'.

    """
    function ExploreThenCommit(ξ::Categorical{Float64, Vector{Float64}}, τ::Int64, cumulative_reward_per_arm_bandit::Vector{Float64}, choices_per_arm::Vector{Int64}; m=10::Int64)
        # Check Arguments
        (m > 0) || throw(ArgumentError("Only positive values can be accepted for m"))
        
        d = length(cumulative_reward_per_arm_bandit)

        if τ <= d*m
            probs(ξ) .*= 0.0
            probs(ξ)[(τ % d) + 1] = 1.0
            return probs(ξ)
        elseif τ == (d*m + 1)
            probs(ξ) .*= 0.0
            probs(ξ)[argmax(cumulative_reward_per_arm_bandit ./ choices_per_arm)] = 1.0
            return probs(ξ)
        else
            return probs(ξ)
        end 
    end
    
    """
    UpperConfidenceBound(ξ::Categorical{Float64, Vector{Float64}}, τ::Int64, choices_per_arm::Vector{Int64}, average_reward_per_arm_bandit::Vector{Float64}; α=3::Int64)

    The upper confidence bound algorithm improves the explore then commit algorithm by taking
    a data driven approach to smoothly transition from exploration to exploitation. It employs
    the principal of optimism in the face of uncertainty to select in each round the arm with
    the potential to be the best one. For more details on the algorithm and its regret bound 
    refer to arXiv:1912.13213v5 pg.115.

    ###Arguments
    - `ξ::Distributions.Categorical{Float64, Vector{Float64}}`: A categorical distribution over the n=|A| possible 
    actions in the bandit game.

    - `τ::Int64`: Current iteration/time step of the bandit game.

     - `choices_per_arm::Vector{Int64, A}`: A vector representing the number of times each of the 'A' possible
    actions was selected in the game's history.

    - `average_reward_per_arm_bandit::Vector{Float64, A}`: A vector representing the average observed reward 
    for each of the 'A' possible actions in the bandit informaiton setting.

    - `α::Float64`: A tunable hpyerparameter greater than 2

    """
    function UpperConfidenceBound(ξ::Categorical{Float64, Vector{Float64}}, τ::Int64, choices_per_arm::Vector{Int64}, average_reward_per_arm_bandit::Vector{Float64}; α=3::Int64)
        # Check Arguments
        (α > 2) || throw(ArgumentError("Only positive values can be accepted for α"))
         
        probs(ξ) .*= 0.0

        if minimum(choices_per_arm) == 0
            probs(ξ)[argmin(choices_per_arm)] = 1.0
            return probs(ξ)
        end

        lower_confidence_band = -1.0 .* average_reward_per_arm_bandit .- sqrt(2*α*log(τ))./choices_per_arm
        probs(ξ)[argmin(lower_confidence_band)] = 1.0
        return probs(ξ)
    end 

    """
    EpsilonGreedy(ξ::Categorical{Float64, Vector{Float64}}, average_reward_per_arm_bandit::Vector{Float64};ϵ=0.1::Float64)

    The Epsilon Greedy algorithm is an alternative to the explore then commit algorithm that trades off
    exploration and exploitation by choosing one of the 'A' possible actions at random with some
    probability ϵ, and playing the performing action on average with probability 1 - ϵ.

    ###Arguments

    - `ξ::Distributions.Categorical{Float64, Vector{Float64}}`: A categorical distribution over the n=|A| possible 
    actions in the bandit game.

    - `average_reward_per_arm_bandit::Vector{Float64, A}`: A vector representing the average observed reward 
    for each of the 'A' possible actions in the bandit informaiton setting.

    - `ϵ::Float64`: Probability value for ϵ determining the proportion of exploration.

    """
    function EpsilonGreedy(ξ::Categorical{Float64, Vector{Float64}}, average_reward_per_arm_bandit::Vector{Float64};ϵ=0.1::Float64)
        # Check Argument Error
        (ϵ >= 0.0 && ϵ <= 1.0) || throw(ArgumentError("Invalid combination of default parameters"))
        
        probs(ξ) .*= 0.0

        if rand(Uniform(0, 1)) < ϵ
            arm = rand(1:length(average_reward_per_arm_bandit))
            probs(ξ)[arm] = 1.0
            return probs(ξ)
        end
        arm = argmax(average_reward_per_arm_bandit)
        probs(ξ)[arm] = 1.0
        return probs(ξ)
    end

    """
    LinearDecayedEpsilonGreedy(T::Int64, τ::Int64, ξ::Categorical{Float64, Vector{Float64}}, average_reward_per_arm_bandit::Vector{Float64};ϵ_start=1.0::Float64, ϵ_end=0.0::Float64)

    The Linear Decayed Epsilon Greedy algorithm is an extends the epsilon greedy algorithm by
    setting a timestep based linear decay schedule for the exploration probability value ϵ. 
    In this way the algorithm explores more in earlier iterations and exploits more in later 
    iterations.

    ###Arguments

    - `ξ::Distributions.Categorical{Float64, Vector{Float64}}`: A categorical distribution over the n=|A| possible 
    actions in the bandit game.
    
    - `τ::Int64`: Current iteration/time step of the bandit game.

    - `T::Int64`: Maximum iteration/time step of the bandit game.

    - `average_reward_per_arm_bandit::Vector{Float64, A}`: A vector representing the average observed reward 
    for each of the 'A' possible actions in the bandit informaiton setting.
    
    - `ϵ_start::Float64`: Inital probability value for ϵ.
    
    - `ϵ_end::Float64`: Terminal probability value for ϵ.

    """
    function LinearDecayedEpsilonGreedy(T::Int64, τ::Int64, ξ::Categorical{Float64, Vector{Float64}}, average_reward_per_arm_bandit::Vector{Float64};ϵ_start=1.0::Float64, ϵ_end=0.0::Float64)
        # Check Arguments
        ((ϵ_start >= 0.0 && ϵ_start <= 1.0) && (ϵ_end >= 0.0 && ϵ_end <= 1.0) && (ϵ_start >= ϵ_end)) || throw(ArgumentError("Invalid combination of default parameters"))
        
        probs(ξ) .*= 0.0
        ϵ = ϵ_start - ((ϵ_start - ϵ_end)/T)*τ
        if rand(Uniform(0, 1)) < ϵ
            arm = rand(1:length(average_reward_per_arm_bandit))
            probs(ξ)[arm] = 1.0
            return probs(ξ)
        end
        arm = argmax(average_reward_per_arm_bandit)
        probs(ξ)[arm] = 1.0
        return probs(ξ)
    end

    """
    ExpDecayedEpsilonGreedy(T::Int64, τ::Int64, ξ::Categorical{Float64, Vector{Float64}}, average_reward_per_arm_bandit::Vector{Float64};ϵ_start=1.0::Float64, ϵ_end=0.001::Float64)

    The Exponential Decayed Epsilon Greedy algorithm is an extends the epsilon greedy algorithm by
    setting a timestep based exponential decay schedule for the exploration probability value ϵ. 
    In this way the algorithm explores more in earlier iterations and exploits more in later 
    iterations.

    ###Arguments

    - `ξ::Distributions.Categorical{Float64, Vector{Float64}}`: A categorical distribution over the n=|A| possible 
    actions in the bandit game.
    
    - `τ::Int64`: Current iteration/time step of the bandit game.

    - `T::Int64`: Maximum iteration/time step of the bandit game.

    - `average_reward_per_arm_bandit::Vector{Float64, A}`: A vector representing the average observed reward 
    for each of the 'A' possible actions in the bandit informaiton setting.
    
    - `ϵ_start::Float64`: Inital probability value for ϵ.
    
    - `ϵ_end::Float64`: Terminal probability value for ϵ.

    """
    function ExpDecayedEpsilonGreedy(T::Int64, τ::Int64, ξ::Categorical{Float64, Vector{Float64}}, average_reward_per_arm_bandit::Vector{Float64};ϵ_start=1.0::Float64, ϵ_end=0.001::Float64)
        # Check Arguments
        ((ϵ_start >= 0.0 && ϵ_start <= 1.0) && (ϵ_end > 0.0 && ϵ_end <= 1.0) && (ϵ_start >= ϵ_end)) || throw(ArgumentError("Invalid combination of default parameters"))
        probs(ξ) .*= 0
        ϵ = (ϵ_end/ϵ_start)^(τ/T)
        u = rand(Uniform(0, 1))
        if u < ϵ
            arm = rand(1:length(average_reward_per_arm_bandit))
            probs(ξ)[arm] = 1
            return probs(ξ)
        end
        arm = argmax(average_reward_per_arm_bandit)
        probs(ξ)[arm] = 1
        return probs(ξ)
    end

    """
    Hedge(ξ, reward_vector; η)

    The hedge algorithm is an algorithm used in the full information case that updates
    the policy at a given time step of the game by applying a smoothed weighted majority
    over the outputs of the actions.

    ###Arguments

    - `ξ::Distributions.Categorical{Float64, Vector{Float64}}`: A categorical distribution over the n=|A| possible 
    actions in the bandit game.

    - `reward_vector::Vector{Float64, A}`: A vector representing the observed losses from each of the 'A'
    possible actions in the bandit game for a given round. 

    - `η::Float64`: A tunable hpyerparameter for the learning rate.

    """
    function Hedge(ξ::Categorical{Float64, Vector{Float64}}, reward_vector::Vector{Float64}; η=0.1::Float64)
        # https://people.eecs.berkeley.edu/~jiantao/2902021spring/scribe/EE290_Lecture_09.pdf
        # Check argument
        (η > 0) || throw(ArgumentError("Only positive values can be accepted for η"))

        updated_probs = probs(ξ) .* exp.(reward_vector .* η)
        return updated_probs ./ sum(updated_probs)
    end


end #Module