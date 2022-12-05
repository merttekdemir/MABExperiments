using Random, Distributions

include("MABStruct.jl"); M = MABStructs;

A = (Normal(0.15, 0.7), Normal(0.54, 0.2), Normal(0.38, 0.5))
ξ = Categorical([1/3, 1/3, 1/3])
game = M.MABStruct(10, A, ξ)

# xi_0 = [1 / len(A) for _ in range(len(A))]  # Starting probability distribution
# T = 10

# A = (Normal(2*Random.rand()-1, Random.rand()*1.5),
#     Normal(2*Random.rand()-1, Random.rand()*1.5))

