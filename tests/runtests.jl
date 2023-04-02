using Test

tests = [
    "test_MABStruct.jl"
    "test_OnlineLearningAlgorithms.jl"
    "test_Utils.jl"
]

for test in tests
    println("Running $test")
    include(test)
end
