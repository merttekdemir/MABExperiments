using Test

# Define the list of tests to perform
tests = [
    "test_MABStruct.jl"
    "test_OnlineLearningAlgorithms.jl"
    "test_Utils.jl"
]

# Iterate over the tests, including them will run all tests
for test in tests
    println("Running $test")
    include(test)
end
