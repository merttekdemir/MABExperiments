using Test

include(joinpath("..", "src", "MABStruct.jl")); M = MABStructs;
include("test_MABStruct.jl"); U = UnitTests

# Run the tests
runtests(tests=U.tests; ncores=ceil(Int, Sys.CPU_THREADS / 2),
exit_on_error=false, revise=false, seed=42)
