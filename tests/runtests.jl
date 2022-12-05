using Test

include(joinpath("..", "src", "MABStruct.jl")); M = MABStructs;


# Run the tests
runtests(tests=t.tests; ncores=ceil(Int, Sys.CPU_THREADS / 2),
exit_on_error=false, revise=false, seed=42)
