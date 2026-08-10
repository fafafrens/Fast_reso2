# Transitional package entry point. The numerical suite remains at the
# repository root until its large `Test.@testset` migration is done separately.
include(joinpath(@__DIR__, "..", "runtests.jl"))
