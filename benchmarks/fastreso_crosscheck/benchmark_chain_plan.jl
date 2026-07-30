const BENCHMARK_DIR = @__DIR__
const REPOSITORY_ROOT = normpath(joinpath(BENCHMARK_DIR, "..", ".."))

include(joinpath(REPOSITORY_ROOT, "CascadeTables.jl"))

using .CascadeTables
using Printf
using Statistics

environment_int(name, default) = parse(Int, get(ENV, name, string(default)))
environment_float(name, default) =
    parse(Float64, get(ENV, name, string(default)))

function main(arguments)
    length(arguments) ≤ 1 ||
        error("usage: julia benchmark_chain_plan.jl [DECAY_LIST]")
    decay_path = isempty(arguments) ?
        joinpath(BENCHMARK_DIR, "fixtures", "rho_chain.dat") :
        only(arguments)
    decay_list = DecayList(decay_path)
    orders = (
        nq = environment_int("FASTRESO_PLAN_NQ", 128),
        nθ = environment_int("FASTRESO_PLAN_NTHETA", 32),
        nµ = environment_int("FASTRESO_PLAN_NMU", 32),
    )

    # Compile the construction path on a tiny grid before measuring it.
    CascadePlan(
        decay_list;
        grid = range(0.0, 0.2, length = 5),
        modes = [0],
        orders,
        width_cut = Inf,
        cache = nothing,
    )

    point_count = environment_int("FASTRESO_PLAN_POINTS", 401)
    maximum_momentum =
        environment_float("FASTRESO_PLAN_PMAX", 8.5)
    temperature = environment_float("FASTRESO_PLAN_T", 0.145)
    grid = range(0.0, maximum_momentum; length = point_count)

    plan = nothing
    GC.gc()
    build_seconds = @elapsed plan = CascadePlan(
        decay_list;
        grid,
        modes = [0],
        orders,
        width_cut = Inf,
        cache = nothing,
    )
    primary(particle, mode, momenta) = [
        begin
            transverse_mass = hypot(particle.mass, momentum)
            iseven(round(Int, particle.degeneracy)) ?
            particle.degeneracy * transverse_mass /
                ((2π)^3 * (exp(transverse_mass / temperature) + 1.0)) :
            particle.degeneracy * transverse_mass /
                ((2π)^3 * expm1(transverse_mass / temperature))
        end
        for momentum in momenta
    ]

    warm_result = runcascade(plan; primary, output = :all)
    sample_count = environment_int("FASTRESO_PLAN_SAMPLES", 50)
    samples = Float64[]
    for _ in 1:sample_count
        push!(
            samples,
            @elapsed(runcascade(plan; primary, output = :all)),
        )
    end
    parent_order = [plan.ids[step.parent] for step in plan.steps]

    @printf(
        "threads=%d particles=%d grid=%d keys=%d steps=%d\n",
        Threads.nthreads(),
        length(plan.ids),
        length(grid),
        length(plan.table.W),
        length(plan.steps),
    )
    println("parent_order=", parent_order)
    @printf("table_plus_plan_build_s=%.6f\n", build_seconds)
    @printf(
        "execution_median_ms=%.6f min_ms=%.6f max_ms=%.6f samples=%d\n",
        1.0e3 * median(samples),
        1.0e3 * minimum(samples),
        1.0e3 * maximum(samples),
        sample_count,
    )

    if haskey(warm_result.F, 211) && haskey(warm_result.F, -211)
        warm_result.F[211][1] == warm_result.F[-211][1] ||
            error("charge-symmetric chain produced asymmetric pion spectra")
    end
    return 0
end

exit(main(ARGS))
