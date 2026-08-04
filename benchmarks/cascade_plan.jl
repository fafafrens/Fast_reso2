# Benchmark cascade plan construction, reuse, and mode-level parallelism.
#
# The default workload uses the complete bundled topology with small, distinct
# deterministic packed matrices. Synthetic matrix construction and
# production-cache I/O are setup costs outside the measurements.
#
#   julia --project -t 2 benchmarks/cascade_plan.jl
#   CASCADE_BENCH_GRID=32 CASCADE_BENCH_SAMPLES=5 \
#       julia --project -t 2 benchmarks/cascade_plan.jl

isdefined(Main, :CascadeTables) ||
    include(joinpath(dirname(@__DIR__), "CascadeTables.jl"))

using .CascadeTables
using LinearAlgebra
using Printf
using Statistics

BLAS.set_num_threads(1)

const ROOT = dirname(@__DIR__)
const GRID_LENGTH = parse(Int, get(ENV, "CASCADE_BENCH_GRID", "16"))
const SAMPLES = parse(Int, get(ENV, "CASCADE_BENCH_SAMPLES", "5"))
const MODES = (0, 2)

GRID_LENGTH ≥ 2 || error("CASCADE_BENCH_GRID must be at least 2")
SAMPLES ≥ 1 || error("CASCADE_BENCH_SAMPLES must be positive")

function deterministic_table(decay_list, grid)
    matrix_type = CascadeTables._tablematrixtype(length(grid))
    matrices =
        Dict{ChannelKey, NTuple{length(MODES), matrix_type}}()
    transfer_count = Ref(0)

    parent_order = sort!(collect(keys(decay_list.byparent)))
    CascadeTables.eachchannelkey(
        decay_list;
        order = parent_order,
    ) do _parent_id, _channel, _daughter_id, key
        transfer_count[] += 1
        get!(matrices, key) do
            key_index = length(matrices) + 1
            scale = Float32(1.0e-5 * (1 + key_index % 13))
            mode_zero = fill(scale, length(grid), length(grid))
            mode_two = fill(-0.07f0 * scale, length(grid), length(grid))
            (
                CascadeTables._packtablematrix(matrix_type, mode_zero),
                CascadeTables._packtablematrix(matrix_type, mode_two),
            )
        end
    end

    table = CascadeTable(
        grid,
        (nq = 32, nθ = 8, nµ = 16, nσ = 8, nma = 16),
        MODES,
        1.0e-3,
        4.0,
        Dict{Symbol, Any}(:kind => "deterministic benchmark"),
        matrices,
    )
    return table, transfer_count[]
end

function measure(run; samples = SAMPLES)
    run()
    times = Float64[]
    allocations = Int[]
    for _ in 1:samples
        GC.gc()
        push!(times, @elapsed run())
    end
    for _ in 1:samples
        GC.gc()
        push!(allocations, @allocated run())
    end
    return (time = median(times), bytes = median(allocations))
end

function print_measurement(label, measurement; relative_to = nothing)
    speedup =
        relative_to === nothing ? "" :
        @sprintf("  %6.2fx", relative_to.time / measurement.time)
    return @printf(
        "  %-26s %9.3f ms  %9.3f MiB%s\n",
        label,
        1.0e3 * measurement.time,
        measurement.bytes / 2.0^20,
        speedup,
    )
end

decay_list = DecayList(
    readparticles(joinpath(ROOT, "data", "particles_QM2016Plus.dat")),
    readdecays(joinpath(ROOT, "data", "decays_QM2016Plus.dat")),
)
grid = range(0.0, 4.0, length = GRID_LENGTH)
table, transfer_count = deterministic_table(decay_list, grid)
plan = CascadePlan(table, decay_list)

primary(particle, mode, momenta) = begin
    mode_scale = mode == 0 ? 1.0 : 0.05
    [
        mode_scale * particle.degeneracy *
            exp(-hypot(particle.mass, momentum) / 0.145)
            for momentum in momenta
    ]
end

wrapped = runcascade(table, decay_list; primary, output = :all)
serial = runcascade(plan; primary, output = :all)
modes_result =
    runcascade(plan; primary, parallel = :modes, output = :all)
wrapped.F == serial.F ||
    error("table wrapper and reused serial plan results differ")
wrapped.Fprim == serial.Fprim ||
    error("table wrapper and reused serial primaries differ")
serial.F == modes_result.F ||
    error("serial and :modes plan results differ")
serial.Fprim == modes_result.Fprim ||
    error("serial and :modes primaries differ")

plan_measurement = measure(() -> CascadePlan(table, decay_list))
wrapper_measurement = measure(
    () -> runcascade(table, decay_list; primary, output = :all),
)
serial_measurement =
    measure(() -> runcascade(plan; primary, output = :all))
modes_measurement = measure(
    () -> runcascade(
        plan;
        primary,
        parallel = :modes,
        output = :all,
    ),
)

group_count = length(plan.groups)
multi_target_count = count(group -> length(group.targets) > 1, plan.groups)
modes_execution =
    Threads.nthreads() > 1 && length(MODES) > 1 ?
    "parallel" :
    "serial fallback"

println("Cascade pipeline deterministic full-topology benchmark")
@printf(
    "  Julia threads: %d; BLAS threads: %d; grid: %d; modes: %s; samples: %d\n",
    Threads.nthreads(),
    BLAS.get_num_threads(),
    GRID_LENGTH,
    repr(MODES),
    SAMPLES,
)
@printf(
    "  particles: %d; keys: %d; transfers: %d; groups: %d; targets: %d; multi-target: %d\n",
    length(plan.ids),
    length(table.W),
    transfer_count,
    group_count,
    length(plan.targets),
    multi_target_count,
)
println(
    "  consistency: table wrapper, reused serial, and :modes " *
    "($modes_execution) outputs are bitwise equal",
)

println("\nPlan compilation (construction only)")
print_measurement("CascadePlan(table, dl)", plan_measurement)

println("\nPublic runcascade (includes spectrum initialization and output)")
print_measurement("table wrapper: compile + run", wrapper_measurement)
print_measurement(
    "reused plan: serial",
    serial_measurement;
    relative_to = wrapper_measurement,
)
print_measurement(
    "reused plan: modes ($modes_execution)",
    modes_measurement;
    relative_to = serial_measurement,
)
