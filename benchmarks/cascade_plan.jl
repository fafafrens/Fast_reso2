# Benchmark the legacy keyed cascade against the compiled flat CascadePlan.
#
# The default workload uses the complete bundled topology with small, distinct
# deterministic matrices. It measures the full numerical apply and grouping
# while excluding matrix construction and production-cache I/O.
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
    matrices = Dict{ChannelKey, Vector{matrix_type}}()
    transfer_count = Ref(0)

    CascadeTables.eachchannelkey(decay_list) do _parent_id, _channel, _daughter_id, key
        transfer_count[] += 1
        get!(matrices, key) do
            key_index = length(matrices) + 1
            scale = Float32(1.0e-5 * (1 + key_index % 13))
            mode_zero = fill(scale, length(grid), length(grid))
            mode_two = fill(-0.07f0 * scale, length(grid), length(grid))
            matrix_type[
                CascadeTables._packtablematrix(matrix_type, mode_zero),
                CascadeTables._packtablematrix(matrix_type, mode_two),
            ]
        end
    end

    table = CascadeTable(
        grid,
        DEFAULT_ORDERS,
        MODES,
        Dict{Symbol, Any}(:kind => "deterministic benchmark"),
        matrices,
    )
    return table, transfer_count[]
end

function normalized_error(candidate, reference)
    max_error = 0.0
    for particle_id in keys(reference.F)
        for mode_index in eachindex(reference.F[particle_id])
            expected = reference.F[particle_id][mode_index]
            actual = candidate.F[particle_id][mode_index]
            max_error = max(
                max_error,
                maximum(abs.(actual .- expected)) /
                    max(maximum(abs.(expected)), 1.0),
            )
            candidate.Fprim[particle_id][mode_index] ==
                reference.Fprim[particle_id][mode_index] ||
                error("Fprim mismatch")
        end
    end
    return max_error
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

function measure_apply(plan, primary, parallel; samples = SAMPLES)
    function fresh_spectra()
        initialized = CascadeTables._initialize_plan_spectra(
            plan,
            primary,
            eachindex(plan.ids),
        )
        return initialized[3]
    end

    warm = fresh_spectra()
    CascadeTables._apply_plan!(warm, plan, parallel)

    times = Float64[]
    allocations = Int[]
    for _ in 1:samples
        spectra_by_mode = fresh_spectra()
        GC.gc()
        elapsed = @elapsed CascadeTables._apply_plan!(
            spectra_by_mode,
            plan,
            parallel,
        )
        push!(times, elapsed)
    end
    for _ in 1:samples
        spectra_by_mode = fresh_spectra()
        GC.gc()
        allocated = @allocated CascadeTables._apply_plan!(
            spectra_by_mode,
            plan,
            parallel,
        )
        push!(allocations, allocated)
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

decay_list =
    DecayList(joinpath(ROOT, "decays_QM2016Plus_massorder.dat"))
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

reference = runcascade(table, decay_list; primary, output = :all)
serial = runcascade(plan; primary, output = :all)
threaded =
    runcascade(plan; primary, parallel = :modes, output = :all)
serial_error = normalized_error(serial, reference)
threaded_error = normalized_error(threaded, reference)
serial.F == threaded.F ||
    error("serial and mode-threaded plan results are not bitwise equal")
max(serial_error, threaded_error) ≤ 1.0e-12 ||
    error("plan/reference normalized error exceeds 1e-12")

compile_measurement = measure(() -> CascadePlan(table, decay_list))
legacy_measurement = measure(
    () -> runcascade(table, decay_list; primary, output = :all),
)
serial_measurement =
    measure(() -> runcascade(plan; primary, output = :all))
threaded_measurement = measure(
    () -> runcascade(
        plan;
        primary,
        parallel = :modes,
        output = :all,
    ),
)
serial_apply = measure_apply(plan, primary, :serial)
threaded_apply = measure_apply(plan, primary, :modes)

group_count = length(plan.groups)
multi_target_count = count(group -> length(group.targets) > 1, plan.groups)
matrix_bytes =
    group_count * length(MODES) * GRID_LENGTH^2 * sizeof(Float32)

println("CascadePlan deterministic full-topology benchmark")
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
@printf(
    "  correctness: serial %.3e; threaded %.3e; serial/threaded bitwise equal\n",
    serial_error,
    threaded_error,
)

println("\nCompilation (reported separately)")
print_measurement("CascadePlan(table, dl)", compile_measurement)

println("\nPublic runcascade (includes output initialization)")
print_measurement("legacy keyed", legacy_measurement)
print_measurement(
    "flat serial",
    serial_measurement;
    relative_to = legacy_measurement,
)
print_measurement(
    "flat mode-threaded",
    threaded_measurement;
    relative_to = legacy_measurement,
)

println("\nCompiled apply only (fresh spectra initialized outside timing)")
print_measurement("flat serial apply", serial_apply)
print_measurement(
    "flat mode-threaded apply",
    threaded_apply;
    relative_to = serial_apply,
)
@printf(
    "  flat logical matrix traffic: %.2f GiB; %.2f GiB/s serial, %.2f GiB/s threaded\n",
    matrix_bytes / 2.0^30,
    matrix_bytes / serial_apply.time / 2.0^30,
    matrix_bytes / threaded_apply.time / 2.0^30,
)
