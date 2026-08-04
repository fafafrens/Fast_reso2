const BENCHMARK_DIR = @__DIR__
const REPOSITORY_ROOT = normpath(joinpath(BENCHMARK_DIR, "..", ".."))

include(joinpath(BENCHMARK_DIR, "CrosscheckTools.jl"))
include(joinpath(REPOSITORY_ROOT, "CascadeTables.jl"))
include(joinpath(REPOSITORY_ROOT, "tools", "convert_decay_data.jl"))

using .CrosscheckTools
using .CascadeTables
using .DecayDataConverter
using FastGaussQuadrature: gausslegendre
using Printf
using Statistics

environment_int(name, default) = parse(Int, get(ENV, name, string(default)))
environment_float(name, default) =
    parse(Float64, get(ENV, name, string(default)))

function file_fnv1a64(path)
    hash = UInt64(0xcbf29ce484222325)
    for byte in read(path)
        hash = (hash ⊻ UInt64(byte)) * UInt64(0x100000001b3)
    end
    return lowercase(string(hash; base = 16, pad = 16))
end

function canonical_decay_list(azyhydro_path)
    return mktempdir() do directory
        particle_path = joinpath(directory, "particles.dat")
        decay_path = joinpath(directory, "decays.dat")
        convert_azyhydro([azyhydro_path], particle_path, decay_path)
        DecayList(readparticles(particle_path), readdecays(decay_path))
    end
end

function thermal_projector(temperature, eta_maximum, order)
    nodes, weights = gausslegendre(order)
    rapidities = @. eta_maximum * (nodes + 1.0) / 2.0
    scaled_weights = @. eta_maximum * weights / 2.0
    cosh_rapidities = cosh.(rapidities)

    return function (particle, momenta)
        # Upstream deliberately leaves photons uninitialized.
        particle.mass > 0.001 || return zeros(length(momenta))
        degeneracy = round(Int, particle.degeneracy)
        is_fermion = iseven(degeneracy)
        result = Vector{Float64}(undef, length(momenta))
        @inbounds for momentum_index in eachindex(momenta)
            transverse_mass = hypot(
                particle.mass,
                momenta[momentum_index],
            )
            integral = 0.0
            for rapidity_index in eachindex(cosh_rapidities)
                energy =
                    transverse_mass * cosh_rapidities[rapidity_index]
                exponent = energy / temperature
                distribution = exponent > 700.0 ? 0.0 :
                    is_fermion ?
                    inv(exp(exponent) + 1.0) :
                    inv(expm1(exponent))
                integral +=
                    scaled_weights[rapidity_index] *
                    energy *
                    distribution
            end
            # TKernel integrates eta over [-eta_max,eta_max] and the source
            # azimuth over [0,2pi]. The exporter and this callback both add
            # the physical phase-space factor (2pi)^-3.
            result[momentum_index] =
                4π * degeneracy * integral / (2π)^3
        end
        return result
    end
end

function main(arguments)
    length(arguments) == 2 || error(
        "usage: julia verify_projected.jl REFERENCE_TSV DECAY_LIST",
    )
    reference_path, decay_path = arguments
    reference = read_reference(reference_path)
    reference.metadata["representation"] == "projected" ||
        error("verify_projected.jl requires a projected reference")
    grid = reference_grid(reference)
    temperature = metadata_float(reference, "temperature")
    eta_maximum = metadata_float(reference, "eta_max")
    expected_hash = get(reference.metadata, "decay_fnv1a64", nothing)
    expected_hash === nothing &&
        error("reference is missing its decay-list fingerprint")
    actual_hash = file_fnv1a64(decay_path)
    expected_hash == actual_hash ||
        error(
            "reference was generated from a different decay list " *
            "(expected $expected_hash, got $actual_hash)",
        )

    raw_decay_list = canonical_decay_list(decay_path)
    # Upstream FastReso is sharp-parent and implements only 2-/3-body rows.
    decay_list = filter(nbody(2, 3), raw_decay_list)
    projector = thermal_projector(
        temperature,
        eta_maximum,
        environment_int("FASTRESO_PROJECTED_NETA", 128),
    )

    primary_by_id = Dict{Int,Vector{Float64}}()
    primary_seconds = @elapsed for particle_id in keys(decay_list.parts)
        primary_by_id[particle_id] =
            projector(decay_list.parts[particle_id], grid)
    end
    primary(particle, mode, requested_grid) = begin
        mode == 0 || return zeros(length(requested_grid))
        requested_grid == grid ||
            error("primary callback received an unexpected grid")
        return copy(primary_by_id[particle.id])
    end

    orders = (
        nq = environment_int("FASTRESO_PROJECTED_NQ", 128),
        nθ = environment_int("FASTRESO_PROJECTED_NTHETA", 32),
        nµ = environment_int("FASTRESO_PROJECTED_NMU", 32),
        nσ = environment_int("FASTRESO_PROJECTED_NSIGMA", 16),
        nma = environment_int("FASTRESO_PROJECTED_NMA", 16),
    )
    cache_path = get(ENV, "FASTRESO_PROJECTED_CACHE", "")
    cache = isempty(cache_path) ? nothing : cache_path

    table = nothing
    table_seconds = @elapsed table = CascadeTable(
        decay_list;
        grid,
        modes = [0],
        orders,
        width_cut = Inf,
        cache,
    )
    plan = nothing
    plan_seconds = @elapsed plan = CascadePlan(table, decay_list)
    result = nothing
    first_run_seconds = @elapsed result = runcascade(
        plan;
        primary,
        output = :all,
    )
    sample_count = environment_int("FASTRESO_PROJECTED_SAMPLES", 20)
    run_samples = Float64[]
    for _ in 1:sample_count
        push!(
            run_samples,
            @elapsed(runcascade(plan; primary, output = :all)),
        )
    end

    minimum_momentum =
        environment_float("FASTRESO_PROJECTED_PMIN", 0.1)
    maximum_momentum =
        environment_float("FASTRESO_PROJECTED_PMAX", 3.0)
    peak_floor =
        environment_float("FASTRESO_PROJECTED_PEAK_FLOOR", 1.0e-10)
    tolerance =
        environment_float("FASTRESO_PROJECTED_RTOL", 2.0e-3)
    peak_absolute_tolerance =
        environment_float("FASTRESO_PROJECTED_PEAK_ATOL", 1.0e-4)
    primary_tolerance =
        environment_float("FASTRESO_PROJECTED_PRIMARY_RTOL", 1.0e-6)

    @printf(
        "Projected thermal crosscheck\n  particles=%d channels=%d grid=%d keys=%d\n  primary_projection=%.6f s table=%.6f s plan=%.6f s\n  first_run=%.6f ms reused_median=%.6f ms (%d samples)\n\n",
        length(decay_list.parts),
        sum(length, values(decay_list.byparent)),
        length(grid),
        length(table.W),
        primary_seconds,
        table_seconds,
        plan_seconds,
        1.0e3 * first_run_seconds,
        1.0e3 * median(run_samples),
        sample_count,
    )

    passed = true
    momenta = collect(grid)
    for particle_id in sort!(collect(keys(reference.spectra)))
        haskey(result.F, particle_id) ||
            error("local result has no requested PDG $particle_id")
        rows = reference.spectra[particle_id]
        reference_primary = [row.primary for row in rows]
        reference_total = [row.total for row in rows]
        reference_feed = reference_total .- reference_primary
        candidate_primary = primary_by_id[particle_id]
        candidate_total = result.F[particle_id][1]
        candidate_feed = candidate_total .- candidate_primary

        primary_summary = error_summary(
            candidate_primary,
            reference_primary,
            momenta;
            peak_floor,
            minimum_momentum,
            maximum_momentum,
        )
        total_summary = error_summary(
            candidate_total,
            reference_total,
            momenta;
            peak_floor,
            minimum_momentum,
            maximum_momentum,
        )
        feed_summary = error_summary(
            candidate_feed,
            reference_feed,
            momenta;
            peak_floor,
            minimum_momentum,
            maximum_momentum,
        )

        println("PDG ", particle_id)
        print_summary(
            "primary",
            primary_summary,
            momenta,
            candidate_primary,
            reference_primary,
        )
        print_summary(
            "total",
            total_summary,
            momenta,
            candidate_total,
            reference_total,
        )
        print_summary(
            "feed-down",
            feed_summary,
            momenta,
            candidate_feed,
            reference_feed,
        )
        println()

        primary_pass =
            primary_summary.max_relative ≤ primary_tolerance ||
            primary_summary.scaled_max_absolute ≤ primary_tolerance
        total_pass =
            total_summary.max_relative ≤ tolerance ||
            total_summary.scaled_max_absolute ≤ peak_absolute_tolerance
        feed_pass =
            feed_summary.max_relative ≤ tolerance ||
            feed_summary.scaled_max_absolute ≤ peak_absolute_tolerance
        passed &= primary_pass && total_pass && feed_pass
    end

    println("projected CascadePlan crosscheck: ", passed ? "PASS" : "FAIL")
    return passed ? 0 : 1
end

exit(main(ARGS))
