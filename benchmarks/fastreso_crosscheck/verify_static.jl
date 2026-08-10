const BENCHMARK_DIR = @__DIR__

include(joinpath(BENCHMARK_DIR, "CrosscheckTools.jl"))

using FastReso2
using .CrosscheckTools
using Printf
using Statistics

thermal_spectrum(mass, degeneracy, temperature) =
    (qT, eta, phi) -> begin
        energy = hypot(mass, qT) * cosh(eta)
        degeneracy * energy / ((2π)^3 * expm1(energy / temperature))
    end

relative_error(candidate, reference) =
    abs(candidate - reference) / max(abs(reference), floatmin(Float64))

function compare_points(label, dataset, particle_id, local_feed, requested)
    rows = closest_rows(dataset.spectra[particle_id], requested)
    errors = Float64[]
    println("\n", label)
    println(" p [GeV]       upstream feed          local feed       relative error")
    elapsed = @elapsed for row in rows
        reference = row.total - row.primary
        candidate = local_feed(row.momentum)
        error = relative_error(candidate, reference)
        push!(errors, error)
        @printf(
            "%8.5f  %19.11e  %19.11e  %14.6e\n",
            row.momentum,
            reference,
            candidate,
            error,
        )
    end
    central = [
        errors[index] for index in eachindex(rows)
        if rows[index].momentum ≤ 0.41
    ]
    @printf(
        "central max %.6e; through 1.2 GeV max %.6e; %.3f s\n",
        maximum(central),
        maximum(errors),
        elapsed,
    )
    return maximum(central), maximum(errors)
end

function yield_error(label, dataset, expected)
    maximum_error = 0.0
    println("\n", label, " integrated-yield closure")
    for particle_id in sort!(collect(keys(expected)))
        actual = dataset.yields[particle_id][2]
        error = relative_error(expected[particle_id], actual)
        maximum_error = max(maximum_error, error)
        @printf(
            "PDG %6d upstream %.12e expected %.12e rel %.3e\n",
            particle_id,
            actual,
            expected[particle_id],
            error,
        )
    end
    return maximum_error
end

function main(arguments)
    length(arguments) == 3 || error(
        "usage: julia verify_static.jl RHO_TSV OMEGA_TSV COMBINED_TSV",
    )
    rho_data, omega_data, combined_data = read_reference.(arguments)
    all(
        data -> data.metadata["representation"] == "static",
        (rho_data, omega_data, combined_data),
    ) || error("verify_static.jl requires static references")

    temperature = metadata_float(combined_data, "temperature")
    all(
        data -> metadata_float(data, "temperature") == temperature,
        (rho_data, omega_data),
    ) || error("reference temperatures differ")

    primary_errors = Float64[]
    for (particle_id, mass) in ((211, 0.140), (-211, 0.140), (111, 0.135))
        primary = thermal_spectrum(mass, 1.0, temperature)
        for row in combined_data.spectra[particle_id]
            row.primary > 1.0e-20 || continue
            push!(
                primary_errors,
                relative_error(
                    primary(row.momentum, 0.0, 0.0),
                    row.primary,
                ),
            )
        end
    end
    primary_maximum = maximum(primary_errors)
    @printf(
        "Primary mapping: max rel %.6e, median rel %.6e\n",
        primary_maximum,
        median(primary_errors),
    )

    rules = Rules(
        nq = environment_int("FASTRESO_STATIC_NQ", 256),
        nθ = environment_int("FASTRESO_STATIC_NTHETA", 64),
        nµ = environment_int("FASTRESO_STATIC_NMU", 64),
    )
    rho_to_charged =
        TwoBodyDecay(0.775, 0.140, 0.140, rules; B = 1.0)
    omega_to_charged =
        ThreeBodyDecay(0.783, 0.140, 0.140, 0.135, rules; B = 0.8959)
    omega_to_neutral =
        ThreeBodyDecay(0.783, 0.135, 0.140, 0.140, rules; B = 0.8959)
    rho_primary = thermal_spectrum(0.775, 3.0, temperature)
    omega_primary = thermal_spectrum(0.783, 3.0, temperature)

    rho_feed(momentum) =
        rho_to_charged(rho_primary, momentum, 0.0, 0.0)
    omega_charged_feed(momentum) =
        omega_to_charged(omega_primary, momentum, 0.0, 0.0)
    omega_neutral_feed(momentum) =
        omega_to_neutral(omega_primary, momentum, 0.0, 0.0)

    # Compile before timing the comparison points.
    rho_feed(0.2)
    omega_charged_feed(0.2)
    omega_neutral_feed(0.2)

    momenta = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20]
    comparisons = [
        compare_points(
            "rho0 -> pi+ pi-",
            rho_data,
            211,
            rho_feed,
            momenta,
        ),
        compare_points(
            "omega -> charged pion",
            omega_data,
            211,
            omega_charged_feed,
            momenta,
        ),
        compare_points(
            "omega -> neutral pion",
            omega_data,
            111,
            omega_neutral_feed,
            momenta,
        ),
        compare_points(
            "combined -> charged pion",
            combined_data,
            211,
            momentum ->
                rho_feed(momentum) + omega_charged_feed(momentum),
            momenta,
        ),
    ]

    rho_yields = rho_data.yields
    omega_yields = omega_data.yields
    combined_yields = combined_data.yields
    yield_errors = [
        yield_error(
            "rho-only",
            rho_data,
            Dict(
                211 => rho_yields[211][1] + rho_yields[113][1],
                -211 => rho_yields[-211][1] + rho_yields[113][1],
                111 => rho_yields[111][1],
            ),
        ),
        yield_error(
            "omega-only",
            omega_data,
            Dict(
                211 => omega_yields[211][1] +
                       0.8959 * omega_yields[223][1],
                -211 => omega_yields[-211][1] +
                        0.8959 * omega_yields[223][1],
                111 => omega_yields[111][1] +
                       0.8959 * omega_yields[223][1],
            ),
        ),
        yield_error(
            "combined",
            combined_data,
            Dict(
                211 => combined_yields[211][1] +
                       combined_yields[113][1] +
                       0.8959 * combined_yields[223][1],
                -211 => combined_yields[-211][1] +
                        combined_yields[113][1] +
                        0.8959 * combined_yields[223][1],
                111 => combined_yields[111][1] +
                       0.8959 * combined_yields[223][1],
            ),
        ),
    ]

    central_tolerance =
        environment_float("FASTRESO_STATIC_CENTRAL_RTOL", 2.0e-6)
    broad_tolerance =
        environment_float("FASTRESO_STATIC_BROAD_RTOL", 1.0e-5)
    passed =
        primary_maximum ≤ 1.0e-10 &&
        all(result -> result[1] ≤ central_tolerance, comparisons) &&
        all(result -> result[2] ≤ broad_tolerance, comparisons) &&
        maximum(yield_errors) ≤ 1.0e-12
    println("\nstatic crosscheck: ", passed ? "PASS" : "FAIL")
    return passed ? 0 : 1
end

exit(main(ARGS))
