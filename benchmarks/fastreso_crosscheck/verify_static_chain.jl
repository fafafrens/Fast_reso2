const BENCHMARK_DIR = @__DIR__

include(joinpath(BENCHMARK_DIR, "CrosscheckTools.jl"))

using FastReso2
using .CrosscheckTools
using Printf
using Statistics

thermal(mass, degeneracy, temperature) =
    (qT, eta, phi) -> begin
        energy = hypot(mass, qT) * cosh(eta)
        degeneracy * energy / ((2π)^3 * expm1(energy / temperature))
    end

function main(arguments)
    length(arguments) == 1 ||
        error("usage: julia verify_static_chain.jl CHAIN_TSV")
    reference = read_reference(only(arguments))
    reference.metadata["representation"] == "static" ||
        error("chain verification requires a static reference")
    temperature = metadata_float(reference, "temperature")

    nq = environment_int("FASTRESO_CHAIN_NQ", 48)
    ntheta = environment_int("FASTRESO_CHAIN_NTHETA", 12)
    nmu = environment_int("FASTRESO_CHAIN_NMU", 12)
    rules = Rules(; nq, nθ = ntheta, nµ = nmu)
    root_to_rho =
        ThreeBodyDecay(1.700, 0.775, 0.140, 0.140, rules; B = 0.15)
    root_to_pion =
        ThreeBodyDecay(1.700, 0.140, 0.775, 0.140, rules; B = 0.15)
    rho_to_pion =
        TwoBodyDecay(0.775, 0.140, 0.140, rules; B = 1.0)
    root_primary = thermal(1.700, 3.0, temperature)
    rho_primary = thermal(0.775, 3.0, temperature)
    rho_after(qT, eta, phi) =
        rho_primary(qT, eta, phi) +
        root_to_rho(root_primary, qT, eta, phi)
    pion_feed(momentum) =
        root_to_pion(root_primary, momentum, 0.0, 0.0) +
        rho_to_pion(rho_after, momentum, 0.0, 0.0)

    rows = closest_rows(
        reference.spectra[211],
        [0.05, 0.10, 0.20, 0.40, 0.80],
    )
    pion_feed(rows[1].momentum)
    candidates = Float64[]
    elapsed = @elapsed for row in rows
        push!(candidates, pion_feed(row.momentum))
    end
    references = [row.total - row.primary for row in rows]
    errors = abs.((candidates .- references) ./ references)

    println("Static chained functor check")
    @printf(
        "  orders nq=%d ntheta=%d nmu=%d; evaluation %.3f s\n",
        nq,
        ntheta,
        nmu,
        elapsed,
    )
    for (row, candidate, expected, error) in
        zip(rows, candidates, references, errors)
        @printf(
            "  p=%7.5f upstream=% .11e local=% .11e rel=%9.3e\n",
            row.momentum,
            expected,
            candidate,
            error,
        )
    end

    yields = reference.yields
    expected_rho = yields[113][1] + 0.15 * yields[30113][1]
    expected_pion =
        yields[211][1] + yields[113][1] + 0.30 * yields[30113][1]
    rho_yield_error =
        abs(yields[113][2] - expected_rho) / expected_rho
    pion_yield_error =
        abs(yields[211][2] - expected_pion) / expected_pion
    @printf(
        "  yield closure rho=%.3e pion=%.3e\n",
        rho_yield_error,
        pion_yield_error,
    )

    tolerance = environment_float("FASTRESO_CHAIN_RTOL", 1.0e-5)
    passed =
        maximum(errors) ≤ tolerance &&
        max(rho_yield_error, pion_yield_error) ≤ 1.0e-12
    println("static chained functor crosscheck: ", passed ? "PASS" : "FAIL")
    return passed ? 0 : 1
end

exit(main(ARGS))
