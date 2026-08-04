"""
    ThermalModel

Statistical hadronization: thermal number densities at chemical freeze-out and
the yields that follow once resonances decay.

    dl = DecayList(readparticles("data/particles_QM2016Plus.dat"),
                   readdecays("data/decays_QM2016Plus.dat"))
    primary, total, feeddown = thermalyields(dl, 0.1565; muB = 0.0, V = 5300.0)

Densities are integrated directly rather than summed as Bessel functions, which
keeps `QuadGK` as the only dependency.

Functions take any object exposing `.mass`, `.width`, `.degeneracy`, `.baryon`,
`.charge`, `.strangeness`, `.charm` and `.bottomness`, rather than annotating
`::ParticleInfo`. Annotating would force this module to include
`ParticleList.jl` and so define a *second* `ParticleInfo` type, distinct from
the one `CascadeTables` hands out — records from a live decay list would then
fail to dispatch here.
"""
module ThermalModel

using QuadGK: quadgk
using FastGaussQuadrature: gausslegendre

export chemicalpotential, thermaldensity, primaryyields, thermalyields

# The momentum integrand is smooth and exponentially damped, so a fixed
# Gauss-Legendre rule beats adaptive quadrature outright: 64 nodes reach machine
# precision (1.4e-15 against a 1e-12 adaptive reference across every species in
# the tables) at a fifth of the cost and with no per-call allocation. Built once
# at load and read-only thereafter, so threads may share it.
const _MOMENTUM_NODES, _MOMENTUM_WEIGHTS = gausslegendre(64)

# How far past the pole the momentum integral runs, in units of temperature.
# The occupancy is below exp(-60) ~ 1e-26 there, far under Float64 resolution
# relative to the peak.
const _MOMENTUM_CUTOFF = 60.0

"Hadrons with half-integer spin are fermions; baryon number identifies them."
_isfermion(particle) = particle.baryon != 0

"""
    chemicalpotential(particle; muB, muS = 0.0, muQ = 0.0, muC = 0.0)

Chemical potential of one species, `B*muB + S*muS + Q*muQ + C*muC`.

At LHC energies `muB` is consistent with zero and the others are fixed by
strangeness neutrality and the colliding nuclei's charge-to-baryon ratio;
they are exposed so a finite-`muB` calculation can supply them.
"""
chemicalpotential(particle; muB, muS = 0.0, muQ = 0.0, muC = 0.0) =
    particle.baryon * muB +
    particle.strangeness * muS +
    particle.charge * muQ +
    particle.charm * muC

"""
    _densityatmass(mass, degeneracy, is_fermion, temperature, mu; quantum, rtol)

    n = g/(2 pi^2) * int_0^inf p^2 dp / (exp((E - mu)/T) + eta)

with `eta = +1` for fermions, `-1` for bosons and `0` for Boltzmann statistics.
"""
function _densityatmass(
        mass, degeneracy, is_fermion, temperature, mu;
        quantum = true,
    )
    mass <= 0 && return 0.0
    eta = quantum ? (is_fermion ? 1.0 : -1.0) : 0.0

    # Truncate where the occupancy has died rather than mapping [0, Inf): the
    # cutoff is set by energy, so it adapts to both the mass and the chemical
    # potential instead of assuming a momentum scale.
    energy_cutoff = max(mass, mu) + _MOMENTUM_CUTOFF * temperature
    momentum_maximum = sqrt(max(
        energy_cutoff^2 - mass^2, (_MOMENTUM_CUTOFF * temperature)^2,
    ))
    half_width = momentum_maximum / 2

    integral = 0.0
    @inbounds for index in eachindex(_MOMENTUM_NODES)
        momentum = half_width * (_MOMENTUM_NODES[index] + 1)
        exponent = (hypot(mass, momentum) - mu) / temperature
        # exp overflows around 709; the occupancy is zero long before that, and
        # for bosons the exp(x) - 1 form loses the cancellation that expm1 keeps.
        exponent > 700 && continue
        occupancy = eta == 0 ? exp(-exponent) :
                    eta > 0 ? inv(exp(exponent) + 1) : inv(expm1(exponent))
        integral += _MOMENTUM_WEIGHTS[index] * half_width * momentum^2 * occupancy
    end
    return degeneracy * integral / (2π^2)
end

"""
    thermaldensity(particle, temperature; muB = 0.0, quantum = true,
                   width = true, threshold = 0.0, k = 4.0)

Number density of one species in GeV^3.

`quantum = false` selects Boltzmann statistics. `width = true` averages the
density over a relativistic Breit-Wigner of the species' own width instead of
using the pole mass.

`rtol` controls only the mass integral; the momentum integral uses the fixed
64-node rule, which is already at machine precision. Adaptive quadrature is kept
here because the integrand is a Lorentzian times a steeply falling density, and
fixed rules converge badly on it — a tangent substitution that flattens the
Breit-Wigner, as `DecaySpectra.parentmassnodes` uses, only moves the steepness
into the endpoints (2e-2 at 24 nodes). The default `1e-6` costs `5e-9`, eight
orders below the uncertainty in the branching ratios these yields are built
from. Tighten it if you are checking convergence rather than doing physics.

`threshold` is the lowest mass the resonance can carry — the summed daughter
masses of its lightest channel. The Breit-Wigner is truncated there rather than
at `mR - k*Gamma`, because a constant-width shape carries weight below the
decay threshold: `spectral_function_plan.md` measures 2.56% of the sigma's
weight below `2*m_pi`. For a density that weight is unphysical mass, so it is
removed and the remainder renormalized.
"""
function thermaldensity(
        particle, temperature;
        muB = 0.0, muS = 0.0, muQ = 0.0, muC = 0.0,
        quantum = true, width = true, threshold = 0.0, k = 4.0,
        rtol = 1.0e-6,
    )
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    mu = chemicalpotential(particle; muB, muS, muQ, muC)
    is_fermion = _isfermion(particle)
    pole_density() = _densityatmass(
        particle.mass, particle.degeneracy, is_fermion, temperature, mu; quantum,
    )

    (!width || particle.width <= 0) && return pole_density()

    lower = max(particle.mass - k * particle.width, threshold)
    upper = particle.mass + k * particle.width
    lower < upper || return pole_density()

    # Relativistic Breit-Wigner in the mass variable, normalized over whatever
    # support survives the threshold cut.
    shape(m) = m * particle.width /
        ((m^2 - particle.mass^2)^2 + (particle.mass * particle.width)^2)
    weight, _ = quadgk(shape, lower, upper; rtol)
    weight > 0 || return pole_density()
    averaged, _ = quadgk(lower, upper; rtol) do m
        shape(m) * _densityatmass(
            m, particle.degeneracy, is_fermion, temperature, mu; quantum,
        )
    end
    return averaged / weight
end

"""
    _thresholds(decay_list) -> Dict{Int,Float64}

Lowest reachable mass per parent: the smallest summed daughter mass over its
channels. Self-channels are skipped — they mark a terminal species, not a decay.
"""
function _thresholds(decay_list)
    catalog = decay_list.catalog.byid
    thresholds = Dict{Int, Float64}()
    for (parent_id, channels) in decay_list.network.byparent
        best = Inf
        for channel in channels
            daughters = channel.daughters
            length(daughters) == 1 && only(daughters) == parent_id && continue
            total = 0.0
            for daughter_id in daughters
                haskey(catalog, daughter_id) || (total = Inf; break)
                total += catalog[daughter_id].mass
            end
            best = min(best, total)
        end
        isfinite(best) && (thresholds[parent_id] = best)
    end
    return thresholds
end

"""
    primaryyields(decay_list, temperature; muB = 0.0, V, kwargs...)

Primary (directly produced) yields, `n_i * V`, keyed by particle ID. `V` is in
fm^3 and converted with hbar*c = 0.1973269804 GeV fm.
"""
function primaryyields(decay_list, temperature; V, kwargs...)
    thresholds = _thresholds(decay_list)
    volume = V / (0.1973269804)^3                       # fm^3 -> GeV^-3
    return Dict(
        particle_id => volume * thermaldensity(
            particle, temperature;
            threshold = get(thresholds, particle_id, 0.0), kwargs...,
        )
        for (particle_id, particle) in decay_list.catalog.byid
    )
end

"""
    thermalyields(decay_list, temperature; muB = 0.0, V, kwargs...)
        -> (primary, total, feeddown)

Yields before and after resonances decay, plus their difference.

Parents are processed in descending mass, which is sufficient ordering because
a parent always outweighs its daughters — `validate` enforces exactly that
threshold condition — so a parent's total is final before it is distributed.
This deliberately avoids `CascadeTables._cascadeorder`, which solves the harder
problem of matrix application order among broad parents that reopen each other.

Daughters are credited with multiplicity, so `eta -> 3 pi0` adds three pions.
"""
function thermalyields(decay_list, temperature; V, kwargs...)
    primary = primaryyields(decay_list, temperature; V, kwargs...)
    total = copy(primary)
    catalog = decay_list.catalog.byid

    parents = sort!(
        collect(keys(decay_list.network.byparent));
        by = parent_id -> (-catalog[parent_id].mass, parent_id),
    )
    for parent_id in parents
        available = get(total, parent_id, 0.0)
        available == 0 && continue
        for channel in decay_list.network.byparent[parent_id]
            daughters = channel.daughters
            length(daughters) == 1 && only(daughters) == parent_id && continue
            contribution = channel.br * available
            for daughter_id in daughters
                haskey(total, daughter_id) || continue
                total[daughter_id] += contribution
            end
        end
    end

    feeddown = Dict(id => total[id] - primary[id] for id in keys(primary))
    return (; primary, total, feeddown)
end

end # module ThermalModel
