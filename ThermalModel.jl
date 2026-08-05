"""
    ThermalModel

Statistical hadronization: thermal number densities at chemical freeze-out and
the yields that follow once resonances decay.

    dl = DecayList(readparticles("data/particles_QM2016Plus.dat"),
                   readdecays("data/decays_QM2016Plus.dat"))
    primary, total, feeddown = thermalyields(dl, 0.1565; muB = 0.0, V = 5300.0)

For a fit, compile the list once and reuse it -- `ThermalPlan` hoists
everything independent of `(T, muB, V)`, precomputes one spectral quadrature
rule per broad species, and chunks the density loop across threads:

    plan = ThermalPlan(dl)
    yields = thermalyields(plan, T; muB, V)      # ~0.7 ms vs ~50 ms direct

Densities are integrated directly rather than summed as Bessel functions, which
keeps `QuadGK` as the only dependency.

Broad resonances are averaged over a spectral shape. `shape = :massdep` (the
default) uses the threshold-vanishing `MassDependentBW`; `:constant` the
constant-width `BreitWigner`. Both are `DecaySpectra`'s own types, so the
cascade and the yields average over the same shape by construction -- what
differs is only the quadrature each drives it through.

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

# The spectral shapes are DecaySpectra's, not copies of them. Including it here
# does create a second instance of that module -- but unlike `ParticleList`,
# nothing of it crosses this boundary: shapes are built and consumed inside
# this module, and the public API speaks in Symbols. What the shared source
# buys is that "the cascade and the yields average over one shape" is true by
# construction rather than by a test comparing two transcriptions.
include(joinpath(@__DIR__, "DecaySpectra.jl"))
using .DecaySpectra: SpectralShape, BreitWigner, MassDependentBW
import .DecaySpectra: _lowerendpointvanishes

export chemicalpotential, thermaldensity, primaryyields, thermalyields,
       ThermalPlan

# The momentum integrand is smooth and exponentially damped, so a fixed
# Gauss-Legendre rule beats adaptive quadrature outright, with no per-call
# allocation. Built once at load and read-only thereafter, so threads share it.
#
# The order is set by what the *result* needs, not by what the rule can reach.
# Measured end to end against a 128-node reference, over a spread of species:
#
#     nodes   max rel dev   us/density
#        16      1.09e-03        0.317
#        32      2.25e-08        0.442
#        48      1.71e-12        0.635
#        64      2.99e-13        0.706
#
# 32 sits a hundredfold inside the 1e-6 the mass integral is held to, and a
# further six orders inside the uncertainty on the branching ratios these
# yields are built from. The higher orders buy precision nothing downstream
# can use, at 1.6x the cost of the innermost loop in the model.
const _MOMENTUM_NODES, _MOMENTUM_WEIGHTS = gausslegendre(32)

# How far past the pole the momentum integral runs, in units of temperature.
# The occupancy is below exp(-60) ~ 1e-26 there, far under Float64 resolution
# relative to the peak.
const _MOMENTUM_CUTOFF = 60.0

"Hadrons with half-integer spin are fermions; baryon number identifies them."
_isfermion(particle) = particle.baryon != 0

"""
    _spectralshape(shape, mass, width, threshold, lower, upper) -> SpectralShape

The mass distribution one resonance is averaged over, on the window
`[lower, upper]`. `:massdep` needs q at the pole, which a species whose pole
sits at or below its own threshold does not have; those reopened states fall
back to the constant shape, exactly as `CascadeTables.decayfor` decides per key.
"""
function _spectralshape(shape, mass, width, threshold, lower, upper)
    shape === :massdep || shape === :constant || throw(ArgumentError(
        "shape must be :massdep or :constant, got $(repr(shape))",
    ))
    if shape === :massdep && threshold < mass
        return MassDependentBW(mass, width, threshold; mmin = lower, mmax = upper)
    end
    return BreitWigner(mass, width; mmin = lower, mmax = upper)
end

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
density over the species' spectral shape instead of using the pole mass;
`shape` picks it. `:massdep` (default) carries the threshold-vanishing
mass-dependent width `Γ(m) ∝ (mR/m)·q(m)/q(mR)` with the symmetric-split
`q(m) = √(m² − threshold²)/2`; `:constant` the constant-width Breit-Wigner.
With `threshold = 0` the two differ only by the frozen width's placement, and
`:massdep` degenerates to a constant width exactly.

`rtol` controls only the mass integral; the momentum integral uses the fixed
32-node rule, which is already far inside every other tolerance here. This
reference path keeps adaptive quadrature; `ThermalPlan` instead precomputes a
rule per species that carries the shape as its weight function, and the suite
holds the two paths together within the model tolerance.

`threshold` is the lowest mass the resonance can carry — the summed daughter
masses of its lightest channel. Both shapes truncate their support there
(`spectral_function_plan.md` measures 2.56% of the sigma's constant-width
weight below `2*m_pi`; for a density that weight is unphysical mass), and
`:massdep` additionally vanishes there by construction.
"""
function thermaldensity(
        particle, temperature;
        muB = 0.0, muS = 0.0, muQ = 0.0, muC = 0.0,
        quantum = true, width = true, shape = :massdep,
        threshold = 0.0, k = 4.0, rtol = 1.0e-6,
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

    # The spectral shape, normalized over whatever support survives the
    # threshold cut.
    spectral = _spectralshape(
        shape, particle.mass, particle.width, threshold, lower, upper)
    weight, _ = quadgk(spectral, lower, upper; rtol)
    weight > 0 || return pole_density()
    averaged, _ = quadgk(lower, upper; rtol) do m
        spectral(m) * _densityatmass(
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
            # Zero-branching rows are skipped, matching
            # CascadeTables.eachchannelkey -- the two modules must agree on
            # every species' threshold or the shapes stop being one shape.
            iszero(channel.br) && continue
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

# ---------------------------------------------------------------------------
# Compiled plan
# ---------------------------------------------------------------------------

"Mass window a resonance is averaged over: `mR +- k*Gamma`, cut at threshold."
function _support(record, threshold, k)
    record.width > 0 || return (0.0, 0.0)
    return (max(record.mass - k * record.width, threshold),
            record.mass + k * record.width)
end

# Dense rules used only when building the plan, to integrate the spectral
# shape against each Lagrange basis function. Their order is irrelevant at
# evaluation. The cosine-mapped variant serves shapes that vanish like a
# square root at the lower endpoint (:massdep at its threshold): the sin
# Jacobian absorbs the vanishing that plain Gauss-Legendre resolves slowly.
const _BUILD_NODES, _BUILD_WEIGHTS = gausslegendre(256)
const _BUILD_COS_NODES, _BUILD_COS_WEIGHTS = let
    angles = π .* (_BUILD_NODES .+ 1) ./ 2
    (-cos.(angles), π / 2 .* _BUILD_WEIGHTS .* sin.(angles))
end

"""
    _spectralrule(spectral::SpectralShape, order) -> (nodes, weights)

A quadrature rule carrying the spectral shape as its weight function:
`sum(w .* f.(nodes))` approximates `int S(m) f(m) dm / int S(m) dm` for smooth
`f`. Nodes are Chebyshev points, weights are `int S(m) l_j(m) dm` with `l_j` the
Lagrange basis through them, so the rule is exact wherever the degree-`order`
interpolant of `f` is.

Handing the peak to the weights is what makes this cheap: adaptive quadrature
subdivides to resolve the Lorentzian and pays a full density evaluation at every
point it adds — 150 per resonance — although the density itself is smooth there.

Nothing here depends on temperature or chemical potential, so a fit builds it
once. The accuracy floor is polynomial approximation of the density across a
window spanning three decades. A shape that vanishes like `√(m − lower)` at its
lower endpoint costs roughly an order there — it asks for the cosine-mapped
build sweep through `_lowerendpointvanishes`, the same trait
`DecaySpectra.parentmassnodes` dispatches on. Against the adaptive reference
the plan deviates by 1.3e-6 at 24 nodes and 3.6e-8 at 32, where it meets the
reference's own tolerance and stops improving — hence the default of 32.
"""
function _spectralrule(spectral::SpectralShape, order)
    lower, upper = DecaySpectra.support(spectral)
    vanishing = _lowerendpointvanishes(spectral)
    half = (upper - lower) / 2
    midpoint = (upper + lower) / 2
    reference = [cos((2j - 1) * π / (2order)) for j in 1:order]
    nodes = midpoint .+ half .* reference
    # Barycentric weights for Chebyshev points of the first kind.
    barycentric = [(-1)^j * sin((2j - 1) * π / (2order)) for j in 1:order]

    build_nodes = vanishing ? _BUILD_COS_NODES : _BUILD_NODES
    build_weights = vanishing ? _BUILD_COS_WEIGHTS : _BUILD_WEIGHTS

    weights = zeros(order)
    basis = zeros(order)
    for index in eachindex(build_nodes)
        x = build_nodes[index]
        weight = build_weights[index] * half * spectral(midpoint + half * x)

        # Barycentric Lagrange basis at x, guarding the node-coincident case.
        hit = findfirst(j -> x == reference[j], 1:order)
        if hit === nothing
            total = 0.0
            for j in 1:order
                basis[j] = barycentric[j] / (x - reference[j])
                total += basis[j]
            end
            basis ./= total
        else
            fill!(basis, 0.0)
            basis[hit] = 1.0
        end
        for j in 1:order
            weights[j] += weight * basis[j]
        end
    end
    total = sum(weights)
    return nodes, (total > 0 ? weights ./ total : fill(1 / order, order))
end

"""
    ThermalPlan(decay_list; mass_order = 32, k = 4.0, shape = :massdep)

Everything about a decay list that does not depend on `(T, muB, V)`, resolved
once: dense species indices, the quantum numbers each density needs, the decay
threshold per resonance, and the feed-down order with its channels flattened
into offset arrays.

Built for fitting, where `thermalyields` runs thousands of times over the same
list. Roughly a fifth of an unplanned call is this setup — the mass-descending
parent sort alone is 8.5 ms of 47 ms on the production list — and the density
loop, which is the rest, parallelizes over species.

    plan = ThermalPlan(decay_list)
    for (T, muB, V) in grid
        yields = thermalyields(plan, T; muB, V)
    end
"""
struct ThermalPlan
    ids::Vector{Int}
    mass::Vector{Float64}
    width::Vector{Float64}
    degeneracy::Vector{Float64}
    isfermion::Vector{Bool}
    baryon::Vector{Float64}
    strangeness::Vector{Float64}
    charge::Vector{Float64}
    charm::Vector{Float64}
    threshold::Vector{Float64}
    # Feed-down, in application order. `channels[parent]` spans
    # `channel_range[parent]`, and channel `c` sends `branching[c]` to the dense
    # daughter indices in `daughters[daughter_range[c]]`.
    order::Vector{Int}
    channel_range::Vector{UnitRange{Int}}
    branching::Vector{Float64}
    daughter_range::Vector{UnitRange{Int}}
    daughters::Vector{Int}
    # A quadrature rule per broad species that carries the Breit-Wigner as its
    # weight function, so a call evaluates the density at `mass_nodes` and takes
    # a dot product instead of driving an adaptive integral through the peak.
    # `mass_range[i]` is empty for a sharp species. Weights are normalized, so
    # no separate normalization is needed.
    mass_range::Vector{UnitRange{Int}}
    mass_nodes::Vector{Float64}
    mass_weights::Vector{Float64}
    # The spectral shape the rules above were built for; evaluation never
    # rebuilds them, so a plan answers for exactly one shape.
    shape::Symbol
end

function ThermalPlan(decay_list; mass_order = 32, k = 4.0, shape = :massdep)
    shape === :massdep || shape === :constant || throw(ArgumentError(
        "shape must be :massdep or :constant, got $(repr(shape))",
    ))
    catalog = decay_list.catalog.byid
    ids = sort!(collect(keys(catalog)))
    index_of = Dict(particle_id => index for (index, particle_id) in pairs(ids))
    records = [catalog[particle_id] for particle_id in ids]
    thresholds = _thresholds(decay_list)

    # Parents heaviest first: a parent always outweighs its daughters, so its
    # total is final before it is distributed.
    parents = sort!(
        collect(keys(decay_list.network.byparent));
        by = parent_id -> (-catalog[parent_id].mass, parent_id),
    )

    order = Int[]
    channel_range = UnitRange{Int}[]
    branching = Float64[]
    daughter_range = UnitRange{Int}[]
    daughters = Int[]
    for parent_id in parents
        first_channel = length(branching) + 1
        for channel in decay_list.network.byparent[parent_id]
            list = channel.daughters
            length(list) == 1 && only(list) == parent_id && continue
            all(d -> haskey(index_of, d), list) || continue
            first_daughter = length(daughters) + 1
            append!(daughters, (index_of[d] for d in list))
            push!(daughter_range, first_daughter:length(daughters))
            push!(branching, channel.br)
        end
        length(branching) >= first_channel || continue
        push!(order, index_of[parent_id])
        push!(channel_range, first_channel:length(branching))
    end

    mass_range = UnitRange{Int}[]
    mass_nodes = Float64[]
    mass_weights = Float64[]
    for record in records
        threshold = get(thresholds, record.id, 0.0)
        lower, upper = _support(record, threshold, k)
        if lower >= upper
            push!(mass_range, 1:0)
            continue
        end
        spectral = _spectralshape(
            shape, record.mass, record.width, threshold, lower, upper)
        nodes, weights = _spectralrule(spectral, mass_order)
        first_node = length(mass_nodes) + 1
        append!(mass_nodes, nodes)
        append!(mass_weights, weights)
        push!(mass_range, first_node:length(mass_nodes))
    end

    return ThermalPlan(
        ids,
        [r.mass for r in records], [r.width for r in records],
        [Float64(r.degeneracy) for r in records], [r.baryon != 0 for r in records],
        [Float64(r.baryon) for r in records], [Float64(r.strangeness) for r in records],
        [Float64(r.charge) for r in records], [Float64(r.charm) for r in records],
        [get(thresholds, particle_id, 0.0) for particle_id in ids],
        order, channel_range, branching, daughter_range, daughters,
        mass_range, mass_nodes, mass_weights,
        shape,
    )
end

Base.length(plan::ThermalPlan) = length(plan.ids)

"""
    thermalyields(plan::ThermalPlan, temperature; muB = 0.0, V, kwargs...)

The same result as the decay-list method, reusing a compiled plan. The density
loop runs in a few chunks per thread; each broad species costs one dot product
over its precomputed spectral rule, each sharp one a single fixed-rule
evaluation. The plan's `shape` was fixed when it was built -- compile another
plan to compare shapes.
"""
function thermalyields(
        plan::ThermalPlan, temperature;
        muB = 0.0, muS = 0.0, muQ = 0.0, muC = 0.0,
        V, quantum = true, width = true,
    )
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    volume = V / (0.1973269804)^3
    species_count = length(plan)
    primary = Vector{Float64}(undef, species_count)

    # Chunked, not per-species. A single density costs 0.4 us for a sharp
    # species and ~10 us for a broad one, so scheduling one task each buries the
    # work in overhead: 1518 tasks cost ~9 ms against ~7 ms of actual
    # computation. A few chunks per thread amortize that while leaving the
    # scheduler enough pieces to balance broad against sharp.
    chunk_count = min(species_count, 4 * Threads.nthreads())
    chunk_size = cld(species_count, chunk_count)
    Threads.@threads :greedy for chunk in 1:chunk_count
        first_index = (chunk - 1) * chunk_size + 1
        last_index = min(chunk * chunk_size, species_count)
        @inbounds for index in first_index:last_index
            mu = plan.baryon[index] * muB + plan.strangeness[index] * muS +
                 plan.charge[index] * muQ + plan.charm[index] * muC
            primary[index] =
                volume * _density(plan, index, temperature, mu; quantum, width)
        end
    end

    total = copy(primary)
    @inbounds for (position, parent) in pairs(plan.order)
        available = total[parent]
        available == 0 && continue
        for channel in plan.channel_range[position]
            contribution = plan.branching[channel] * available
            for slot in plan.daughter_range[channel]
                total[plan.daughters[slot]] += contribution
            end
        end
    end

    keyed(values) = Dict(zip(plan.ids, values))
    return (
        primary = keyed(primary),
        total = keyed(total),
        feeddown = keyed(total .- primary),
    )
end

"Density for one plan entry, mirroring `thermaldensity` on a flattened record."
function _density(plan::ThermalPlan, index, temperature, mu; quantum, width)
    mass, resonance_width = plan.mass[index], plan.width[index]
    degeneracy, is_fermion = plan.degeneracy[index], plan.isfermion[index]
    pole() = _densityatmass(mass, degeneracy, is_fermion, temperature, mu; quantum)
    (!width || resonance_width <= 0) && return pole()

    range = plan.mass_range[index]
    isempty(range) && return pole()
    averaged = 0.0
    @inbounds for slot in range
        averaged += plan.mass_weights[slot] * _densityatmass(
            plan.mass_nodes[slot], degeneracy, is_fermion, temperature, mu; quantum,
        )
    end
    return averaged
end

end # module ThermalModel
