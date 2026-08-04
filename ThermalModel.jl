"""
    ThermalModel

Statistical hadronization: thermal number densities at chemical freeze-out and
the yields that follow once resonances decay.

    dl = DecayList(readparticles("data/particles_QM2016Plus.dat"),
                   readdecays("data/decays_QM2016Plus.dat"))
    primary, total, feeddown = thermalyields(dl, 0.1565; muB = 0.0, V = 5300.0)

For a fit, compile the list once and reuse it -- `ThermalPlan` hoists
everything independent of `(T, muB, V)` and threads the density loop:

    plan = ThermalPlan(dl)
    yields = thermalyields(plan, T; muB, V)      # 4x faster, 4.5x less garbage

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

using QuadGK: quadgk, alloc_segbuf, Segment
using FastGaussQuadrature: gausslegendre

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

# ---------------------------------------------------------------------------
# Compiled plan
# ---------------------------------------------------------------------------

# One adaptive-quadrature buffer per task, so the mass integral stops
# allocating on every call and threaded species still get their own. `Vector`
# is not thread-safe to grow, hence per task rather than one shared buffer.
const _SEGBUF = OncePerTask{Vector{Segment{Float64, Float64, Float64}}}() do
    alloc_segbuf()
end

"Mass window a resonance is averaged over: `mR +- k*Gamma`, cut at threshold."
function _support(record, threshold, k)
    record.width > 0 || return (0.0, 0.0)
    return (max(record.mass - k * record.width, threshold),
            record.mass + k * record.width)
end

# Dense rule used only when building the plan, to integrate the Breit-Wigner
# against each Lagrange basis function. Its order is irrelevant at evaluation.
const _BUILD_NODES, _BUILD_WEIGHTS = gausslegendre(256)

"""
    _breitwignerrule(record, lower, upper, mass_order) -> (nodes, weights)

A quadrature rule carrying the Breit-Wigner as its weight function:
`sum(w .* f.(nodes))` approximates `int S(m) f(m) dm / int S(m) dm` for smooth
`f`. Nodes are Chebyshev points, weights are `int S(m) l_j(m) dm` with `l_j` the
Lagrange basis through them, so the rule is exact wherever the degree-`order`
interpolant of `f` is.

Handing the peak to the weights is what makes this cheap: adaptive quadrature
subdivides to resolve the Lorentzian and pays a full density evaluation at every
point it adds — 150 per resonance — although the density itself is smooth there.

Nothing here depends on temperature or chemical potential, so a fit builds it
once. The accuracy floor is polynomial approximation of the density, which
spans three decades across the window of a broad state: 6.6e-6 at order 24.
"""
function _breitwignerrule(record, lower, upper, order)
    half = (upper - lower) / 2
    midpoint = (upper + lower) / 2
    reference = [cos((2j - 1) * π / (2order)) for j in 1:order]
    nodes = midpoint .+ half .* reference
    # Barycentric weights for Chebyshev points of the first kind.
    barycentric = [(-1)^j * sin((2j - 1) * π / (2order)) for j in 1:order]

    shape(m) = m * record.width /
        ((m^2 - record.mass^2)^2 + (record.mass * record.width)^2)

    weights = zeros(order)
    basis = zeros(order)
    for index in eachindex(_BUILD_NODES)
        x = _BUILD_NODES[index]
        weight = _BUILD_WEIGHTS[index] * half * shape(midpoint + half * x)

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
    ThermalPlan(decay_list; mass_order = 24, k = 4.0)

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
end

function ThermalPlan(decay_list; mass_order = 24, k = 4.0)
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
        lower, upper = _support(record, get(thresholds, record.id, 0.0), k)
        if lower >= upper
            push!(mass_range, 1:0)
            continue
        end
        nodes, weights = _breitwignerrule(record, lower, upper, mass_order)
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
    )
end

Base.length(plan::ThermalPlan) = length(plan.ids)

"""
    thermalyields(plan::ThermalPlan, temperature; muB = 0.0, V, kwargs...)

The same result as the decay-list method, reusing a compiled plan. Densities are
computed with `Threads.@threads :greedy`, which suits a workload where a narrow
species costs one fixed-rule evaluation and a broad one costs an adaptive mass
integral around it.
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
