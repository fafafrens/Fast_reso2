"""
    DecayModel

Resolve particle catalogs and decay networks into the validated model shared by
the cascade and thermal paths. This module owns topology filtering, model
composition, spectral thresholds, feed-down ordering, and physics validation;
numerical consumers import the same `DecayList` type from here.
"""
module DecayModel

using Printf
using ..ParticleList: ParticleInfo, DecayChannel, ParticleCatalog, DecayNetwork,
    isospin, isbaryon, ismeson, isstrange
import ..ParticleList: particle, particleids

export DecayList,
    allparticles, stableparticles,
    ChannelFilter, channelfilter, nbody, parentmass, daughtermass, brabove,
    fromparent, todaughter, fromid, toid, baryonicparent, mesonicparent,
    parentstrangeness, strangeparent, charmparent, parentisospin,
    feeddown_order, validate

struct _ResolvedModelToken end
const _RESOLVED_MODEL = _ResolvedModelToken()

struct DecayList
    catalog::ParticleCatalog
    network::DecayNetwork
    stable_ids::Vector{Int}
    thresholds::Dict{Int, Float64}
    path::String
    sha::String

    function DecayList(
            catalog::ParticleCatalog,
            network::DecayNetwork,
            stable_ids::Vector{Int},
            thresholds::Dict{Int, Float64},
            path::String,
            sha::String,
            ::_ResolvedModelToken,
        )
        return new(catalog, network, stable_ids, thresholds, path, sha)
    end
end

_resolvedmodel(catalog, network, stable_ids, thresholds, path, sha) = DecayList(
    catalog,
    network,
    stable_ids,
    thresholds,
    String(path),
    String(sha),
    _RESOLVED_MODEL,
)

# Compatibility projections for the original public fields. Literal
# `decay_list.parts` and `.byparent` remain inferred and allocation-free.
function Base.getproperty(decay_list::DecayList, name::Symbol)
    name === :parts && return getfield(getfield(decay_list, :catalog), :byid)
    name === :byparent &&
        return getfield(getfield(decay_list, :network), :byparent)
    return getfield(decay_list, name)
end

Base.propertynames(::DecayList, ::Bool = false) =
    (:catalog, :network, :stable_ids, :thresholds, :path, :sha, :parts, :byparent)

particle(decay_list::DecayList, id_or_name) =
    particle(decay_list.catalog, id_or_name)
particleids(decay_list::DecayList, label::AbstractString) =
    particleids(decay_list.catalog, label)

function _resolvedstableids(
        catalog::ParticleCatalog,
        network::DecayNetwork,
    )
    particles = catalog.byid

    # Store `(owner, missing_id)` pairs so resolution can report all missing
    # references at once. A missing declared parent is represented as `(id, id)`.
    unresolved = Tuple{Int, Int}[]

    # Decaying declarations survive topology filtering even when a parent's
    # visible channel block becomes empty, so validate them independently of
    # `network.byparent`.
    for parent_id in network.decaying_ids
        haskey(particles, parent_id) || push!(unresolved, (parent_id, parent_id))
    end

    for (parent_id, channels) in network.byparent
        haskey(particles, parent_id) || begin
            push!(unresolved, (parent_id, parent_id))
            continue
        end

        for channel in channels
            # The dictionary key is the authoritative block owner. Keeping it
            # consistent with every row avoids ambiguous merge/filter results.
            channel.parent == parent_id || throw(ArgumentError(
                "channel parent $(channel.parent) disagrees with network " *
                    "parent $parent_id",
            ))
            isfinite(channel.br) || throw(ArgumentError(
                "channel for parent $parent_id has nonfinite branching ratio " *
                    "$(channel.br)",
            ))
            multiplicity = length(channel.daughters)
            if multiplicity == 1
                # Canonical and legacy inputs use exactly `parent -> parent`
                # with BR 1 as a terminal marker; it is not a physical decay.
                only(channel.daughters) == parent_id || throw(ArgumentError(
                    "one-body channel $parent_id -> " *
                        "$(only(channel.daughters)) is not a terminal self-channel",
                ))
                length(channels) == 1 || throw(ArgumentError(
                    "terminal self-channel for particle $parent_id must be " *
                        "the only row in its parent block",
                ))
                channel.br == 1.0 || throw(ArgumentError(
                    "terminal self-channel for particle $parent_id must have BR 1",
                ))
            elseif !(2 <= multiplicity <= 4)
                throw(ArgumentError(
                    "channel for parent $parent_id has unsupported " *
                        "multiplicity $multiplicity; runnable decay lists " *
                        "support 2-, 3-, and 4-body channels",
                ))
            end
            for daughter_id in channel.daughters
                haskey(particles, daughter_id) ||
                    push!(unresolved, (parent_id, daughter_id))
            end
        end
    end

    if !isempty(unresolved)
        # Deduplicate and sort for a deterministic, bounded diagnostic instead
        # of failing on whichever missing reference dictionary order finds first.
        sort!(unique!(unresolved))
        preview = join(
            ("$parent_id -> $daughter_id" for
                (parent_id, daughter_id) in first(unresolved, 8)),
            ", ",
        )
        suffix = length(unresolved) > 8 ?
            " (and $(length(unresolved) - 8) more)" : ""
        throw(ArgumentError(
            "decay network has unresolved particle references: " * preview * suffix,
        ))
    end

    # Stability is catalog membership minus the declared decaying set, not
    # `keys(byparent)`: filtering may remove every active channel for a parent,
    # but that must not silently turn the parent into a terminal output.
    decaying = Set(network.decaying_ids)
    return sort!([
        particle_id for particle_id in keys(particles)
            if particle_id ∉ decaying
    ])
end

function _modelprovenance(catalog::ParticleCatalog, network::DecayNetwork)
    same_source =
        catalog.path == network.path && catalog.sha == network.sha
    path = same_source ? catalog.path : join(
        filter(value -> !isempty(value), (catalog.path, network.path)),
        " + ",
    )
    sha = same_source ? catalog.sha : join(
        filter(value -> !isempty(value), (catalog.sha, network.sha)),
        "+",
    )
    return path, sha
end

function _resolve(
        catalog::ParticleCatalog,
        network::DecayNetwork;
        thresholds = nothing,
        provenance = nothing,
    )
    stable_ids = _resolvedstableids(catalog, network)
    resolved_thresholds = thresholds === nothing ?
        _resolvedthresholds(catalog, network) :
        Dict{Int, Float64}(thresholds)
    path, sha = provenance === nothing ?
        _modelprovenance(catalog, network) : provenance
    return _resolvedmodel(
        catalog,
        network,
        stable_ids,
        resolved_thresholds,
        path,
        sha,
    )
end

"""
    _resolvedthresholds(catalog, network) -> Dict{Int, Float64}

Lowest physical mass per decaying species: the smallest summed daughter mass
over its nonzero, nonterminal channels. This is resolved model metadata rather
than filtered-topology metadata.
"""
function _resolvedthresholds(
        catalog::ParticleCatalog,
        network::DecayNetwork,
    )
    thresholds = Dict{Int, Float64}()
    for (parent_id, channels) in network.byparent
        best = Inf
        for channel in channels
            iszero(channel.br) && continue
            daughters = channel.daughters
            length(daughters) == 1 && only(daughters) == parent_id && continue
            best = min(
                best,
                sum(catalog.byid[daughter_id].mass for daughter_id in daughters),
            )
        end
        isfinite(best) && (thresholds[parent_id] = best)
    end
    return thresholds
end

"""
    DecayList(catalog::ParticleCatalog, network::DecayNetwork)

Resolve a catalog and network into the strict model accepted by table, plan,
filtering, validation, cascade, and thermal APIs.
"""
function DecayList(catalog::ParticleCatalog, network::DecayNetwork)
    return _resolve(catalog, network)
end

# Compatibility constructors used by programmatic decay lists.
function DecayList(
        particles::Dict{Int, ParticleInfo},
        channels_by_parent::Dict{Int, Vector{DecayChannel}},
        path::AbstractString,
        sha::AbstractString,
    )
    catalog = ParticleCatalog(
        particles;
        path = String(path),
        sha = String(sha),
    )
    network = DecayNetwork(
        channels_by_parent;
        path = String(path),
        sha = String(sha),
    )
    return DecayList(catalog, network)
end

function DecayList(
        particles::Dict{Int, ParticleInfo},
        channels_by_parent::Dict{Int, Vector{DecayChannel}},
        stable_ids,
        path::AbstractString,
        sha::AbstractString,
    )
    catalog = ParticleCatalog(
        particles;
        path = String(path),
        sha = String(sha),
    )
    requested_stable_ids = sort!(unique!(collect(Int, stable_ids)))
    network = DecayNetwork(
        channels_by_parent;
        decaying_ids = setdiff(keys(particles), requested_stable_ids),
        path = String(path),
        sha = String(sha),
    )
    decay_list = DecayList(catalog, network)
    decay_list.stable_ids == requested_stable_ids || throw(ArgumentError(
        "declared stable particle IDs disagree with resolved topology: " *
            "requested $(requested_stable_ids), resolved $(decay_list.stable_ids)",
    ))
    return decay_list
end

"""
    merge(first::DecayList, remaining::DecayList...) -> DecayList

Copy or combine resolved decay models. Catalog and network conflicts are
rejected by their respective merge methods. Thresholds from filtered inputs
remain resolved-model metadata and are preserved in the result.
"""
function Base.merge(
        first::DecayList,
        remaining::DecayList...,
    )
    decay_lists = (first, remaining...)
    catalog = merge((decay_list.catalog for decay_list in decay_lists)...)
    network = merge((decay_list.network for decay_list in decay_lists)...)
    thresholds = _resolvedthresholds(catalog, network)
    for decay_list in decay_lists
        for (parent_id, threshold) in decay_list.thresholds
            thresholds[parent_id] = min(
                get(thresholds, parent_id, Inf),
                threshold,
            )
        end
    end
    source_paths = [
        decay_list.path for decay_list in decay_lists
            if !isempty(decay_list.path)
    ]
    source_hashes = [
        decay_list.sha for decay_list in decay_lists
            if !isempty(decay_list.sha)
    ]
    return _resolve(
        catalog,
        network,
        thresholds = thresholds,
        provenance = (
            join(source_paths, " + "),
            join(source_hashes, "+"),
        ),
    )
end

"Return every catalog particle in ascending particle-ID order."
allparticles(catalog::ParticleCatalog) = [
    catalog.byid[particle_id]
        for particle_id in sort!(collect(keys(catalog.byid)))
]
allparticles(decay_list::DecayList) = allparticles(decay_list.catalog)

"Return terminal particles in ascending particle-ID order."
stableparticles(decay_list::DecayList) =
    [decay_list.catalog.byid[particle_id] for particle_id in decay_list.stable_ids]

# ---------------------------------------------------------------------------
# Channel filtering
# ---------------------------------------------------------------------------

struct ChannelFilter{Fun}
    keep::Fun
end

(selection::ChannelFilter)(parent, channel, daughters) =
    selection.keep(parent, channel, daughters)::Bool

function Base.:&(left::ChannelFilter, right::ChannelFilter)
    keep(parent, channel, daughters) =
        left(parent, channel, daughters) &&
        right(parent, channel, daughters)
    return ChannelFilter(keep)
end

function Base.:|(left::ChannelFilter, right::ChannelFilter)
    keep(parent, channel, daughters) =
        left(parent, channel, daughters) ||
        right(parent, channel, daughters)
    return ChannelFilter(keep)
end

function Base.:!(selection::ChannelFilter)
    keep(parent, channel, daughters) =
        !selection(parent, channel, daughters)
    return ChannelFilter(keep)
end

"""
    filter(selection::ChannelFilter, decay_list::DecayList) -> DecayList

Restrict the active topology while preserving the resolved stable set,
threshold metadata, particle catalog, and source provenance.
"""
function Base.filter(selection::ChannelFilter, decay_list::DecayList)
    channels_by_parent = Dict{Int, Vector{DecayChannel}}()

    for (parent_id, channels) in decay_list.network.byparent
        parent = decay_list.catalog.byid[parent_id]
        selected = DecayChannel[]

        for channel in channels
            daughters = ParticleInfo[
                decay_list.catalog.byid[daughter_id]
                    for daughter_id in channel.daughters
            ]
            selection(parent, channel, daughters) && push!(selected, channel)
        end

        isempty(selected) || (channels_by_parent[parent_id] = selected)
    end

    network = DecayNetwork(
        channels_by_parent;
        decaying_ids = decay_list.network.decaying_ids,
        path = decay_list.network.path,
        sha = decay_list.network.sha,
    )
    return _resolve(
        decay_list.catalog,
        network,
        thresholds = decay_list.thresholds,
        provenance = (decay_list.path, decay_list.sha),
    )
end

function Base.filter(::ChannelFilter, ::DecayNetwork)
    throw(ArgumentError(
        "a DecayNetwork has unresolved particle IDs; construct a DecayList " *
            "with its ParticleCatalog before filtering",
    ))
end

nbody(multiplicities::Integer...) = ChannelFilter(
    (_parent, channel, _daughters) ->
        length(channel.daughters) in multiplicities,
)
brabove(minimum_branching_ratio) = ChannelFilter(
    (_parent, channel, _daughters) ->
        channel.br >= minimum_branching_ratio,
)
parentmass(minimum_mass, maximum_mass) = ChannelFilter(
    (parent, _channel, _daughters) ->
        minimum_mass <= parent.mass <= maximum_mass,
)
fromparent(names::AbstractString...) =
    ChannelFilter((parent, _channel, _daughters) -> parent.name in names)
fromid(ids::Integer...) =
    ChannelFilter((parent, _channel, _daughters) -> parent.id in ids)
baryonicparent() =
    ChannelFilter((parent, _channel, _daughters) -> isbaryon(parent))
mesonicparent() =
    ChannelFilter((parent, _channel, _daughters) -> ismeson(parent))
parentstrangeness(strangenesses::Integer...) = ChannelFilter(
    (parent, _channel, _daughters) ->
        parent.strangeness in strangenesses,
)
strangeparent() =
    ChannelFilter((parent, _channel, _daughters) -> isstrange(parent))
charmparent() =
    ChannelFilter((parent, _channel, _daughters) -> parent.charm != 0)

function parentisospin(isospins::Real...)
    twice_isospins = map(isospin -> round(Int, 2 * isospin), isospins)
    return ChannelFilter(
        (parent, _channel, _daughters) ->
            parent.twiceisospin in twice_isospins,
    )
end

daughtermass(minimum_mass, maximum_mass) = ChannelFilter(
    (_parent, _channel, daughters) ->
        all(daughter -> minimum_mass <= daughter.mass <= maximum_mass, daughters),
)
todaughter(names::AbstractString...) = ChannelFilter(
    (_parent, _channel, daughters) ->
        any(daughter -> daughter.name in names, daughters),
)
toid(ids::Integer...) = ChannelFilter(
    (_parent, _channel, daughters) ->
        any(daughter -> daughter.id in ids, daughters),
)
channelfilter(predicate) = ChannelFilter(predicate)

# ---------------------------------------------------------------------------
# Shared feeder-before-fed ordering
# ---------------------------------------------------------------------------

function _feeddown_dependencies(decay_list::DecayList)
    # Ordering needs only the existence of an edge, not its multiplicity or
    # branching weight. A Set therefore collapses repeated daughters and the
    # same parent-daughter relation appearing in multiple channels.
    dependencies = Set{Tuple{Int, Int}}()

    # Only parents with an active channel block participate in the feed-down
    # sweep. Stable daughters need no dependency because they are never
    # processed as a later parent step.
    decaying = decay_list.network.byparent
    for (parent_id, channels) in decaying
        for channel in channels
            # Zero-weight rows transfer nothing and must not constrain order.
            iszero(channel.br) && continue
            daughters = channel.daughters

            # A one-body `parent -> parent` row is a terminal marker, not a
            # physical self-loop in the decay graph.
            length(daughters) == 1 && only(daughters) == parent_id && continue
            for daughter_id in daughters
                # If the daughter also decays, process the feeder parent first
                # so the daughter receives its complete yield before its turn.
                haskey(decaying, daughter_id) &&
                    push!(dependencies, (parent_id, daughter_id))
            end
        end
    end
    return dependencies
end

"""
    feeddown_order(decay_list; dependencies = nothing, cycles = :error)

Return a deterministic feeder-before-fed parent order. By default dependencies
are all nonzero decay edges whose daughter also decays. Callers with a narrower
active topology may supply `(parent_id, daughter_id)` pairs explicitly.

Strongly connected components are detected before ordering. `cycles = :error`
rejects them because one sweep cannot be exact; `cycles = :warn` preserves the
cascade's historical mass-ordered approximation within each component.
"""
function feeddown_order(
        decay_list::DecayList;
        dependencies = nothing,
        cycles::Symbol = :error,
    )
    cycles === :error || cycles === :warn || throw(ArgumentError(
        "cycles must be :error or :warn, got $(repr(cycles))",
    ))

    # Normalize the selected edges into a deterministic adjacency list over
    # active decaying parents. Including parents with no outgoing dependency
    # keeps disconnected and terminal-in-the-subgraph species in the result.
    selected_dependencies =
        dependencies === nothing ? _feeddown_dependencies(decay_list) : dependencies
    decaying_parents = sort!(collect(keys(decay_list.network.byparent)))
    downstream = Dict(parent_id => Int[] for parent_id in decaying_parents)

    # Explicit caller-supplied dependencies must describe this active graph;
    # accepting an absent or stable endpoint would make the final order partial.
    for (parent_id, daughter_id) in selected_dependencies
        haskey(downstream, parent_id) || throw(ArgumentError(
            "feed-down dependency has unknown parent $parent_id",
        ))
        haskey(downstream, daughter_id) || throw(ArgumentError(
            "feed-down dependency $parent_id -> $daughter_id targets a " *
                "non-decaying species",
        ))
        push!(downstream[parent_id], daughter_id)
    end

    # Repeated edges do not change ordering. Sorting also makes the depth-first
    # traversal and resulting component numbering reproducible.
    foreach(daughters -> sort!(unique!(daughters)), values(downstream))

    # Tarjan strongly connected component state. `indices` is discovery order;
    # `lowlinks` records the earliest active vertex reachable from each parent.
    # Ref provides a mutable counter to the nested visitor without a boxed local.
    next_index = Ref(0)
    indices = Dict{Int, Int}()
    lowlinks = Dict{Int, Int}()
    stack = Int[]
    on_stack = Set{Int}()
    components = Vector{Vector{Int}}()

    function visit!(parent_id)
        # Discover the parent and keep it active until its complete strongly
        # connected component has been identified.
        next_index[] += 1
        indices[parent_id] = next_index[]
        lowlinks[parent_id] = next_index[]
        push!(stack, parent_id)
        push!(on_stack, parent_id)

        for daughter_id in downstream[parent_id]
            if !haskey(indices, daughter_id)
                # Tree edge: visit the daughter, then propagate the earliest
                # vertex reachable through its subtree.
                visit!(daughter_id)
                lowlinks[parent_id] = min(
                    lowlinks[parent_id], lowlinks[daughter_id],
                )
            elseif daughter_id in on_stack
                # Back edge into the active DFS stack closes or participates in
                # a cycle, so the daughter's discovery index constrains lowlink.
                lowlinks[parent_id] = min(
                    lowlinks[parent_id], indices[daughter_id],
                )
            end
        end

        # Equal lowlink and discovery indices mark an SCC root. Pop through the
        # root to recover exactly that component from the active stack.
        if lowlinks[parent_id] == indices[parent_id]
            component = Int[]
            while true
                member = pop!(stack)
                delete!(on_stack, member)
                push!(component, member)
                member == parent_id && break
            end

            # Acyclic components contain one species. For a permitted cyclic
            # component, mass-descending order preserves the historical
            # deterministic one-pass approximation; particle ID breaks ties.
            sort!(
                component;
                by = member -> (-decay_list.catalog.byid[member].mass, member),
            )
            push!(components, component)
        end
        return nothing
    end

    # Tarjan must start from every unvisited parent because the feed-down graph
    # can contain multiple disconnected components.
    for parent_id in decaying_parents
        haskey(indices, parent_id) || visit!(parent_id)
    end

    # Collapse every SCC to one vertex. `component_of` provides the species-to-
    # component lookup needed to translate the original edges.
    component_of = Dict{Int, Int}()
    for (component_id, component) in pairs(components)
        for parent_id in component
            component_of[parent_id] = component_id
        end
    end

    # Build the condensation graph. Edges within one SCC disappear; edges
    # between SCCs form a directed acyclic graph by construction.
    component_downstream = [Int[] for _ in components]
    for (parent_id, daughter_ids) in downstream
        parent_component = component_of[parent_id]
        for daughter_id in daughter_ids
            daughter_component = component_of[daughter_id]
            parent_component == daughter_component ||
                push!(component_downstream[parent_component], daughter_component)
        end
    end

    # Kahn's algorithm below needs each component's number of upstream feeders.
    # Deduplicate first because several particle edges can connect the same SCCs.
    incoming_count = zeros(Int, length(components))
    for daughter_components in component_downstream
        sort!(unique!(daughter_components))
        for daughter_component in daughter_components
            incoming_count[daughter_component] += 1
        end
    end

    # Multi-species SCCs are cyclic; a singleton is cyclic only when it has an
    # explicit self-edge. One feed-down sweep cannot solve either exactly.
    cyclic_components = [
        component for component in components
            if length(component) > 1 ||
                first(component) in downstream[first(component)]
    ]
    if !isempty(cyclic_components)
        cycle_count = length(cyclic_components)
        cycle_species = sum(length, cyclic_components)
        message = "feed-down graph has $cycle_count cycle components " *
            "involving $cycle_species species"
        if cycles === :error
            throw(ArgumentError(
                message * "; one-pass feed-down is not exact for cyclic topology",
            ))
        end
        @warn message * "; members ordered by mass within each component, " *
            "cyclic feed-down approximate"
    end

    # Topologically sort the acyclic component graph. Zero-incoming components
    # are complete feeders and therefore safe to process next.
    order = Int[]
    ready_components = findall(iszero, incoming_count)
    while !isempty(ready_components)
        # Several components can be ready simultaneously. Select the one with
        # the heaviest leading member, then the lowest particle ID, to keep the
        # historical mass preference without overriding graph dependencies.
        # `argmax(f, itr)` returns the winning component ID, not its list index.
        component_id = argmax(
            candidate -> begin
                parent_id = first(components[candidate])
                (decay_list.catalog.byid[parent_id].mass, -parent_id)
            end,
            ready_components,
        )
        deleteat!(
            ready_components,
            findfirst(==(component_id), ready_components),
        )

        # In `cycles = :warn` mode, members of a cyclic SCC are appended in the
        # deterministic mass order established when the component was formed.
        append!(order, components[component_id])

        # Removing this feeder edge may make downstream components ready.
        for daughter_component in component_downstream[component_id]
            incoming_count[daughter_component] -= 1
            incoming_count[daughter_component] == 0 &&
                push!(ready_components, daughter_component)
        end
    end

    return order
end

# ---------------------------------------------------------------------------
# Physics validation
# ---------------------------------------------------------------------------

function validate(decay_list::DecayList; brtol = 0.02, verbose = true)
    channel_count = 0
    one_body_count = 0
    unresolved_count = 0
    invalid_branching_ratio_count = 0
    strangeness_change_count = 0
    charm_change_count = 0
    bottomness_change_count = 0
    subthreshold_count = 0
    duplicate_channel_count = 0

    charge_violations = Tuple{Int, Vector{Int}}[]
    baryon_violations = Tuple{Int, Vector{Int}}[]
    multiplicity_counts = Dict{Int, Int}()
    branching_ratio_sums = Dict(
        parent_id => 0.0 for parent_id in decay_list.network.decaying_ids
    )

    for (parent_id, channels) in decay_list.network.byparent
        parent = decay_list.catalog.byid[parent_id]
        seen_channels = Set{Tuple{Float64, Tuple}}()

        for channel in channels
            daughter_ids = channel.daughters
            channel_signature = (channel.br, Tuple(sort(daughter_ids)))
            if channel_signature in seen_channels
                duplicate_channel_count += 1
            else
                push!(seen_channels, channel_signature)
            end
            multiplicity = length(daughter_ids)
            multiplicity_counts[multiplicity] =
                get(multiplicity_counts, multiplicity, 0) + 1

            if multiplicity == 1
                one_body_count += 1
                continue
            end

            channel_count += 1
            branching_ratio_sums[parent_id] += channel.br

            if !(0.0 <= channel.br <= 1 + 1.0e-6)
                invalid_branching_ratio_count += 1
            end
            if !all(
                    daughter_id -> haskey(decay_list.catalog.byid, daughter_id),
                    daughter_ids,
                )
                unresolved_count += 1
                continue
            end

            daughters = (
                decay_list.catalog.byid[daughter_id] for daughter_id in daughter_ids
            )
            if sum(daughter -> daughter.charge, daughters) != parent.charge
                push!(charge_violations, (parent_id, daughter_ids))
            end
            if sum(daughter -> daughter.baryon, daughters) != parent.baryon
                push!(baryon_violations, (parent_id, daughter_ids))
            end
            if sum(daughter -> daughter.strangeness, daughters) != parent.strangeness
                strangeness_change_count += 1
            end
            if sum(daughter -> daughter.charm, daughters) != parent.charm
                charm_change_count += 1
            end
            if sum(daughter -> daughter.bottomness, daughters) != parent.bottomness
                bottomness_change_count += 1
            end
            if parent.mass + 1.0e-6 < sum(daughter -> daughter.mass, daughters)
                subthreshold_count += 1
            end
        end
    end

    branching_ratio_deviations = [
        abs(branching_ratio_sum - 1)
            for branching_ratio_sum in values(branching_ratio_sums)
    ]
    maximum_branching_deviation = isempty(branching_ratio_deviations) ?
        0.0 : maximum(branching_ratio_deviations)
    stats = (
        particles = length(decay_list.catalog.byid),
        channels = channel_count,
        selfstable = length(decay_list.stable_ids),
        selfchannels = one_body_count,
        multiplicity = sort!(collect(multiplicity_counts)),
        unresolved = unresolved_count,
        br_out_of_range = invalid_branching_ratio_count,
        charge_violations = length(charge_violations),
        baryon_violations = length(baryon_violations),
        strangeness_changing = strangeness_change_count,
        charm_changing = charm_change_count,
        bottomness_changing = bottomness_change_count,
        duplicate_channels = duplicate_channel_count,
        subthreshold = subthreshold_count,
        br_norm_outliers = count(>(brtol), branching_ratio_deviations),
        br_maxdev = maximum_branching_deviation,
    )

    if verbose
        _print_validation_report(
            decay_list,
            stats,
            charge_violations,
            baryon_violations,
            branching_ratio_sums,
            brtol,
        )
    end
    return stats
end

function _print_validation_report(
        decay_list,
        stats,
        charge_violations,
        baryon_violations,
        branching_ratio_sums,
        branching_ratio_tolerance,
    )
    particle_name(particle_id) = haskey(decay_list.catalog.byid, particle_id) ?
        decay_list.catalog.byid[particle_id].name :
        "?$particle_id"
    count_status(count) = count == 0 ? "OK" : "*** $count ***"

    println("── decay-list validation ", repeat("─", 44))
    @printf(
        "  %d particles, %d decay channels (%d stable), multiplicities %s\n",
        stats.particles,
        stats.channels,
        stats.selfstable,
        string(stats.multiplicity),
    )
    println("  conservation (must be zero):")
    @printf("    charge violations ........ %s\n", count_status(stats.charge_violations))
    @printf("    baryon-number violations . %s\n", count_status(stats.baryon_violations))
    @printf("    unresolved daughters ..... %s\n", count_status(stats.unresolved))
    @printf("    BR out of [0,1] .......... %s\n", count_status(stats.br_out_of_range))
    @printf("    exact duplicate channels . %s\n", count_status(stats.duplicate_channels))
    @printf(
        "  branching ratios: max |ΣBR-1| = %.2e,  %s parents beyond %.0f%%\n",
        stats.br_maxdev,
        count_status(stats.br_norm_outliers),
        100 * branching_ratio_tolerance,
    )
    @printf(
        "  informational: %d strangeness-, %d charm-, %d bottomness-changing (weak),  %d closed-at-pole\n",
        stats.strangeness_changing,
        stats.charm_changing,
        stats.bottomness_changing,
        stats.subthreshold,
    )

    for (label, violations) in (
            ("charge", charge_violations),
            ("baryon", baryon_violations),
        )
        isempty(violations) && continue
        println("  first $label violations:")
        for (parent_id, daughter_ids) in first(violations, 3)
            daughter_names =
                (particle_name(daughter_id) for daughter_id in daughter_ids)
            @printf(
                "    %s -> %s\n",
                particle_name(parent_id),
                join(daughter_names, " "),
            )
        end
    end

    worst_branching_sums = sort!(
        collect(branching_ratio_sums);
        by = branching_entry -> abs(last(branching_entry) - 1),
        rev = true,
    )
    has_outlier =
        !isempty(worst_branching_sums) &&
        abs(worst_branching_sums[1][2] - 1) > branching_ratio_tolerance
    if has_outlier
        println("  parents with worst ΣBR:")
        for (parent_id, branching_ratio_sum) in first(worst_branching_sums, 3)
            @printf(
                "    %-12s ΣBR = %.3f\n",
                particle_name(parent_id),
                branching_ratio_sum,
            )
        end
    end
    return println(repeat("─", 70))
end

end # module DecayModel
