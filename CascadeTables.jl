"""
    CascadeTables

Structured, persistent decay-matrix tables for the full resonance cascade.

    dl = DecayList(
        readparticles("particles.dat"),
        readdecays("decays.dat"),
    )
    res = runcascade(                                 # build, compile, and run
        dl; primary = myprimaries, cache = "Wtables.jls",
    )

Design (see matrix_notes.pdf for details):
  * decay matrices are serialized; source path and hash remain in table metadata;
  * matrices are keyed by `ChannelKey` (the masses plus the parent width,
    observed daughter fixed, unobserved partners sorted), allowing physically
    equivalent channels to share an entry;
  * missing keys are built on demand and `prune!` removes unreferenced keys;
  * matrices use packed Float32 row bands while spectra accumulate in Float64;
  * table construction parallelizes over independent channel keys;
  * `CascadePlan` resolves keys and topology to flat, dense arrays and can run
    independent Fourier modes on separate Julia tasks.
"""
module CascadeTables

include("DecaySpectra.jl")
include("ParticleList.jl")
include("RowBandMatrix.jl")
using .DecaySpectra
using .ParticleList
import .ParticleList: particle, particleids
using Serialization, Dates, Printf
using LinearAlgebra: axpy!
import LinearAlgebra: mul!
include("CascadeLinearAlgebra.jl")

export ParticleCatalog, DecayNetwork, DecayList,
    readparticles, readdecays,
    particle, particleids,
    ChannelKey, CascadeTable, CascadePlan, runcascade,
    prune!, validate,
    savetables, loadtables, ParticleInfo, DecayChannel,
    allparticles, stableparticles,
    isospin, spin, isbaryon, ismeson, isantiparticle, isstrange, ischarm,
    isbottom, quarkdigits, hasflavour,
    ChannelFilter, channelfilter, nbody, parentmass, daughtermass, brabove,
    fromparent, todaughter, fromid, toid, baryonicparent, mesonicparent,
    parentstrangeness, strangeparent, charmparent, parentisospin

# Increment when the serialized layout or generated matrix values change.
const FORMAT = 11
const CACHE_MAGIC = "FastReso.CascadeTables\n"

# ---------------------------------------------------------------------------
# Resolved model: a particle catalog plus an independently parsed decay network
# ---------------------------------------------------------------------------

struct _ResolvedModelToken end
const _RESOLVED_MODEL = _ResolvedModelToken()

struct DecayList
    catalog::ParticleCatalog
    network::DecayNetwork
    stable_ids::Vector{Int}
    path::String
    sha::String

    function DecayList(
            catalog::ParticleCatalog,
            network::DecayNetwork,
            stable_ids::Vector{Int},
            path::String,
            sha::String,
            ::_ResolvedModelToken,
        )
        return new(catalog, network, stable_ids, path, sha)
    end
end

_resolvedmodel(catalog, network, stable_ids, path, sha) = DecayList(
    catalog,
    network,
    stable_ids,
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
    (:catalog, :network, :stable_ids, :path, :sha, :parts, :byparent)

particle(decay_list::DecayList, id_or_name) =
    particle(decay_list.catalog, id_or_name)
particleids(decay_list::DecayList, label::AbstractString) =
    particleids(decay_list.catalog, label)

function _resolvedstableids(
        catalog::ParticleCatalog,
        network::DecayNetwork,
    )
    particles = catalog.byid
    unresolved = Tuple{Int, Int}[]

    for parent_id in network.decaying_ids
        haskey(particles, parent_id) || push!(unresolved, (parent_id, parent_id))
    end
    for parent_id in network.terminal_ids
        haskey(particles, parent_id) || push!(unresolved, (parent_id, parent_id))
    end

    for (parent_id, channels) in network.byparent
        haskey(particles, parent_id) || begin
            push!(unresolved, (parent_id, parent_id))
            continue
        end

        for channel in channels
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
            elseif !(2 ≤ multiplicity ≤ 4)
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

"""
    DecayList(catalog::ParticleCatalog, network::DecayNetwork)

Resolve a catalog and network into the strict model accepted by table, plan,
filtering, validation, and cascade APIs. Every parent and daughter ID must be
present in `catalog`. Merge all cross-referencing catalogs and networks before
calling this constructor.
"""
function DecayList(catalog::ParticleCatalog, network::DecayNetwork)
    stable_ids = _resolvedstableids(catalog, network)
    path, sha = _modelprovenance(catalog, network)
    return _resolvedmodel(catalog, network, stable_ids, path, sha)
end

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
    network = DecayNetwork(
        channels_by_parent;
        decaying_ids = setdiff(keys(particles), stable_ids),
        terminal_ids = stable_ids,
        path = String(path),
        sha = String(sha),
    )
    return DecayList(catalog, network)
end

"""
    merge(first::DecayList, remaining::DecayList...) -> DecayList

Copy or combine resolved decay lists. A particle may occur in several inputs
when its physical fields agree; alternate names become catalog aliases. A
parent channel block may also repeat when
it contains the same multiset of channels; channel-row and daughter ordering
do not matter, but branching ratios must agree exactly. Conflicting particle
or channel definitions are rejected instead of overwritten or concatenated.
For cross-list daughters, merge all catalogs and networks before constructing
the resolved `DecayList` so every source is present at the resolution boundary.

The result owns new outer dictionaries and a new terminal-ID vector, preserves
the terminal-particle classification of every input, and records source paths
and hashes in input order. A repeated particle is terminal only when every
input containing it marks it terminal. Particle records and channel vectors
are shared with the first defining input and treated as read-only.
"""
function Base.merge(
        first::DecayList,
        remaining::DecayList...,
    )
    decay_lists = (first, remaining...)
    catalog = merge((decay_list.catalog for decay_list in decay_lists)...)
    network = merge((decay_list.network for decay_list in decay_lists)...)
    resolved = DecayList(catalog, network)
    source_paths = [
        decay_list.path for decay_list in decay_lists
            if !isempty(decay_list.path)
    ]
    source_hashes = [
        decay_list.sha for decay_list in decay_lists
            if !isempty(decay_list.sha)
    ]
    return _resolvedmodel(
        catalog,
        network,
        resolved.stable_ids,
        join(source_paths, " + "),
        join(source_hashes, "+"),
    )
end

"""
    allparticles(decay_list::DecayList) -> Vector{ParticleInfo}

Return every particle defined by the decay list in ascending particle-ID order.
"""
allparticles(catalog::ParticleCatalog) = [
    catalog.byid[particle_id]
        for particle_id in sort!(collect(keys(catalog.byid)))
]
allparticles(decay_list::DecayList) = allparticles(decay_list.catalog)

"""
    stableparticles(decay_list::DecayList) -> Vector{ParticleInfo}

Return particles marked terminal by the decay list in ascending particle-ID
order. A sole one-body self-channel marks an explicit terminal; particles not
marked decaying are also terminal, including particles without a channel block.
"""
stableparticles(decay_list::DecayList) =
    [decay_list.catalog.byid[particle_id] for particle_id in decay_list.stable_ids]

# ---------------------------------------------------------------------------
# Channel filtering
# ---------------------------------------------------------------------------
# Filters operate on `(parent, channel, daughters)` and combine with `&`, `|`, `!`.

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
    filter(f::ChannelFilter, dl::DecayList) -> DecayList

Restrict `dl` to the channels satisfying `f`, keeping every species so all
daughters still resolve but pruning `byparent`. The original terminal-particle
set is preserved. The result flows unchanged through `CascadeTable` and
`runcascade`; since the masses are unchanged, its decay-matrix keys are a
subset of the full list's, so filtering an *apply* reuses a full cached table
with no rebuild.
"""
function Base.filter(selection::ChannelFilter, dl::DecayList)
    channels_by_parent = Dict{Int, Vector{DecayChannel}}()

    for (parent_id, channels) in dl.network.byparent
        parent = dl.catalog.byid[parent_id]
        selected = DecayChannel[]

        for channel in channels
            daughter_ids = channel.daughters
            daughters = ParticleInfo[
                dl.catalog.byid[daughter_id] for daughter_id in daughter_ids
            ]
            if selection(parent, channel, daughters)
                push!(selected, channel)
            end
        end

        if !isempty(selected)
            channels_by_parent[parent_id] = selected
        end
    end

    network = DecayNetwork(
        channels_by_parent;
        decaying_ids = dl.network.decaying_ids,
        terminal_ids = dl.network.terminal_ids,
        path = dl.network.path,
        sha = dl.network.sha,
    )
    resolved = DecayList(dl.catalog, network)
    return _resolvedmodel(
        dl.catalog,
        network,
        resolved.stable_ids,
        dl.path,
        dl.sha,
    )
end

function Base.filter(::ChannelFilter, ::DecayNetwork)
    throw(ArgumentError(
        "a DecayNetwork has unresolved particle IDs; construct a DecayList " *
            "with its ParticleCatalog before filtering",
    ))
end

# --- channel properties ----------------------------------------------------
"Keep channels whose daughter count is one of `multiplicities`."
nbody(multiplicities::Integer...) =
    ChannelFilter(
    (_parent, channel, _daughters) ->
    length(channel.daughters) in multiplicities
)
"Keep channels with branching ratio ≥ `minimum_branching_ratio`."
brabove(minimum_branching_ratio) =
    ChannelFilter(
    (_parent, channel, _daughters) ->
    channel.br ≥ minimum_branching_ratio
)

# --- parent properties -----------------------------------------------------
"Keep channels whose parent mass lies in `[minimum_mass, maximum_mass]`."
parentmass(minimum_mass, maximum_mass) =
    ChannelFilter(
    (parent, _channel, _daughters) ->
    minimum_mass ≤ parent.mass ≤ maximum_mass
)
"Keep channels whose parent name is one of `names`."
fromparent(names::AbstractString...) =
    ChannelFilter((parent, _channel, _daughters) -> parent.name in names)
"Keep channels whose parent id is one of `ids`."
fromid(ids::Integer...) =
    ChannelFilter((parent, _channel, _daughters) -> parent.id in ids)
"Keep channels whose parent is a baryon (nonzero baryon number)."
baryonicparent() =
    ChannelFilter((parent, _channel, _daughters) -> isbaryon(parent))
"Keep channels whose parent is a meson (zero baryon number)."
mesonicparent() =
    ChannelFilter((parent, _channel, _daughters) -> ismeson(parent))
"Keep channels whose parent strangeness is one of `strangenesses`."
parentstrangeness(strangenesses::Integer...) =
    ChannelFilter(
    (parent, _channel, _daughters) ->
    parent.strangeness in strangenesses
)
"Keep channels whose parent carries nonzero strangeness."
strangeparent() =
    ChannelFilter((parent, _channel, _daughters) -> isstrange(parent))
"Keep channels whose parent carries nonzero charm."
charmparent() =
    ChannelFilter((parent, _channel, _daughters) -> parent.charm != 0)
"Keep channels whose parent isospin is one of `isospins` (physical values)."
function parentisospin(isospins::Real...)
    twice_isospins = map(isospin -> round(Int, 2 * isospin), isospins)
    return ChannelFilter(
        (parent, _channel, _daughters) ->
        parent.twiceisospin in twice_isospins
    )
end

# --- daughter properties ---------------------------------------------------
"Keep channels whose daughter masses all lie in `[minimum_mass, maximum_mass]`."
daughtermass(minimum_mass, maximum_mass) =
    ChannelFilter(
    (_parent, _channel, daughters) ->
    all(daughter -> minimum_mass ≤ daughter.mass ≤ maximum_mass, daughters)
)
"Keep channels with at least one daughter named in `names`."
todaughter(names::AbstractString...) =
    ChannelFilter(
    (_parent, _channel, daughters) ->
    any(daughter -> daughter.name in names, daughters)
)
"Keep channels with at least one daughter id in `ids`."
toid(ids::Integer...) =
    ChannelFilter(
    (_parent, _channel, daughters) ->
    any(daughter -> daughter.id in ids, daughters)
)

"Wrap a raw predicate `(parent, channel, daughters) -> Bool` as a `ChannelFilter`."
channelfilter(predicate) = ChannelFilter(predicate)

# ---------------------------------------------------------------------------
# Channel key: the masses and parent width that determine a decay matrix,
# with the canonicalizing invariants enforced by the constructor
# ---------------------------------------------------------------------------

struct ChannelKey
    # Field order is serialized; changing it requires a FORMAT bump.
    nbody::Int8
    ma::Float64          # parent mass
    mb::Float64          # observed daughter; never sorted with partners
    Γ::Float64           # parent width; zero selects the sharp path
    mth::Float64         # species decay threshold; zero on sharp keys, where
                         # no spectral shape is built and sharing stays maximal
    mc::Float64          # partner masses, sorted and zero-padded
    md::Float64
    me::Float64
end

function ChannelKey(
        parent_mass::Real, observed_mass::Real,
        partner_masses::AbstractVector{<:Real},
        parent_width::Real = 0.0,
        species_threshold::Real = 0.0,
    )
    partner_count = length(partner_masses)
    1 ≤ partner_count ≤ 3 ||
        throw(ArgumentError("ChannelKey supports 2-, 3-, and 4-body only"))

    sorted_partners = sort!(collect(Float64, partner_masses))
    padded_partners =
        ntuple(index -> index ≤ partner_count ? sorted_partners[index] : 0.0, 3)
    return ChannelKey(
        Int8(partner_count + 1), parent_mass, observed_mass,
        parent_width, species_threshold, padded_partners...
    )
end

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

struct CascadeTable{
        G <: AbstractRange{Float64},
        N,
        M <: RowBandMatrix{Float32},
    }
    grid::G
    orders::NamedTuple{(:nq, :nθ, :nµ, :nσ, :nma), Tuple{Int, Int, Int, Int, Int}}
    modes::NTuple{N, Int}
    width_cut::Float64
    width_window::Float64
    spectral::Symbol     # broad-parent mass distribution: :massdep or :constant
    meta::Dict{Symbol, Any}
    W::Dict{ChannelKey, NTuple{N, M}}
end

function _tablematrixtype(grid_length::Int)
    column_type = _rowbandcolumntype(grid_length)
    offset_type =
        _rowbandoffsettype(Base.checked_mul(grid_length, grid_length))
    return RowBandMatrix{Float32, column_type, offset_type}
end

function _packtablematrix(
        ::Type{RowBandMatrix{Float32, I, O}},
        matrix::AbstractMatrix{Float32},
    ) where {I <: Integer, O <: Integer}
    return RowBandMatrix(matrix, I, O)
end

struct CascadeTarget
    daughter::Int
    weight::Float64
end

struct CascadeGroup{N, M <: RowBandMatrix{Float32}}
    matrices::NTuple{N, M}
    targets::UnitRange{Int}
end

struct CascadeStep
    parent::Int
    groups::UnitRange{Int}
end

"""
    CascadePlan(table, decay_list)

Compile a filtered or complete decay topology into dense particle indices,
direct matrix references, and flat step/group/target arrays. The table remains
the keyed construction and cache representation; every public cascade path
executes through this compiled representation.
"""
struct CascadePlan{
        T <: CascadeTable,
        N,
        M <: RowBandMatrix{Float32},
    }
    table::T
    ids::Vector{Int}
    particles::Vector{ParticleInfo}
    stable_indices::Vector{Int}
    steps::Vector{CascadeStep}
    groups::Vector{CascadeGroup{N, M}}
    targets::Vector{CascadeTarget}
end

"""
    stableparticles(plan::CascadePlan) -> Vector{ParticleInfo}

Return the plan's terminal particles in ascending particle-ID order.
"""
stableparticles(plan::CascadePlan) = plan.particles[plan.stable_indices]

"""
    allparticles(plan::CascadePlan) -> Vector{ParticleInfo}

Return every particle compiled into the plan in ascending particle-ID order.
The returned vector is independent of the plan's internal storage.
"""
allparticles(plan::CascadePlan) = copy(plan.particles)

mutable struct _CascadeGroupBuilder{N, M <: RowBandMatrix{Float32}}
    matrices::NTuple{N, M}
    daughters::Vector{Int}
    weights::Dict{Int, Float64}
end

"""
    eachchannelkey(visit, dl; order = keys(dl.network.byparent),
                   width_cut = 1.0e-3, width_window = 4.0)

Call `visit(parent_id, channel, daughter_id, key)` for every enrolled
observed-daughter slot. Table construction and plan compilation share the same
threshold, width-tail, nonpositive-mass, and self-slot rules. Returns
enrollment and skip statistics. Rows with exactly zero branching weight are
ignored. Multiplicity and particle references were already checked when the
`DecayList` was resolved.
"""
function eachchannelkey(
        visit,
        dl::DecayList;
        order = keys(dl.network.byparent),
        width_cut::Float64 = 1.0e-3,
        width_window::Float64 = 4.0,
    )
    enrollment_stats = Dict(
        :self => 0, :closed => 0, :reopened => 0, :γdaughter => 0,
    )

    for parent_id in order
        if !haskey(dl.network.byparent, parent_id)
            continue
        end

        parent = dl.catalog.byid[parent_id]
        parent_mass = parent.mass
        parent_width = parent.width
        broad_parent = parent_width > width_cut

        # The species decay threshold: the lightest channel's summed daughter
        # masses (zero-branching rows are ignored here exactly as below). It
        # rides in broad keys so `decayfor` can build a threshold-vanishing
        # shape -- the shape belongs to the species, while each channel's own
        # closure keeps being handled by node dropping. Sharp keys carry 0.
        species_threshold = 0.0
        if broad_parent
            species_threshold = Inf
            for channel in dl.network.byparent[parent_id]
                iszero(channel.br) && continue
                channel_daughters = channel.daughters
                length(channel_daughters) == 1 &&
                    only(channel_daughters) == parent_id && continue
                species_threshold = min(species_threshold, sum(
                    dl.catalog.byid[daughter_id].mass
                        for daughter_id in channel_daughters
                ))
            end
            isfinite(species_threshold) || (species_threshold = 0.0)
        end

        for channel in dl.network.byparent[parent_id]
            iszero(channel.br) && continue
            daughter_ids = channel.daughters
            multiplicity = length(daughter_ids)

            if multiplicity == 1
                enrollment_stats[:self] += 1
                continue
            end
            daughter_masses = [
                dl.catalog.byid[daughter_id].mass for daughter_id in daughter_ids
            ]
            threshold = sum(daughter_masses) + 1.0e-6
            open_at_pole = parent_mass > threshold
            open_in_tail = broad_parent &&
                parent_mass + width_window * parent_width > threshold

            if !open_at_pole && !open_in_tail
                enrollment_stats[:closed] += 1
                continue
            end
            if !open_at_pole
                enrollment_stats[:reopened] += 1
            end

            key_parent_width = broad_parent ? parent_width : 0.0
            for (daughter_index, daughter_id) in enumerate(daughter_ids)
                daughter_mass = daughter_masses[daughter_index]

                if daughter_mass ≤ 0
                    enrollment_stats[:γdaughter] += 1
                    continue
                end
                if daughter_id == parent_id
                    # Keeps the in-place apply free of input/output aliasing.
                    continue
                end

                partner_masses = [
                    daughter_masses[index]
                        for index in eachindex(daughter_masses)
                        if index != daughter_index
                ]
                key = ChannelKey(
                    parent_mass,
                    daughter_mass,
                    partner_masses,
                    key_parent_width,
                    key_parent_width > 0 ? species_threshold : 0.0,
                )
                visit(parent_id, channel, daughter_id, key)
            end
        end
    end
    return enrollment_stats
end

# Every matrix key the cascade will read, together with enrollment statistics.
function neededkeys(
        decay_list::DecayList;
        width_cut::Float64 = 1.0e-3,
        width_window::Float64 = 4.0,
    )
    required_keys = Set{ChannelKey}()
    enrollment_stats = eachchannelkey(
        (_parent_id, _channel, _daughter_id, key) -> push!(required_keys, key),
        decay_list,
        width_cut = width_cut,
        width_window = width_window,
    )
    return required_keys, enrollment_stats
end

function _cascadeorder(dl::DecayList, dependencies)
    decaying_parents = sort!(collect(keys(dl.network.byparent)))
    downstream = Dict(parent_id => Int[] for parent_id in decaying_parents)

    for (parent_id, daughter_id) in dependencies
        push!(downstream[parent_id], daughter_id)
    end
    foreach(daughters -> sort!(unique!(daughters)), values(downstream))

    # Collapse every mutually dependent set before sorting. The resulting
    # component graph is acyclic, so only edges inside a component need an
    # approximate order.
    next_index = 0
    indices = Dict{Int, Int}()
    lowlinks = Dict{Int, Int}()
    stack = Int[]
    on_stack = Set{Int}()
    components = Vector{Vector{Int}}()

    function visit!(parent_id)
        next_index += 1
        indices[parent_id] = next_index
        lowlinks[parent_id] = next_index
        push!(stack, parent_id)
        push!(on_stack, parent_id)

        for daughter_id in downstream[parent_id]
            if !haskey(indices, daughter_id)
                visit!(daughter_id)
                lowlinks[parent_id] = min(
                    lowlinks[parent_id], lowlinks[daughter_id],
                )
            elseif daughter_id in on_stack
                lowlinks[parent_id] = min(
                    lowlinks[parent_id], indices[daughter_id],
                )
            end
        end

        if lowlinks[parent_id] == indices[parent_id]
            component = Int[]
            while true
                member = pop!(stack)
                delete!(on_stack, member)
                push!(component, member)
                member == parent_id && break
            end
            sort!(
                component;
                by = member -> (-dl.catalog.byid[member].mass, member),
            )
            push!(components, component)
        end
        return nothing
    end

    for parent_id in decaying_parents
        haskey(indices, parent_id) || visit!(parent_id)
    end

    component_of = Dict{Int, Int}()
    for (component_id, component) in pairs(components)
        for parent_id in component
            component_of[parent_id] = component_id
        end
    end

    component_downstream = [Int[] for _ in components]
    for (parent_id, daughter_ids) in downstream
        parent_component = component_of[parent_id]
        for daughter_id in daughter_ids
            daughter_component = component_of[daughter_id]
            parent_component == daughter_component ||
                push!(component_downstream[parent_component], daughter_component)
        end
    end

    incoming_count = zeros(Int, length(components))
    for daughter_components in component_downstream
        sort!(unique!(daughter_components))
        for daughter_component in daughter_components
            incoming_count[daughter_component] += 1
        end
    end

    cycle_count = 0
    cycle_species = 0
    for component in components
        cyclic = length(component) > 1 ||
            first(component) in downstream[first(component)]
        if cyclic
            cycle_count += 1
            cycle_species += length(component)
        end
    end
    if cycle_count > 0
        @warn "cascade feed-down graph has $cycle_count cycle components " *
              "involving $cycle_species species; members ordered by mass " *
              "within each component, cyclic feed-down approximate"
    end

    order = Int[]
    ready_components = findall(iszero, incoming_count)
    while !isempty(ready_components)
        component_id = argmax(
            candidate -> begin
                parent_id = first(components[candidate])
                (dl.catalog.byid[parent_id].mass, -parent_id)
            end,
            ready_components,
        )
        deleteat!(
            ready_components,
            findfirst(==(component_id), ready_components),
        )
        append!(order, components[component_id])

        for daughter_component in component_downstream[component_id]
            incoming_count[daughter_component] -= 1
            incoming_count[daughter_component] == 0 &&
                push!(ready_components, daughter_component)
        end
    end

    return order
end

function CascadePlan(
        table::CascadeTable{G, N, M},
        decay_list::DecayList,
    ) where {G, N, M}
    # Particle IDs remain the public API, while the numerical loop uses these
    # stable dense indices for direct vector access.
    ids = sort!(collect(keys(decay_list.catalog.byid)))
    particles = [decay_list.catalog.byid[particle_id] for particle_id in ids]
    dense_index = Dict(
        particle_id => index for (index, particle_id) in pairs(ids)
    )
    stable_indices = [
        dense_index[particle_id] for particle_id in decay_list.stable_ids
    ]

    # These dictionaries exist only while compiling the topology. Builders
    # preserve first-seen group/target order; the finalized plan below contains
    # only flat vectors, ranges, and dense indices.
    group_builders =
        Dict{Int, Vector{_CascadeGroupBuilder{N, M}}}()
    group_positions = Dict{Int, Dict{ChannelKey, Int}}()
    dependencies = Set{Tuple{Int, Int}}()

    # A deterministic traversal makes group order and construction failures
    # independent of Dict insertion order.
    compile_order = sort!(collect(keys(decay_list.network.byparent)))
    eachchannelkey(
        decay_list;
        order = compile_order,
        width_cut = table.width_cut,
        width_window = table.width_window,
    ) do parent_id, channel, daughter_id, key
        builders = get!(group_builders, parent_id) do
            _CascadeGroupBuilder{N, M}[]
        end
        positions = get!(group_positions, parent_id) do
            Dict{ChannelKey, Int}()
        end

        group_index = get(positions, key, 0)
        if group_index == 0
            push!(
                builders,
                _CascadeGroupBuilder{N, M}(
                    table.W[key],
                    Int[],
                    Dict{Int, Float64}(),
                ),
            )
            group_index = length(builders)
            positions[key] = group_index
        end

        # Sum the branching weight for every matching daughter slot. This
        # folds repeated daughters and equivalent channels into one target.
        daughter = dense_index[daughter_id]
        builder = builders[group_index]
        if haskey(builder.weights, daughter)
            builder.weights[daughter] += channel.br
        else
            push!(builder.daughters, daughter)
            builder.weights[daughter] = channel.br
        end

        # Only daughters that also decay constrain the feeder-before-fed
        # parent order. The Set removes duplicate edges from repeated slots.
        if haskey(decay_list.network.byparent, daughter_id)
            push!(dependencies, (parent_id, daughter_id))
        end
    end

    parent_order = _cascadeorder(decay_list, dependencies)

    # Flatten the temporary parent/group/target hierarchy in cascade order.
    # UnitRanges retain the hierarchy without allocating nested vectors.
    steps = CascadeStep[]
    groups = CascadeGroup{N, M}[]
    targets = CascadeTarget[]

    for parent_id in parent_order
        builders = get(group_builders, parent_id, nothing)
        builders === nothing && continue

        group_start = length(groups) + 1
        for builder in builders
            target_start = length(targets) + 1
            for daughter in builder.daughters
                push!(
                    targets,
                    CascadeTarget(daughter, builder.weights[daughter]),
                )
            end
            push!(
                groups,
                CascadeGroup{N, M}(
                    builder.matrices,
                    target_start:length(targets),
                ),
            )
        end
        push!(
            steps,
            CascadeStep(
                dense_index[parent_id],
                group_start:length(groups),
            ),
        )
    end

    return CascadePlan{typeof(table), N, M}(
        table,
        ids,
        particles,
        stable_indices,
        steps,
        groups,
        targets,
    )
end

function decayfor(
        key::ChannelKey,
        rules::Rules,
        width_window::Float64,
        spectral::Symbol,
    )
    if key.Γ > 0
        minimum_mass = max(key.ma - width_window * key.Γ, 0.05)
        maximum_mass = key.ma + width_window * key.Γ
        # The threshold-vanishing shape needs q at the pole; a species whose
        # pole sits at or below its own threshold (a reopened state -- the same
        # ones the :reopened enrollment stat counts) has none, and stays on the
        # constant shape.
        mass_distribution = spectral === :massdep && key.mth < key.ma ?
            MassDependentBW(
                key.ma, key.Γ, key.mth;
                mmin = minimum_mass,
                mmax = maximum_mass,
            ) :
            BreitWigner(
                key.ma, key.Γ;
                mmin = minimum_mass,
                mmax = maximum_mass,
            )
        if key.nbody == 2
            return BroadParent(key.mb, key.mc, mass_distribution, rules)
        elseif key.nbody == 3
            return BroadParent(key.mb, key.mc, key.md, mass_distribution, rules)
        else
            return BroadParent(
                key.mb, key.mc, key.md, key.me,
                mass_distribution, rules
            )
        end
    end

    if key.nbody == 2
        return TwoBodyDecay(key.ma, key.mb, key.mc, rules)
    elseif key.nbody == 3
        return ThreeBodyDecay(key.ma, key.mb, key.mc, key.md, rules)
    else
        return FourBodyDecay(key.ma, key.mb, key.mc, key.md, key.me, rules)
    end
end

function buildkeys!(
        table::CascadeTable{G, N, M},
        keys_to_build::Vector{ChannelKey},
    ) where {G, N, M}
    if N == 0
        for key in keys_to_build
            table.W[key] = ()
        end
        return length(keys_to_build)
    end

    modes = table.modes
    rules = Rules(; table.orders...)
    grid_length = length(table.grid)

    # Tasks can migrate between threads, so scratch storage belongs to each task.
    scratch_for_task = OncePerTask{NTuple{N, Vector{Float64}}}(
        () -> rowscratch(modes, grid_length),
    )
    matrices_per_key =
        Vector{NTuple{N, M}}(undef, length(keys_to_build))

    Threads.@threads :greedy for key_index in eachindex(keys_to_build)
        decay = decayfor(
            keys_to_build[key_index], rules,
            table.width_window, table.spectral,
        )
        dense_matrices =
            map(_mode -> zeros(Float32, grid_length, grid_length), modes)
        # The outer key loop already uses the thread pool.
        decaymatrices!(
            dense_matrices, decay, modes,
            table.grid, table.grid; threaded = false,
            scratch = scratch_for_task()
        )

        packed_matrices = ntuple(Val(N)) do mode_index
            _packtablematrix(M, dense_matrices[mode_index])
        end
        matrices_per_key[key_index] = packed_matrices
    end

    # Dict mutation is kept serial because Dict is not thread-safe.
    for (key, mode_matrices) in zip(keys_to_build, matrices_per_key)
        table.W[key] = mode_matrices
    end
    return length(keys_to_build)
end

# Build the matrices required by `decay_list` that are not already present.
function ensure!(table::CascadeTable, decay_list::DecayList)
    required_keys, _ = neededkeys(
        decay_list;
        width_cut = table.width_cut,
        width_window = table.width_window,
    )
    missing_keys = [key for key in required_keys if !haskey(table.W, key)]
    if isempty(missing_keys)
        return 0
    end
    return buildkeys!(table, missing_keys)
end

"Remove matrices not referenced by `decay_list`; return the orphan count."
function prune!(table::CascadeTable, decay_list::DecayList)
    required_keys, _ = neededkeys(
        decay_list;
        width_cut = table.width_cut,
        width_window = table.width_window,
    )
    orphaned_keys = [key for key in keys(table.W) if key ∉ required_keys]
    for key in orphaned_keys
        delete!(table.W, key)
    end
    return length(orphaned_keys)
end

function _table_metadata(decay_list::DecayList)
    return Dict{Symbol, Any}(
        :sha => decay_list.sha,
        :path => decay_list.path,
        :created => string(now()),
    )
end

"""
    CascadeTable(dl; grid = range(0, 4.0, length = 134),
                 orders = (;), modes = [0, 2],
                 width_cut = 1.0e-3, width_window = 4.0,
                 cache = nothing, rebuild = false)

Load the tables from `cache` if present and compatible (same format, grid,
orders, modes, `width_cut`, and `width_window`), else build and save. `orders`
may be any subset of
`(nq = 32, nθ = 8, nµ = 16, nσ = 8, nma = 16)` — omitted fields are filled
with those defaults. A decay-list hash mismatch keeps the tables (lenient
policy — keys self-invalidate through their masses) but warns and builds any
missing keys on demand.

`modes` accepts any iterable of integers and is stored as an immutable tuple,
so its length specializes the fused matrix build.

Parents with `width > width_cut` use the finite-width `BroadParent` path.
Set `width_cut = Inf` to make every parent sharp at its pole mass, or `0.0` to
use every positive width. Broad-parent mass integration is truncated to
`max(mass - width_window * width, 0.05 GeV)` through
`mass + width_window * width`. Both options are concrete `Float64` table
fields and cache-compatibility settings; their physical ranges are caller
responsibility.

`spectral` selects the broad-parent mass distribution: `:massdep` (default)
averages over the threshold-vanishing `MassDependentBW`, whose width
`Γ(m) ∝ (mR/m)·q(m)/q(mR)` dies at the species' decay threshold; `:constant`
keeps the constant-width `BreitWigner`. Species whose pole sits at or below
their own threshold use the constant shape either way. Like the width options
it is a concrete table field and a cache-compatibility setting.

`grid` is an `AbstractRange{Float64}`. Matrix assembly trusts that range and
uses the uniform-grid interpolation kernel without validation scans.
"""
function CascadeTable(
        decay_list::DecayList;
        grid::AbstractRange{Float64} = range(0.0, 4.0, length = 134),
        orders = (;),
        modes = [0, 2],
        width_cut::Float64 = 1.0e-3,
        width_window::Float64 = 4.0,
        spectral::Symbol = :massdep,
        cache = nothing, rebuild = false
    )
    spectral === :massdep || spectral === :constant || throw(ArgumentError(
        "spectral must be :massdep or :constant, got $(repr(spectral))",
    ))
    default_orders = (nq = 32, nθ = 8, nµ = 16, nσ = 8, nma = 16)
    resolved_orders = merge(default_orders, orders)
    selected_modes = Tuple(collect(Int, modes))

    if cache !== nothing && !rebuild
        cached_table = loadtables(cache)
        cache_matches =
            cached_table !== nothing &&
            cached_table.grid == grid &&
            cached_table.orders == resolved_orders &&
            cached_table.modes == selected_modes &&
            cached_table.width_cut == width_cut &&
            cached_table.width_window == width_window &&
            cached_table.spectral == spectral

        if cache_matches
            if !isequal(
                    get(cached_table.meta, :sha, nothing),
                    decay_list.sha,
                )
                @warn "decay list changed since tables were built; building missing keys on demand (prune! removes orphans)"
            end
            built_count = ensure!(cached_table, decay_list)
            if built_count > 0
                savetables(cached_table, cache)
            end
            return cached_table
        end
    end

    matrix_type = _tablematrixtype(length(grid))
    table = CascadeTable{
        typeof(grid),
        length(selected_modes),
        matrix_type,
    }(
        grid,
        resolved_orders,
        selected_modes,
        width_cut,
        width_window,
        spectral,
        _table_metadata(decay_list),
        Dict{
            ChannelKey,
            NTuple{length(selected_modes), matrix_type},
        }(),
    )
    ensure!(table, decay_list)
    if cache !== nothing
        savetables(table, cache)
    end
    return table
end

"""
    CascadePlan(decay_list::DecayList; table_options...)

Build or load the [`CascadeTable`](@ref) and compile `decay_list` into one
prepared object. All keyword arguments are forwarded to `CascadeTable`.
Reuse the returned plan across `runcascade` calls with different primary
callbacks.
"""
function CascadePlan(decay_list::DecayList; table_options...)
    table = CascadeTable(decay_list; table_options...)
    return CascadePlan(table, decay_list)
end

# ---------------------------------------------------------------------------
# Persistence (Serialization stdlib: local cache; format is version-checked)
# ---------------------------------------------------------------------------

function savetables(table::CascadeTable, path::AbstractString)
    return open(path, "w") do io
        write(io, CACHE_MAGIC)
        write(io, UInt8(FORMAT))
        serialize(
            io,
            (
                table.grid,
                table.orders,
                table.modes,
                table.width_cut,
                table.width_window,
                table.spectral,
                table.meta,
                table.W,
            ),
        )
    end
end

function loadtables(path::AbstractString)
    stored_data = try
        open(path, "r") do io
            magic = read(io, ncodeunits(CACHE_MAGIC))
            magic == codeunits(CACHE_MAGIC) || return nothing
            read(io, UInt8) == FORMAT || return nothing
            return deserialize(io)
        end
    catch
        return nothing
    end
    stored_data isa Tuple && length(stored_data) == 8 || return nothing
    grid, orders, modes, width_cut, width_window, spectral, metadata, matrices =
        stored_data
    grid isa AbstractRange{Float64} || return nothing
    modes isa Tuple{Vararg{Int}} || return nothing
    width_cut isa Float64 || return nothing
    width_window isa Float64 || return nothing
    spectral === :massdep || spectral === :constant || return nothing
    metadata isa Dict{Symbol, Any} || return nothing
    return _restoretable(
        grid,
        orders,
        modes,
        width_cut,
        width_window,
        spectral,
        metadata,
        matrices,
    )
end

function _restoretable(
        grid::G,
        orders,
        modes::NTuple{N, Int},
        width_cut::Float64,
        width_window::Float64,
        spectral::Symbol,
        metadata,
        matrices::Dict{ChannelKey, V},
    ) where {G, N, V <: Tuple}
    matrix_type = try
        _tablematrixtype(length(grid))
    catch
        return nothing
    end
    V === NTuple{N, matrix_type} || return nothing
    expected_size = (length(grid), length(grid))
    try
        for mode_matrices in values(matrices)
            for matrix in mode_matrices
                size(matrix) == expected_size || return nothing
                _validate_rowband_storage(
                    matrix.column_count,
                    matrix.column_starts,
                    matrix.value_offsets,
                    matrix.values,
                )
            end
        end
        return CascadeTable{G, N, matrix_type}(
            grid,
            orders,
            modes,
            width_cut,
            width_window,
            spectral,
            metadata,
            matrices,
        )
    catch
        return nothing
    end
end

_restoretable(
    _grid,
    _orders,
    _modes,
    _width_cut,
    _width_window,
    _spectral,
    _metadata,
    _matrices,
) = nothing

# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------

function _initialize_plan_spectra(
        plan::CascadePlan{T, N, M},
        primary,
        output_indices,
    ) where {T, N, M}
    species_count = length(plan.ids)
    grid_length = length(plan.table.grid)

    # Each mode task receives one dense particle-indexed vector, avoiding a
    # particle→mode container lookup in the numerical loop.
    spectra_by_mode = ntuple(
        _ -> Vector{Vector{Float64}}(undef, species_count),
        Val(N),
    )

    for dense_index in eachindex(plan.ids)
        particle_id = plan.ids[dense_index]
        particle = plan.particles[dense_index]

        for mode_index in 1:N
            mode = plan.table.modes[mode_index]
            spectrum =
                Vector{Float64}(primary(particle, mode, plan.table.grid))
            length(spectrum) == grid_length ||
                throw(
                DimensionMismatch(
                    "primary spectrum for particle $particle_id, mode $mode has " *
                        "length $(length(spectrum)); expected $grid_length",
                )
            )
            spectra_by_mode[mode_index][dense_index] = spectrum
        end
    end

    output_count = length(output_indices)
    spectra = Dict{Int, Vector{Vector{Float64}}}()
    primary_spectra = Dict{Int, Vector{Vector{Float64}}}()
    sizehint!(spectra, output_count)
    sizehint!(primary_spectra, output_count)
    for dense_index in output_indices
        particle_id = plan.ids[dense_index]
        mode_spectra = [
            spectra_by_mode[mode_index][dense_index] for mode_index in 1:N
        ]
        spectra[particle_id] = mode_spectra
        primary_spectra[particle_id] = copy.(mode_spectra)
    end
    return spectra, primary_spectra, spectra_by_mode
end

function _apply_plan_mode!(
        mode_spectra::Vector{Vector{Float64}},
        plan::CascadePlan{T, N, M},
        mode_index::Int,
        scratch::Vector{Float64},
    ) where {T, N, M}
    @inbounds for step in plan.steps
        parent_spectrum = mode_spectra[step.parent]

        for group_index in step.groups
            group = plan.groups[group_index]
            matrix = group.matrices[mode_index]

            if length(group.targets) == 1
                target = plan.targets[first(group.targets)]
                daughter_spectrum = mode_spectra[target.daughter]
                mul!(
                    daughter_spectrum,
                    matrix,
                    parent_spectrum,
                    target.weight,
                    1.0,
                )
            else
                mul!(scratch, matrix, parent_spectrum)
                for target_index in group.targets
                    target = plan.targets[target_index]
                    daughter_spectrum = mode_spectra[target.daughter]
                    _cascade_axpy!(
                        target.weight,
                        scratch,
                        daughter_spectrum,
                    )
                end
            end
        end
    end
    return mode_spectra
end

function _apply_plan!(
        spectra_by_mode::NTuple{N, Vector{Vector{Float64}}},
        plan::CascadePlan{T, N, M},
        parallel::Symbol,
    ) where {T, N, M}
    grid_length = length(plan.table.grid)
    if parallel === :modes && Threads.nthreads() > 1 && N > 1
        @sync for mode_index in 1:N
            Threads.@spawn begin
                # Deliberately not named `scratch`. `@spawn` builds a closure,
                # and a name assigned both inside it and in the serial branch
                # below is treated as one shared variable: Julia boxes it, and
                # `_apply_plan!` infers it as `Core.Box`. Distinct names keep
                # each buffer local to its branch and the function inferred.
                mode_scratch = Vector{Float64}(undef, grid_length)
                _apply_plan_mode!(
                    spectra_by_mode[mode_index],
                    plan,
                    mode_index,
                    mode_scratch,
                )
            end
        end
    else
        scratch = Vector{Float64}(undef, grid_length)
        for mode_index in 1:N
            _apply_plan_mode!(
                spectra_by_mode[mode_index],
                plan,
                mode_index,
                scratch,
            )
        end
    end
    return spectra_by_mode
end

function _check_run_options(parallel, output)
    parallel === :serial || parallel === :modes ||
        throw(
        ArgumentError(
            "parallel must be :serial or :modes, got $(repr(parallel))",
        )
    )
    output === :stable || output === :all ||
        throw(
        ArgumentError(
            "output must be :stable or :all, got $(repr(output))",
        )
    )
    return nothing
end

"""
    runcascade(plan::CascadePlan; primary, parallel = :serial,
               output = :stable) -> (F, Fprim)

Run a compiled plan without topology traversal, key construction, or dictionary
lookup in the numerical loop. `parallel = :modes` assigns each Fourier mode to
an independent task when multiple Julia threads and modes are available.
Primary callbacks are always initialized serially. A plan is read-only and may
be shared by concurrent calls whose callbacks and outputs are independent.
Only stable particles are returned by default; use `output = :all` to retain
intermediate resonances for diagnostics.
"""
function runcascade(
        plan::CascadePlan;
        primary,
        parallel = :serial,
        output = :stable,
    )
    _check_run_options(parallel, output)
    output_indices = output === :stable ?
        plan.stable_indices :
        eachindex(plan.ids)
    spectra, primary_spectra, spectra_by_mode =
        _initialize_plan_spectra(plan, primary, output_indices)
    _apply_plan!(spectra_by_mode, plan, parallel::Symbol)
    return (F = spectra, Fprim = primary_spectra)
end

"""
    runcascade(table::CascadeTable, decay_list::DecayList;
               primary, parallel = :serial, output = :stable) -> (F, Fprim)

Compile `decay_list` against an existing table, then run the resulting
[`CascadePlan`](@ref). Construct the plan explicitly when reusing the same
table and topology with multiple primary callbacks.
"""
function runcascade(
        table::CascadeTable,
        decay_list::DecayList;
        primary,
        parallel = :serial,
        output = :stable,
    )
    _check_run_options(parallel, output)
    plan = CascadePlan(table, decay_list)
    return runcascade(plan; primary, parallel, output)
end

"""
    runcascade(decay_list::DecayList; primary,
               grid = range(0, 4.0, length = 134),
               orders = (;), modes = [0, 2],
               width_cut = 1.0e-3, width_window = 4.0,
               cache = nothing, rebuild = false,
               parallel = :serial, output = :stable) -> (F, Fprim)

Build or load a table, compile a plan, and run the cascade in one call.
`grid`, `orders`, `modes`, `width_cut`, `width_window`, `cache`, and `rebuild`
configure the table;
`primary`, `parallel`, and `output` configure execution.

This is the convenient one-shot interface. Construct a [`CascadePlan`](@ref)
explicitly when it will be reused with multiple primary callbacks.
"""
function runcascade(
        decay_list::DecayList;
        primary,
        parallel = :serial,
        output = :stable,
        table_options...,
    )
    _check_run_options(parallel, output)
    plan = CascadePlan(decay_list; table_options...)
    return runcascade(plan; primary, parallel, output)
end

# ---------------------------------------------------------------------------
# Decay-list validation: conservation laws + branching-ratio sanity
# ---------------------------------------------------------------------------

"""
    validate(dl::DecayList; brtol = 0.02, verbose = true) -> NamedTuple

Check every decay channel of `dl` against the physics it must obey and return
the statistics (printing a report when `verbose`). Self-contained — no PDG
needed. The checks, and how to read them:

  * **charge** and **baryon number** — conserved by *every* channel (strong, EM,
    weak alike); any nonzero count is a data error.
  * **branching ratios** — `ΣBR ≈ 1` per decaying parent (`brtol`), and each
    `BR` lies in `[0, 1]` up to a numerical tolerance.
  * **exact duplicate channels** — repeated `(BR, daughter multiset)` rows are
    counted so accidental double weighting is visible without rejecting raw
    legacy data.
  * **daughter resolution** — every daughter id must be a known species.
  * **terminal particles** — `selfstable` reports the resolved terminal set;
    `selfchannels` separately counts explicit one-body self rows because split
    particle files may omit those rows.
  * **strangeness / charm / bottomness change** — *informational*: zero means a
    strong/EM-only list; nonzero counts the weak flavour-changing decays.
  * **threshold** — channels kinematically closed at the parent pole mass.
"""
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

            if !(0.0 ≤ channel.br ≤ 1 + 1.0e-6)
                invalid_branching_ratio_count += 1
            end
            if !all(
                    daughter_id -> haskey(decay_list.catalog.byid, daughter_id),
                    daughter_ids
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
        multiplicity = sort(collect(multiplicity_counts)),
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
            decay_list, stats, charge_violations,
            baryon_violations, branching_ratio_sums, brtol
        )
    end
    return stats
end

function _print_validation_report(
        decay_list, stats, charge_violations,
        baryon_violations, branching_ratio_sums,
        branching_ratio_tolerance
    )
    particle_name(particle_id) = haskey(decay_list.catalog.byid, particle_id) ?
        decay_list.catalog.byid[particle_id].name :
        "?$particle_id"
    count_status(count) = count == 0 ? "OK" : "*** $count ***"

    println("── decay-list validation ", repeat("─", 44))
    @printf(
        "  %d particles, %d decay channels (%d stable), multiplicities %s\n",
        stats.particles, stats.channels, stats.selfstable, string(stats.multiplicity)
    )
    println("  conservation (must be zero):")
    @printf(
        "    charge violations ........ %s\n",
        count_status(stats.charge_violations)
    )
    @printf(
        "    baryon-number violations . %s\n",
        count_status(stats.baryon_violations)
    )
    @printf(
        "    unresolved daughters ..... %s\n",
        count_status(stats.unresolved)
    )
    @printf(
        "    BR out of [0,1] .......... %s\n",
        count_status(stats.br_out_of_range)
    )
    @printf(
        "    exact duplicate channels . %s\n",
        count_status(stats.duplicate_channels)
    )
    @printf(
        "  branching ratios: max |ΣBR-1| = %.2e,  %s parents beyond %.0f%%\n",
        stats.br_maxdev, count_status(stats.br_norm_outliers),
        100 * branching_ratio_tolerance
    )
    @printf(
        "  informational: %d strangeness-, %d charm-, %d bottomness-changing (weak),  %d closed-at-pole\n",
        stats.strangeness_changing, stats.charm_changing,
        stats.bottomness_changing, stats.subthreshold
    )

    for (label, violations) in (
            ("charge", charge_violations),
            ("baryon", baryon_violations),
        )
        if isempty(violations)
            continue
        end
        println("  first $label violations:")
        for (parent_id, daughter_ids) in first(violations, 3)
            daughter_names =
                (particle_name(daughter_id) for daughter_id in daughter_ids)
            @printf(
                "    %s -> %s\n",
                particle_name(parent_id), join(daughter_names, " ")
            )
        end
    end

    worst_branching_sums = sort(
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
                particle_name(parent_id), branching_ratio_sum
            )
        end
    end
    return println(repeat("─", 70))
end

end # module
