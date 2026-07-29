"""
    CascadeTables

Structured, persistent decay-matrix tables for the full resonance cascade.

    dl = DecayList("decays_QM2016Plus_massorder.dat")
    res = runcascade(                                 # build, compile, and run
        dl; primary = myprimaries, cache = "Wtables.jls",
    )

Design (see matrix_notes.pdf for details):
  * decay matrices are serialized; source identity and bytes remain in the
    table metadata for provenance;
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
using Serialization, SHA, Dates, Printf
using LinearAlgebra: axpy!
import LinearAlgebra: mul!
include("CascadeLinearAlgebra.jl")

export DecayList, ChannelKey, CascadeTable, CascadePlan, runcascade,
    prune!, validate, DEFAULT_ORDERS,
    savetables, loadtables, ParticleInfo, DecayChannel,
    stableparticles,
    isospin, spin, isbaryon, ismeson, isantiparticle, isstrange,
    ChannelFilter, channelfilter, nbody, parentmass, daughtermass, brabove,
    fromparent, todaughter, fromid, toid, baryonicparent, mesonicparent,
    parentstrangeness, strangeparent, charmparent, parentisospin

# Increment when the serialized layout or generated matrix values change.
const FORMAT = 8
const CACHE_MAGIC = "FastReso.CascadeTables\n"

# Parents wider than WIDTHCUT are averaged within a window of up to ±WIDTHK·Γ.
const WIDTHCUT = 0.001                 # GeV
const WIDTHK = 4.0

"Default quadrature orders. `CascadeTable` accepts any subset and fills in the rest."
const DEFAULT_ORDERS = (nq = 32, nθ = 8, nµ = 16, nσ = 8, nma = 16)

# ---------------------------------------------------------------------------
# Decay list wrapper: parsed structures + identity of the file that made them
# ---------------------------------------------------------------------------

struct DecayList
    parts::Dict{Int, ParticleInfo}
    byparent::Dict{Int, Vector{DecayChannel}}
    stable_ids::Vector{Int}
    path::String
    sha::String
end

function _stableparticleids(particles, channels_by_parent)
    stable_ids = Int[]
    for particle_id in keys(particles)
        channels = get(channels_by_parent, particle_id, nothing)
        if channels === nothing || all(
                channel ->
                length(channel.daughters) == 1 &&
                    only(channel.daughters) == particle_id,
                channels,
            )
            push!(stable_ids, particle_id)
        end
    end
    return sort!(stable_ids)
end

function DecayList(
        particles::Dict{Int, ParticleInfo},
        channels_by_parent::Dict{Int, Vector{DecayChannel}},
        path::AbstractString,
        sha::AbstractString,
    )
    return DecayList(
        particles,
        channels_by_parent,
        _stableparticleids(particles, channels_by_parent),
        String(path),
        String(sha),
    )
end

function DecayList(path::AbstractString)
    particles, channels_by_parent = read_decaylist(path)
    return DecayList(
        particles, channels_by_parent, String(path),
        bytes2hex(sha256(read(path)))
    )
end

"""
    stableparticles(decay_list::DecayList) -> Vector{ParticleInfo}

Return particles marked terminal by the decay list in ascending particle-ID
order. Terminal particles use one-body self-channels in file-backed lists;
particles without a channel block are also treated as terminal for
programmatically constructed lists.
"""
stableparticles(decay_list::DecayList) =
    [decay_list.parts[particle_id] for particle_id in decay_list.stable_ids]

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

    for (parent_id, channels) in dl.byparent
        parent = dl.parts[parent_id]
        selected = DecayChannel[]

        for channel in channels
            daughter_ids = channel.daughters
            if !all(daughter_id -> haskey(dl.parts, daughter_id), daughter_ids)
                continue
            end

            daughters = ParticleInfo[
                dl.parts[daughter_id] for daughter_id in daughter_ids
            ]
            if selection(parent, channel, daughters)
                push!(selected, channel)
            end
        end

        if !isempty(selected)
            channels_by_parent[parent_id] = selected
        end
    end

    return DecayList(
        dl.parts,
        channels_by_parent,
        dl.stable_ids,
        dl.path,
        dl.sha,
    )
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
    mc::Float64          # partner masses, sorted and zero-padded
    md::Float64
    me::Float64
end

function ChannelKey(
        parent_mass::Real, observed_mass::Real,
        partner_masses::AbstractVector{<:Real},
        parent_width::Real = 0.0
    )
    partner_count = length(partner_masses)
    1 ≤ partner_count ≤ 3 ||
        throw(ArgumentError("ChannelKey supports 2-, 3-, and 4-body only"))

    sorted_partners = sort!(collect(Float64, partner_masses))
    padded_partners =
        ntuple(index -> index ≤ partner_count ? sorted_partners[index] : 0.0, 3)
    return ChannelKey(
        Int8(partner_count + 1), parent_mass, observed_mass,
        parent_width, padded_partners...
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
    meta::Dict{Symbol, Any}
    W::Dict{ChannelKey, Vector{M}}
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

function _packtablematrices(
        grid,
        matrices::Dict{ChannelKey, Vector{Matrix{Float32}}},
    )
    matrix_type = _tablematrixtype(length(grid))
    packed = Dict{ChannelKey, Vector{matrix_type}}()
    sizehint!(packed, length(matrices))

    for (key, mode_matrices) in matrices
        packed_modes = Vector{matrix_type}(undef, length(mode_matrices))
        for mode_index in eachindex(mode_matrices)
            packed_modes[mode_index] =
                _packtablematrix(matrix_type, mode_matrices[mode_index])
        end
        packed[key] = packed_modes
    end
    return packed
end

function CascadeTable(
        grid,
        orders,
        modes,
        metadata,
        matrices::Dict{ChannelKey, Vector{Matrix{Float32}}},
    )
    return CascadeTable(
        grid,
        orders,
        modes,
        metadata,
        _packtablematrices(grid, matrices),
    )
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
the keyed construction and cache representation; a plan is an in-memory
snapshot for repeated [`runcascade`](@ref) calls.
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

mutable struct _CascadeGroupBuilder{N, M <: RowBandMatrix{Float32}}
    matrices::NTuple{N, M}
    daughters::Vector{Int}
    weights::Dict{Int, Float64}
end

"""
    eachchannelkey(visit, dl; order = keys(dl.byparent))

Call `visit(parent_id, channel, daughter_id, key)` for every enrolled
observed-daughter slot. Table construction and cascade execution share the same
multiplicity, threshold, width-tail, unresolved-daughter, nonpositive-mass, and
self-slot rules. Returns enrollment and skip statistics.
"""
function eachchannelkey(visit, dl::DecayList; order = keys(dl.byparent))
    enrollment_stats = Dict(
        :self => 0, :nd5 => 0, :closed => 0, :reopened => 0,
        :γdaughter => 0, :unknown => 0
    )

    for parent_id in order
        if !haskey(dl.byparent, parent_id)
            continue
        end

        parent = dl.parts[parent_id]
        parent_mass = parent.mass
        parent_width = parent.width
        broad_parent = parent_width > WIDTHCUT

        for channel in dl.byparent[parent_id]
            daughter_ids = channel.daughters
            multiplicity = length(daughter_ids)

            if multiplicity == 1
                enrollment_stats[:self] += 1
                continue
            end
            if multiplicity ≥ 5
                enrollment_stats[:nd5] += 1
                continue
            end
            if !all(daughter_id -> haskey(dl.parts, daughter_id), daughter_ids)
                enrollment_stats[:unknown] += 1
                continue
            end

            daughter_masses = [
                dl.parts[daughter_id].mass for daughter_id in daughter_ids
            ]
            threshold = sum(daughter_masses) + 1.0e-6
            open_at_pole = parent_mass > threshold
            open_in_tail = broad_parent &&
                parent_mass + WIDTHK * parent_width > threshold

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
                )
                visit(parent_id, channel, daughter_id, key)
            end
        end
    end
    return enrollment_stats
end

# Every matrix key the cascade will read, together with enrollment statistics.
function neededkeys(decay_list::DecayList)
    required_keys = Set{ChannelKey}()
    enrollment_stats = eachchannelkey(
        (_parent_id, _channel, _daughter_id, key) -> push!(required_keys, key),
        decay_list,
    )
    return required_keys, enrollment_stats
end

"""
    cascadeorder(dl::DecayList) -> Vector{Int}

Order decaying parents feeder-before-fed, so every parent spectrum is complete
before that parent decays. This is a topological sort of the enrolled feed-down
graph. Among parents whose inputs are ready, process the heaviest first and use
particle ID as a deterministic tie-break.

A mass-only ordering is insufficient because a broad parent's spectral tail
can reopen a channel to a heavier daughter. If the graph contains a cycle, no
exact single-sweep order exists; append all parents blocked by that cycle in
descending mass order and warn that their cross feed-down is approximate.
"""
function cascadeorder(dl::DecayList)
    dependencies = Tuple{Int, Int}[]

    # An enrolled parent → daughter slot is a dependency only when the daughter
    # has a parent block in the cascade sweep.
    eachchannelkey(dl) do parent_id, _channel, daughter_id, _key
        if daughter_id != parent_id && haskey(dl.byparent, daughter_id)
            push!(dependencies, (parent_id, daughter_id))
        end
    end
    return _cascadeorder(dl, dependencies)
end

function _cascadeorder(dl::DecayList, dependencies)
    decaying_parents = collect(keys(dl.byparent))
    incoming_count = Dict(parent_id => 0 for parent_id in decaying_parents)
    downstream = Dict(parent_id => Int[] for parent_id in decaying_parents)

    for (parent_id, daughter_id) in dependencies
        push!(downstream[parent_id], daughter_id)
        incoming_count[daughter_id] += 1
    end

    order = Int[]
    unprocessed = Set(decaying_parents)

    while !isempty(unprocessed)
        ready_parents = [
            parent_id for parent_id in unprocessed
                if incoming_count[parent_id] == 0
        ]

        if isempty(ready_parents)
            blocked_parents = sort!(
                collect(unprocessed);
                by = parent_id -> (-dl.parts[parent_id].mass, parent_id),
            )
            @warn "cascade feed-down graph has a cycle (mutually reopening broad \
                   parents); $(length(blocked_parents)) species ordered by mass, cross \
                   feed-down approximate" maxlog = 1
            append!(order, blocked_parents)
            break
        end

        parent_id = argmax(
            candidate -> (dl.parts[candidate].mass, -candidate),
            ready_parents,
        )
        push!(order, parent_id)
        delete!(unprocessed, parent_id)

        for daughter_id in downstream[parent_id]
            incoming_count[daughter_id] -= 1
        end
    end

    return order
end

function _planmatrices(
        table::CascadeTable{G, N, M},
        key::ChannelKey,
    ) where {G, N, M}
    matrices = table.W[key]
    length(matrices) == N ||
        throw(
        DimensionMismatch(
            "table key $key has $(length(matrices)) matrices for $N modes",
        )
    )

    grid_length = length(table.grid)
    expected_size = (grid_length, grid_length)
    for (mode_index, matrix) in pairs(matrices)
        size(matrix) == expected_size ||
            throw(
            DimensionMismatch(
                "table key $key mode index $mode_index has size $(size(matrix)); " *
                    "expected $expected_size",
            )
        )
    end
    return ntuple(mode_index -> matrices[mode_index], Val(N))
end

function CascadePlan(
        table::CascadeTable{G, N, M},
        decay_list::DecayList,
    ) where {G, N, M}
    # Particle IDs remain the public API, while the numerical loop uses these
    # stable dense indices for direct vector access.
    ids = sort!(collect(keys(decay_list.parts)))
    particles = [decay_list.parts[particle_id] for particle_id in ids]
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
    # Equal keys share one tuple of references to table matrices. ntuple in
    # _planmatrices copies references, never the matrix payload.
    matrix_cache =
        Dict{ChannelKey, NTuple{N, M}}()
    dependencies = Set{Tuple{Int, Int}}()

    # A deterministic traversal makes group order and construction failures
    # independent of Dict insertion order.
    compile_order = sort!(collect(keys(decay_list.byparent)))
    eachchannelkey(decay_list; order = compile_order) do parent_id, channel, daughter_id, key
        builders = get!(group_builders, parent_id) do
            _CascadeGroupBuilder{N, M}[]
        end
        positions = get!(group_positions, parent_id) do
            Dict{ChannelKey, Int}()
        end

        group_index = get(positions, key, 0)
        if group_index == 0
            matrices = get!(matrix_cache, key) do
                _planmatrices(table, key)
            end
            push!(
                builders,
                _CascadeGroupBuilder{N, M}(
                    matrices,
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
        if daughter_id != parent_id && haskey(decay_list.byparent, daughter_id)
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

function decayfor(key::ChannelKey, rules::Rules)
    if key.Γ > 0
        minimum_mass = max(key.ma - WIDTHK * key.Γ, 0.05)
        maximum_mass = key.ma + WIDTHK * key.Γ
        mass_distribution = BreitWigner(
            key.ma, key.Γ;
            mmin = minimum_mass,
            mmax = maximum_mass
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
    modes = table.modes
    rules = Rules(; table.orders...)
    grid_length = length(table.grid)

    # Tasks can migrate between threads, so scratch storage belongs to each task.
    scratch_for_task = OncePerTask{NTuple{N, Vector{Float64}}}(
        () -> rowscratch(modes, grid_length),
    )
    matrices_per_key = Vector{Vector{M}}(undef, length(keys_to_build))

    Threads.@threads :greedy for key_index in eachindex(keys_to_build)
        decay = decayfor(keys_to_build[key_index], rules)
        dense_matrices =
            map(_mode -> zeros(Float32, grid_length, grid_length), modes)
        # The outer key loop already uses the thread pool.
        decaymatrices!(
            dense_matrices, decay, modes,
            table.grid, table.grid; threaded = false,
            scratch = scratch_for_task()
        )

        packed_matrices = Vector{M}(undef, N)
        for mode_index in 1:N
            packed_matrices[mode_index] =
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
    required_keys, _ = neededkeys(decay_list)
    missing_keys = [key for key in required_keys if !haskey(table.W, key)]
    if isempty(missing_keys)
        return 0
    end
    return buildkeys!(table, missing_keys)
end

"Remove matrices not referenced by `decay_list`; return the orphan count."
function prune!(table::CascadeTable, decay_list::DecayList)
    required_keys, _ = neededkeys(decay_list)
    orphaned_keys = [key for key in keys(table.W) if key ∉ required_keys]
    for key in orphaned_keys
        delete!(table.W, key)
    end
    return length(orphaned_keys)
end

function _table_metadata(decay_list::DecayList)
    source_bytes = isfile(decay_list.path) ? read(decay_list.path) : UInt8[]
    return Dict{Symbol, Any}(
        :sha => decay_list.sha,
        :path => decay_list.path,
        :file => source_bytes,
        :created => string(now()),
    )
end

"""
    CascadeTable(dl; grid = range(0, 4.0, length = 134),
                 orders = DEFAULT_ORDERS, modes = [0, 2],
                 cache = nothing, rebuild = false)

Load the tables from `cache` if present and compatible (same format, grid,
orders, modes), else build and save. `orders` may be any subset of
[`DEFAULT_ORDERS`](@ref) — the rest are filled in. A decay-list hash mismatch
keeps the tables (lenient policy — keys self-invalidate through their masses)
but warns and builds any missing keys on demand.

`modes` accepts any iterable of integers and is stored as an immutable tuple,
so its length specializes the fused matrix build.

`grid` is an `AbstractRange{Float64}`. Matrix assembly trusts that range and
uses the uniform-grid interpolation kernel without validation scans.
"""
function CascadeTable(
        decay_list::DecayList;
        grid::AbstractRange{Float64} = range(0.0, 4.0, length = 134),
        orders = DEFAULT_ORDERS,
        modes = [0, 2], cache = nothing, rebuild = false
    )
    resolved_orders = merge(DEFAULT_ORDERS, orders)
    selected_modes = Tuple(collect(Int, modes))

    if cache !== nothing && !rebuild
        cached_table = loadtables(cache)
        cache_matches =
            cached_table !== nothing &&
            cached_table.grid == grid &&
            cached_table.orders == resolved_orders &&
            cached_table.modes == selected_modes

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
    table = CascadeTable(
        grid,
        resolved_orders,
        selected_modes,
        _table_metadata(decay_list),
        Dict{ChannelKey, Vector{matrix_type}}(),
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
        # Keep modes as a vector so the serialized payload is independent of N.
        serialize(
            io,
            (
                table.grid,
                table.orders,
                collect(table.modes),
                table.meta,
                table.W,
            ),
        )
    end
end

function loadtables(path::AbstractString)
    isfile(path) || return nothing

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
    stored_data isa Tuple && length(stored_data) == 5 || return nothing
    grid, orders, modes, metadata, matrices = stored_data
    grid isa AbstractRange{Float64} || return nothing
    modes isa Union{AbstractVector, Tuple} || return nothing
    all(mode -> mode isa Integer, modes) || return nothing
    metadata isa Dict{Symbol, Any} || return nothing
    normalized_modes = try
        Tuple(collect(Int, modes))
    catch
        return nothing
    end
    return _restoretable(
        grid,
        orders,
        normalized_modes,
        metadata,
        matrices,
    )
end

function _restoretable(
        grid,
        orders,
        modes::NTuple{N, Int},
        metadata,
        matrices::Dict{ChannelKey, Vector{M}},
    ) where {N, M <: RowBandMatrix{Float32}}
    matrix_type = try
        _tablematrixtype(length(grid))
    catch
        return nothing
    end
    M === matrix_type || return nothing
    expected_size = (length(grid), length(grid))
    try
        for mode_matrices in values(matrices)
            length(mode_matrices) == N || return nothing
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
        return CascadeTable(
            grid,
            orders,
            modes,
            metadata,
            matrices,
        )
    catch
        return nothing
    end
end

_restoretable(_grid, _orders, _modes, _metadata, _matrices) = nothing

# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------

"""
    runcascade(t, dl; primary, output = :stable) -> (F, Fprim)

Single feeder-before-fed sweep: every species' spectrum is complete before it
decays. Each channel adds `BR · W_n · F_parent,n` to each daughter and mode.
`primary(p::ParticleInfo, n, grid)` supplies the mode-`n` primary spectrum of
species `p` on `table.grid`, which may be an immutable range and must be treated
as read-only. Returns stable-species vectors of total and primary spectra;
index `i` corresponds to Fourier mode `table.modes[i]`. Use `output = :all`
to retain intermediate resonances for diagnostics.
"""
function runcascade(
        table::CascadeTable,
        decay_list::DecayList;
        primary,
        output = :stable,
    )
    output === :stable || output === :all ||
        throw(
        ArgumentError(
            "output must be :stable or :all, got $(repr(output))",
        )
    )
    output_ids = output === :stable ?
        decay_list.stable_ids :
        sort!(collect(keys(decay_list.parts)))

    spectra = Dict(
        particle_id => [
                Vector{Float64}(primary(particle, mode, table.grid))
                for mode in table.modes
            ]
            for (particle_id, particle) in decay_list.parts
    )
    primary_spectra = Dict{Int, Vector{Vector{Float64}}}()
    sizehint!(primary_spectra, length(output_ids))
    for particle_id in output_ids
        primary_spectra[particle_id] = [
            copy(mode_spectrum) for mode_spectrum in spectra[particle_id]
        ]
    end

    parent_order = cascadeorder(decay_list)
    eachchannelkey(decay_list; order = parent_order) do parent_id, channel, daughter_id, key
        decay_matrices = table.W[key]
        parent_spectra = spectra[parent_id]
        daughter_spectra = spectra[daughter_id]

        for mode_index in eachindex(table.modes)
            mul!(
                daughter_spectra[mode_index],
                decay_matrices[mode_index],
                parent_spectra[mode_index],
                channel.br,
                1.0,
            )
        end
    end

    if output === :all
        output_spectra = spectra
    else
        output_spectra = Dict{Int, Vector{Vector{Float64}}}()
        sizehint!(output_spectra, length(output_ids))
        for particle_id in output_ids
            output_spectra[particle_id] = spectra[particle_id]
        end
    end
    return (F = output_spectra, Fprim = primary_spectra)
end

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
                scratch = Vector{Float64}(undef, grid_length)
                _apply_plan_mode!(
                    spectra_by_mode[mode_index],
                    plan,
                    mode_index,
                    scratch,
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
    runcascade(decay_list::DecayList; primary,
               grid = range(0, 4.0, length = 134),
               orders = DEFAULT_ORDERS, modes = [0, 2],
               cache = nothing, rebuild = false,
               parallel = :serial, output = :stable) -> (F, Fprim)

Build or load a table, compile a plan, and run the cascade in one call.
`grid`, `orders`, `modes`, `cache`, and `rebuild` configure the table;
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
  * **daughter resolution** — every daughter id must be a known species.
  * **one-body rows** — counted as stable self-channels, following the decay
    list format convention.
  * **strangeness / charm change** — *informational*: zero means a strong/EM-only
    list; nonzero counts the weak (ΔS or ΔC) decays it includes.
  * **threshold** — channels kinematically closed at the parent pole mass.
"""
function validate(decay_list::DecayList; brtol = 0.02, verbose = true)
    channel_count = 0
    one_body_count = 0
    unresolved_count = 0
    invalid_branching_ratio_count = 0
    strangeness_change_count = 0
    charm_change_count = 0
    subthreshold_count = 0

    charge_violations = Tuple{Int, Vector{Int}}[]
    baryon_violations = Tuple{Int, Vector{Int}}[]
    multiplicity_counts = Dict{Int, Int}()
    branching_ratio_sums = Dict{Int, Float64}()

    for (parent_id, channels) in decay_list.byparent
        parent = decay_list.parts[parent_id]
        branching_ratio_sum = 0.0
        has_nonself_channel = false

        for channel in channels
            daughter_ids = channel.daughters
            multiplicity = length(daughter_ids)
            multiplicity_counts[multiplicity] =
                get(multiplicity_counts, multiplicity, 0) + 1

            if multiplicity == 1
                one_body_count += 1
                continue
            end

            has_nonself_channel = true
            channel_count += 1
            branching_ratio_sum += channel.br

            if !(0.0 ≤ channel.br ≤ 1 + 1.0e-6)
                invalid_branching_ratio_count += 1
            end
            if !all(
                    daughter_id -> haskey(decay_list.parts, daughter_id),
                    daughter_ids
                )
                unresolved_count += 1
                continue
            end

            daughters = (
                decay_list.parts[daughter_id] for daughter_id in daughter_ids
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
            if parent.mass + 1.0e-6 < sum(daughter -> daughter.mass, daughters)
                subthreshold_count += 1
            end
        end

        if has_nonself_channel
            branching_ratio_sums[parent_id] = branching_ratio_sum
        end
    end

    branching_ratio_deviations = [
        abs(branching_ratio_sum - 1)
            for branching_ratio_sum in values(branching_ratio_sums)
    ]
    maximum_branching_deviation = isempty(branching_ratio_deviations) ?
        0.0 : maximum(branching_ratio_deviations)
    stats = (
        particles = length(decay_list.parts),
        channels = channel_count,
        selfstable = one_body_count,
        multiplicity = sort(collect(multiplicity_counts)),
        unresolved = unresolved_count,
        br_out_of_range = invalid_branching_ratio_count,
        charge_violations = length(charge_violations),
        baryon_violations = length(baryon_violations),
        strangeness_changing = strangeness_change_count,
        charm_changing = charm_change_count,
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
    particle_name(particle_id) = haskey(decay_list.parts, particle_id) ?
        decay_list.parts[particle_id].name :
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
        "  branching ratios: max |ΣBR-1| = %.2e,  %s parents beyond %.0f%%\n",
        stats.br_maxdev, count_status(stats.br_norm_outliers),
        100 * branching_ratio_tolerance
    )
    @printf(
        "  informational: %d strangeness-changing, %d charm-changing (weak decays),  %d closed-at-pole\n",
        stats.strangeness_changing, stats.charm_changing, stats.subthreshold
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
