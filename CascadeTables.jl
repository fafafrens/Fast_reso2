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

include("RowBandMatrix.jl")
using ..DecaySpectra
using ..ParticleList
using ..DecayModel
import ..DecayModel: allparticles, stableparticles
using Serialization, Dates
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
    isbottom,
    ChannelFilter, channelfilter, nbody, parentmass, daughtermass, brabove,
    fromparent, todaughter, fromid, toid, baryonicparent, mesonicparent,
    parentstrangeness, strangeparent, charmparent, parentisospin,
    feeddown_order

# Increment when the serialized layout or generated matrix values change.
const FORMAT = 12
const CACHE_MAGIC = "FastReso.CascadeTables\n"

# ---------------------------------------------------------------------------
# Channel key: the masses and parent width that determine a decay matrix,
# with the canonicalizing invariants enforced by the constructor
# ---------------------------------------------------------------------------

struct _ValidatedChannelKeyToken end
const _VALIDATED_CHANNEL_KEY = _ValidatedChannelKeyToken()

struct ChannelKey
    # Field order is serialized; changing it requires a FORMAT bump.
    # Scalars are deliberately canonical Float64 cache metadata. A generic
    # precision policy would need to parameterize the full particle/table
    # pipeline, not only its dictionary key.
    nbody::Int8
    ma::Float64          # parent mass
    mb::Float64          # observed daughter; never sorted with partners
    Γ::Float64           # parent width; zero selects the sharp path
    mth::Float64         # species decay threshold; zero on sharp keys, where
                         # no spectral shape is built and sharing stays maximal
    mc::Float64          # partner masses, sorted and zero-padded
    md::Float64
    me::Float64

    function ChannelKey(
            nbody::Int8,
            ma::Float64,
            mb::Float64,
            Γ::Float64,
            mth::Float64,
            mc::Float64,
            md::Float64,
            me::Float64,
            ::_ValidatedChannelKeyToken,
        )
        return new(nbody, ma, mb, Γ, mth, mc, md, me)
    end
end

function ChannelKey(
        nbody::Integer,
        parent_mass::Real,
        observed_mass::Real,
        parent_width::Real,
        species_threshold::Real,
        mc::Real,
        md::Real,
        me::Real,
    )
    nbody in 2:4 || throw(ArgumentError(
        "ChannelKey supports 2-, 3-, and 4-body only",
    ))
    values = map(
        Float64,
        (
            parent_mass,
            observed_mass,
            parent_width,
            species_threshold,
            mc,
            md,
            me,
        ),
    )
    all(isfinite, values) || throw(ArgumentError(
        "ChannelKey masses, width, and threshold must be finite",
    ))
    all(>=(0.0), values) || throw(ArgumentError(
        "ChannelKey masses, width, and threshold must be nonnegative",
    ))

    ma, mb, Γ, mth, canonical_mc, canonical_md, canonical_me = values
    Γ == 0.0 && mth != 0.0 && throw(ArgumentError(
        "a zero-width ChannelKey must have a zero threshold",
    ))
    # Keep validation on the fixed fields. Slicing the tuple with the runtime
    # `nbody` produces `Tuple{Vararg{Float64}}` in typed IR, even though this
    # constructor's final return type remains `ChannelKey`.
    if nbody == 2
        iszero(canonical_md) && iszero(canonical_me) || throw(ArgumentError(
            "ChannelKey unused partner slots must be zero-padded",
        ))
    elseif nbody == 3
        !isless(canonical_md, canonical_mc) || throw(ArgumentError(
            "ChannelKey partner masses must be sorted",
        ))
        iszero(canonical_me) || throw(ArgumentError(
            "ChannelKey unused partner slots must be zero-padded",
        ))
    else
        !isless(canonical_md, canonical_mc) &&
            !isless(canonical_me, canonical_md) || throw(ArgumentError(
            "ChannelKey partner masses must be sorted",
        ))
    end

    return ChannelKey(
        Int8(nbody), ma, mb, Γ, mth,
        canonical_mc, canonical_md, canonical_me,
        _VALIDATED_CHANNEL_KEY,
    )
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

    first_index = firstindex(partner_masses)
    first_mass = Float64(partner_masses[first_index])
    if partner_count == 1
        return ChannelKey(
            2, parent_mass, observed_mass, parent_width, species_threshold,
            first_mass, 0.0, 0.0,
        )
    end

    second_index = nextind(partner_masses, first_index)
    second_mass = Float64(partner_masses[second_index])
    if partner_count == 2
        mc, md = sort((
            first_mass,
            second_mass,
        ))
        return ChannelKey(
            3, parent_mass, observed_mass, parent_width, species_threshold,
            mc, md, 0.0,
        )
    end

    third_index = nextind(partner_masses, second_index)
    third_mass = Float64(partner_masses[third_index])
    mc, md, me = sort((first_mass, second_mass, third_mass))
    return ChannelKey(
        4, parent_mass, observed_mass, parent_width, species_threshold,
        mc, md, me,
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

        # The species threshold is resolved before topology filtering, so a
        # filtered plan remains a key-subset of a table built from the full
        # model. Each channel's own closure is still handled below.
        species_threshold = broad_parent ? get(dl.thresholds, parent_id, 0.0) : 0.0

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

_cascadeorder(decay_list::DecayList, dependencies) =
    feeddown_order(decay_list; dependencies = dependencies, cycles = :warn)

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

end # module
