"""
    CascadeTables

Structured, persistent decay-matrix tables for the full resonance cascade.

    dl = DecayList("decays_QM2016Plus_massorder.dat")
    t  = CascadeTable(dl; cache = "Wtables.jls")     # load, or build + save
    res = runcascade(t, dl; primary = myprimaries)   # per-species, per-mode

Design (see matrix_notes.pdf for details):
  * decay matrices are serialized; source identity and bytes remain in the
    table metadata for provenance;
  * matrices are keyed by `ChannelKey` (the masses plus the parent width,
    observed daughter fixed, unobserved partners sorted), allowing physically
    equivalent channels to share an entry;
  * missing keys are built on demand and `prune!` removes unreferenced keys;
  * matrices are stored as Float32 while spectra accumulate in Float64;
  * table construction parallelizes over independent channel keys.
"""
module CascadeTables

include("DecaySpectra.jl")
include("ParticleList.jl")
using .DecaySpectra
using .ParticleList
using Serialization, SHA, Dates, Printf
using LinearAlgebra: mul!

export DecayList, ChannelKey, CascadeTable, runcascade, prune!, validate, DEFAULT_ORDERS,
       savetables, loadtables, ParticleInfo, DecayChannel,
       isospin, spin, isbaryon, ismeson, isantiparticle, isstrange,
       ChannelFilter, channelfilter, nbody, parentmass, daughtermass, brabove,
       fromparent, todaughter, fromid, toid, baryonicparent, mesonicparent,
       parentstrangeness, strangeparent, charmparent, parentisospin

# Increment when the serialized layout or generated matrix values change.
const FORMAT = 7

# Parents wider than WIDTHCUT are averaged within a window of up to ±WIDTHK·Γ.
const WIDTHCUT = 0.001                 # GeV
const WIDTHK   = 4.0

"Default quadrature orders. `CascadeTable` accepts any subset and fills in the rest."
const DEFAULT_ORDERS = (nq=32, nθ=8, nµ=16, nσ=8, nma=16)

# ---------------------------------------------------------------------------
# Decay list wrapper: parsed structures + identity of the file that made them
# ---------------------------------------------------------------------------

struct DecayList
    parts::Dict{Int,ParticleInfo}
    byparent::Dict{Int,Vector{DecayChannel}}
    path::String
    sha::String
end

function DecayList(path::AbstractString)
    particles, channels_by_parent = read_decaylist(path)
    return DecayList(particles, channels_by_parent, String(path),
                     bytes2hex(sha256(read(path))))
end

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

Restrict `dl` to the channels satisfying `f`, keeping every species (so all
daughters still resolve and appear as cascade leaves) but pruning `byparent`.
The result flows unchanged through `CascadeTable` and `runcascade`; since the
masses are unchanged, its decay-matrix keys are a subset of the full list's, so
filtering an *apply* reuses a full cached table with no rebuild.
"""
function Base.filter(selection::ChannelFilter, dl::DecayList)
    channels_by_parent = Dict{Int,Vector{DecayChannel}}()

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

    return DecayList(dl.parts, channels_by_parent, dl.path, dl.sha)
end

# --- channel properties ----------------------------------------------------
"Keep channels whose daughter count is one of `multiplicities`."
nbody(multiplicities::Integer...) =
    ChannelFilter((_parent, channel, _daughters) ->
        length(channel.daughters) in multiplicities)
"Keep channels with branching ratio ≥ `minimum_branching_ratio`."
brabove(minimum_branching_ratio) =
    ChannelFilter((_parent, channel, _daughters) ->
        channel.br ≥ minimum_branching_ratio)

# --- parent properties -----------------------------------------------------
"Keep channels whose parent mass lies in `[minimum_mass, maximum_mass]`."
parentmass(minimum_mass, maximum_mass) =
    ChannelFilter((parent, _channel, _daughters) ->
        minimum_mass ≤ parent.mass ≤ maximum_mass)
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
    ChannelFilter((parent, _channel, _daughters) ->
        parent.strangeness in strangenesses)
"Keep channels whose parent carries nonzero strangeness."
strangeparent() =
    ChannelFilter((parent, _channel, _daughters) -> isstrange(parent))
"Keep channels whose parent carries nonzero charm."
charmparent() =
    ChannelFilter((parent, _channel, _daughters) -> parent.charm != 0)
"Keep channels whose parent isospin is one of `isospins` (physical values)."
function parentisospin(isospins::Real...)
    twice_isospins = map(isospin -> round(Int, 2 * isospin), isospins)
    return ChannelFilter((parent, _channel, _daughters) ->
        parent.twiceisospin in twice_isospins)
end

# --- daughter properties ---------------------------------------------------
"Keep channels whose daughter masses all lie in `[minimum_mass, maximum_mass]`."
daughtermass(minimum_mass, maximum_mass) =
    ChannelFilter((_parent, _channel, daughters) ->
        all(daughter -> minimum_mass ≤ daughter.mass ≤ maximum_mass, daughters))
"Keep channels with at least one daughter named in `names`."
todaughter(names::AbstractString...) =
    ChannelFilter((_parent, _channel, daughters) ->
        any(daughter -> daughter.name in names, daughters))
"Keep channels with at least one daughter id in `ids`."
toid(ids::Integer...) =
    ChannelFilter((_parent, _channel, daughters) ->
        any(daughter -> daughter.id in ids, daughters))

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

function ChannelKey(parent_mass::Real, observed_mass::Real,
                    partner_masses::AbstractVector{<:Real},
                    parent_width::Real = 0.0)
    partner_count = length(partner_masses)
    1 ≤ partner_count ≤ 3 ||
        throw(ArgumentError("ChannelKey supports 2-, 3-, and 4-body only"))

    sorted_partners = sort!(collect(Float64, partner_masses))
    padded_partners =
        ntuple(index -> index ≤ partner_count ? sorted_partners[index] : 0.0, 3)
    return ChannelKey(Int8(partner_count + 1), parent_mass, observed_mass,
                      parent_width, padded_partners...)
end

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

struct CascadeTable{G<:AbstractRange{Float64}}
    grid::G
    orders::NamedTuple{(:nq, :nθ, :nµ, :nσ, :nma),Tuple{Int,Int,Int,Int,Int}}
    modes::Vector{Int}
    meta::Dict{Symbol,Any}
    W::Dict{ChannelKey,Vector{Matrix{Float32}}}   # one matrix per mode
end

"""
    eachchannelkey(visit, dl; order = keys(dl.byparent))

Call `visit(parent_id, channel, daughter_id, key)` for every enrolled
observed-daughter slot. Table construction and cascade execution share the same
multiplicity, threshold, width-tail, unresolved-daughter, nonpositive-mass, and
self-slot rules. Returns enrollment and skip statistics.
"""
function eachchannelkey(visit, dl::DecayList; order = keys(dl.byparent))
    enrollment_stats = Dict(:self => 0, :nd5 => 0, :closed => 0, :reopened => 0,
                            :γdaughter => 0, :unknown => 0)

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
            threshold = sum(daughter_masses) + 1e-6
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
graph. Among parents whose inputs are ready, process the heaviest first.

A mass-only ordering is insufficient because a broad parent's spectral tail
can reopen a channel to a heavier daughter. If the graph contains a cycle, no
exact single-sweep order exists; append all parents blocked by that cycle in
descending mass order and warn that their cross feed-down is approximate.
"""
function cascadeorder(dl::DecayList)
    decaying_parents = collect(keys(dl.byparent))
    incoming_count = Dict(parent_id => 0 for parent_id in decaying_parents)
    downstream = Dict(parent_id => Int[] for parent_id in decaying_parents)

    # An enrolled parent → daughter slot is a dependency only when the daughter
    # also decays. Stable daughters are outputs, not steps in the cascade sweep.
    eachchannelkey(dl) do parent_id, _channel, daughter_id, _key
        daughter_decays = haskey(dl.byparent, daughter_id)
        if daughter_id != parent_id && daughter_decays
            push!(downstream[parent_id], daughter_id)
            incoming_count[daughter_id] += 1
        end
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
                by = parent_id -> dl.parts[parent_id].mass,
                rev = true,
            )
            @warn "cascade feed-down graph has a cycle (mutually reopening broad \
                   parents); $(length(blocked_parents)) species ordered by mass, cross \
                   feed-down approximate" maxlog = 1
            append!(order, blocked_parents)
            break
        end

        parent_id = argmax(
            candidate -> dl.parts[candidate].mass,
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

function decayfor(key::ChannelKey, rules::Rules)
    if key.Γ > 0
        minimum_mass = max(key.ma - WIDTHK * key.Γ, 0.05)
        maximum_mass = key.ma + WIDTHK * key.Γ
        mass_distribution = BreitWigner(key.ma, key.Γ;
                                        mmin = minimum_mass,
                                        mmax = maximum_mass)
        if key.nbody == 2
            return BroadParent(key.mb, key.mc, mass_distribution, rules)
        elseif key.nbody == 3
            return BroadParent(key.mb, key.mc, key.md, mass_distribution, rules)
        else
            return BroadParent(key.mb, key.mc, key.md, key.me,
                               mass_distribution, rules)
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

# Convert the runtime mode vector to a tuple before entering the threaded build,
# allowing `_buildkeys!` to specialize on the number of modes.
function buildkeys!(table::CascadeTable, keys_to_build::Vector{ChannelKey})
    mode_tuple = Tuple(table.modes)
    rules = Rules(; table.orders...)
    return _buildkeys!(table, keys_to_build, mode_tuple, rules)
end

function _buildkeys!(
    table::CascadeTable,
    keys_to_build::Vector{ChannelKey},
    mode_tuple::NTuple{N,Int},
    rules::Rules,
) where {N}
    grid_length = length(table.grid)

    # Tasks can migrate between threads, so scratch storage belongs to each task.
    scratch_for_task = OncePerTask{NTuple{N,Vector{Float64}}}(
        () -> rowscratch(mode_tuple, grid_length),
    )
    matrices_per_key = [
        map(_mode -> zeros(Float32, grid_length, grid_length), mode_tuple)
        for _ in eachindex(keys_to_build)
    ]

    Threads.@threads :greedy for key_index in eachindex(keys_to_build)
        decay = decayfor(keys_to_build[key_index], rules)
        # The outer key loop already uses the thread pool.
        decaymatrices!(matrices_per_key[key_index], decay, mode_tuple,
                       table.grid, table.grid; threaded = false,
                       scratch = scratch_for_task())
    end

    # Dict mutation is kept serial because Dict is not thread-safe.
    for (key, mode_matrices) in zip(keys_to_build, matrices_per_key)
        table.W[key] = collect(mode_matrices)
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
    return Dict{Symbol,Any}(
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

`grid` is an `AbstractRange{Float64}`. Matrix assembly trusts that range and
uses the uniform-grid interpolation kernel without validation scans.
"""
function CascadeTable(decay_list::DecayList;
                      grid::AbstractRange{Float64} = range(0.0, 4.0, length = 134),
                      orders = DEFAULT_ORDERS,
                      modes = [0, 2], cache = nothing, rebuild = false)
    resolved_orders = merge(DEFAULT_ORDERS, orders)
    selected_modes = collect(Int, modes)

    if cache !== nothing && !rebuild
        cached_table = loadtables(cache)
        cache_matches =
            cached_table !== nothing &&
            cached_table.grid == grid &&
            cached_table.orders == resolved_orders &&
            cached_table.modes == selected_modes

        if cache_matches
            if cached_table.meta[:sha] != decay_list.sha
                @warn "decay list changed since tables were built; building missing keys on demand (prune! removes orphans)"
            end
            built_count = ensure!(cached_table, decay_list)
            if built_count > 0
                savetables(cached_table, cache)
            end
            return cached_table
        end
    end

    table = CascadeTable(
        grid,
        resolved_orders,
        selected_modes,
        _table_metadata(decay_list),
        Dict{ChannelKey,Vector{Matrix{Float32}}}(),
    )
    ensure!(table, decay_list)
    if cache !== nothing
        savetables(table, cache)
    end
    return table
end

# ---------------------------------------------------------------------------
# Persistence (Serialization stdlib: local cache; format is version-checked)
# ---------------------------------------------------------------------------

function savetables(table::CascadeTable, path::AbstractString)
    stored_data = (FORMAT, table.grid, table.orders, table.modes,
                   table.meta, table.W)
    return open(path, "w") do io
        serialize(io, stored_data)
    end
end

function loadtables(path::AbstractString)
    isfile(path) || return nothing

    stored_data = try
        deserialize(path)
    catch
        return nothing
    end
    valid_format =
        stored_data isa Tuple &&
        length(stored_data) == 6 &&
        stored_data[1] == FORMAT
    valid_format || return nothing

    _, grid, orders, modes, metadata, matrices = stored_data
    grid isa AbstractRange{Float64} || return nothing
    return CascadeTable(grid, orders, modes, metadata, matrices)
end

# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------

"""
    runcascade(t, dl; primary) -> (F, Fprim)

Single feeder-before-fed sweep: every species' spectrum is complete before it
decays. Each channel adds `BR · W_n · F_parent,n` to each daughter and mode.
`primary(p::ParticleInfo, n, grid)` supplies the mode-`n` primary spectrum of
species `p` on `table.grid`, which may be an immutable range and must be treated
as read-only. Returns per-species vectors of total and primary spectra;
index `i` corresponds to Fourier mode `table.modes[i]`.
"""
function runcascade(table::CascadeTable, decay_list::DecayList; primary)
    spectra = Dict(
        particle_id => [
            Vector{Float64}(primary(particle, mode, table.grid))
            for mode in table.modes
        ]
        for (particle_id, particle) in decay_list.parts
    )
    primary_spectra = Dict(
        particle_id => [copy(mode_spectrum) for mode_spectrum in mode_spectra]
        for (particle_id, mode_spectra) in spectra
    )

    parent_order = cascadeorder(decay_list)
    eachchannelkey(decay_list; order = parent_order) do parent_id, channel, daughter_id, key
        decay_matrices = table.W[key]
        parent_spectra = spectra[parent_id]
        daughter_spectra = spectra[daughter_id]

        for mode_index in eachindex(table.modes)
            # Five-argument mul! accumulates without a temporary product.
            mul!(daughter_spectra[mode_index], decay_matrices[mode_index],
                 parent_spectra[mode_index], channel.br, 1.0)
        end
    end

    return (F = spectra, Fprim = primary_spectra)
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

    charge_violations = Tuple{Int,Vector{Int}}[]
    baryon_violations = Tuple{Int,Vector{Int}}[]
    multiplicity_counts = Dict{Int,Int}()
    branching_ratio_sums = Dict{Int,Float64}()

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

            if !(0.0 ≤ channel.br ≤ 1 + 1e-6)
                invalid_branching_ratio_count += 1
            end
            if !all(daughter_id -> haskey(decay_list.parts, daughter_id),
                    daughter_ids)
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
            if parent.mass + 1e-6 < sum(daughter -> daughter.mass, daughters)
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
        _print_validation_report(decay_list, stats, charge_violations,
                                 baryon_violations, branching_ratio_sums, brtol)
    end
    return stats
end

function _print_validation_report(decay_list, stats, charge_violations,
                                  baryon_violations, branching_ratio_sums,
                                  branching_ratio_tolerance)
    particle_name(particle_id) = haskey(decay_list.parts, particle_id) ?
                                 decay_list.parts[particle_id].name :
                                 "?$particle_id"
    count_status(count) = count == 0 ? "OK" : "*** $count ***"

    println("── decay-list validation ", repeat("─", 44))
    @printf("  %d particles, %d decay channels (%d stable), multiplicities %s\n",
            stats.particles, stats.channels, stats.selfstable, string(stats.multiplicity))
    println("  conservation (must be zero):")
    @printf("    charge violations ........ %s\n",
            count_status(stats.charge_violations))
    @printf("    baryon-number violations . %s\n",
            count_status(stats.baryon_violations))
    @printf("    unresolved daughters ..... %s\n",
            count_status(stats.unresolved))
    @printf("    BR out of [0,1] .......... %s\n",
            count_status(stats.br_out_of_range))
    @printf("  branching ratios: max |ΣBR-1| = %.2e,  %s parents beyond %.0f%%\n",
            stats.br_maxdev, count_status(stats.br_norm_outliers),
            100 * branching_ratio_tolerance)
    @printf("  informational: %d strangeness-changing, %d charm-changing (weak decays),  %d closed-at-pole\n",
            stats.strangeness_changing, stats.charm_changing, stats.subthreshold)

    for (label, violations) in (("charge", charge_violations),
                                ("baryon", baryon_violations))
        if isempty(violations)
            continue
        end
        println("  first $label violations:")
        for (parent_id, daughter_ids) in first(violations, 3)
            daughter_names =
                (particle_name(daughter_id) for daughter_id in daughter_ids)
            @printf("    %s -> %s\n",
                    particle_name(parent_id), join(daughter_names, " "))
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
            @printf("    %-12s ΣBR = %.3f\n",
                    particle_name(parent_id), branching_ratio_sum)
        end
    end
    println(repeat("─", 70))
end

end # module
