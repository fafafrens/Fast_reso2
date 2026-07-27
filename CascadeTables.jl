"""
    CascadeTables

Structured, persistent decay-matrix tables for the full resonance cascade.

    dl = DecayList("decays_QM2016Plus_massorder.dat")
    t  = CascadeTable(dl; cache = "Wtables.jls")     # load, or build + save
    res = runcascade(t, dl; primary = myprimaries)   # per-species, per-mode

Design (see matrix_notes.pdf and the project discussion):
  * inputs are hashed and embedded, derived data is serialized: the decay
    list is re-parsed each session (0.1 s) and its SHA256 + raw bytes travel
    in the table metadata; only the expensive matrices are persisted;
  * matrices are keyed by `ChannelKey` (the masses plus the parent width,
    observed daughter slot fixed, unobserved partners sorted), so
    charge/antiparticle partners share one entry, a changed BR invalidates
    nothing, and a changed mass or width self-invalidates by missing the key;
  * invalidation is lenient: on a decay-list hash mismatch the tables are
    kept, missing keys are built on demand, and orphans (keys no channel
    references) are reported; `prune!` removes them;
  * matrices are stored Float32 (rounding 6e-8, far below the quadrature
    setting); spectra accumulate in Float64;
  * builds parallelize over unique keys (`Threads.@threads :greedy`): measured
    per-key cost spans ~509 us (2-body sharp) to ~5.5 ms (3-body broad, the
    nma-fold spectral average), so greedy scheduling balances the ~11x spread.
"""
module CascadeTables

include("DecaySpectra.jl");  using .DecaySpectra
include("ParticleList.jl");  using .ParticleList
using Serialization, SHA, Dates, Printf
using LinearAlgebra: mul!

export DecayList, ChannelKey, CascadeTable, runcascade, prune!, validate, DEFAULT_ORDERS,
       savetables, loadtables, ParticleInfo, DecayChannel,
       isospin, spin, isbaryon, ismeson, isantiparticle, isstrange,
       ChannelFilter, channelfilter, nbody, parentmass, daughtermass, brabove,
       fromparent, todaughter, fromid, toid, baryonicparent, mesonicparent,
       parentstrangeness, strangeparent, charmparent, parentisospin

const FORMAT = 7                       # bump on any layout OR matrix-value change
                                       # (2: 4-body key + nσ; 3: Chebyshev–Gauss Z-integral;
                                       #  4: parent-width key Γ + nma order;
                                       #  5: ChannelKey field reorder — Γ before mc/md/me;
                                       #  6: parent-mass quadrature split at the channel
                                       #     threshold — affects every broad key whose BW
                                       #     window dips below it, ~70% of them;
                                       #  7: cascade grids are stored as ranges)

# Finite parent width (BroadParent): parents broader than WIDTHCUT get the
# spectral-function mass average, truncated to ±WIDTHK·Γ; narrower ones stay
# sharp (identical to before).
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
    parts, byparent = read_decaylist(path)
    DecayList(parts, byparent, String(path), bytes2hex(sha256(read(path))))
end

# ---------------------------------------------------------------------------
# Channel filtering: restrict which decays a cascade uses (the "topology" knob)
# ---------------------------------------------------------------------------
# A ChannelFilter wraps a predicate over one channel, evaluated with full
# context — parent info, the channel, and the daughter infos — so it can select
# on multiplicity, masses, names, ids, quantum numbers, or branching ratio.
# Combine with &, |, ! (defined on the type — no piracy). Apply to a DecayList
# with `filter`, giving a restricted list that flows unchanged through
# `CascadeTable` (build) and `runcascade` (apply). See the README.

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
"Keep channels whose daughter count is one of `ns` — e.g. `nbody(2)`, `nbody(2, 3)`."
nbody(ns::Integer...) =
    ChannelFilter((_parent, channel, _daughters) -> length(channel.daughters) in ns)
"Keep channels with branching ratio ≥ `x`."
brabove(x) =
    ChannelFilter((_parent, channel, _daughters) -> channel.br ≥ x)

# --- parent properties -----------------------------------------------------
"Keep channels whose parent mass lies in `[lo, hi]`."
parentmass(lo, hi) =
    ChannelFilter((parent, _channel, _daughters) -> lo ≤ parent.mass ≤ hi)
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
"Keep channels whose parent strangeness is one of `ss` — e.g. `parentstrangeness(1, -1)`."
parentstrangeness(ss::Integer...) =
    ChannelFilter((parent, _channel, _daughters) -> parent.strangeness in ss)
"Keep channels whose parent carries nonzero strangeness."
strangeparent() =
    ChannelFilter((parent, _channel, _daughters) -> isstrange(parent))
"Keep channels whose parent carries nonzero charm."
charmparent() =
    ChannelFilter((parent, _channel, _daughters) -> parent.charm != 0)
"Keep channels whose parent isospin is one of `is` — pass physical values, e.g. `parentisospin(0.5, 1.5)`."
function parentisospin(isospins::Real...)
    twice_isospins = map(isospin -> round(Int, 2isospin), isospins)
    return ChannelFilter(
        (parent, _channel, _daughters) ->
            parent.twiceisospin in twice_isospins,
    )
end

# --- daughter properties ---------------------------------------------------
"Keep channels all of whose daughter masses lie in `[lo, hi]`."
daughtermass(lo, hi) =
    ChannelFilter(
        (_parent, _channel, daughters) ->
            all(daughter -> lo ≤ daughter.mass ≤ hi, daughters),
    )
"Keep channels with at least one daughter named in `names`."
todaughter(names::AbstractString...) =
    ChannelFilter(
        (_parent, _channel, daughters) ->
            any(daughter -> daughter.name in names, daughters),
    )
"Keep channels with at least one daughter id in `ids`."
toid(ids::Integer...) =
    ChannelFilter(
        (_parent, _channel, daughters) ->
            any(daughter -> daughter.id in ids, daughters),
    )

"Wrap a raw predicate `(p, ch, ds) -> Bool` as a `ChannelFilter`."
channelfilter(predicate) = ChannelFilter(predicate)

# ---------------------------------------------------------------------------
# Channel key: the masses and parent width that determine a decay matrix,
# with the canonicalizing invariants enforced by the constructor
# ---------------------------------------------------------------------------

struct ChannelKey
    nbody::Int8
    ma::Float64
    mb::Float64          # observed daughter — slot is load-bearing (E_b, band, 1/p_b)
    Γ::Float64           # parent width: 0.0 = sharp; >0 = BroadParent mass average
    mc::Float64          # unobserved partner(s), sorted mc ≤ md ≤ me — a trailing
    md::Float64          # splat block, 0.0-padded (md=me=0 two-body, me=0 three-body)
    me::Float64
end

function ChannelKey(ma::Real, mb::Real, rest::AbstractVector{<:Real}, Γ::Real = 0.0)
    n = length(rest)
    1 ≤ n ≤ 3 || throw(ArgumentError("ChannelKey supports 2-, 3-, and 4-body only"))
    s = sort!(collect(Float64, rest))                # partners canonicalized (sort of 1 = no-op)
    partners = ntuple(i -> i ≤ n ? s[i] : 0.0, 3)    # zero-pad to fixed (mc, md, me)
    return ChannelKey(Int8(n + 1), ma, mb, Γ, partners...)   # Γ, then splat the partner block
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

# Single source of truth for the cascade's channel enrollment. For every decay
# channel of `dl` that survives the skip rules — multiplicity (2–4 body), the
# threshold (open at the pole mass, or reopened by a broad parent's spectral
# tail), and the photon-slot filter — this invokes `f(pid, ch, did, key)` once
# per observed-daughter slot, with the `ChannelKey` the matrix is stored under.
# Both the table build (`neededkeys`) and the apply (`runcascade`) walk this, so
# they can never disagree on which keys exist or how they are keyed. `order` is
# the parent visitation order (the apply needs each parent spectrum complete
# before it decays — see `cascadeorder`); the default suffices for the build,
# which only enumerates keys. Returns the skip statistics.
function eachchannelkey(f, dl::DecayList; order = keys(dl.byparent))
    stats = Dict(:self => 0, :nd5 => 0, :closed => 0, :reopened => 0,
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
                stats[:self] += 1
                continue
            end
            if multiplicity ≥ 5
                stats[:nd5] += 1
                continue
            end
            if !all(daughter_id -> haskey(dl.parts, daughter_id), daughter_ids)
                stats[:unknown] += 1
                continue
            end

            daughter_masses = [
                dl.parts[daughter_id].mass for daughter_id in daughter_ids
            ]
            threshold = sum(daughter_masses) + 1e-6
            open_at_pole = parent_mass > threshold
            open_in_tail =
                broad_parent && parent_mass + WIDTHK * parent_width > threshold

            if !open_at_pole && !open_in_tail
                stats[:closed] += 1
                continue
            end
            if !open_at_pole
                stats[:reopened] += 1
            end

            key_width = broad_parent ? parent_width : 0.0
            for (slot, daughter_id) in enumerate(daughter_ids)
                daughter_mass = daughter_masses[slot]

                if daughter_mass ≤ 0
                    stats[:γdaughter] += 1
                    continue
                end
                if daughter_id == parent_id
                    # Keeps the in-place apply free of input/output aliasing.
                    continue
                end

                partner_masses = [
                    daughter_masses[index]
                    for index in eachindex(daughter_masses) if index != slot
                ]
                key = ChannelKey(
                    parent_mass,
                    daughter_mass,
                    partner_masses,
                    key_width,
                )
                f(parent_id, channel, daughter_id, key)
            end
        end
    end
    return stats
end

# Every matrix key the cascade will read, with the skip statistics.
function neededkeys(dl::DecayList)
    need = Set{ChannelKey}()
    stats = eachchannelkey((_pid, _ch, _did, key) -> push!(need, key), dl)
    return need, stats
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

# The decay map for one key. `k.nbody` is a runtime value, so this must branch
# (the result is a small Union that the DecayMatrix call then splits on) — a
# branch is cheaper than Val-dispatch, and this is cold build-time code.
function decayfor(k::ChannelKey, r::Rules)
    if k.Γ > 0                                    # broad parent → BroadParent mass average
        bw = BreitWigner(k.ma, k.Γ; mmin = max(k.ma - WIDTHK * k.Γ, 0.05),
                                    mmax = k.ma + WIDTHK * k.Γ)
        return k.nbody == 2 ? BroadParent(k.mb, k.mc, bw, r) :
               k.nbody == 3 ? BroadParent(k.mb, k.mc, k.md, bw, r) :
                              BroadParent(k.mb, k.mc, k.md, k.me, bw, r)
    end
    k.nbody == 2 ? TwoBodyDecay(k.ma, k.mb, k.mc, r) :
    k.nbody == 3 ? ThreeBodyDecay(k.ma, k.mb, k.mc, k.md, r) :
                   FourBodyDecay(k.ma, k.mb, k.mc, k.md, k.me, r)
end

# Parallel build of the listed keys (independent work; :greedy balances the
# ~11x per-key cost spread, driven by the broad/nma-averaged keys). Results
# land in a preallocated vector,
# the Dict is filled serially afterward — no lock needed.
function buildkeys!(t::CascadeTable, keyvec::Vector{ChannelKey})
    # `Tuple(t.modes)` has a *runtime* length, so every type derived from it —
    # the output element type, the scratch type, `w` in the store loop — infers
    # abstractly (out::Vector, w::Any) even though all of them are concrete at
    # run time. Split here: the barrier below is specialized on N, which makes
    # them inferrable and removes a dynamic dispatch per key. The values are
    # unchanged; this is purely where inference gets to see the width.
    _buildkeys!(t, keyvec, Tuple(t.modes), Rules(; t.orders...))
end

# `rules`: one set for the whole build, constructed *before* any task starts.
# Rules are immutable and read-only, so every key shares them with no
# synchronization and nothing can observe a half-built one. At the default
# orders this replaces ~173k per-channel copies of five distinct rules.
function _buildkeys!(t::CascadeTable, keyvec::Vector{ChannelKey},
                     mtup::NTuple{N,Int}, rules::Rules) where {N}
    nq = length(t.grid)
    # Row accumulators, one set per *task* rather than one per key. `:greedy`
    # spawns nthreads() tasks that pull from a shared queue and exposes no task
    # index, so OncePerTask is what ties a buffer set to the task that will reuse
    # it for every key it draws — nthreads() allocations instead of one per key.
    # Per-task, not per-thread, is the safe granularity: a task keeps its own set
    # even if it migrates between threads, so no two tasks can ever alias.
    scratch = OncePerTask{NTuple{N,Vector{Float64}}}(() -> rowscratch(mtup, nq))
    # The outputs are allocated per key, up front, and filled in place. They
    # cannot be pooled the way the row buffers are: `buf` is dead once a row is
    # written, but these matrices are *retained* in t.W, so each key needs its
    # own. A pooled set would have to be copied out — which allocates exactly
    # this much and adds a memcpy (measured: 668 MB either way, no faster).
    out = [map(_ -> zeros(Float32, nq, nq), mtup) for _ in eachindex(keyvec)]
    Threads.@threads :greedy for i in eachindex(keyvec)
        dec = decayfor(keyvec[i], rules)
        # threaded=false: the key loop above already saturates the pool; a nested
        # @threads over pT rows would only oversubscribe it (much slower at 8+
        # threads). Float32 storage: accumulation stays Float64 per row and is
        # rounded once, so no throwaway Float64 matrix. Passing `mtup` rather than
        # t.modes skips the vector entry point's per-key Tuple(modes) conversion.
        decaymatrices!(out[i], dec, mtup, t.grid, t.grid;
                       threaded = false, scratch = scratch())
    end
    # Dict writes stay out of the threaded loop: Dict is not concurrent, and
    # writing it from tasks silently drops entries (measured 199034 of 200000).
    for (k, w) in zip(keyvec, out)
        t.W[k] = collect(w)
    end
    return length(keyvec)
end

# Build whatever `dl` needs that the tables lack; returns the number built.
function ensure!(t::CascadeTable, dl::DecayList)
    need, _ = neededkeys(dl)
    missing = [k for k in need if !haskey(t.W, k)]
    isempty(missing) && return 0
    return buildkeys!(t, missing)
end

"Remove matrices no channel of `dl` references; returns the orphan count."
function prune!(t::CascadeTable, dl::DecayList)
    need, _ = neededkeys(dl)
    orphans = [k for k in keys(t.W) if k ∉ need]
    foreach(k -> delete!(t.W, k), orphans)
    return length(orphans)
end

metafor(dl::DecayList) = Dict{Symbol,Any}(
    :sha => dl.sha, :path => dl.path,
    :file => isfile(dl.path) ? read(dl.path) : UInt8[],   # provenance copy (~1 MB)
    :created => string(now()))

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
function CascadeTable(dl::DecayList;
                      grid::AbstractRange{Float64} = range(0.0, 4.0, length = 134),
                      orders = DEFAULT_ORDERS,
                      modes = [0, 2], cache = nothing, rebuild = false)
    # any subset is accepted; an unknown key makes the NamedTuple the wrong shape
    # for CascadeTable.orders and fails loudly rather than being silently ignored
    ordersnt = merge(DEFAULT_ORDERS, orders)
    modesv = collect(Int, modes)
    if cache !== nothing && !rebuild
        t = loadtables(cache)
        if t !== nothing && t.grid == grid && t.orders == ordersnt && t.modes == modesv
            t.meta[:sha] == dl.sha ||
                @warn "decay list changed since tables were built; building missing keys on demand (prune! removes orphans)"
            nbuilt = ensure!(t, dl)
            nbuilt > 0 && savetables(t, cache)
            return t
        end
    end
    t = CascadeTable(grid, ordersnt, modesv, metafor(dl),
                     Dict{ChannelKey,Vector{Matrix{Float32}}}())
    ensure!(t, dl)
    cache !== nothing && savetables(t, cache)
    return t
end

# ---------------------------------------------------------------------------
# Persistence (Serialization stdlib: local cache; format is version-checked)
# ---------------------------------------------------------------------------

savetables(t::CascadeTable, path::AbstractString) =
    open(io -> serialize(io, (FORMAT, t.grid, t.orders, t.modes, t.meta, t.W)), path, "w")

function loadtables(path::AbstractString)
    isfile(path) || return nothing
    data = try
        deserialize(path)
    catch
        return nothing            # unreadable (e.g. other Julia version): rebuild
    end
    data isa Tuple && length(data) == 6 && data[1] == FORMAT || return nothing
    _, grid, orders, modes, meta, W = data
    grid isa AbstractRange{Float64} || return nothing
    return CascadeTable(grid, orders, modes, meta, W)
end

# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------

"""
    runcascade(t, dl; primary) -> (F, Fprim)

Single mass-descending sweep: every species' spectrum is complete before it
decays; each channel adds `BR · W_n · F_parent,n` into each daughter, per
mode. `primary(p::ParticleInfo, n, grid)` supplies the mode-`n` primary
spectrum of species `p` on `grid::AbstractVector{Float64}`. The grid may be
an immutable range and must be treated as read-only. Returns per-species
vectors of per-mode spectra (`F[id][m]` for mode `t.modes[m]`), total and
primary.
"""
function runcascade(t::CascadeTable, dl::DecayList; primary)
    F = Dict(id => [Vector{Float64}(primary(p, n, t.grid)) for n in t.modes]
             for (id, p) in dl.parts)
    Fprim = Dict(id => [copy(v) for v in vs] for (id, vs) in F)
    # walk the enrolled channels feeder-before-fed (cascadeorder): F[pid] is
    # complete (primary + all feed-down) by the time pid decays, even when a
    # reopened channel feeds a heavier daughter. Enrollment and key layout are the
    # shared eachchannelkey, identical to the build — no drift possible.
    eachchannelkey(dl; order = cascadeorder(dl)) do pid, ch, did, key
        Ws = t.W[key]
        Fp, Fd = F[pid], F[did]
        for m in eachindex(t.modes)
            # F[did] += br · Ws · F[pid], in place: 5-arg mul!(C, A, B, α, β) is
            # C ← α·A·B + β·C, so no per-channel GEMV temporary. did ≠ pid
            # (enforced in eachchannelkey), so C never aliases B.
            mul!(Fd[m], Ws[m], Fp[m], ch.br, 1.0)
        end
    end
    return (F = F, Fprim = Fprim)
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
    `BR ∈ [0, 1]`; outliers flag a malformed or truncated list.
  * **daughter resolution** — every daughter id must be a known species.
  * **strangeness / charm change** — *informational*: zero means a strong/EM-only
    list; nonzero counts the weak (ΔS or ΔC) decays it includes.
  * **threshold** — channels closed at the pole mass (open only via finite width).
"""
function validate(dl::DecayList; brtol = 0.02, verbose = true)
    nchan = 0; nself = 0; nunresolved = 0; brbad = 0
    schange = 0; cchange = 0; subthresh = 0
    qviol = Tuple{Int,Vector{Int}}[]; bviol = Tuple{Int,Vector{Int}}[]
    ndist = Dict{Int,Int}(); brsum = Dict{Int,Float64}()
    for (pid, chans) in dl.byparent
        p = dl.parts[pid]; s = 0.0; hasreal = false
        for ch in chans
            ds = ch.daughters; nd = length(ds)
            ndist[nd] = get(ndist, nd, 0) + 1
            nd == 1 && (nself += 1; continue)          # stable self-channel
            hasreal = true; nchan += 1; s += ch.br
            (0.0 ≤ ch.br ≤ 1 + 1e-6) || (brbad += 1)
            all(d -> haskey(dl.parts, d), ds) || (nunresolved += 1; continue)
            di = (dl.parts[d] for d in ds)
            sum(d -> d.charge,      di) == p.charge      || push!(qviol, (pid, ds))
            sum(d -> d.baryon,      di) == p.baryon      || push!(bviol, (pid, ds))
            sum(d -> d.strangeness, di) == p.strangeness || (schange += 1)
            sum(d -> d.charm,       di) == p.charm       || (cchange += 1)
            p.mass + 1e-6 < sum(d -> d.mass, di)         && (subthresh += 1)
        end
        hasreal && (brsum[pid] = s)
    end
    brdev = [abs(v - 1) for v in values(brsum)]
    stats = (particles = length(dl.parts), channels = nchan, selfstable = nself,
             multiplicity = sort(collect(ndist)),
             unresolved = nunresolved, br_out_of_range = brbad,
             charge_violations = length(qviol), baryon_violations = length(bviol),
             strangeness_changing = schange, charm_changing = cchange,
             subthreshold = subthresh,
             br_norm_outliers = count(>(brtol), brdev),
             br_maxdev = isempty(brdev) ? 0.0 : maximum(brdev))
    verbose && _report(dl, stats, qviol, bviol, brsum, brtol)
    return stats
end

function _report(dl, st, qviol, bviol, brsum, brtol)
    name(id) = haskey(dl.parts, id) ? dl.parts[id].name : "?$id"
    ok(n) = n == 0 ? "OK" : "*** $n ***"
    println("── decay-list validation ", repeat("─", 44))
    @printf("  %d particles, %d decay channels (%d stable), multiplicities %s\n",
            st.particles, st.channels, st.selfstable, string(st.multiplicity))
    println("  conservation (must be zero):")
    @printf("    charge violations ........ %s\n", ok(st.charge_violations))
    @printf("    baryon-number violations . %s\n", ok(st.baryon_violations))
    @printf("    unresolved daughters ..... %s\n", ok(st.unresolved))
    @printf("    BR out of [0,1] .......... %s\n", ok(st.br_out_of_range))
    @printf("  branching ratios: max |ΣBR-1| = %.2e,  %s parents beyond %.0f%%\n",
            st.br_maxdev, ok(st.br_norm_outliers), 100brtol)
    @printf("  informational: %d strangeness-changing, %d charm-changing (weak decays),  %d closed-at-pole\n",
            st.strangeness_changing, st.charm_changing, st.subthreshold)
    for (label, v) in (("charge", qviol), ("baryon", bviol))
        isempty(v) && continue
        println("  first $label violations:")
        for (pid, ds) in first(v, 3)
            @printf("    %s -> %s\n", name(pid), join((name(d) for d in ds), " "))
        end
    end
    worst = sort(collect(brsum); by = kv -> abs(kv[2] - 1), rev = true)
    if !isempty(worst) && abs(worst[1][2] - 1) > brtol
        println("  parents with worst ΣBR:")
        for (pid, s) in first(worst, 3)
            @printf("    %-12s ΣBR = %.3f\n", name(pid), s)
        end
    end
    println(repeat("─", 70))
end

end # module
