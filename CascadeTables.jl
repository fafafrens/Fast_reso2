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
  * matrices are keyed by `ChannelKey` (masses only, observed daughter slot
    fixed, unobserved partners sorted), so charge/antiparticle partners
    share one entry, a changed BR invalidates nothing, and a changed mass
    self-invalidates by missing the key;
  * invalidation is lenient: on a decay-list hash mismatch the tables are
    kept, missing keys are built on demand, and orphans (keys no channel
    references) are reported; `prune!` removes them;
  * matrices are stored Float32 (rounding 6e-8, far below the quadrature
    setting); spectra accumulate in Float64;
  * builds parallelize over unique keys (`Threads.@threads :greedy` —
    three-body keys cost ~8x two-body, so greedy scheduling balances).
"""
module CascadeTables

include("DecaySpectra.jl");  using .DecaySpectra
include("ParticleList.jl");  using .ParticleList
using Serialization, SHA, Dates

export DecayList, ChannelKey, CascadeTable, runcascade, prune!,
       savetables, loadtables, ParticleInfo, DecayChannel,
       ChannelFilter, channelfilter, nbody, parentmass, daughtermass, brabove,
       fromparent, todaughter, fromid, toid, baryonicparent, mesonicparent

const FORMAT = 2                       # bump on any layout change (2: 4-body key + nσ)

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

struct ChannelFilter
    keep::Function     # (p::ParticleInfo, ch::DecayChannel, ds::Vector{ParticleInfo}) -> Bool
end
(f::ChannelFilter)(p, ch, ds) = f.keep(p, ch, ds)::Bool

Base.:&(a::ChannelFilter, b::ChannelFilter) = ChannelFilter((p, ch, ds) -> a(p, ch, ds) && b(p, ch, ds))
Base.:|(a::ChannelFilter, b::ChannelFilter) = ChannelFilter((p, ch, ds) -> a(p, ch, ds) || b(p, ch, ds))
Base.:!(a::ChannelFilter)                   = ChannelFilter((p, ch, ds) -> !a(p, ch, ds))

"""
    filter(f::ChannelFilter, dl::DecayList) -> DecayList

Restrict `dl` to the channels satisfying `f`, keeping every species (so all
daughters still resolve and appear as cascade leaves) but pruning `byparent`.
The result flows unchanged through `CascadeTable` and `runcascade`; since the
masses are unchanged, its decay-matrix keys are a subset of the full list's, so
filtering an *apply* reuses a full cached table with no rebuild.
"""
function Base.filter(f::ChannelFilter, dl::DecayList)
    bp = Dict{Int,Vector{DecayChannel}}()
    for (pid, chans) in dl.byparent
        p = dl.parts[pid]
        kept = [ch for ch in chans
                if all(d -> haskey(dl.parts, d), ch.daughters) &&
                   f(p, ch, ParticleInfo[dl.parts[d] for d in ch.daughters])]
        isempty(kept) || (bp[pid] = kept)
    end
    return DecayList(dl.parts, bp, dl.path, dl.sha)
end

# --- ready-made filters (each returns a ChannelFilter) ---------------------
"Keep channels whose daughter count is one of `ns` — e.g. `nbody(2)`, `nbody(2, 3)`."
nbody(ns::Integer...)                = ChannelFilter((p, ch, ds) -> length(ch.daughters) in ns)
"Keep channels whose parent mass lies in `[lo, hi]`."
parentmass(lo, hi)                   = ChannelFilter((p, ch, ds) -> lo ≤ p.mass ≤ hi)
"Keep channels all of whose daughter masses lie in `[lo, hi]`."
daughtermass(lo, hi)                 = ChannelFilter((p, ch, ds) -> all(d -> lo ≤ d.mass ≤ hi, ds))
"Keep channels with branching ratio ≥ `x`."
brabove(x)                           = ChannelFilter((p, ch, ds) -> ch.br ≥ x)
"Keep channels whose parent name is one of `names`."
fromparent(names::AbstractString...) = ChannelFilter((p, ch, ds) -> p.name in names)
"Keep channels with at least one daughter named in `names`."
todaughter(names::AbstractString...) = ChannelFilter((p, ch, ds) -> any(d -> d.name in names, ds))
"Keep channels whose parent id is one of `ids`."
fromid(ids::Integer...)              = ChannelFilter((p, ch, ds) -> p.id in ids)
"Keep channels with at least one daughter id in `ids`."
toid(ids::Integer...)                = ChannelFilter((p, ch, ds) -> any(d -> d.id in ids, ds))
"Keep channels whose parent is a baryon (nonzero baryon number)."
baryonicparent()                     = ChannelFilter((p, ch, ds) -> p.baryon != 0)
"Keep channels whose parent is a meson (zero baryon number)."
mesonicparent()                      = ChannelFilter((p, ch, ds) -> p.baryon == 0)
"Wrap a raw predicate `(p, ch, ds) -> Bool` as a `ChannelFilter`."
channelfilter(f)                     = ChannelFilter(f)

# ---------------------------------------------------------------------------
# Channel key: the masses that determine a decay matrix, invariants enforced
# ---------------------------------------------------------------------------

struct ChannelKey
    nbody::Int8
    ma::Float64
    mb::Float64          # observed daughter — slot is load-bearing (E_b, band, 1/p_b)
    mc::Float64          # unobserved partner(s), sorted mc ≤ md ≤ me
    md::Float64          # 0.0 for two-body
    me::Float64          # 0.0 for two-/three-body
end

function ChannelKey(ma::Real, mb::Real, rest::AbstractVector{<:Real})
    if length(rest) == 1
        return ChannelKey(Int8(2), ma, mb, rest[1], 0.0, 0.0)
    elseif length(rest) == 2
        lo, hi = minmax(rest[1], rest[2])
        return ChannelKey(Int8(3), ma, mb, lo, hi, 0.0)
    elseif length(rest) == 3
        s = sort(collect(Float64, rest))
        return ChannelKey(Int8(4), ma, mb, s[1], s[2], s[3])
    end
    throw(ArgumentError("ChannelKey supports 2-, 3-, and 4-body only"))
end

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

struct CascadeTable
    grid::Vector{Float64}
    orders::NamedTuple{(:nq, :nθ, :nµ, :nσ),Tuple{Int,Int,Int,Int}}
    modes::Vector{Int}
    meta::Dict{Symbol,Any}
    W::Dict{ChannelKey,Vector{Matrix{Float32}}}   # one matrix per mode
end

# The skip rules of the cascade, applied once to enumerate every needed key.
# Returns the key set and the skip statistics.
function neededkeys(dl::DecayList)
    need = Set{ChannelKey}()
    stats = Dict(:self => 0, :nd5 => 0, :closed => 0, :γdaughter => 0, :unknown => 0)
    for (pid, chans) in dl.byparent
        ma = dl.parts[pid].mass
        for ch in chans
            ds = ch.daughters
            nd = length(ds)
            nd == 1 && (stats[:self] += 1; continue)          # stable self-channel
            nd ≥ 5 && (stats[:nd5] += 1; continue)            # no 5-body kernel (none in data)
            all(d -> haskey(dl.parts, d), ds) || (stats[:unknown] += 1; continue)
            ms = [dl.parts[d].mass for d in ds]
            # closed at nominal masses (finite-width only) + p_b > 0 guard
            ma > sum(ms) + 1e-6 || (stats[:closed] += 1; continue)
            for (slot, _) in enumerate(ds)
                mb = ms[slot]
                mb > 0 || (stats[:γdaughter] += 1; continue)  # photon not observable
                push!(need, ChannelKey(ma, mb, [ms[t] for t in eachindex(ms) if t != slot]))
            end
        end
    end
    return need, stats
end

# Parallel build of the listed keys (independent work; :greedy balances the
# ~8x cost gap between 3- and 2-body). Results land in a preallocated vector,
# the Dict is filled serially afterward — no lock needed.
function buildkeys!(t::CascadeTable, keyvec::Vector{ChannelKey})
    (; nq, nθ, nµ, nσ) = t.orders
    out = Vector{Vector{Matrix{Float32}}}(undef, length(keyvec))
    Threads.@threads :greedy for i in eachindex(keyvec)
        k = keyvec[i]
        dec = k.nbody == 2 ? TwoBodyDecay(k.ma, k.mb, k.mc; nq, nθ) :
              k.nbody == 3 ? ThreeBodyDecay(k.ma, k.mb, k.mc, k.md; nµ, nq, nθ) :
                             FourBodyDecay(k.ma, k.mb, k.mc, k.md, k.me; nµ, nσ, nq, nθ)
        out[i] = [Float32.(DecayMatrix(dec, n, t.grid, t.grid).W) for n in t.modes]
    end
    for (k, w) in zip(keyvec, out)
        t.W[k] = w
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
                 orders = (nq = 32, nθ = 8, nµ = 8, nσ = 8), modes = [0, 2],
                 cache = nothing, rebuild = false)

Load the tables from `cache` if present and compatible (same format, grid,
orders, modes), else build and save. A decay-list hash mismatch keeps the
tables (lenient policy — keys self-invalidate through their masses) but
warns and builds any missing keys on demand.
"""
function CascadeTable(dl::DecayList;
                      grid = range(0.0, 4.0, length = 134),
                      orders = (nq = 32, nθ = 8, nµ = 8, nσ = 8),
                      modes = [0, 2], cache = nothing, rebuild = false)
    gridv = collect(Float64, grid)
    ordersnt = NamedTuple{(:nq, :nθ, :nµ, :nσ)}(          # get(): tolerate callers omitting nσ
        (orders.nq, orders.nθ, orders.nµ, get(orders, :nσ, 8)))
    modesv = collect(Int, modes)
    if cache !== nothing && !rebuild
        t = loadtables(cache)
        if t !== nothing && t.grid == gridv && t.orders == ordersnt && t.modes == modesv
            t.meta[:sha] == dl.sha ||
                @warn "decay list changed since tables were built; building missing keys on demand (prune! removes orphans)"
            nbuilt = ensure!(t, dl)
            nbuilt > 0 && savetables(t, cache)
            return t
        end
    end
    t = CascadeTable(gridv, ordersnt, modesv, metafor(dl),
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
spectrum of species `p` on the grid. Returns per-species vectors of
per-mode spectra (`F[id][m]` for mode `t.modes[m]`), total and primary.
"""
function runcascade(t::CascadeTable, dl::DecayList; primary)
    F = Dict(id => [Vector{Float64}(primary(p, n, t.grid)) for n in t.modes]
             for (id, p) in dl.parts)
    Fprim = Dict(id => [copy(v) for v in vs] for (id, vs) in F)
    order = sort(collect(keys(dl.parts)); by = id -> dl.parts[id].mass, rev = true)
    for pid in order
        haskey(dl.byparent, pid) || continue
        ma = dl.parts[pid].mass
        Fp = F[pid]
        for ch in dl.byparent[pid]
            ds = ch.daughters
            nd = length(ds)
            (nd == 1 || nd ≥ 5) && continue
            all(d -> haskey(dl.parts, d), ds) || continue
            ms = [dl.parts[d].mass for d in ds]
            ma > sum(ms) + 1e-6 || continue
            for (slot, did) in enumerate(ds)
                mb = ms[slot]
                mb > 0 || continue
                key = ChannelKey(ma, mb, [ms[x] for x in eachindex(ms) if x != slot])
                Ws = t.W[key]
                for m in eachindex(t.modes)
                    F[did][m] .+= ch.br .* (Ws[m] * Fp[m])
                end
            end
        end
    end
    return (F = F, Fprim = Fprim)
end

end # module
