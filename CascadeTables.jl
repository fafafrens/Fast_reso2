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
       fromparent, todaughter, fromid, toid, baryonicparent, mesonicparent,
       parentstrangeness, strangeparent, charmparent, parentisospin

const FORMAT = 5                       # bump on any layout OR matrix-value change
                                       # (2: 4-body key + nσ; 3: Chebyshev–Gauss Z-integral;
                                       #  4: parent-width key Γ + nma order;
                                       #  5: ChannelKey field reorder — Γ before mc/md/me)

# Finite parent width (BroadParent): parents broader than WIDTHCUT get the
# spectral-function mass average, truncated to ±WIDTHK·Γ; narrower ones stay
# sharp (identical to before).
const WIDTHCUT = 0.001                 # GeV
const WIDTHK   = 4.0

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
    keep::Fun     # (p::ParticleInfo, ch::DecayChannel, ds::Vector{ParticleInfo}) -> Bool
end
(f::ChannelFilter)(p, ch, ds)  = f.keep(p, ch, ds)::Bool

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
"Keep channels whose parent strangeness is one of `ss` — e.g. `parentstrangeness(1, -1)`."
parentstrangeness(ss::Integer...)    = ChannelFilter((p, ch, ds) -> p.strangeness in ss)
"Keep channels whose parent carries nonzero strangeness."
strangeparent()                      = ChannelFilter((p, ch, ds) -> p.strangeness != 0)
"Keep channels whose parent carries nonzero charm."
charmparent()                        = ChannelFilter((p, ch, ds) -> p.charm != 0)
"Keep channels whose parent isospin is one of `is` — pass physical values, e.g. `parentisospin(0.5, 1.5)`."
parentisospin(is::Real...)           = ChannelFilter((p, ch, ds) -> any(i -> round(Int, 2i) == p.twiceisospin, is))
"Wrap a raw predicate `(p, ch, ds) -> Bool` as a `ChannelFilter`."
channelfilter(f)                     = ChannelFilter(f)

# ---------------------------------------------------------------------------
# Channel key: the masses that determine a decay matrix, invariants enforced
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

struct CascadeTable
    grid::Vector{Float64}
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
# they can never disagree on which keys exist or how they are keyed. Pass
# `ordered = true` to visit parents mass-descending (the apply needs each parent
# spectrum complete before it decays). Returns the skip statistics.
function eachchannelkey(f, dl::DecayList; ordered::Bool = false)
    stats = Dict(:self => 0, :nd5 => 0, :closed => 0, :reopened => 0,
                 :γdaughter => 0, :unknown => 0)
    pids = collect(keys(dl.byparent))
    ordered && sort!(pids; by = pid -> dl.parts[pid].mass, rev = true)
    for pid in pids
        ma = dl.parts[pid].mass
        Γa = dl.parts[pid].width
        broad = Γa > WIDTHCUT
        for ch in dl.byparent[pid]
            ds = ch.daughters
            nd = length(ds)
            nd == 1 && (stats[:self] += 1; continue)          # stable self-channel
            nd ≥ 5 && (stats[:nd5] += 1; continue)            # no 5-body kernel (none in data)
            all(d -> haskey(dl.parts, d), ds) || (stats[:unknown] += 1; continue)
            ms = [dl.parts[d].mass for d in ds]
            Σm = sum(ms)
            openpole = ma > Σm + 1e-6                          # open at the pole mass
            # a broad parent's spectral tail can reopen a nominally-closed channel
            reopenable = broad && (ma + WIDTHK * Γa > Σm + 1e-6)
            openpole || reopenable || (stats[:closed] += 1; continue)
            openpole || (stats[:reopened] += 1)               # closed at pole, tail-reopened
            Γk = broad ? Γa : 0.0                             # 0 ⇒ sharp key
            for (slot, did) in enumerate(ds)
                mb = ms[slot]
                mb > 0 || (stats[:γdaughter] += 1; continue)  # photon not observable
                key = ChannelKey(ma, mb, [ms[t] for t in eachindex(ms) if t != slot], Γk)
                f(pid, ch, did, key)
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

# The decay map for one key. `k.nbody` is a runtime value, so this must branch
# (the result is a small Union that the DecayMatrix call then splits on) — a
# branch is cheaper than Val-dispatch, and this is cold build-time code.
function decayfor(k::ChannelKey; nq, nθ, nµ, nσ, nma)
    if k.Γ > 0                                    # broad parent → BroadParent mass average
        bw = BreitWigner(k.ma, k.Γ; mmin = max(k.ma - WIDTHK * k.Γ, 0.05),
                                    mmax = k.ma + WIDTHK * k.Γ)
        return k.nbody == 2 ? BroadParent(k.mb, k.mc, bw; nma, nq, nθ) :
               k.nbody == 3 ? BroadParent(k.mb, k.mc, k.md, bw; nma, nµ, nq, nθ) :
                              BroadParent(k.mb, k.mc, k.md, k.me, bw; nma, nσ, nµ, nq, nθ)
    end
    k.nbody == 2 ? TwoBodyDecay(k.ma, k.mb, k.mc; nq, nθ) :
    k.nbody == 3 ? ThreeBodyDecay(k.ma, k.mb, k.mc, k.md; nµ, nq, nθ) :
                   FourBodyDecay(k.ma, k.mb, k.mc, k.md, k.me; nµ, nσ, nq, nθ)
end

# Parallel build of the listed keys (independent work; :greedy balances the
# ~8x cost gap between 3- and 2-body). Results land in a preallocated vector,
# the Dict is filled serially afterward — no lock needed.
function buildkeys!(t::CascadeTable, keyvec::Vector{ChannelKey})
    (; nq, nθ, nµ, nσ, nma) = t.orders
    warmnodes!(nq, nθ, nµ, nσ)              # populate the node caches before threading
    out = Vector{Vector{Matrix{Float32}}}(undef, length(keyvec))
    Threads.@threads :greedy for i in eachindex(keyvec)
        dec = decayfor(keyvec[i]; nq, nθ, nµ, nσ, nma)
        # threaded=false: the key loop above already saturates the pool; a nested
        # @threads over pT rows would only oversubscribe it (much slower at 8+ threads).
        # T=Float32: the matrix is built straight into Float32 storage (accumulation
        # stays Float64 per row, rounded once), skipping a throwaway Float64 matrix.
        out[i] = decaymatrices(dec, t.modes, t.grid, t.grid; threaded = false, T = Float32)
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
                 orders = (nq = 32, nθ = 8, nµ = 8, nσ = 8, nma = 8), modes = [0, 2],
                 cache = nothing, rebuild = false)

Load the tables from `cache` if present and compatible (same format, grid,
orders, modes), else build and save. A decay-list hash mismatch keeps the
tables (lenient policy — keys self-invalidate through their masses) but
warns and builds any missing keys on demand.
"""
function CascadeTable(dl::DecayList;
                      grid = range(0.0, 4.0, length = 134),
                      orders = (nq = 32, nθ = 8, nµ = 8, nσ = 8, nma = 8),
                      modes = [0, 2], cache = nothing, rebuild = false)
    gridv = collect(Float64, grid)
    ordersnt = NamedTuple{(:nq, :nθ, :nµ, :nσ, :nma)}(    # get(): tolerate callers omitting nσ/nma
        (orders.nq, orders.nθ, orders.nµ, get(orders, :nσ, 8), get(orders, :nma, 8)))
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
    # walk the enrolled channels parent-mass-descending: F[pid] is complete
    # (primary + all feed-down) by the time pid decays. Enrollment and key layout
    # are the shared eachchannelkey, identical to the build — no drift possible.
    eachchannelkey(dl; ordered = true) do pid, ch, did, key
        Ws = t.W[key]
        Fp = F[pid]
        for m in eachindex(t.modes)
            F[did][m] .+= ch.br .* (Ws[m] * Fp[m])
        end
    end
    return (F = F, Fprim = Fprim)
end

end # module
