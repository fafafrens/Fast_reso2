# ============================================================================
# Self-contained reference cascade (single mode n = 0, matrices built inline).
#
# This is the pedagogical / dependency-light version of the resonance decay
# cascade. It builds every decay matrix on the fly with a small evicting cache
# and never touches disk. Compare with `cascade.jl`, the production version
# built on the `CascadeTables` module, which
#   * persists all matrices to disk (~2.5 GB Float32) so reruns cost seconds,
#   * builds them in parallel over unique keys rather than inline,
#   * carries Fourier modes n = 0 and n = 2 together (→ v₂ feed-down),
#   * wraps the loose dicts below in typed, hash-fingerprinted structures.
#
# Keep this file as the minimal illustration of *what the cascade is*: one
# mass-descending sweep of F[daughter] += BR · W · F[parent]. It is also the
# regression reference — its F_total/F_primary ratios must match cascade.jl.
# Run with `julia -t 8 cascade_simple.jl` for the threaded matrix builds.
# ============================================================================

# Guarded includes: safe to re-include this script in a live session — a bare
# include would define a second incarnation of each module and make their
# exported names (DecayChannel, TwoBodyDecay, …) ambiguous in Main.
isdefined(Main, :DecaySpectra) || include("DecaySpectra.jl")
isdefined(Main, :ParticleList) || include("ParticleList.jl")
using .DecaySpectra, .ParticleList
using Printf

const T = 0.145                                  # GeV
const grid = collect(range(0.0, 5.0, length = 168))   # common grid, h ≈ 0.03
const NQ, NΘ, NΜ, NΣ = 32, 8, 8, 8              # quadrature orders (~1e-4 level)

parts, byparent = read_decaylist("decays_QM2016Plus_massorder.dat")

# primary spectra (boost-invariant Boltzmann, spin degeneracy)
primary(p) = [p.degeneracy * exp(-hypot(p.mass, q) / T) for q in grid]
F = Dict(id => primary(p) for (id, p) in parts)
Fprim = Dict(id => copy(v) for (id, v) in F)

function run_cascade!(F, parts, byparent, grid)
    # Decays only go downhill in mass, so a single heavy→light sweep suffices:
    # when species X is reached, everything that can feed X has already been
    # processed and F[X] is complete (primary + all feed-down) before X decays.
    order = sort(collect(keys(parts)); by = id -> parts[id].mass, rev = true)

    # W depends only on the masses (and mode), not on charges or the source,
    # so charge/antiparticle partners share one matrix. Key layout:
    # (nbody, ma, mb, mc[, md[, me]]) — mb is the OBSERVED daughter and must keep
    # its slot (the kernel singles it out via E_b, the qT band and 1/p_b), while
    # the unobserved partners are interchangeable (they enter only through the
    # symmetric µ machinery) and are sorted to canonicalize the key.
    cache = Dict{NTuple{6,Float64},Matrix{Float64}}()
    seen = Set{NTuple{6,Float64}}()        # distinguishes rebuilds from new keys

    stats = Dict(:built => 0, :applied => 0, :closed => 0, :nd5 => 0,
                 :self => 0, :γdaughter => 0, :unknown => 0, :unique => 0)
    for pid in order
        haskey(byparent, pid) || continue
        Fp = F[pid]              # complete by construction (see ordering above)
        ma = parts[pid].mass
        for ch in byparent[pid]
            ds = ch.daughters
            nd = length(ds)
            # stable particles are encoded as a self-channel (nd = 1)
            nd == 1 && (stats[:self] += 1; continue)
            # no 5-body kernel (none in this list); counted and reported
            nd ≥ 5 && (stats[:nd5] += 1; continue)
            all(d -> haskey(parts, d), ds) || (stats[:unknown] += 1; continue)
            ms = [parts[d].mass for d in ds]
            # channels closed at nominal masses proceed only via finite widths,
            # which the sharp-mass kernel does not model; the margin also keeps
            # p_b > 0 (p_b = 0 would make the 1/p_b prefactor 0/0)
            ma > sum(ms) + 1e-6 || (stats[:closed] += 1; continue)
            # one channel feeds EVERY daughter: each slot is "observed" once,
            # with the remaining daughters as the effective pair — identical
            # daughters (e.g. π⁰π⁰) then get the correct multiplicity for free
            for (slot, did) in enumerate(ds)
                mb = ms[slot]
                # kernel needs mb > 0: photons are not tracked as daughters,
                # but stay valid as the unobserved partner (mc = 0, e.g. Σ⁰→Λγ)
                mb > 0 || (stats[:γdaughter] += 1; continue)
                rest = sort!([ms[t] for t in eachindex(ms) if t != slot])
                key = nd == 2 ? (2.0, ma, mb, rest[1], 0.0, 0.0) :
                      nd == 3 ? (3.0, ma, mb, rest[1], rest[2], 0.0) :
                                (4.0, ma, mb, rest[1], rest[2], rest[3])
                W = get!(cache, key) do
                    stats[:built] += 1
                    key in seen || (push!(seen, key); stats[:unique] += 1)
                    dec = nd == 2 ? TwoBodyDecay(ma, mb, rest[1]; nq = NQ, nθ = NΘ) :
                          nd == 3 ? ThreeBodyDecay(ma, mb, rest[1], rest[2]; nµ = NΜ, nq = NQ, nθ = NΘ) :
                                    FourBodyDecay(ma, mb, rest[1], rest[2], rest[3];
                                                  nµ = NΜ, nσ = NΣ, nq = NQ, nθ = NΘ)
                    DecayMatrix(dec, 0, grid, grid).W
                end
                # matrices are built with B = 1; the branching ratio is applied
                # here so one cached W serves channels with different BRs
                F[did] .+= ch.br .* (W * Fp)
                stats[:applied] += 1
                # crude eviction (~440 MB cap): safe because equal-mass partners
                # sit adjacent in the mass-ordered sweep, so keys are reused
                # shortly after creation; clearing only costs rebuild time
                length(cache) > 2000 && empty!(cache)
            end
        end
    end
    return stats
end

t0 = time()
stats = run_cascade!(F, parts, byparent, grid)
@printf("cascade done in %.1f s:  %d matrix builds (%d unique), %d applications, %d threads\n",
        time() - t0, stats[:built], stats[:unique], stats[:applied], Threads.nthreads())
@printf("skipped:  %d stable self-channels, %d 5-body, %d closed-at-nominal-mass, %d γ daughters, %d unknown ids\n",
        stats[:self], stats[:nd5], stats[:closed], stats[:γdaughter], stats[:unknown])

# ---------------------------------------------------------------- results
idx(pT) = argmin(abs.(grid .- pT))
println("\nFeed-down enhancement F_total/F_primary at midrapidity:")
println(" particle |  pT=0.3  |  pT=0.6  |  pT=1.0  |  pT=2.0")
for (name, id) in (("pi+", 211), ("K+", 321), ("p", 2212), ("Lambda", 3122))
    haskey(F, id) || continue
    r = [F[id][idx(p)] / Fprim[id][idx(p)] for p in (0.3, 0.6, 1.0, 2.0)]
    @printf(" %8s |  %6.2f  |  %6.2f  |  %6.2f  |  %6.2f\n", name, r...)
end
