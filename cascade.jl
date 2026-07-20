# Full resonance-decay cascade via CascadeTables: persistent per-channel
# decay matrices (modes n = 0 and n = 2), lenient mass-keyed caching, single
# mass-descending sweep. First run builds and saves the tables (~2.5 GB,
# Float32); later runs load them and the cascade itself takes ~2 s.
#
# For the minimal, dependency-light illustration of the same physics (single
# mode, matrices built inline, no persistence), see `cascade_simple.jl`; its
# F_total/F_primary ratios are the regression reference for this file.

isdefined(Main, :CascadeTables) || include("CascadeTables.jl")
using .CascadeTables
using Printf

const T   = 0.145        # GeV
const v2p = 0.05         # primary elliptic-flow amplitude (all species)

dl = DecayList("decays_QM2016Plus_massorder.dat")

t0 = time()
tab = CascadeTable(dl; cache = "Wtables.jls")
@printf("tables ready in %.1f s:  %d keys × %d modes, grid %d pts [%.1f, %.1f] GeV\n",
        time() - t0, length(tab.W), length(tab.modes), length(tab.grid),
        first(tab.grid), last(tab.grid))

# primaries: Boltzmann with degeneracy; flat v₂ = v2p feeding the n = 2 mode
primary(p, n, grid) =
    n == 0 ? [p.degeneracy * exp(-hypot(p.mass, q) / T) for q in grid] :
    n == 2 ? [v2p * p.degeneracy * exp(-hypot(p.mass, q) / T) for q in grid] :
             zeros(length(grid))

t0 = time()
res = runcascade(tab, dl; primary)
@printf("cascade (both modes) in %.2f s\n\n", time() - t0)

idx(pT) = argmin(abs.(tab.grid .- pT))
println("Feed-down enhancement F_tot/F_prim and v₂ after feed-down (primary v₂ = $(v2p)):")
println(" particle |      pT=0.3      |      pT=0.6      |      pT=1.0")
println("          |  ratio |   v₂    |  ratio |   v₂    |  ratio |   v₂")
for (name, id) in (("pi+", 211), ("K+", 321), ("p", 2212), ("Lambda", 3122))
    haskey(res.F, id) || continue
    row = String[]
    for pT in (0.3, 0.6, 1.0)
        i = idx(pT)
        ratio = res.F[id][1][i] / res.Fprim[id][1][i]
        v2b = res.F[id][2][i] / res.F[id][1][i]
        push!(row, @sprintf("%6.2f | %+.4f", ratio, v2b))
    end
    @printf(" %8s | %s | %s | %s\n", name, row...)
end
