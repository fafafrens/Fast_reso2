# ============================================================================
# Physics deliverable: resonance-decay feed-down on spectra and elliptic flow.
#
# Boltzmann primaries with a pT-dependent *input* v₂ (the same smooth ansatz for
# every primary species). The full mass-ordered cascade then redistributes yield
# and v₂ into the stable species π/K/p/Λ. Because the input v₂ is identical for
# all species, any species-dependence in the TOTAL v₂ is a genuine feed-down
# effect — not a flow input. (This source has no radial-flow blast wave, so it is
# not meant to reproduce experimental mass ordering; it isolates what the DECAYS
# do to a given input.)
#
# Produces feeddown_physics.png: (left) invariant spectra, primary vs total;
# (right) v₂(pT), input vs total.
# Run: julia --project -t auto figures/physics_feeddown.jl
# ============================================================================

const ROOT = dirname(@__DIR__)      # repo root: data lives there, figures here
using FastReso2
using Printf

const T = 0.145                                   # GeV
v2in(pT) = 0.20 * pT / (pT + 1.5)                 # input v₂: rises, saturates ~0.2

primary(p, n, grid) =
    n == 0 ? [p.degeneracy * exp(-hypot(p.mass, q) / T) for q in grid] :
    n == 2 ? [v2in(q) * p.degeneracy * exp(-hypot(p.mass, q) / T) for q in grid] :
             zeros(length(grid))

const DATA = joinpath(ROOT, "data")
dl = DecayList(
    readparticles(joinpath(DATA, "particles_QM2016Plus.dat")),
    readdecays(joinpath(DATA, "decays_QM2016Plus.dat")),
)
t0  = time()
plan = CascadePlan(dl; cache = nothing)            # build in memory, without a cache file
tab = plan.table
@printf("table + plan ready in %.1f s:  %d keys × %d modes\n", time() - t0,
        length(tab.W), length(tab.modes))
res = runcascade(plan; primary)

grid = tab.grid
idx(pT) = argmin(abs.(grid .- pT))
species = (("π⁺", 211, "#2a78d6"), ("K⁺", 321, "#008300"),
           ("p", 2212, "#d55181"), ("Λ", 3122, "#e08214"))

# ---- summary table --------------------------------------------------------
println("\n feed-down:  yield enhancement F_tot/F_prim,  and v₂ (input → total)")
@printf(" %-4s |  %-21s |  %-21s\n", "", "pT = 0.5 GeV", "pT = 1.5 GeV")
@printf(" %-4s | %6s %7s %7s | %6s %7s %7s\n", "sp", "enh", "v2in", "v2tot", "enh", "v2in", "v2tot")
for (nm, id, _) in species
    haskey(res.F, id) || continue
    row = String[]
    for pT in (0.5, 1.5)
        i = idx(pT)
        enh = res.F[id][1][i] / res.Fprim[id][1][i]
        v2t = res.F[id][2][i] / res.F[id][1][i]
        push!(row, @sprintf("%6.2f %7.4f %7.4f", enh, v2in(pT), v2t))
    end
    @printf(" %-4s | %s | %s\n", nm, row...)
end

# ---- figure ---------------------------------------------------------------
ENV["GKSwstype"] = "100"                           # headless GR
using Plots
mask = findall(q -> 0.15 ≤ q ≤ 3.0, grid); pTs = grid[mask]
common = (lw = 2, gridalpha = 0.15, foreground_color_legend = nothing,
          background_color = :white, framestyle = :box, tickfontsize = 9,
          guidefontsize = 10, legendfontsize = 8, titlefontsize = 10)

pspec = plot(; xlabel = "pT  [GeV]", ylabel = "E dN/d³p   (arb.)", yscale = :log10,
             title = "spectra:  primary (dashed) vs total", legend = :topright, common...)
pv2 = plot(; xlabel = "pT  [GeV]", ylabel = "v₂",
           title = "elliptic flow:  input (dashed) vs total", legend = :topleft, common...)
psup = plot(; xlabel = "pT  [GeV]", ylabel = "v₂ total / v₂ input",
            title = "feed-down v₂ suppression", legend = :bottomright, common...)
for (nm, id, col) in species
    haskey(res.F, id) || continue
    plot!(pspec, pTs, res.Fprim[id][1][mask]; color = col, ls = :dash, label = "", common...)
    plot!(pspec, pTs, res.F[id][1][mask];     color = col, label = nm, common...)
    v2tot = res.F[id][2][mask] ./ res.F[id][1][mask]
    plot!(pv2, pTs, v2tot; color = col, label = nm, common...)
    plot!(psup, pTs, v2tot ./ v2in.(pTs); color = col, label = nm, common...)
end
plot!(pv2, pTs, v2in.(pTs); color = :gray, ls = :dash, label = "input v₂", common...)
hline!(psup, [1.0]; color = :gray, ls = :dash, label = "no feed-down", common...)

fig = plot(pspec, pv2, psup; layout = (1, 3), size = (1360, 400), margin = 5Plots.mm)
savefig(fig, joinpath(@__DIR__, "feeddown_physics.png"))
println("\nsaved feeddown_physics.png")
