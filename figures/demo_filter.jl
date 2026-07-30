# Feed-down enhancement by decay topology, using channel filtering.
#
# For π⁺, K⁺, p: how much of the feed-down comes from 2-body, 2+3-body, and
# 2+3+4-body decays — three NESTED filters (2 ⊆ 23 ⊆ 234) applied to the ONE
# cached table (build once, filter freely). See the README "Channel filtering".

isdefined(Main, :CascadeTables) || include("../CascadeTables.jl")

const ROOT = dirname(@__DIR__)      # repo root: data lives there, figures here
using .CascadeTables
using Printf

const T = 0.145        # GeV

dl  = DecayList(joinpath(ROOT, "decays_QM2016Plus_massorder.dat"))
tab = CascadeTable(dl; cache = joinpath(ROOT, "Wtables.jls"))     # full table, loaded (or built once)

# mode-0 Boltzmann primaries with degeneracy (no v₂ needed for the ratio)
primary(p, n, grid) =
    n == 0 ? [p.degeneracy * exp(-hypot(p.mass, q) / T) for q in grid] :
             zeros(length(grid))

# Three nested decay-topology restrictions compile three plans backed by the
# same table. Matrix payloads are reused; only the topology-specific plan
# changes.
topologies = (("2-body",     filter(nbody(2),       dl)),
              ("2+3-body",   filter(nbody(2, 3),    dl)),
              ("2+3+4-body", filter(nbody(2, 3, 4), dl)))
plans = [(lbl, CascadePlan(tab, d)) for (lbl, d) in topologies]
results = [(lbl, runcascade(plan; primary)) for (lbl, plan) in plans]

# ---------------------------------------------------------------------------
# Plot: F_tot/F_prim vs pT, one panel per species, sequential ramp over topology
# (ordered data ⇒ single-hue light→dark; darker = more decays ⇒ higher curve)
# ---------------------------------------------------------------------------
ENV["GKSwstype"] = "100"           # headless GR
using Plots

ramp = ("#9ecae1", "#4292c6", "#08519c")     # ColorBrewer Blues: light→dark, CVD-safe
mask = findall(q -> 0.15 ≤ q ≤ 2.5, tab.grid)
pTs  = tab.grid[mask]
common = (lw = 2, gridalpha = 0.15, foreground_color_legend = nothing,
          background_color = :white, framestyle = :box, tickfontsize = 9,
          guidefontsize = 10, legendfontsize = 8, titlefontsize = 11)

panels = Any[]
for (k, (name, id)) in enumerate((("π⁺", 211), ("K⁺", 321), ("p", 2212)))
    pl = plot(; xlabel = "pT  [GeV]", ylabel = k == 1 ? "F_tot / F_prim" : "",
              title = name, legend = k == 1 ? :topright : false, common...)
    for (j, (lbl, res)) in enumerate(results)
        plot!(pl, pTs, res.F[id][1][mask] ./ res.Fprim[id][1][mask];
              label = lbl, color = ramp[j], common...)
    end
    push!(panels, pl)
end
fig = plot(panels...; layout = (1, 3), size = (1150, 380), margin = 5Plots.mm)
savefig(fig, joinpath(@__DIR__, "feeddown_topology.png"))
println("plot saved to feeddown_topology.png")

# ---------------------------------------------------------------------------
# Plot: the overall spectra F = E dN/d³p vs pT — top row log-y, bottom row
# linear-y — per species, three nested topologies + the primary (no-decay)
# reference. Log shows the shape over decades; linear shows the low-pT bulk.
# ---------------------------------------------------------------------------
speclist = (("π⁺", 211), ("K⁺", 321), ("p", 2212))
panels2 = Any[]
for (r, logy) in enumerate((true, false))
    for (c, (name, id)) in enumerate(speclist)
        first = (r == 1 && c == 1)
        pl = plot(; xlabel = "pT  [GeV]",
                  ylabel = c == 1 ? (logy ? "F = E dN/d³p  (log)" : "F  (linear)") : "",
                  yscale = logy ? :log10 : :identity,
                  title = r == 1 ? name : "",
                  legend = first ? :topright : false, common...)
        plot!(pl, pTs, results[1][2].Fprim[id][1][mask]; label = "primary",
              color = "#999999", ls = :dash, common...)          # same for all topologies
        for (j, (lbl, res)) in enumerate(results)
            plot!(pl, pTs, res.F[id][1][mask]; label = lbl, color = ramp[j], common...)
        end
        push!(panels2, pl)
    end
end
fig2 = plot(panels2...; layout = (2, 3), size = (1150, 720), margin = 5Plots.mm)
savefig(fig2, joinpath(@__DIR__, "feeddown_spectra.png"))
println("plot saved to feeddown_spectra.png")

# print the endpoint numbers so the figure has a text companion
println("\nF_tot/F_prim at pT ≈ 0.6:")
i06 = findmin(abs.(tab.grid .- 0.6))[2]
for (name, id) in (("π⁺", 211), ("K⁺", 321), ("p", 2212))
    vals = [res.F[id][1][i06] / res.Fprim[id][1][i06] for (_, res) in results]
    @printf("  %-3s  2-body %.3f   2+3 %.3f   2+3+4 %.3f\n", name, vals...)
end
