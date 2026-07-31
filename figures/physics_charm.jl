# ============================================================================
# Charm feed-down: open-charm hadrons cascade to the ground-state D mesons and Λc.
# Statistical-hadronization model: thermal Boltzmann primaries (deg·e^{-m_T/T})
# enhanced by a charm fugacity g_c — thermal *shape*, hard-production *yield*.
# Only open-charm states (charm ≠ 0) carry a primary, so each D⁰/D⁺/D_s/Λc total
# is direct production PLUS strong feed-down from excited charm (D*, D₁, D₂*, Σc,
# …). g_c cancels in the feed-down ratio F_tot/F_direct.
#
# Needs the charm+bottom supplement.  Run: julia -t auto physics_charm.jl
# ============================================================================

isdefined(Main, :CascadeTables) || include("../CascadeTables.jl")

const ROOT = dirname(@__DIR__)      # repo root: data lives there, figures here
using .CascadeTables
using Printf

# Merge the main list (light daughters) with the charm+bottom supplement.
# Supplement channels may refer directly to particle IDs from the base list.
dl = merge(
    DecayList(joinpath(ROOT, "decays_QM2016Plus_massorder.dat")),
    DecayList(joinpath(ROOT, "decays_charm_bottom.dat")),
)
dlc = filter(charmparent(), dl)                   # only charm-parent decays matter

const T  = 0.145                                  # freeze-out temperature
const gc = 30.0                                   # charm enhancement (fugacity) above thermal
# statistical hadronization of charm: thermal Boltzmann shape, enhanced by the
# charm fugacity g_c (one power per charm quark) so the yield matches hard
# production rather than the tiny thermal e^{-m_c/T}. g_c cancels in F/F_direct.
therm(m, pT) = exp(-hypot(m, pT) / T)
v2c(pT) = 0.08 * pT / (pT + 2.0)                  # modest charm input v₂
primary(p, n, grid) =
    p.charm == 0 ? zeros(length(grid)) :          # light/hidden: no charm source
    n == 0 ? [gc * p.degeneracy * therm(p.mass, q)          for q in grid] :
    n == 2 ? [v2c(q) * gc * p.degeneracy * therm(p.mass, q) for q in grid] :
             zeros(length(grid))

t0  = time()
# charm sits at high mass/momentum → extend the grid to 8 GeV (same ~0.03 spacing)
plan = CascadePlan(dlc; grid = range(0.0, 8.0, length = 268), cache = nothing)
tab = plan.table
@printf("charm table + plan ready in %.1f s:  %d keys\n", time() - t0, length(tab.W))
res = runcascade(plan; primary)

grid = tab.grid
idx(pT) = argmin(abs.(grid .- pT))
charm = (("D⁰", 421, "#2a78d6"), ("D⁺", 411, "#008300"),
         ("D_s⁺", 431, "#d55181"), ("Λc⁺", 4122, "#e08214"))

println("\n charm feed-down:  F_tot/F_direct (from excited charm),  and v₂")
@printf(" %-5s |  %-15s |  %-15s\n", "", "pT = 1 GeV", "pT = 3 GeV")
@printf(" %-5s | %7s %8s | %7s %8s\n", "sp", "enh", "v2tot", "enh", "v2tot")
for (nm, id, _) in charm
    haskey(res.F, id) || continue
    row = String[]
    for pT in (1.0, 3.0)
        i = idx(pT)
        enh = res.F[id][1][i] / max(res.Fprim[id][1][i], eps())
        v2t = res.F[id][2][i] / res.F[id][1][i]
        push!(row, @sprintf("%7.2f %8.4f", enh, v2t))
    end
    @printf(" %-5s | %s | %s\n", nm, row...)
end

# ---- figure ---------------------------------------------------------------
ENV["GKSwstype"] = "100"
using Plots
mask = findall(q -> 0.2 ≤ q ≤ 6.0, grid); pTs = grid[mask]   # stay off the grid-edge artifact
common = (lw = 2, gridalpha = 0.15, foreground_color_legend = nothing,
          background_color = :white, framestyle = :box, tickfontsize = 9,
          guidefontsize = 10, legendfontsize = 8, titlefontsize = 10)

ps = plot(; xlabel = "pT  [GeV]", ylabel = "E dN/d³p   (arb.)", yscale = :log10,
          title = "charm spectra:  direct (dashed) vs total", legend = :topright, common...)
pe = plot(; xlabel = "pT  [GeV]", ylabel = "F_tot / F_direct",
          title = "charm feed-down enhancement", legend = :topright, common...)
psup = plot(; xlabel = "pT  [GeV]", ylabel = "v₂ total / v₂ input",
            title = "charm v₂ suppression", legend = :bottomright, common...)
for (nm, id, col) in charm
    haskey(res.F, id) || continue
    plot!(ps, pTs, res.Fprim[id][1][mask]; color = col, ls = :dash, label = "", common...)
    plot!(ps, pTs, res.F[id][1][mask];     color = col, label = nm, common...)
    plot!(pe, pTs, res.F[id][1][mask] ./ res.Fprim[id][1][mask]; color = col, label = nm, common...)
    plot!(psup, pTs, (res.F[id][2][mask] ./ res.F[id][1][mask]) ./ v2c.(pTs);
          color = col, label = nm, common...)
end
hline!(psup, [1.0]; color = :gray, ls = :dash, label = "no feed-down", common...)
fig = plot(ps, pe, psup; layout = (1, 3), size = (1360, 400), margin = 5Plots.mm)
savefig(fig, joinpath(@__DIR__, "feeddown_charm.png"))
println("\nsaved feeddown_charm.png")
