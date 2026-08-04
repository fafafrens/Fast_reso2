# ============================================================================
# Statistical hadronization: thermal yields with resonance feed-down.
#
# Number densities at chemical freeze-out (quantum statistics, finite widths),
# multiplied by the fireball volume and then fed down through the decay list.
# Parameters follow Andronic et al. for Pb-Pb 0-10% at the LHC.
#
# Three lists are run, each a superset of the one before it, so every difference
# between the curves is attributable to the species that were added:
#
#   light_pdg          the 351 species carrying a PDG MC ID
#   QM2016Plus         the production list, 1518 species
#   QM2016Plus + sigma the same plus f0(500)
#
# The sigma is the most extreme entry in any of these tables -- m = 0.475 GeV
# with Gamma = 0.55 -- so it is also the sharpest test of the finite-width
# average. Four widths below its pole mass is negative, and a Breit-Wigner
# integrated from there would credit the thermal density with unphysical mass;
# `thermaldensity` cuts the support at the 2*m_pi decay threshold instead. It
# decays only to pions, so it moves the pion yield and nothing else.
#
# Two caveats worth reading before comparing with data:
#
#   * The light table's branching ratios are curated, not measured: 1167 of its
#     1518 species are quark-model states with hand-assigned BRs, and
#     pdg_audit.md records that only eta and omega regenerate from PDG. Agreement
#     at the 10% level is the realistic target.
#   * More states means more feed-down. QM2016Plus carries far more resonances
#     than a PDG-only list, so its ratios sit above the published SHM values;
#     the light_pdg layer is printed alongside to show the size of that effect.
#
# Run: julia --project -t auto figures/thermal_yields.jl
# ============================================================================

isdefined(Main, :CascadeTables) || include("../CascadeTables.jl")
isdefined(Main, :ThermalModel) || include("../ThermalModel.jl")

const ROOT = dirname(@__DIR__)
using .CascadeTables
using .ThermalModel
using Printf

# PyPlot is a figure-only dependency: installed into the shared default
# environment rather than the project, which describes what the library needs.
using Pkg
let project = Base.active_project()
    if Base.find_package("PyPlot") === nothing
        @info "installing PyPlot into the default environment (not Project.toml)"
        Pkg.activate(); Pkg.add("PyPlot"); Pkg.activate(project)
    end
end
using PyPlot

function plot_params(; gui = false, pHeight = 2.3)
    close("all")
    pygui(gui)
    rc("text", usetex = "true")
    rc("font", family = "serif")
    rc("font", size = 12)
    rc("xtick", labelsize = 12)
    rc("ytick", labelsize = 12)
    rc("lines", markersize = 1)
    rc("legend", fontsize = 10)
    rc("axes", titlesize = 12)
    rc("axes", titlepad = 10)
    return nothing
end
pHeight = 2.3
pWidth = 1.3333333 * pHeight
plot_params()

const T   = 0.1565      # GeV, chemical freeze-out
const MUB = 0.0         # GeV, consistent with zero at LHC energies
const VOL = 5300.0      # fm^3, dV/dy for Pb-Pb 0-10%

const SIGMA = 9000221   # f0(500)

const SPECIES = [
    (211,  "pi+",    raw"$\pi^+$"),
    (321,  "K+",     raw"$K^+$"),
    (311,  "K0",     raw"$K^0$"),
    (2212, "p",      raw"$p$"),
    (3122, "Lambda", raw"$\Lambda$"),
    (3312, "Xi-",    raw"$\Xi^-$"),
    (3334, "Omega-", raw"$\Omega^-$"),
]

load(layers) = DecayList(
    reduce(merge, [readparticles(joinpath(ROOT, "data", "particles_$n.dat")) for n in layers]),
    reduce(merge, [readdecays(joinpath(ROOT, "data", "decays_$n.dat")) for n in layers]),
)

function report(label, layers)
    decay_list = load(layers)
    elapsed = @elapsed yields = thermalyields(decay_list, T; muB = MUB, V = VOL)
    @printf("\n%s  (%d species, %.1f s)\n", label, length(decay_list.catalog.byid), elapsed)
    @printf("  %-9s %10s %10s %10s %8s\n", "species", "primary", "feed-down", "total", "feed/tot")
    for (particle_id, name, _) in SPECIES
        haskey(yields.total, particle_id) || continue
        total = yields.total[particle_id]
        @printf("  %-9s %10.2f %10.2f %10.2f %8.2f\n",
                name, yields.primary[particle_id], yields.feeddown[particle_id],
                total, yields.feeddown[particle_id] / total)
    end
    @printf("  ratios: K+/pi+ = %.4f   p/pi+ = %.4f   Lambda/pi+ = %.4f\n",
            yields.total[321] / yields.total[211],
            yields.total[2212] / yields.total[211],
            yields.total[3122] / yields.total[211])
    return yields
end

@printf("T = %.4f GeV   muB = %.3f GeV   V = %.0f fm^3\n", T, MUB, VOL)
pdg_only = report("light_pdg (PDG-numbered species only)", ["light_pdg"])
full     = report("QM2016Plus (production list)", ["QM2016Plus"])
sigma    = report("QM2016Plus + f0(500)", ["QM2016Plus", "sigma"])

# ---------------------------------------------------------------------------
# What the sigma alone does. It is one species out of 1519 and decays only to
# pions, so it is the one case here where a single entry's effect is separable.
# ---------------------------------------------------------------------------
@printf("\nf0(500), m = 0.475 GeV, Gamma = 0.55 GeV, support cut at 2*m_pi = %.4f\n",
        2 * 0.13957)
@printf("  primary yield of the sigma itself   %8.2f\n", sigma.primary[SIGMA])
for (particle_id, name, _) in SPECIES
    before, after = full.total[particle_id], sigma.total[particle_id]
    @printf("  %-9s %10.2f -> %10.2f  %+7.2f%%\n",
            name, before, after, 100 * (after - before) / before)
end
@printf("  K+/pi+  %.4f -> %.4f     p/pi+  %.4f -> %.4f\n",
        full.total[321] / full.total[211], sigma.total[321] / sigma.total[211],
        full.total[2212] / full.total[211], sigma.total[2212] / sigma.total[211])

# ---------------------------------------------------------------------------
# Left:   yields per species, primary against total, for all three lists.
# Middle: what fraction of each total arrives as feed-down.
# Right:  what each added layer is worth, as a percentage of the total.
#
# The third panel exists because the first cannot show the sigma. Adding it
# moves the pion yield by 3.4%, which on a log axis spanning three decades is
# less than the marker. Plotted as a relative change next to what the
# quark-model states contribute, both are legible on one linear scale.
#
# In the first two panels the series are drawn at small x-offsets. Without them
# the QM2016Plus and +sigma markers sit exactly on top of each other everywhere
# except the pion -- which is the point being made, but is invisible if they
# overlap.
# ---------------------------------------------------------------------------
labels = [label for (_, _, label) in SPECIES]
positions = collect(0:length(SPECIES) - 1)
present(y) = [haskey(y.total, id) ? y.total[id] : NaN for (id, _, _) in SPECIES]
primaries(y) = [haskey(y.primary, id) ? y.primary[id] : NaN for (id, _, _) in SPECIES]
fractions(y) = [haskey(y.total, id) && y.total[id] > 0 ?
                y.feeddown[id] / y.total[id] : NaN for (id, _, _) in SPECIES]

figure_handle, axes = subplots(1, 3, figsize = (pWidth * 3.9, pHeight * 1.25))
left, right, delta = axes[1], axes[2], axes[3]

left.plot(positions .- 0.18, primaries(full), "o", color = "0.55", ms = 6,
          mfc = "none", mew = 1.2, label = "primary (thermal)")
left.plot(positions .- 0.06, present(pdg_only), "^", color = "C1", ms = 6,
          mfc = "none", mew = 1.2, label = raw"total, PDG-numbered only")
left.plot(positions .+ 0.06, present(full), "s", color = "C0", ms = 6,
          label = "total, QM2016Plus")
left.plot(positions .+ 0.18, present(sigma), "D", color = "C2", ms = 5,
          mfc = "none", mew = 1.2, label = raw"total, $+\,f_0(500)$")
left.set_yscale("log")
left.set_xticks(positions); left.set_xticklabels(labels)
left.set_ylabel(raw"$\mathrm{d}N/\mathrm{d}y$")
left.set_title("thermal yields with feed-down")
left.legend(frameon = false, loc = "lower left")
left.grid(alpha = 0.25, axis = "y")

right.plot(positions .- 0.12, fractions(pdg_only), "^", color = "C1", ms = 6,
           mfc = "none", mew = 1.2, label = "PDG-numbered only")
right.plot(positions, fractions(full), "s", color = "C0", ms = 6,
           label = "QM2016Plus")
right.plot(positions .+ 0.12, fractions(sigma), "D", color = "C2", ms = 5,
           mfc = "none", mew = 1.2, label = raw"$+\,f_0(500)$")
right.set_xticks(positions); right.set_xticklabels(labels)
right.set_ylim(-0.04, 1.02)
right.set_ylabel("feed-down fraction of total")
right.set_title("how much arrives from decays")
right.legend(frameon = false, loc = "lower left")
right.grid(alpha = 0.25, axis = "y")

percent(after, before) = [100 * (after.total[id] - before.total[id]) / before.total[id]
                          for (id, _, _) in SPECIES]
delta.bar(positions .- 0.19, percent(full, pdg_only), 0.36, color = "C0",
          label = "quark-model states")
delta.bar(positions .+ 0.19, percent(sigma, full), 0.36, color = "C2",
          label = raw"$f_0(500)$")
delta.axhline(0, color = "0.3", lw = 0.8)
delta.set_xticks(positions); delta.set_xticklabels(labels)
delta.set_ylabel("change in total yield [\\%]")
delta.set_title("what each layer adds")
delta.legend(frameon = false, loc = "upper left")
delta.grid(alpha = 0.25, axis = "y")

figure_handle.text(0.5, 0.955,
    "\$T = $(T)\\,\\mathrm{GeV},\\quad \\mu_B = 0,\\quad V = $(Int(VOL))\\,\\mathrm{fm}^3\$",
    ha = "center", fontsize = 11)
figure_handle.subplots_adjust(left = 0.07, right = 0.985, top = 0.80, bottom = 0.11, wspace = 0.30)
output = joinpath(@__DIR__, "thermal_yields.png")
figure_handle.savefig(output, dpi = 200)
@printf("\nwrote %s\n", output)

println("""

Measured, Pb-Pb 0-10% (ALICE): K+/pi+ ~ 0.15, p/pi+ ~ 0.05.
The excess above those values is dominated by feed-down from the quark-model
resonances, which is why all three lists are shown.""")
