# ============================================================================
# Statistical hadronization: thermal yields with resonance feed-down.
#
# Number densities at chemical freeze-out (quantum statistics, finite widths),
# multiplied by the fireball volume and then fed down through the decay list.
# Parameters follow Andronic et al. for Pb-Pb 0-10% at the LHC.
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
full = report("QM2016Plus (production list)", ["QM2016Plus"])
pdg_only = report("light_pdg (PDG-numbered species only)", ["light_pdg"])

# ---------------------------------------------------------------------------
# Left: yields per species, primary against total, for both lists.
# Right: what fraction of each total arrives as feed-down.
# ---------------------------------------------------------------------------
labels = [label for (_, _, label) in SPECIES]
positions = collect(0:length(SPECIES) - 1)
present(y) = [haskey(y.total, id) ? y.total[id] : NaN for (id, _, _) in SPECIES]
primaries(y) = [haskey(y.primary, id) ? y.primary[id] : NaN for (id, _, _) in SPECIES]
fractions(y) = [haskey(y.total, id) && y.total[id] > 0 ?
                y.feeddown[id] / y.total[id] : NaN for (id, _, _) in SPECIES]

figure_handle, axes = subplots(1, 2, figsize = (pWidth * 2.6, pHeight * 1.25))
left, right = axes[1], axes[2]

left.plot(positions, primaries(full), "o", color = "0.55", ms = 6,
          mfc = "none", mew = 1.2, label = "primary (thermal)")
left.plot(positions, present(full), "s", color = "C0", ms = 6,
          label = "total, QM2016Plus")
left.plot(positions, present(pdg_only), "^", color = "C1", ms = 6,
          mfc = "none", mew = 1.2, label = raw"total, PDG-numbered only")
left.set_yscale("log")
left.set_xticks(positions); left.set_xticklabels(labels)
left.set_ylabel(raw"$\mathrm{d}N/\mathrm{d}y$")
left.set_title("thermal yields with feed-down")
left.legend(frameon = false, loc = "lower left")
left.grid(alpha = 0.25, axis = "y")

right.plot(positions, fractions(full), "s", color = "C0", ms = 6,
           label = "QM2016Plus")
right.plot(positions, fractions(pdg_only), "^", color = "C1", ms = 6,
           mfc = "none", mew = 1.2, label = "PDG-numbered only")
right.set_xticks(positions); right.set_xticklabels(labels)
right.set_ylim(0, 1)
right.set_ylabel("feed-down fraction of total")
right.set_title("how much arrives from decays")
right.legend(frameon = false, loc = "lower left")
right.grid(alpha = 0.25, axis = "y")

figure_handle.text(0.5, 0.945,
    "\$T = $(T)\\,\\mathrm{GeV},\\quad \\mu_B = 0,\\quad V = $(Int(VOL))\\,\\mathrm{fm}^3\$",
    ha = "center", fontsize = 11)
figure_handle.subplots_adjust(left = 0.09, right = 0.98, top = 0.84, bottom = 0.11, wspace = 0.28)
output = joinpath(@__DIR__, "thermal_yields.png")
figure_handle.savefig(output, dpi = 200)
@printf("\nwrote %s\n", output)

println("""

Measured, Pb-Pb 0-10% (ALICE): K+/pi+ ~ 0.15, p/pi+ ~ 0.05.
The excess above those values is dominated by feed-down from the quark-model
resonances, which is why both lists are shown.""")
