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

const T   = 0.1565      # GeV, chemical freeze-out
const MUB = 0.0         # GeV, consistent with zero at LHC energies
const VOL = 5300.0      # fm^3, dV/dy for Pb-Pb 0-10%

const SPECIES = [
    (211, "pi+"), (321, "K+"), (310, "K0S"), (2212, "p"),
    (3122, "Lambda"), (3312, "Xi-"), (3334, "Omega-"),
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
    for (particle_id, name) in SPECIES
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
report("QM2016Plus (production list)", ["QM2016Plus"])
report("light_pdg (PDG-numbered species only)", ["light_pdg"])

println("""

Measured, Pb-Pb 0-10% (ALICE): K+/pi+ ~ 0.15, p/pi+ ~ 0.05.
The excess above those values is dominated by feed-down from the quark-model
resonances, which is why both lists are shown.""")
