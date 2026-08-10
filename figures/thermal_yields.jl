# ============================================================================
# Statistical hadronization: thermal yields with resonance feed-down.
#
# Number densities at chemical freeze-out (quantum statistics, finite widths),
# multiplied by the fireball volume and then fed down through the decay list.
# Parameters follow Andronic et al. for Pb-Pb 0-10% at the LHC.
#
# Five modelling choices are separated here, each measured against the one
# before it, so no effect is attributed to more than one cause:
#
#   light_pdg               the 351 species carrying a PDG MC ID
#   QM2016Plus              the quark-model list, 1518 species
#   QM2016Plus_isofix       three corrected isospin decompositions
#   + sigma                 f0(500)
#   pole mass vs width      the same list with the mass average off
#   spectral shape          threshold-vanishing Gamma(m) vs constant Gamma
#
# The last four are small in species count and large in effect, which is the
# point of separating them.
#
# `isofix` patches eta, a1(1260) and K(1630), whose isospin decompositions were
# wrong. All three feed pions and nothing else, so it moves the pion yield by
# -3% and leaves every other species identical to the digit.
#
# The sigma is the most extreme entry in any of these tables -- m = 0.475 GeV
# with Gamma = 0.55 -- so it is also the sharpest test of the finite-width
# average. Four widths below its pole mass is negative, and a Breit-Wigner
# integrated from there would credit the thermal density with unphysical mass;
# `thermaldensity` cuts the support at the 2*m_pi decay threshold instead. It
# decays only to pions, so it moves the pion yield and nothing else.
#
# Finite widths are the largest correction and the least uniform: the proton is
# fed by the broad Delta(1232) and the N* tower, the Omega- by a handful of
# narrow states. Turning them off is not a small perturbation on the ratios; it
# is a different model.
#
# Which spectral shape carries the width is a physics choice in its own right,
# so it gets its own bar. The default `:massdep` gives each resonance a
# threshold-vanishing width Gamma(m) ~ (mR/m) q(m)/q(mR), so it carries no
# weight below the mass of its own decay products; `:constant` freezes
# Gamma(m) = Gamma. At l = 0 the two do not order universally -- the truncation
# removes low-mass weight while the tail adds high-mass weight, and which wins
# depends on where the threshold sits inside the window. The sigma, whose
# threshold cuts deep into its window, is where they part company.
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

const ROOT = dirname(@__DIR__)
using FastReso2
using Printf

ENV["GKSwstype"] = "100" # headless GR
using Plots

const T = 0.1565      # GeV, chemical freeze-out
const MUB = 0.0         # GeV, consistent with zero at LHC energies
const VOL = 5300.0      # fm^3, dV/dy for Pb-Pb 0-10%

const SIGMA = 9000221   # f0(500)

const SPECIES = [
    (211, "pi+", "π⁺"),
    (321, "K+", "K⁺"),
    (311, "K0", "K⁰"),
    (2212, "p", "p"),
    (3122, "Lambda", "Λ"),
    (3312, "Xi-", "Ξ⁻"),
    (3334, "Omega-", "Ω⁻"),
]

# isofix + sigma reproduces the whole-file QM2016Plus_isofix_sigma exactly --
# every yield agrees to 0.0e+00 -- so the layered form is used, which lets the
# sigma be switched on and off without swapping tables.
const ISOFIX_SIGMA = ["QM2016Plus_isofix", "sigma"]

load(layers) = DecayList(
    reduce(merge, [readparticles(joinpath(ROOT, "data", "particles_$n.dat")) for n in layers]),
    reduce(merge, [readdecays(joinpath(ROOT, "data", "decays_$n.dat")) for n in layers]),
)

function report(label, layers; width = true, shape = :massdep)
    decay_list = load(layers)
    elapsed = @elapsed yields = thermalyields(
        decay_list, T; muB = MUB, V = VOL, width = width, shape = shape
    )
    @printf("\n%s  (%d species, %.1f s)\n", label, length(decay_list.catalog.byid), elapsed)
    @printf("  %-9s %10s %10s %10s %8s\n", "species", "primary", "feed-down", "total", "feed/tot")
    for (particle_id, name, _) in SPECIES
        haskey(yields.total, particle_id) || continue
        total = yields.total[particle_id]
        @printf(
            "  %-9s %10.2f %10.2f %10.2f %8.2f\n",
            name, yields.primary[particle_id], yields.feeddown[particle_id],
            total, yields.feeddown[particle_id] / total
        )
    end
    @printf(
        "  ratios: K+/pi+ = %.4f   p/pi+ = %.4f   Lambda/pi+ = %.4f\n",
        yields.total[321] / yields.total[211],
        yields.total[2212] / yields.total[211],
        yields.total[3122] / yields.total[211]
    )
    return yields
end

@printf("T = %.4f GeV   muB = %.3f GeV   V = %.0f fm^3\n", T, MUB, VOL)
pdg_only = report("light_pdg (PDG-numbered species only)", ["light_pdg"])
raw_qm = report("QM2016Plus (uncorrected isospin)", ["QM2016Plus"])
full = report("QM2016Plus_isofix (production list)", ["QM2016Plus_isofix"])
sigma = report("QM2016Plus_isofix + f0(500)", ISOFIX_SIGMA)
pole = report(
    "QM2016Plus_isofix + f0(500), pole mass (no width average)",
    ISOFIX_SIGMA; width = false
)
constant = report(
    "QM2016Plus_isofix + f0(500), constant-width shape",
    ISOFIX_SIGMA; shape = :constant
)

# ---------------------------------------------------------------------------
# What the sigma alone does. It is one species out of 1519 and decays only to
# pions, so it is the one case here where a single entry's effect is separable.
# ---------------------------------------------------------------------------
@printf(
    "\nf0(500), m = 0.475 GeV, Gamma = 0.55 GeV, support cut at 2*m_pi = %.4f\n",
    2 * 0.13957
)
@printf("  primary yield of the sigma itself   %8.2f\n", sigma.primary[SIGMA])
for (particle_id, name, _) in SPECIES
    before, after = full.total[particle_id], sigma.total[particle_id]
    @printf(
        "  %-9s %10.2f -> %10.2f  %+7.2f%%\n",
        name, before, after, 100 * (after - before) / before
    )
end
@printf(
    "  K+/pi+  %.4f -> %.4f     p/pi+  %.4f -> %.4f\n",
    full.total[321] / full.total[211], sigma.total[321] / sigma.total[211],
    full.total[2212] / full.total[211], sigma.total[2212] / sigma.total[211]
)

# ---------------------------------------------------------------------------
# What the finite widths are worth. Unlike the three list changes above this
# one touches every species, and unevenly: the proton is fed by the broad
# Delta(1232) and the N* tower, the Omega- by a handful of narrow states.
# ---------------------------------------------------------------------------
println("\npole mass vs mass average, and the two shapes, on the same list")
@printf(
    "  %-9s %10s %10s %10s %9s %9s\n",
    "species", "pole", "constant", "massdep", "width[%]", "shape[%]"
)
for (particle_id, name, _) in SPECIES
    p0 = pole.total[particle_id]
    pc = constant.total[particle_id]
    pm = sigma.total[particle_id]
    @printf(
        "  %-9s %10.2f %10.2f %10.2f %+8.2f%% %+8.2f%%\n",
        name, p0, pc, pm, 100 * (pm - p0) / p0, 100 * (pm - pc) / pc
    )
end
ratio(y, a, b) = y.total[a] / y.total[b]
@printf(
    "  K+/pi+  pole %.4f  constant %.4f  massdep %.4f\n",
    ratio(pole, 321, 211), ratio(constant, 321, 211), ratio(sigma, 321, 211)
)
@printf(
    "  p/pi+   pole %.4f  constant %.4f  massdep %.4f\n",
    ratio(pole, 2212, 211), ratio(constant, 2212, 211), ratio(sigma, 2212, 211)
)

# ---------------------------------------------------------------------------
# Left:   yields per species, primary against total, for the three lists.
# Middle: what fraction of each total arrives as feed-down.
# Right:  what each modelling choice is worth, as a percentage of the total.
#
# The third panel exists because the first cannot show any of the small
# choices. The sigma moves the pion yield by 3.4% and isofix by -3%, both less
# than a marker on a log axis spanning three decades. As relative changes they
# sit on one linear scale beside the two large effects, the quark-model states
# and the finite widths.
#
# In the first two panels the series are drawn at small x-offsets. Without them
# the isofix and +sigma markers sit exactly on top of each other everywhere
# except the pion -- which is the point being made, but is invisible if they
# overlap.
# ---------------------------------------------------------------------------
labels = [label for (_, _, label) in SPECIES]
positions = collect(0:(length(SPECIES) - 1))
present(y) = [haskey(y.total, id) ? y.total[id] : NaN for (id, _, _) in SPECIES]
primaries(y) = [haskey(y.primary, id) ? y.primary[id] : NaN for (id, _, _) in SPECIES]
fractions(y) = [
    haskey(y.total, id) && y.total[id] > 0 ?
        y.feeddown[id] / y.total[id] : NaN for (id, _, _) in SPECIES
]

common = (
    gridalpha = 0.25,
    foreground_color_legend = nothing,
    background_color = :white,
    framestyle = :box,
    tickfontsize = 10,
    guidefontsize = 11,
    legendfontsize = 8,
    titlefontsize = 11,
)

left = plot(;
    xticks = (positions, labels),
    ylabel = "dN/dy",
    yscale = :log10,
    title = "thermal yields with feed-down",
    legend = :bottomleft,
    common...,
)
scatter!(
    left, positions .- 0.18, primaries(full);
    color = "#8c8c8c", marker = :circle, markersize = 6,
    markercolor = :white, markerstrokewidth = 1.2,
    label = "primary (thermal)"
)
scatter!(
    left, positions .- 0.06, present(pdg_only);
    color = "#ff7f0e", marker = :utriangle, markersize = 6,
    markercolor = :white, markerstrokewidth = 1.2,
    label = "total, PDG-numbered only"
)
scatter!(
    left, positions .+ 0.06, present(full);
    color = "#1f77b4", marker = :square, markersize = 6,
    label = "total, QM2016Plus_isofix"
)
scatter!(
    left, positions .+ 0.18, present(sigma);
    color = "#2ca02c", marker = :diamond, markersize = 5,
    markercolor = :white, markerstrokewidth = 1.2,
    label = "total, + f₀(500)"
)

right = plot(;
    xticks = (positions, labels),
    ylims = (-0.04, 1.02),
    ylabel = "feed-down fraction of total",
    title = "how much arrives from decays",
    legend = :bottomleft,
    common...,
)
scatter!(
    right, positions .- 0.12, fractions(pdg_only);
    color = "#ff7f0e", marker = :utriangle, markersize = 6,
    markercolor = :white, markerstrokewidth = 1.2,
    label = "PDG-numbered only"
)
scatter!(
    right, positions, fractions(full);
    color = "#1f77b4", marker = :square, markersize = 6,
    label = "QM2016Plus_isofix"
)
scatter!(
    right, positions .+ 0.12, fractions(sigma);
    color = "#2ca02c", marker = :diamond, markersize = 5,
    markercolor = :white, markerstrokewidth = 1.2,
    label = "+ f₀(500)"
)

percent(after, before) = [
    100 * (after.total[id] - before.total[id]) / before.total[id]
        for (id, _, _) in SPECIES
]
# Against `raw_qm`, not `full`: `full` already carries the isospin fix, which
# is its own bar, so measuring the quark-model states against it would count
# that step twice and contradict "each measured against the one before it".
delta = plot(;
    xticks = (positions, labels),
    ylabel = "change in total yield [%]",
    title = "what each choice is worth",
    legend = :topleft,
    common...,
)
bar!(
    delta, positions .- 0.32, percent(raw_qm, pdg_only);
    bar_width = 0.15, color = "#1f77b4", label = "quark-model states"
)
bar!(
    delta, positions .- 0.16, percent(sigma, pole);
    bar_width = 0.15, color = "#9467bd", label = "finite widths"
)
bar!(
    delta, positions, percent(sigma, constant);
    bar_width = 0.15, color = "#8c564b", label = "shape: Γ(m) vs Γ"
)
bar!(
    delta, positions .+ 0.16, percent(sigma, full);
    bar_width = 0.15, color = "#2ca02c", label = "f₀(500)"
)
bar!(
    delta, positions .+ 0.32, percent(full, raw_qm);
    bar_width = 0.15, color = "#d62728", label = "isospin fix"
)
hline!(delta, [0.0]; color = "#4c4c4c", linewidth = 0.8, label = "")

figure_handle = plot(
    left, right, delta;
    layout = (1, 3),
    size = (2391, 575),
    margin = 5Plots.mm,
    plot_title = "T = $(T) GeV,  μB = 0,  V = $(Int(VOL)) fm³",
    plot_titlefontsize = 11,
)
output = joinpath(@__DIR__, "thermal_yields.png")
savefig(figure_handle, output)
@printf("\nwrote %s\n", output)

println(
    """

    Measured, Pb-Pb 0-10% (ALICE): K+/pi+ ~ 0.15, p/pi+ ~ 0.05.
    The excess above those values is dominated by feed-down from the quark-model
    resonances, which is why every list is shown separately. Finite widths are the
    largest single correction and the least uniform across species: dropping them
    is not a small perturbation on the ratios. The spectral shape carrying that
    width is a separate choice, and the sigma is where the two shapes part."""
)
