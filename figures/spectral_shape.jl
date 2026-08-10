# ============================================================================
# What the spectral shape does to pT spectra.
#
# `data/README.md` and the thermal figure quantify the shape choice in *yields*
# — pT-integrated numbers. This asks the sharper question: does it change the
# shape of the spectrum, or only its normalization?
#
# The two options are the mass distribution a broad parent decays from:
#
#   :massdep    Gamma(m) ~ (mR/m) q(m)/q(mR), vanishing at the decay threshold
#   :constant   Gamma(m) = Gamma, the constant-width Breit-Wigner
#
# Both tables are built from one decay list on one grid, with one thermal
# source, so the only thing that differs between them is the parent-mass
# distribution each broad channel integrates over. Everything downstream —
# kernel, matrices, cascade order — is identical.
#
# The expected mechanism, and what the ratio panel tests: a resonance averaged
# over lower masses gives its daughters less momentum, so suppressing the
# low-mass tail should *harden* the daughter spectrum. A pure normalization
# shift would show as a flat ratio; a genuine shape change as a sloped one.
#
# One thing this figure deliberately does NOT contain. Every parent enters with
# the same primary spectrum under both shapes, so what is isolated here is the
# *decay kinematics* -- how the parent-mass distribution maps parent momentum
# onto daughter momentum. The other half of the physics, how many parents a
# given shape produces at this temperature, lives in `thermal_yields.jl`: there
# the shape moves the proton yield by -7.5%. The two are separable and this
# figure is the kinematic half.
#
# Run: julia --project -t auto figures/spectral_shape.jl
# ============================================================================

const ROOT = dirname(@__DIR__)
using FastReso2
using Printf

ENV["GKSwstype"] = "100" # headless GR
using Plots

const T = 0.1565                       # GeV, chemical freeze-out
const GRID = range(0.0, 4.0, length = 134)
const DATA = joinpath(ROOT, "data")

# Static thermal source, mode 0 only: F_a = g exp(-m_T/T), the invariant
# spectrum every primary resonance is born with. Mode 2 is not built — this
# figure is about the spectrum, not about v2.
primary(particle, mode, momenta) =
    mode == 0 ?
    [particle.degeneracy * exp(-hypot(particle.mass, q) / T) for q in momenta] :
    zeros(length(momenta))

decay_list = DecayList(
    reduce(
        merge, [
            readparticles(joinpath(DATA, "particles_$n.dat"))
                for n in ("QM2016Plus_isofix", "sigma")
        ]
    ),
    reduce(
        merge, [
            readdecays(joinpath(DATA, "decays_$n.dat"))
                for n in ("QM2016Plus_isofix", "sigma")
        ]
    ),
)

results = Dict{Symbol, Any}()
for shape in (:massdep, :constant)
    local table, plan
    elapsed = @elapsed begin
        table = CascadeTable(
            decay_list; grid = GRID, modes = [0], spectral = shape, cache = nothing,
        )
        plan = CascadePlan(table, decay_list)
        results[shape] = runcascade(plan; primary)
    end
    @printf(
        "%-9s  %d keys, %d broad  built and run in %.1f s\n",
        shape, length(table.W), count(key -> key.Γ > 0, keys(table.W)), elapsed
    )
end

const SPECIES = [
    (211, "π⁺", "#1f77b4"),
    (321, "K⁺", "#2ca02c"),
    (2212, "p", "#d62728"),
    (3122, "Λ", "#ff7f0e"),
]

spectrum(shape, id) = results[shape].F[id][1]
grid = collect(GRID)
window = findall(q -> 0.15 ≤ q ≤ 3.0, grid)
pTs = grid[window]

common = (
    gridalpha = 0.25,
    foreground_color_legend = nothing,
    background_color = :white,
    framestyle = :box,
    tickfontsize = 10,
    guidefontsize = 11,
    legendfontsize = 9,
    titlefontsize = 11,
)

left = plot(;
    xlabel = "pT [GeV]",
    ylabel = "E dN/d³p [arb.]",
    yscale = :log10,
    title = "spectra: Γ(m) solid, constant Γ dashed",
    legend = :topright,
    common...,
)

for (id, label, color) in SPECIES
    plot!(
        left, pTs, spectrum(:constant, id)[window];
        color, linestyle = :dash, linewidth = 1.4, label = ""
    )
    plot!(
        left, pTs, spectrum(:massdep, id)[window];
        color, linewidth = 1.8, label
    )
end

# The ratio is the point: flat means the shape only renormalizes, sloped means
# it redistributes momentum.
right = plot(;
    xlabel = "pT [GeV]",
    ylabel = "Γ(m) / constant Γ",
    title = "ratio: flat = normalization, sloped = harder",
    legend = :topleft,
    legend_columns = 2,
    common...,
)
for (id, label, color) in SPECIES
    ratio = spectrum(:massdep, id)[window] ./ spectrum(:constant, id)[window]
    plot!(right, pTs, ratio; color, linewidth = 1.8, label)
end
hline!(right, [1.0]; color = "#4c4c4c", linewidth = 0.8, label = "")

figure_handle = plot(
    left, right;
    layout = (1, 2),
    size = (1594, 575),
    margin = 5Plots.mm,
    plot_title = "T = $(T) GeV, static thermal source, QM2016Plus_isofix + f₀(500)",
    plot_titlefontsize = 11,
)
output = joinpath(@__DIR__, "spectral_shape.png")
savefig(figure_handle, output)
@printf("\nwrote %s\n", output)

println("\nratio  Gamma(m) / constant, by pT")
@printf("  %-10s %8s %8s %8s %8s %8s\n", "species", "0.2", "0.5", "1.0", "2.0", "3.0")
for (id, _, _) in SPECIES
    ratio = spectrum(:massdep, id) ./ spectrum(:constant, id)
    at(pT) = ratio[argmin(abs.(grid .- pT))]
    name = decay_list.catalog.byid[id].name
    @printf(
        "  %-10s %8.4f %8.4f %8.4f %8.4f %8.4f\n",
        name, at(0.2), at(0.5), at(1.0), at(2.0), at(3.0)
    )
end
