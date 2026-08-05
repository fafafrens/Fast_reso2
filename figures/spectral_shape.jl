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

isdefined(Main, :CascadeTables) || include("../CascadeTables.jl")

const ROOT = dirname(@__DIR__)
using .CascadeTables
using Printf

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
    reduce(merge, [readparticles(joinpath(DATA, "particles_$n.dat"))
                   for n in ("QM2016Plus_isofix", "sigma")]),
    reduce(merge, [readdecays(joinpath(DATA, "decays_$n.dat"))
                   for n in ("QM2016Plus_isofix", "sigma")]),
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
    @printf("%-9s  %d keys, %d broad  built and run in %.1f s\n",
            shape, length(table.W), count(key -> key.Γ > 0, keys(table.W)), elapsed)
end

const SPECIES = [
    (211,  raw"$\pi^+$", "C0"),
    (321,  raw"$K^+$",   "C2"),
    (2212, raw"$p$",     "C3"),
    (3122, raw"$\Lambda$", "C1"),
]

spectrum(shape, id) = results[shape].F[id][1]
grid = collect(GRID)
window = findall(q -> 0.15 ≤ q ≤ 3.0, grid)
pTs = grid[window]

figure_handle, axes = subplots(1, 2, figsize = (pWidth * 2.6, pHeight * 1.25))
left, right = axes[1], axes[2]

for (id, label, color) in SPECIES
    left.plot(pTs, spectrum(:constant, id)[window], "--", color = color, lw = 1.4)
    left.plot(pTs, spectrum(:massdep, id)[window], "-", color = color, lw = 1.8,
              label = label)
end
left.set_yscale("log")
left.set_xlabel(raw"$p_T$ [GeV]")
left.set_ylabel(raw"$E\,\mathrm{d}N/\mathrm{d}^3p$  [arb.]")
left.set_title(raw"spectra: $\Gamma(m)$ solid, constant $\Gamma$ dashed")
left.legend(frameon = false, loc = "upper right")
left.grid(alpha = 0.25)

# The ratio is the point: flat means the shape only renormalizes, sloped means
# it redistributes momentum.
for (id, label, color) in SPECIES
    ratio = spectrum(:massdep, id)[window] ./ spectrum(:constant, id)[window]
    right.plot(pTs, ratio, "-", color = color, lw = 1.8, label = label)
end
right.axhline(1, color = "0.3", lw = 0.8)
right.set_xlabel(raw"$p_T$ [GeV]")
right.set_ylabel(raw"$\Gamma(m)\ /\ $constant $\Gamma$")
right.set_title("ratio: flat = normalization, sloped = harder")
right.legend(frameon = false, loc = "upper left", ncol = 2)
right.grid(alpha = 0.25)

figure_handle.text(0.5, 0.955,
    "\$T = $(T)\\,\\mathrm{GeV}\$, static thermal source, QM2016Plus\\_isofix + \$f_0(500)\$",
    ha = "center", fontsize = 11)
figure_handle.subplots_adjust(left = 0.09, right = 0.98, top = 0.80,
                              bottom = 0.20, wspace = 0.28)
output = joinpath(@__DIR__, "spectral_shape.png")
figure_handle.savefig(output, dpi = 200)
@printf("\nwrote %s\n", output)

println("\nratio  Gamma(m) / constant, by pT")
@printf("  %-10s %8s %8s %8s %8s %8s\n", "species", "0.2", "0.5", "1.0", "2.0", "3.0")
for (id, _, _) in SPECIES
    ratio = spectrum(:massdep, id) ./ spectrum(:constant, id)
    at(pT) = ratio[argmin(abs.(grid .- pT))]
    name = decay_list.catalog.byid[id].name
    @printf("  %-10s %8.4f %8.4f %8.4f %8.4f %8.4f\n",
            name, at(0.2), at(0.5), at(1.0), at(2.0), at(3.0))
end
