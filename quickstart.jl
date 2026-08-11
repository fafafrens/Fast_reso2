## 1. One decay, on its own
using FastReso2
using Printf

ρ = TwoBodyDecay(0.7752, 0.1396, 0.1396)
Fa(qT, η, ϕ) = exp(-hypot(0.7752, qT) * cosh(η) / 0.155)
@printf("pion from ρ at pT = 0.5: %.4e\n", ρ(Fa, 0.5, 0.0, 0.0))

grid = range(0.0, 4.0, length = 134)
W = decaymatrices(ρ, [0, 2], grid, grid)
@printf("W[n=0] is %s — daughter = W * parent\n", size(W[1]))

## 2. The whole cascade
const DATA = joinpath(@__DIR__, "data")
dl = DecayList(
    readparticles(joinpath(DATA, "particles_QM2016Plus_isofix.dat")),
    readdecays(joinpath(DATA, "decays_QM2016Plus_isofix.dat")),
)
T = 0.155
primary(p, n, qs) =
    (n == 0 ? 1.0 : 0.04) * p.degeneracy *
    [exp(-hypot(p.mass, q) / T) for q in qs]

qs = range(0.0, 4.0, length = 61)
res = runcascade(
    dl;
    primary,
    grid = qs,
    spectral = :massdep,
    parallel = :modes,
)

## 3. Read it out
π⁺ = 211
@printf("\n  pT    primary       +feed-down   ratio     v₂\n")
for i in (1, 16, 31)
    prim, tot = res.Fprim[π⁺][1][i], res.F[π⁺][1][i]
    @printf(
        "%5.2f  %.4e   %.4e   %5.2f   %+.3f\n",
        qs[i], prim, tot, tot / prim, res.F[π⁺][2][i] / tot,
    )
end

## 4. Plot it — labels are LaTeX, so the figure drops straight into matrix_notes.tex
ENV["GKSwstype"] = "100"           # headless GR
using Plots
using LaTeXStrings

species = [(211, L"\pi^+", "#2a78d6"), (321, L"K^+", "#008300"),
           (2212, L"p", "#d55181")]
# Font sizes are set for the figure at \linewidth in matrix_notes.tex, where the
# 920 px width is scaled down by ~0.7; on screen they look slightly large.
common = (lw = 2, gridalpha = 0.15, foreground_color_legend = nothing,
          background_color = :white, framestyle = :box, tickfontsize = 11,
          guidefontsize = 14, legendfontsize = 11, titlefontsize = 13)

p1 = plot(; yscale = :log10, xlabel = L"p_T\ \mathrm{[GeV]}",
          ylabel = L"E\,\mathrm{d}N/\mathrm{d}^3p\ \ \mathrm{(arb.)}",
          title = "spectra: primary (dashed) vs total", common...)
p2 = plot(; xlabel = L"p_T\ \mathrm{[GeV]}", ylabel = L"v_2",
          legend = :bottomright, title = "elliptic flow after feed-down",
          common...)
for (id, label, color) in species
    plot!(p1, qs, res.F[id][1]; label, color, common...)
    plot!(p1, qs, res.Fprim[id][1]; label = "", color, ls = :dash, common...)
    plot!(p2, qs, res.F[id][2] ./ res.F[id][1]; label, color, common...)
end
hline!(p2, [0.04]; label = L"\mathrm{input}\ v_2", color = :gray,
       ls = :dash, lw = 2)

fig = plot(p1, p2; layout = (1, 2), size = (920, 380), margin = 5Plots.mm)
out = joinpath(@__DIR__, "figures", "quickstart.png")
savefig(fig, out)
println("\nplot saved to ", out)
