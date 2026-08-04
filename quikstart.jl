## 1. One decay, on its own
include("DecaySpectra.jl")
using .DecaySpectra
using Printf
using Plots
using LaTeXStrings                                # L"..." labels, rendered by GR


## 2. The whole cascade
include("CascadeTables.jl"); using .CascadeTables

const DATA = joinpath(@__DIR__, "data")
base_catalog = readparticles(joinpath(DATA, "particles_QM2016Plus.dat"))
base_network = readdecays(joinpath(DATA, "decays_QM2016Plus.dat"))
dl = DecayList(base_catalog, base_network)

charm_catalog = merge(
    base_catalog,
    readparticles(joinpath(DATA, "particles_charm_bottom.dat")),
)
charm_network = merge(
    base_network,
    readdecays(joinpath(DATA, "decays_charm_bottom.dat")),
)
dl_charm = DecayList(charm_catalog, charm_network)
T  = 0.155
primary(p, n, qs) =                               # thermal source, 4% v₂ on mode 2
    (n == 0 ? 1.0 : 0.04) * p.degeneracy * [exp(-hypot(p.mass, q) / T) for q in qs]

qs = range(0.0, 6.0, length = 151)
res = runcascade(dl; primary, grid = qs, parallel = :modes)
res_charm = runcascade(dl_charm; primary, grid = qs, parallel = :modes)
## 3. Read it out
π⁺ = 211
π0 = 111
π⁻=-211
prot=2212
K⁺=321
@printf("\n  pT    primary       +feed-down   ratio     v₂\n")
for i in (1, 16, 31)
    prim, tot = res.Fprim[π⁺][1][i], res.F[π⁺][1][i]
    @printf("%5.2f  %.4e   %.4e   %5.2f   %+.3f\n",
            qs[i], prim, tot, tot / prim, res.F[π⁺][2][i] / tot)
end

plot(qs,res.Fprim[π⁺][1],yaxis=:log10)
plot!(qs,res.F[π⁺][1])


plot(qs,res.Fprim[prot][1])
plot!(qs,res.F[prot][1])


plot(qs,res.Fprim[K⁺][1])
plot!(qs,res.F[K⁺][1])

p=plot(qs, res.Fprim[π0][1];
     label  = L"\pi^{0}\ \mathrm{primary}",
     xlabel = L"p_T\ [\mathrm{GeV}]",
     ylabel = L"dN/(2\pi p_T\,dp_T\,dy)\ [\mathrm{GeV}^{-2}]",
     title  = L"\mathrm{Pion\ spectra:\ primary\ vs.\ total}", legend = :topright, lw = 2, lc = 1, ls = :dash)
plot!(qs, res.F[π0][1];    label = L"\pi^{0}\ \mathrm{total}",    lw = 2, lc = 1)
plot!(qs, res.Fprim[π⁺][1]; label = L"\pi^{+}\ \mathrm{primary}", lw = 2, lc = 2, ls = :dash)
plot!(qs, res.F[π⁺][1];     label = L"\pi^{+}\ \mathrm{total}",   lw = 2, lc = 2)
plot!(qs, res.Fprim[π⁻][1]; label = L"\pi^{-}\ \mathrm{primary}", lw = 2, lc = 3, ls = :dash)
plot!(qs, res.F[π⁻][1];     label = L"\pi^{-}\ \mathrm{total}",   lw = 2, lc = 3)

savefig( p,"pions.pdf")

available_particles = sort(
    stableparticles(dl_charm);
    by = p -> (lowercase(p.name), p.id),
)

foreach(p -> println(p.id, "  ", p.name), available_particles)

function plot_charm_spectra(result, grid, particles)
    spectrum_plot = plot(;
        xlabel = L"p_T\ [\mathrm{GeV}]",
        ylabel = L"F_0(p_T)\ [\mathrm{arb.\ units}]",
        title = L"\mathrm{Charm\ spectra:\ primary\ vs.\ cascade}",
        yscale = :log10,
        legend = :topright,
        framestyle = :box,
        gridalpha = 0.2,
    )

    for (color, (particle_id, tex_name)) in enumerate(particles)
        haskey(result.F, particle_id) ||
            throw(ArgumentError("particle ID $particle_id is not in the cascade output"))

        plot!(
            spectrum_plot,
            grid,
            result.Fprim[particle_id][1];
            label = latexstring(tex_name, raw"\ \mathrm{primary}"),
            color,
            linestyle = :dash,
            linewidth = 2,
        )
        plot!(
            spectrum_plot,
            grid,
            result.F[particle_id][1];
            label = latexstring(tex_name, raw"\ \mathrm{total}"),
            color,
            linewidth = 2,
        )
    end

    return spectrum_plot
end

charm_particles = [
   # (421, raw"D^{0}"),
   # (443, raw"J/\psi"),
   ( 431, raw"D_s^{+}"),
(4122, raw"\Lambda_c^{+}")
]

charm_spectra = plot_charm_spectra(res_charm, qs, charm_particles)
savefig(charm_spectra, "charm_spectra.pdf")
