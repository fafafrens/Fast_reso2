## 1. One decay, on its own
include("DecaySpectra.jl")
using .DecaySpectra
using Printf
using Plots
using LaTeXStrings                                # L"..." labels, rendered by GR

ρ = TwoBodyDecay(0.7752, 0.1396, 0.1396)          # ρ → π + π  (parent, observed, partner)
Fa(qT, η, ϕ) = exp(-hypot(0.7752, qT) * cosh(η) / 0.155)     # any parent spectrum
@printf("pion from ρ at pT = 0.5: %.4e\n", ρ(Fa, 0.5, 0.0, 0.0))

grid = range(0.0, 4.0, length = 134)              # parent and daughter share the grid
W = decaymatrices(ρ, [0, 2], grid, grid)          # one matrix per Fourier mode
@printf("W[n=0] is %s — daughter = W * parent\n", size(W[1]))

## 2. The whole cascade
include("CascadeTables.jl"); using .CascadeTables

dl = DecayList("decays_QM2016Plus_massorder_isofix.dat")
T  = 0.155
primary(p, n, qs) =                               # thermal source, 4% v₂ on mode 2
    (n == 0 ? 1.0 : 0.04) * p.degeneracy * [exp(-hypot(p.mass, q) / T) for q in qs]

qs    = range(0.0, 4.0, length = 61)
table = CascadeTable(dl; grid = qs)               # build every channel matrix (~10 s)
plan  = CascadePlan(table, dl)                    # compile the topology once
res   = runcascade(plan; primary, parallel = :modes)

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
plot(qs,res.Fprim[π⁺][1])
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

available = sort(
    [dl.parts[id] for id in keys(res.F)];
    by = p -> (lowercase(p.name), p.id),
)

foreach(p -> println(p.id, "  ", p.name), available)

res.F