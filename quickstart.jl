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
