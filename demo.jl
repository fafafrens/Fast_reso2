# Example usage of DecaySpectra.jl: pion feed-down spectra from ρ → ππ
# (two-body, Eq. 32) and ω → π⁺π⁻π⁰ (three-body, Eq. 59) for a static
# thermal source with elliptic-flow modulation.

include("DecaySpectra.jl")
using .DecaySpectra
using Printf

# `const` matters: non-const globals captured in the spectra closures make the
# innermost integrand dynamically typed and ~1000× slower.
const T  = 0.150               # GeV
const v2 = 0.05

ρ = TwoBodyDecay(0.7752, 0.1396, 0.1396; B = 1.0)             # ρ⁰ → π⁺π⁻
ω = ThreeBodyDecay(0.7827, 0.1396, 0.1396, 0.1350; B = 0.892) # ω → π⁺π⁻π⁰

# parent invariant spectra F_a = E dN/d³p, Eq. (14): Boltzmann × (1 + 2 v₂ cos 2ϕ)
Fρ(qT, η, ϕ) = exp(-hypot(0.7752, qT) * cosh(η) / T) * (1 + 2v2 * cos(2ϕ))
Fω(qT, η, ϕ) = exp(-hypot(0.7827, qT) * cosh(η) / T) * (1 + 2v2 * cos(2ϕ))

# Fourier-mode inputs (modulated-input trick, Eqs. 41/62): n = 0 and n = ±2
Fρ0(qT, η, ϕ) = exp(-hypot(0.7752, qT) * cosh(η) / T)
Fρ2(qT, η, ϕ) = v2 * Fρ0(qT, η, 0.0) * cos(2ϕ)
Fω0(qT, η, ϕ) = exp(-hypot(0.7827, qT) * cosh(η) / T)
Fω2(qT, η, ϕ) = v2 * Fω0(qT, η, 0.0) * cos(2ϕ)

println("Pion feed-down at midrapidity (ηp = 0), static thermal source, T = $(T) GeV")
println(" pT [GeV] |  F_π from ρ→ππ  |  v₂(π) |  F_π from ω→πππ |  v₂(π)")
println("-"^70)
for pT in 0.2:0.2:1.6
    fρ0 = ρ(Fρ0, pT, 0.0, 0.0)
    fρ2 = ρ(Fρ2, pT, 0.0, 0.0)
    fω0 = ω(Fω0, pT, 0.0, 0.0)
    fω2 = ω(Fω2, pT, 0.0, 0.0)
    @printf(" %8.2f | %.6e | %+.4f | %.6e | %+.4f\n",
            pT, fρ0, fρ2 / fρ0, fω0, fω2 / fω0)
end

# full ϕ-dependent spectrum at one point, Eq. (32)/(59)
pT, ηp, ϕp = 0.6, 0.0, π / 4
@printf("\nF_π(pT=%.1f, ηp=%.1f, ϕp=π/4):  ρ→ππ: %.6e   ω→πππ: %.6e\n",
        pT, ηp, ρ(Fρ, pT, ηp, ϕp), ω(Fω, pT, ηp, ϕp))

# ---------------------------------------------------------------------------
# Plot: feed-down spectra and v₂ vs pT
# ---------------------------------------------------------------------------
ENV["GKSwstype"] = "100"           # headless GR
using Plots

pTs = 0.1:0.05:2.0
sρ = similar(collect(pTs)); vρ = similar(sρ)
sω = similar(sρ);           vω = similar(sρ)
sπ = similar(sρ)                          # primary (thermal) pions, no decay
for (i, pT) in enumerate(pTs)
    f0 = ρ(Fρ0, pT, 0.0, 0.0); sρ[i] = f0; vρ[i] = ρ(Fρ2, pT, 0.0, 0.0) / f0
    g0 = ω(Fω0, pT, 0.0, 0.0); sω[i] = g0; vω[i] = ω(Fω2, pT, 0.0, 0.0) / g0
    sπ[i] = exp(-hypot(0.1396, pT) / T)   # F at ηp = 0; its v₂ is the input v2, flat
end

col_ρ, col_ω, col_π = "#2a78d6", "#008300", "#d55181"   # entity-stable series colors
common = (lw = 2, gridalpha = 0.15, foreground_color_legend = nothing,
          background_color = :white, framestyle = :box, tickfontsize = 9,
          guidefontsize = 10, legendfontsize = 9)

p1 = plot(pTs, sρ; yscale = :log10, label = "ρ → ππ", color = col_ρ,
          xlabel = "pT  [GeV]", ylabel = "F_π = E dN/d³p", title = "pion spectra",
          titlefontsize = 10, common...)
plot!(p1, pTs, sω; label = "ω → π⁺π⁻π⁰", color = col_ω, common...)
plot!(p1, pTs, sπ; label = "primary π", color = col_π, common...)

p2 = plot(pTs, vρ; label = "ρ → ππ", color = col_ρ,
          xlabel = "pT  [GeV]", ylabel = "v₂(π)", title = "pion elliptic flow",
          titlefontsize = 10, legend = :bottomright, common...)
plot!(p2, pTs, vω; label = "ω → π⁺π⁻π⁰", color = col_ω, common...)
plot!(p2, pTs, fill(v2, length(pTs)); label = "primary π", color = col_π, common...)

fig = plot(p1, p2; layout = (1, 2), size = (920, 380), margin = 5Plots.mm)
savefig(fig, "demo_feeddown.png")
println("\nplot saved to demo_feeddown.png")

# ---------------------------------------------------------------------------
# Decay kernel as a precomputed matrix:  F_b,n(pTs) = W_n * F_a,n(qgrid)
# (η-independent source; see DecayMatrix docstring)
# ---------------------------------------------------------------------------
const qgrid = range(0.0, 4.2, length = 121)
FBI(qT)  = exp(-hypot(0.7752, qT) / T)    # boost-invariant ρ source, n = 0 mode
FBIω(qT) = exp(-hypot(0.7827, qT) / T)

t0 = time()
M0 = DecayMatrix(ρ, 0, pTs, qgrid)
M2 = DecayMatrix(ρ, 2, pTs, qgrid)
Mω = DecayMatrix(ω, 0, pTs, qgrid)
@printf("assembled ρ W₀, W₂ and ω W₀  (%d×%d) in %.2f s\n",
        length(pTs), length(qgrid), time() - t0)

t0 = time()
fb0_fun = [ρ((qT, η, ϕ) -> FBI(qT),                pT, 0.0, 0.0) for pT in pTs]
fb2_fun = [ρ((qT, η, ϕ) -> v2 * FBI(qT) * cos(2ϕ), pT, 0.0, 0.0) for pT in pTs]
fω0_fun = [ω((qT, η, ϕ) -> FBIω(qT),               pT, 0.0, 0.0) for pT in pTs]
tfun = time() - t0
applymat() = (M0(FBI), M2(q -> v2 * FBI(q)), Mω(FBIω))
applymat()                                 # warm up BLAS/compilation
tmat = minimum(@elapsed(applymat()) for _ in 1:200)
fb0_mat, fb2_mat, fω0_mat = applymat()
@printf("matrix vs functor:  ρ n=0 %.1e   ρ v₂ dev %.1e   ω n=0 %.1e\n",
        maximum(abs.(fb0_mat ./ fb0_fun .- 1)),
        maximum(abs.(fb2_mat ./ fb0_mat .- fb2_fun ./ fb0_fun)),
        maximum(abs.(fω0_mat ./ fω0_fun .- 1)))
@printf("apply time, all %d pT points, 3 spectra:  matrix %.0f µs   functor %.0f ms  (×%.0f)\n",
        length(pTs), 1e6 * tmat, 1e3 * tfun, tfun / tmat)

M0.W

