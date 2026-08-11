# Why the qT interval is split at qT*, and where the nodes land.
#
# Reproduces the qT-integrand of the two-body kernel (Eq. tensor in
# matrix_notes.tex) for ρ → ππ, at one pT below the rest-frame daughter
# momentum p_b (log singularity inside the band: two segments) and one above
# it (singularity outside the band: one segment). The third panel measures what
# the split buys.
#
# Produces figures/quadrature_nodes.png, included in matrix_notes.tex.

using FastGaussQuadrature
using Printf

# --- the rule under discussion, as implemented in src/DecaySpectra.jl -------
# Cosine-mapped Gauss–Legendre: x = mid + half·xᵢ, weights absorb the sinθ
# Jacobian so 1/√ endpoint singularities cancel identically.
function cosnodes(order::Int)
    x, w = gausslegendre(order)
    θ = @. π * (x + 1) / 2
    return (-cos.(θ), @. π / 2 * w * sin(θ))
end

# --- ρ → ππ ----------------------------------------------------------------
const ma, mb, mc = 0.7752, 0.1396, 0.1396
const Eb = (ma^2 + mb^2 - mc^2) / (2ma)
const pb = sqrt(Eb^2 - mb^2)                  # rest-frame daughter momentum
const T  = 0.155
Fa(qT, η, ϕ) = exp(-hypot(ma, qT) * cosh(η) / T)   # boost-invariant source

# Band edges and the split point (Sec. logsing): qT* = −qT⁻ sits inside the
# band exactly when pT < p_b.
function band(pT)
    mbT = hypot(mb, pT)
    qTmin = max(ma * (Eb * pT - pb * mbT) / mb^2, 0.0)
    qTmax = ma * (Eb * pT + pb * mbT) / mb^2
    qTstar = ma * (pb * mbT - Eb * pT) / mb^2
    return qTmin, qTmax, qTstar
end

# The qT-integrand: the inner Δη integral, including the qT and half-width
# factors that the outer rule multiplies.
function integrand(pT, qT, rule)
    C, mbT = ma * Eb, hypot(mb, pT)
    M, Q = hypot(ma, qT) * mbT, qT * pT
    coshmax = (C + Q) / M
    coshmax > 1 || return 0.0
    ηmax = acosh(coshmax)
    coshmin = (C - Q) / M
    ηmin = coshmin > 1 ? acosh(coshmin) : 0.0
    half, mid = (ηmax - ηmin) / 2, (ηmax + ηmin) / 2
    nodes, weights = rule
    s = 0.0
    for i in eachindex(nodes)
        Δη = mid + half * nodes[i]
        D = M * cosh(Δη) - C
        S2 = (Q - D) * (Q + D)
        S2 > 0 || continue
        α = acos(clamp(D / Q, -1, 1))
        s += weights[i] / sqrt(S2) *
             (Fa(qT, Δη, α) + Fa(qT, Δη, -α) + Fa(qT, -Δη, α) + Fa(qT, -Δη, -α))
    end
    return qT * half * s
end

# The outer rule, with the split either honored or suppressed.
function kernel(pT; nq, split::Bool, nθ = 64)
    θrule = cosnodes(nθ)
    qTmin, qTmax, qTstar = band(pT)
    segments = (split && qTmin < qTstar < qTmax) ?
               ((qTmin, qTstar), (qTstar, qTmax)) : ((qTmin, qTmax),)
    nodes, weights = cosnodes(nq)
    total = 0.0
    for (lo, hi) in segments
        lo < hi || continue
        half, mid = (hi - lo) / 2, (hi + lo) / 2
        for i in eachindex(nodes)
            total += weights[i] * half * integrand(pT, mid + half * nodes[i], θrule)
        end
    end
    return total
end

# qT node positions actually used, for overlay.
function qTnodes(pT; nq, split::Bool)
    qTmin, qTmax, qTstar = band(pT)
    segments = (split && qTmin < qTstar < qTmax) ?
               ((qTmin, qTstar), (qTstar, qTmax)) : ((qTmin, qTmax),)
    nodes, _ = cosnodes(nq)
    return [((hi - lo) / 2) * x + (hi + lo) / 2
            for (lo, hi) in segments if lo < hi for x in nodes]
end

const PT_SPLIT = 0.15      # < p_b: qT* inside the band  → two segments
const PT_ONE   = 0.80      # > p_b: qT* ≤ 0              → one segment

@printf("ρ → ππ:  p_b = %.4f GeV\n", pb)
for pT in (PT_SPLIT, PT_ONE)
    qTmin, qTmax, qTstar = band(pT)
    @printf("  pT = %.2f:  band [%.4f, %.4f]  qT* = %+.4f  %s\n",
            pT, qTmin, qTmax, qTstar,
            qTmin < qTstar < qTmax ? "→ two segments" : "→ one segment")
end

# --- convergence: what the split is worth ----------------------------------
const ORDERS = [8, 16, 32, 64, 128, 256]
reference = kernel(PT_SPLIT; nq = 4096, split = true)
err(split) = [abs(kernel(PT_SPLIT; nq, split) / reference - 1) for nq in ORDERS]
err_split, err_one = err(true), err(false)

println("\n nq | split at qT* | single segment")
for (i, nq) in enumerate(ORDERS)
    @printf("%4d | %12.3e | %.3e\n", nq, err_split[i], err_one[i])
end
rate(e) = log2(e[end-1] / e[end])
@printf("\nobserved order: split %.1f, single segment %.1f (error ~ nq^-p)\n",
        rate(err_split), rate(err_one))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
ENV["GKSwstype"] = "100"           # headless GR
using Plots
using LaTeXStrings

# Font sizes are set for three panels at \linewidth in matrix_notes.tex, where
# the 1380 px width is scaled down by ~0.46; on screen they look large.
common = (lw = 2.5, gridalpha = 0.15, foreground_color_legend = nothing,
          background_color = :white, framestyle = :box, tickfontsize = 15,
          guidefontsize = 19, legendfontsize = 14, titlefontsize = 17)
col_curve, col_one, col_two = "#3d3d3d", "#d55181", "#2a78d6"

function integrand_panel(pT, title, xlims; nq = 16, legend = :topright)
    θrule = cosnodes(64)
    qTmin, qTmax, qTstar = band(pT)
    split_here = qTmin < qTstar < qTmax
    # dense sampling, refined near the split so the logarithm is visible
    qs = sort(vcat(range(qTmin, qTmax, length = 1200),
                   split_here ?
                       qTstar .+ vcat(-exp10.(range(-6, -1.0, length = 200)),
                                       exp10.(range(-6, -1.0, length = 200))) :
                       Float64[]))
    qs = filter(q -> qTmin < q < qTmax, qs)
    ys = [integrand(pT, q, θrule) for q in qs]
    ymax = maximum(ys[qs .≥ xlims[1]])

    p = plot(qs, ys; color = col_curve, label = "", legend, xlims,
             ylims = (-0.10ymax, 1.12ymax), xlabel = L"q_T\ \mathrm{[GeV]}",
             ylabel = L"q_T\ \mathrm{integrand}", title, common...)
    split_here && vline!(p, [qTstar]; color = :black, ls = :dot, lw = 1.5,
                         label = L"q_T^{*}", common...)
    # node rugs below the curve; markers continue past the axis limit
    rows = split_here ?
        ((qTnodes(pT; nq, split = false), col_one, "single segment", :circle, -0.035),
         (qTnodes(pT; nq, split = true),  col_two, "split at "*L"q_T^{*}",
          :utriangle, -0.075)) :
        ((qTnodes(pT; nq, split = false), col_two, "one segment suffices",
          :circle, -0.035),)
    for (nodes, color, lab, shape, frac) in rows
        scatter!(p, nodes, fill(frac * ymax, length(nodes)); color, label = lab,
                 marker = shape, markersize = 4, markerstrokewidth = 0.4, common...)
    end
    return p
end

p1 = integrand_panel(PT_SPLIT,
        L"p_T < p_b:\ \ q_T^{*}\ \mathrm{inside\ the\ band}", (0.0, 1.6))
p2 = integrand_panel(PT_ONE,
        L"p_T > p_b:\ \ q_T^{*} \leq 0", (0.6539, 4.0))

p3 = plot(ORDERS, err_one; color = col_one, marker = :circle, markersize = 4,
          xscale = :log10, yscale = :log10, label = "single segment",
          xlabel = L"n_q", ylabel = L"|K_{n_q}/K_{\mathrm{ref}} - 1|",
          title = "convergence at "*L"p_T = 0.15\ \mathrm{GeV}",
          legend = :bottomleft, common...)
plot!(p3, ORDERS, err_split; color = col_two, marker = :utriangle, markersize = 5,
      label = "split at "*L"q_T^{*}", common...)
plot!(p3, ORDERS, err_split[2] .* (ORDERS ./ ORDERS[2]) .^ (-4.0);
      color = :gray, ls = :dash, lw = 1.5, label = L"\propto n_q^{-4}", common...)

fig = plot(p1, p2, p3; layout = (1, 3), size = (1380, 470),
           left_margin = 9Plots.mm, bottom_margin = 11Plots.mm,
           top_margin = 4Plots.mm, right_margin = 5Plots.mm)
out = joinpath(@__DIR__, "quadrature_nodes.png")
savefig(fig, out)
println("\nplot saved to ", out)
