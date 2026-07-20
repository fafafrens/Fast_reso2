"""
    LagrangeWeights

Local Lagrange interpolation weights on a uniform grid, for scattering values
at arbitrary points onto grid columns (e.g. assembling a decay-kernel matrix).

    j0, w = lagrangeweights(x, x1, h, N, P)

returns the stencil start index `j0` and the `P` weights `w` such that

    F(x) ≈ sum(w[m] * F[j0 + m - 1] for m in 1:P)

on the grid xᵢ = x1 + (i−1)h, i = 1…N. The stencil is centered on the cell
containing `x` and clamped at the boundaries (one-sided interpolation there).
Error is O(h^P) for smooth F.

Small stencils use hard-coded formulas (P = 2 linear, 3 quadratic, 4 cubic);
larger P falls back to the generic Lagrange product
ℓ_m(ξ) = Π_{k≠m} (ξ−k)/(m−k) on the local coordinate ξ ∈ [0, P−1].
The weights always sum to 1 (a constant is reproduced exactly) and reduce to
(…,0,1,0,…) when `x` is a grid node.
"""
module LagrangeWeights

export lagrangeweights, lagrangederiv, lagrangederiv2, lagrangeboth, lagrangeall

# ---------------------------------------------------------------------------
# weights on the local coordinate ξ (stencil nodes sit at ξ = 0, 1, …, P−1)
# ---------------------------------------------------------------------------

# linear, nodes {0, 1}
lweights(ξ, ::Val{2}) = (1 - ξ, ξ)

# quadratic, nodes {0, 1, 2}
lweights(ξ, ::Val{3}) = ((ξ - 1) * (ξ - 2) / 2,
                         -ξ * (ξ - 2),
                         ξ * (ξ - 1) / 2)

# cubic, nodes {0, 1, 2, 3} (t = ξ − 1 is the position inside the middle cell)
function lweights(ξ, ::Val{4})
    t = ξ - 1
    (-t * (t - 1) * (t - 2) / 6,
     (t^2 - 1) * (t - 2) / 2,
     -t * (t + 1) * (t - 2) / 2,
     t * (t^2 - 1) / 6)
end

# generic fallback for any stencil size: Lagrange product formula
@inline function lweights(ξ, ::Val{P}) where {P}
    ntuple(Val(P)) do m
        w = 1.0
        for k in 0:(P - 1)
            k == m - 1 && continue
            w *= (ξ - k) / (m - 1 - k)
        end
        w
    end
end

@inline function lweights(ξ, P::Integer) 
    ntuple(P) do m
        w = 1.0
        for k in 0:(P - 1)
            k == m - 1 && continue
            w *= (ξ - k) / (m - 1 - k)
        end
        w
    end
end

# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

"""
    lagrangeweights(x, x1, h, N, P::Integer = 4) -> (j0, w::NTuple{P,Float64})

Stencil start `j0` and `P` Lagrange weights for interpolating at `x` on the
uniform grid xᵢ = x1 + (i−1)h with `N` points. Pass `P` as `Val(P)` in hot
loops to keep the tuple length inferable.

"""
lagrangeweights(x, x1, h, N, P::Integer = 4) = lagrangeweights(x, x1, h, N, Val(P))

function lagrangeweights(x, x1, h, N, ::Val{P}) where {P}
    2 ≤ P ≤ N || throw(ArgumentError("need 2 ≤ P ≤ N, got P = $P, N = $N"))
    j = floor(Int, (x - x1) / h) + 1               # cell: x ∈ [x_j, x_{j+1})
    j0 = clamp(j - (P - 2) ÷ 2, 1, N - P + 1)      # centered stencil, clamped
    ξ = (x - (x1 + (j0 - 1) * h)) / h              # local coordinate in [0, P−1]
    return j0, lweights(ξ, Val(P))
end

"""
    lagrangeweights(x, xs::AbstractVector, P::Integer = 4) -> (j0, w::NTuple{P,Float64})

Non-uniform-grid version: `xs` are the actual node positions (sorted,
strictly increasing). Same contract: F(x) ≈ Σₘ w[m] F[j0+m−1]. The weights
are the generic Lagrange product on the true nodes,
ℓ_m(x) = Π_{k≠m} (x − xs[k]) / (xs[m] − xs[k]), restricted to the P-point
stencil centered on the cell containing `x` (clamped at the boundaries).
Exact for polynomials of degree < P on any node spacing.
"""
function lagrangeweights(x, xs::AbstractVector, ::Val{P}) where {P}
    N = length(xs)
    2 ≤ P ≤ N || throw(ArgumentError("need 2 ≤ P ≤ N, got P = $P, N = $N"))
    j = clamp(searchsortedlast(xs, x), 1, N - 1)   # cell: x ∈ [xs[j], xs[j+1])
    j0 = clamp(j - (P - 2) ÷ 2, 1, N - P + 1)      # centered stencil, clamped
    w = ntuple(Val(P)) do m
        num, den = 1.0, 1.0
        @inbounds for k in 1:P
            k == m && continue
            num *= x - xs[j0 + k - 1]
            den *= xs[j0 + m - 1] - xs[j0 + k - 1]
        end
        num / den
    end
    return j0, w
end

function lagrangeweights(x, xs::AbstractVector, P::Integer= 4)
    N = length(xs)
    2 ≤ P ≤ N || throw(ArgumentError("need 2 ≤ P ≤ N, got P = $P, N = $N"))
    j = clamp(searchsortedlast(xs, x), 1, N - 1)   # cell: x ∈ [xs[j], xs[j+1])
    j0 = clamp(j - (P - 2) ÷ 2, 1, N - P + 1)      # centered stencil, clamped
    w = ntuple(P) do m
        num, den = 1.0, 1.0
        @inbounds for k in 1:P
            k == m && continue
            num *= x - xs[j0 + k - 1]
            den *= xs[j0 + m - 1] - xs[j0 + k - 1]
        end
        num / den
    end
    return j0, w
end

# ---------------------------------------------------------------------------
# derivative weights: d/dξ of the local Lagrange basis (nodes ξ = 0,…,P−1)
# ---------------------------------------------------------------------------
# dℓ_m/dξ; the public entry points multiply by 1/h to give d/dx on the grid.
# Explicit for P = 2,3,4 (differentiated hardcoded bases), generic otherwise.
# All sum to 0 (the derivative of the reproduced constant is zero).

dlweights(ξ, ::Val{2}) = (-1.0, 1.0)                     # constant slope

dlweights(ξ, ::Val{3}) = (ξ - 1.5, -2ξ + 2, ξ - 0.5)

function dlweights(ξ, ::Val{4})                          # t = ξ − 1 (mid cell)
    t = ξ - 1
    (-(3t^2 - 6t + 2) / 6,
     (3t^2 - 4t - 1) / 2,
     -(3t^2 - 2t - 2) / 2,
     (3t^2 - 1) / 6)
end

# generic: dℓ_m/dξ = Σ_{i≠c} 1/(c−i) · Π_{k≠c,i} (ξ−k)/(c−k),  c = m−1
@inline function dlweights(ξ, ::Val{P}) where {P}
    ntuple(Val(P)) do m
        c = m - 1
        s = 0.0
        for i in 0:(P - 1)
            i == c && continue
            term = 1.0 / (c - i)
            for k in 0:(P - 1)
                (k == c || k == i) && continue
                term *= (ξ - k) / (c - k)
            end
            s += term
        end
        s
    end
end

"""
    lagrangederiv(x, x1, h, N, P::Integer = 4) -> (j0, w::NTuple{P,Float64})

Weights for the derivative of the Lagrange interpolant on the uniform grid
xᵢ = x1 + (i−1)h: F′(x) ≈ Σₘ w[m] F[j0+m−1], exact to O(h^{P−1}) and exact
for polynomials of degree < P. Same stencil placement as `lagrangeweights`.
"""
lagrangederiv(x, x1, h, N, P::Integer = 4) = lagrangederiv(x, x1, h, N, Val(P))

function lagrangederiv(x, x1, h, N, ::Val{P}) where {P}
    2 ≤ P ≤ N || throw(ArgumentError("need 2 ≤ P ≤ N, got P = $P, N = $N"))
    j = floor(Int, (x - x1) / h) + 1
    j0 = clamp(j - (P - 2) ÷ 2, 1, N - P + 1)
    ξ = (x - (x1 + (j0 - 1) * h)) / h
    return j0, dlweights(ξ, Val(P)) ./ h                 # chain rule d/dx = (1/h) d/dξ
end

"""
    lagrangederiv(x, xs::AbstractVector, P::Integer = 4) -> (j0, w::NTuple{P,Float64})

Non-uniform-grid derivative weights: `xs` the actual sorted node positions,
F′(x) ≈ Σₘ w[m] F[j0+m−1]. Uses the product-rule form of dℓ_m/dx on the true
nodes (valid at nodes too — no division by x−xs[k]).
"""
lagrangederiv(x, xs::AbstractVector, P::Integer = 4) = lagrangederiv(x, xs, Val(P))

function lagrangederiv(x, xs::AbstractVector, ::Val{P}) where {P}
    N = length(xs)
    2 ≤ P ≤ N || throw(ArgumentError("need 2 ≤ P ≤ N, got P = $P, N = $N"))
    j = clamp(searchsortedlast(xs, x), 1, N - 1)
    j0 = clamp(j - (P - 2) ÷ 2, 1, N - P + 1)
    w = ntuple(Val(P)) do m
        xm = xs[j0 + m - 1]
        s = 0.0
        @inbounds for i in 1:P
            i == m && continue
            term = 1.0 / (xm - xs[j0 + i - 1])
            for k in 1:P
                (k == m || k == i) && continue
                term *= (x - xs[j0 + k - 1]) / (xm - xs[j0 + k - 1])
            end
            s += term
        end
        s
    end
    return j0, w
end

# ---------------------------------------------------------------------------
# second-derivative weights: d²/dξ² of the local Lagrange basis
# ---------------------------------------------------------------------------
# Explicit for P = 2,3,4 (differentiated twice), generic otherwise. All sum
# to 0. P = 2 is identically 0 (a line has no curvature); P = 3 is the
# classic second difference (1, −2, 1).

d2lweights(ξ, ::Val{2}) = (0.0, 0.0)

d2lweights(ξ, ::Val{3}) = (1.0, -2.0, 1.0)

function d2lweights(ξ, ::Val{4})                         # t = ξ − 1 (mid cell)
    t = ξ - 1
    (1 - t, 3t - 2, 1 - 3t, t)
end

# generic: d²ℓ_m/dξ² = Σ_{i≠c} Σ_{j≠c,i} 1/((c−i)(c−j)) Π_{k≠c,i,j} (ξ−k)/(c−k)
@inline function d2lweights(ξ, ::Val{P}) where {P}
    ntuple(Val(P)) do m
        c = m - 1
        s = 0.0
        for i in 0:(P - 1)
            i == c && continue
            for j in 0:(P - 1)
                (j == c || j == i) && continue
                term = 1.0 / ((c - i) * (c - j))
                for k in 0:(P - 1)
                    (k == c || k == i || k == j) && continue
                    term *= (ξ - k) / (c - k)
                end
                s += term
            end
        end
        s
    end
end

"""
    lagrangederiv2(x, x1, h, N, P::Integer = 4) -> (j0, w::NTuple{P,Float64})

Weights for the second derivative of the Lagrange interpolant on the uniform
grid: F″(x) ≈ Σₘ w[m] F[j0+m−1], exact for polynomials of degree < P (needs
P ≥ 3 for a nonzero result). Same stencil placement as `lagrangeweights`.
"""
lagrangederiv2(x, x1, h, N, P::Integer = 4) = lagrangederiv2(x, x1, h, N, Val(P))

function lagrangederiv2(x, x1, h, N, ::Val{P}) where {P}
    2 ≤ P ≤ N || throw(ArgumentError("need 2 ≤ P ≤ N, got P = $P, N = $N"))
    j = floor(Int, (x - x1) / h) + 1
    j0 = clamp(j - (P - 2) ÷ 2, 1, N - P + 1)
    ξ = (x - (x1 + (j0 - 1) * h)) / h
    return j0, d2lweights(ξ, Val(P)) ./ h^2               # d²/dx² = (1/h²) d²/dξ²
end

# non-uniform second-derivative product form: Σ over ordered pairs (i,j)
_nud2(x, xs, j0, ::Val{P}) where {P} = ntuple(Val(P)) do m
    xm = xs[j0 + m - 1]
    s = 0.0
    @inbounds for i in 1:P
        i == m && continue
        for j in 1:P
            (j == m || j == i) && continue
            term = 1.0 / ((xm - xs[j0 + i - 1]) * (xm - xs[j0 + j - 1]))
            for k in 1:P
                (k == m || k == i || k == j) && continue
                term *= (x - xs[j0 + k - 1]) / (xm - xs[j0 + k - 1])
            end
            s += term
        end
    end
    s
end

"""
    lagrangederiv2(x, xs::AbstractVector, P::Integer = 4) -> (j0, w::NTuple{P,Float64})

Non-uniform-grid second-derivative weights on the actual node positions.
"""
lagrangederiv2(x, xs::AbstractVector, P::Integer = 4) = lagrangederiv2(x, xs, Val(P))

function lagrangederiv2(x, xs::AbstractVector, ::Val{P}) where {P}
    N = length(xs)
    2 ≤ P ≤ N || throw(ArgumentError("need 2 ≤ P ≤ N, got P = $P, N = $N"))
    j = clamp(searchsortedlast(xs, x), 1, N - 1)
    j0 = clamp(j - (P - 2) ÷ 2, 1, N - P + 1)
    return j0, _nud2(x, xs, j0, Val(P))
end

# ---------------------------------------------------------------------------
# value + derivative together (shares the stencil placement)
# ---------------------------------------------------------------------------

"""
    lagrangeboth(x, x1, h, N, P::Integer = 4) -> (j0, w, dw)

Value and derivative weights at one point, sharing the stencil placement:
F(x) ≈ Σₘ w[m] F[j0+m−1] and F′(x) ≈ Σₘ dw[m] F[j0+m−1]. Cheaper than a
separate `lagrangeweights` + `lagrangederiv` (one floor/clamp/ξ, not two).
"""
lagrangeboth(x, x1, h, N, P::Integer = 4) = lagrangeboth(x, x1, h, N, Val(P))

function lagrangeboth(x, x1, h, N, ::Val{P}) where {P}
    2 ≤ P ≤ N || throw(ArgumentError("need 2 ≤ P ≤ N, got P = $P, N = $N"))
    j = floor(Int, (x - x1) / h) + 1
    j0 = clamp(j - (P - 2) ÷ 2, 1, N - P + 1)
    ξ = (x - (x1 + (j0 - 1) * h)) / h
    return j0, lweights(ξ, Val(P)), dlweights(ξ, Val(P)) ./ h
end

"""
    lagrangeboth(x, xs::AbstractVector, P::Integer = 4) -> (j0, w, dw)

Non-uniform-grid value and derivative weights, sharing the cell search.
"""
lagrangeboth(x, xs::AbstractVector, P::Integer = 4) = lagrangeboth(x, xs, Val(P))

function lagrangeboth(x, xs::AbstractVector, ::Val{P}) where {P}
    N = length(xs)
    2 ≤ P ≤ N || throw(ArgumentError("need 2 ≤ P ≤ N, got P = $P, N = $N"))
    j = clamp(searchsortedlast(xs, x), 1, N - 1)
    j0 = clamp(j - (P - 2) ÷ 2, 1, N - P + 1)
    w = ntuple(Val(P)) do m
        num, den = 1.0, 1.0
        @inbounds for k in 1:P
            k == m && continue
            num *= x - xs[j0 + k - 1]
            den *= xs[j0 + m - 1] - xs[j0 + k - 1]
        end
        num / den
    end
    dw = ntuple(Val(P)) do m
        xm = xs[j0 + m - 1]
        s = 0.0
        @inbounds for i in 1:P
            i == m && continue
            term = 1.0 / (xm - xs[j0 + i - 1])
            for k in 1:P
                (k == m || k == i) && continue
                term *= (x - xs[j0 + k - 1]) / (xm - xs[j0 + k - 1])
            end
            s += term
        end
        s
    end
    return j0, w, dw
end

# ---------------------------------------------------------------------------
# value + first + second derivative together (shares the stencil placement)
# ---------------------------------------------------------------------------

"""
    lagrangeall(x, x1, h, N, P::Integer = 4) -> (j0, w, dw, d2w)

Value, first- and second-derivative weights at one point on the uniform
grid, sharing the stencil placement: F, F′, F″ ≈ Σₘ (w|dw|d2w)[m] F[j0+m−1].
"""
lagrangeall(x, x1, h, N, P::Integer = 4) = lagrangeall(x, x1, h, N, Val(P))

function lagrangeall(x, x1, h, N, ::Val{P}) where {P}
    2 ≤ P ≤ N || throw(ArgumentError("need 2 ≤ P ≤ N, got P = $P, N = $N"))
    j = floor(Int, (x - x1) / h) + 1
    j0 = clamp(j - (P - 2) ÷ 2, 1, N - P + 1)
    ξ = (x - (x1 + (j0 - 1) * h)) / h
    return j0, lweights(ξ, Val(P)), dlweights(ξ, Val(P)) ./ h, d2lweights(ξ, Val(P)) ./ h^2
end

"""
    lagrangeall(x, xs::AbstractVector, P::Integer = 4) -> (j0, w, dw, d2w)

Non-uniform-grid value, first- and second-derivative weights, sharing the
cell search.
"""
lagrangeall(x, xs::AbstractVector, P::Integer = 4) = lagrangeall(x, xs, Val(P))

function lagrangeall(x, xs::AbstractVector, ::Val{P}) where {P}
    N = length(xs)
    2 ≤ P ≤ N || throw(ArgumentError("need 2 ≤ P ≤ N, got P = $P, N = $N"))
    j = clamp(searchsortedlast(xs, x), 1, N - 1)
    j0 = clamp(j - (P - 2) ÷ 2, 1, N - P + 1)
    w = ntuple(Val(P)) do m
        num, den = 1.0, 1.0
        @inbounds for k in 1:P
            k == m && continue
            num *= x - xs[j0 + k - 1]
            den *= xs[j0 + m - 1] - xs[j0 + k - 1]
        end
        num / den
    end
    dw = ntuple(Val(P)) do m
        xm = xs[j0 + m - 1]
        s = 0.0
        @inbounds for i in 1:P
            i == m && continue
            term = 1.0 / (xm - xs[j0 + i - 1])
            for k in 1:P
                (k == m || k == i) && continue
                term *= (x - xs[j0 + k - 1]) / (xm - xs[j0 + k - 1])
            end
            s += term
        end
        s
    end
    return j0, w, dw, _nud2(x, xs, j0, Val(P))
end

end # module
