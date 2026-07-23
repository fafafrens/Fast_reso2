"""
    ParticleList

Reader for the hydro-code decay list format (`decays_QM2016Plus_massorder.dat`):
per species a 12-field header

    id  name  mass  width  degeneracy  B  S  C  Bot  I  Q  n_channels

followed by `n_channels` 8-field lines

    id  n_daughters  BR  d1 d2 d3 d4 d5      (unused slots are 0)

Stable particles carry one self-channel (n_daughters = 1, daughter = id).
"""
module ParticleList

export ParticleInfo, DecayChannel, read_decaylist,
       isospin, isbaryon, ismeson, isantiparticle, isstrange

struct ParticleInfo
    id::Int
    name::String
    mass::Float64
    width::Float64
    degeneracy::Float64
    baryon::Int
    charge::Int
    strangeness::Int      # S  (field 7)
    charm::Int            # C  (field 8) — all zero in the QM2016 list
    twiceisospin::Int     # 2I (field 10 × 2): integer 0,1,2,3 for I = 0,½,1,1½
end

# 7-arg form (pre-quantum-number callers, tests) defaults S = C = 0, 2I = 0.
ParticleInfo(id, name, mass, width, deg, baryon, charge) =
    ParticleInfo(id, name, mass, width, deg, baryon, charge, 0, 0, 0)

# --- derived quantities (exact from stored fields) -------------------------
"Physical isospin I (half-integer), recovered from the stored 2I."
isospin(p::ParticleInfo)        = p.twiceisospin / 2
"True for a baryon (nonzero baryon number)."
isbaryon(p::ParticleInfo)       = p.baryon != 0
"True for a meson (zero baryon number)."
ismeson(p::ParticleInfo)        = p.baryon == 0
"True for an antiparticle (negative id in this list's convention)."
isantiparticle(p::ParticleInfo) = p.id < 0
"True for a strange hadron (nonzero strangeness)."
isstrange(p::ParticleInfo)      = p.strangeness != 0

struct DecayChannel
    parent::Int
    br::Float64
    daughters::Vector{Int}
end

"""
    read_decaylist(path) -> (particles, byparent)

`particles :: Dict{Int,ParticleInfo}` keyed by id, and
`byparent :: Dict{Int,Vector{DecayChannel}}` — every particle's channels
grouped under its id (absent key = no channel lines for that id).
"""
function read_decaylist(path::AbstractString)
    particles = Dict{Int,ParticleInfo}()
    byparent = Dict{Int,Vector{DecayChannel}}()
    for ln in eachline(path)
        s = split(ln)                          # whitespace/tab tolerant
        (isempty(s) || startswith(s[1], "#")) && continue
        # The two line types are distinguished purely by field count: headers
        # have 12 fields, channel lines always 8 (5 daughter slots, 0-padded).
        # Safe because particle names contain no whitespace in this format.
        if length(s) == 12
            # header: id name mass width degeneracy B S C Bot I Q n_channels
            #         1  2    3    4     5          6 7 8 9   10 11 12
            # (Bot (9) and the redundant trailing count (12) are skipped;
            #  channels carry their own parent id, so 12 is never needed)
            id = parse(Int, s[1])
            particles[id] = ParticleInfo(id, String(s[2]),
                parse(Float64, s[3]), parse(Float64, s[4]), parse(Float64, s[5]),
                round(Int, parse(Float64, s[6])), round(Int, parse(Float64, s[11])),
                round(Int, parse(Float64, s[7])), round(Int, parse(Float64, s[8])),
                round(Int, 2 * parse(Float64, s[10])))     # store 2I as an integer
        elseif length(s) == 8
            # channel: id n_daughters BR d1 d2 d3 d4 d5
            # only the first n_daughters slots are meaningful, the rest are 0.
            # Stable particles appear as a self-channel (n_daughters = 1,
            # d1 = id, BR = 1) — kept as-is; the cascade skips them by nd == 1.
            parent = parse(Int, s[1])
            nd = parse(Int, s[2])
            br = parse(Float64, s[3])
            ds = [parse(Int, s[3 + k]) for k in 1:nd]
            push!(get!(byparent, parent, DecayChannel[]), DecayChannel(parent, br, ds))
        else
            # fail loudly: a silently mis-split line would corrupt the cascade
            error("unrecognized line ($(length(s)) fields): $ln")
        end
    end
    return particles, byparent
end

end # module
