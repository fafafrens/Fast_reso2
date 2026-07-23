"""Generate a charm+bottom decay-list supplement in the hydro-code format.

Header (12 fields):  id name mass width degeneracy B S C Bot I Q n_channels
Channel (8 fields):  id n_daughters BR d1 d2 d3 d4 d5     (unused slots 0)

Flavour quantum numbers are derived from the PDG Monte-Carlo id's quark digits;
masses/widths/J/I/charge come from the PDG 2026 database.
"""
import os
import pdg
from fractions import Fraction

api = pdg.connect()

# paths relative to the repo root (this script lives in tools/); override with
# RESO_REPO. Needs the PDG package: `pip install pdg`.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("RESO_REPO", os.path.dirname(HERE))
MAIN = os.path.join(REPO, "decays_QM2016Plus_massorder.dat")
OUT = os.path.join(REPO, "decays_charm_bottom.dat")

# ---------------------------------------------------------------- quark content
UP_TYPE = {2, 4, 6}          # u, c, t


def quark_digits(mcid):
    """(digits, is_baryon) from the PDG MC numbering scheme."""
    s = str(abs(mcid))
    if len(s) < 3:
        return [], False
    if len(s) >= 4 and s[-4] != '0':          # 3 quark digits -> baryon
        return [int(s[-4]), int(s[-3]), int(s[-2])], True
    return [int(s[-3]), int(s[-2])], False    # 2 quark digits -> meson


def constituents(mcid):
    """Signed quark flavours: +f = quark, -f = antiquark."""
    d, isbar = quark_digits(mcid)
    if not d:
        return []
    sgn = 1 if mcid > 0 else -1
    if isbar:
        return [sgn * f for f in d]           # baryon: all quarks (anti- if mcid<0)
    a, b = d                                  # meson, a >= b
    # positive code: heavier quark is a quark if up-type, else an antiquark
    q = [a, -b] if a in UP_TYPE else [-a, b]
    return [sgn * x for x in q]


def flavour(mcid):
    """(B, S, C, Bot) from quark content. s:S=-1, c:C=+1, b:Bot=-1 (antiquarks flip)."""
    c = constituents(mcid)
    if not c:
        return 0, 0, 0, 0
    _, isbar = quark_digits(mcid)
    B = (1 if mcid > 0 else -1) if isbar else 0
    S = sum(-1 if f == 3 else (1 if f == -3 else 0) for f in c)
    C = sum(1 if f == 4 else (-1 if f == -4 else 0) for f in c)
    Bt = sum(-1 if f == 5 else (1 if f == -5 else 0) for f in c)
    return B, S, C, Bt


def heavy(mcid):
    c = constituents(mcid)
    return any(abs(f) in (4, 5) for f in c)


def num(x, default=0.0):
    if x is None:
        return default
    try:
        return float(Fraction(str(x).strip()))
    except Exception:
        return default


# ---------------------------------------------------------------- collect states
states = []
for obj in api.get_particles():
    try:
        it = list(obj)
    except TypeError:
        it = [obj]
    for q in it:
        if getattr(q, 'mcid', None) is None:
            continue
        if not heavy(q.mcid):
            continue
        states.append(q)

# de-duplicate by mcid, and add missing antiparticles by conjugation
by_id = {}
for p in states:
    by_id.setdefault(p.mcid, p)

PHOTON = 22
SEP = {'-->', ',', '+', 'X'}


def width_of(p):
    try:
        return float(p.width or 0.0)
    except Exception:
        return 0.0


def mass_of(p):
    try:
        return float(p.mass or 0.0)
    except Exception:
        return 0.0


def charge_of(p):
    try:
        return int(round(float(p.charge)))
    except Exception:
        return 0


def products_of(bf, parent):
    """Resolve a mode's decay products to mcids, or None if not resolvable.

    Handles: separator tokens, multipliers, and a single generic (multiplet)
    item fixed uniquely by charge conservation (e.g. Sigma_c++ -> Lambda_c+ pi).
    """
    try:
        prods = bf.decay_products
    except Exception:
        return None
    fixed, generic = [], []
    for dp in prods:
        item = dp.item
        nm = getattr(item, 'name', '')
        if nm in SEP or not nm:
            continue
        mult = int(getattr(dp, 'multiplier', 1) or 1)
        if getattr(item, 'has_particle', False):
            try:
                pid = item.particle.mcid
            except Exception:
                return None
            if pid is None:
                return None
            fixed.extend([pid] * mult)
        else:
            generic.append((item, mult))
    if len(generic) > 1:
        return None                       # >1 ambiguous item: needs Clebsch-Gordan
    if generic:
        item, mult = generic[0]
        if mult != 1:
            return None
        try:
            cands = [c for c in item.particles if getattr(c, 'mcid', None) is not None]
        except Exception:
            return None
        need = charge_of(parent) - sum(_charge_by_id(i) for i in fixed)
        hit = [c for c in cands if charge_of(c) == need]
        if len(hit) != 1:
            return None                   # not uniquely fixed by charge
        fixed.append(hit[0].mcid)
    return fixed or None


_charge_cache = {}


def _charge_by_id(mcid):
    if mcid in _charge_cache:
        return _charge_cache[mcid]
    q = 0
    try:
        q = charge_of(api.get_particle_by_mcid(mcid))
    except Exception:
        pass
    _charge_cache[mcid] = q
    return q


# ---------------------------------------------------------------- build entries
WEAK = 1e-6          # width below this => treat as stable leaf (weak decay)

# universe of ids, so charge conjugation knows which daughters are self-conjugate
# (a daughter d conjugates to -d only if -d is a real, distinct particle).
main_ids = set()
for ln in open(MAIN):
    f = ln.split()
    if len(f) == 12:
        main_ids.add(int(f[0]))
known_ids = main_ids | set(by_id)


def conj(d):
    return -d if (-d) in known_ids else d      # self-conjugate daughters stay


def width_info(p):
    """(value, measured?). An unmeasured width (PDG upper limit only) comes back
    as None -> (0.0, False)."""
    try:
        w = p.width
    except Exception:
        return 0.0, False
    return (float(w), True) if w is not None else (0.0, False)


def is_leaf(p):
    """A weakly-decaying ground state — measured, tiny width — is a stable
    cascade end-product. An excited state with an *unmeasured* width is NOT a
    leaf: it still decays (usually narrow EM/ππ), so we resolve its modes and
    let it decay sharply (header width 0)."""
    w, measured = width_info(p)
    return measured and 0.0 < w <= WEAK


def resolve_channels(p):
    """Charge-consistent (particle-convention) strong/EM modes, or None."""
    modes = []
    try:
        bfs = list(p.exclusive_branching_fractions())
    except Exception:
        bfs = []
    for bf in bfs:
        ds = products_of(bf, p)
        if not ds or not (2 <= len(ds) <= 4):
            continue                           # cascade kernels cover 2-4 body
        if any(abs(d) < 100 and d != PHOTON for d in ds):
            continue                           # skip leptonic/neutrino modes
        if any(d not in known_ids for d in ds):
            continue                           # daughter outside the list (K_S/K_L, exotic f0)
        if sum(_charge_by_id(d) for d in ds) != charge_of(p):
            continue                           # multiplet-level mode for a sibling charge state
        modes.append((bf.value, ds))
    if len(modes) == 1:
        return [(1.0, modes[0][1])]            # single mode => BR = 1
    known = [v for v, _ in modes if v is not None]
    if modes and sum(known) > 0:
        tot = sum(known)
        ch = [((v if v is not None else 0.0) / tot, ds) for v, ds in modes]
        return [(b, ds) for b, ds in ch if b > 0]
    return None


# resolve decays for non-leaf particles (mcid > 0); antiparticles conjugate theirs
pos_channels = {m: resolve_channels(p) for m, p in by_id.items()
                if m > 0 and not is_leaf(p)}

entries = []
for mcid, p in sorted(by_id.items(), key=lambda kv: mass_of(kv[1])):
    B, S, C, Bt = flavour(mcid)
    J = num(getattr(p, 'quantum_j', None), 0.0)
    I = num(getattr(p, 'quantum_i', None), 0.0)
    name = (p.name or f"id{mcid}").replace(' ', '')
    if is_leaf(p):
        chans = None                           # weakly-decaying ground state
    elif mcid > 0:
        chans = pos_channels.get(mcid)
    else:
        part = pos_channels.get(-mcid)         # conjugate the particle's channels
        chans = None if part is None else [(br, [conj(d) for d in ds]) for br, ds in part]
    if not chans:
        chans = [(1.0, [mcid])]                # stable leaf: self-channel
    entries.append(dict(id=mcid, name=name, mass=mass_of(p), width=width_of(p),
                        deg=2 * J + 1, B=B, S=S, C=C, Bt=Bt, I=I,
                        Q=charge_of(p), chans=chans))

# ---------------------------------------------------------------- emit
def fmt(x):
    return f"{x:g}"


out = []
for e in sorted(entries, key=lambda e: -e['mass']):     # mass-descending, like the main list
    out.append("\t".join([str(e['id']), e['name'], f"{e['mass']:.5f}", f"{e['width']:.6g}",
                          fmt(e['deg']), str(e['B']), str(e['S']), str(e['C']),
                          str(e['Bt']), fmt(e['I']), str(e['Q']), str(len(e['chans']))]))
    for br, ds in e['chans']:
        slots = list(ds) + [0] * (5 - len(ds))
        out.append("\t".join([str(e['id']), str(len(ds)), f"{br:.6g}"] + [str(x) for x in slots]))

open(OUT, "w").write("\n".join(out) + "\n")

nstrong = sum(1 for e in entries if not (len(e['chans']) == 1 and e['chans'][0][1] == [e['id']]))
ncharm = sum(1 for e in entries if e['C'] != 0)
nbot = sum(1 for e in entries if e['Bt'] != 0)
nhidden = sum(1 for e in entries if e['C'] == 0 and e['Bt'] == 0)
print(f"states: {len(entries)}  (open charm {ncharm}, open bottom {nbot}, hidden-flavour {nhidden})")
print(f"decaying parents: {nstrong},  stable leaves: {len(entries)-nstrong}")
print(f"channels: {sum(len(e['chans']) for e in entries)}")
