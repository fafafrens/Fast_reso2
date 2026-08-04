"""PDG -> decay-table extraction.

Reads the PDG Python API and reports which decay channels can be taken from it,
which must be supplied by hand, and why. It does not write tables; it produces
channels plus a coverage number, and the caller decides.

The central quantity is *coverage*: the fraction of a parent's width that the
retained channels account for. Coverage near 1 means the retained rows are a
complete decomposition and may be renormalized. Coverage well below 1 means PDG
did not quantify the dominant mode, and renormalizing would invent branching
ratios that pass every downstream check. See `channels`.

Requires the `pdg` package (`pip install pdg`); everything else is stdlib.
"""
from fractions import Fraction as F
from math import factorial as fac
import itertools

NEUTRINOS = {12, 14, 16}
CHARGED_LEPTONS = {11, 13, 15}
PHOTON = 22

# quark digit -> (slot in the flavour tuple, sign carried by the quark).
# Top is absent on purpose: it decays before it hadronizes, so no top hadrons
# exist and the decay-table format has no topness column.
FLAVOURS = {3: (1, -1),    # s -> strangeness -1
            4: (2, +1),    # c -> charm       +1
            5: (3, -1)}    # b -> bottomness  -1


# --------------------------------------------------------------- quantum numbers
def flavour(mcid):
    """(B, S, C, Bot) decoded from the Monte-Carlo ID digits."""
    d = str(abs(mcid)).zfill(4)
    sign = 1 if mcid > 0 else -1
    q1, q2, q3 = int(d[-4]), int(d[-3]), int(d[-2])
    out = [0, 0, 0, 0]
    if q1:                                      # baryon: three quark digits
        out[0] = sign
        for digit in (q1, q2, q3):
            if digit in FLAVOURS:
                slot, carried = FLAVOURS[digit]
                out[slot] += carried * sign
        return tuple(out)
    # Meson: q2 is the heavier flavour. For a positive ID it is a quark when
    # up-type (u, c, t) and an antiquark when down-type (d, s, b); q3 is the
    # opposite. Getting this backwards silently mislabels every strange meson.
    k2 = 1 if q2 in (2, 4, 6) else -1           # +1 quark, -1 antiquark
    for digit, k in ((q2, k2), (q3, -k2)):
        if digit in FLAVOURS:
            slot, carried = FLAVOURS[digit]
            out[slot] += carried * k * sign
    return tuple(out)


def interaction(parent, daughters):
    """'strong' | 'em' | 'weak' for one channel.

    Only neutrinos force `weak`; a charged-lepton pair is a virtual photon and
    so electromagnetic (rho0 -> e+ e- is EM, not weak).
    """
    if any(abs(m) in NEUTRINOS for m in daughters):
        return "weak"
    if tuple(sum(flavour(m)[i] for m in daughters) for i in range(4)) != flavour(parent):
        return "weak"                           # a flavour quantum number changed
    if any(m == PHOTON or abs(m) in CHARGED_LEPTONS for m in daughters):
        return "em"
    return "strong"


# ------------------------------------------------------------- Clebsch-Gordan
def cg2(j1, m1, j2, m2, J, M):
    """|<j1 m1 j2 m2 | J M>|^2, exact, via the Racah formula."""
    if m1 + m2 != M or not abs(j1 - j2) <= J <= j1 + j2:
        return F(0)
    if any(abs(m) > j for j, m in ((j1, m1), (j2, m2), (J, M))):
        return F(0)
    prefactor = (F(2 * J + 1)
                 * F(fac(int(J + j1 - j2)) * fac(int(J - j1 + j2)) * fac(int(j1 + j2 - J)),
                     fac(int(j1 + j2 + J + 1)))
                 * F(fac(int(J + M)) * fac(int(J - M)) * fac(int(j1 - m1))
                     * fac(int(j1 + m1)) * fac(int(j2 - m2)) * fac(int(j2 + m2))))
    total = F(0)
    for k in range(200):
        args = [j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k,
                J - j2 + m1 + k, J - j1 - m2 + k]
        if any(a < 0 or a != int(a) for a in args):
            continue
        denominator = fac(k)
        for a in args:
            denominator *= fac(int(a))
        total += F((-1) ** k, denominator)
    return prefactor * total * total


class Model:
    """PDG lookups plus the isospin bookkeeping, cached per particle."""

    def __init__(self, api):
        self.api = api
        self._charge, self._isospin = {}, {}

    def charge(self, mcid):
        if mcid not in self._charge:
            self._charge[mcid] = round(self.api.get_particle_by_mcid(mcid).charge)
        return self._charge[mcid]

    def isospin(self, mcid):
        """Total isospin I, or None when PDG records it as ambiguous ('0,1')."""
        if mcid not in self._isospin:
            value = self.api.get_particle_by_mcid(mcid).quantum_I
            try:
                self._isospin[mcid] = F(value)
            except (TypeError, ValueError):
                self._isospin[mcid] = None
        return self._isospin[mcid]

    def i3(self, mcid):
        """Isospin projection, from Gell-Mann-Nishijima."""
        baryon, strange, charm, bottom = flavour(mcid)
        return F(self.charge(mcid)) - F(baryon + strange + charm + bottom, 2)

    def conserves(self, parent, daughters):
        if sum(self.charge(m) for m in daughters) != self.charge(parent):
            return False
        return tuple(sum(flavour(m)[i] for m in daughters)
                     for i in range(4)) == flavour(parent)


# ------------------------------------------------------------------- the rules
def reject(bf):
    """Row-level veto. None means the row is usable.

    `is_limit` and `value is None` are independent: a non-limit mode can still
    carry no number, and for rho and Delta that is the dominant mode.
    """
    if list(bf.get_children()): return "group"          # a heading, not a decay
    if bf.is_limit:             return "limit"
    if bf.value is None:        return "no-value"       # qualitative ('seen')
    if bf.value <= 0:           return "non-positive"   # interference term
    if bf.value > 1:            return "relative-only"  # ratios, not fractions
    return None


def concrete(bf, parent, model):
    """Every allowed concrete daughter combination for one row.

    Generic products ('N', 'pi') are expanded through `PdgItem.particles`;
    text, inclusive and particle-list items cannot be resolved.
    """
    slots = []
    for product in bf.decay_products:
        item = product.item
        if item.item_type in ("T", "I", "L"):
            return []
        members = [p.mcid for p in item.particles]
        if not members:
            return []
        slots += [members] * int(product.multiplier or 1)
    return [c for c in itertools.product(*slots) if model.conserves(parent, c)]


def isospin_weights(combinations, parent, model):
    """Relative weights for the charge states of one mode.

    Two-body combinations are weighted by |CG|^2, merged over orderings of the
    same final state (which is why rho+ -> pi+ pi0 comes out at 1, not 1/2).
    Higher multiplicities have no single-CG answer and are split evenly; the
    caller is told via the returned flag.
    """
    I, I3 = model.isospin(parent), model.i3(parent)
    weights, exact = {}, True
    for combination in combinations:
        key = tuple(sorted(combination))
        if len(combination) == 2 and I is not None:
            a, b = combination
            ia, ib = model.isospin(a), model.isospin(b)
            if ia is None or ib is None:
                exact = False
                weights[key] = weights.get(key, F(0)) + F(1, len(combinations))
                continue
            weights[key] = weights.get(key, F(0)) + cg2(ia, model.i3(a),
                                                        ib, model.i3(b), I, I3)
        else:
            exact = False
            weights[key] = weights.get(key, F(0)) + F(1, len(combinations))
    total = sum(weights.values())
    if not total:
        return {}, False
    return {k: v / total for k, v in weights.items()}, exact


# ----------------------------------------------------------------- the driver
def channels(mcid, model, min_coverage=0.9, max_coverage=1.02):
    """Decay channels for one species.

    Returns (channels, report). `channels` maps a sorted daughter tuple to a
    branching ratio normalized over what was retained, or None when the result
    is not trustworthy. `report` always explains the outcome.

    A species whose every channel is weak is terminal in a strong/EM cascade:
    it decays long after freeze-out, so it takes a self-channel and zero width
    rather than a decay table.
    """
    api = model.api
    report = {"reasons": {}, "coverage": 0.0, "sources": [], "approximate": 0}

    def note(reason):
        report["reasons"][reason] = report["reasons"].get(reason, 0) + 1

    kinds, merged = set(), {}
    for bf in api.get_particle_by_mcid(mcid).exclusive_branching_fractions(
            include_subdecays=True):
        reason = reject(bf)
        if reason:
            note(reason)
            continue
        combinations = concrete(bf, mcid, model)
        if not combinations:
            note("unresolvable")
            continue
        combinations = [c for c in combinations
                        if interaction(mcid, c) != "weak"
                        and not any(abs(m) in NEUTRINOS | CHARGED_LEPTONS for m in c)
                        and not all(m == PHOTON for m in c)
                        and 2 <= len(c) <= 4]
        kinds.update(interaction(mcid, c) for c in combinations)
        if not combinations:
            note("filtered")
            continue
        weights, exact = isospin_weights(combinations, mcid, model)
        if not exact:
            report["approximate"] += 1
        for key, weight in weights.items():
            merged[key] = merged.get(key, 0.0) + bf.value * float(weight)

    if merged:
        report["sources"].append("pdg")

    coverage = sum(merged.values())
    report["coverage"] = coverage

    if not merged:
        report["verdict"] = "terminal" if not kinds else "empty"
        return None, report
    if coverage < min_coverage:
        report["verdict"] = "incomplete"          # PDG never quantified the main mode
        return None, report
    if coverage > max_coverage:
        report["verdict"] = "over-unity"          # overlapping or relative rows
        return None, report

    report["verdict"] = "ok"
    return {k: v / coverage for k, v in merged.items()}, report
