"""Audit the decay list against the PDG database. Reports only -- writes no .dat.

    pip install pdg
    python3 tools/audit_pdg.py [--importance importance.csv] [-o pdg_audit.md]

For every table entry carrying a standard PDG MC id, compares mass, width, spin
degeneracy (2J+1), isospin and the channel list against `pdg`, and writes a
markdown report ranked by how much each particle actually matters for the pion
spectrum.

Ranking input (optional): a CSV of per-particle leave-one-out shares --
id,name,mass,pi0_low,pi0_mid,pip_low,pip_mid -- where each column is the
fraction of that pion spectrum that disappears when the particle's decays are
switched off (low = 0.33 GeV, mid = 1.0 GeV). Without it the report falls back
to a thermal-abundance estimate, which over-weights heavy states.

Three things the probe established, which is why this is an audit and not an
importer:

  * PDG carries no number for many dominant modes.  rho --> pi pi, the single
    largest pion source in the table, has value None (it is "dominant", quoted
    as ~100% in the listings text).  Renormalising over the modes that *do*
    carry numbers would hand the rho a 50% branch to pi+ pi- gamma.  Any
    automated import must treat "no value" as "do not touch", not as zero.
  * Many modes are generic multiplet items (rho --> "pi pi", a_1 --> "3 pi",
    Sigma(1385) --> "Lambda pi").  Resolving them to charge states needs
    Clebsch-Gordan, which PDG does not supply.  Modes with more than one
    generic item are reported as UNRESOLVED rather than guessed at.
  * a_1(1260) has only generic, value-less modes (3 pi, K K pi, pi gamma), so
    PDG cannot constrain it at all -- the isospin argument used in
    make_isofix_decays.py remains the only available basis for that entry.
"""
import argparse
import csv
import os
from collections import defaultdict

import pdg

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("RESO_REPO", os.path.dirname(HERE))
SRC = os.path.join(REPO, "decays_QM2016Plus_massorder.dat")

PHOTON = 22
MASS_TOL = 0.005          # GeV, absolute
WIDTH_TOL = 0.30          # relative
BR_TOL = 0.05             # absolute, on a renormalised BR


# ------------------------------------------------------------------ table side
def read_table(path):
    parts, chans = {}, defaultdict(list)
    for line in open(path):
        f = line.split()
        if len(f) == 12:
            parts[int(f[0])] = dict(
                id=int(f[0]), name=f[1], mass=float(f[2]), width=float(f[3]),
                deg=float(f[4]), B=int(float(f[6 - 1])), S=int(float(f[7 - 1])),
                twoI=round(2 * float(f[10 - 1])), Q=int(float(f[11 - 1])))
        elif len(f) == 8:
            n = int(f[1])
            chans[int(f[0])].append((float(f[2]),
                                     tuple(sorted(int(f[3 + i]) for i in range(n)))))
    return parts, chans


# -------------------------------------------------------------------- pdg side
class Pdg:
    def __init__(self):
        self.api = pdg.connect()
        self._charge = {}

    def particle(self, mcid):
        try:
            return self.api.get_particle_by_mcid(mcid)
        except Exception:
            return None

    def charge(self, mcid):
        if mcid not in self._charge:
            p = self.particle(mcid)
            self._charge[mcid] = 0 if p is None else int(round(float(p.charge or 0)))
        return self._charge[mcid]

    def modes(self, p, known_ids):
        """(resolved, unresolved, measured_total).

        `exclusive_branching_fractions()` returns PDG's *top-level* partition,
        which for the eta is {neutral modes, charged modes} -- 2gamma and 3pi0
        sit a level below and never appear.  So the full `branching_fractions()`
        list is used instead, and the grouping rows are dropped by requiring
        every decay product to resolve to an explicit mcid: groupings ("neutral
        modes", "K pi", "(K pi)+-") are always generic, real modes are not.
        Nesting attributes do not do this job -- summing over `subdecays`
        double counts the eta to exactly 2.0000.

        measured_total sums the identifiable measured modes whether or not this
        table can represent them, so a caller can tell how much of the width
        PDG actually pins down (rho: 0.011, a_1: 0.000).
        """
        resolved, unresolved, measured_total = [], [], 0.0
        try:
            bfs = list(p.branching_fractions())
        except Exception:
            return resolved, unresolved, measured_total
        for bf in bfs:
            fixed, generic = [], 0
            try:
                prods = list(bf.decay_products)
            except Exception:
                continue
            for dp in prods:
                item = dp.item
                mult = int(getattr(dp, "multiplier", 1) or 1)
                if getattr(item, "has_particle", False):
                    try:
                        mcid = item.particle.mcid
                    except Exception:
                        mcid = None
                    if mcid is None:
                        generic += mult
                    else:
                        fixed.extend([mcid] * mult)
                else:
                    generic += mult
            desc = bf.description
            val = bf.value
            lim = bool(getattr(bf, "is_limit", False))
            if generic:
                unresolved.append((val, lim, desc, generic))
                continue                                  # grouping or multiplet row
            if len(fixed) < 2:
                continue
            if sum(self.charge(d) for d in fixed) != int(round(float(p.charge or 0))):
                continue                                  # sibling charge state
            # identifiable and charge-consistent: it counts towards how much of
            # the width PDG pins down, even if this table cannot carry it.
            if val is not None and not lim:
                measured_total += float(val)
            if len(fixed) > 5:
                continue
            if any(abs(d) < 100 and d != PHOTON for d in fixed):
                continue                                  # leptonic / neutrino
            if any(d not in known_ids for d in fixed):
                continue                                  # daughter not in table
            resolved.append((val, lim, tuple(sorted(fixed)), desc))
        return resolved, unresolved, measured_total


# ------------------------------------------------------------------- reporting
def fmt_ids(ids, parts):
    return " ".join(parts[i]["name"] if i in parts else str(i) for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--importance", default=None)
    ap.add_argument("-o", "--out", default=os.path.join(REPO, "pdg_audit.md"))
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    parts, chans = read_table(SRC)
    known = set(parts)
    db = Pdg()

    imp = {}
    if args.importance and os.path.exists(args.importance):
        for row in csv.DictReader(open(args.importance)):
            imp[int(row["id"])] = max(float(row["pi0_low"]), float(row["pi0_mid"]),
                                      float(row["pip_low"]), float(row["pip_mid"]))

    findings, coverage, usable = [], defaultdict(int), []
    for pid, t in sorted(parts.items()):
        if pid <= 0 or abs(pid) >= 1_000_000:
            coverage["ad-hoc id (no PDG entry)" if abs(pid) >= 1_000_000
                     else "antiparticle (conjugate)"] += 1
            continue
        p = db.particle(pid)
        if p is None:
            coverage["standard id, absent from PDG"] += 1
            continue

        head = []
        pm = float(p.mass) if p.mass is not None else None
        if pm is not None and abs(pm - t["mass"]) > MASS_TOL:
            head.append(f"mass {t['mass']:.4f} -> PDG {pm:.4f}")
        pw = float(p.width) if p.width is not None else None
        if pw is not None and pw > 0 and t["width"] > 0:
            if abs(pw - t["width"]) / max(pw, 1e-12) > WIDTH_TOL:
                head.append(f"width {t['width']:.3e} -> PDG {pw:.3e} "
                            f"({t['width'] / pw:.0f}x)" if pw else "")
        J = getattr(p, "quantum_J", None)
        if J not in (None, "", "?"):
            try:
                deg = 2 * float(eval(str(J).replace("/", "/"))) + 1
                if abs(deg - t["deg"]) > 0.01:
                    head.append(f"degeneracy {t['deg']:.0f} -> 2J+1 = {deg:.0f}")
            except Exception:
                pass
        I = getattr(p, "quantum_I", None)
        if I not in (None, "", "?"):
            try:
                twoI = round(2 * float(eval(str(I))))
                if twoI != t["twoI"]:
                    head.append(f"2I {t['twoI']} -> PDG {twoI}")
            except Exception:
                pass

        tab_modes = chans.get(pid, [])
        selfstable = len(tab_modes) == 1 and tab_modes[0][1] == (pid,)
        resolved, unresolved, numeric_total = db.modes(p, known)
        numeric = [m for m in resolved if m[0] is not None and not m[1]]
        here = sum(m[0] for m in numeric)

        # Renormalising is only meaningful when PDG's measured modes actually
        # account for the width, and when the modes this table can represent are
        # most of that.  Otherwise the tail gets inflated to 100% -- renormalised
        # over its 7 measured modes the rho would get a 91% branch to pi+ pi- gamma.
        trust = (not selfstable and numeric_total >= 0.90
                 and here >= 0.50 * numeric_total)
        if not (resolved or unresolved):
            coverage["PDG modes: none"] += 1
        elif trust:
            coverage["PDG modes: usable (measured, representable)"] += 1
            usable.append(f"{t['name']} (`{pid}`)")
        else:
            coverage["PDG modes: cannot constrain (unmeasured/generic)"] += 1

        ch = []
        if trust:
            tab_by_ids = {ids: br for br, ids in tab_modes}
            for val, _, ids, desc in numeric:
                share = val / here
                if ids not in tab_by_ids:
                    if share > BR_TOL:
                        ch.append(f"MISSING  `{fmt_ids(ids, parts)}` — PDG "
                                  f"{val:.4f}, renormalised {share:.3f}")
                elif abs(tab_by_ids[ids] - share) > BR_TOL:
                    ch.append(f"BR-DIFF  `{fmt_ids(ids, parts)}` — table "
                              f"{tab_by_ids[ids]:.3f} vs PDG {share:.3f}")
            pdg_ids = {m[2] for m in numeric}
            for br, ids in tab_modes:
                if ids not in pdg_ids and br > BR_TOL:
                    ch.append(f"EXTRA    `{fmt_ids(ids, parts)}` — table "
                              f"{br:.3f}, no measured PDG counterpart")
        blocked = []
        if not trust and not selfstable and imp.get(pid, 0) > 0.005:
            blocked = [f"{d} = {'limit' if l else ('no value' if v is None else f'{v:.4g}')}"
                       for v, l, d, _ in unresolved][:3]
            blocked += [f"{d} = no value" for v, l, _, d in resolved
                        if v is None][:3]

        if head or ch or blocked:
            findings.append(dict(id=pid, name=t["name"], mass=t["mass"],
                                 imp=imp.get(pid, 0.0), head=head, ch=ch,
                                 blocked=blocked, cover=numeric_total, trust=trust))

    findings.sort(key=lambda f: -f["imp"])
    with open(args.out, "w") as fh:
        w = fh.write
        w("# PDG audit of `decays_QM2016Plus_massorder.dat`\n\n")
        w(f"PDG edition {db.api.edition}. Report only -- no `.dat` file was modified.\n\n")
        w("`importance` = largest fraction of the pi0 or pi+ spectrum (at 0.33 or "
          "1.0 GeV) that disappears when this particle's decays are switched off.\n\n")
        w("## Coverage\n\n| class | particles |\n|---|---|\n")
        for k, v in sorted(coverage.items(), key=lambda kv: -kv[1]):
            w(f"| {k} | {v} |\n")
        w("\nParticles whose channel list PDG can regenerate on its own: "
          + (", ".join(usable) if usable else "none") + ".\n")
        w("""
## Method, and why most entries cannot be refreshed

A mode is only compared when PDG measures enough of the width to renormalise
against: identifiable, charge-consistent modes must cover >= 90% of it, and the
modes this table can represent must be >= 50% of that. Three structural reasons
block the rest.

1. **No number on the dominant mode.** `rho --> pi pi` is quoted as ~100% in the
   listings text but carries no value in the database; the rho's measured modes
   sum to 1.1%, all of them radiative. Renormalising over what *is* measured
   would give the rho a 91% branch to `pi+ pi- gamma`. Same for `K*(892) --> K pi`,
   `Delta(1232) --> N pi` (0.994, but generic) and `a_1(1260)`, which has no
   numeric mode at all.
2. **Generic multiplet rows.** `pi pi`, `3 pi`, `N pi`, `Lambda pi` name a
   multiplet, not charge states. Splitting them needs Clebsch-Gordan, which PDG
   does not supply -- exactly the gap the a1 and K(1630) patches fill by hand.
3. **Grouping rows.** `exclusive_branching_fractions()` returns PDG's top-level
   partition: for the eta that is {neutral modes, charged modes}, and `2gamma` /
   `3pi0` never appear. Descending to the real modes and dropping the groupings
   is what the explicit-mcid filter in `Pdg.modes` does.

So a PDG refresh is a per-entry judgement, not an import. What the audit is good
for is catching the cases where PDG *does* have the numbers and the table
disagrees.

""")
        w(f"## Findings ({len(findings)} particles)\n\n")
        for f in findings[: args.top]:
            w(f"### {f['name']} (`{f['id']}`), m = {f['mass']:.3f}, "
              f"importance {100 * f['imp']:.1f}%\n\n")
            for h in f["head"]:
                w(f"- **header** {h}\n")
            for c in f["ch"]:
                w(f"- {c}\n")
            if f["blocked"]:
                w(f"- _PDG cannot constrain these branching ratios: measured "
                  f"modes cover only {100 * f['cover']:.1f}% of the width._ "
                  + "; ".join(f"`{b}`" for b in f["blocked"]) + "\n")
            w("\n")
    print(f"wrote {args.out}: {len(findings)} particles with findings")
    for k, v in sorted(coverage.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<45} {v:5d}")


if __name__ == "__main__":
    main()
