#!/usr/bin/env python3
"""Generate the charm+bottom decay-list supplement.

Header (12 fields): id name mass width degeneracy B S C Bot I Q n_channels
Channel (8 fields): id n_daughters BR d1 d2 d3 d4 d5 (unused slots are zero)

The checked-in supplement is generated from the pinned PDG 2026 API. Missing
required particle properties are errors, except for two documented neutral
Sigma_b isospin estimates. By default, numerical non-limit branching fractions
are kept as absolute probabilities: unsupported, unquantified, and limit-only
modes are reported and are not silently redistributed over retained modes.
"""

from __future__ import annotations

import argparse
import math
import os
import stat
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    from pdg.errors import PdgAmbiguousValueError, PdgNoDataError
except ModuleNotFoundError:
    # Keep pure helper and artifact tests runnable without installing the
    # generator-only PDG dependency.
    class PdgNoDataError(Exception):
        pass

    class PdgAmbiguousValueError(Exception):
        pass


PINNED_PDG_VERSION = "2026.0"
PINNED_PDG_EDITION = "2026"
PHOTON = 22
SEPARATORS = {"-->", ",", "+", "X"}
UP_TYPE = {2, 4, 6}
WEAK_WIDTH = 1e-6
WIDTH_CUT = 0.001
WIDTH_WINDOW = 4.0

# The neutral Sigma_b states have PDG entries but no measured mass. Preserve the
# complete isospin multiplets using an explicit charged-partner midpoint. The
# same rule is applied to their widths and to the corresponding antiparticles.
ISOSPIN_PARTNERS = {
    5212: (5112, 5222),
    5214: (5114, 5224),
}


class GenerationError(RuntimeError):
    """The PDG input cannot be converted without inventing silent defaults."""


@dataclass(frozen=True)
class ParticleData:
    id: int
    name: str
    mass: float
    width: float
    degeneracy: int
    baryon: int
    strangeness: int
    charm: int
    bottomness: int
    isospin: float
    charge: int


@dataclass(frozen=True)
class Entry:
    particle: ParticleData
    channels: tuple[tuple[float, tuple[int, ...]], ...]


@dataclass(frozen=True)
class PdgProvenance:
    package_version: str
    edition: str
    data_release: str
    schema_version: str
    license: str
    citation: str


@dataclass
class GenerationReport:
    dropped_modes: Counter[str] = field(default_factory=Counter)
    dropped_examples: dict[str, list[str]] = field(default_factory=dict)
    branching_totals: dict[int, float] = field(default_factory=dict)
    estimates: dict[tuple[int, str], tuple[float, tuple[int, ...]]] = field(
        default_factory=dict
    )
    missing_widths: set[int] = field(default_factory=set)
    unquantified_terminals: set[int] = field(default_factory=set)
    conditional_parents: set[int] = field(default_factory=set)

    def drop(self, reason: str, parent, branching_fraction) -> None:
        self.dropped_modes[reason] += 1
        examples = self.dropped_examples.setdefault(reason, [])
        if len(examples) < 3:
            description = getattr(branching_fraction, "description", "?")
            examples.append(f"{parent.mcid} {parent.name}: {description}")


def quark_digits(mcid: int) -> tuple[list[int], bool]:
    """Return the quark digits and whether the Monte-Carlo ID is a baryon."""
    digits = str(abs(mcid))
    if len(digits) < 3:
        return [], False
    if len(digits) >= 4 and digits[-4] != "0":
        return [int(digits[-4]), int(digits[-3]), int(digits[-2])], True
    return [int(digits[-3]), int(digits[-2])], False


def constituents(mcid: int) -> list[int]:
    """Signed quark flavours, where positive values denote quarks."""
    digits, is_baryon = quark_digits(mcid)
    if not digits:
        return []

    sign = 1 if mcid > 0 else -1
    if is_baryon:
        return [sign * flavour_id for flavour_id in digits]

    heavy_flavour, light_flavour = digits
    particle_content = (
        [heavy_flavour, -light_flavour]
        if heavy_flavour in UP_TYPE
        else [-heavy_flavour, light_flavour]
    )
    return [sign * flavour_id for flavour_id in particle_content]


def flavour(mcid: int) -> tuple[int, int, int, int]:
    """Return (baryon number, strangeness, charm, bottomness)."""
    content = constituents(mcid)
    if not content:
        return 0, 0, 0, 0

    _, is_baryon = quark_digits(mcid)
    baryon = (1 if mcid > 0 else -1) if is_baryon else 0
    strangeness = sum(
        -1 if flavour_id == 3 else 1 if flavour_id == -3 else 0
        for flavour_id in content
    )
    charm = sum(
        1 if flavour_id == 4 else -1 if flavour_id == -4 else 0
        for flavour_id in content
    )
    bottomness = sum(
        -1 if flavour_id == 5 else 1 if flavour_id == -5 else 0
        for flavour_id in content
    )
    return baryon, strangeness, charm, bottomness


def heavy(mcid: int) -> bool:
    return any(abs(flavour_id) in (4, 5) for flavour_id in constituents(mcid))


def number(value) -> float:
    """Parse PDG integer/fraction strings without silently accepting bad data."""
    if value is None:
        raise GenerationError("required numerical value is missing")
    try:
        result = float(Fraction(str(value).strip()))
    except (ValueError, ZeroDivisionError) as error:
        raise GenerationError(f"invalid numerical value {value!r}") from error
    if not math.isfinite(result):
        raise GenerationError(f"non-finite numerical value {value!r}")
    return result


def _required_float(particle, attribute: str) -> float:
    try:
        raw_value = getattr(particle, attribute)
        result = float(raw_value)
    except Exception as error:
        raise GenerationError(
            f"{particle.mcid} {particle.name}: missing or invalid {attribute}"
        ) from error
    if not math.isfinite(result):
        raise GenerationError(
            f"{particle.mcid} {particle.name}: non-finite {attribute} {result}"
        )
    return result


class ParticleProperties:
    """Validated PDG properties with explicit neutral-Sigma_b estimates."""

    def __init__(self, particles: dict[int, object], report: GenerationReport):
        self.particles = particles
        self.report = report
        self._masses: dict[int, float] = {}
        self._widths: dict[int, tuple[float, bool]] = {}
        self._charges: dict[int, int] = {}

    def _partner_average(self, mcid: int, attribute: str) -> float:
        partner_ids = ISOSPIN_PARTNERS[abs(mcid)]
        values = []
        for partner_id in partner_ids:
            try:
                partner = self.particles[partner_id]
            except KeyError as error:
                raise GenerationError(
                    f"{mcid}: missing isospin partner {partner_id}"
                ) from error
            value = _required_float(partner, attribute)
            if value <= 0:
                raise GenerationError(
                    f"{mcid}: isospin partner {partner_id} has invalid "
                    f"{attribute} {value}"
                )
            values.append(value)

        estimate = math.fsum(values) / len(values)
        self.report.estimates[(abs(mcid), attribute)] = (
            estimate,
            partner_ids,
        )
        return estimate

    def mass(self, particle) -> float:
        mcid = int(particle.mcid)
        if mcid in self._masses:
            return self._masses[mcid]

        try:
            mass = _required_float(particle, "mass")
        except GenerationError as error:
            if (
                abs(mcid) not in ISOSPIN_PARTNERS
                or not isinstance(error.__cause__, PdgNoDataError)
            ):
                raise
            mass = self._partner_average(mcid, "mass")

        if mass <= 0:
            raise GenerationError(
                f"{mcid} {particle.name}: mass must be positive, got {mass}"
            )
        self._masses[mcid] = mass
        return mass

    def width_info(self, particle) -> tuple[float, bool]:
        mcid = int(particle.mcid)
        if mcid in self._widths:
            return self._widths[mcid]

        try:
            raw_width = particle.width
        except PdgNoDataError:
            raw_width = None

        if raw_width is None:
            width, measured = 0.0, False
            self.report.missing_widths.add(mcid)
        else:
            try:
                width = float(raw_width)
            except (TypeError, ValueError, OverflowError) as error:
                raise GenerationError(
                    f"{mcid} {particle.name}: invalid width {raw_width!r}"
                ) from error
            if not math.isfinite(width) or width < 0:
                raise GenerationError(
                    f"{mcid} {particle.name}: invalid width {width}"
                )
            measured = width > 0

        has_width_entry = getattr(particle, "has_width_entry", None)
        has_lifetime_entry = getattr(particle, "has_lifetime_entry", None)
        has_width_data = (
            bool(has_width_entry)
            if has_width_entry is not None
            else raw_width is not None
        )
        has_lifetime_data = (
            bool(has_lifetime_entry)
            if has_lifetime_entry is not None
            else False
        )
        if width == 0 and not has_width_data and not has_lifetime_data:
            self.report.missing_widths.add(mcid)

        # PDG represents the unmeasured neutral Sigma_b widths as zero. Use the
        # same explicit isospin estimate as for their missing masses.
        if width == 0 and abs(mcid) in ISOSPIN_PARTNERS:
            width = self._partner_average(mcid, "width")
            measured = False
            self.report.missing_widths.discard(mcid)

        result = (width, measured)
        self._widths[mcid] = result
        return result

    def charge(self, particle) -> int:
        mcid = int(particle.mcid)
        if mcid in self._charges:
            return self._charges[mcid]

        raw_charge = _required_float(particle, "charge")
        charge = round(raw_charge)
        if not math.isclose(raw_charge, charge, abs_tol=1e-9):
            raise GenerationError(
                f"{mcid} {particle.name}: non-integral hadron charge {raw_charge}"
            )
        self._charges[mcid] = charge
        return charge

    def particle_data(self, particle) -> ParticleData:
        mcid = int(particle.mcid)
        spin = number(getattr(particle, "quantum_J", None))
        isospin = number(getattr(particle, "quantum_I", None))
        degeneracy_value = 2 * spin + 1
        degeneracy = round(degeneracy_value)
        if not math.isclose(degeneracy_value, degeneracy, abs_tol=1e-9):
            raise GenerationError(
                f"{mcid} {particle.name}: invalid spin degeneracy "
                f"{degeneracy_value}"
            )

        baryon, strangeness, charm, bottomness = flavour(mcid)
        width, _ = self.width_info(particle)
        name = (particle.name or f"id{mcid}").replace(" ", "")
        return ParticleData(
            id=mcid,
            name=name,
            mass=self.mass(particle),
            width=width,
            degeneracy=degeneracy,
            baryon=baryon,
            strangeness=strangeness,
            charm=charm,
            bottomness=bottomness,
            isospin=isospin,
            charge=self.charge(particle),
        )


def collect_heavy_particles(api) -> dict[int, object]:
    particles: dict[int, object] = {}
    for particle_group in api.get_particles():
        candidates = (
            [particle_group]
            if getattr(particle_group, "mcid", None) is not None
            else list(particle_group)
        )
        for particle in candidates:
            mcid = getattr(particle, "mcid", None)
            if mcid is None or not heavy(mcid):
                continue
            particles.setdefault(int(mcid), particle)
    return particles


def read_main_particles(path: Path) -> dict[int, ParticleData]:
    particles: dict[int, ParticleData] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) == 8:
                continue
            if len(fields) != 12:
                raise GenerationError(
                    f"{path}:{line_number}: expected 12-field header or "
                    "8-field channel"
                )
            mcid = int(fields[0])
            if mcid in particles:
                raise GenerationError(
                    f"{path}:{line_number}: duplicate particle ID {mcid}"
                )
            particle = ParticleData(
                id=mcid,
                name=fields[1],
                mass=float(fields[2]),
                width=float(fields[3]),
                degeneracy=round(float(fields[4])),
                baryon=round(float(fields[5])),
                strangeness=round(float(fields[6])),
                charm=round(float(fields[7])),
                bottomness=round(float(fields[8])),
                isospin=float(fields[9]),
                charge=round(float(fields[10])),
            )
            if (
                not math.isfinite(particle.mass)
                or particle.mass < 0
                or not math.isfinite(particle.width)
                or particle.width < 0
                or particle.degeneracy <= 0
            ):
                raise GenerationError(
                    f"{path}:{line_number}: invalid particle metadata"
                )
            particles[mcid] = particle
    if not particles:
        raise GenerationError(f"{path}: no particle headers found")
    return particles


class HeavyFlavourGenerator:
    def __init__(
        self,
        api,
        main_particles: dict[int, ParticleData],
        *,
        branching_policy: str = "absolute",
    ):
        if branching_policy not in {"absolute", "conditional"}:
            raise ValueError(f"unknown branching policy {branching_policy!r}")
        self.api = api
        self.main_particles = main_particles
        self.branching_policy = branching_policy
        self.report = GenerationReport()
        self.particles = collect_heavy_particles(api)
        self.properties = ParticleProperties(self.particles, self.report)
        self.known_ids = set(main_particles) | set(self.particles)
        self._charge_cache = {
            mcid: particle.charge for mcid, particle in main_particles.items()
        }

        # Resolve every required property before channel construction. This
        # prevents a partly generated file from concealing a bad PDG entry.
        for particle in self.particles.values():
            self.properties.particle_data(particle)
        missing_antiparticles = sorted({
            abs(particle.mcid)
            for particle in self.particles.values()
            if not particle.self_conjugate
            and -particle.mcid not in self.particles
        })
        if missing_antiparticles:
            raise GenerationError(
                "missing antiparticles for MC IDs "
                f"{missing_antiparticles[:8]}"
            )

    def charge_by_id(self, mcid: int) -> int:
        if mcid in self._charge_cache:
            return self._charge_cache[mcid]
        if mcid in self.particles:
            charge = self.properties.charge(self.particles[mcid])
        else:
            try:
                particle = self.api.get_particle_by_mcid(mcid)
            except Exception as error:
                raise GenerationError(
                    f"cannot resolve charge for daughter {mcid}"
                ) from error
            charge = self.properties.charge(particle)
        self._charge_cache[mcid] = charge
        return charge

    def products_of(self, branching_fraction, parent) -> list[int] | None:
        """Resolve a PDG mode to concrete daughter MC IDs."""
        try:
            decay_products = branching_fraction.decay_products
        except (PdgNoDataError, PdgAmbiguousValueError):
            return None

        fixed: list[int] = []
        generic = []
        for decay_product in decay_products:
            item = decay_product.item
            name = getattr(item, "name", "")
            if name in SEPARATORS or not name:
                continue
            multiplier = int(getattr(decay_product, "multiplier", 1) or 1)
            if getattr(item, "has_particle", False):
                try:
                    daughter_id = item.particle.mcid
                except (PdgNoDataError, PdgAmbiguousValueError):
                    return None
                if daughter_id is None:
                    return None
                fixed.extend([int(daughter_id)] * multiplier)
            else:
                generic.append((item, multiplier))

        if len(generic) > 1:
            return None
        if generic:
            item, multiplier = generic[0]
            if multiplier != 1:
                return None
            try:
                candidates = [
                    candidate
                    for candidate in item.particles
                    if getattr(candidate, "mcid", None) is not None
                ]
            except (PdgNoDataError, PdgAmbiguousValueError):
                return None
            needed_charge = self.properties.charge(parent) - sum(
                self.charge_by_id(daughter_id) for daughter_id in fixed
            )
            matches = [
                candidate
                for candidate in candidates
                if self.properties.charge(candidate) == needed_charge
            ]
            if len(matches) != 1:
                return None
            fixed.append(int(matches[0].mcid))
        return fixed or None

    @staticmethod
    def _known_branching_value(branching_fraction) -> float | None:
        if getattr(branching_fraction, "is_limit", False):
            return None
        raw_value = branching_fraction.value
        if raw_value is None:
            return None
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as error:
            raise GenerationError(
                f"invalid branching ratio {raw_value!r} for "
                f"{getattr(branching_fraction, 'description', '?')}"
            ) from error
        if not math.isfinite(value) or not (0 < value <= 1):
            raise GenerationError(
                f"invalid branching ratio {value!r} for "
                f"{getattr(branching_fraction, 'description', '?')}"
            )
        return value

    def resolve_channels(
        self, particle
    ) -> tuple[tuple[float, tuple[int, ...]], ...]:
        branching_fractions = list(particle.exclusive_branching_fractions())
        relative_table = any(
            getattr(branching_fraction, "value_text", None) == "DEFINED AS 1"
            for branching_fraction in branching_fractions
        )

        retained: list[tuple[float, tuple[int, ...]]] = []

        for branching_fraction in branching_fractions:
            if getattr(branching_fraction, "is_limit", False):
                self.report.drop("limit", particle, branching_fraction)
                continue
            if relative_table:
                self.report.drop(
                    "relative-branching-ratio", particle, branching_fraction
                )
                continue

            daughters = self.products_of(branching_fraction, particle)
            if not daughters or not (2 <= len(daughters) <= 4):
                self.report.drop(
                    "unresolved-or-unsupported-products",
                    particle,
                    branching_fraction,
                )
                continue
            if any(
                abs(daughter_id) < 100 and daughter_id != PHOTON
                for daughter_id in daughters
            ):
                self.report.drop("leptonic", particle, branching_fraction)
                continue
            if any(
                daughter_id not in self.known_ids for daughter_id in daughters
            ):
                self.report.drop(
                    "daughter-outside-model", particle, branching_fraction
                )
                continue
            if sum(self.charge_by_id(daughter_id) for daughter_id in daughters) != (
                self.properties.charge(particle)
            ):
                self.report.drop(
                    "charge-mismatched-multiplet", particle, branching_fraction
                )
                continue

            value = self._known_branching_value(branching_fraction)
            if value is None:
                self.report.drop(
                    "unquantified-branching-ratio",
                    particle,
                    branching_fraction,
                )
                continue
            retained.append((value, tuple(daughters)))

        retained_total = math.fsum(value for value, _ in retained)
        self.report.branching_totals[int(particle.mcid)] = retained_total
        retained.sort(key=lambda channel: (channel[1], channel[0]))

        if self.branching_policy == "conditional" and retained_total > 0:
            self.report.conditional_parents.add(int(particle.mcid))
            return tuple(
                (value / retained_total, daughters)
                for value, daughters in retained
            )
        return tuple(retained)

    def is_leaf(self, particle) -> bool:
        width, measured = self.properties.width_info(particle)
        return measured and 0 < width <= WEAK_WIDTH

    def conjugate(self, daughter_id: int) -> int:
        return -daughter_id if -daughter_id in self.known_ids else daughter_id

    def build(self) -> list[Entry]:
        positive_channels = {
            mcid: self.resolve_channels(particle)
            for mcid, particle in self.particles.items()
            if mcid > 0 and not self.is_leaf(particle)
        }

        entries = []
        for mcid, particle in sorted(
            self.particles.items(),
            key=lambda item: (-self.properties.mass(item[1]), item[0]),
        ):
            particle_data = self.properties.particle_data(particle)
            if self.is_leaf(particle):
                channels: tuple[tuple[float, tuple[int, ...]], ...] = ()
            elif mcid > 0:
                channels = positive_channels.get(mcid, ())
            else:
                particle_channels = positive_channels.get(-mcid, ())
                channels = tuple(
                    (
                        branching_ratio,
                        tuple(
                            self.conjugate(daughter_id)
                            for daughter_id in daughters
                        ),
                    )
                    for branching_ratio, daughters in particle_channels
                )

            if not channels:
                channels = ((1.0, (mcid,)),)
                if not self.is_leaf(particle):
                    self.report.unquantified_terminals.add(mcid)
            entries.append(Entry(particle_data, channels))
        return entries


def validate_entries(
    entries: list[Entry], main_particles: dict[int, ParticleData]
) -> None:
    if not entries:
        raise GenerationError("generator produced no particles")
    generated = {entry.particle.id: entry.particle for entry in entries}
    if len(generated) != len(entries):
        raise GenerationError("generated particle IDs are not unique")
    combined = dict(main_particles)
    overlap = set(combined) & set(generated)
    if overlap:
        raise GenerationError(
            f"generated IDs duplicate the main list: {sorted(overlap)[:8]}"
        )
    combined.update(generated)

    for entry in entries:
        parent = entry.particle
        if not math.isfinite(parent.mass) or parent.mass <= 0:
            raise GenerationError(f"{parent.id}: invalid mass {parent.mass}")
        if not math.isfinite(parent.width) or parent.width < 0:
            raise GenerationError(f"{parent.id}: invalid width {parent.width}")
        if parent.degeneracy <= 0:
            raise GenerationError(
                f"{parent.id}: invalid degeneracy {parent.degeneracy}"
            )

        self_channels = [
            channel for channel in entry.channels if len(channel[1]) == 1
        ]
        if self_channels and (
            len(entry.channels) != 1
            or self_channels[0] != (1.0, (parent.id,))
        ):
            raise GenerationError(
                f"{parent.id}: a terminal self-channel must be the only channel"
            )

        nonself_branching_sum = math.fsum(
            branching_ratio
            for branching_ratio, daughter_ids in entry.channels
            if len(daughter_ids) > 1
        )
        if nonself_branching_sum > 1 + 2e-8:
            raise GenerationError(
                f"{parent.id}: retained branching ratios sum to "
                f"{nonself_branching_sum}"
            )

        for branching_ratio, daughter_ids in entry.channels:
            if not math.isfinite(branching_ratio) or not (
                0 < branching_ratio <= 1
            ):
                raise GenerationError(
                    f"{parent.id}: invalid branching ratio {branching_ratio}"
                )
            if len(daughter_ids) == 1:
                if daughter_ids != (parent.id,) or branching_ratio != 1:
                    raise GenerationError(
                        f"{parent.id}: invalid terminal self-channel"
                    )
                continue
            if not 2 <= len(daughter_ids) <= 4:
                raise GenerationError(
                    f"{parent.id}: unsupported daughter count "
                    f"{len(daughter_ids)}"
                )
            try:
                daughters = [combined[daughter_id] for daughter_id in daughter_ids]
            except KeyError as error:
                raise GenerationError(
                    f"{parent.id}: unresolved daughter {error.args[0]}"
                ) from error

            for label in (
                "baryon",
                "strangeness",
                "charm",
                "bottomness",
                "charge",
            ):
                if sum(getattr(daughter, label) for daughter in daughters) != (
                    getattr(parent, label)
                ):
                    raise GenerationError(
                        f"{parent.id} -> {daughter_ids}: {label} is not conserved"
                    )

            threshold = math.fsum(daughter.mass for daughter in daughters) + 1e-6
            open_at_pole = parent.mass > threshold
            open_in_tail = (
                parent.width > WIDTH_CUT
                and parent.mass + WIDTH_WINDOW * parent.width > threshold
            )
            if not open_at_pole and not open_in_tail:
                raise GenerationError(
                    f"{parent.id} -> {daughter_ids}: channel is kinematically "
                    f"closed (parent {parent.mass:g}, threshold {threshold:g})"
                )


def _format_number(value: float) -> str:
    return f"{value:g}"


def render(
    entries: list[Entry],
    report: GenerationReport,
    *,
    provenance: PdgProvenance,
    branching_policy: str,
) -> str:
    def append_id_lines(label: str, particle_ids: list[int]) -> None:
        lines.append(f"# {label}:")
        for start in range(0, len(particle_ids), 12):
            lines.append(
                "#   "
                + ",".join(
                    str(mcid) for mcid in particle_ids[start : start + 12]
                )
            )

    lines = [
        "# Generated by tools/gen_heavy_flavour.py with "
        f"pdg=={provenance.package_version}, edition {provenance.edition}.",
        f"# PDG data release={provenance.data_release}, "
        f"schema={provenance.schema_version}, license={provenance.license}.",
        f"# PDG citation: {provenance.citation}.",
        f"# branching-policy={branching_policy}: limits, relative-only tables, "
        "and unquantified or unsupported modes are omitted.",
    ]
    if branching_policy == "absolute":
        lines.append(
            "# Retained branching fractions are absolute PDG values and are "
            "not renormalized."
        )
    else:
        lines.append(
            "# Retained branching fractions are explicitly conditional and "
            "renormalized per parent."
        )

    for (mcid, attribute), (estimate, partner_ids) in sorted(
        report.estimates.items()
    ):
        lines.append(
            f"# estimate {mcid} {attribute}={estimate:.9g} from charged "
            f"isospin partners {partner_ids[0]},{partner_ids[1]}."
        )
    if report.dropped_modes:
        lines.append(
            "# omitted PDG modes: "
            + ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(report.dropped_modes.items())
            )
        )
    if report.missing_widths:
        append_id_lines(
            "unmeasured widths represented by the sharp-width path",
            sorted(report.missing_widths),
        )
    if report.unquantified_terminals:
        append_id_lines(
            "model-terminal states without quantified retained modes",
            sorted(report.unquantified_terminals),
        )
    incomplete = [
        (mcid, retained)
        for mcid, retained in report.branching_totals.items()
        if retained < 1 - 1e-12
    ]
    if incomplete:
        lines.append("# retained absolute branching-fraction sums below one:")
        branching_fields = [
            f"{mcid}={branching_sum:.6g}"
            for mcid, branching_sum in sorted(incomplete)
        ]
        for start in range(0, len(branching_fields), 8):
            lines.append("#   " + ",".join(branching_fields[start : start + 8]))

    for entry in entries:
        particle = entry.particle
        lines.append(
            "\t".join(
                [
                    str(particle.id),
                    particle.name,
                    f"{particle.mass:.5f}",
                    f"{particle.width:.6g}",
                    str(particle.degeneracy),
                    str(particle.baryon),
                    str(particle.strangeness),
                    str(particle.charm),
                    str(particle.bottomness),
                    _format_number(particle.isospin),
                    str(particle.charge),
                    str(len(entry.channels)),
                ]
            )
        )
        for branching_ratio, daughters in entry.channels:
            slots = list(daughters) + [0] * (5 - len(daughters))
            lines.append(
                "\t".join(
                    [
                        str(particle.id),
                        str(len(daughters)),
                        f"{branching_ratio:.9g}",
                        *(str(daughter_id) for daughter_id in slots),
                    ]
                )
            )
    return "\n".join(lines) + "\n"


def parse_rendered_entries(content: str) -> list[Entry]:
    """Parse rendered output so validation covers its rounded representation."""
    entries: list[Entry] = []
    current_particle: ParticleData | None = None
    current_channels: list[tuple[float, tuple[int, ...]]] = []
    declared_channel_count = 0

    def finish_current() -> None:
        nonlocal current_particle, current_channels
        if current_particle is None:
            return
        if len(current_channels) != declared_channel_count:
            raise GenerationError(
                f"{current_particle.id}: rendered header declares "
                f"{declared_channel_count} channels, found "
                f"{len(current_channels)}"
            )
        entries.append(Entry(current_particle, tuple(current_channels)))
        current_particle = None
        current_channels = []

    for line_number, line in enumerate(content.splitlines(), start=1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) == 12:
            finish_current()
            current_particle = ParticleData(
                id=int(fields[0]),
                name=fields[1],
                mass=float(fields[2]),
                width=float(fields[3]),
                degeneracy=int(fields[4]),
                baryon=int(fields[5]),
                strangeness=int(fields[6]),
                charm=int(fields[7]),
                bottomness=int(fields[8]),
                isospin=float(fields[9]),
                charge=int(fields[10]),
            )
            declared_channel_count = int(fields[11])
        elif len(fields) == 8:
            if current_particle is None:
                raise GenerationError(
                    f"rendered line {line_number}: channel before header"
                )
            parent_id = int(fields[0])
            if parent_id != current_particle.id:
                raise GenerationError(
                    f"rendered line {line_number}: channel parent {parent_id} "
                    f"does not match header {current_particle.id}"
                )
            daughter_count = int(fields[1])
            daughter_slots = tuple(int(value) for value in fields[3:])
            if not 1 <= daughter_count <= 5 or any(
                daughter_slots[index] != 0
                for index in range(daughter_count, 5)
            ):
                raise GenerationError(
                    f"rendered line {line_number}: invalid daughter padding"
                )
            current_channels.append(
                (
                    float(fields[2]),
                    daughter_slots[:daughter_count],
                )
            )
        else:
            raise GenerationError(
                f"rendered line {line_number}: malformed field count"
            )
    finish_current()
    return entries


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        # mkstemp defaults to 0600. The generated physics table is repository
        # data, so install it with the normal world-readable file mode.
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
        if stat.S_IMODE(path.stat().st_mode) != 0o644:
            raise GenerationError(f"{path}: failed to install mode 0644")
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def print_report(entries: list[Entry], report: GenerationReport) -> None:
    decaying = sum(
        not (
            len(entry.channels) == 1
            and entry.channels[0][1] == (entry.particle.id,)
        )
        for entry in entries
    )
    open_charm = sum(entry.particle.charm != 0 for entry in entries)
    open_bottom = sum(entry.particle.bottomness != 0 for entry in entries)
    charm_bottom = sum(
        entry.particle.charm != 0 and entry.particle.bottomness != 0
        for entry in entries
    )
    hidden = sum(
        entry.particle.charm == 0 and entry.particle.bottomness == 0
        for entry in entries
    )
    print(
        f"states: {len(entries)} (open charm {open_charm}, "
        f"open bottom {open_bottom}, charm+bottom overlap {charm_bottom}, "
        f"hidden-flavour {hidden})"
    )
    print(
        f"decaying parents: {decaying}, model terminals: {len(entries) - decaying}"
    )
    print(f"channels: {sum(len(entry.channels) for entry in entries)}")

    if report.estimates:
        for (mcid, attribute), (estimate, partner_ids) in sorted(
            report.estimates.items()
        ):
            print(
                f"estimate: {mcid} {attribute}={estimate:.9g} "
                f"from {partner_ids}"
            )
    if report.dropped_modes:
        summary = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(report.dropped_modes.items())
        )
        print(f"dropped modes: {summary}")
        for reason, examples in sorted(report.dropped_examples.items()):
            for example in examples:
                print(f"  {reason}: {example}")
    if report.unquantified_terminals:
        print(
            "unquantified model terminals: "
            + ",".join(
                str(mcid) for mcid in sorted(report.unquantified_terminals)
            )
        )

    incomplete = [
        (mcid, retained)
        for mcid, retained in report.branching_totals.items()
        if retained < 1 - 1e-12
    ]
    if incomplete:
        worst = sorted(
            incomplete,
            key=lambda item: item[1],
        )[:5]
        details = ", ".join(
            f"{mcid}:ΣBR={retained:.6g}"
            for mcid, retained in worst
        )
        print(
            f"partial absolute BR: {len(incomplete)} parents; "
            f"lowest {details}"
        )


def _required_api_info(api, key: str) -> str:
    try:
        value = str(api.info(key)).strip()
    except Exception as error:
        raise GenerationError(f"PDG metadata {key!r} is unavailable") from error
    if not value or "\n" in value or "\r" in value:
        raise GenerationError(f"PDG metadata {key!r} is invalid")
    return value.replace("<i>", "").replace("</i>", "")


def _connect_pinned_pdg() -> tuple[object, PdgProvenance]:
    try:
        installed_version = version("pdg")
    except PackageNotFoundError as error:
        raise GenerationError(
            "missing PDG dependency; run "
            "`python -m pip install -r tools/requirements-heavy-flavour.txt`"
        ) from error
    if installed_version != PINNED_PDG_VERSION:
        raise GenerationError(
            f"pdg=={PINNED_PDG_VERSION} required, found {installed_version}"
        )

    try:
        import pdg
    except ImportError as error:
        raise GenerationError("could not import the pinned pdg package") from error
    module_version = str(getattr(pdg, "__version__", ""))
    if module_version != installed_version:
        raise GenerationError(
            "imported pdg module version does not match installed package: "
            f"{module_version!r} != {installed_version!r}"
        )
    api = pdg.connect()
    edition = str(getattr(api, "edition", ""))
    if edition != PINNED_PDG_EDITION:
        raise GenerationError(
            f"PDG edition {PINNED_PDG_EDITION} required, found {edition!r}"
        )
    provenance = PdgProvenance(
        package_version=installed_version,
        edition=edition,
        data_release=_required_api_info(api, "data_release"),
        schema_version=_required_api_info(api, "schema_version"),
        license=_required_api_info(api, "license"),
        citation=_required_api_info(api, "citation"),
    )
    return api, provenance


def parse_arguments(arguments=None):
    repository = Path(
        os.environ.get("RESO_REPO", Path(__file__).resolve().parents[1])
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main",
        type=Path,
        default=repository / "data/azyhydro/decays_QM2016Plus_massorder.dat",
        help="base light-flavour decay list",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "data/azyhydro/decays_charm_bottom.dat",
        help="generated supplement",
    )
    parser.add_argument(
        "--branching-policy",
        choices=("absolute", "conditional"),
        default="absolute",
        help=(
            "keep absolute PDG values (default) or normalize retained "
            "absolute modes"
        ),
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--check",
        action="store_true",
        help="fail if the output differs instead of replacing it",
    )
    output_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report without reading or writing the output",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output when its generated contents differ",
    )
    return parser.parse_args(arguments)


def main(arguments=None) -> int:
    options = parse_arguments(arguments)
    api, provenance = _connect_pinned_pdg()
    main_particles = read_main_particles(options.main)
    generator = HeavyFlavourGenerator(
        api,
        main_particles,
        branching_policy=options.branching_policy,
    )
    entries = generator.build()
    validate_entries(entries, main_particles)
    content = render(
        entries,
        generator.report,
        provenance=provenance,
        branching_policy=options.branching_policy,
    )
    validate_entries(parse_rendered_entries(content), main_particles)
    print_report(entries, generator.report)

    if options.dry_run:
        return 0
    if options.check:
        try:
            current = options.output.read_text(encoding="utf-8")
        except FileNotFoundError:
            current = None
        if current != content:
            print(f"{options.output}: generated supplement is stale")
            return 1
        print(f"{options.output}: up to date")
        return 0

    if options.output.exists():
        current = options.output.read_text(encoding="utf-8")
        if current == content:
            print(f"{options.output}: already up to date")
            return 0
        if not options.force:
            raise GenerationError(
                f"{options.output} exists and differs; inspect with --check, "
                "then pass --force to replace it atomically"
            )

    write_atomic(options.output, content)
    print(f"wrote {options.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationError as error:
        raise SystemExit(f"error: {error}") from error
