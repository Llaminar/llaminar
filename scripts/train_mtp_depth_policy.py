#!/usr/bin/env python3
"""Fit the generated MTP depth policy from measured fixed-depth economics.

The single trainer consumes complete same-prompt fixed-depth controls, derives
measured acceptable target sets in the authored economic-equivalence band,
and fits minimum contiguous acceptance regions. Ties prefer training-only
startup coverage before measured log throughput. Midpoint boundaries and
positive support floors are emitted through the existing six-decimal ABI.
Binary true negatives and a single exact preferred label are not evidence
that a selected target is economical. Exact preferred-label accuracy remains
a separately named diagnostic.

Startup remains a training-only geometric-mean decision; held-out summaries
never choose startup, target regions or cutpoints. Both training and held-out
reports evaluate actual ordered emitted targets, coverage and measured regret.
Real unprofiled learned-Dynamic validation remains required before promotion.

Explicit controller_* evidence pairs unchanged unprofiled fixed-depth costs
with an untimed, unflushed Observe window. Its eligible counters exclude clipped
transactions exactly as the live controller does. Physical acceptance_pct and
accepted/drafted fields remain separate work statistics. Incomplete controller
evidence is fatal; it must never silently select the legacy aggregate feature.
Positive learned moves additionally require an observed full accepted prefix in
that same eligible window. The fitter and ordered evaluator enforce the shared
runtime guard. Legacy percentage-only rows retain unknown full-prefix evidence:
they may support holds or demotions, but cannot certify a promotion.

New complete scopes bind physical verifier strategy, uniform TP/PP, capacity,
loaded model geometry and frozen hardware architecture. Every unmeasured scope,
including installed generic policy, can be preserved without recertifying it.
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import math
import re
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable


DEPTH_VARIANTS = {f"fixed_d{depth}": depth for depth in range(1, 16)}


SCOPE_COLUMNS = ("verifier_width_policy", "tp_width", "pp_stages", "retained_depth",
                 "hidden_dim", "main_layers", "expert_count", "experts_per_token",
                 "compute_capability_major", "compute_capability_minor")
WIDTH_POLICIES = {
    "bounded_logical_bucket": "BoundedLogicalBucket",
    "dynamic_device_envelope": "DynamicDeviceEnvelope",
    "exact_logical_rows": "ExactLogicalRows",
}


@dataclass(frozen=True, order=True)
class PolicyScope:
    """Complete immutable runtime economics, or the preserved generic domain."""

    verifier_width_policy: str = ""
    tp_width: int = 0
    pp_stages: int = 0
    retained_depth: int = 0
    hidden_dim: int = 0
    main_layers: int = 0
    expert_count: int = 0
    experts_per_token: int = 0

    compute_capability_major: int = 0
    compute_capability_minor: int = 0

    def valid(self) -> bool:
        """Reject partial scopes rather than widening a measured overlay."""
        if not self.verifier_width_policy:
            return self == PolicyScope()
        return (self.verifier_width_policy in WIDTH_POLICIES and
                self.tp_width > 0 and self.pp_stages > 0 and 1 <= self.retained_depth <= 15 and
                self.hidden_dim > 0 and self.main_layers > 0 and
                self.compute_capability_major > 0 and self.compute_capability_minor >= 0 and
                ((self.expert_count == 0 and self.experts_per_token == 0) or
                 0 < self.experts_per_token <= self.expert_count))

    def group_key(self) -> tuple[str, ...]:
        """Keep economics and train/holdout grouping on the same frozen identity."""
        return tuple(f"{name}={getattr(self, name)}" for name in SCOPE_COLUMNS)


def policy_domain(row) -> tuple[str, str, str, PolicyScope]:
    """Whole policy domain; preserving an unmeasured scope never erases another."""
    return row.backend, row.model_class, row.mode, row.scope


def scope_from_summary(row: dict[str, str]) -> PolicyScope:
    """Accept legacy all-absent scopes; require authoritative complete new scopes."""
    present = [bool(row.get(name, "")) for name in SCOPE_COLUMNS]
    if not any(present):
        return PolicyScope()
    if not all(present):
        raise ValueError("partial learned-policy scope in benchmark summary")
    if row.get("model_class") not in ("dense", "moe"):
        raise ValueError("scoped measurements require explicit loaded model_class")
    if _backend_from_device(row.get("device", "")) == "any":
        raise ValueError("scoped measurements require a known actual backend")
    try:
        scope = PolicyScope(row[SCOPE_COLUMNS[0]], *(int(row[name]) for name in SCOPE_COLUMNS[1:]))
    except ValueError as error:
        raise ValueError("invalid integer learned-policy scope") from error
    if not scope.valid():
        raise ValueError("invalid complete learned-policy scope")
    if (scope.expert_count > 0) != (row["model_class"] == "moe"):
        raise ValueError("loaded model_class disagrees with scoped expert geometry")
    return scope


@dataclass(frozen=True)
class FixedDepthRow:
    """One fixed-depth benchmark row used as a training example."""

    group_key: tuple[str, ...]
    backend: str
    model_class: str
    mode: str
    depth: int
    decode_tps: float
    acceptance_rate: float
    held_out: bool | None = None
    scope: PolicyScope = PolicyScope()
    acceptance_source: str = "physical_aggregate"
    full_accepts_lower_bound: int | None = None  # None is unknown evidence, never a measured zero.


@dataclass(frozen=True)
class LabeledExample:
    """A fixed-depth row with its optimal action label for that lane."""

    backend: str
    model_class: str
    mode: str
    depth: int
    target_depth: int
    acceptance_rate: float
    action: str
    group_key: tuple[str, ...]
    held_out: bool | None = None
    scope: PolicyScope = PolicyScope()
    relative_tps: tuple[tuple[int, float], ...] = ()
    startup_depth: int | None = None
    full_accepts_lower_bound: int | None = None


@dataclass(frozen=True)
class LearnedRule:
    """A generated C++ rule and its training support."""

    backend: str
    model_class: str
    mode: str
    depth: int
    target_depth: int
    min_acceptance: float
    max_acceptance: float
    max_zero_accept: float
    min_full_accept: float
    delta: int
    label: str
    train_correct: int
    train_total: int
    holdout_correct: int
    holdout_total: int
    scope: PolicyScope = PolicyScope()
    positive_support: int = 0


@dataclass(frozen=True)
class LearnedStartup:
    """One domain's measured admission choice; it does not constrain adaptation."""

    backend: str
    model_class: str
    mode: str
    depth: int
    train_groups: int
    relative_throughput: float
    scope: PolicyScope = PolicyScope()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Benchmark summary.tsv. May be supplied multiple times.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument(
        "--holdout-input", action="append", type=Path, default=[],
        help="Independent holdout summaries. With these, --input is training-only.",
    )
    parser.add_argument(
        "--base-policy", type=Path,
        help="Preserve every unmeasured backend/model/mode/complete-scope domain, including legacy generic rows.",
    )
    parser.add_argument(
        "--holdout-modulus",
        type=int,
        default=4,
        help=(
            "Stable group hash modulus; one bucket is holdout. Set to zero "
            "to train on the complete corpus without a holdout split."
        ),
    )
    parser.add_argument(
        "--holdout-bucket",
        type=int,
        default=0,
        help="Stable group hash bucket reserved for holdout.",
    )
    parser.add_argument(
        "--min-holdout-accuracy",
        type=float,
        default=0.0,
        help="Minimum economic admissibility of actually selected holdout targets per observed region; no binary true negatives.",
    )
    parser.add_argument("--min-ordered-holdout-coverage", type=float, default=0.0,
                        help="Minimum fraction of holdout windows matched by the whole ordered policy.")
    parser.add_argument("--min-ordered-holdout-accuracy", type=float, default=0.0,
                        help="Minimum economically admissible selected-target fraction over ALL holdout observations (misses fail); exact preferred labels are diagnostic.")
    parser.add_argument("--max-ordered-holdout-regret", type=float,
                        help="Maximum matched holdout TPS loss versus its measured best fixed depth.")
    parser.add_argument(
        "--min-train-accuracy",
        type=float,
        default=0.75,
        help=(
            "Minimum selected-target economic admissibility on training support; "
            "every fitted region must in addition satisfy its per-example economic loss bound."
        ),
    )
    parser.add_argument(
        "--hold-acceptance-margin",
        type=float,
        default=0.05,
        help=(
            "Expand only the lowest hold support floor by this margin. If expansion "
            "reaches serialized zero, retain the measured positive floor; never alter internal partition cuts."
        ),
    )
    parser.add_argument(
        "--startup-tie-tolerance", type=float, default=0.05,
        help="Economic equivalence band for measured targets and startup tie preference; zero requires the exact throughput winner.",
    )
    return parser.parse_args()


def _to_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _group_key(
    row: dict[str, str],
    source_id: int | None = None,
) -> tuple[str, ...]:
    """Build a stable same-run lane key while ignoring metrics and variants.

    Fixed d1/d2/d3 rows are comparable only inside one benchmark lane.  The
    deterministic source ordinal keeps two summaries with the same
    device/model/mode from overwriting each other, while the lane fields keep
    multi-topology and multi-request-batch rows separated within one summary.
    Filesystem paths are deliberately excluded: relocating an identical corpus
    must not reshuffle train and holdout domains.
    """

    preferred = [
        key
        for key in (
            "topology",
            "device",
            "model",
            "mode",
            "case",
            "decode_tokens",
            "request_batch",
        )
        if key in row
    ]
    if preferred:
        parts: list[str] = []
        if source_id is not None:
            parts.append(f"source_id={source_id}")
        parts.extend(f"{key}={row.get(key, '')}" for key in preferred)
        return tuple(parts)
    fallback_parts = (
        [f"source_id={source_id}"]
        if source_id is not None
        else []
    )
    fallback_parts.extend(
        row[key]
        for key in sorted(row)
        if not key.startswith("controller_") and key
        not in {
            "variant",
            "decode_tps",
            "speedup_vs_baseline",
            "overall_tps",
            "acceptance_pct",
            "accepted",
            "drafted",
            "verifier_runs",
            "rejected",
            "json",
            "perfstats",
        }
    )
    return tuple(fallback_parts)


def _backend_from_device(device: str) -> str:
    """Return the coarse backend class used by the generated C++ policy."""

    normalized = device.lower()
    if normalized.startswith("cuda"):
        return "cuda"
    if normalized.startswith("rocm"):
        return "rocm"
    if normalized.startswith("cpu"):
        return "cpu"
    return "any"


def _model_class_from_summary(model: str) -> str:
    """Return the coarse model class used by the generated C++ policy."""

    normalized = model.replace("_", "-").lower()
    if not normalized:
        return "any"
    if "moe" in normalized:
        return "moe"
    return "dense"


def _canonical_mode(mode: str) -> str:
    """Resolve summary spellings before grouping, replacement, or C++ emission."""
    normalized = mode.replace("_", "-").lower()
    if normalized == "greedy":
        return "greedy"
    if normalized in {"stochastic", "speculative-sampling", "sampling"}:
        return "stochastic"
    raise ValueError(f"unknown MTP verification mode in evidence: {mode!r}")


CONTROLLER_OBSERVATION_COLUMNS = (
    "controller_accepted", "controller_attempted", "controller_runs",
    "controller_full_accepts", "controller_zero_accepts", "controller_rollbacks",
    "controller_observation", "controller_observation_semantics",
    "controller_measurement_iterations",
)


def acceptance_from_summary(row: dict[str, str], depth: int) -> tuple[float, str]:
    """Read explicit eligible-window evidence without altering measured costs.

    The producer owns native identity, output equality and the unflushed-window
    proof. This parser validates the counter algebra and refuses partial or
    mislabeled evidence. Legacy summaries without controller columns retain
    their explicitly reported physical-aggregate feature.
    """
    physical = _to_float(row, "acceptance_pct", float("nan")) / 100.0
    if not math.isfinite(physical) or not 0 <= physical <= 1:
        raise ValueError("invalid fixed-depth physical aggregate acceptance")
    if not any(name.startswith("controller_") for name in row):
        return physical, "physical_aggregate"
    if any(not row.get(name) for name in CONTROLLER_OBSERVATION_COLUMNS):
        raise ValueError("incomplete controller observation evidence")
    if row["controller_observation_semantics"] != "eligible_unflushed_observe_v1":
        raise ValueError("unknown controller observation semantics")
    counts = {}
    for name in (*CONTROLLER_OBSERVATION_COLUMNS[:6], "controller_measurement_iterations"):
        value = row[name]
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError("invalid integer controller observation: " + name)
        counts[name] = int(value)
    runs = counts["controller_runs"]
    attempted = counts["controller_attempted"]
    accepted = counts["controller_accepted"]
    full = counts["controller_full_accepts"]
    zero = counts["controller_zero_accepts"]
    partial = runs - full - zero
    if (not 1 <= depth <= 15 or runs <= 0 or
            counts["controller_measurement_iterations"] <= 0 or
            attempted != runs * depth or not 0 <= accepted <= attempted or
            partial < 0 or counts["controller_rollbacks"] > runs - full or
            not full * depth + partial <= accepted <= full * depth + partial * (depth - 1)):
        raise ValueError("inconsistent controller observation counters")
    return accepted / attempted, "eligible_unflushed_observe_v1"


def full_accepts_lower_bound_from_summary(row: dict[str, str], depth: int, source: str) -> int | None:
    """Retain authoritative eligible full counts without reinterpreting legacy rates.

    Eligible Observe counts were already checked by acceptance_from_summary.
    Legacy physical counts do not establish the same eligible observation
    window. Their promotion evidence stays unknown, never a fabricated zero.
    """
    if source == "eligible_unflushed_observe_v1":
        return int(row["controller_full_accepts"])
    return None


def promotion_supported(target: int, example: LabeledExample) -> bool:
    """Mirror the native per-row positive-delta guard without fabricating counts."""
    if target <= example.depth:
        return True
    count = example.full_accepts_lower_bound
    if count is None:
        return False
    if type(count) is not int or count < 0:
        raise ValueError("invalid full-prefix promotion evidence")
    return count > 0


def validate_acceptance_sources(rows: Iterable[FixedDepthRow]) -> None:
    """One learned domain cannot mix physical and controller-eligible features."""
    sources = {}
    for row in rows:
        sources.setdefault(policy_domain(row), set()).add(row.acceptance_source)
    if any(len(observed) != 1 for observed in sources.values()):
        raise ValueError("mixed acceptance observation semantics in one learned domain")


def load_fixed_rows(paths: Iterable[Path], held_out: bool | None = None) -> list[FixedDepthRow]:
    """Read measured depths, keeping explicit train/holdout membership immutable."""
    rows: list[FixedDepthRow] = []
    for source_id, path in enumerate(paths):
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                variant = row.get("variant", "")
                if variant not in DEPTH_VARIANTS:
                    continue
                if row.get("success", "true").lower() == "false":
                    continue
                throughput = _to_float(row, "decode_tps", float("nan"))
                acceptance, acceptance_source = acceptance_from_summary(row, DEPTH_VARIANTS[variant])
                if not math.isfinite(throughput) or throughput <= 0 or not 0 <= acceptance <= 1:
                    raise ValueError(f"invalid fixed-depth measurement in {path}: {variant}")
                scope = scope_from_summary(row)
                group = _group_key(row, source_id)
                if scope.verifier_width_policy:
                    group = (*group, *scope.group_key())
                if held_out is not None:
                    group = (f"split={'holdout' if held_out else 'train'}", *group)
                rows.append(
                    FixedDepthRow(
                        group_key=group,
                        backend=_backend_from_device(row.get("device", "")),
                        model_class=_model_class_from_summary(row.get("model_class", row.get("model", ""))),
                        mode=_canonical_mode(row.get("mode", "")),
                        depth=DEPTH_VARIANTS[variant],
                        decode_tps=throughput,
                        acceptance_rate=acceptance,
                        held_out=held_out, scope=scope, acceptance_source=acceptance_source,
                        full_accepts_lower_bound=full_accepts_lower_bound_from_summary(row, DEPTH_VARIANTS[variant], acceptance_source),
                    )
                )
    validate_acceptance_sources(rows)
    return rows


def label_examples(
    rows: list[FixedDepthRow], startups: Iterable[LearnedStartup] = (),
    tie_tolerance: float = 0.0,
) -> list[LabeledExample]:
    """Label measured economics; d1/d2/d3 remain mandatory controls.

    A precomputed training-only startup may resolve near-equivalent outcomes.
    The same throughput condition scores holdouts, but holdouts never select
    the startup or its tolerance. Large deficits retain their exact winning
    label and therefore still fail the ordinary classification gate.
    """
    if not 0.0 <= tie_tolerance < 1.0:
        raise ValueError("startup tie tolerance must be in [0, 1)")
    preferred = {policy_domain(row): row.depth for row in startups}
    grouped: dict[tuple[str, ...], list[FixedDepthRow]] = {}
    for row in rows:
        grouped.setdefault(row.group_key, []).append(row)

    examples: list[LabeledExample] = []
    for group_key, group_rows in grouped.items():
        by_depth = {row.depth: row for row in group_rows}
        if len(by_depth) != len(group_rows):
            raise ValueError(f"duplicate fixed-depth rows in benchmark lane {group_key}")
        if not all(depth in by_depth for depth in (1, 2, 3)):
            continue
        best_depth = max(
            by_depth,
            key=lambda depth: (by_depth[depth].decode_tps, -depth),
        )
        first = group_rows[0]
        startup = preferred.get(policy_domain(first))
        if (startup in by_depth and
                by_depth[startup].decode_tps >= (1.0 - tie_tolerance) * by_depth[best_depth].decode_tps):
            best_depth = startup
        for depth, row in sorted(by_depth.items()):
            if depth < best_depth:
                action = "promote"
            elif depth > best_depth:
                action = "demote"
            else:
                action = "hold"
            examples.append(
                LabeledExample(
                    backend=row.backend,
                    model_class=row.model_class,
                    mode=row.mode,
                    depth=depth,
                    target_depth=best_depth,
                    acceptance_rate=row.acceptance_rate,
                    action=action,
                    group_key=group_key,
                    held_out=row.held_out, scope=row.scope,
                    relative_tps=tuple((candidate, item.decode_tps / max(x.decode_tps for x in group_rows))
                                       for candidate, item in sorted(by_depth.items())),
                    startup_depth=preferred.get(policy_domain(row)),
                    full_accepts_lower_bound=row.full_accepts_lower_bound,
                )
            )
    return examples


def is_holdout(group_key: tuple[str, ...], modulus: int, bucket: int) -> bool:
    """Return whether a semantic lane belongs to the deterministic holdout.

    A zero modulus is an explicit full-corpus training mode.  It is useful for
    small synthetic regressions where probabilistically withholding the only
    example for a domain would make the expected rule table ill-defined.
    """

    if modulus == 0:
        return False
    joined = "\x1f".join(group_key).encode("utf-8")
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], "little") % modulus == bucket


def learn_startups(rows: list[FixedDepthRow], modulus: int, bucket: int) -> list[LearnedStartup]:
    """Choose a common measured depth using equal-weight training requests.

    Normalizing each request before taking a geometric mean prevents a fast,
    repetitive prompt from outweighing slower prose solely because its absolute
    tok/s is higher. Only depths measured on every complete training request in
    a domain are eligible. A hold rule describes a live acceptance interval;
    it is deliberately not an admission recommendation.
    """
    training_groups = {
        example.group_key for example in label_examples(rows)
        if not (example.held_out if example.held_out is not None
                else is_holdout(example.group_key, modulus, bucket))
    }
    if not training_groups:
        raise ValueError("no complete training requests for learned startup")
    domains: dict[tuple[str, str, str], dict[tuple[str, ...], dict[int, float]]] = {}
    for row in rows:
        if row.group_key in training_groups:
            domain = policy_domain(row)
            domains.setdefault(domain, {}).setdefault(row.group_key, {})[row.depth] = row.decode_tps
    choices = []
    for domain, groups in sorted(domains.items()):
        common = set.intersection(*(set(group) for group in groups.values()))
        scores = {
            depth: math.fsum(math.log(group[depth] / max(group[d] for d in common))
                             for group in groups.values()) / len(groups)
            for depth in common
        }
        winner = max(common, key=lambda depth: (scores[depth], -depth))
        choices.append(LearnedStartup(*domain[:3], winner, len(groups), math.exp(scores[winner]), scope=domain[3]))
    return choices


RATE_SCALE = 1_000_000


def _within_loss(loss: float, tolerance: float) -> bool:
    """Inclusive economic bound, allowing arithmetic roundoff only."""
    return loss - tolerance <= 8 * max(math.ulp(loss), math.ulp(tolerance))


def _economic_rates(example: LabeledExample) -> dict[int, float]:
    """Require measured economics rather than manufacture support from a class label."""
    rates = dict(example.relative_tps)
    if (not rates or len(rates) != len(example.relative_tps) or
            any(not 1 <= depth <= 15 or not math.isfinite(rate) or not 0 < rate <= 1
                for depth, rate in rates.items()) or max(rates.values()) != 1):
        raise ValueError("economic interval fitting requires complete measured relative throughput support")
    return rates


def _cut_ppm(left: float, right: float) -> int | None:
    """Nearest training midpoint satisfying left<=emitted cut<right exactly.

    Both runtime comparisons are inclusive. Ascending first-match order owns
    the shared cut, with no1PPM uncovered interval. Equal or sub-resolution
    conflicting observations are inseparable under the emitted scalar ABI.
    """
    if not 0 <= left < right <= 1:
        return None
    # Decimal authoring avoids a floating product such as .000123*1e6
    # becoming123.00000000000001 and incorrectly discarding legal cut123.
    left_decimal, right_decimal = Decimal(str(left)), Decimal(str(right))
    low = int((left_decimal * RATE_SCALE).to_integral_value(rounding=ROUND_CEILING))
    high = int((right_decimal * RATE_SCALE).to_integral_value(rounding=ROUND_CEILING)) - 1
    if low > high:
        return None
    midpoint = int(((left_decimal + right_decimal) * RATE_SCALE / 2).to_integral_value(rounding=ROUND_HALF_UP))
    cut = min(high, max(low, midpoint))
    return cut if left <= cut / RATE_SCALE < right else None


def _outer_floor_ppm(current: int, target: int, minimum: float, hold_margin: float) -> int | None:
    """Preserve evidence-supported promotion/hold floors; demotions may extend to zero.

    A hold expansion that reaches serialized zero uses its unexpanded measured
    positive floor. If even that floor is unrepresentable, this candidate is
    invalid; no invented epsilon may turn zero acceptance into a learned hold.
    """
    if target < current:
        return 0
    measured = int((Decimal(str(minimum)) * RATE_SCALE).to_integral_value(rounding=ROUND_FLOOR))
    floor = measured
    if target == current:
        expanded = int(((Decimal(str(minimum))-Decimal(str(hold_margin))) * RATE_SCALE).to_integral_value(rounding=ROUND_FLOOR))
        if expanded > 0:
            floor = expanded
    if floor <= 0 or floor / RATE_SCALE > minimum:
        return None
    return floor


def _interval_matches(rule: LearnedRule, example: LabeledExample) -> bool:
    """Score exactly the six-decimal emitted predicates in runtime row order."""
    minimum = float(f"{rule.min_acceptance:.6f}")
    maximum = float(f"{rule.max_acceptance:.6f}")
    return (rule.depth == example.depth and policy_domain(rule) == policy_domain(example) and
            minimum <= example.acceptance_rate <= maximum and
            promotion_supported(rule.target_depth, example))


def learn_rules(
    examples: list[LabeledExample], holdout_modulus: int, holdout_bucket: int,
    min_train_accuracy: float, hold_acceptance_margin: float,
    economic_tolerance: float = .05,
) -> list[LearnedRule]:
    """Fit minimum contiguous economic regions using training observations only.

    Lexicographic objective: minimum region count, maximum training startup
    coverage, maximum measured log-relative throughput, deterministic ties.
    Targets must be economically admissible for EVERY supported training
    observation; binary true negatives never contribute support. Held-out data
    only score the frozen emitted ordered policy after fitting completes.
    """
    if not 0 <= economic_tolerance < 1 or not 0 <= hold_acceptance_margin <= 1:
        raise ValueError("invalid economic tolerance or hold support margin")
    if not 0 <= min_train_accuracy <= 1:
        raise ValueError("economic training accuracy must be in [0,1]")
    def held_out(example):
        return example.held_out if example.held_out is not None else is_holdout(example.group_key, holdout_modulus, holdout_bucket)
    train = [example for example in examples if not held_out(example)]
    if not train:
        raise ValueError("no training examples; holdout data must never be used for fitting")
    rules = []
    lanes = sorted({(*policy_domain(example), example.depth) for example in train})
    for backend, model, mode, scope, depth in lanes:
        points = sorted((example for example in train if
                         (*policy_domain(example), example.depth) == (backend, model, mode, scope, depth)),
                        key=lambda example: (example.acceptance_rate, example.group_key))
        rates = [_economic_rates(example) for example in points]
        if any(not 0 <= point.acceptance_rate <= 1 or not math.isfinite(point.acceptance_rate) for point in points):
            raise ValueError("invalid observed acceptance support")
        startups = {point.startup_depth for point in points}
        if len(startups) != 1:
            raise ValueError("training scope has inconsistent measured startup identity")
        startup = next(iter(startups))
        acceptable = [{target for target, rate in row.items()
                       if _within_loss(1 - rate, economic_tolerance) and promotion_supported(target, point)}
                      for point, row in zip(points, rates)]
        if any(not allowed and point.full_accepts_lower_bound is None for point, allowed in zip(points, acceptable)):
            raise ValueError("no supported economic partition: positive near-best targets lack eligible full-prefix evidence")
        # State: region count, negative startup coverage, negative log rate,
        # deterministic path (start,end,target,lowerPPM,upperPPM).
        states = [None] * (len(points) + 1)
        states[0] = (0, 0, 0.0, ())
        for end in range(1, len(points) + 1):
            allowed = None
            for start in range(end - 1, -1, -1):
                allowed = acceptable[start].copy() if allowed is None else allowed & acceptable[start]
                if not allowed:
                    break
                previous = states[start]
                if previous is None:
                    continue
                lower_cut = None if start == 0 else _cut_ppm(points[start-1].acceptance_rate, points[start].acceptance_rate)
                upper = RATE_SCALE if end == len(points) else _cut_ppm(points[end-1].acceptance_rate, points[end].acceptance_rate)
                if upper is None or (start and lower_cut is None):
                    continue
                for target in sorted(allowed):
                    lower = lower_cut if start else _outer_floor_ppm(depth, target, points[0].acceptance_rate, hold_acceptance_margin)
                    if lower is None or lower > upper:
                        continue
                    if any(not lower / RATE_SCALE <= point.acceptance_rate <= upper / RATE_SCALE for point in points[start:end]):
                        continue
                    entry = (start, end, target, lower, upper)
                    path = previous[3] + (entry,)
                    log_score = math.fsum(math.log(rates[index][chosen])
                        for begin, stop, chosen, _, _ in path for index in range(begin, stop))
                    candidate = (previous[0]+1, previous[1] - ((end-start) if target == startup else 0), -log_score, path)
                    if states[end] is None or candidate < states[end]:
                        states[end] = candidate
        if states[-1] is None:
            raise ValueError(f"no supported economic partition at {backend}/{model}/{mode} depth{depth}: inseparable observations or invalid positive support")
        for ordinal, (start, end, target, minimum, maximum) in enumerate(states[-1][3]):
            action = "promote" if target > depth else "demote" if target < depth else "hold"
            support = end-start
            rules.append(LearnedRule(
                backend, model, mode, depth, target, minimum/RATE_SCALE, maximum/RATE_SCALE,
                1.0, 0.0, target-depth,
                f"trained_{backend}_{model}_{mode}_{action}_d{depth}_to_d{target}_region{ordinal}",
                support, support, 0, 0, scope, support))
    # Validate the actual row order/quantized predicates, not the abstract DP.
    for example in train:
        selected = next((rule for rule in rules if _interval_matches(rule, example)), None)
        if selected is None or not _within_loss(1-_economic_rates(example)[selected.target_depth], economic_tolerance):
            raise ValueError("emitted ordered policy misses or exceeds training economic loss")
    # Region accuracy is economic precision among ACTUALLY selected holdout
    # examples. Empty regions receive no invented evidence or true-negative credit.
    totals = [[0, 0] for _ in rules]
    for example in examples:
        if not held_out(example):
            continue
        for index, rule in enumerate(rules):
            if _interval_matches(rule, example):
                totals[index][1] += 1
                totals[index][0] += _within_loss(1-_economic_rates(example).get(rule.target_depth, 0), economic_tolerance)
                break
    return [replace(rule, holdout_correct=totals[index][0], holdout_total=totals[index][1]) for index, rule in enumerate(rules)]


def _generated_table_lines(path: Path, name: str) -> list[str]:
    """Read exactly one generator-owned table; missing/hand-edited ABI is fatal."""
    matches = re.findall(r"\b" + re.escape(name) + r"\[\] = \{\n(.*?)\n\};",
                         path.read_text(encoding="utf-8"), re.DOTALL)
    if len(matches) != 1:
        raise ValueError(f"malformed generated table {name} in {path}")
    return [line for line in matches[0].splitlines() if line.strip()]


SCOPE_CPP_PATTERN = (r'(?:, \{MTPVerifierPhysicalWidthPolicy::'
                     r'(BoundedLogicalBucket|DynamicDeviceEnvelope|ExactLogicalRows), '
                     r'(\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)\})?')


def _scope_from_cpp(groups: tuple[str | None, ...]) -> PolicyScope:
    """Decode this generator's optional complete-scope suffix."""
    if groups[0] is None:
        return PolicyScope()
    width = next(key for key, value in WIDTH_POLICIES.items() if value == groups[0])
    scope = PolicyScope(width, *(int(value) for value in groups[1:]))
    if not scope.valid():
        raise ValueError("invalid generated policy scope")
    return scope


def cpp_scope_suffix(scope: PolicyScope) -> str:
    """Old rows remain byte-compatible; new rows carry all immutable keys."""
    if not scope.valid():
        raise ValueError("invalid generated policy scope")
    if not scope.verifier_width_policy:
        return ""
    return (", {MTPVerifierPhysicalWidthPolicy::" + WIDTH_POLICIES[scope.verifier_width_policy] +
            ", " + ", ".join(str(getattr(scope, name)) for name in SCOPE_COLUMNS[1:]) + "}")


def preserve_untrained_domains(
    path: Path, trained: list[LearnedRule | LearnedStartup],
) -> list[LearnedRule]:
    """Replace whole measured domains without erasing unrelated backend policies.

    Only this tool's generated row grammar is accepted. Partial or malformed
    rows are fatal: a narrow ROCm refresh must not silently discard CUDA/CPU or
    greedy/dense behavior. Preserved rows are not credited as fresh evidence.
    """
    domains = {policy_domain(rule) for rule in trained}
    pattern = re.compile(
        r'\s*\{MTPVerifyMode::(Greedy|SpeculativeSampling), '
        r'MTPDepthPolicyBackend::(Any|CPU|CUDA|ROCm), '
        r'MTPDepthPolicyModelClass::(Any|Dense|MoE), '
        r'(\d+), ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+), '
        r'([+-]?\d+), "([A-Za-z0-9_]+)"' + SCOPE_CPP_PATTERN + r'\},\s*'
    )
    preserved = []
    seen = 0
    for line in _generated_table_lines(path, "kMTPGeneratedDepthPolicyRules"):
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed generated policy row in {path}: {line}")
        seen += 1
        verify, backend, model, depth, minimum, maximum, zero, full, delta, label = match.groups()[:10]
        scope = _scope_from_cpp(match.groups()[10:])
        mode = "greedy" if verify == "Greedy" else "stochastic"
        domain = (backend.lower(), model.lower(), mode, scope)
        if domain in domains:
            continue
        preserved.append(LearnedRule(
            backend=domain[0], model_class=domain[1], mode=mode, depth=int(depth),
            target_depth=int(depth) + int(delta), min_acceptance=float(minimum),
            max_acceptance=float(maximum), max_zero_accept=float(zero),
            min_full_accept=float(full), delta=int(delta), label=label,
            train_correct=0, train_total=0, holdout_correct=0, holdout_total=0, scope=scope,
        ))
    if not seen:
        raise ValueError(f"base policy contains no generated rules: {path}")
    return preserved


def preserve_untrained_startups(
    path: Path, trained: list[LearnedRule | LearnedStartup],
) -> list[LearnedStartup]:
    """Preserve explicit admission choices alongside untouched transition rows."""
    domains = {policy_domain(row) for row in trained}
    pattern = re.compile(
        r'\s*\{MTPVerifyMode::(Greedy|SpeculativeSampling), '
        r'MTPDepthPolicyBackend::(Any|CPU|CUDA|ROCm), '
        r'MTPDepthPolicyModelClass::(Any|Dense|MoE), (\d+)' + SCOPE_CPP_PATTERN + r'\},\s*')
    preserved = []
    seen = set()
    for line in _generated_table_lines(path, "kMTPGeneratedDepthPolicyStartups"):
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed generated startup row in {path}: {line}")
        verify, backend, model, depth = match.groups()[:4]
        scope = _scope_from_cpp(match.groups()[4:])
        domain = (backend.lower(), model.lower(),
                  "greedy" if verify == "Greedy" else "stochastic", scope)
        if domain in seen or not 1 <= int(depth) <= 15:
            raise ValueError(f"duplicate or invalid generated startup in {path}: {line}")
        seen.add(domain)
        if domain not in domains:
            preserved.append(LearnedStartup(*domain[:3], int(depth), 0, 0.0, scope=scope))
    return preserved


def cpp_verify_mode(mode: str) -> str:
    """Return the C++ enum expression for a benchmark summary mode."""

    if _canonical_mode(mode) == "greedy":
        return "MTPVerifyMode::Greedy"
    return "MTPVerifyMode::SpeculativeSampling"


def cpp_backend(backend: str) -> str:
    """Return the C++ enum expression for a benchmark device backend."""

    normalized = backend.replace("_", "-").lower()
    if normalized == "cuda":
        return "MTPDepthPolicyBackend::CUDA"
    if normalized == "rocm":
        return "MTPDepthPolicyBackend::ROCm"
    if normalized == "cpu":
        return "MTPDepthPolicyBackend::CPU"
    return "MTPDepthPolicyBackend::Any"


def cpp_model_class(model_class: str) -> str:
    """Return the C++ enum expression for a benchmark model class."""

    normalized = model_class.replace("_", "-").lower()
    if normalized == "dense":
        return "MTPDepthPolicyModelClass::Dense"
    if normalized == "moe":
        return "MTPDepthPolicyModelClass::MoE"
    return "MTPDepthPolicyModelClass::Any"


def write_include(path: Path, rules: list[LearnedRule], source_count: int,
                  startups: list[LearnedStartup], preserved_count: int = 0) -> None:
    """Emit the single admission/adaptation policy artifact consumed at runtime."""
    startup_domains = {policy_domain(row) for row in startups}
    if len(startup_domains) != len(startups) or any(not 1 <= row.depth <= 15 for row in startups):
        raise ValueError("duplicate or invalid generated startup choice")
    rule_domains = {policy_domain(row) for row in rules}
    if rule_domains - startup_domains:
        raise ValueError(f"missing measured startup for policy domains: {rule_domains - startup_domains}")
    lines = [
        "/**",
        " * @file MTPDepthPolicyGenerated.inc",
        " * @brief Measured startup choices and live acceptance-window transitions.",
        " *",
        " * Auto-generated by scripts/train_mtp_depth_policy.py.",
        " * Fresh regions: minimum contiguous economic partition; preserved rows retain their original evidence.",
        f" * Labelled fixed-depth examples (including holdout): {source_count}.",
        f" * Preserved untrained-domain rows: {preserved_count}.",
        " * Do not hand-edit rule rows; regenerate from benchmark summaries.",
        " */",
        "static constexpr MTPGeneratedDepthPolicyRule kMTPGeneratedDepthPolicyRules[] = {",
    ]
    for rule in rules:
        lines.append(
            "    "
            f"{{{cpp_verify_mode(rule.mode)}, "
            f"{cpp_backend(rule.backend)}, "
            f"{cpp_model_class(rule.model_class)}, "
            f"{rule.depth}, {rule.min_acceptance:.6f}, {rule.max_acceptance:.6f}, "
            f"{rule.max_zero_accept:.6f}, {rule.min_full_accept:.6f}, "
            f"{rule.delta:+d}, \"{rule.label}\"{cpp_scope_suffix(rule.scope)}}},"
        )
    lines.extend(
        [
            "};",
            "",
            "static constexpr MTPGeneratedDepthPolicyStartup kMTPGeneratedDepthPolicyStartups[] = {",
        ])
    for choice in startups:
        lines.append(
            f"    {{{cpp_verify_mode(choice.mode)}, {cpp_backend(choice.backend)}, "
            f"{cpp_model_class(choice.model_class)}, {choice.depth}{cpp_scope_suffix(choice.scope)}}},")
    lines.extend(
        [
            "};",
            "",
            "static constexpr const char *kMTPGeneratedDepthPolicySource =",
            "    \"trained from benchmark summary TSV\";",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _row_keys_match(row, example) -> bool:
    """Mirror runtime wildcard keys; the complete scope is selected separately."""
    return (row.backend in ("any", example.backend) and
            row.model_class in ("any", example.model_class) and row.mode == example.mode)


def selected_policy_scope(example, startups: Iterable[LearnedStartup]) -> PolicyScope:
    """Select the specific whole domain before acceptance/depth/startup bounds."""
    if example.scope.verifier_width_policy and any(
            row.scope == example.scope and _row_keys_match(row, example) for row in startups):
        return example.scope
    return PolicyScope()


def ordered_policy_report(
    rules: list[LearnedRule], startups: list[LearnedStartup],
    examples: list[LabeledExample], fixed_rows: list[FixedDepthRow],
    modulus: int, bucket: int, held_out: bool, economic_tolerance: float = .05,
) -> dict[str, int | float | str | None]:
    """Score actual first-match targets, coverage, shadowing and measured economic loss.

    Economic admissibility, not equality to the preferred startup-tied label,
    defines accuracy. Exact preferred-label accuracy is reported separately.
    Individual binary true negatives do not certify a selected action. Misses
    count against whole-policy accuracy and are never silently treated as zero
    economic regret: generic handwritten runtime decisions need real dynamic
    held-out measurements. Throughput loss uses the same prompt's measured
    best fixed control, independent of near-best label smoothing.
    """
    rates: dict[tuple[str, ...], dict[int, float]] = {}
    for row in fixed_rows:
        rates.setdefault(row.group_key, {})[row.depth] = row.decode_tps
    total = matched = correct = exact_preferred = overshadowed = 0
    regrets = []
    for example in examples:
        is_held = example.held_out if example.held_out is not None else is_holdout(example.group_key, modulus, bucket)
        if is_held != held_out:
            continue
        total += 1
        scope = selected_policy_scope(example, startups)
        matches = [row for row in rules if _row_keys_match(row, example) and
                   row.scope == scope and row.depth == example.depth and
                   row.min_acceptance <= example.acceptance_rate <= row.max_acceptance and
                   promotion_supported(example.depth + row.delta, example) and
                   row.max_zero_accept >= 1.0 and row.min_full_accept <= 0.0]
        if not matches:
            continue
        matched += 1
        overshadowed += len(matches) - 1
        selected = example.depth + matches[0].delta
        exact_preferred += selected == example.target_depth
        by_depth = rates.get(example.group_key, {})
        if selected not in by_depth:
            raise ValueError("ordered policy selects an unmeasured target depth")
        best = max(by_depth.values())
        loss = (best - by_depth[selected]) / best
        regrets.append(loss)
        correct += _within_loss(loss, economic_tolerance)
    return {"total": total, "matched": matched, "correct": correct,
            "unmatched": total - matched, "overshadowed_rule_matches": overshadowed,
            "coverage": matched / total if total else 0.0,
            "accuracy": correct / total if total else 0.0,
            "accuracy_semantics": "selected_target_economic_admissibility",
            "exact_preferred_correct": exact_preferred,
            "exact_preferred_accuracy": exact_preferred / total if total else 0.0,
            "regret_observations": len(regrets),
            "mean_regret": math.fsum(regrets) / len(regrets) if regrets else None,
            "max_regret": max(regrets) if regrets else None}


def validate_ordered_holdout(report: dict, coverage: float, accuracy: float, regret: float | None) -> None:
    """Fail requested whole-policy gates without crediting absent evidence."""
    if not 0 <= coverage <= 1 or not 0 <= accuracy <= 1 or (regret is not None and not 0 <= regret <= 1):
        raise ValueError("ordered holdout thresholds must be in [0, 1]")
    if coverage == 0 and accuracy == 0 and regret is None:
        return
    if not report["total"]:
        raise ValueError("ordered policy holdout evidence is required")
    if report["coverage"] < coverage or report["accuracy"] < accuracy:
        raise ValueError("ordered policy holdout coverage/accuracy below threshold")
    if regret is not None:
        observed = report["max_regret"]
        # The gate is inclusive. Permit only arithmetic roundoff, not an
        # economic tolerance wider than the authored maximum loss.
        roundoff = 0.0 if observed is None else 8 * max(math.ulp(observed), math.ulp(regret))
        if observed is None or observed - regret > roundoff:
            raise ValueError("ordered policy holdout selected-depth regret exceeds threshold or is unavailable")


def validate_holdout_domains(
    rules: list[LearnedRule], startups: list[LearnedStartup], examples: list[LabeledExample],
    fixed_rows: list[FixedDepthRow], modulus: int, bucket: int, min_region_accuracy: float,
    min_coverage: float, min_accuracy: float, max_regret: float | None,
    economic_tolerance: float,
) -> None:
    """Require held-out evidence for EVERY freshly trained domain when any gate is requested.

    Individual regions not visited by independent holdouts remain uncredited.
    Domain-local gates prevent another well-covered domain hiding their absence
    or a local economic failure. Preserved installed domains are not fresh fits.
    """
    requested = min_region_accuracy > 0 or min_coverage > 0 or min_accuracy > 0 or max_regret is not None
    if not requested:
        return
    domains = {policy_domain(row) for row in [*rules, *startups]}
    for domain in sorted(domains):
        observed = [example for example in examples if policy_domain(example) == domain]
        report = ordered_policy_report(rules, startups, observed, fixed_rows, modulus, bucket, True, economic_tolerance)
        if not report["total"]:
            raise ValueError(f"heldout evidence required for every newly trained domain: {domain}")
        if min_region_accuracy > 0 and not report["matched"]:
            raise ValueError(f"selected heldout evidence required for economic region accuracy: {domain}")
        validate_ordered_holdout(report, min_coverage, min_accuracy, max_regret)


def write_summary(path: Path, rules: list[LearnedRule], examples: list[LabeledExample],
                  startups: list[LearnedStartup], tie_tolerance: float,
                  fixed_rows: list[FixedDepthRow] = (), modulus: int = 0, bucket: int = 0) -> None:
    lines = [
        f"examples={len(examples)}",
        "acceptance_sources=" + json.dumps(sorted({row.acceptance_source for row in fixed_rows})),
        "fit_objective=minimum_contiguous_economic_regions_v1",
        "accuracy_semantics=selected_target_economic_admissibility; exact_preferred_accuracy=diagnostic",
        f"startup_tie_tolerance={tie_tolerance:.6f}",
        f"economic_equivalence_loss_bound={tie_tolerance:.6f}",
        "rules:",
    ]
    for rule in rules:
        holdout_ratio = (
            rule.holdout_correct / rule.holdout_total
            if rule.holdout_total
            else 0.0
        )
        train_ratio = rule.train_correct / rule.train_total if rule.train_total else 0.0
        lines.append(
            f"- {rule.label}: backend={rule.backend} mode={rule.mode} "
            f"model_class={rule.model_class} depth={rule.depth} delta={rule.delta:+d} "
            f"target_depth={rule.target_depth} scope={rule.scope} positive_support={rule.positive_support} "
            f"acceptance=[{rule.min_acceptance:.6f},{rule.max_acceptance:.6f}] "
            f"train={rule.train_correct}/{rule.train_total} ({train_ratio:.3f}) "
            f"holdout={rule.holdout_correct}/{rule.holdout_total} ({holdout_ratio:.3f})"
        )
    lines.append("startups (training requests only):")
    for choice in startups:
        lines.append(f"- {choice.backend}/{choice.model_class}/{choice.mode}: "
                     f"depth={choice.depth} scope={choice.scope} groups={choice.train_groups} "
                     f"geomean_relative_throughput={choice.relative_throughput:.6f}")
    if fixed_rows:
        for held_out in (False, True):
            report = ordered_policy_report(rules, startups, examples, fixed_rows, modulus, bucket, held_out, tie_tolerance)
            lines.append(f"ordered_{'holdout' if held_out else 'train'}=" + json.dumps(report, sort_keys=True))
        lines.append("Ordered regret describes matched fixed-control targets; real held-out dynamic vs installed/fixed runs remain required.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.holdout_modulus < 0:
        raise SystemExit("--holdout-modulus must be non-negative")
    if args.holdout_modulus == 0 and args.holdout_bucket != 0:
        raise SystemExit("--holdout-bucket must be zero when holdout is disabled")
    if args.holdout_modulus > 0 and not (
        0 <= args.holdout_bucket < args.holdout_modulus
    ):
        raise SystemExit("--holdout-bucket must be in [0, holdout-modulus)")
    if args.min_train_accuracy < 0.0 or args.min_train_accuracy > 1.0:
        raise SystemExit("--min-train-accuracy must be in [0, 1]")
    if args.min_holdout_accuracy < 0.0 or args.min_holdout_accuracy > 1.0:
        raise SystemExit("--min-holdout-accuracy must be in [0, 1]")
    if args.hold_acceptance_margin < 0.0 or args.hold_acceptance_margin > 1.0:
        raise SystemExit("--hold-acceptance-margin must be in [0, 1]")
    if not 0.0 <= args.startup_tie_tolerance < 1.0:
        raise SystemExit("--startup-tie-tolerance must be in [0, 1)")

    # Explicit holdout inputs are kept out of training even when the automatic
    # hash split is disabled. They cannot change thresholds or initial depths.
    fixed_rows = load_fixed_rows(args.input, False if args.holdout_input else None)
    fixed_rows += load_fixed_rows(args.holdout_input, True)
    validate_acceptance_sources(fixed_rows)
    startups = learn_startups(fixed_rows, args.holdout_modulus, args.holdout_bucket)
    examples = label_examples(fixed_rows, startups, args.startup_tie_tolerance)
    if not examples:
        raise SystemExit("no complete fixed_d1/fixed_d2/fixed_d3 groups found")

    rules = learn_rules(
        examples,
        args.holdout_modulus,
        args.holdout_bucket,
        args.min_train_accuracy,
        args.hold_acceptance_margin,
        args.startup_tie_tolerance,
    )
    if not rules:
        raise SystemExit("no generated depth-policy rules could be learned")

    validate_holdout_domains(rules, startups, examples, fixed_rows,
        args.holdout_modulus, args.holdout_bucket, args.min_holdout_accuracy,
        args.min_ordered_holdout_coverage, args.min_ordered_holdout_accuracy,
        args.max_ordered_holdout_regret, args.startup_tie_tolerance)
    for rule in rules:
        if rule.holdout_total == 0:
            continue # This region is unobserved/uncredited; requested evidence is required per fresh domain above.
        accuracy = rule.holdout_correct / rule.holdout_total
        if accuracy < args.min_holdout_accuracy:
            raise SystemExit(
                f"holdout economic admissibility below threshold for {rule.label}: "
                f"{accuracy:.3f} < {args.min_holdout_accuracy:.3f}"
            )
    ordered_train = ordered_policy_report(rules, startups, examples, fixed_rows,
        args.holdout_modulus, args.holdout_bucket, False, args.startup_tie_tolerance)
    validate_ordered_holdout(ordered_train, 1.0, 1.0, args.startup_tie_tolerance)
    ordered_holdout = ordered_policy_report(rules, startups, examples, fixed_rows,
        args.holdout_modulus, args.holdout_bucket, True, args.startup_tie_tolerance)
    validate_ordered_holdout(ordered_holdout, args.min_ordered_holdout_coverage,
                             args.min_ordered_holdout_accuracy, args.max_ordered_holdout_regret)
    trained = [*rules, *startups]
    preserved = preserve_untrained_domains(args.base_policy, trained) if args.base_policy else []
    preserved_startups = preserve_untrained_startups(args.base_policy, trained) if args.base_policy else []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_include(args.output, preserved + rules, len(examples),
                  preserved_startups + startups, len(preserved))
    write_summary(args.summary, rules, examples, startups, args.startup_tie_tolerance,
                  fixed_rows, args.holdout_modulus, args.holdout_bucket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
