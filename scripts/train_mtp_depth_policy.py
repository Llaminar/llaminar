#!/usr/bin/env python3
"""Train and emit the generated MTP dynamic-depth policy include.

The trainer consumes one or more benchmark `summary.tsv` files from
`run_mtp_iteration_benchmark_matrix.sh` or `run_mtp_depth_hysteresis_sweep.sh`.
It derives labels from same-run fixed-depth rows (depths 1 through 15): if d3 is the fastest
lane, a healthy depth-1 window should learn a direct promote-to-depth-3 rule;
if fixed d1 is fastest, depth-2/depth-3 windows should learn direct demotions.
The winning depth also emits a hold guardrail so handwritten hysteresis does
not fight the trained result. Startup is a separate measured decision: choose
the best geometric-mean relative throughput across training requests, not the
largest depth which happened to win one request. Holdouts never choose startup.
Within a declared near-best throughput band, transition labels prefer that
startup instead of chasing tiny per-prompt wins. Outside the band the actual
fixed-depth winner remains the target, including on held-out requests.

This is deliberately not runtime ML.  The output is a compact, deterministic
C++ table that `MTPDepthController` can evaluate cheaply during decode.
Explicit holdout files never participate in fitting. A scoped refresh may
preserve generated rows for every unmeasured backend/model/sampling domain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEPTH_VARIANTS = {f"fixed_d{depth}": depth for depth in range(1, 16)}


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


@dataclass(frozen=True)
class LearnedStartup:
    """One domain's measured admission choice; it does not constrain adaptation."""

    backend: str
    model_class: str
    mode: str
    depth: int
    train_groups: int
    relative_throughput: float


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
        help="Preserve generated rows for backend/model/mode domains absent from this training run.",
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
        help="Fail if any learned action with holdout examples scores below this ratio.",
    )
    parser.add_argument(
        "--min-train-accuracy",
        type=float,
        default=0.75,
        help=(
            "Skip generated rules whose best acceptance-window predicate scores "
            "below this ratio on the training split."
        ),
    )
    parser.add_argument(
        "--hold-acceptance-margin",
        type=float,
        default=0.05,
        help=(
            "Subtract this margin from a learned hold threshold so aggregate "
            "fixed-depth evidence protects nearby live windows without masking "
            "clearly bad acceptance."
        ),
    )
    parser.add_argument(
        "--startup-tie-tolerance", type=float, default=0.05,
        help="Prefer measured startup when within this fraction of a request's best fixed throughput; zero requires the exact winner.",
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
        if key
        not in {
            "variant",
            "decode_tps",
            "speedup_vs_baseline",
            "overall_tps",
            "acceptance_pct",
            "accepted",
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
                acceptance = _to_float(row, "acceptance_pct", float("nan")) / 100.0
                if not math.isfinite(throughput) or throughput <= 0 or not 0 <= acceptance <= 1:
                    raise ValueError(f"invalid fixed-depth measurement in {path}: {variant}")
                group = _group_key(row, source_id)
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
                        held_out=held_out,
                    )
                )
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
    preferred = {(row.backend, row.model_class, row.mode): row.depth for row in startups}
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
        startup = preferred.get((first.backend, first.model_class, first.mode))
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
                    held_out=row.held_out,
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
            domain = (row.backend, row.model_class, row.mode)
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
        choices.append(LearnedStartup(*domain, winner, len(groups), math.exp(scores[winner])))
    return choices


def _accuracy_interval(
    examples: list[LabeledExample],
    backend: str,
    model_class: str,
    mode: str,
    depth: int,
    target_depth: int,
    action: str,
    min_acceptance: float,
    max_acceptance: float,
) -> tuple[int, int]:
    total = 0
    correct = 0
    for example in examples:
        if (
            example.backend != backend
            or example.model_class != model_class
            or example.mode != mode
            or example.depth != depth
        ):
            continue
        total += 1
        predicted = min_acceptance <= example.acceptance_rate <= max_acceptance
        expected = (
            example.action == action and
            example.target_depth == target_depth
        )
        if predicted == expected:
            correct += 1
    return correct, total


def _choose_acceptance_interval(
    examples: list[LabeledExample],
    backend: str,
    model_class: str,
    mode: str,
    depth: int,
    target_depth: int,
    action: str,
) -> tuple[float, float, int, int]:
    best_min_acceptance = 0.0
    best_max_acceptance = 1.0
    best_correct = -1
    best_total = 0
    for min_point in range(0, 101):
        min_acceptance = min_point / 100.0
        for max_point in range(min_point, 101):
            max_acceptance = max_point / 100.0
            correct, total = _accuracy_interval(
                examples,
                backend,
                model_class,
                mode,
                depth,
                target_depth,
                action,
                min_acceptance,
                max_acceptance,
            )
            if total == 0:
                continue
            better = correct > best_correct
            if correct == best_correct:
                if action == "demote":
                    # Demotion rules should remain low-acceptance guards.
                    better = (
                        min_acceptance < best_min_acceptance or
                        (
                            min_acceptance == best_min_acceptance and
                            max_acceptance < best_max_acceptance
                        )
                    )
                else:
                    # Promotions/holds prefer the widest high side that still
                    # explains the data, then the strongest lower bound.  This
                    # preserves the old high-acceptance behavior when possible,
                    # but can also express a bounded "probe the next depth"
                    # region when a low-to-moderate lane wins in benchmark data.
                    better = (
                        max_acceptance > best_max_acceptance or
                        (
                            max_acceptance == best_max_acceptance and
                            min_acceptance > best_min_acceptance
                        )
                    )
            if better:
                best_min_acceptance = min_acceptance
                best_max_acceptance = max_acceptance
                best_correct = correct
                best_total = total
    return best_min_acceptance, best_max_acceptance, best_correct, best_total


def learn_rules(
    examples: list[LabeledExample],
    holdout_modulus: int,
    holdout_bucket: int,
    min_train_accuracy: float,
    hold_acceptance_margin: float,
) -> list[LearnedRule]:
    """Fit training observations only; score the exact emitted predicate."""
    def held_out(example: LabeledExample) -> bool:
        return (example.held_out if example.held_out is not None else
                is_holdout(example.group_key, holdout_modulus, holdout_bucket))

    train = [
        example
        for example in examples
        if not held_out(example)
    ]
    holdout = [
        example
        for example in examples
        if held_out(example)
    ]
    if not train:
        raise ValueError("no training examples; holdout data must never be used for fitting")

    rules: list[LearnedRule] = []
    rule_groups = sorted(
        {(example.backend, example.model_class, example.mode) for example in examples}
    )
    for backend, model_class, mode in rule_groups:
        action_groups = sorted(
            {
                (example.depth, example.target_depth, example.action)
                for example in train
                if example.backend == backend
                and example.model_class == model_class
                and example.mode == mode
            }
        )
        for depth, target_depth, action in action_groups:
            if action == "promote" and target_depth <= depth:
                continue
            if action == "demote" and target_depth >= depth:
                continue
            if action == "hold" and target_depth != depth:
                continue

            (
                min_acceptance,
                max_acceptance,
                train_correct,
                train_total,
            ) = _choose_acceptance_interval(
                train,
                backend,
                model_class,
                mode,
                depth,
                target_depth,
                action,
            )
            if action == "hold":
                min_acceptance = max(0.0, min_acceptance - hold_acceptance_margin)
            # The installed predicate includes the hold margin. Validate that
            # predicate, not a narrower one which will never execute.
            train_correct, train_total = _accuracy_interval(
                train, backend, model_class, mode, depth, target_depth, action,
                min_acceptance, max_acceptance,
            )
            train_accuracy = train_correct / train_total if train_total else 0.0
            if train_accuracy < min_train_accuracy:
                continue
            holdout_correct, holdout_total = _accuracy_interval(
                holdout,
                backend,
                model_class,
                mode,
                depth,
                target_depth,
                action,
                min_acceptance,
                max_acceptance,
            )
            rules.append(
                LearnedRule(
                    backend=backend,
                    model_class=model_class,
                    mode=mode,
                    depth=depth,
                    target_depth=target_depth,
                    min_acceptance=min_acceptance,
                    max_acceptance=max_acceptance,
                    max_zero_accept=1.0,
                    min_full_accept=0.0,
                    delta=target_depth - depth,
                    label=(
                        f"trained_{backend}_{model_class}_{mode}_{action}_"
                        f"d{depth}_to_d{target_depth}"
                    ),
                    train_correct=train_correct,
                    train_total=train_total,
                    holdout_correct=holdout_correct,
                    holdout_total=holdout_total,
                )
            )

    return rules


def _generated_table_lines(path: Path, name: str) -> list[str]:
    """Read exactly one generator-owned table; missing/hand-edited ABI is fatal."""
    matches = re.findall(r"\b" + re.escape(name) + r"\[\] = \{\n(.*?)\n\};",
                         path.read_text(encoding="utf-8"), re.DOTALL)
    if len(matches) != 1:
        raise ValueError(f"malformed generated table {name} in {path}")
    return [line for line in matches[0].splitlines() if line.strip()]


def preserve_untrained_domains(
    path: Path, trained: list[LearnedRule | LearnedStartup],
) -> list[LearnedRule]:
    """Replace whole measured domains without erasing unrelated backend policies.

    Only this tool's generated row grammar is accepted. Partial or malformed
    rows are fatal: a narrow ROCm refresh must not silently discard CUDA/CPU or
    greedy/dense behavior. Preserved rows are not credited as fresh evidence.
    """
    domains = {(rule.backend, rule.model_class, rule.mode) for rule in trained}
    pattern = re.compile(
        r'\s*\{MTPVerifyMode::(Greedy|SpeculativeSampling), '
        r'MTPDepthPolicyBackend::(Any|CPU|CUDA|ROCm), '
        r'MTPDepthPolicyModelClass::(Any|Dense|MoE), '
        r'(\d+), ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+), '
        r'([+-]?\d+), "([A-Za-z0-9_]+)"\},\s*'
    )
    preserved = []
    seen = 0
    for line in _generated_table_lines(path, "kMTPGeneratedDepthPolicyRules"):
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed generated policy row in {path}: {line}")
        seen += 1
        verify, backend, model, depth, minimum, maximum, zero, full, delta, label = match.groups()
        mode = "greedy" if verify == "Greedy" else "stochastic"
        domain = (backend.lower(), model.lower(), mode)
        if domain in domains:
            continue
        preserved.append(LearnedRule(
            backend=domain[0], model_class=domain[1], mode=mode, depth=int(depth),
            target_depth=int(depth) + int(delta), min_acceptance=float(minimum),
            max_acceptance=float(maximum), max_zero_accept=float(zero),
            min_full_accept=float(full), delta=int(delta), label=label,
            train_correct=0, train_total=0, holdout_correct=0, holdout_total=0,
        ))
    if not seen:
        raise ValueError(f"base policy contains no generated rules: {path}")
    return preserved


def preserve_untrained_startups(
    path: Path, trained: list[LearnedRule | LearnedStartup],
) -> list[LearnedStartup]:
    """Preserve explicit admission choices alongside untouched transition rows."""
    domains = {(row.backend, row.model_class, row.mode) for row in trained}
    pattern = re.compile(
        r'\s*\{MTPVerifyMode::(Greedy|SpeculativeSampling), '
        r'MTPDepthPolicyBackend::(Any|CPU|CUDA|ROCm), '
        r'MTPDepthPolicyModelClass::(Any|Dense|MoE), (\d+)\},\s*')
    preserved = []
    seen = set()
    for line in _generated_table_lines(path, "kMTPGeneratedDepthPolicyStartups"):
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed generated startup row in {path}: {line}")
        verify, backend, model, depth = match.groups()
        domain = (backend.lower(), model.lower(),
                  "greedy" if verify == "Greedy" else "stochastic")
        if domain in seen or not 1 <= int(depth) <= 15:
            raise ValueError(f"duplicate or invalid generated startup in {path}: {line}")
        seen.add(domain)
        if domain not in domains:
            preserved.append(LearnedStartup(*domain, int(depth), 0, 0.0))
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
    startup_domains = {(row.backend, row.model_class, row.mode) for row in startups}
    if len(startup_domains) != len(startups) or any(not 1 <= row.depth <= 15 for row in startups):
        raise ValueError("duplicate or invalid generated startup choice")
    rule_domains = {(row.backend, row.model_class, row.mode) for row in rules}
    if rule_domains - startup_domains:
        raise ValueError(f"missing measured startup for policy domains: {rule_domains - startup_domains}")
    lines = [
        "/**",
        " * @file MTPDepthPolicyGenerated.inc",
        " * @brief Measured startup choices and live acceptance-window transitions.",
        " *",
        " * Auto-generated by scripts/train_mtp_depth_policy.py.",
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
            f"{rule.delta:+d}, \"{rule.label}\"}},"
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
            f"{cpp_model_class(choice.model_class)}, {choice.depth}}},")
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


def write_summary(path: Path, rules: list[LearnedRule], examples: list[LabeledExample],
                  startups: list[LearnedStartup], tie_tolerance: float) -> None:
    lines = [
        f"examples={len(examples)}",
        f"startup_tie_tolerance={tie_tolerance:.6f}",
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
            f"target_depth={rule.target_depth} "
            f"acceptance=[{rule.min_acceptance:.2f},{rule.max_acceptance:.2f}] "
            f"train={rule.train_correct}/{rule.train_total} ({train_ratio:.3f}) "
            f"holdout={rule.holdout_correct}/{rule.holdout_total} ({holdout_ratio:.3f})"
        )
    lines.append("startups (training requests only):")
    for choice in startups:
        lines.append(f"- {choice.backend}/{choice.model_class}/{choice.mode}: "
                     f"depth={choice.depth} groups={choice.train_groups} "
                     f"geomean_relative_throughput={choice.relative_throughput:.6f}")
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
    )
    if not rules:
        raise SystemExit("no generated depth-policy rules could be learned")

    for rule in rules:
        if rule.holdout_total == 0:
            if args.min_holdout_accuracy > 0.0:
                raise SystemExit(f"holdout accuracy requested but no holdout examples for {rule.label}")
            continue
        accuracy = rule.holdout_correct / rule.holdout_total
        if accuracy < args.min_holdout_accuracy:
            raise SystemExit(
                f"holdout accuracy below threshold for {rule.label}: "
                f"{accuracy:.3f} < {args.min_holdout_accuracy:.3f}"
            )
    trained = [*rules, *startups]
    preserved = preserve_untrained_domains(args.base_policy, trained) if args.base_policy else []
    preserved_startups = preserve_untrained_startups(args.base_policy, trained) if args.base_policy else []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_include(args.output, preserved + rules, len(examples),
                  preserved_startups + startups, len(preserved))
    write_summary(args.summary, rules, examples, startups, args.startup_tie_tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
