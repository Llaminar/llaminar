#!/usr/bin/env python3
"""Train and emit the generated MTP dynamic-depth policy include.

The trainer consumes one or more benchmark `summary.tsv` files from
`run_mtp_iteration_benchmark_matrix.sh` or `run_mtp_depth_hysteresis_sweep.sh`.
It derives labels from same-run fixed-depth rows: if fixed d3 is the fastest
lane, a healthy depth-1 window should learn a direct promote-to-depth-3 rule;
if fixed d1 is fastest, depth-2/depth-3 windows should learn direct demotions.
The winning depth also emits a hold guardrail so handwritten hysteresis does
not fight the trained result.

This is deliberately not runtime ML.  The output is a compact, deterministic
C++ table that `MTPDepthController` can evaluate cheaply during decode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEPTH_VARIANTS = {
    "fixed_d1": 1,
    "fixed_d2": 2,
    "fixed_d3": 3,
}


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
        "--holdout-modulus",
        type=int,
        default=4,
        help="Stable group hash modulus; one bucket is holdout.",
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
    return parser.parse_args()


def _to_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _group_key(row: dict[str, str]) -> tuple[str, ...]:
    """Build a stable lane key while ignoring metrics and policy variants."""

    preferred = [key for key in ("device", "model", "mode", "case") if key in row]
    if preferred:
        return tuple(row.get(key, "") for key in preferred)
    return tuple(
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


def load_fixed_rows(paths: Iterable[Path]) -> list[FixedDepthRow]:
    rows: list[FixedDepthRow] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                variant = row.get("variant", "")
                if variant not in DEPTH_VARIANTS:
                    continue
                if row.get("success", "true").lower() == "false":
                    continue
                rows.append(
                    FixedDepthRow(
                        group_key=_group_key(row),
                        backend=_backend_from_device(row.get("device", "")),
                        model_class=_model_class_from_summary(row.get("model", "")),
                        mode=row.get("mode", ""),
                        depth=DEPTH_VARIANTS[variant],
                        decode_tps=_to_float(row, "decode_tps"),
                        acceptance_rate=_to_float(row, "acceptance_pct") / 100.0,
                    )
                )
    return rows


def label_examples(rows: list[FixedDepthRow]) -> list[LabeledExample]:
    grouped: dict[tuple[str, ...], list[FixedDepthRow]] = {}
    for row in rows:
        grouped.setdefault(row.group_key, []).append(row)

    examples: list[LabeledExample] = []
    for group_key, group_rows in grouped.items():
        by_depth = {row.depth: row for row in group_rows}
        if not all(depth in by_depth for depth in (1, 2, 3)):
            continue
        best_depth = max(
            (1, 2, 3),
            key=lambda depth: (by_depth[depth].decode_tps, depth),
        )
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
                )
            )
    return examples


def is_holdout(group_key: tuple[str, ...], modulus: int, bucket: int) -> bool:
    joined = "\x1f".join(group_key).encode("utf-8")
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], "little") % modulus == bucket


def _accuracy(
    examples: list[LabeledExample],
    backend: str,
    model_class: str,
    mode: str,
    depth: int,
    target_depth: int,
    action: str,
    threshold: float,
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
        if action == "promote":
            predicted = example.acceptance_rate >= threshold
        elif action == "demote":
            predicted = example.acceptance_rate <= threshold
        else:
            predicted = example.acceptance_rate >= threshold
        expected = (
            example.action == action and
            example.target_depth == target_depth
        )
        if predicted == expected:
            correct += 1
    return correct, total


def _choose_threshold(
    examples: list[LabeledExample],
    backend: str,
    model_class: str,
    mode: str,
    depth: int,
    target_depth: int,
    action: str,
) -> tuple[float, int, int]:
    best_threshold = 0.0
    best_correct = -1
    best_total = 0
    for point in range(0, 101):
        threshold = point / 100.0
        correct, total = _accuracy(
            examples,
            backend,
            model_class,
            mode,
            depth,
            target_depth,
            action,
            threshold,
        )
        if total == 0:
            continue
        if correct > best_correct:
            best_threshold = threshold
            best_correct = correct
            best_total = total
        elif correct == best_correct:
            # Prefer the conservative equally correct rule.  Promotions should
            # require the strongest acceptance signal still supported by the
            # training data, while demotions should only fire on the weakest
            # acceptance signal that still explains the data.  This prevents an
            # offline table from turning "all zero accepted" into a generated
            # promotion just because every depth-1 training lane eventually
            # preferred a deeper fixed-depth benchmark.
            if action in {"promote", "hold"} and threshold > best_threshold:
                best_threshold = threshold
            if action == "demote" and threshold < best_threshold:
                best_threshold = threshold
    return best_threshold, best_correct, best_total


def learn_rules(
    examples: list[LabeledExample],
    holdout_modulus: int,
    holdout_bucket: int,
    min_train_accuracy: float,
    hold_acceptance_margin: float,
) -> list[LearnedRule]:
    train = [
        example
        for example in examples
        if not is_holdout(example.group_key, holdout_modulus, holdout_bucket)
    ]
    holdout = [
        example
        for example in examples
        if is_holdout(example.group_key, holdout_modulus, holdout_bucket)
    ]
    if not train:
        train = examples
        holdout = []

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

            threshold, train_correct, train_total = _choose_threshold(
                train,
                backend,
                model_class,
                mode,
                depth,
                target_depth,
                action,
            )
            train_accuracy = train_correct / train_total if train_total else 0.0
            if train_accuracy < min_train_accuracy:
                continue
            holdout_correct, holdout_total = _accuracy(
                holdout,
                backend,
                model_class,
                mode,
                depth,
                target_depth,
                action,
                threshold,
            )
            min_acceptance = threshold
            max_acceptance = 1.0
            if action == "demote":
                min_acceptance = 0.0
                max_acceptance = threshold
            elif action == "hold":
                min_acceptance = max(0.0, threshold - hold_acceptance_margin)
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


def cpp_verify_mode(mode: str) -> str:
    """Return the C++ enum expression for a benchmark summary mode."""

    normalized = mode.replace("_", "-").lower()
    if normalized == "greedy":
        return "MTPVerifyMode::Greedy"
    if normalized in {"stochastic", "speculative-sampling", "sampling"}:
        return "MTPVerifyMode::SpeculativeSampling"
    return "MTPVerifyMode::Greedy"


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


def write_include(path: Path, rules: list[LearnedRule], source_count: int) -> None:
    lines = [
        "/*",
        " * Auto-generated by scripts/train_mtp_depth_policy.py.",
        f" * Training examples: {source_count}.",
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
            "static constexpr const char *kMTPGeneratedDepthPolicySource =",
            "    \"trained from benchmark summary TSV\";",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, rules: list[LearnedRule], examples: list[LabeledExample]) -> None:
    lines = [
        f"examples={len(examples)}",
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.min_train_accuracy < 0.0 or args.min_train_accuracy > 1.0:
        raise SystemExit("--min-train-accuracy must be in [0, 1]")
    if args.min_holdout_accuracy < 0.0 or args.min_holdout_accuracy > 1.0:
        raise SystemExit("--min-holdout-accuracy must be in [0, 1]")
    if args.hold_acceptance_margin < 0.0 or args.hold_acceptance_margin > 1.0:
        raise SystemExit("--hold-acceptance-margin must be in [0, 1]")

    fixed_rows = load_fixed_rows(args.input)
    examples = label_examples(fixed_rows)
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

    saw_holdout = False
    for rule in rules:
        if rule.holdout_total == 0:
            continue
        saw_holdout = True
        accuracy = rule.holdout_correct / rule.holdout_total
        if accuracy < args.min_holdout_accuracy:
            raise SystemExit(
                f"holdout accuracy below threshold for {rule.label}: "
                f"{accuracy:.3f} < {args.min_holdout_accuracy:.3f}"
            )
    if args.min_holdout_accuracy > 0.0 and not saw_holdout:
        raise SystemExit("holdout accuracy requested but no holdout examples were selected")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_include(args.output, rules, len(examples))
    write_summary(args.summary, rules, examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
