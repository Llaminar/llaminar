#!/usr/bin/env python3
"""Analyze graph-native MoE overlay benchmark sweep outputs.

The Phase 19 benchmark runner writes one subdirectory per config with:
command.txt, exit_code.txt, stdout_stderr.log, and optionally
moe_overlay_profile.csv. This tool turns those artifacts into a conservative
Markdown report without requiring third-party Python packages.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROFILE_SUM_FIELDS: Tuple[str, ...] = (
    "routed_entries",
    "selected_rows",
    "transfer_bytes",
    "outbound_bytes",
    "return_bytes",
    "inbound_rows",
    "compact_dispatch_bytes",
    "compact_return_bytes",
    "dense_bytes_avoided",
    "cpu_fallback_rows",
    "gpu_cached_rows",
)

CAPACITY_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "all-fit": 3,
}

NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")


@dataclass
class BenchmarkTiming:
    prefill_tokens: Optional[float] = None
    prefill_time_ms: Optional[float] = None
    prefill_throughput_tps: Optional[float] = None
    decode_tokens: Optional[float] = None
    decode_time_ms: Optional[float] = None
    decode_throughput_tps: Optional[float] = None
    total_time_ms: Optional[float] = None
    overall_throughput_tps: Optional[float] = None

    def has_timing(self) -> bool:
        return any(
            value is not None
            for value in (
                self.prefill_time_ms,
                self.prefill_throughput_tps,
                self.decode_time_ms,
                self.decode_throughput_tps,
                self.total_time_ms,
                self.overall_throughput_tps,
            )
        )

    def best_throughput(self) -> Optional[float]:
        for value in (
            self.overall_throughput_tps,
            self.decode_throughput_tps,
            self.prefill_throughput_tps,
        ):
            if value is not None:
                return value
        return None


@dataclass
class ConfigSummary:
    name: str
    path: Path
    family: str
    capacity_bucket: str
    placement_policy: str
    command: Optional[str] = None
    exit_code: Optional[int] = None
    log_present: bool = False
    csv_present: bool = False
    csv_rows: int = 0
    csv_headers: Sequence[str] = field(default_factory=tuple)
    totals: Dict[str, float] = field(default_factory=dict)
    ms_by_field: Dict[str, float] = field(default_factory=dict)
    phase_compute_ms: Dict[str, float] = field(default_factory=dict)
    timing: BenchmarkTiming = field(default_factory=BenchmarkTiming)
    prefill_gemm_evidence: bool = False
    decode_gemv_evidence: bool = False
    notes: List[str] = field(default_factory=list)

    def has_profile_field(self, field_name: str) -> bool:
        return field_name in self.csv_headers

    def total(self, field_name: str) -> Optional[float]:
        if not self.has_profile_field(field_name):
            return None
        return self.totals.get(field_name, 0.0)

    def sparse_wait_ms(self) -> Optional[float]:
        if not self.ms_by_field:
            return None
        wait_fields = [
            name
            for name in self.ms_by_field
            if "wait" in name or name in ("domain_reduce_ms", "cross_domain_reduce_ms")
        ]
        if not wait_fields:
            return None
        return sum(self.ms_by_field[name] for name in wait_fields)

    def scatter_ms(self) -> Optional[float]:
        if not self.ms_by_field:
            return None
        fields = [name for name in self.ms_by_field if "scatter" in name]
        if not fields:
            return None
        return sum(self.ms_by_field[name] for name in fields)

    def import_ms(self) -> Optional[float]:
        if not self.ms_by_field:
            return None
        fields = [name for name in self.ms_by_field if "import" in name]
        if not fields:
            return None
        return sum(self.ms_by_field[name] for name in fields)

    def sparse_transport_ms(self) -> Optional[float]:
        parts = [self.sparse_wait_ms(), self.scatter_ms(), self.import_ms()]
        values = [value for value in parts if value is not None]
        if not values:
            return None
        return sum(values)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a graph-native MoE overlay benchmark run directory."
    )
    parser.add_argument("run_dir", type=Path, help="Run directory produced by the Phase 19 sweep script")
    parser.add_argument("--output", type=Path, help="Optional Markdown output path")
    return parser.parse_args(argv)


def read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"<failed to read {path}: {exc}>"


def first_number(text: str) -> Optional[float]:
    match = NUMBER_RE.search(text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def parse_numeric(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    number = first_number(text)
    return number if number is not None else 0.0


def classify_config_name(name: str) -> Tuple[str, str, str]:
    lower = name.lower()
    has_cpu = "mixed" in lower or "cpu_cold" in lower or re.search(r"(^|_)cpu($|_)", lower) is not None
    if has_cpu:
        family = "mixed GPU/CPU"
    elif "all_gpu" in lower or "cuda_hot_rocm_warm" in lower or "rocm_warm" in lower:
        family = "all-GPU"
    else:
        family = "unknown"

    if "all_fit" in lower or "all-fit" in lower:
        capacity = "all-fit"
    elif "high" in lower:
        capacity = "high"
    elif "medium" in lower:
        capacity = "medium"
    elif "low" in lower:
        capacity = "low"
    else:
        capacity = "not encoded"

    if "static" in lower:
        placement = "static"
    elif "rebalanced" in lower or "rebalance" in lower:
        placement = "rebalanced"
    else:
        placement = "not encoded"

    return family, capacity, placement


def parse_exit_code(path: Path, summary: ConfigSummary) -> None:
    text = read_text(path)
    if text is None:
        summary.notes.append("missing exit_code.txt")
        return
    try:
        summary.exit_code = int(text.strip())
    except ValueError:
        summary.notes.append("unparseable exit_code.txt")


def parse_benchmark_log(text: str, summary: ConfigSummary) -> None:
    current_phase: Optional[str] = None
    lower_text = text.lower()
    summary.prefill_gemm_evidence = bool(
        re.search(r"prefill[^\n]*(gemm|matrix[- ]matrix)", lower_text)
        or re.search(r"(gemm|matrix[- ]matrix)[^\n]*prefill", lower_text)
    )
    summary.decode_gemv_evidence = bool(
        re.search(r"decode[^\n]*(gemv|matrix[- ]vector)", lower_text)
        or re.search(r"(gemv|matrix[- ]vector)[^\n]*decode", lower_text)
    )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if not line:
            continue

        if "prefill" in lower:
            current_phase = "prefill"
        elif "decode" in lower:
            current_phase = "decode"
        elif "total" in lower:
            current_phase = None

        if "overall" in lower and "tok/s" in lower:
            summary.timing.overall_throughput_tps = first_number(line)
            continue
        if "total" in lower and "time" in lower and "ms" in lower:
            summary.timing.total_time_ms = first_number(line)
            continue

        if current_phase == "prefill":
            if "tokens" in lower:
                summary.timing.prefill_tokens = first_number(line)
            elif "time" in lower and "ms" in lower:
                summary.timing.prefill_time_ms = first_number(line)
            elif "throughput" in lower and "tok/s" in lower:
                summary.timing.prefill_throughput_tps = first_number(line)
        elif current_phase == "decode":
            if "tokens" in lower:
                summary.timing.decode_tokens = first_number(line)
            elif "time" in lower and "ms" in lower:
                summary.timing.decode_time_ms = first_number(line)
            elif "throughput" in lower and "tok/s" in lower:
                summary.timing.decode_throughput_tps = first_number(line)


def parse_profile_csv(path: Path, summary: ConfigSummary) -> None:
    if not path.exists():
        summary.notes.append("missing moe_overlay_profile.csv")
        return

    summary.csv_present = True
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = tuple(reader.fieldnames or ())
            summary.csv_headers = headers
            ms_fields = tuple(header for header in headers if header.endswith("_ms"))
            for row in reader:
                summary.csv_rows += 1
                phase = row.get("phase", "unknown") or "unknown"
                for field_name in PROFILE_SUM_FIELDS:
                    if field_name in headers:
                        summary.totals[field_name] = summary.totals.get(field_name, 0.0) + parse_numeric(row.get(field_name))
                for field_name in ms_fields:
                    summary.ms_by_field[field_name] = summary.ms_by_field.get(field_name, 0.0) + parse_numeric(row.get(field_name))
                if "compute_ms" in headers:
                    summary.phase_compute_ms[phase] = summary.phase_compute_ms.get(phase, 0.0) + parse_numeric(row.get("compute_ms"))
    except csv.Error as exc:
        summary.notes.append(f"CSV parse error: {exc}")
    except OSError as exc:
        summary.notes.append(f"CSV read error: {exc}")

    if summary.csv_present and summary.csv_rows == 0:
        summary.notes.append("moe_overlay_profile.csv has no data rows")


def discover_config_dirs(run_dir: Path) -> List[Path]:
    config_files = {"command.txt", "exit_code.txt", "stdout_stderr.log", "moe_overlay_profile.csv"}
    candidates = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if any((child / file_name).exists() for file_name in config_files):
            candidates.append(child)
    return candidates


def analyze_config(config_dir: Path) -> ConfigSummary:
    family, capacity, placement = classify_config_name(config_dir.name)
    summary = ConfigSummary(
        name=config_dir.name,
        path=config_dir,
        family=family,
        capacity_bucket=capacity,
        placement_policy=placement,
    )

    command_text = read_text(config_dir / "command.txt")
    if command_text is None:
        summary.notes.append("missing command.txt")
    else:
        summary.command = " ".join(command_text.strip().split())

    parse_exit_code(config_dir / "exit_code.txt", summary)

    log_text = read_text(config_dir / "stdout_stderr.log")
    if log_text is None:
        summary.notes.append("missing stdout_stderr.log")
    else:
        summary.log_present = True
        parse_benchmark_log(log_text, summary)

    parse_profile_csv(config_dir / "moe_overlay_profile.csv", summary)
    if summary.exit_code not in (None, 0):
        summary.notes.append(f"config exited with code {summary.exit_code}")
    return summary


def fmt_int(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{int(round(value)):,}"


def fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def fmt_rate(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def md_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    out = []
    out.append("| " + " | ".join(md_escape(header) for header in headers) + " |")
    out.append("| " + " | ".join("---" for _header in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(md_escape(value) for value in row) + " |")
    return "\n".join(out)


def average(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def family_configs(configs: Sequence[ConfigSummary], family: str) -> List[ConfigSummary]:
    return [config for config in configs if config.family == family]


def timing_candidates(configs: Sequence[ConfigSummary]) -> List[ConfigSummary]:
    return [config for config in configs if config.timing.has_timing()]


def observation_all_gpu_vs_mixed(configs: Sequence[ConfigSummary]) -> str:
    all_gpu = [config for config in family_configs(configs, "all-GPU") if config.timing.has_timing()]
    all_fit = [config for config in all_gpu if config.capacity_bucket == "all-fit"]
    all_gpu_candidates = all_fit if all_fit else all_gpu
    mixed = [config for config in family_configs(configs, "mixed GPU/CPU") if config.timing.has_timing()]
    if not all_gpu_candidates or not mixed:
        return (
            "- All-GPU vs mixed timing: insufficient data - need benchmark timing/throughput "
            "for at least one all-GPU config and one mixed GPU/CPU config."
        )

    comparisons: List[Tuple[str, Optional[float], Optional[float]]] = [
        (
            "prefill tok/s",
            average(config.timing.prefill_throughput_tps for config in all_gpu_candidates),
            average(config.timing.prefill_throughput_tps for config in mixed),
        ),
        (
            "decode tok/s",
            average(config.timing.decode_throughput_tps for config in all_gpu_candidates),
            average(config.timing.decode_throughput_tps for config in mixed),
        ),
        (
            "overall tok/s",
            average(config.timing.overall_throughput_tps for config in all_gpu_candidates),
            average(config.timing.overall_throughput_tps for config in mixed),
        ),
    ]
    usable = [(label, left, right) for label, left, right in comparisons if left is not None and right is not None]
    if not usable:
        return (
            "- All-GPU vs mixed timing: insufficient data - logs exist, but no comparable "
            "prefill/decode/overall throughput values were parsed."
        )
    observed = all(left > right for _, left, right in usable)
    status = "observed" if observed else "not observed"
    detail = "; ".join(f"{label} all-GPU={left:.2f}, mixed={right:.2f}" for label, left, right in usable)
    return f"- All-GPU vs mixed timing: {status} - {detail}."


def observation_gpu_budget_trend(configs: Sequence[ConfigSummary]) -> str:
    observations: List[str] = []
    for family in ("all-GPU", "mixed GPU/CPU"):
        candidates = [
            config
            for config in family_configs(configs, family)
            if config.capacity_bucket in CAPACITY_ORDER and config.timing.best_throughput() is not None
        ]
        candidates.sort(key=lambda config: CAPACITY_ORDER[config.capacity_bucket])
        if len(candidates) < 2:
            continue
        rates = [config.timing.best_throughput() for config in candidates]
        assert all(rate is not None for rate in rates)
        nondecreasing = all(rates[index] <= rates[index + 1] for index in range(len(rates) - 1))
        capacities = ", ".join(f"{config.capacity_bucket}={config.timing.best_throughput():.2f}" for config in candidates)
        observations.append(f"{family}: {'monotonic' if nondecreasing else 'not monotonic'} ({capacities})")

    if not observations:
        return (
            "- GPU budget throughput trend: insufficient data - need two or more low/medium/high/all-fit "
            "configs in the same family with parsed timing/throughput."
        )
    status = "observed" if all("monotonic" in text and "not monotonic" not in text for text in observations) else "not observed"
    return f"- GPU budget throughput trend: {status} - {'; '.join(observations)}."


def observation_cpu_fallback(configs: Sequence[ConfigSummary]) -> str:
    with_field = [config for config in configs if config.has_profile_field("cpu_fallback_rows")]
    if not with_field:
        return "- CPU fallback row correlation: insufficient data - no profiler CSV with cpu_fallback_rows was available."
    all_gpu_rows = sum(config.total("cpu_fallback_rows") or 0.0 for config in family_configs(with_field, "all-GPU"))
    mixed_rows = sum(config.total("cpu_fallback_rows") or 0.0 for config in family_configs(with_field, "mixed GPU/CPU"))
    if mixed_rows > 0 and all_gpu_rows == 0:
        status = "observed"
    elif mixed_rows > all_gpu_rows:
        status = "observed with caveats"
    else:
        status = "not observed"
    return (
        f"- CPU fallback row correlation: {status} - all-GPU CPU fallback rows={fmt_int(all_gpu_rows)}, "
        f"mixed GPU/CPU CPU fallback rows={fmt_int(mixed_rows)}."
    )


def observation_prefill_decode_shape(configs: Sequence[ConfigSummary]) -> str:
    has_prefill_decode_timing = any(
        config.timing.prefill_time_ms is not None and config.timing.decode_time_ms is not None
        for config in configs
    )
    if not has_prefill_decode_timing:
        return (
            "- Prefill/decode kernel shape: insufficient data - benchmark logs did not include both "
            "prefill and decode timing."
        )
    if any(config.prefill_gemm_evidence for config in configs) and any(config.decode_gemv_evidence for config in configs):
        return "- Prefill/decode kernel shape: observed - logs contain prefill GEMM and decode GEMV evidence."
    return (
        "- Prefill/decode kernel shape: insufficient data - timing is present, but logs do not include "
        "kernel-class evidence for prefill GEMM-like and decode GEMV-like behavior."
    )


def observation_sparse_transport(configs: Sequence[ConfigSummary]) -> str:
    profiled = [config for config in configs if config.csv_present and config.sparse_transport_ms() is not None]
    if not profiled:
        return (
            "- Sparse transport bottleneck check: insufficient data - no profiler rows with wait/scatter/import "
            "timing were available."
        )
    transport_ms = sum(config.sparse_transport_ms() or 0.0 for config in profiled)
    compute_ms = sum(config.ms_by_field.get("compute_ms", 0.0) for config in profiled)
    if compute_ms <= 0.0:
        return (
            f"- Sparse transport bottleneck check: diagnostic - sparse transport timing={transport_ms:.3f} ms, "
            "but compute_ms is missing or zero."
        )
    share = transport_ms / (transport_ms + compute_ms)
    if transport_ms >= compute_ms:
        verdict = "transport is at least as large as compute; inspect sparse pack/unpack and collectives before kernel tuning"
    else:
        verdict = "transport is below compute in this data; keep monitoring before GPU-native pack/unpack work"
    return (
        f"- Sparse transport bottleneck check: diagnostic - sparse transport={transport_ms:.3f} ms, "
        f"compute={compute_ms:.3f} ms, transport_share={share:.1%}; {verdict}."
    )


def render_config_table(configs: Sequence[ConfigSummary]) -> str:
    rows = []
    for config in configs:
        rows.append(
            (
                config.name,
                config.family,
                config.capacity_bucket,
                config.placement_policy,
                "n/a" if config.exit_code is None else config.exit_code,
                "yes" if config.log_present else "missing",
                f"{config.csv_rows} rows" if config.csv_present else "missing",
                fmt_int(config.total("selected_rows")),
                fmt_int(config.total("inbound_rows")),
                fmt_int(config.total("compact_dispatch_bytes")),
                fmt_int(config.total("compact_return_bytes")),
                fmt_int(config.total("dense_bytes_avoided")),
                fmt_int(config.total("cpu_fallback_rows")),
                fmt_int(config.total("gpu_cached_rows")),
                fmt_ms(config.ms_by_field.get("compute_ms") if "compute_ms" in config.ms_by_field else None),
                fmt_ms(config.sparse_wait_ms()),
                fmt_ms(config.scatter_ms()),
                fmt_ms(config.import_ms()),
                fmt_ms(config.timing.prefill_time_ms),
                fmt_rate(config.timing.prefill_throughput_tps),
                fmt_ms(config.timing.decode_time_ms),
                fmt_rate(config.timing.decode_throughput_tps),
                fmt_ms(config.timing.total_time_ms),
                fmt_rate(config.timing.overall_throughput_tps),
            )
        )
    return md_table(
        (
            "Config",
            "Family",
            "Capacity",
            "Placement",
            "Exit",
            "Log",
            "CSV",
            "Selected rows",
            "Inbound rows",
            "Dispatch bytes",
            "Return bytes",
            "Dense bytes avoided",
            "CPU fallback rows",
            "GPU cached rows",
            "Compute ms",
            "Sparse wait ms",
            "Scatter ms",
            "Import ms",
            "Prefill ms",
            "Prefill tok/s",
            "Decode ms",
            "Decode tok/s",
            "Total ms",
            "Overall tok/s",
        ),
        rows,
    )


def render_phase_compute(configs: Sequence[ConfigSummary]) -> str:
    rows = []
    for config in configs:
        if not config.phase_compute_ms:
            continue
        for phase, compute_ms in sorted(config.phase_compute_ms.items()):
            rows.append((config.name, phase, fmt_ms(compute_ms)))
    if not rows:
        return "No profiler compute_ms phase breakdown was available."
    return md_table(("Config", "Profiler phase", "Compute ms"), rows)


def render_completeness(configs: Sequence[ConfigSummary]) -> str:
    rows = []
    for config in configs:
        notes = list(config.notes)
        if config.command is None:
            notes.append("command unavailable")
        if not config.timing.has_timing():
            notes.append("benchmark timing unavailable")
        if not notes:
            notes.append("complete")
        rows.append((config.name, "; ".join(notes)))
    return md_table(("Config", "Completeness / diagnostics"), rows)


def render_tuning_signals(configs: Sequence[ConfigSummary]) -> str:
    lines: List[str] = []
    incomplete = [config.name for config in configs if config.notes or not config.timing.has_timing()]
    if incomplete:
        lines.append(
            "- Complete missing logs/CSVs before changing placement hysteresis or threshold defaults: "
            + ", ".join(incomplete)
            + "."
        )

    mixed_fallback_rows = sum(config.total("cpu_fallback_rows") or 0.0 for config in family_configs(configs, "mixed GPU/CPU"))
    if mixed_fallback_rows > 0:
        lines.append(
            f"- Mixed GPU/CPU configs routed {fmt_int(mixed_fallback_rows)} rows through CPU fallback; "
            "compare this against all-fit GPU timing before raising or lowering GPU hot-cache budgets."
        )

    profiled = [config for config in configs if config.csv_present and config.sparse_transport_ms() is not None]
    if profiled:
        transport_ms = sum(config.sparse_transport_ms() or 0.0 for config in profiled)
        compute_ms = sum(config.ms_by_field.get("compute_ms", 0.0) for config in profiled)
        if compute_ms > 0.0:
            transport_share = transport_ms / (transport_ms + compute_ms)
            lines.append(
                f"- Sparse transport share is {transport_share:.1%} of profiled compute+transport time; "
                "prioritize GPU-native pack/unpack only when real runs show this share competing with compute."
            )
        else:
            lines.append("- Sparse transport timing exists, but compute_ms is missing or zero; inspect profiler coverage before tuning kernels.")

    for family in ("all-GPU", "mixed GPU/CPU"):
        timed_capacities = {
            config.capacity_bucket
            for config in family_configs(configs, family)
            if config.capacity_bucket in CAPACITY_ORDER and config.timing.best_throughput() is not None
        }
        if len(timed_capacities) < 2:
            lines.append(
                f"- {family} budget tuning needs at least two timed low/medium/high/all-fit buckets; "
                f"currently has {len(timed_capacities)}."
            )

    if not lines:
        lines.append("- No tuning recommendation is made from this data alone; inspect the expected-observation status lines first.")
    return "\n".join(lines)


def render_commands(configs: Sequence[ConfigSummary]) -> str:
    lines: List[str] = []
    for config in configs:
        if config.command:
            lines.append(f"- `{config.name}`: `{config.command}`")
        else:
            lines.append(f"- `{config.name}`: command.txt missing")
    return "\n".join(lines)


def render_report(run_dir: Path, configs: Sequence[ConfigSummary]) -> str:
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    metadata_present = (run_dir / "metadata.txt").exists()
    hardware_present = (run_dir / "hardware_inventory.txt").exists()

    sections: List[str] = []
    sections.append("# MoE Graph-Native Overlay Benchmark Analysis")
    sections.append("")
    sections.append(f"Generated: {generated}")
    sections.append(f"Run directory: `{run_dir}`")
    sections.append(f"Configs analyzed: {len(configs)}")
    sections.append(f"metadata.txt: {'present' if metadata_present else 'missing'}")
    sections.append(f"hardware_inventory.txt: {'present' if hardware_present else 'missing'}")
    sections.append("")
    sections.append("This report is an analyzer output, not a performance claim. Status lines say `insufficient data` when the required benchmark timing or profiler evidence is absent.")

    sections.append("\n## Config Summary")
    sections.append(render_config_table(configs) if configs else "No config directories were found.")

    all_gpu = family_configs(configs, "all-GPU")
    mixed = family_configs(configs, "mixed GPU/CPU")

    sections.append("\n## All-GPU Configs")
    sections.append(render_config_table(all_gpu) if all_gpu else "No all-GPU configs were classified from filenames.")

    sections.append("\n## Mixed GPU/CPU Configs")
    sections.append(render_config_table(mixed) if mixed else "No mixed GPU/CPU configs were classified from filenames.")

    sections.append("\n## Profiler Phase Compute Time")
    sections.append(render_phase_compute(configs))

    sections.append("\n## Data Completeness")
    sections.append(render_completeness(configs) if configs else "No data files were available.")

    sections.append("\n## Expected Observations")
    sections.append(observation_all_gpu_vs_mixed(configs))
    sections.append(observation_gpu_budget_trend(configs))
    sections.append(observation_cpu_fallback(configs))
    sections.append(observation_prefill_decode_shape(configs))
    sections.append(observation_sparse_transport(configs))

    sections.append("\n## Sparse Transport Notes")
    sections.append(
        "Use `Sparse wait ms`, `Scatter ms`, `Import ms`, `Dispatch bytes`, and `Return bytes` "
        "to decide whether GPU-native pack/unpack or collective work should precede expert kernel tuning. "
        "Dense bytes avoided is a savings counter, not a throughput result by itself."
    )

    sections.append("\n## Tuning Signals")
    sections.append(render_tuning_signals(configs))

    sections.append("\n## Commands")
    sections.append(render_commands(configs) if configs else "No command.txt files were found.")

    return "\n".join(sections).rstrip() + "\n"


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir
    if not run_dir.is_dir():
        print(f"error: run directory does not exist: {run_dir}", file=sys.stderr)
        return 2

    config_dirs = discover_config_dirs(run_dir)
    configs = [analyze_config(config_dir) for config_dir in config_dirs]
    report = render_report(run_dir, configs)
    print(report, end="")

    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8")
        except OSError as exc:
            print(f"error: failed to write {args.output}: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))