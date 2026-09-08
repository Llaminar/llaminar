#!/usr/bin/env python3
"""! @file run_cpu_fa2_tile_tournament.py
@brief Process-isolated aggregate certification for CPU FA2 tile dispatch.

One production server owns one model attention geometry and one native K/V
format. Running every unrelated model and format through one OpenMP process
creates allocator, worker-pool, and cache history that no production request
observes. This driver gives every ``(format, head dimension)`` family a fresh
worker process while deliberately keeping serial decode, grouped verification,
and prefill together inside that process. It then certifies the aggregate p95
regret across the complete 81-domain representative matrix.

The driver also pins workers to one hardware thread per physical core on one
socket. Hyperthreads and cross-socket first-touch effects are therefore not
silently admitted into policy evidence. Child failures, malformed CSV, missing
domains, mixed code-generation/runtime ISA profiles, and duplicate disagreement
are fatal.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from typing import Iterable, Sequence


NATIVE_FORMATS: tuple[str, ...] = (
    "fp32",
    "fp16",
    "bf16",
    "q16_1",
    "q8_1",
    "tq4_tq4",
    "tq4_tq8",
    "tq8_tq4",
    "tq8_tq8",
)
HEAD_DIMS: tuple[int, ...] = (64, 128, 256)
EXPECTED_REGIMES: frozenset[str] = frozenset(
    ("decode", "grouped_verifier", "prefill")
)
CSV_COLUMNS = 23


@dataclass(frozen=True)
class DomainRegret:
    """One unique production domain and its paired tournament regret."""

    sources: str
    format_name: str
    codegen_isa: str
    isa: str
    regime: str
    query_rows: int
    kv_rows: int
    heads: int
    kv_heads: int
    head_dim: int
    gqa_rep: int
    regret_percent: float

    @property
    def identity(self) -> tuple[object, ...]:
        """Return fields that must identify one domain across candidate rows."""

        return (
            self.sources,
            self.format_name,
            self.codegen_isa,
            self.isa,
            self.regime,
            self.query_rows,
            self.kv_rows,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.gqa_rep,
        )


def nearest_rank_percentile(values: Sequence[float], probability: float) -> float:
    """Return the nearest-rank percentile used by the C++ policy gate."""

    if not values:
        raise ValueError("cannot calculate a percentile over no values")
    if not 0.0 < probability <= 1.0:
        raise ValueError("percentile probability must be in (0, 1]")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def parse_cpu_list(text: str) -> list[int]:
    """Parse Linux taskset syntax into a sorted unique CPU inventory."""

    cpus: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            raise ValueError("CPU list contains an empty component")
        if "-" in item:
            begin_text, end_text = item.split("-", maxsplit=1)
            begin = int(begin_text)
            end = int(end_text)
            if begin < 0 or end < begin:
                raise ValueError(f"invalid CPU range: {item}")
            cpus.update(range(begin, end + 1))
        else:
            cpu = int(item)
            if cpu < 0:
                raise ValueError(f"invalid CPU id: {cpu}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU list is empty")
    return sorted(cpus)


def discover_single_socket_physical_cpus() -> list[int]:
    """Choose one allowed hardware thread per core on one physical package."""

    packages: dict[int, dict[int, int]] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = int((topology / "physical_package_id").read_text().strip())
            core = int((topology / "core_id").read_text().strip())
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"cannot authenticate physical topology for CPU {cpu}: {error}"
            ) from error
        packages.setdefault(package, {}).setdefault(core, cpu)

    if not packages:
        raise RuntimeError("current affinity contains no online physical cores")
    package, cores = min(
        packages.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    selected = sorted(cores.values())
    if not selected:
        raise RuntimeError(f"physical package {package} contains no usable cores")
    return selected


def parse_family_output(
    output: str,
    expected_format: str,
    expected_head_dim: int,
) -> list[DomainRegret]:
    """Parse and authenticate one child family's candidate CSV rows."""

    domains: dict[tuple[object, ...], DomainRegret] = {}
    candidate_counts: dict[tuple[object, ...], int] = {}
    for line in output.splitlines():
        columns = line.strip().split(",")
        if len(columns) != CSV_COLUMNS or columns[1] != expected_format:
            continue
        try:
            domain = DomainRegret(
                sources=columns[0],
                format_name=columns[1],
                codegen_isa=columns[2],
                isa=columns[3],
                regime=columns[4],
                query_rows=int(columns[5]),
                kv_rows=int(columns[6]),
                heads=int(columns[7]),
                kv_heads=int(columns[8]),
                head_dim=int(columns[9]),
                gqa_rep=int(columns[10]),
                regret_percent=float(columns[18]),
            )
        except ValueError as error:
            raise RuntimeError(f"malformed tournament CSV row: {line}") from error
        if domain.head_dim != expected_head_dim:
            raise RuntimeError(
                f"worker emitted head_dim={domain.head_dim}, expected "
                f"{expected_head_dim}"
            )
        prior = domains.get(domain.identity)
        if prior is not None and not math.isclose(
            prior.regret_percent,
            domain.regret_percent,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                f"candidate rows disagree for domain {domain.identity}"
            )
        domains[domain.identity] = domain
        candidate_counts[domain.identity] = (
            candidate_counts.get(domain.identity, 0) + 1
        )

    regimes = {domain.regime for domain in domains.values()}
    if regimes != EXPECTED_REGIMES:
        raise RuntimeError(
            f"{expected_format}/hd{expected_head_dim} emitted regimes "
            f"{sorted(regimes)}, expected {sorted(EXPECTED_REGIMES)}"
        )
    invalid_counts = {
        identity: count
        for identity, count in candidate_counts.items()
        if count != 7
    }
    if invalid_counts:
        raise RuntimeError(
            "each domain must emit all seven compiled tile candidates: "
            f"{invalid_counts}"
        )
    return sorted(domains.values(), key=lambda domain: domain.identity)


def aggregate_family_repeats(
    repeats: Sequence[Sequence[DomainRegret]],
) -> list[DomainRegret]:
    """Form one median-regret record per domain across fresh worker repeats."""

    if not repeats:
        raise ValueError("family certification requires at least one repeat")
    expected = {domain.identity for domain in repeats[0]}
    observations: dict[tuple[object, ...], list[float]] = {
        identity: [] for identity in expected
    }
    templates = {domain.identity: domain for domain in repeats[0]}
    for repeat_index, repeat in enumerate(repeats):
        identities = {domain.identity for domain in repeat}
        if identities != expected:
            raise RuntimeError(
                f"family repeat {repeat_index} changed domain inventory"
            )
        for domain in repeat:
            observations[domain.identity].append(domain.regret_percent)
    return sorted(
        (
            replace(
                templates[identity],
                regret_percent=nearest_rank_percentile(regrets, 0.50),
            )
            for identity, regrets in observations.items()
        ),
        key=lambda domain: domain.identity,
    )


def run_family(
    binary: Path,
    format_name: str,
    head_dim: int,
    worker_cpus: Sequence[int],
) -> list[DomainRegret]:
    """Execute one production family in a fresh, socket-pinned process."""

    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset is required for CPU tournament isolation")
    cpu_list = ",".join(str(cpu) for cpu in worker_cpus)
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(len(worker_cpus)),
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "LLAMINAR_CPU_FA2_TOURNAMENT_FORMAT": format_name,
            "LLAMINAR_CPU_FA2_TOURNAMENT_HEAD_DIM": str(head_dim),
            "LLAMINAR_CPU_FA2_TOURNAMENT_MAX_P95_REGRET": "100000",
        }
    )
    command = [
        taskset,
        "-c",
        cpu_list,
        str(binary),
        "--gtest_filter="
        "Perf__CPUFlashAttentionTileTournament.RepresentativeAllFormatPolicyGate",
    ]
    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed for {format_name}/hd{head_dim} "
            f"with status {completed.returncode}:\n{completed.stdout}"
        )
    return parse_family_output(completed.stdout, format_name, head_dim)


def certify(
    binary: Path,
    worker_cpus: Sequence[int],
    maximum_p95_regret: float,
    family_repeats: int,
) -> int:
    """Run the complete process-isolated matrix and enforce aggregate p95."""

    all_domains: list[DomainRegret] = []
    print(
        "CPU_FA2_PROCESS_ISOLATION,workers="
        f"{len(worker_cpus)},repeats={family_repeats},"
        f"cpus={','.join(map(str, worker_cpus))}",
        flush=True,
    )
    for head_dim in HEAD_DIMS:
        for format_name in NATIVE_FORMATS:
            repeat_results = [
                run_family(binary, format_name, head_dim, worker_cpus)
                for _ in range(family_repeats)
            ]
            family = aggregate_family_repeats(repeat_results)
            all_domains.extend(family)
            family_max = max(domain.regret_percent for domain in family)
            print(
                "CPU_FA2_FAMILY,"
                f"format={format_name},head_dim={head_dim},"
                f"domains={len(family)},max_regret_pct={family_max:.3f}",
                flush=True,
            )

    expected_domains = len(NATIVE_FORMATS) * len(HEAD_DIMS) * len(EXPECTED_REGIMES)
    if len(all_domains) != expected_domains:
        raise RuntimeError(
            f"aggregate matrix emitted {len(all_domains)} domains, "
            f"expected {expected_domains}"
        )
    isas = {domain.isa for domain in all_domains}
    if len(isas) != 1:
        raise RuntimeError(f"aggregate matrix mixed active ISAs: {sorted(isas)}")
    codegen_isas = {domain.codegen_isa for domain in all_domains}
    if len(codegen_isas) != 1:
        raise RuntimeError(
            "aggregate matrix mixed code-generation ISAs: "
            f"{sorted(codegen_isas)}"
        )

    regrets = [domain.regret_percent for domain in all_domains]
    p50 = nearest_rank_percentile(regrets, 0.50)
    p95 = nearest_rank_percentile(regrets, 0.95)
    maximum = max(regrets)
    print(
        "CPU_FA2_POLICY_SUMMARY,"
        f"codegen_isa={next(iter(codegen_isas))},"
        f"runtime_isa={next(iter(isas))},domains={len(regrets)},"
        f"p50_regret_pct={p50:.3f},p95_regret_pct={p95:.3f},"
        f"max_regret_pct={maximum:.3f}"
    )
    for domain in sorted(
        all_domains,
        key=lambda item: item.regret_percent,
        reverse=True,
    )[:10]:
        print(
            "CPU_FA2_POLICY_TAIL,"
            f"regret_pct={domain.regret_percent:.3f},"
            f"codegen_isa={domain.codegen_isa},runtime_isa={domain.isa},"
            f"format={domain.format_name},head_dim={domain.head_dim},"
            f"heads={domain.heads},kv_heads={domain.kv_heads},"
            f"gqa_rep={domain.gqa_rep},sources={domain.sources},"
            f"regime={domain.regime},M={domain.query_rows},KV={domain.kv_rows}"
        )
    if p95 > maximum_p95_regret:
        print(
            f"CPU FA2 aggregate p95 regret {p95:.3f}% exceeds "
            f"{maximum_p95_regret:.3f}%",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line contract used by CTest and developers."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--cpu-list", help="explicit physical CPU list for taskset")
    parser.add_argument("--max-p95-regret", type=float, default=5.0)
    parser.add_argument(
        "--family-repeats",
        type=int,
        default=3,
        help="odd number of fresh worker processes per format/head family",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Validate arguments, resolve topology, and run aggregate certification."""

    arguments = build_parser().parse_args(argv)
    binary = arguments.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"tournament binary is not executable: {binary}")
    if arguments.max_p95_regret < 0.0:
        raise ValueError("--max-p95-regret must be non-negative")
    if arguments.family_repeats <= 0 or arguments.family_repeats % 2 == 0:
        raise ValueError("--family-repeats must be a positive odd number")
    worker_cpus = (
        parse_cpu_list(arguments.cpu_list)
        if arguments.cpu_list
        else discover_single_socket_physical_cpus()
    )
    return certify(
        binary,
        worker_cpus,
        arguments.max_p95_regret,
        arguments.family_repeats,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"CPU FA2 tournament driver failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
