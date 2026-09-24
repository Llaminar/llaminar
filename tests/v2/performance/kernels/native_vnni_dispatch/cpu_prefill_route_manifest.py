"""Typed production-route evidence for the CPU ordinary-prefill planner.

The serial-M1 CPU tile configuration depends on the measured machine's cache
topology and OpenMP width.  The timing planner must therefore consume route
decisions emitted by the production C++ ``computeTileConfig`` helper instead of
copying that heuristic into Python.  This module validates and indexes those
small, timing-free manifests before any expensive corpus collection begins.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CPU_PREFILL_ROUTE_SCHEMA = "cpu-prefill-serial-route-v1"
CPU_PREFILL_FULL_K_BUNDLE = (
    "single-native-vnni-prefill:serial-full-k:fp32-output:v2"
)
CPU_PREFILL_KPART_BUNDLE = (
    "single-native-vnni-prefill:serial-kpart:fp32-output:v2"
)
CPU_PREFILL_ROUTE_BUNDLES = frozenset({
    CPU_PREFILL_FULL_K_BUNDLE,
    CPU_PREFILL_KPART_BUNDLE,
})

REQUIRED_COLUMNS = frozenset({
    "schema_version",
    "shape",
    "n",
    "k",
    "execution_codebook",
    "payload_bytes",
    "threads",
    "isa_regime",
    "k_tiles",
    "bundle_signature",
})


@dataclass(frozen=True, order=True)
class CPUPrefillSerialRoute:
    """One C++-observed serial arithmetic route on a CPU ISA surface."""

    execution_codebook: int
    shape_name: str
    n: int
    k: int
    isa_regime: str
    payload_bytes: int
    threads: int
    k_tiles: int
    bundle_signature: str

    @property
    def serial_kpart(self) -> bool:
        """Return whether grouped rows must inherit a serial K reduction."""

        return self.bundle_signature == CPU_PREFILL_KPART_BUNDLE


class CPUPrefillSerialRouteManifest:
    """Immutable, duplicate-free route index from one collection host."""

    def __init__(self, routes: Iterable[CPUPrefillSerialRoute]) -> None:
        indexed: dict[tuple[int, str, str], CPUPrefillSerialRoute] = {}
        for route in routes:
            key = (
                route.execution_codebook,
                route.shape_name,
                route.isa_regime,
            )
            if key in indexed:
                raise ValueError(f"duplicate CPU prefill serial route {key}")
            indexed[key] = route
        if not indexed:
            raise ValueError("CPU prefill serial route manifest is empty")
        self._routes = indexed
        self.thread_count()

    def route_for(
        self,
        execution_codebook: int,
        shape_name: str,
        isa_regime: str,
    ) -> CPUPrefillSerialRoute:
        """Resolve one required production route without a guessed default."""

        key = (execution_codebook, shape_name, isa_regime)
        try:
            return self._routes[key]
        except KeyError as error:
            raise ValueError(
                "CPU prefill serial route manifest lacks "
                f"codebook={execution_codebook} shape={shape_name} "
                f"regime={isa_regime}"
            ) from error

    def routes(self) -> tuple[CPUPrefillSerialRoute, ...]:
        """Return canonical manifest-order-independent route records."""

        return tuple(sorted(self._routes.values()))

    def thread_count(self) -> int:
        """Return the one OpenMP width shared by every measured route.

        CPU route selection depends on the OpenMP team width. Mixing manifests
        produced with different widths would make the route digest internally
        contradictory even when every individual row is otherwise valid.
        """

        counts = {route.threads for route in self._routes.values()}
        if len(counts) != 1:
            raise ValueError(
                "CPU prefill serial route manifests use inconsistent thread counts"
            )
        return next(iter(counts))

    def digest(self) -> str:
        """Bind a collection contract to exact C++ route evidence."""

        payload = [
            {
                "execution_codebook": route.execution_codebook,
                "shape": route.shape_name,
                "n": route.n,
                "k": route.k,
                "isa_regime": route.isa_regime,
                "payload_bytes": route.payload_bytes,
                "threads": route.threads,
                "k_tiles": route.k_tiles,
                "bundle_signature": route.bundle_signature,
            }
            for route in self.routes()
        ]
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


def read_cpu_prefill_route_manifests(
    paths: Iterable[Path],
) -> CPUPrefillSerialRouteManifest:
    """Read strict C++ route CSVs and reject contradictory route claims."""

    routes: list[CPUPrefillSerialRoute] = []
    for path in (Path(item) for item in paths):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path}: missing CPU prefill route columns {sorted(missing)}"
                )
            for row_number, raw in enumerate(reader, start=2):
                if raw["schema_version"].strip() != CPU_PREFILL_ROUTE_SCHEMA:
                    raise ValueError(
                        f"{path}:{row_number}: unsupported route schema"
                    )
                route = CPUPrefillSerialRoute(
                    execution_codebook=int(raw["execution_codebook"]),
                    shape_name=raw["shape"].strip(),
                    n=int(raw["n"]),
                    k=int(raw["k"]),
                    isa_regime=raw["isa_regime"].strip(),
                    payload_bytes=int(raw["payload_bytes"]),
                    threads=int(raw["threads"]),
                    k_tiles=int(raw["k_tiles"]),
                    bundle_signature=raw["bundle_signature"].strip(),
                )
                if (
                    not route.shape_name
                    or route.n <= 0
                    or route.k <= 0
                    or route.k % 32 != 0
                    or route.execution_codebook < 0
                    or route.execution_codebook > 255
                    or route.payload_bytes <= 0
                    or route.threads <= 0
                    or route.k_tiles < 0
                ):
                    raise ValueError(
                        f"{path}:{row_number}: invalid CPU prefill route values"
                    )
                if route.bundle_signature not in CPU_PREFILL_ROUTE_BUNDLES:
                    raise ValueError(
                        f"{path}:{row_number}: unknown arithmetic bundle"
                    )
                if route.serial_kpart != (route.k_tiles > 1):
                    raise ValueError(
                        f"{path}:{row_number}: K tiles disagree with bundle"
                    )
                routes.append(route)
    return CPUPrefillSerialRouteManifest(routes)
