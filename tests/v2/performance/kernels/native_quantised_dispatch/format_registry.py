"""Versioned source, prepared-family, and runtime NativeQuantised registry.

The old ``native_quantised_codebooks.py`` table represented one integer as both the
source tensor codebook and the runtime dispatch codebook. That is false for the
IQ4_XS family (source4, physical24) and raw INT8 family: GPU preparation normalizes Q8_0, Q8_1, and Q8_K to execution
codebook 19, while CPU retains source codebooks 19, 20, and 21. Keeping both
identities here prevents trainer rows from silently selecting a policy the
production runtime cannot distinguish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Iterable


FORMAT_REGISTRY_VERSION = "native-quantised-formats-v1"


@dataclass(frozen=True)
class NativeQuantisedFormatSpec:
    """One source tensor format and its backend preparation identities."""

    label: str
    source_codebook_id: int
    payload_bytes: int
    source_is_superblock: bool
    gpu_execution_codebook_id: int
    cpu_execution_codebook_id: int
    is_asymmetric: bool = False

    def runtime_codebook(self, backend: str) -> int:
        """Return the codebook visible to one production backend resolver."""

        normalized = backend.lower()
        if normalized in {"cuda", "rocm"}:
            return self.gpu_execution_codebook_id
        if normalized == "cpu":
            return self.cpu_execution_codebook_id
        raise ValueError(f"unknown NativeQuantised backend {backend!r}")

    def prepared_family(self, backend: str) -> str:
        """Return the runtime-representable prepared-family identifier."""

        codebook = self.runtime_codebook(backend)
        return f"NativeQuantised_{backend.lower()}_CB{codebook}"

    def arithmetic_policy_codebook(self, backend: str) -> int:
        """Return source-owned arithmetic identity without relabeling payloads.

        GPU IQ4_XS executes physical24 but preserves source4's reduction
        policy, as does IQ4_NL. Raw/prepared metadata remains physical24.
        Other registered GPU aliases already normalize to their arithmetic
        key (including Q8 aliases); CPU retains its original source codebook.
        """

        if backend.lower() in {"cuda", "rocm"} and self.source_codebook_id == 4:
            return self.source_codebook_id
        return self.runtime_codebook(backend)

    def packing_abi(self, backend: str) -> str:
        """Return the versioned normalized packing ABI for this backend."""

        codebook = self.runtime_codebook(backend)
        return f"native-quantised-{backend.lower()}-cb{codebook}-v1"


FORMAT_SPECS: tuple[NativeQuantisedFormatSpec, ...] = (
    NativeQuantisedFormatSpec("Q4_0", 0, 16, False, 0, 0),
    NativeQuantisedFormatSpec("IQ4_NL", 4, 16, False, 4, 4),
    NativeQuantisedFormatSpec("IQ4_XS", 4, 16, True, 24, 4),
    NativeQuantisedFormatSpec("Q4_1", 5, 16, False, 5, 5, True),
    NativeQuantisedFormatSpec("Q4_K", 5, 16, True, 5, 5, True),
    NativeQuantisedFormatSpec("Q5_0", 6, 20, False, 6, 6),
    NativeQuantisedFormatSpec("Q5_1", 7, 20, False, 7, 7, True),
    NativeQuantisedFormatSpec("Q5_K", 7, 20, True, 7, 7, True),
    NativeQuantisedFormatSpec("Q6_K", 8, 24, True, 8, 8),
    NativeQuantisedFormatSpec("Q3_K", 9, 12, True, 9, 9),
    NativeQuantisedFormatSpec("Q2_K", 10, 8, True, 10, 10, True),
    NativeQuantisedFormatSpec("IQ3_S", 11, 13, True, 11, 11),
    NativeQuantisedFormatSpec("IQ3_XXS", 12, 12, True, 12, 12),
    NativeQuantisedFormatSpec("IQ2_S", 13, 9, True, 13, 13),
    NativeQuantisedFormatSpec("IQ2_XS", 14, 9, True, 14, 14),
    NativeQuantisedFormatSpec("IQ2_XXS", 15, 8, True, 15, 15),
    NativeQuantisedFormatSpec("IQ1_S", 16, 6, True, 16, 16, True),
    NativeQuantisedFormatSpec("IQ1_M", 17, 6, True, 17, 17),
    NativeQuantisedFormatSpec("Q8_0", 19, 32, False, 19, 19),
    NativeQuantisedFormatSpec("Q8_1", 20, 32, False, 19, 20),
    NativeQuantisedFormatSpec("Q8_K", 21, 32, True, 19, 21),
)

FORMAT_BY_LABEL = {spec.label: spec for spec in FORMAT_SPECS}

# Compatibility views used by existing backend analyzers during migration.
# ``FORMAT_TO_CODEBOOK`` deliberately remains the source codebook identity.
FORMAT_TO_CODEBOOK = OrderedDict(
    (spec.label, spec.source_codebook_id) for spec in FORMAT_SPECS
)
CODEBOOK_TO_FORMAT = {
    codebook: "/".join(spec.label for spec in FORMAT_SPECS
                       if spec.source_codebook_id == codebook)
    for codebook in sorted({spec.source_codebook_id for spec in FORMAT_SPECS})
}
CODEBOOK_PAYLOAD_BYTES = {
    spec.source_codebook_id: spec.payload_bytes for spec in FORMAT_SPECS
}
GPU_FORMAT_TO_EXECUTION_CODEBOOK = OrderedDict(
    (spec.label, spec.gpu_execution_codebook_id) for spec in FORMAT_SPECS
)


def format_spec(label: str) -> NativeQuantisedFormatSpec:
    """Resolve a canonical source-format label or fail loudly."""

    normalized = label.strip().upper()
    try:
        return FORMAT_BY_LABEL[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown NativeQuantised source format {label!r}") from exc


def runtime_aliases(backend: str, runtime_codebook: int) -> tuple[str, ...]:
    """Return every source format indistinguishable at a backend dispatch key."""

    return tuple(
        spec.label
        for spec in FORMAT_SPECS
        if spec.runtime_codebook(backend) == runtime_codebook
    )


def arithmetic_policy_aliases(backend: str, policy_codebook: int) -> tuple[str, ...]:
    """Return source aliases sharing one queried arithmetic policy key."""

    return tuple(spec.label for spec in FORMAT_SPECS
                 if spec.arithmetic_policy_codebook(backend) == policy_codebook)


def registry_digest(specs: Iterable[NativeQuantisedFormatSpec] = FORMAT_SPECS) -> str:
    """Hash the ordered registry for policy provenance and stale-table checks."""

    payload = {
        "version": FORMAT_REGISTRY_VERSION,
        "formats": [asdict(spec) for spec in specs],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def infer_format_from_filename(path) -> str | None:
    """Infer a format label from a sweep filename, preferring longer names."""

    name = path.stem.lower().replace("_", "")
    for candidate in sorted(FORMAT_BY_LABEL, key=len, reverse=True):
        if candidate.lower().replace("_", "") in name:
            return candidate
    return None


def main() -> int:
    """Expose the canonical all-format inventory to shell transactions."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Print the ordered comma-separated production format inventory",
    )
    args = parser.parse_args()
    if not args.labels:
        parser.error("use --labels")
    print(",".join(spec.label for spec in FORMAT_SPECS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
