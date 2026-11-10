"""Native GPU target identity for exact publication, separate from qualification.

Historical observation architecture labels remain readable. New exact entries
require an explicit native target and retain it in their runtime/retention key.
Parsing a target establishes identity only: neither an accepted spelling nor a
nearest-runtime selection certifies native measurements on that architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping


@dataclass(frozen=True, order=True)
class GPUArchitecture:
    """Lossless coordinates shared with NativeQuantisedArchitecture in C++."""

    backend: str
    major: int
    minor: int
    stepping: int = 0

    def __post_init__(self) -> None:
        """Reject malformed or backend-foreign coordinates before publication."""
        if self.backend not in {"cuda", "rocm"}:
            raise ValueError("GPU architecture requires CUDA or ROCm backend")
        if any(type(value) is not int or not 0 <= value <= 0xffffffff
               for value in (self.major, self.minor, self.stepping)) or self.major == 0:
            raise ValueError("GPU architecture requires positive native major and uint32 coordinates")
        if self.minor > (9 if self.backend == "cuda" else 15) or (self.backend == "cuda" and self.stepping != 0) or self.stepping > 15:
            raise ValueError("GPU architecture coordinates do not have a canonical native target")

    @classmethod
    def parse(cls, backend: str, target: str) -> "GPUArchitecture":
        """Read canonical native targets, rejecting historical/marketing labels.

        Discovery owns normalization of cached native metadata. Feature suffixes
        such as xnack remain physical receipt facts, rather than being silently
        discarded by an emitter while accepting an arbitrary architecture label.
        """
        if not isinstance(target, str):
            raise ValueError("GPU exact publication requires a canonical native target")
        if backend == "cuda":
            match = re.fullmatch(r"sm([1-9][0-9]*)([0-9])", target)
            if match:
                return cls(backend, int(match[1]), int(match[2]))
        elif backend == "rocm":
            match = re.fullmatch(r"gfx([1-9][0-9]*)([0-9a-f])([0-9a-f])", target)
            if match:
                return cls(backend, int(match[1]), int(match[2], 16), int(match[3], 16))
        raise ValueError(f"GPU exact publication requires canonical {backend} native architecture, got {target!r}")

    @property
    def native_target(self) -> str:
        """Return the exact native architecture without physical-device aliases."""
        if self.backend == "cuda":
            return f"sm{self.major}{self.minor}"
        return f"gfx{self.major}{self.minor:x}{self.stepping:x}"

    def cpp_literal(self) -> str:
        """Render the structured production factory without a lossy packed ID."""
        coordinates = (self.major, self.minor) if self.backend == "cuda" else (self.major, self.minor, self.stepping)
        return ("::llaminar2::native_quantised::NativeQuantisedArchitecture::" + self.backend + "(" +
                ", ".join(f"{value}u" for value in coordinates) + ")")

    def canonical_mapping(self) -> dict[str, object]:
        """Preserve both native spelling and typed coordinates in audit receipts."""
        return dict(backend=self.backend, major=self.major, minor=self.minor,
                    stepping=self.stepping, native_target=self.native_target)


def require_single_gpu_architecture(backend: str, targets: Iterable[str]) -> GPUArchitecture:
    """Reject mixed/absent target evidence before selecting or emitting winners."""
    architectures = {GPUArchitecture.parse(backend, target) for target in targets}
    if len(architectures) != 1:
        raise ValueError("one GPU exact collection must contain one authenticated native architecture")
    return next(iter(architectures))
