"""Pure prefill timing-interval contract; execution always retains actual M.

These helpers validate explicitly new qualification inputs, never promote old
exact records or substitute for native byte, resource, profiler and timing gates.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

UINT32_MAX = 0xFFFFFFFF
_ALLOWED = {(128, 511), (512, UINT32_MAX), (128, 255), (256, 511),
            (512, 1023), (1024, UINT32_MAX)}

@dataclass(frozen=True, order=True)
class PrefillRowRegime:
    """Inclusive prefill interval; first_rows is the actually measured anchor."""
    first_rows: int
    last_rows: int

    def __post_init__(self) -> None:
        """Reject unsupported intervals instead of inventing a broader tuning scope."""
        if (type(self.first_rows) is not int or type(self.last_rows) is not int
                or self.scalars() not in _ALLOWED):
            raise ValueError("unsupported prefill timing interval")

    def scalars(self) -> tuple[int, int]:
        """Return the explicit generated interval without inferring a scope."""
        return self.first_rows, self.last_rows

    def contains(self, actual_rows: int) -> bool:
        """Test membership without projecting execution geometry onto the anchor."""
        if type(actual_rows) is not int or not 1 <= actual_rows <= UINT32_MAX:
            raise ValueError("actual prefill rows must be positive uint32")
        return self.first_rows <= actual_rows <= self.last_rows


def prefill_row_regime(actual_rows: int) -> PrefillRowRegime | None:
    """Return the requested default collection interval; below128 stays exact-M."""
    if type(actual_rows) is not int or not 1 <= actual_rows <= UINT32_MAX:
        raise ValueError("actual prefill rows must be positive uint32")
    if actual_rows < 128:
        return None
    return PrefillRowRegime(128, 511) if actual_rows < 512 else PrefillRowRegime(512, UINT32_MAX)


def validate_prefill_regime_check(
    regime: PrefillRowRegime, *, check_rows: int,
    anchor_identity: Sequence[object], check_identity: Sequence[object],
    chosen_candidate: str, required_candidates: Iterable[str],
    timings_us: Mapping[str, float],
) -> bool:
    """Require a same-identity actual-M control and report a >5% competitor win.

    Identities come from the existing validated raw evidence, excluding only M;
    all architecture/execution/numerical discriminators remain present. This is
    a timing check, not a new evidence authority. Missing candidates, differing
    classes and wrong check rows are diagnostic errors rather than passes.
    A true result requires an explicit split/recollection before interval install.
    """
    expected = { (128, 511): 256, (512, UINT32_MAX): 1024 }.get(regime.scalars())
    if type(check_rows) is not int or expected is None or check_rows != expected:
        raise ValueError("prefill regime requires its actual256/1024 check row")
    if (not anchor_identity or not check_identity
            or tuple(anchor_identity) != tuple(check_identity)):
        raise ValueError("prefill check changed its execution/architecture identity")
    candidates = tuple(required_candidates)
    if (not candidates or len(set(candidates)) != len(candidates)
            or any(not isinstance(c, str) or not c for c in candidates)
            or set(timings_us) != set(candidates) or chosen_candidate not in candidates):
        raise ValueError("prefill check is missing the chosen/competing candidate matrix")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in timings_us.values()):
        raise ValueError("prefill check requires positive finite native timings")
    chosen = timings_us[chosen_candidate]
    return any(candidate != chosen_candidate and value < chosen * 0.95
               for candidate, value in timings_us.items())


def prefill_regime_measurement_m_values(existing_values: Iterable[int]) -> tuple[int, ...]:
    """Project new collection onto fine rows below128, anchors and actual check rows."""
    values = tuple(existing_values)
    if any(type(m) is not int or not 1 <= m <= UINT32_MAX for m in values):
        raise ValueError("prefill measurement rows must be positive uint32")
    return tuple(sorted({m for m in values if m < 128} | {128, 256, 512, 1024}))
