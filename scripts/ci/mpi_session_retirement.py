## @file mpi_session_retirement.py
#  @brief Passive native-owner retirement checks at campaign scheduling boundaries.
"""The C++ process owner deletes backing; Python only observes its completed work."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time
from typing import Iterable, Iterator


def owned_session_bases(inventory: dict) -> dict[str, Path]:
    """Read the original roots from CMake's actual argv, never reconstruct their names."""
    result: dict[str, Path] = {}
    for test in inventory.get("tests", []):
        properties = {item["name"]: item["value"] for item in test.get("properties", [])}
        if "ProcessOwned" not in properties.get("LABELS", []):
            continue
        command = test.get("command", [])
        if (len(command) < 5 or Path(command[0]).name != "llaminar_mpi_session"
                or command[1] != "--session-base" or command[3] != "--"):
            raise RuntimeError(f"Missing declared native process owner: {test['name']}")
        base = Path(command[2])
        if not base.is_absolute() or not base.is_dir():
            raise RuntimeError(f"Unavailable declared native session base: {base}")
        if base in result.values():
            raise RuntimeError(f"Native test jobs share a session base: {base}")
        result[test["name"]] = base
    return result


def require_retired(bases: Iterable[Path], *, timeout_seconds: float = 36.0) -> None:
    """Wait for actual owner deletion; retain ambiguous files and fail admission.

    A killed CTest/proxy can leave its independent guardian retiring children.
    An empty process group alone cannot certify that boundary. The guardian
    removes shared backing before its private temporary root, and only after
    all actual descendants retire. This observer never deletes or signals.
    """
    roots = tuple(bases)
    deadline = time.monotonic() + timeout_seconds
    while True:
        pending = []
        for base in roots:
            if not base.is_dir():
                raise RuntimeError(f"Native session base disappeared: {base}")
            pending.extend(path for path in base.iterdir() if path.name.startswith("llaminar-mpi-"))
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Native process/file retirement is incomplete; backing retained: "
                               + ", ".join(str(path) for path in pending))
        time.sleep(0.05)


@contextmanager
def retirement_scope(bases: Iterable[Path]) -> Iterator[None]:
    """Exclude a dirty predecessor and await native cleanup even on cancellation/errors."""
    roots = tuple(bases)
    if not roots:
        yield
        return
    require_retired(roots, timeout_seconds=0.0)
    try:
        yield
    finally:
        require_retired(roots)
