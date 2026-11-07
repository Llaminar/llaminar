"""@file host_cores.py
@brief One physical-core worker authority for offline NativeVNNI tooling.

Offline corpus hashing, policy fitting, bootstrap and profiler reduction run
one CPU-bound process per physical core: SMT siblings contend for the same
execution resources. A physical core is a kernel thread-sibling set within a
package. Presentation core labels are not identities: hybrid ARM hosts such as
GB10 repeat ``core_id`` across clusters, so counting ``(package, core_id)``
pairs there reports ten cores for twenty. Only the process affinity mask is
counted, so container and NUMA pinning remain authoritative. A platform
without complete topology metadata reports its visible logical count rather
than guessing an SMT ratio.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def physical_core_count(
    *,
    affinity: Iterable[int] | None = None,
    topology_root: Path = Path("/sys/devices/system/cpu"),
) -> int:
    """Return the affinity-visible physical cores, at least one.

    @param affinity Logical CPUs to count; the process affinity by default.
    @param topology_root Linux CPU sysfs root (a fixture seam for tests).
    @return Distinct ``(package, thread-sibling set)`` identities, or the
        visible logical count when any CPU lacks complete topology.
    """

    if affinity is None:
        try:
            visible_cpus = tuple(sorted(os.sched_getaffinity(0)))
        except AttributeError:
            visible_cpus = tuple(range(os.cpu_count() or 1))
    else:
        visible_cpus = tuple(sorted(set(affinity)))
    if not visible_cpus:
        return 1

    physical_cores: set[tuple[str, str]] = set()
    for cpu in visible_cpus:
        topology = topology_root / f"cpu{cpu}" / "topology"
        try:
            package = (topology / "physical_package_id").read_text(
                encoding="ascii"
            ).strip()
            siblings = (topology / "thread_siblings_list").read_text(
                encoding="ascii"
            ).strip()
        except (OSError, UnicodeError):
            return len(visible_cpus)
        if not package or not siblings:
            return len(visible_cpus)
        physical_cores.add((package, siblings))
    return max(1, len(physical_cores))
