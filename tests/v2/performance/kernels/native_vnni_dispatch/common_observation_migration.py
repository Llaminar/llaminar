"""Authenticate a deterministic NativeVNNI common-observation schema upgrade.

Timing corpora are expensive and immutable, while the derived common CSV schema
may gain learner metadata or deterministic launch features. This validator lets
an old corpus be re-adapted without pretending the derived files are bytewise
identical. Every historical shared field is compared exactly, row-for-row; only
the explicitly versioned learner label may change, and only the reviewed current
columns may be appended.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path


ALLOWED_ADDED_COLUMNS = (
    "launch_k_tiles",
    "launch_n_block_chunks",
    "adaptive_timing_evidence",
)
ALLOWED_CHANGED_COLUMNS = frozenset({"learner_version"})


@dataclass(frozen=True)
class CommonObservationMigrationReport:
    """Authenticated row count and the one permitted version transition."""

    row_count: int
    retained_learner_version: str
    replayed_learner_version: str


def validate_common_observation_migration(
    retained_path: Path,
    replayed_path: Path,
    *,
    require_learner_transition: bool = False,
) -> CommonObservationMigrationReport:
    """Prove that replay changed schema metadata but no measured evidence."""

    with retained_path.open(newline="", encoding="utf-8") as retained_handle, \
         replayed_path.open(newline="", encoding="utf-8") as replayed_handle:
        retained = csv.DictReader(retained_handle)
        replayed = csv.DictReader(replayed_handle)
        retained_fields = tuple(retained.fieldnames or ())
        replayed_fields = tuple(replayed.fieldnames or ())
        if not retained_fields or not replayed_fields:
            raise ValueError("common observation migration requires CSV headers")
        added = tuple(field for field in replayed_fields if field not in retained_fields)
        removed = tuple(field for field in retained_fields if field not in replayed_fields)
        if removed:
            raise ValueError(
                f"common observation schema removed retained columns: {removed}"
            )
        if added != ALLOWED_ADDED_COLUMNS:
            raise ValueError(
                "common observation schema added an unreviewed column set: "
                f"expected={ALLOWED_ADDED_COLUMNS} actual={added}"
            )
        shared_replayed_order = tuple(
            field for field in replayed_fields if field in retained_fields
        )
        if shared_replayed_order != retained_fields:
            raise ValueError("common observation schema reordered retained columns")

        retained_learner_versions: set[str] = set()
        replayed_learner_versions: set[str] = set()
        row_count = 0
        sentinel = object()
        for row_index, pair in enumerate(
            zip_longest(retained, replayed, fillvalue=sentinel),
            start=1,
        ):
            retained_row, replayed_row = pair
            if retained_row is sentinel or replayed_row is sentinel:
                raise ValueError(
                    "common observation migration changed row count at "
                    f"row {row_index}"
                )
            row_count += 1
            retained_learner_versions.add(retained_row["learner_version"])
            replayed_learner_versions.add(replayed_row["learner_version"])
            for field in retained_fields:
                if field in ALLOWED_CHANGED_COLUMNS:
                    continue
                if retained_row[field] != replayed_row[field]:
                    raise ValueError(
                        "common observation migration changed measured evidence: "
                        f"row={row_index} field={field} "
                        f"retained={retained_row[field]!r} "
                        f"replayed={replayed_row[field]!r}"
                    )

    if row_count == 0:
        raise ValueError("common observation migration has no rows")
    if len(retained_learner_versions) != 1 or len(replayed_learner_versions) != 1:
        raise ValueError("common observation learner versions are not uniform")
    retained_version = next(iter(retained_learner_versions))
    replayed_version = next(iter(replayed_learner_versions))
    if require_learner_transition and retained_version == replayed_version:
        raise ValueError("common observation replay did not change learner version")
    return CommonObservationMigrationReport(
        row_count=row_count,
        retained_learner_version=retained_version,
        replayed_learner_version=replayed_version,
    )


def main() -> int:
    """Validate one retained/replayed pair from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained", required=True, type=Path)
    parser.add_argument("--replayed", required=True, type=Path)
    parser.add_argument("--require-learner-transition", action="store_true")
    args = parser.parse_args()
    report = validate_common_observation_migration(
        args.retained,
        args.replayed,
        require_learner_transition=args.require_learner_transition,
    )
    print(
        "authenticated common-observation migration: "
        f"rows={report.row_count} "
        f"learner={report.retained_learner_version}->"
        f"{report.replayed_learner_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
