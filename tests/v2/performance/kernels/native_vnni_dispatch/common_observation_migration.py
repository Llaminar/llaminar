"""Authenticate a deterministic NativeVNNI common-observation replay.

Timing corpora are expensive and immutable, while the derived common CSV schema
may gain learner metadata or deterministic launch features. This validator lets
an old corpus be re-adapted without pretending the derived files are bytewise
identical. It also authenticates the narrower case in which an already-current
common corpus is rebound to a regenerated raw-corpus identity. Every measured
field is compared exactly, row-for-row; only explicitly reviewed metadata may
change, and only the reviewed current columns may be appended.
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
CURRENT_SCHEMA_REBIND_CHANGED_COLUMNS = frozenset({
    "corpus_id",
    "learner_version",
})


@dataclass(frozen=True)
class CommonObservationMigrationReport:
    """Authenticated row count and the one permitted version transition."""

    row_count: int
    retained_learner_version: str
    replayed_learner_version: str
    current_schema_rebind: bool


def finalize_common_observation_migration(
    retained_path: Path,
    replayed_path: Path,
    report: CommonObservationMigrationReport,
) -> bool:
    """Finalize an authenticated replay and preserve paid corpus identity.

    An already-current replay with an unchanged learner does not need any
    generated columns from ``replayed_path``. In that exact case, atomically
    restore the retained file at the canonical path so profiler request and
    feature digests remain bound to their original timing corpus. Historical
    schema upgrades keep the replayed file and discard the retained snapshot.

    Returns ``True`` when the retained identity was restored.
    """

    preserve_retained = (
        report.current_schema_rebind
        and report.retained_learner_version == report.replayed_learner_version
    )
    if preserve_retained:
        retained_path.replace(replayed_path)
    else:
        retained_path.unlink()
    return preserve_retained


def validate_common_observation_migration(
    retained_path: Path,
    replayed_path: Path,
    *,
    require_learner_transition: bool = False,
    allow_current_schema_rebind: bool = False,
    require_transition: bool = False,
) -> CommonObservationMigrationReport:
    """Prove that replay changed only reviewed metadata, never evidence.

    A historical schema upgrade must append exactly
    :data:`ALLOWED_ADDED_COLUMNS`. An already-current replay is accepted only
    when ``allow_current_schema_rebind`` is explicit; in that mode the schema
    must already contain every reviewed column and only ``corpus_id`` and
    ``learner_version`` may differ. The corpus identity is metadata here: the
    row-by-row comparison below independently proves that the adapted measured
    evidence did not change.
    """

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
        reviewed_schema_upgrade = added == ALLOWED_ADDED_COLUMNS
        current_schema_rebind = (
            allow_current_schema_rebind
            and not added
            and all(field in retained_fields for field in ALLOWED_ADDED_COLUMNS)
        )
        if not reviewed_schema_upgrade and not current_schema_rebind:
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
        transition_observed = bool(added)
        allowed_changed_columns = (
            CURRENT_SCHEMA_REBIND_CHANGED_COLUMNS
            if current_schema_rebind
            else ALLOWED_CHANGED_COLUMNS
        )
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
                if field in allowed_changed_columns:
                    transition_observed |= retained_row[field] != replayed_row[field]
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
    if require_transition and not transition_observed:
        raise ValueError("common observation replay did not change reviewed metadata")
    return CommonObservationMigrationReport(
        row_count=row_count,
        retained_learner_version=retained_version,
        replayed_learner_version=replayed_version,
        current_schema_rebind=current_schema_rebind,
    )


def main() -> int:
    """Validate one retained/replayed pair from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained", required=True, type=Path)
    parser.add_argument("--replayed", required=True, type=Path)
    parser.add_argument("--require-learner-transition", action="store_true")
    parser.add_argument("--allow-current-schema-rebind", action="store_true")
    parser.add_argument("--require-transition", action="store_true")
    parser.add_argument("--finalize-retained-identity", action="store_true")
    args = parser.parse_args()
    report = validate_common_observation_migration(
        args.retained,
        args.replayed,
        require_learner_transition=args.require_learner_transition,
        allow_current_schema_rebind=args.allow_current_schema_rebind,
        require_transition=args.require_transition,
    )
    print(
        "authenticated common-observation migration: "
        f"rows={report.row_count} "
        f"learner={report.retained_learner_version}->"
        f"{report.replayed_learner_version}"
    )
    if args.finalize_retained_identity:
        restored = finalize_common_observation_migration(
            args.retained,
            args.replayed,
            report,
        )
        print(
            "common-observation replay disposition: "
            + (
                "restored retained current-schema corpus identity"
                if restored
                else "kept regenerated schema-upgrade corpus"
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
