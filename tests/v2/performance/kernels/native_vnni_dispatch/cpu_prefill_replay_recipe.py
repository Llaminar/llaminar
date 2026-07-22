"""Authenticate and expand one complete CPU-prefill policy replay recipe.

CPU NativeVNNI prefill tuning can accumulate several immutable evidence
generations: an original timing corpus, an anchored candidate expansion, a
split migration, generic-refinement rounds, and isolated profiler launches.
Passing those artifacts as an untyped list of shell options is both difficult
to review and unsafe. A path can be perfectly valid while playing the wrong
role in the transaction.

This module makes that lineage explicit. The recipe binds every artifact by
SHA-256, keeps historical lineage plans separate from current refinement
plans, authenticates their embedded relationships, and states whether the
large common-observation checkpoint is absent, a real prefix, or complete.
The refresh script consumes the emitted typed records; operators do not
reconstruct the historical command line by hand.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

from .adapters.evidence import raw_corpus_id
from .cpu_prefill_candidate_expansion import (
    read_cpu_prefill_candidate_expansion_plan,
)
from .cpu_prefill_burned_seal import (
    read_cpu_prefill_burned_seal_transaction,
)
from .cpu_prefill_development_lineage import (
    read_cpu_prefill_development_lineage_plan,
)
from .cpu_prefill_generic_refinement import (
    CPUPrefillGenericRefinementPlan,
    read_cpu_prefill_generic_refinement_plan,
)
from .cpu_prefill_route_manifest import read_cpu_prefill_route_manifests
from .cpu_prefill_split_manifest import load_cpu_prefill_split_manifest
from .cpu_prefill_training_plan import (
    validate_cpu_prefill_serial_route_manifest,
)
from .candidate_observation import read_observation_csv
from .profiler_evidence import (
    compose_profiler_evidence,
    read_profiler_evidence_manifest,
    read_profiler_request_manifest,
)


SCHEMA_VERSION = "native-vnni-cpu-prefill-replay-recipe-v4"
CHECKPOINT_STATES = frozenset(("rebuild", "prefix", "complete"))
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


class CPUPrefillReplayRecipeError(ValueError):
    """Report an invalid, incomplete, or unauthenticated replay recipe."""


def _sha256(path: Path) -> str:
    """Return one streaming content identity without copying a large corpus."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(value: object) -> str:
    """Return the deterministic identity of one JSON-compatible value."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require_mapping(
    raw: object,
    required: set[str],
    context: str,
) -> Mapping[str, object]:
    """Return a strict JSON object or explain its exact schema mismatch."""

    if not isinstance(raw, dict) or set(raw) != required:
        actual = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise CPUPrefillReplayRecipeError(
            f"{context} schema is invalid: expected={sorted(required)} "
            f"actual={actual}"
        )
    return raw


def _require_sequence(raw: object, context: str) -> Sequence[object]:
    """Return a JSON list while rejecting strings and mapping iteration."""

    if not isinstance(raw, list):
        raise CPUPrefillReplayRecipeError(f"{context} must be a list")
    return raw


def _safe_relative_path(raw: object, context: str) -> PurePosixPath:
    """Parse a portable path and reject absolute paths and control bytes."""

    if not isinstance(raw, str) or not raw or "\0" in raw:
        raise CPUPrefillReplayRecipeError(f"{context} path is invalid")
    if "\n" in raw or "\r" in raw or "\t" in raw:
        raise CPUPrefillReplayRecipeError(
            f"{context} path contains a shell-record delimiter"
        )
    path = PurePosixPath(raw)
    if path.is_absolute():
        raise CPUPrefillReplayRecipeError(
            f"{context} path must be relative to its declared base"
        )
    return path


@dataclass(frozen=True)
class Artifact:
    """One immutable recipe- or repository-relative evidence payload."""

    base: str
    path: PurePosixPath
    sha256: str

    @classmethod
    def from_json(cls, raw: object, context: str) -> "Artifact":
        """Parse one strict content-addressed artifact record."""

        value = _require_mapping(raw, {"base", "path", "sha256"}, context)
        base = value["base"]
        if base not in {"recipe", "repository"}:
            raise CPUPrefillReplayRecipeError(
                f"{context} base must be 'recipe' or 'repository'"
            )
        digest = value["sha256"]
        if (
            not isinstance(digest, str)
            or not digest.startswith("sha256:")
            or len(digest) != 71
            or any(
                character not in "0123456789abcdef"
                for character in digest.removeprefix("sha256:")
            )
        ):
            raise CPUPrefillReplayRecipeError(
                f"{context} SHA-256 identity is invalid"
            )
        return cls(
            base=str(base),
            path=_safe_relative_path(value["path"], context),
            sha256=digest,
        )

    def resolve(self, recipe_path: Path) -> Path:
        """Resolve this payload against its declared, reviewable root."""

        root = recipe_path.resolve().parent if self.base == "recipe" else REPOSITORY_ROOT
        return root.joinpath(*self.path.parts).resolve()

    def canonical_mapping(self) -> dict[str, str]:
        """Return the stable representation included in the recipe digest."""

        return {
            "base": self.base,
            "path": self.path.as_posix(),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class AdditiveTransaction:
    """One complete-registry timing addition and its optional current plan."""

    aggregate: Artifact
    timing: Artifact
    generic_refinement_plan: Artifact | None


@dataclass(frozen=True)
class ProfilerTransaction:
    """One isolated per-physical-candidate profiler evidence transaction."""

    requests: Artifact
    evidence: Artifact
    observations: Artifact


@dataclass(frozen=True)
class Checkpoint:
    """Explicit lifecycle state for the large adapted observation checkpoint."""

    state: str
    path: PurePosixPath
    sha256: str | None
    raw_corpus_id: str | None
    prefix_input_count: int | None


@dataclass(frozen=True)
class CPUPrefillReplayRecipe:
    """The complete typed provenance needed for one fit-only CPU replay."""

    path: Path
    recipe_digest: str
    target_split_manifest: Artifact
    target_route_manifests: tuple[Artifact, ...]
    sealed_candidate_route_manifest: Artifact
    candidate_plan: Artifact
    candidate_source_aggregate: Artifact
    candidate_source_timing: Artifact
    candidate_expansion_aggregate: Artifact
    candidate_expansion_timing: Artifact
    additives: tuple[AdditiveTransaction, ...]
    primary_profiler_transaction: ProfilerTransaction
    profiler_transactions: tuple[ProfilerTransaction, ...]
    burned_seal_development_transactions: tuple[Artifact, ...]
    lineage_plan: Artifact
    lineage_source_aggregate: Artifact
    lineage_source_timing: Artifact
    lineage_source_split_manifest: Artifact
    lineage_source_route_manifests: tuple[Artifact, ...]
    lineage_historical_refinement_plans: tuple[Artifact, ...]
    lineage_additional_split_manifests: tuple[Artifact, ...]
    checkpoint: Checkpoint

    def development_aggregates(self) -> tuple[Path, ...]:
        """Return the ordered raw aggregates consumed by the adapter."""

        return (
            self.candidate_source_aggregate.resolve(self.path),
            *(item.aggregate.resolve(self.path) for item in self.additives),
        )

    def development_timings(self) -> tuple[Path, ...]:
        """Return timing sidecars in the exact aggregate order."""

        return (
            self.candidate_source_timing.resolve(self.path),
            *(item.timing.resolve(self.path) for item in self.additives),
        )

    def context_corpus_id(self, input_count: int | None = None) -> str:
        """Reproduce the adapter identity for a complete or prefix checkpoint."""

        aggregates = self.development_aggregates()
        timings = self.development_timings()
        selected = len(aggregates) if input_count is None else input_count
        if not 0 < selected <= len(aggregates):
            raise CPUPrefillReplayRecipeError(
                "checkpoint input count is outside the development transaction"
            )
        return raw_corpus_id((
            *aggregates[:selected],
            self.candidate_expansion_aggregate.resolve(self.path),
            *timings[:selected],
            self.candidate_expansion_timing.resolve(self.path),
        ))

    def checkpoint_path(self) -> Path:
        """Resolve the mutable checkpoint destination owned by this recipe."""

        return self.path.resolve().parent.joinpath(
            *self.checkpoint.path.parts
        ).resolve()


def _parse_artifact_list(
    raw: object,
    context: str,
    *,
    allow_empty: bool = False,
) -> tuple[Artifact, ...]:
    """Parse a list of immutable artifacts with stable indexed diagnostics."""

    values = _require_sequence(raw, context)
    if not values and not allow_empty:
        raise CPUPrefillReplayRecipeError(f"{context} must not be empty")
    return tuple(
        Artifact.from_json(item, f"{context}[{index}]")
        for index, item in enumerate(values)
    )


def _parse_checkpoint(raw: object) -> Checkpoint:
    """Parse checkpoint state without inferring lifecycle from file existence."""

    value = _require_mapping(
        raw,
        {"state", "path", "sha256", "raw_corpus_id", "prefix_input_count"},
        "checkpoint",
    )
    state = value["state"]
    if state not in CHECKPOINT_STATES:
        raise CPUPrefillReplayRecipeError(
            f"checkpoint state must be one of {sorted(CHECKPOINT_STATES)}"
        )
    digest = value["sha256"]
    corpus_id = value["raw_corpus_id"]
    prefix_count = value["prefix_input_count"]
    if state == "rebuild":
        if digest is not None or corpus_id is not None or prefix_count is not None:
            raise CPUPrefillReplayRecipeError(
                "rebuild checkpoint must not claim prior evidence identity"
            )
    else:
        for name, candidate in (("sha256", digest), ("raw_corpus_id", corpus_id)):
            if not isinstance(candidate, str) or not candidate.startswith("sha256:"):
                raise CPUPrefillReplayRecipeError(
                    f"{state} checkpoint requires a canonical {name}"
                )
        if state == "complete" and prefix_count is not None:
            raise CPUPrefillReplayRecipeError(
                "complete checkpoint cannot declare a prefix input count"
            )
        if state == "prefix" and (
            type(prefix_count) is not int or prefix_count <= 0
        ):
            raise CPUPrefillReplayRecipeError(
                "prefix checkpoint requires a positive input count"
            )
    return Checkpoint(
        state=str(state),
        path=_safe_relative_path(value["path"], "checkpoint"),
        sha256=None if digest is None else str(digest),
        raw_corpus_id=None if corpus_id is None else str(corpus_id),
        prefix_input_count=None if prefix_count is None else int(prefix_count),
    )


def load_recipe(path: Path) -> CPUPrefillReplayRecipe:
    """Parse one strict replay recipe and authenticate its self-digest."""

    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CPUPrefillReplayRecipeError(
            f"cannot read CPU prefill replay recipe {path}: {error}"
        ) from error
    required = {
        "schema_version",
        "recipe_digest",
        "target",
        "candidate_expansion",
        "additive_transactions",
        "primary_profiler_transaction",
        "profiler_transactions",
        "burned_seal_development_transactions",
        "development_lineage",
        "checkpoint",
    }
    root = _require_mapping(raw, required, "recipe root")
    if root["schema_version"] != SCHEMA_VERSION:
        raise CPUPrefillReplayRecipeError("unsupported CPU prefill replay recipe")
    authenticated = dict(root)
    claimed_digest = authenticated.pop("recipe_digest")
    if claimed_digest != _canonical_digest(authenticated):
        raise CPUPrefillReplayRecipeError("CPU prefill replay recipe digest changed")

    target = _require_mapping(
        root["target"],
        {
            "split_manifest",
            "route_manifests",
            "sealed_candidate_route_manifest",
        },
        "target",
    )
    expansion = _require_mapping(
        root["candidate_expansion"],
        {
            "plan",
            "source_aggregate",
            "source_timing",
            "expansion_aggregate",
            "expansion_timing",
        },
        "candidate_expansion",
    )
    lineage = _require_mapping(
        root["development_lineage"],
        {
            "plan",
            "source_aggregate",
            "source_timing",
            "source_split_manifest",
            "source_route_manifests",
            "historical_refinement_plans",
            "additional_split_manifests",
        },
        "development_lineage",
    )

    additives = []
    for index, item in enumerate(_require_sequence(
        root["additive_transactions"], "additive_transactions"
    )):
        value = _require_mapping(
            item,
            {"aggregate", "timing", "generic_refinement_plan"},
            f"additive_transactions[{index}]",
        )
        plan = value["generic_refinement_plan"]
        additives.append(AdditiveTransaction(
            aggregate=Artifact.from_json(
                value["aggregate"], f"additive_transactions[{index}].aggregate"
            ),
            timing=Artifact.from_json(
                value["timing"], f"additive_transactions[{index}].timing"
            ),
            generic_refinement_plan=(
                None
                if plan is None
                else Artifact.from_json(
                    plan,
                    f"additive_transactions[{index}].generic_refinement_plan",
                )
            ),
        ))
    if not additives:
        raise CPUPrefillReplayRecipeError(
            "CPU prefill replay recipe has no additive transactions"
        )

    primary_profiler_value = _require_mapping(
        root["primary_profiler_transaction"],
        {"requests", "evidence", "observations"},
        "primary_profiler_transaction",
    )
    primary_profiler = ProfilerTransaction(
        requests=Artifact.from_json(
            primary_profiler_value["requests"],
            "primary_profiler_transaction.requests",
        ),
        evidence=Artifact.from_json(
            primary_profiler_value["evidence"],
            "primary_profiler_transaction.evidence",
        ),
        observations=Artifact.from_json(
            primary_profiler_value["observations"],
            "primary_profiler_transaction.observations",
        ),
    )

    profilers = []
    for index, item in enumerate(_require_sequence(
        root["profiler_transactions"], "profiler_transactions"
    )):
        value = _require_mapping(
            item,
            {"requests", "evidence", "observations"},
            f"profiler_transactions[{index}]",
        )
        profilers.append(ProfilerTransaction(
            requests=Artifact.from_json(
                value["requests"], f"profiler_transactions[{index}].requests"
            ),
            evidence=Artifact.from_json(
                value["evidence"], f"profiler_transactions[{index}].evidence"
            ),
            observations=Artifact.from_json(
                value["observations"],
                f"profiler_transactions[{index}].observations",
            ),
        ))

    return CPUPrefillReplayRecipe(
        path=path,
        recipe_digest=str(claimed_digest),
        target_split_manifest=Artifact.from_json(
            target["split_manifest"], "target.split_manifest"
        ),
        target_route_manifests=_parse_artifact_list(
            target["route_manifests"], "target.route_manifests"
        ),
        sealed_candidate_route_manifest=Artifact.from_json(
            target["sealed_candidate_route_manifest"],
            "target.sealed_candidate_route_manifest",
        ),
        candidate_plan=Artifact.from_json(expansion["plan"], "candidate_expansion.plan"),
        candidate_source_aggregate=Artifact.from_json(
            expansion["source_aggregate"], "candidate_expansion.source_aggregate"
        ),
        candidate_source_timing=Artifact.from_json(
            expansion["source_timing"], "candidate_expansion.source_timing"
        ),
        candidate_expansion_aggregate=Artifact.from_json(
            expansion["expansion_aggregate"],
            "candidate_expansion.expansion_aggregate",
        ),
        candidate_expansion_timing=Artifact.from_json(
            expansion["expansion_timing"], "candidate_expansion.expansion_timing"
        ),
        additives=tuple(additives),
        primary_profiler_transaction=primary_profiler,
        profiler_transactions=tuple(profilers),
        burned_seal_development_transactions=_parse_artifact_list(
            root["burned_seal_development_transactions"],
            "burned_seal_development_transactions",
            allow_empty=True,
        ),
        lineage_plan=Artifact.from_json(lineage["plan"], "development_lineage.plan"),
        lineage_source_aggregate=Artifact.from_json(
            lineage["source_aggregate"], "development_lineage.source_aggregate"
        ),
        lineage_source_timing=Artifact.from_json(
            lineage["source_timing"], "development_lineage.source_timing"
        ),
        lineage_source_split_manifest=Artifact.from_json(
            lineage["source_split_manifest"],
            "development_lineage.source_split_manifest",
        ),
        lineage_source_route_manifests=_parse_artifact_list(
            lineage["source_route_manifests"],
            "development_lineage.source_route_manifests",
        ),
        lineage_historical_refinement_plans=_parse_artifact_list(
            lineage["historical_refinement_plans"],
            "development_lineage.historical_refinement_plans",
        ),
        lineage_additional_split_manifests=_parse_artifact_list(
            lineage["additional_split_manifests"],
            "development_lineage.additional_split_manifests",
            allow_empty=True,
        ),
        checkpoint=_parse_checkpoint(root["checkpoint"]),
    )


def _all_artifacts(recipe: CPUPrefillReplayRecipe) -> tuple[Artifact, ...]:
    """Enumerate each immutable recipe payload exactly once for hashing."""

    artifacts = [
        recipe.target_split_manifest,
        *recipe.target_route_manifests,
        recipe.sealed_candidate_route_manifest,
        recipe.candidate_plan,
        recipe.candidate_source_aggregate,
        recipe.candidate_source_timing,
        recipe.candidate_expansion_aggregate,
        recipe.candidate_expansion_timing,
        recipe.lineage_plan,
        recipe.lineage_source_aggregate,
        recipe.lineage_source_timing,
        recipe.lineage_source_split_manifest,
        *recipe.lineage_source_route_manifests,
        *recipe.lineage_historical_refinement_plans,
        *recipe.lineage_additional_split_manifests,
        *recipe.burned_seal_development_transactions,
        recipe.primary_profiler_transaction.requests,
        recipe.primary_profiler_transaction.evidence,
        recipe.primary_profiler_transaction.observations,
    ]
    for item in recipe.additives:
        artifacts.extend((item.aggregate, item.timing))
        if item.generic_refinement_plan is not None:
            artifacts.append(item.generic_refinement_plan)
    for item in recipe.profiler_transactions:
        artifacts.extend((item.requests, item.evidence, item.observations))
    unique: dict[tuple[str, str], Artifact] = {}
    for artifact in artifacts:
        key = artifact.base, artifact.path.as_posix()
        prior = unique.get(key)
        if prior is not None and prior.sha256 != artifact.sha256:
            raise CPUPrefillReplayRecipeError(
                f"recipe repeats {artifact.path} with conflicting digests"
            )
        unique[key] = artifact
    return tuple(unique.values())


def _read_historical_plans(
    recipe: CPUPrefillReplayRecipe,
    route_manifest: object,
    split_manifests: Iterable[object],
) -> tuple[CPUPrefillGenericRefinementPlan, ...]:
    """Authenticate each historical plan against its own reviewed split."""

    by_digest = {}
    for manifest in split_manifests:
        for digest in manifest.accepted_digests():
            if digest in by_digest and by_digest[digest] is not manifest:
                raise CPUPrefillReplayRecipeError(
                    f"duplicate lineage source split digest: {digest}"
                )
            by_digest[digest] = manifest
    plans = []
    for artifact in recipe.lineage_historical_refinement_plans:
        path = artifact.resolve(recipe.path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        split_digest = raw.get("split_manifest_digest") if isinstance(raw, dict) else None
        if split_digest not in by_digest:
            raise CPUPrefillReplayRecipeError(
                f"{path}: no declared historical split matches {split_digest}"
            )
        plans.append(read_cpu_prefill_generic_refinement_plan(
            path, route_manifest, by_digest[split_digest]
        ))
    return tuple(plans)


def _checkpoint_first_corpus_id(path: Path) -> str:
    """Read only the first adapted row to authenticate checkpoint ownership."""

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
    except (OSError, StopIteration, csv.Error) as error:
        raise CPUPrefillReplayRecipeError(
            f"cannot read checkpoint transaction identity from {path}: {error}"
        ) from error
    corpus_id = row.get("corpus_id", "")
    if not corpus_id.startswith("sha256:"):
        raise CPUPrefillReplayRecipeError(
            f"checkpoint {path} has no canonical corpus identity"
        )
    return corpus_id


def _checkpoint_candidate_plan_digest(path: Path) -> str:
    """Return the expansion plan actually embedded in normalized checkpoint rows.

    The first expansion row can occur well into a complete checkpoint, but
    scanning its compact CSV is still much cheaper than loading and validating
    every observation. Preflight uses this identity to reject a checkpoint
    paired with a different candidate-expansion generation before adaptation.
    """

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw = row.get("adaptive_timing_evidence", "")
                if "candidate_expansion_normalization" not in raw:
                    continue
                evidence = json.loads(raw)
                proof = evidence.get("candidate_expansion_normalization")
                digest = proof.get("plan_digest") if isinstance(proof, dict) else None
                if not isinstance(digest, str) or not digest.startswith("sha256:"):
                    raise CPUPrefillReplayRecipeError(
                        f"checkpoint {path} has malformed expansion provenance"
                    )
                return digest
    except (OSError, csv.Error, json.JSONDecodeError) as error:
        raise CPUPrefillReplayRecipeError(
            f"cannot inspect checkpoint expansion provenance in {path}: {error}"
        ) from error
    raise CPUPrefillReplayRecipeError(
        f"checkpoint {path} contains no candidate-expansion provenance"
    )


def authenticate_recipe(recipe: CPUPrefillReplayRecipe) -> str:
    """Authenticate all artifacts and return the checkpoint replay action.

    The result is ``complete`` or ``prefix`` when the checkpoint belongs to
    the raw paths in this workspace. A content-authenticated checkpoint copied
    to another workspace receives ``rebase`` because the historical adapter
    corpus identity deliberately includes source paths. Rebase validates the
    complete plan partition and changes provenance only; it does not reinterpret
    historical timing under a newer adapter implementation.
    """

    for artifact in _all_artifacts(recipe):
        path = artifact.resolve(recipe.path)
        if path.is_symlink() or not path.is_file():
            raise CPUPrefillReplayRecipeError(
                f"CPU prefill replay artifact is missing: {path}"
            )
        actual = _sha256(path)
        if actual != artifact.sha256:
            raise CPUPrefillReplayRecipeError(
                f"CPU prefill replay artifact digest changed: {path}"
            )

    target_split = load_cpu_prefill_split_manifest(
        recipe.target_split_manifest.resolve(recipe.path)
    )
    target_routes = read_cpu_prefill_route_manifests(
        artifact.resolve(recipe.path) for artifact in recipe.target_route_manifests
    )
    sealed_candidate_routes = read_cpu_prefill_route_manifests((
        recipe.sealed_candidate_route_manifest.resolve(recipe.path),
    ))
    if any(
        not route.shape_name.startswith("CPUPrefillAutoRefine_")
        for route in sealed_candidate_routes.routes()
    ):
        raise CPUPrefillReplayRecipeError(
            "sealed CPU prefill candidate routes contain a declared model shape"
        )
    try:
        validate_cpu_prefill_serial_route_manifest(target_routes)
    except ValueError as error:
        raise CPUPrefillReplayRecipeError(
            "target CPU prefill route manifest is incomplete for the current "
            f"production matrix: {error}"
        ) from error
    source_split = load_cpu_prefill_split_manifest(
        recipe.lineage_source_split_manifest.resolve(recipe.path)
    )
    additional_splits = tuple(
        load_cpu_prefill_split_manifest(artifact.resolve(recipe.path))
        for artifact in recipe.lineage_additional_split_manifests
    )
    source_routes = read_cpu_prefill_route_manifests(
        artifact.resolve(recipe.path)
        for artifact in recipe.lineage_source_route_manifests
    )
    historical_plans = _read_historical_plans(
        recipe,
        source_routes,
        (source_split, *additional_splits),
    )
    read_cpu_prefill_development_lineage_plan(
        recipe.lineage_plan.resolve(recipe.path),
        recipe.lineage_source_aggregate.resolve(recipe.path),
        recipe.lineage_source_timing.resolve(recipe.path),
        source_split,
        target_split,
        source_routes,
        target_routes,
        historical_plans,
    )
    candidate_plan = read_cpu_prefill_candidate_expansion_plan(
        recipe.candidate_plan.resolve(recipe.path),
        source_aggregate=recipe.candidate_source_aggregate.resolve(recipe.path),
        source_timing=recipe.candidate_source_timing.resolve(recipe.path),
        collection_build_digest=None,
        authenticate_current_implementation=False,
    )
    burned_transactions = tuple(
        read_cpu_prefill_burned_seal_transaction(
            artifact.resolve(recipe.path)
        )
        for artifact in recipe.burned_seal_development_transactions
    )
    if len({item.transaction_digest for item in burned_transactions}) != len(
        burned_transactions
    ):
        raise CPUPrefillReplayRecipeError(
            "burned-seal development transactions are duplicated"
        )

    current_plans = tuple(
        read_cpu_prefill_generic_refinement_plan(
            item.generic_refinement_plan.resolve(recipe.path),
            target_routes,
            target_split,
        )
        for item in recipe.additives
        if item.generic_refinement_plan is not None
    )
    historical_digests = {item.digest() for item in historical_plans}
    current_digests = {item.digest() for item in current_plans}
    duplicate_roles = historical_digests & current_digests
    if duplicate_roles:
        raise CPUPrefillReplayRecipeError(
            "refinement plans appear in both historical and current roles: "
            f"{sorted(duplicate_roles)}"
        )
    if len(current_digests) != len(current_plans):
        raise CPUPrefillReplayRecipeError(
            "current generic-refinement plans contain duplicate transactions"
        )

    checkpoint = recipe.checkpoint
    if checkpoint.state == "rebuild":
        return "rebuild"
    path = recipe.checkpoint_path()
    if path.is_symlink() or not path.is_file():
        raise CPUPrefillReplayRecipeError(
            f"declared {checkpoint.state} checkpoint is missing: {path}"
        )
    if _sha256(path) != checkpoint.sha256:
        raise CPUPrefillReplayRecipeError(
            f"declared {checkpoint.state} checkpoint digest changed: {path}"
        )
    embedded_candidate_plan = _checkpoint_candidate_plan_digest(path)
    if embedded_candidate_plan != candidate_plan.digest():
        raise CPUPrefillReplayRecipeError(
            "checkpoint candidate-expansion plan differs from replay recipe: "
            f"checkpoint={embedded_candidate_plan} "
            f"recipe={candidate_plan.digest()}"
        )
    represented_count = (
        checkpoint.prefix_input_count
        if checkpoint.state == "prefix"
        else len(recipe.development_aggregates())
    )
    expected = recipe.context_corpus_id(represented_count)
    observed = _checkpoint_first_corpus_id(path)
    if observed == expected:
        return checkpoint.state
    if observed == checkpoint.raw_corpus_id:
        return "rebase"
    raise CPUPrefillReplayRecipeError(
        "checkpoint corpus identity changed independently of its recipe"
    )


def shell_records(
    recipe: CPUPrefillReplayRecipe,
    checkpoint_action: str,
) -> tuple[tuple[str, str], ...]:
    """Expand a recipe into typed records consumed without shell evaluation."""

    records: list[tuple[str, str]] = [
        ("recipe_digest", recipe.recipe_digest),
        ("checkpoint_action", checkpoint_action),
        ("checkpoint_path", str(recipe.checkpoint_path())),
        (
            "context_corpus_id",
            (
                str(recipe.checkpoint.raw_corpus_id)
                if checkpoint_action == "complete"
                else recipe.context_corpus_id()
            ),
        ),
        (
            "target_split_manifest",
            str(recipe.target_split_manifest.resolve(recipe.path)),
        ),
    ]
    if checkpoint_action == "prefix":
        records.append((
            "checkpoint_prefix_input_count",
            str(recipe.checkpoint.prefix_input_count),
        ))
    records.extend(
        ("target_route_manifest", str(item.resolve(recipe.path)))
        for item in recipe.target_route_manifests
    )
    records.append((
        "sealed_candidate_route_manifest",
        str(recipe.sealed_candidate_route_manifest.resolve(recipe.path)),
    ))
    records.extend((
        ("candidate_plan", str(recipe.candidate_plan.resolve(recipe.path))),
        (
            "candidate_source_aggregate",
            str(recipe.candidate_source_aggregate.resolve(recipe.path)),
        ),
        (
            "candidate_source_timing",
            str(recipe.candidate_source_timing.resolve(recipe.path)),
        ),
        (
            "candidate_expansion_aggregate",
            str(recipe.candidate_expansion_aggregate.resolve(recipe.path)),
        ),
        (
            "candidate_expansion_timing",
            str(recipe.candidate_expansion_timing.resolve(recipe.path)),
        ),
        ("lineage_plan", str(recipe.lineage_plan.resolve(recipe.path))),
        (
            "lineage_source_aggregate",
            str(recipe.lineage_source_aggregate.resolve(recipe.path)),
        ),
        (
            "lineage_source_timing",
            str(recipe.lineage_source_timing.resolve(recipe.path)),
        ),
        (
            "lineage_source_split_manifest",
            str(recipe.lineage_source_split_manifest.resolve(recipe.path)),
        ),
    ))
    records.extend(
        ("lineage_source_route_manifest", str(item.resolve(recipe.path)))
        for item in recipe.lineage_source_route_manifests
    )
    records.extend(
        ("lineage_historical_refinement_plan", str(item.resolve(recipe.path)))
        for item in recipe.lineage_historical_refinement_plans
    )
    records.extend(
        ("lineage_additional_split_manifest", str(item.resolve(recipe.path)))
        for item in recipe.lineage_additional_split_manifests
    )
    for item in recipe.additives:
        records.append(("additive_aggregate", str(item.aggregate.resolve(recipe.path))))
        records.append(("additive_timing", str(item.timing.resolve(recipe.path))))
        if item.generic_refinement_plan is not None:
            records.append((
                "current_generic_refinement_plan",
                str(item.generic_refinement_plan.resolve(recipe.path)),
            ))
    records.extend((
        (
            "primary_profiler_requests",
            str(recipe.primary_profiler_transaction.requests.resolve(recipe.path)),
        ),
        (
            "primary_profiler_evidence",
            str(recipe.primary_profiler_transaction.evidence.resolve(recipe.path)),
        ),
        (
            "primary_profiler_observations",
            str(recipe.primary_profiler_transaction.observations.resolve(recipe.path)),
        ),
    ))
    for item in recipe.profiler_transactions:
        records.extend((
            ("profiler_requests", str(item.requests.resolve(recipe.path))),
            ("profiler_evidence", str(item.evidence.resolve(recipe.path))),
            ("profiler_observations", str(item.observations.resolve(recipe.path))),
        ))
    records.extend(
        (
            "burned_sealed_development_manifest",
            str(item.resolve(recipe.path)),
        )
        for item in recipe.burned_seal_development_transactions
    )
    return tuple(records)


def finalize_checkpoint(recipe_path: Path, checkpoint_path: Path) -> None:
    """Atomically promote a rebuilt checkpoint to explicit complete state."""

    recipe = load_recipe(recipe_path)
    expected_path = recipe.checkpoint_path()
    if Path(checkpoint_path).resolve() != expected_path:
        raise CPUPrefillReplayRecipeError(
            f"checkpoint finalization path differs from recipe: {checkpoint_path}"
        )
    expected_corpus_id = recipe.context_corpus_id()
    if _checkpoint_first_corpus_id(expected_path) != expected_corpus_id:
        raise CPUPrefillReplayRecipeError(
            "rebuilt checkpoint does not belong to the complete recipe transaction"
        )
    raw = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    raw["checkpoint"] = {
        "state": "complete",
        "path": recipe.checkpoint.path.as_posix(),
        "sha256": _sha256(expected_path),
        "raw_corpus_id": expected_corpus_id,
        "prefix_input_count": None,
    }
    authenticated = dict(raw)
    authenticated.pop("recipe_digest", None)
    raw["recipe_digest"] = _canonical_digest(authenticated)
    temporary = Path(recipe_path).with_name(
        f"{Path(recipe_path).name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, recipe_path)


def relocate_recipe(source_path: Path, output_path: Path) -> None:
    """Stage an authenticated replay recipe in a clean transaction directory.

    A post-freeze CPU-prefill seal must never share its output names with the
    seal that was burned into development evidence.  Merely copying a recipe
    is insufficient because recipe-relative profiler and burned-seal
    artifacts would then resolve against the new directory.  This operation
    preserves those immutable source locations by rewriting each
    recipe-relative artifact path relative to the destination recipe, copies
    the mutable common-observation checkpoint, and authenticates the relocated
    transaction before publishing it.

    The large timing and profiler payloads remain in place.  Only the adapted
    checkpoint is copied because the refresh driver owns that filename in its
    output directory and may replace it when a future recipe explicitly asks
    for a rebase or rebuild.
    """

    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    if source_path == output_path:
        raise CPUPrefillReplayRecipeError(
            "CPU prefill replay relocation requires a distinct output path"
        )

    source_recipe = load_recipe(source_path)
    authenticate_recipe(source_recipe)
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def relocate_artifacts(value: object) -> None:
        """Retarget every strict recipe-relative artifact in one JSON tree."""

        if isinstance(value, dict):
            if set(value) == {"base", "path", "sha256"} and value["base"] == "recipe":
                source_artifact = source_path.parent.joinpath(
                    *PurePosixPath(str(value["path"])).parts
                ).resolve()
                value["path"] = PurePosixPath(os.path.relpath(
                    source_artifact,
                    output_path.parent,
                )).as_posix()
            for child in value.values():
                relocate_artifacts(child)
        elif isinstance(value, list):
            for child in value:
                relocate_artifacts(child)

    relocate_artifacts(raw)
    authenticated = dict(raw)
    authenticated.pop("recipe_digest", None)
    raw["recipe_digest"] = _canonical_digest(authenticated)

    source_checkpoint = source_recipe.checkpoint_path()
    destination_checkpoint = output_path.parent.joinpath(
        *source_recipe.checkpoint.path.parts
    ).resolve()
    if source_recipe.checkpoint.state != "rebuild":
        destination_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_temporary = destination_checkpoint.with_name(
            f"{destination_checkpoint.name}.{os.getpid()}.tmp"
        )
        shutil.copyfile(source_checkpoint, checkpoint_temporary)
        os.replace(checkpoint_temporary, destination_checkpoint)

    temporary = output_path.with_name(f"{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)

    relocated = load_recipe(output_path)
    action = authenticate_recipe(relocated)
    if action != source_recipe.checkpoint.state:
        raise CPUPrefillReplayRecipeError(
            "relocated CPU prefill replay checkpoint changed lifecycle state: "
            f"expected={source_recipe.checkpoint.state} actual={action}"
        )


def add_burned_seal_transaction(
    recipe_path: Path,
    transaction_path: Path,
) -> None:
    """Atomically append one authenticated failed seal to a v4 replay recipe.

    The failed-seal manifest authenticates its own raw aggregate, timing
    sidecar, witness plan, and measurement provenance before the recipe is
    rewritten. Older recipes must first name an authenticated primary profiler
    transaction; inferring one from output-directory filenames would recreate
    the implicit-base ambiguity that v4 removes.
    """

    recipe_path = Path(recipe_path)
    transaction_path = Path(transaction_path)
    read_cpu_prefill_burned_seal_transaction(transaction_path)
    raw = json.loads(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CPUPrefillReplayRecipeError("recipe root is not an object")
    schema = raw.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise CPUPrefillReplayRecipeError(
            "burned-seal updates require a v4 CPU prefill replay recipe with "
            "an explicit primary profiler transaction"
        )

    relative = PurePosixPath(os.path.relpath(
        transaction_path.resolve(),
        recipe_path.resolve().parent,
    ))
    artifact = {
        "base": "recipe",
        "path": relative.as_posix(),
        "sha256": _sha256(transaction_path),
    }
    transactions = raw["burned_seal_development_transactions"]
    if not isinstance(transactions, list):
        raise CPUPrefillReplayRecipeError(
            "burned-seal recipe field is not a list"
        )
    if artifact in transactions:
        raise CPUPrefillReplayRecipeError(
            "burned-seal transaction is already present in replay recipe"
        )
    transactions.append(artifact)
    authenticated = dict(raw)
    authenticated.pop("recipe_digest", None)
    raw["recipe_digest"] = _canonical_digest(authenticated)
    temporary = recipe_path.with_name(f"{recipe_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, recipe_path)
    load_recipe(recipe_path)


def set_primary_profiler_transaction(
    recipe_path: Path,
    requests_path: Path,
    evidence_path: Path,
    observations_path: Path,
    *,
    clear_additive_transactions: bool,
) -> None:
    """Install one authenticated matched-anchor primary profiler transaction.

    The operation is the supported v3-to-v4 migration. It authenticates the
    compact timing witnesses, request derivation, evidence coverage, and
    matched candidate anchors before changing recipe bytes. Existing additive
    profiler transactions are never discarded implicitly; a replacement
    corpus must request that destructive lineage edit explicitly.
    """

    recipe_path = Path(recipe_path)
    requests_path = Path(requests_path)
    evidence_path = Path(evidence_path)
    observations_path = Path(observations_path)

    raw = json.loads(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CPUPrefillReplayRecipeError("recipe root is not an object")
    authenticated_source = dict(raw)
    claimed_digest = authenticated_source.pop("recipe_digest", None)
    if claimed_digest != _canonical_digest(authenticated_source):
        raise CPUPrefillReplayRecipeError(
            "CPU prefill replay recipe digest changed before profiler migration"
        )
    schema = raw.get("schema_version")
    if schema not in {
        "native-vnni-cpu-prefill-replay-recipe-v3",
        SCHEMA_VERSION,
    }:
        raise CPUPrefillReplayRecipeError(
            "primary profiler migration requires a v3 or v4 replay recipe"
        )
    additions = raw.get("profiler_transactions")
    if not isinstance(additions, list):
        raise CPUPrefillReplayRecipeError(
            "profiler_transactions must be a JSON list"
        )
    if additions and not clear_additive_transactions:
        raise CPUPrefillReplayRecipeError(
            "recipe has additive profiler transactions; pass "
            "--clear-additive-profiler-transactions only when the new primary "
            "transaction replaces their complete candidate surface"
        )

    requests = read_profiler_request_manifest(requests_path)
    evidence = read_profiler_evidence_manifest(evidence_path)
    observations = read_observation_csv((observations_path,))
    compose_profiler_evidence(((observations, requests, evidence),))

    def artifact(path: Path) -> dict[str, str]:
        relative = PurePosixPath(os.path.relpath(
            path.resolve(),
            recipe_path.resolve().parent,
        ))
        return {
            "base": "recipe",
            "path": relative.as_posix(),
            "sha256": _sha256(path),
        }

    raw["primary_profiler_transaction"] = {
        "requests": artifact(requests_path),
        "evidence": artifact(evidence_path),
        "observations": artifact(observations_path),
    }
    if clear_additive_transactions:
        raw["profiler_transactions"] = []
    raw["schema_version"] = SCHEMA_VERSION
    authenticated = dict(raw)
    authenticated.pop("recipe_digest", None)
    raw["recipe_digest"] = _canonical_digest(authenticated)

    temporary = recipe_path.with_name(f"{recipe_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, recipe_path)
    load_recipe(recipe_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Expose authenticated preflight, shell expansion, and finalization."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--recipe", type=Path, required=True)
    preflight.add_argument("--emit-shell-records", action="store_true")
    finalize = subparsers.add_parser("finalize-checkpoint")
    finalize.add_argument("--recipe", type=Path, required=True)
    finalize.add_argument("--checkpoint", type=Path, required=True)
    add_burned = subparsers.add_parser("add-burned-seal")
    add_burned.add_argument("--recipe", type=Path, required=True)
    add_burned.add_argument("--transaction", type=Path, required=True)
    set_primary = subparsers.add_parser("set-primary-profiler")
    set_primary.add_argument("--recipe", type=Path, required=True)
    set_primary.add_argument("--requests", type=Path, required=True)
    set_primary.add_argument("--evidence", type=Path, required=True)
    set_primary.add_argument("--observations", type=Path, required=True)
    set_primary.add_argument(
        "--clear-additive-profiler-transactions",
        action="store_true",
    )
    relocate = subparsers.add_parser("relocate")
    relocate.add_argument("--recipe", type=Path, required=True)
    relocate.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    if arguments.command == "finalize-checkpoint":
        finalize_checkpoint(arguments.recipe, arguments.checkpoint)
        print(f"finalized CPU prefill replay checkpoint: {arguments.checkpoint}")
        return 0
    if arguments.command == "add-burned-seal":
        add_burned_seal_transaction(
            arguments.recipe,
            arguments.transaction,
        )
        print(
            "added CPU prefill burned seal to replay recipe: "
            f"{arguments.transaction}"
        )
        return 0
    if arguments.command == "set-primary-profiler":
        set_primary_profiler_transaction(
            arguments.recipe,
            arguments.requests,
            arguments.evidence,
            arguments.observations,
            clear_additive_transactions=(
                arguments.clear_additive_profiler_transactions
            ),
        )
        print(
            "installed CPU prefill primary profiler transaction: "
            f"{arguments.requests}"
        )
        return 0
    if arguments.command == "relocate":
        relocate_recipe(arguments.recipe, arguments.output)
        print(
            "relocated authenticated CPU prefill replay recipe: "
            f"{arguments.recipe} -> {arguments.output}"
        )
        return 0

    recipe = load_recipe(arguments.recipe)
    action = authenticate_recipe(recipe)
    if arguments.emit_shell_records:
        for kind, value in shell_records(recipe, action):
            print(f"{kind}\t{value}")
    else:
        print(
            "authenticated CPU prefill replay recipe "
            f"digest={recipe.recipe_digest} "
            f"additives={len(recipe.additives)} "
            "historical_plans="
            f"{len(recipe.lineage_historical_refinement_plans)} "
            "current_plans="
            f"{sum(item.generic_refinement_plan is not None for item in recipe.additives)} "
            "primary_profiler_transaction=1 "
            f"additive_profiler_transactions={len(recipe.profiler_transactions)} "
            "burned_seal_transactions="
            f"{len(recipe.burned_seal_development_transactions)} "
            f"checkpoint_action={action}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
