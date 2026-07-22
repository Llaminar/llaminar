"""Regression tests for authenticated CPU-prefill replay recipes."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PYTHON_ROOT = REPOSITORY_ROOT / "tests" / "v2" / "performance" / "kernels"
sys.path.insert(0, str(PYTHON_ROOT))

from native_vnni_dispatch.cpu_prefill_replay_recipe import (  # noqa: E402
    CPUPrefillReplayRecipe,
    CPUPrefillReplayRecipeError,
    _canonical_digest,
    add_burned_seal_transaction,
    authenticate_recipe,
    finalize_checkpoint,
    load_recipe,
    relocate_recipe,
    set_primary_profiler_transaction,
    shell_records,
)


def _sha256(path: Path) -> str:
    """Return a compact fixture artifact identity."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeSplit:
    """Provide only the digest contract used by recipe split routing."""

    def __init__(self, digest: str) -> None:
        self._digest = digest

    def digest(self) -> str:
        return self._digest

    def accepted_digests(self) -> frozenset[str]:
        return frozenset((self._digest,))


class _FakePlan:
    """Provide stable plan identity without constructing a real corpus plan."""

    def __init__(self, digest: str) -> None:
        self._digest = digest

    def digest(self) -> str:
        return self._digest


class _FakeRoutes:
    """Expose an empty typed route inventory to recipe preflight tests."""

    @staticmethod
    def routes() -> tuple[object, ...]:
        return ()


class CPUPrefillReplayRecipeTest(unittest.TestCase):
    """Prove recipe roles and checkpoint state fail closed."""

    def _artifact(
        self,
        directory: Path,
        name: str,
        content: str = "fixture\n",
    ) -> dict[str, str]:
        path = directory / name
        path.write_text(content, encoding="utf-8")
        return {
            "base": "recipe",
            "path": name,
            "sha256": _sha256(path),
        }

    def _write_recipe(
        self,
        directory: Path,
        *,
        checkpoint: dict[str, object] | None = None,
    ) -> Path:
        artifacts = {
            name: self._artifact(
                directory,
                name,
                (
                    '{"split_manifest_digest":"old-split"}\n'
                    if name == "historical-plan.json"
                    else "fixture\n"
                ),
            )
            for name in (
                "target-split.json",
                "target-route.csv",
                "sealed-candidate-route.csv",
                "candidate-plan.json",
                "base.csv",
                "base.timing.csv",
                "expansion.csv",
                "expansion.timing.csv",
                "increment.csv",
                "increment.timing.csv",
                "current-plan.json",
                "lineage-plan.json",
                "lineage-source.csv",
                "lineage-source.timing.csv",
                "source-split.json",
                "old-split.json",
                "source-route.csv",
                "historical-plan.json",
                "profiler-requests.json",
                "profiler-evidence.json",
                "profiler-observations.csv",
            )
        }
        payload: dict[str, object] = {
            "schema_version": "native-vnni-cpu-prefill-replay-recipe-v4",
            "target": {
                "split_manifest": artifacts["target-split.json"],
                "route_manifests": [artifacts["target-route.csv"]],
                "sealed_candidate_route_manifest": artifacts[
                    "sealed-candidate-route.csv"
                ],
            },
            "candidate_expansion": {
                "plan": artifacts["candidate-plan.json"],
                "source_aggregate": artifacts["base.csv"],
                "source_timing": artifacts["base.timing.csv"],
                "expansion_aggregate": artifacts["expansion.csv"],
                "expansion_timing": artifacts["expansion.timing.csv"],
            },
            "additive_transactions": [{
                "aggregate": artifacts["increment.csv"],
                "timing": artifacts["increment.timing.csv"],
                "generic_refinement_plan": artifacts["current-plan.json"],
            }],
            "primary_profiler_transaction": {
                "requests": artifacts["profiler-requests.json"],
                "evidence": artifacts["profiler-evidence.json"],
                "observations": artifacts["profiler-observations.csv"],
            },
            "profiler_transactions": [],
            "burned_seal_development_transactions": [],
            "development_lineage": {
                "plan": artifacts["lineage-plan.json"],
                "source_aggregate": artifacts["lineage-source.csv"],
                "source_timing": artifacts["lineage-source.timing.csv"],
                "source_split_manifest": artifacts["source-split.json"],
                "source_route_manifests": [artifacts["source-route.csv"]],
                "historical_refinement_plans": [
                    artifacts["historical-plan.json"]
                ],
                "additional_split_manifests": [artifacts["old-split.json"]],
            },
            "checkpoint": checkpoint or {
                "state": "rebuild",
                "path": "checkpoint.csv",
                "sha256": None,
                "raw_corpus_id": None,
                "prefix_input_count": None,
            },
        }
        payload["recipe_digest"] = _canonical_digest(payload)
        output = directory / "cpu_prefill_replay_recipe.v1.json"
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output

    def _authentication_patches(
        self,
        *,
        duplicate_split: bool = False,
        duplicate_plan_role: bool = False,
        incomplete_target_routes: bool = False,
    ) -> ExitStack:
        stack = ExitStack()
        split_digests = iter((
            "target-split",
            "source-split",
            "source-split" if duplicate_split else "old-split",
        ))
        stack.enter_context(mock.patch(
            "native_vnni_dispatch.cpu_prefill_replay_recipe."
            "load_cpu_prefill_split_manifest",
            side_effect=lambda _path: _FakeSplit(next(split_digests)),
        ))
        stack.enter_context(mock.patch(
            "native_vnni_dispatch.cpu_prefill_replay_recipe."
            "read_cpu_prefill_route_manifests",
            side_effect=(_FakeRoutes(), _FakeRoutes(), _FakeRoutes()),
        ))
        stack.enter_context(mock.patch(
            "native_vnni_dispatch.cpu_prefill_replay_recipe."
            "validate_cpu_prefill_serial_route_manifest",
            side_effect=(
                ValueError("missing codebook/shape/ISA route")
                if incomplete_target_routes
                else None
            ),
        ))
        plan_digests = (
            ("same-plan", "same-plan")
            if duplicate_plan_role
            else ("historical-plan", "current-plan")
        )
        plans = iter(_FakePlan(value) for value in plan_digests)
        stack.enter_context(mock.patch(
            "native_vnni_dispatch.cpu_prefill_replay_recipe."
            "read_cpu_prefill_generic_refinement_plan",
            side_effect=lambda *_args, **_kwargs: next(plans),
        ))
        stack.enter_context(mock.patch(
            "native_vnni_dispatch.cpu_prefill_replay_recipe."
            "read_cpu_prefill_development_lineage_plan",
        ))
        stack.enter_context(mock.patch(
            "native_vnni_dispatch.cpu_prefill_replay_recipe."
            "read_cpu_prefill_candidate_expansion_plan",
            return_value=_FakePlan("candidate-plan"),
        ))
        stack.enter_context(mock.patch(
            "native_vnni_dispatch.cpu_prefill_replay_recipe."
            "_checkpoint_candidate_plan_digest",
            return_value="candidate-plan",
        ))
        return stack

    def test_preflight_authenticates_typed_rebuild_recipe(self) -> None:
        """A valid recipe reaches a timing-free rebuild decision."""

        with tempfile.TemporaryDirectory() as temporary:
            recipe = load_recipe(self._write_recipe(Path(temporary)))
            with self._authentication_patches():
                self.assertEqual(authenticate_recipe(recipe), "rebuild")

    def test_primary_profiler_transaction_is_explicit_shell_input(self) -> None:
        """Replay replaces the implicit profiler base with one named triplet."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recipe = load_recipe(self._write_recipe(directory))
            records = dict(shell_records(recipe, "rebuild"))

            self.assertEqual(
                records["primary_profiler_requests"],
                str(directory / "profiler-requests.json"),
            )
            self.assertEqual(
                records["primary_profiler_evidence"],
                str(directory / "profiler-evidence.json"),
            )
            self.assertEqual(
                records["primary_profiler_observations"],
                str(directory / "profiler-observations.csv"),
            )

    def test_legacy_implicit_profiler_base_is_rejected(self) -> None:
        """A v3 recipe cannot silently compose against output-directory files."""

        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_recipe(Path(temporary))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["schema_version"] = (
                "native-vnni-cpu-prefill-replay-recipe-v3"
            )
            payload.pop("primary_profiler_transaction")
            authenticated = dict(payload)
            authenticated.pop("recipe_digest")
            payload["recipe_digest"] = _canonical_digest(authenticated)
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                CPUPrefillReplayRecipeError,
                "schema is invalid|unsupported",
            ):
                load_recipe(path)

    def test_primary_profiler_command_migrates_v3_explicitly(self) -> None:
        """The supported migration authenticates and deliberately clears additions."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self._write_recipe(directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            old_primary = payload.pop("primary_profiler_transaction")
            payload["profiler_transactions"] = [old_primary]
            payload["schema_version"] = (
                "native-vnni-cpu-prefill-replay-recipe-v3"
            )
            authenticated = dict(payload)
            authenticated.pop("recipe_digest")
            payload["recipe_digest"] = _canonical_digest(authenticated)
            path.write_text(json.dumps(payload), encoding="utf-8")

            profiler_patches = (
                mock.patch(
                    "native_vnni_dispatch.cpu_prefill_replay_recipe."
                    "read_profiler_request_manifest"
                ),
                mock.patch(
                    "native_vnni_dispatch.cpu_prefill_replay_recipe."
                    "read_profiler_evidence_manifest"
                ),
                mock.patch(
                    "native_vnni_dispatch.cpu_prefill_replay_recipe."
                    "read_observation_csv"
                ),
                mock.patch(
                    "native_vnni_dispatch.cpu_prefill_replay_recipe."
                    "compose_profiler_evidence"
                ),
            )
            with ExitStack() as stack:
                for profiler_patch in profiler_patches:
                    stack.enter_context(profiler_patch)
                with self.assertRaisesRegex(
                    CPUPrefillReplayRecipeError,
                    "--clear-additive-profiler-transactions",
                ):
                    set_primary_profiler_transaction(
                        path,
                        directory / "profiler-requests.json",
                        directory / "profiler-evidence.json",
                        directory / "profiler-observations.csv",
                        clear_additive_transactions=False,
                    )
                set_primary_profiler_transaction(
                    path,
                    directory / "profiler-requests.json",
                    directory / "profiler-evidence.json",
                    directory / "profiler-observations.csv",
                    clear_additive_transactions=True,
                )

            migrated = load_recipe(path)
            self.assertEqual(
                migrated.primary_profiler_transaction.requests.resolve(path),
                directory / "profiler-requests.json",
            )
            self.assertEqual(migrated.profiler_transactions, ())

    def test_primary_profiler_migration_checks_recipe_before_evidence(self) -> None:
        """A changed recipe fails before parsing any profiler payload."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self._write_recipe(directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["checkpoint"]["path"] = "tampered.csv"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with mock.patch(
                "native_vnni_dispatch.cpu_prefill_replay_recipe."
                "read_profiler_request_manifest",
                side_effect=AssertionError("profiler parsing ran"),
            ):
                with self.assertRaisesRegex(
                    CPUPrefillReplayRecipeError,
                    "digest changed before profiler migration",
                ):
                    set_primary_profiler_transaction(
                        path,
                        directory / "profiler-requests.json",
                        directory / "profiler-evidence.json",
                        directory / "profiler-observations.csv",
                        clear_additive_transactions=False,
                    )

    def test_burned_seal_is_added_as_a_typed_shell_record(self) -> None:
        """Operators never reconstruct failed-seal analyzer arguments by hand."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recipe_path = self._write_recipe(directory)
            transaction_path = directory / "burned-seal.json"
            transaction_path.write_text("fixture\n", encoding="utf-8")
            with mock.patch(
                "native_vnni_dispatch.cpu_prefill_replay_recipe."
                "read_cpu_prefill_burned_seal_transaction"
            ):
                add_burned_seal_transaction(recipe_path, transaction_path)

            recipe = load_recipe(recipe_path)
            self.assertEqual(
                dict(shell_records(recipe, "rebuild"))[
                    "burned_sealed_development_manifest"
                ],
                str(transaction_path),
            )

    def test_duplicate_historical_split_digest_is_rejected(self) -> None:
        """The primary source split cannot be repeated as an older split."""

        with tempfile.TemporaryDirectory() as temporary:
            recipe = load_recipe(self._write_recipe(Path(temporary)))
            with self._authentication_patches(duplicate_split=True):
                with self.assertRaisesRegex(
                    CPUPrefillReplayRecipeError,
                    "duplicate lineage source split digest",
                ):
                    authenticate_recipe(recipe)

    def test_incomplete_target_route_matrix_fails_before_fitting(self) -> None:
        """Recipe preflight owns current route totality, not final freeze."""

        with tempfile.TemporaryDirectory() as temporary:
            recipe = load_recipe(self._write_recipe(Path(temporary)))
            with self._authentication_patches(incomplete_target_routes=True):
                with self.assertRaisesRegex(
                    CPUPrefillReplayRecipeError,
                    "target CPU prefill route manifest is incomplete",
                ):
                    authenticate_recipe(recipe)

    def test_historical_plan_cannot_reappear_as_current_refinement(self) -> None:
        """Plan roles are explicit and cross-role duplicates fail preflight."""

        with tempfile.TemporaryDirectory() as temporary:
            recipe = load_recipe(self._write_recipe(Path(temporary)))
            with self._authentication_patches(duplicate_plan_role=True):
                with self.assertRaisesRegex(
                    CPUPrefillReplayRecipeError,
                    "both historical and current roles",
                ):
                    authenticate_recipe(recipe)

    def test_relocated_complete_checkpoint_requests_rebase(self) -> None:
        """Valid checkpoint bytes are rebased when raw path identity changes."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint_path = directory / "checkpoint.csv"
            with checkpoint_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("corpus_id",))
                writer.writeheader()
                writer.writerow({"corpus_id": "sha256:old-workspace"})
            recipe_path = self._write_recipe(directory, checkpoint={
                "state": "complete",
                "path": "checkpoint.csv",
                "sha256": _sha256(checkpoint_path),
                "raw_corpus_id": "sha256:old-workspace",
                "prefix_input_count": None,
            })
            recipe = load_recipe(recipe_path)
            with self._authentication_patches(), mock.patch.object(
                CPUPrefillReplayRecipe,
                "context_corpus_id",
                return_value="sha256:new-workspace",
            ):
                self.assertEqual(authenticate_recipe(recipe), "rebase")

    def test_relocate_stages_clean_output_and_preserves_source_artifacts(self) -> None:
        """A fresh seal directory retains immutable evidence by relative path."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "fresh-seal"
            source.mkdir()
            checkpoint_path = source / "checkpoint.csv"
            checkpoint_path.write_text(
                "corpus_id\nsha256:" + "a" * 64 + "\n",
                encoding="utf-8",
            )
            recipe_path = self._write_recipe(source, checkpoint={
                "state": "complete",
                "path": "checkpoint.csv",
                "sha256": _sha256(checkpoint_path),
                "raw_corpus_id": "sha256:" + "a" * 64,
                "prefix_input_count": None,
            })
            output_path = destination / "cpu_prefill_replay_recipe.v1.json"

            with mock.patch(
                "native_vnni_dispatch.cpu_prefill_replay_recipe."
                "authenticate_recipe",
                return_value="complete",
            ):
                relocate_recipe(recipe_path, output_path)

            relocated = load_recipe(output_path)
            self.assertEqual(
                relocated.primary_profiler_transaction.requests.resolve(output_path),
                (source / "profiler-requests.json").resolve(),
            )
            self.assertEqual(
                (destination / "checkpoint.csv").read_bytes(),
                checkpoint_path.read_bytes(),
            )
            self.assertNotEqual(
                relocated.checkpoint_path(),
                checkpoint_path.resolve(),
            )

    def test_checkpoint_expansion_plan_mismatch_fails_preflight(self) -> None:
        """A complete checkpoint cannot be paired with a convenient later plan."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint_path = directory / "checkpoint.csv"
            with checkpoint_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("corpus_id",))
                writer.writeheader()
                writer.writerow({"corpus_id": "sha256:old-workspace"})
            recipe = load_recipe(self._write_recipe(directory, checkpoint={
                "state": "complete",
                "path": "checkpoint.csv",
                "sha256": _sha256(checkpoint_path),
                "raw_corpus_id": "sha256:old-workspace",
                "prefix_input_count": None,
            }))
            with self._authentication_patches(), mock.patch(
                "native_vnni_dispatch.cpu_prefill_replay_recipe."
                "_checkpoint_candidate_plan_digest",
                return_value="different-plan",
            ):
                with self.assertRaisesRegex(
                    CPUPrefillReplayRecipeError,
                    "candidate-expansion plan differs",
                ):
                    authenticate_recipe(recipe)

    def test_finalize_checkpoint_records_complete_transaction(self) -> None:
        """A rebuilt checkpoint is atomically promoted to explicit complete state."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recipe_path = self._write_recipe(directory)
            recipe = load_recipe(recipe_path)
            corpus_id = recipe.context_corpus_id()
            checkpoint_path = directory / "checkpoint.csv"
            with checkpoint_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("corpus_id",))
                writer.writeheader()
                writer.writerow({"corpus_id": corpus_id})

            finalize_checkpoint(recipe_path, checkpoint_path)

            finalized = load_recipe(recipe_path)
            self.assertEqual(finalized.checkpoint.state, "complete")
            self.assertEqual(finalized.checkpoint.sha256, _sha256(checkpoint_path))
            self.assertEqual(finalized.checkpoint.raw_corpus_id, corpus_id)

    def test_complete_recipe_publishes_prevalidated_context_identity(self) -> None:
        """Fit invocations do not re-hash raw sidecars after preflight."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint_path = directory / "checkpoint.csv"
            checkpoint_path.write_text(
                "corpus_id\nsha256:"
                + "a" * 64
                + "\n",
                encoding="utf-8",
            )
            recipe = load_recipe(self._write_recipe(directory, checkpoint={
                "state": "complete",
                "path": "checkpoint.csv",
                "sha256": _sha256(checkpoint_path),
                "raw_corpus_id": "sha256:" + "a" * 64,
                "prefix_input_count": None,
            }))

            with mock.patch.object(
                CPUPrefillReplayRecipe,
                "context_corpus_id",
                side_effect=AssertionError("complete identity was recomputed"),
            ):
                records = dict(shell_records(recipe, "complete"))

            self.assertEqual(
                records["context_corpus_id"],
                "sha256:" + "a" * 64,
            )


if __name__ == "__main__":
    unittest.main()
