#!/usr/bin/env python3
"""Unit tests for production parity campaign discovery and sharding."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "run_production_parity_campaigns.py"
TMPFS_SETUP_SCRIPT = (
    REPO_ROOT / "scripts" / "ci" / "setup_production_parity_tmpfs.sh"
)
SPEC = importlib.util.spec_from_file_location("production_parity_campaigns", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
campaigns = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaigns
SPEC.loader.exec_module(campaigns)


def ctest_document(*names: str) -> str:
    """Build the minimal CTest JSON shape used by discovery."""

    model_path = "/models/test.gguf"
    return json.dumps(
        {
            "tests": [
                {
                    "name": name,
                    "properties": [
                        {
                            "name": "LABELS",
                            "value": [
                                "V2",
                                "Campaign",
                                "ProductionPath",
                                "AllPrecisions",
                                "WholeMatrix75MinuteTarget",
                            ],
                        },
                        {
                            "name": "ENVIRONMENT",
                            "value": [
                                *sorted(campaigns.REQUIRED_CAMPAIGN_ENV),
                                "LLAMINAR_PERF_STATS_FILTER=forward_graph",
                                f"{campaigns.MODEL_MANIFEST_ENV}={model_path}",
                            ],
                        },
                        {
                            "name": "TIMEOUT",
                            "value": campaigns.REGISTERED_TIMEOUT_SECONDS,
                        },
                        {"name": "REQUIRED_FILES", "value": [model_path]},
                    ],
                    "command": [
                        "fake_gtest",
                        "--gtest_filter=Suite.ProductionParity/CPU_KV_FP16",
                    ],
                }
                for name in names
            ]
        }
    )


def preflight_ctest_document(*names: str) -> str:
    """Build a model-free Integration registration for preflight discovery."""

    return json.dumps(
        {
            "tests": [
                {
                    "name": name,
                    "properties": [
                        {
                            "name": "LABELS",
                            "value": [
                                "V2",
                                "Integration",
                                campaigns.PRODUCTION_PARITY_PREFLIGHT_LABEL,
                            ],
                        },
                        {"name": "TIMEOUT", "value": 30.0},
                    ],
                    "command": ["fake_integration_test"],
                }
                for name in names
            ]
        }
    )


class ProductionParityCampaignTest(unittest.TestCase):
    @staticmethod
    def write_canonical_artifacts(directory: Path) -> None:
        """Materialize the smallest schema-valid campaign evidence fixture."""

        directory.mkdir(parents=True)
        for filename, header in campaigns.REQUIRED_CSV_HEADERS.items():
            payload = header + "\n"
            if filename in campaigns.CSV_ARTIFACTS_REQUIRING_DATA:
                payload += ",".join(
                    "value" for _ in next(campaigns.csv.reader([header]))
                ) + "\n"
            (directory / filename).write_text(payload, encoding="utf-8")

    @mock.patch.object(campaigns.subprocess, "run")
    def test_preflight_inventory_is_discovered_from_model_free_integration_label(
        self,
        run: mock.Mock,
    ) -> None:
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=preflight_ctest_document(
                "V2_Integration_ParityCellLifecycle_MPI2",
                "V2_Integration_ParityCellLifecycle_MPI1",
            ),
            stderr="",
        )

        discovered = campaigns.discover_production_parity_preflight_tests(
            Path("build")
        )

        self.assertEqual(
            discovered,
            (
                "V2_Integration_ParityCellLifecycle_MPI1",
                "V2_Integration_ParityCellLifecycle_MPI2",
            ),
        )
        command = run.call_args.args[0]
        self.assertIn("--show-only=json-v1", command)
        self.assertIn(
            f"^{campaigns.PRODUCTION_PARITY_PREFLIGHT_LABEL}$",
            command,
        )

    @mock.patch.object(campaigns.subprocess, "run")
    def test_preflight_inventory_rejects_model_fixture_dependency(
        self,
        run: mock.Mock,
    ) -> None:
        document = json.loads(
            preflight_ctest_document("V2_Integration_InvalidPreflight")
        )
        document["tests"][0]["properties"].append(
            {"name": "FIXTURES_REQUIRED", "value": ["V2_Models"]}
        )
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(document), stderr=""
        )

        with self.assertRaisesRegex(RuntimeError, "requires a fixture"):
            campaigns.discover_production_parity_preflight_tests(Path("build"))

    @mock.patch.object(campaigns, "_run_process", return_value=0)
    @mock.patch.object(
        campaigns,
        "discover_production_parity_preflight_tests",
        return_value=("V2_Integration_ParityCellLifecycle_MPI1",),
    )
    def test_preflight_executes_the_ctest_label_before_campaign_admission(
        self,
        discover: mock.Mock,
        run_process: mock.Mock,
    ) -> None:
        return_code, _, tests = campaigns.run_production_parity_preflight(
            Path("build"), 60.0
        )

        self.assertEqual(return_code, 0)
        self.assertEqual(
            tests,
            ("V2_Integration_ParityCellLifecycle_MPI1",),
        )
        discover.assert_called_once_with(Path("build"))
        command = run_process.call_args.args[0]
        self.assertIn("--parallel", command)
        self.assertIn("--no-tests=error", command)
        self.assertIn(
            f"^{campaigns.PRODUCTION_PARITY_PREFLIGHT_LABEL}$",
            command,
        )

    def test_main_orders_preflight_before_model_fixture_and_staging(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        main = source[source.index("def main(") :]
        preflight = main.index("run_production_parity_preflight(")
        fixture = main.index("prepare_model_fixture(")
        staging = main.index("stage_models_in_ramdisk(")
        self.assertLess(preflight, fixture)
        self.assertLess(fixture, staging)

    def test_every_production_parity_source_uses_the_typed_definition_expander(
        self,
    ) -> None:
        """Prevent model fixtures from rebuilding a private stringly matrix."""

        parity_root = REPO_ROOT / "tests" / "v2" / "integration" / "parity"
        offenders: list[str] = []
        for source in sorted(parity_root.rglob("*.cpp")):
            contents = source.read_text(encoding="utf-8")
            if "ProductionParity" not in contents:
                continue
            if (
                "ModelParityDefinition" not in contents
                or "expandModelParityDefinition" not in contents
            ):
                offenders.append(str(source.relative_to(REPO_ROOT)))

        self.assertEqual(
            offenders,
            [],
            "ProductionParity sources must declare and expand the canonical "
            "typed model/topology matrix: " + ", ".join(offenders),
        )

    def test_every_production_parity_body_is_parameterized(self) -> None:
        """Forbid fixed TEST/TEST_F cells beside the typed case expander."""

        parity_root = REPO_ROOT / "tests" / "v2" / "integration" / "parity"
        offenders: list[str] = []
        fixed_test = re.compile(r"\bTEST(?:_F)?\s*\(")
        for source in sorted(parity_root.rglob("*.cpp")):
            contents = source.read_text(encoding="utf-8")
            for match in fixed_test.finditer(contents):
                declaration = contents[match.start() : match.start() + 512]
                declaration = declaration.split("{", maxsplit=1)[0]
                if "ProductionParity" in declaration:
                    line = contents.count("\n", 0, match.start()) + 1
                    offenders.append(
                        f"{source.relative_to(REPO_ROOT)}:{line}"
                    )

        self.assertEqual(
            offenders,
            [],
            "ProductionParity bodies must be TEST_P cases generated from "
            "ModelParityCase: " + ", ".join(offenders),
        )

    def test_persistent_tmpfs_setup_has_a_device_free_help_path(self) -> None:
        """Keep the privileged mount boundary explicit and inspectable."""

        result = subprocess.run(
            ["bash", str(TMPFS_SETUP_SCRIPT), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--status", result.stdout)
        self.assertIn("--unmount", result.stdout)
        self.assertIn("/mnt/llaminar-production-parity", result.stdout)

    def test_classifies_single_backend_and_kv_precision(self) -> None:
        group = campaigns.classify_campaign(
            "V2_Integration_Parity_Qwen2_ProductionCampaign_CPU_KV_Q8_1"
        )

        self.assertEqual(group.backends, "CPU")
        self.assertEqual(group.kv_precision, "Q8_1")
        self.assertEqual(group.name, "cpu__kv-q8_1")

    def test_classifies_heterogeneous_backend_signature(self) -> None:
        group = campaigns.classify_campaign(
            "V2_Integration_Parity_Qwen35_ProductionCampaign_CUDA_ROCm_CPU_KV_FP16"
        )

        self.assertEqual(group.backends, "CPU+CUDA+ROCm")
        self.assertEqual(group.kv_precision, "FP16")

    def test_missing_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no backend identity"):
            campaigns.classify_campaign(
                "V2_Integration_Parity_Qwen2_ProductionCampaign_ALL_PRECISIONS"
            )

    def test_classifies_all_precision_process_campaign(self) -> None:
        group = campaigns.classify_campaign(
            "V2_Integration_Parity_Qwen2_"
            "ProductionCampaign_ROCm_ALL_PRECISIONS"
        )

        self.assertEqual(group.backends, "ROCm")
        self.assertEqual(group.kv_precision, "ALL")

    @mock.patch.object(campaigns.subprocess, "run")
    def test_discovery_uses_campaign_label_and_production_name(self, run: mock.Mock) -> None:
        included = (
            "V2_Integration_Parity_Qwen2_"
            "ProductionCampaign_CUDA_ALL_PRECISIONS"
        )
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=ctest_document(
                included,
                "V2_Unit_NotProductionCampaign_CPU_ALL_PRECISIONS",
            ),
            stderr="",
        )

        cells = campaigns.discover_campaigns(Path("build"))

        self.assertEqual([cell.name for cell in cells], [included])
        self.assertEqual(
            cells[0].gtest_cases,
            ("Suite.ProductionParity/CPU_KV_FP16",),
        )
        self.assertEqual(cells[0].model_files, ("/models/test.gguf",))
        run.assert_called_once()

    @mock.patch.object(campaigns.subprocess, "run")
    def test_focused_discovery_ignores_stale_unselected_registration(
        self, run: mock.Mock
    ) -> None:
        """A named slice validates its own CTest authority, not stale siblings."""

        selected = (
            "V2_Integration_Parity_Qwen2_SingleDevice_"
            "ProductionCampaign_CPU_ALL_PRECISIONS"
        )
        stale = (
            "V2_Integration_Parity_Qwen2_LocalTP_"
            "ProductionCampaign_CUDA_ALL_PRECISIONS"
        )
        document = json.loads(ctest_document(selected, stale))
        stale_properties = document["tests"][1]["properties"]
        stale_properties[:] = [
            property_value
            for property_value in stale_properties
            if property_value["name"] != "REQUIRED_FILES"
        ]
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(document), stderr=""
        )

        cells = campaigns.discover_campaigns(
            Path("build"),
            campaign_regex=re.escape(selected),
        )

        self.assertEqual([cell.name for cell in cells], [selected])

    def test_groups_are_deterministic_and_filterable(self) -> None:
        cells = [
            campaigns.CampaignCell(
                "z_ProductionCampaign_CUDA_ALL_PRECISIONS",
                campaigns.CampaignGroup("CUDA", "ALL"),
            ),
            campaigns.CampaignCell(
                "a_ProductionCampaign_CUDA_ALL_PRECISIONS",
                campaigns.CampaignGroup("CUDA", "ALL"),
            ),
            campaigns.CampaignCell(
                "b_ProductionCampaign_CPU_ALL_PRECISIONS",
                campaigns.CampaignGroup("CPU", "ALL"),
            ),
        ]

        selected = campaigns.filter_campaigns(cells, "CUDA", "ALL")
        grouped = campaigns.group_campaigns(selected)

        self.assertEqual(len(grouped), 1)
        self.assertEqual(
            [cell.name for cell in next(iter(grouped.values()))],
            [
                "a_ProductionCampaign_CUDA_ALL_PRECISIONS",
                "z_ProductionCampaign_CUDA_ALL_PRECISIONS",
            ],
        )

    def test_campaign_name_slice_preserves_discovered_matrix_authority(self) -> None:
        cells = [
            campaigns.CampaignCell(
                "Qwen35_ProductionCampaign_CUDA_ALL_PRECISIONS",
                campaigns.CampaignGroup("CUDA", "ALL"),
            ),
            campaigns.CampaignCell(
                "Qwen35MoEExpertOverlay_ProductionCampaign_CUDA_ALL_PRECISIONS",
                campaigns.CampaignGroup("CUDA", "ALL"),
            ),
            campaigns.CampaignCell(
                "Qwen35MoEGraphNative_ProductionCampaign_CUDA_ALL_PRECISIONS",
                campaigns.CampaignGroup("CUDA", "ALL"),
            ),
        ]

        selected = campaigns.filter_campaigns(
            cells,
            ".*",
            ".*",
            ".*",
            ".*(?:ExpertOverlay|MoEGraphNative).*",
        )

        self.assertEqual(
            [cell.name for cell in selected],
            ["Qwen35_ProductionCampaign_CUDA_ALL_PRECISIONS"],
        )

    def test_exact_ctest_regex_escapes_test_names(self) -> None:
        cell = campaigns.CampaignCell(
            "Suite.ProductionCampaign/CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "FP16"),
        )

        regex = campaigns._ctest_exact_regex([cell])

        self.assertRegex(cell.name, regex)
        self.assertNotRegex("x" + cell.name, regex)
        self.assertNotIn("?:", regex)

    def test_exact_cell_watch_kills_stuck_process_group_and_names_cell(self) -> None:
        """A silent generated cell must fail in its own bounded lifetime."""

        with tempfile.TemporaryDirectory() as raw_directory:
            progress = Path(raw_directory) / "cell" / "test_log.txt"
            case = "MatrixSuite.ProductionParity/StuckCell"
            script = (
                "from pathlib import Path; import time; "
                f"p=Path({str(progress)!r}); p.parent.mkdir(parents=True); "
                "p.write_text('started'); time.sleep(30)"
            )
            evidence = campaigns.ProcessTimeoutEvidence()
            started = campaigns.time.monotonic()

            return_code = campaigns._run_process(
                [sys.executable, "-c", script],
                5.0,
                exact_cell_watch=campaigns.ExactCellTimeoutWatch(
                    timeout_seconds=0.2,
                    progress_files=((case, progress),),
                    not_before_wall_time_ns=campaigns.time.time_ns(),
                ),
                timeout_evidence=evidence,
            )

        self.assertEqual(return_code, 124)
        self.assertLess(campaigns.time.monotonic() - started, 2.0)
        self.assertEqual(
            evidence.kind,
            campaigns.ProcessTimeoutKind.EXACT_CELL,
        )
        self.assertEqual(evidence.exact_gtest_case, case)

    def test_exact_cell_watch_renews_deadline_only_on_next_cell(self) -> None:
        """Sequential cells may exceed one cell budget in aggregate."""

        with tempfile.TemporaryDirectory() as raw_directory:
            first = Path(raw_directory) / "first" / "test_log.txt"
            second = Path(raw_directory) / "second" / "test_log.txt"
            script = (
                "from pathlib import Path; import time; "
                f"a=Path({str(first)!r}); b=Path({str(second)!r}); "
                "a.parent.mkdir(parents=True); a.write_text('started'); "
                "time.sleep(0.3); "
                "b.parent.mkdir(parents=True); b.write_text('started'); "
                "time.sleep(0.3)"
            )
            evidence = campaigns.ProcessTimeoutEvidence()
            started = campaigns.time.monotonic()

            return_code = campaigns._run_process(
                [sys.executable, "-c", script],
                5.0,
                exact_cell_watch=campaigns.ExactCellTimeoutWatch(
                    timeout_seconds=0.5,
                    progress_files=(
                        ("MatrixSuite.ProductionParity/First", first),
                        ("MatrixSuite.ProductionParity/Second", second),
                    ),
                    not_before_wall_time_ns=campaigns.time.time_ns(),
                ),
                timeout_evidence=evidence,
            )

        self.assertEqual(return_code, 0)
        self.assertGreater(campaigns.time.monotonic() - started, 0.5)
        self.assertEqual(evidence.kind, campaigns.ProcessTimeoutKind.NONE)

    @mock.patch.object(campaigns, "_run_process")
    def test_campaign_receives_only_the_completion_timeout_remaining(
        self, run_process: mock.Mock
    ) -> None:
        cell = campaigns.CampaignCell(
            "V2_Integration_Parity_Qwen35_LocalTP_"
            "ProductionCampaign_ROCm_ALL_PRECISIONS",
            campaigns.CampaignGroup("ROCm", "ALL"),
        )
        run_process.return_value = 0

        environment = {
            "LLAMINAR_PRODUCTION_PARITY_DIGEST_CACHE": "/tmp/digest-proof"
        }
        result = campaigns.run_campaign(
            Path("build"),
            cell,
            campaigns.COMPLETION_TIMEOUT_SECONDS,
            target_seconds=campaigns.GLOBAL_TARGET_SECONDS,
            environment_overrides=environment,
        )

        self.assertEqual(
            result.test_type, "V2_Integration_Parity_Qwen35_LocalTP"
        )
        self.assertEqual(result.campaign, cell.name)
        self.assertEqual(result.backends, "ROCm")
        self.assertEqual(result.precision_set, "ALL")
        self.assertEqual(result.precision_types, ("TOPOLOGY_DEFAULT",))
        self.assertEqual(result.gtest_cases, ())
        self.assertTrue(result.target_met)
        command = run_process.call_args.args[0]
        self.assertRegex(cell.name, command[command.index("-R") + 1])
        self.assertIn("--no-tests=error", command)
        self.assertEqual(
            command[command.index("-FA") + 1],
            campaigns.MODEL_FIXTURE_NAME,
        )
        self.assertEqual(
            run_process.call_args.args[1],
            campaigns.COMPLETION_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            run_process.call_args.kwargs["environment_overrides"],
            environment,
        )
        self.assertIsNone(run_process.call_args.kwargs["exact_cell_watch"])
        self.assertIsInstance(
            run_process.call_args.kwargs["timeout_evidence"],
            campaigns.ProcessTimeoutEvidence,
        )
        self.assertEqual(
            result.exact_cell_timeout_seconds,
            campaigns.EXACT_CELL_TIMEOUT_SECONDS,
        )

    def test_campaign_artifacts_must_be_fresh_complete_and_schema_exact(self) -> None:
        cell = campaigns.CampaignCell(
            "V2_Parity_ProductionCampaign_CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "ALL"),
            gtest_cases=("MatrixSuite.ProductionParity/CPU_KV_FP16",),
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            result_directory = root / campaigns._gtest_artifact_directory_name(
                cell.gtest_cases[0]
            )
            started_ns = campaigns.time.time_ns()
            self.write_canonical_artifacts(result_directory)

            count, directories, errors = campaigns.validate_campaign_artifacts(
                cell,
                started_ns,
                revision_results_root=root,
            )

            self.assertEqual(count, len(campaigns.REQUIRED_CSV_HEADERS))
            self.assertEqual(directories, (str(result_directory),))
            self.assertEqual(errors, ())

            (result_directory / "prefill_stages.csv").unlink()
            count, _, errors = campaigns.validate_campaign_artifacts(
                cell,
                started_ns,
                revision_results_root=root,
            )

        self.assertEqual(count, len(campaigns.REQUIRED_CSV_HEADERS) - 1)
        self.assertTrue(
            any("missing prefill_stages.csv" in error for error in errors)
        )

    def test_typed_mtp_cell_requires_exact_transaction_evidence(self) -> None:
        gtest_case = (
            "MatrixSuite.ProductionParity/"
            "Qwen36_CUDA0_ActFP32_KVFP16_MTPDepth15"
        )
        cell = campaigns.CampaignCell(
            "V2_Parity_ProductionCampaign_CUDA_ALL_PRECISIONS",
            campaigns.CampaignGroup("CUDA", "ALL"),
            gtest_cases=(gtest_case,),
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            result_directory = root / campaigns._gtest_artifact_directory_name(
                gtest_case
            )
            self.write_canonical_artifacts(result_directory)

            count, _, errors = campaigns.validate_campaign_artifacts(
                cell,
                campaigns.time.time_ns(),
                revision_results_root=root,
            )
            self.assertEqual(count, len(campaigns.REQUIRED_CSV_HEADERS))
            self.assertTrue(
                any("missing mtp_transactions.csv" in error for error in errors)
            )

            columns = next(
                campaigns.csv.reader([campaigns.MTP_TRANSACTIONS_HEADER])
            )
            self.assertIn("dynamic_policy_witness_executed", columns)
            self.assertIn(
                "dynamic_policy_witness_serial_oracle_tokens", columns
            )
            (result_directory / "mtp_transactions.csv").write_text(
                campaigns.MTP_TRANSACTIONS_HEADER
                + "\n"
                + ",".join("value" for _ in columns)
                + "\n",
                encoding="utf-8",
            )
            count, _, errors = campaigns.validate_campaign_artifacts(
                cell,
                campaigns.time.time_ns(),
                revision_results_root=root,
            )

        self.assertEqual(count, len(campaigns.REQUIRED_CSV_HEADERS) + 1)
        self.assertEqual(errors, ())

    @mock.patch.object(
        campaigns,
        "validate_campaign_artifacts",
        return_value=(6, (), ("missing prefill_stages.csv",)),
    )
    @mock.patch.object(campaigns, "_run_process", return_value=0)
    def test_passing_ctest_cannot_hide_an_artifact_contract_failure(
        self,
        _run_process: mock.Mock,
        _validate: mock.Mock,
    ) -> None:
        cell = campaigns.CampaignCell(
            "V2_Parity_ProductionCampaign_CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "ALL"),
            gtest_cases=("MatrixSuite.ProductionParity/CPU_KV_FP16",),
        )

        result = campaigns.run_campaign(Path("build"), cell, 30.0)

        self.assertEqual(result.return_code, 126)
        self.assertEqual(result.outcome, "artifact_contract_failed")
        self.assertFalse(result.artifact_contract_passed)
        self.assertEqual(result.validated_artifact_file_count, 6)

    @mock.patch.object(
        campaigns,
        "validate_campaign_artifacts",
        return_value=(0, (), ()),
    )
    @mock.patch.object(campaigns, "_run_process")
    def test_campaign_reports_exact_timed_out_cell(
        self,
        run_process: mock.Mock,
        _validate: mock.Mock,
    ) -> None:
        """The report must distinguish one cell timeout from the global guard."""

        case = "MatrixSuite.ProductionParity/Depth1"
        cell = campaigns.CampaignCell(
            "V2_Parity_ProductionCampaign_CUDA_ALL_PRECISIONS",
            campaigns.CampaignGroup("CUDA", "ALL"),
            gtest_cases=(case,),
        )

        def expire_cell(*_args: object, **kwargs: object) -> int:
            evidence = kwargs["timeout_evidence"]
            assert isinstance(evidence, campaigns.ProcessTimeoutEvidence)
            evidence.kind = campaigns.ProcessTimeoutKind.EXACT_CELL
            evidence.exact_gtest_case = case
            return 124

        run_process.side_effect = expire_cell

        result = campaigns.run_campaign(
            Path("build"),
            cell,
            21600.0,
            artifact_results_root=Path("artifacts"),
        )

        self.assertEqual(result.return_code, 124)
        self.assertEqual(result.outcome, "exact_cell_timeout")
        self.assertEqual(result.timed_out_gtest_case, case)
        self.assertEqual(
            result.exact_cell_timeout_seconds,
            campaigns.EXACT_CELL_TIMEOUT_SECONDS,
        )

    @mock.patch.object(campaigns, "_run_process", return_value=0)
    def test_soft_target_never_becomes_the_process_timeout(
        self, run_process: mock.Mock
    ) -> None:
        cell = campaigns.CampaignCell(
            "V2_Parity_ProductionCampaign_CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "ALL"),
        )

        result = campaigns.run_campaign(
            Path("build"),
            cell,
            12345.0,
            global_started_at=campaigns.time.monotonic() - 2.0,
            target_seconds=1.0,
        )

        self.assertEqual(result.return_code, 0)
        self.assertFalse(result.target_met)
        self.assertEqual(run_process.call_args.args[1], 12345.0)
        self.assertEqual(result.outcome, "completed")

    def test_scheduler_overlaps_only_backend_disjoint_campaigns(self) -> None:
        cells = campaigns.scheduling_order(
            [
                campaigns.CampaignCell(
                    "cpu_ProductionCampaign_CPU_ALL_PRECISIONS",
                    campaigns.CampaignGroup("CPU", "ALL"),
                ),
                campaigns.CampaignCell(
                    "cuda_ProductionCampaign_CUDA_ALL_PRECISIONS",
                    campaigns.CampaignGroup("CUDA", "ALL"),
                ),
                campaigns.CampaignCell(
                    "rocm_ProductionCampaign_ROCm_ALL_PRECISIONS",
                    campaigns.CampaignGroup("ROCm", "ALL"),
                ),
                campaigns.CampaignCell(
                    "hybrid_ProductionCampaign_CPU_CUDA_ALL_PRECISIONS",
                    campaigns.CampaignGroup("CPU+CUDA", "ALL"),
                ),
            ]
        )

        selected = campaigns.select_runnable_campaigns(cells, set())

        self.assertEqual(
            [cell.group.backends for cell in selected],
            ["CPU+CUDA", "ROCm"],
        )
        claimed: set[str] = set()
        for cell in selected:
            resources = campaigns.campaign_resources(cell)
            self.assertTrue(claimed.isdisjoint(resources))
            claimed.update(resources)

        with_cuda_busy = campaigns.select_runnable_campaigns(cells, {"CUDA"})
        self.assertEqual(
            [cell.group.backends for cell in with_cuda_busy],
            ["CPU", "ROCm"],
        )

    @mock.patch.object(campaigns, "run_campaign")
    def test_matrix_admits_remaining_campaigns_after_target_is_missed(
        self, run_campaign: mock.Mock
    ) -> None:
        cells = [
            campaigns.CampaignCell(
                f"case_{index}_ProductionCampaign_CPU_ALL_PRECISIONS",
                campaigns.CampaignGroup("CPU", "ALL"),
            )
            for index in range(2)
        ]

        def completed_result(
            _build_dir: Path,
            cell: campaigns.CampaignCell,
            _completion_timeout_seconds: float,
            **_kwargs: object,
        ) -> campaigns.CampaignResult:
            return campaigns.CampaignResult(
                campaign=cell.name,
                test_type=cell.test_type,
                backends=cell.group.backends,
                precision_set=cell.group.kv_precision,
                precision_types=cell.precision_types,
                gtest_cases=cell.gtest_cases,
                elapsed_seconds=0.1,
                target_seconds=1.0,
                return_code=0,
                target_met=False,
                started_offset_seconds=2.0,
                finished_offset_seconds=2.1,
            )

        run_campaign.side_effect = completed_result
        results, _ = campaigns.run_campaign_matrix(
            Path("build"),
            cells,
            target_seconds=1.0,
            completion_timeout_seconds=30.0,
            global_started_at=campaigns.time.monotonic() - 2.0,
        )

        self.assertEqual(run_campaign.call_count, 2)
        self.assertEqual([result.campaign for result in results], [c.name for c in cells])
        self.assertTrue(all(result.return_code == 0 for result in results))
        self.assertTrue(all(not result.target_met for result in results))

    def test_ctest_exact_regex_selects_one_real_registered_test(self) -> None:
        """The emitted expression must be understood by CTest, not just Python."""

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            test_name = "Suite.ProductionCampaign/CPU_ALL_PRECISIONS"
            (directory / "CTestTestfile.cmake").write_text(
                f'add_test("{test_name}" "/bin/true")\n',
                encoding="utf-8",
            )
            cell = campaigns.CampaignCell(
                test_name,
                campaigns.CampaignGroup("CPU", "ALL"),
            )
            completed = subprocess.run(
                [
                    "ctest",
                    "--test-dir",
                    str(directory),
                    "--no-tests=error",
                    "-R",
                    campaigns._ctest_exact_regex([cell]),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stderr or completed.stdout,
        )
        # CTest 4.x removed the redundant `0 tests failed` clause while
        # retaining the same success/count semantics. Accept both renderings;
        # the return code and exact total are the contract under test.
        self.assertRegex(
            completed.stdout,
            r"100% tests passed(?:, 0 tests failed)? out of 1",
        )

    def test_duration_rejects_non_finite_values(self) -> None:
        for value in ("nan", "inf", "-inf", "0"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    campaigns.argparse.ArgumentTypeError,
                    "finite and positive",
                ):
                    campaigns._positive_seconds(value)

    def test_cmake_discovery_groups_production_cells_by_backend(self) -> None:
        discovery = (
            REPO_ROOT / "tests" / "v2" / "cmake" /
            "V2ParityTestDiscovery.cmake"
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            executable = directory / "fake_gtest"
            executable.write_text(
                "#!/bin/sh\n"
                "cat <<'EOF'\n"
                "MatrixSuite.\n"
                "  ProductionParity/CPU_KV_FP16\n"
                "  ProductionParity/CPU_KV_Q8_1\n"
                "  ProductionParity/CUDA_KV_FP16\n"
                "  ProductionParity/CUDA_ROCm_CPU_KV_FP16\n"
                "  FocusedInfrastructure/CPU\n"
                "EOF\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            model = directory / "test.gguf"
            model.write_bytes(b"gguf")
            cpu_model = directory / "cpu.gguf"
            cuda_model = directory / "cuda.gguf"
            rocm_model = directory / "rocm.gguf"
            for backend_model in (cpu_model, cuda_model, rocm_model):
                backend_model.write_bytes(backend_model.name.encode("utf-8"))
            generated = directory / "discovered.cmake"

            completed = subprocess.run(
                [
                    "cmake",
                    f"-DTEST_EXECUTABLE={executable}",
                    f"-DCTEST_FILE={generated}",
                    "-DTEST_PREFIX=V2_Integration_Parity_Fake",
                    "-DLABELS=V2;Integration;Parity;CPU;CUDA;ROCm;NCCL;RCCL",
                    "-DMPI_PROCS=1",
                    "-DNUM_SOCKETS=1",
                    "-DCORES_PER_SOCKET=4",
                    f"-DWORKING_DIR={directory}",
                    "-DTIMEOUT=90",
                    f"-DMODEL_FILES_SERIALIZED={model}",
                    f"-DCPU_MODEL_FILES_SERIALIZED={cpu_model}",
                    f"-DCUDA_MODEL_FILES_SERIALIZED={cuda_model}",
                    f"-DROCM_MODEL_FILES_SERIALIZED={rocm_model}",
                    "-P",
                    str(discovery),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stderr or completed.stdout,
            )
            registration = generated.read_text(encoding="utf-8")

        self.assertEqual(
            registration.count(
                'add_test("V2_Integration_Parity_Fake_ProductionCampaign_CPU_ALL_PRECISIONS"'
            ),
            1,
        )
        self.assertEqual(
            registration.count(
                'add_test("V2_Integration_Parity_Fake_ProductionCampaign_CUDA_ALL_PRECISIONS"'
            ),
            1,
        )
        self.assertEqual(
            registration.count(
                'add_test("V2_Integration_Parity_Fake_ProductionCampaign_CPU_CUDA_ROCm_ALL_PRECISIONS"'
            ),
            1,
        )
        self.assertIn(
            "MatrixSuite.ProductionParity/CPU_KV_FP16:"
            "MatrixSuite.ProductionParity/CPU_KV_Q8_1",
            registration,
        )
        self.assertIn("LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN=1", registration)
        cpu_campaign = registration.split(
            'add_test("V2_Integration_Parity_Fake_ProductionCampaign_CPU_ALL_PRECISIONS"',
            1,
        )[1].split(
            'add_test("V2_Integration_Parity_Fake_ProductionCampaign_CUDA_ALL_PRECISIONS"',
            1,
        )[0]
        cuda_campaign = registration.split(
            'add_test("V2_Integration_Parity_Fake_ProductionCampaign_CUDA_ALL_PRECISIONS"',
            1,
        )[1].split(
            'add_test("V2_Integration_Parity_Fake_ProductionCampaign_CPU_CUDA_ROCm_ALL_PRECISIONS"',
            1,
        )[0]
        hybrid_campaign = registration.split(
            'add_test("V2_Integration_Parity_Fake_ProductionCampaign_CPU_CUDA_ROCm_ALL_PRECISIONS"',
            1,
        )[1]
        self.assertIn("LLAMINAR_LOG_LEVEL=INFO", cpu_campaign)
        self.assertIn(
            f"{campaigns.MODEL_MANIFEST_ENV}={model}|{cpu_model}",
            cpu_campaign,
        )
        self.assertIn(f'REQUIRED_FILES "{model};{cpu_model}"', cpu_campaign)
        self.assertIn("LLAMINAR_FORCE_CPU_ONLY_STARTUP=1", cpu_campaign)
        self.assertNotIn("LLAMINAR_SKIP_ROCM_STARTUP=1", cpu_campaign)
        self.assertNotIn("NCCL", cpu_campaign)
        self.assertNotIn("RCCL", cpu_campaign)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", cpu_campaign)
        self.assertIn("LLAMINAR_LOG_LEVEL=INFO", cuda_campaign)
        self.assertIn(
            f"{campaigns.MODEL_MANIFEST_ENV}={model}|{cuda_model}",
            cuda_campaign,
        )
        self.assertIn(f'REQUIRED_FILES "{model};{cuda_model}"', cuda_campaign)
        self.assertIn("LLAMINAR_SKIP_ROCM_STARTUP=1", cuda_campaign)
        self.assertNotIn("LLAMINAR_FORCE_CPU_ONLY_STARTUP=1", cuda_campaign)
        self.assertIn("NCCL", cuda_campaign)
        self.assertNotIn("RCCL", cuda_campaign)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", cuda_campaign)
        self.assertIn("LLAMINAR_LOG_LEVEL=INFO", hybrid_campaign)
        self.assertIn(
            f"{campaigns.MODEL_MANIFEST_ENV}="
            f"{model}|{cpu_model}|{cuda_model}|{rocm_model}",
            hybrid_campaign,
        )
        self.assertIn(
            f'REQUIRED_FILES "{model};{cpu_model};{cuda_model};{rocm_model}"',
            hybrid_campaign,
        )
        self.assertNotIn("LLAMINAR_FORCE_CPU_ONLY_STARTUP=1", hybrid_campaign)
        self.assertNotIn("LLAMINAR_SKIP_CUDA_STARTUP=1", hybrid_campaign)
        self.assertNotIn("LLAMINAR_SKIP_ROCM_STARTUP=1", hybrid_campaign)
        self.assertIn("NCCL", hybrid_campaign)
        self.assertIn("RCCL", hybrid_campaign)
        self.assertIn("HSA_OVERRIDE_GFX_VERSION=9.0.6", hybrid_campaign)
        self.assertIn(
            f'TIMEOUT "{campaigns.REGISTERED_TIMEOUT_SECONDS:g}"',
            registration,
        )
        self.assertIn("FocusedInfrastructure_CPU", registration)
        self.assertNotIn("ProductionParity_CPU_KV_FP16", registration)

    def test_cmake_discovery_shards_campaigns_by_explicit_mpi_world(self) -> None:
        discovery = (
            REPO_ROOT / "tests" / "v2" / "cmake" /
            "V2ParityTestDiscovery.cmake"
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            executable = directory / "fake_gtest"
            executable.write_text(
                "#!/bin/sh\n"
                "cat <<'EOF'\n"
                "MatrixSuite.\n"
                "  ProductionParity/NodeTP_2xMPI_CPU\n"
                "  ProductionParity/NodeTP_4xMPI_CPU\n"
                "EOF\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            model = directory / "test.gguf"
            model.write_bytes(b"gguf")
            generated = directory / "discovered.cmake"

            completed = subprocess.run(
                [
                    "cmake",
                    f"-DTEST_EXECUTABLE={executable}",
                    f"-DCTEST_FILE={generated}",
                    "-DTEST_PREFIX=V2_Integration_Parity_FakeNodeTP",
                    "-DLABELS=V2;Integration;Parity;CPU;MPI",
                    "-DMPI_PROCS=2",
                    "-DNUM_SOCKETS=2",
                    "-DCORES_PER_SOCKET=4",
                    f"-DWORKING_DIR={directory}",
                    "-DTIMEOUT=90",
                    f"-DMODEL_FILES_SERIALIZED={model}",
                    "-P",
                    str(discovery),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stderr or completed.stdout,
            )
            registration = generated.read_text(encoding="utf-8")

        default_name = (
            "V2_Integration_Parity_FakeNodeTP_"
            "ProductionCampaign_CPU_ALL_PRECISIONS"
        )
        four_rank_name = (
            "V2_Integration_Parity_FakeNodeTP_"
            "ProductionCampaign_CPU_MPI_4_ALL_PRECISIONS"
        )
        self.assertIn(f'add_test("{default_name}"', registration)
        self.assertIn(f'add_test("{four_rank_name}"', registration)
        self.assertIn(
            '"-np" "2"',
            registration.split(f'add_test("{default_name}"', 1)[1].split("\n", 1)[0],
        )
        self.assertIn(
            '"-np" "4"',
            registration.split(f'add_test("{four_rank_name}"', 1)[1].split("\n", 1)[0],
        )
        self.assertIn("ProductionParity/NodeTP_2xMPI_CPU", registration)
        self.assertIn("ProductionParity/NodeTP_4xMPI_CPU", registration)

    def test_cmake_discovery_honors_focused_test_mpi_world_suffix(self) -> None:
        """A focused topology suffix overrides the target's default MPI world."""

        discovery = (
            REPO_ROOT / "tests" / "v2" / "cmake" /
            "V2ParityTestDiscovery.cmake"
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            executable = directory / "fake_gtest"
            executable.write_text(
                "#!/bin/sh\n"
                "cat <<'EOF'\n"
                "FocusedSuite.\n"
                "  SingleRankControl\n"
                "  MixedROCmCPUStaticPlacement_2xMPI\n"
                "EOF\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            generated = directory / "discovered.cmake"

            completed = subprocess.run(
                [
                    "cmake",
                    f"-DTEST_EXECUTABLE={executable}",
                    f"-DCTEST_FILE={generated}",
                    "-DTEST_PREFIX=V2_Integration_Parity_FocusedMPI",
                    "-DLABELS=V2;Integration;Parity;CPU;ROCm;MPI",
                    "-DMPI_PROCS=1",
                    "-DNUM_SOCKETS=2",
                    "-DCORES_PER_SOCKET=4",
                    f"-DWORKING_DIR={directory}",
                    "-DTIMEOUT=90",
                    "-DNO_MODELS=ON",
                    "-P",
                    str(discovery),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stderr or completed.stdout,
            )
            registration = generated.read_text(encoding="utf-8")

        control_name = (
            "V2_Integration_Parity_FocusedMPI_"
            "FocusedSuite_SingleRankControl"
        )
        mixed_name = (
            "V2_Integration_Parity_FocusedMPI_"
            "FocusedSuite_MixedROCmCPUStaticPlacement_2xMPI"
        )
        self.assertIn(f'add_test("{control_name}"', registration)
        self.assertIn(f'add_test("{mixed_name}"', registration)
        control_command = registration.split(
            f'add_test("{control_name}"', 1
        )[1].split("\n", 1)[0]
        mixed_command = registration.split(
            f'add_test("{mixed_name}"', 1
        )[1].split("\n", 1)[0]
        self.assertIn('"-np" "1"', control_command)
        self.assertIn('"-np" "2"', mixed_command)

    def test_cmake_discovery_isolates_current_batch_llep_policy(self) -> None:
        """Unfinished LLEP can be omitted without bypassing canonical campaigns."""

        discovery = (
            REPO_ROOT / "tests" / "v2" / "cmake" /
            "V2ParityTestDiscovery.cmake"
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            executable = directory / "fake_gtest"
            executable.write_text(
                "#!/bin/sh\n"
                "cat <<'EOF'\n"
                "MatrixSuite.\n"
                "  ProductionParity/CPU_Static_Ordinal\n"
                "  ProductionParity/CPU_DynamicMaintenance_Ordinal\n"
                "  ProductionParity/CPU_CurrentBatchLLEP_Ordinal\n"
                "EOF\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            model = directory / "test.gguf"
            model.write_bytes(b"gguf")
            generated = directory / "discovered.cmake"

            completed = subprocess.run(
                [
                    "cmake",
                    f"-DTEST_EXECUTABLE={executable}",
                    f"-DCTEST_FILE={generated}",
                    "-DTEST_PREFIX=V2_Integration_Parity_FakePolicy",
                    "-DLABELS=V2;Integration;Parity;CPU;Static;Dynamic;LLEP",
                    "-DMPI_PROCS=1",
                    "-DNUM_SOCKETS=1",
                    "-DCORES_PER_SOCKET=4",
                    f"-DWORKING_DIR={directory}",
                    f"-DMODEL_FILES_SERIALIZED={model}",
                    "-P",
                    str(discovery),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stderr or completed.stdout,
            )
            registration = generated.read_text(encoding="utf-8")

        main_name = (
            "V2_Integration_Parity_FakePolicy_"
            "ProductionCampaign_CPU_ALL_PRECISIONS"
        )
        llep_name = (
            "V2_Integration_Parity_FakePolicy_LLEP_"
            "FocusedPolicySlice_CPU_ALL_PRECISIONS"
        )
        self.assertIn(f'add_test("{main_name}"', registration)
        self.assertIn(f'add_test("{llep_name}"', registration)
        main_command = registration.split(
            f'add_test("{main_name}"', 1
        )[1].split("\n", 1)[0]
        llep_command = registration.split(
            f'add_test("{llep_name}"', 1
        )[1].split("\n", 1)[0]
        self.assertIn("ProductionParity/CPU_Static_Ordinal", main_command)
        self.assertIn(
            "ProductionParity/CPU_DynamicMaintenance_Ordinal",
            main_command,
        )
        self.assertNotIn("CurrentBatchLLEP", main_command)
        self.assertIn(
            "ProductionParity/CPU_CurrentBatchLLEP_Ordinal",
            llep_command,
        )
        llep_registration = registration.split(
            f'add_test("{llep_name}"', 1
        )[1]
        llep_properties = llep_registration.split(
            "set_tests_properties", 1
        )[1].split(")\n\n", 1)[0]
        self.assertIn("UnfinishedPolicy", llep_properties)
        self.assertNotIn("WholeMatrix75MinuteTarget", llep_properties)
        self.assertNotRegex(llep_properties, r'LABELS "[^"]*Campaign')

    def test_cmake_discovery_isolates_declared_application_lifetimes(self) -> None:
        discovery = (
            REPO_ROOT / "tests" / "v2" / "cmake" /
            "V2ParityTestDiscovery.cmake"
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            executable = directory / "fake_gtest"
            executable.write_text(
                "#!/bin/sh\n"
                "cat <<'EOF'\n"
                "MatrixSuite.\n"
                "  ProductionParity/CUDA_ROCm_CPU\n"
                "  ProductionParity/SegmentedPrefill_CUDA_ROCm_CPU\n"
                "EOF\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            model = directory / "test.gguf"
            model.write_bytes(b"gguf")
            generated = directory / "discovered.cmake"

            completed = subprocess.run(
                [
                    "cmake",
                    f"-DTEST_EXECUTABLE={executable}",
                    f"-DCTEST_FILE={generated}",
                    "-DTEST_PREFIX=V2_Integration_Parity_FakeOverlay",
                    "-DLABELS=V2;Integration;Parity;CPU;CUDA;ROCm;MPI",
                    "-DMPI_PROCS=2",
                    "-DNUM_SOCKETS=2",
                    "-DCORES_PER_SOCKET=4",
                    f"-DWORKING_DIR={directory}",
                    f"-DMODEL_FILES_SERIALIZED={model}",
                    "-DPRODUCTION_ISOLATED_TESTS_SERIALIZED=SegmentedPrefill",
                    "-P",
                    str(discovery),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stderr or completed.stdout,
            )
            registration = generated.read_text(encoding="utf-8")

        ordinary_name = (
            "V2_Integration_Parity_FakeOverlay_ProductionCampaign_"
            "CPU_CUDA_ROCm_ALL_PRECISIONS"
        )
        isolated_name_match = re.search(
            r'add_test\("(V2_Integration_Parity_FakeOverlay_'
            r'Isolated_SegmentedPrefill_[0-9a-f]{12}_ProductionCampaign_'
            r'CPU_CUDA_ROCm_ALL_PRECISIONS)"',
            registration,
        )
        self.assertIn(f'add_test("{ordinary_name}"', registration)
        self.assertIsNotNone(isolated_name_match)
        assert isolated_name_match is not None
        ordinary_command = registration.split(
            f'add_test("{ordinary_name}"', 1
        )[1].split("\n", 1)[0]
        isolated_command = registration.split(
            f'add_test("{isolated_name_match.group(1)}"', 1
        )[1].split("\n", 1)[0]
        self.assertIn("ProductionParity/CUDA_ROCm_CPU", ordinary_command)
        self.assertNotIn("SegmentedPrefill", ordinary_command)
        self.assertIn(
            "ProductionParity/SegmentedPrefill_CUDA_ROCm_CPU",
            isolated_command,
        )
        self.assertIn('LABELS "', registration)
        self.assertIn("FreshMPIWorld", registration)

    def test_exact_precision_types_are_derived_from_gtest_cells(self) -> None:
        cell = campaigns.CampaignCell(
            "V2_Parity_ProductionCampaign_CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "ALL"),
            (
                "Suite.ProductionParity/CPU_KV_TQ",
                "Suite.ProductionParity/CPU_KV_FP16",
                "Suite.ProductionParity/CPU_KV_Q8_1",
                "Suite.ProductionParity/"
                "Qwen36_CUDA0_ActFP32_KVFP32_MTPDepth3",
            ),
        )

        self.assertEqual(
            cell.precision_types,
            ("FP16", "FP32", "Q8_1", "TQ"),
        )

    def test_selected_models_expand_split_gguf_and_deduplicate_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            part_one = directory / "model-00001-of-00002.gguf"
            part_two = directory / "model-00002-of-00002.gguf"
            part_one.write_bytes(b"one")
            part_two.write_bytes(b"two")
            alias = directory / "alias.gguf"
            alias.symlink_to(part_one)
            cells = [
                campaigns.CampaignCell(
                    "cell_ProductionCampaign_CPU_ALL_PRECISIONS",
                    campaigns.CampaignGroup("CPU", "ALL"),
                    model_files=(str(part_one), str(alias)),
                )
            ]

            selected = campaigns.selected_model_files(cells)

        self.assertEqual(selected, (part_one.resolve(), part_two.resolve()))

    @mock.patch.object(campaigns, "_filesystem_type", return_value="tmpfs")
    def test_ramdisk_staging_records_source_digest_and_publishes_read_only_models(
        self, filesystem_type: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = directory / "model.gguf"
            payload = (b"real-weights" * 1024) + b"tail"
            source.write_bytes(payload)
            cell = campaigns.CampaignCell(
                "cell_ProductionCampaign_CPU_ALL_PRECISIONS",
                campaigns.CampaignGroup("CPU", "ALL"),
                model_files=(str(source),),
            )
            staging = directory / "staged"

            evidence, actual_filesystem = campaigns.stage_models_in_ramdisk(
                [cell], staging, campaigns.time.monotonic() + 30.0
            )

            staged = staging / source.name
            self.assertEqual(staged.read_bytes(), payload)
            self.assertEqual(staged.stat().st_mode & 0o777, 0o400)
            self.assertEqual(actual_filesystem, "tmpfs")
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0].size_bytes, len(payload))
            self.assertEqual(
                evidence[0].sha256,
                campaigns.hashlib.sha256(payload).hexdigest(),
            )
            filesystem_type.assert_called_once_with(staging)

    @mock.patch.object(campaigns, "_filesystem_type", return_value="tmpfs")
    def test_persistent_ramdisk_cache_reuses_identity_stable_hits_without_reread(
        self, _: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = directory / "model.gguf"
            payload = (b"persistent-real-weights" * 1024) + b"tail"
            source.write_bytes(payload)
            cell = campaigns.CampaignCell(
                "cell_ProductionCampaign_CPU_ALL_PRECISIONS",
                campaigns.CampaignGroup("CPU", "ALL"),
                model_files=(str(source),),
            )
            staging = directory / "cache" / "models"

            with mock.patch.object(
                campaigns,
                "_stage_one_model",
                wraps=campaigns._stage_one_model,
            ) as stage_one, mock.patch.object(
                campaigns,
                "_hash_file",
                wraps=campaigns._hash_file,
            ) as hash_file:
                first, _ = campaigns.stage_models_in_ramdisk(
                    [cell],
                    staging,
                    campaigns.time.monotonic() + 30.0,
                    persistent=True,
                )
                second, _ = campaigns.stage_models_in_ramdisk(
                    [cell],
                    staging,
                    campaigns.time.monotonic() + 30.0,
                    persistent=True,
                )

            self.assertEqual(stage_one.call_count, 1)
            self.assertEqual(
                hash_file.call_count,
                0,
                "staging must leave the one destination SHA pass to the "
                "reference-pack authentication gate",
            )
            self.assertEqual(first[0].cache_status, "copied")
            self.assertEqual(second[0].cache_status, "reused")
            self.assertEqual(second[0].sha256, first[0].sha256)
            self.assertEqual((staging / source.name).read_bytes(), payload)
            manifest = json.loads(
                (staging.parent / "model-cache-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["schema_version"],
                campaigns.PERSISTENT_MODEL_CACHE_SCHEMA_VERSION,
            )
            self.assertEqual(
                manifest["models"][source.name]["cached_identity"],
                list(campaigns._source_identity((staging / source.name).stat())),
            )

    @mock.patch.object(campaigns, "_filesystem_type", return_value="tmpfs")
    def test_persistent_ramdisk_cache_refreshes_changed_source_atomically(
        self, _: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = directory / "model.gguf"
            source.write_bytes(b"old-real-weights")
            cell = campaigns.CampaignCell(
                "cell_ProductionCampaign_CPU_ALL_PRECISIONS",
                campaigns.CampaignGroup("CPU", "ALL"),
                model_files=(str(source),),
            )
            staging = directory / "cache" / "models"
            campaigns.stage_models_in_ramdisk(
                [cell],
                staging,
                campaigns.time.monotonic() + 30.0,
                persistent=True,
            )

            source.write_bytes(b"new-real-weights")
            refreshed, _ = campaigns.stage_models_in_ramdisk(
                [cell],
                staging,
                campaigns.time.monotonic() + 30.0,
                persistent=True,
            )

            self.assertEqual(refreshed[0].cache_status, "copied")
            self.assertEqual(
                (staging / source.name).read_bytes(),
                b"new-real-weights",
            )

    @mock.patch.object(campaigns, "_filesystem_type", return_value="tmpfs")
    def test_persistent_workspace_survives_exit_for_manual_cleanup(
        self, _: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            ramdisk = Path(raw_directory)
            root = ramdisk / "stable-parity-cache"
            try:
                with campaigns.model_staging_workspace(
                    ramdisk,
                    Path("stable-parity-cache"),
                    campaigns.time.monotonic() + 30.0,
                ) as workspace:
                    marker = workspace.root / "operator-owned-marker"
                    marker.write_text("retained", encoding="utf-8")
                    self.assertTrue(workspace.persistent)
                    self.assertEqual(workspace.mode, "persistent")

                self.assertEqual(marker.read_text(encoding="utf-8"), "retained")
                self.assertEqual(root.stat().st_mode & 0o777, 0o500)
            finally:
                if root.exists():
                    campaigns._set_persistent_cache_write_access(
                        root,
                        writable=True,
                    )

    @mock.patch.object(campaigns, "_filesystem_type", return_value="tmpfs")
    def test_persistent_workspace_seals_models_while_digests_remain_writable(
        self, _: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            ramdisk = Path(raw_directory)
            root = ramdisk / "stable-parity-cache"
            try:
                with campaigns.model_staging_workspace(
                    ramdisk,
                    Path("stable-parity-cache"),
                    campaigns.time.monotonic() + 30.0,
                ) as workspace:
                    workspace.models.mkdir()
                    workspace.digests.mkdir()
                    published_model = workspace.models / "model.gguf"
                    published_model.write_bytes(b"published-model")

                    workspace.protect_published_models()

                    self.assertEqual(workspace.root.stat().st_mode & 0o777, 0o500)
                    self.assertEqual(workspace.models.stat().st_mode & 0o777, 0o500)
                    self.assertEqual(workspace.digests.stat().st_mode & 0o777, 0o700)
                    if os.geteuid() != 0:
                        with self.assertRaises(PermissionError):
                            published_model.unlink()
                    digest = workspace.digests / "model.sha256"
                    digest.write_text("authenticated", encoding="utf-8")
                    self.assertEqual(
                        digest.read_text(encoding="utf-8"),
                        "authenticated",
                    )

                self.assertEqual(root.stat().st_mode & 0o777, 0o500)
                self.assertEqual((root / "models").stat().st_mode & 0o777, 0o500)
                self.assertEqual((root / "digests").stat().st_mode & 0o777, 0o500)
                if os.geteuid() != 0:
                    with self.assertRaises(PermissionError):
                        published_model.unlink()
            finally:
                if root.exists():
                    campaigns._set_persistent_cache_write_access(
                        root,
                        writable=True,
                    )

    @mock.patch.object(campaigns, "_filesystem_type", return_value="tmpfs")
    def test_persistent_workspace_reclaims_only_interrupted_transactions(
        self, _: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            ramdisk = Path(raw_directory)
            root = ramdisk / "stable-parity-cache"
            try:
                models = root / "models"
                models.mkdir(parents=True)
                interrupted_copy = models / ".model.gguf.copying-4242"
                interrupted_copy.write_bytes(b"partial-model")
                interrupted_manifest = (
                    root
                    / ".model-cache-manifest.json.publishing-4242-123456789"
                )
                interrupted_manifest.write_text("partial", encoding="utf-8")
                published_model = models / "model.gguf"
                published_model.write_bytes(b"published-model")
                operator_marker = root / "operator-owned-marker"
                operator_marker.write_text("retained", encoding="utf-8")

                with campaigns.model_staging_workspace(
                    ramdisk,
                    Path("stable-parity-cache"),
                    campaigns.time.monotonic() + 30.0,
                ) as workspace:
                    self.assertEqual(workspace.root, root)
                    self.assertFalse(interrupted_copy.exists())
                    self.assertFalse(interrupted_manifest.exists())
                    self.assertEqual(
                        published_model.read_bytes(),
                        b"published-model",
                    )
                    self.assertEqual(
                        operator_marker.read_text(encoding="utf-8"),
                        "retained",
                    )
            finally:
                if root.exists():
                    campaigns._set_persistent_cache_write_access(
                        root,
                        writable=True,
                    )

    @mock.patch.object(campaigns, "_filesystem_type", return_value="tmpfs")
    def test_persistent_workspace_rejects_transaction_directory(
        self, _: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            ramdisk = Path(raw_directory)
            root = ramdisk / "stable-parity-cache"
            try:
                transaction = root / "models" / ".model.gguf.copying-4242"
                transaction.mkdir(parents=True)

                with self.assertRaisesRegex(
                    campaigns.ModelStagingError,
                    "transaction path is unexpectedly a directory",
                ):
                    with campaigns.model_staging_workspace(
                        ramdisk,
                        Path("stable-parity-cache"),
                        campaigns.time.monotonic() + 30.0,
                    ):
                        self.fail("corrupt transaction directory was accepted")
            finally:
                if root.exists():
                    campaigns._set_persistent_cache_write_access(
                        root,
                        writable=True,
                    )

    def test_persistent_cache_cli_is_explicit(self) -> None:
        args = campaigns.parse_args(
            ["--persistent-model-cache-dir", "/dev/shm/llaminar-parity-cache"]
        )

        self.assertEqual(
            args.persistent_model_cache_dir,
            Path("/dev/shm/llaminar-parity-cache"),
        )

    @mock.patch.object(campaigns, "_filesystem_type", return_value="tmpfs")
    def test_persistent_workspace_rejects_session_ipc_tmpfs(
        self, _: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            session_ipc = Path(raw_directory)
            with mock.patch.object(
                campaigns,
                "SESSION_IPC_RAMDISK_ROOT",
                session_ipc,
            ):
                with self.assertRaisesRegex(
                    campaigns.ModelStagingError,
                    "requires a dedicated tmpfs mount outside",
                ):
                    with campaigns.model_staging_workspace(
                        session_ipc,
                        Path("persistent-cache"),
                        campaigns.time.monotonic() + 30.0,
                    ):
                        self.fail("session IPC was accepted as persistent storage")

    def test_stage_models_only_requires_a_persistent_cache(self) -> None:
        with self.assertRaises(SystemExit):
            campaigns.parse_args(["--stage-models-only"])

    def test_stage_models_only_and_list_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            campaigns.parse_args(
                [
                    "--stage-models-only",
                    "--persistent-model-cache-dir",
                    "stable-cache",
                    "--list",
                ]
            )

    @mock.patch.object(campaigns, "prepare_model_fixture", return_value=(0, 0.5))
    @mock.patch.object(campaigns, "_filesystem_type", return_value="tmpfs")
    def test_stage_models_only_uses_the_persistent_campaign_lifecycle(
        self,
        _: mock.Mock,
        prepare_fixture: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            ramdisk = Path(raw_directory)
            source = ramdisk / "source.gguf"
            source.write_bytes(b"real-weights")
            cell = campaigns.CampaignCell(
                "cell_ProductionCampaign_CPU_ALL_PRECISIONS",
                campaigns.CampaignGroup("CPU", "ALL"),
                model_files=(str(source),),
            )
            args = campaigns.parse_args(
                [
                    "--stage-models-only",
                    "--model-ramdisk-root",
                    str(ramdisk),
                    "--persistent-model-cache-dir",
                    "stable-cache",
                ]
            )
            cache = ramdisk / "stable-cache"
            try:
                self.assertEqual(
                    campaigns.stage_selected_models_only(args, (cell,)),
                    0,
                )
                prepare_fixture.assert_called_once()
                self.assertEqual(
                    (cache / "models" / source.name).read_bytes(),
                    b"real-weights",
                )
                self.assertEqual(cache.stat().st_mode & 0o777, 0o500)
                self.assertEqual(
                    (cache / "models").stat().st_mode & 0o777,
                    0o500,
                )
            finally:
                if cache.exists():
                    campaigns._set_persistent_cache_write_access(
                        cache,
                        writable=True,
                    )

    @mock.patch.object(campaigns, "_filesystem_type", return_value="ext4")
    def test_ramdisk_staging_rejects_non_memory_filesystem(
        self, _: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = directory / "model.gguf"
            source.write_bytes(b"weights")
            cell = campaigns.CampaignCell(
                "cell_ProductionCampaign_CPU_ALL_PRECISIONS",
                campaigns.CampaignGroup("CPU", "ALL"),
                model_files=(str(source),),
            )

            with self.assertRaisesRegex(
                campaigns.ModelStagingError, "requires tmpfs/ramfs"
            ):
                campaigns.stage_models_in_ramdisk(
                    [cell], directory / "staged", campaigns.time.monotonic() + 30.0
                )

    @mock.patch.object(campaigns.subprocess, "run")
    def test_discovery_rejects_wildcard_campaign_filters(
        self, run: mock.Mock
    ) -> None:
        name = "V2_Parity_ProductionCampaign_CUDA_ALL_PRECISIONS"
        document = json.loads(ctest_document(name))
        document["tests"][0]["command"][-1] = (
            "--gtest_filter=Suite.ProductionParity/*CUDA*"
        )
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(document), stderr=""
        )

        with self.assertRaisesRegex(RuntimeError, "wildcard-free|enumerate exact"):
            campaigns.discover_campaigns(Path("build"))


if __name__ == "__main__":
    unittest.main()
