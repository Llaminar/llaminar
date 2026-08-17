#!/usr/bin/env python3
"""Unit tests for production parity campaign discovery and sharding."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "run_production_parity_campaigns.py"
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
                                "WholeMatrixOneHourTarget",
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
        self.assertIn("100% tests passed, 0 tests failed out of 1", completed.stdout)

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
                    "-DLABELS=V2;Integration;Parity;CPU;CUDA",
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
        self.assertIn("LLAMINAR_LOG_LEVEL=INFO", cuda_campaign)
        self.assertIn(
            f"{campaigns.MODEL_MANIFEST_ENV}={model}|{cuda_model}",
            cuda_campaign,
        )
        self.assertIn(f'REQUIRED_FILES "{model};{cuda_model}"', cuda_campaign)
        self.assertIn("LLAMINAR_SKIP_ROCM_STARTUP=1", cuda_campaign)
        self.assertNotIn("LLAMINAR_FORCE_CPU_ONLY_STARTUP=1", cuda_campaign)
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
            ),
        )

        self.assertEqual(cell.precision_types, ("FP16", "Q8_1", "TQ"))

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

    @mock.patch.object(campaigns, "_filesystem_type", return_value="tmpfs")
    def test_persistent_workspace_reclaims_only_interrupted_transactions(
        self, _: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            ramdisk = Path(raw_directory)
            root = ramdisk / "stable-parity-cache"
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
                self.assertEqual(published_model.read_bytes(), b"published-model")
                self.assertEqual(
                    operator_marker.read_text(encoding="utf-8"),
                    "retained",
                )

    @mock.patch.object(campaigns, "_filesystem_type", return_value="tmpfs")
    def test_persistent_workspace_rejects_transaction_directory(
        self, _: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            ramdisk = Path(raw_directory)
            transaction = (
                ramdisk
                / "stable-parity-cache"
                / "models"
                / ".model.gguf.copying-4242"
            )
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

    def test_persistent_cache_cli_is_explicit(self) -> None:
        args = campaigns.parse_args(
            ["--persistent-model-cache-dir", "/dev/shm/llaminar-parity-cache"]
        )

        self.assertEqual(
            args.persistent_model_cache_dir,
            Path("/dev/shm/llaminar-parity-cache"),
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
