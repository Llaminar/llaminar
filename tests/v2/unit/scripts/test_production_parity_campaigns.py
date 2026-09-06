#!/usr/bin/env python3
"""Device-free production campaign discovery, staging and watchdog contracts.

Synthetic artifacts cover the canonical matrix without loading models. Deadline
transition proofs use a controlled clock; explicit process-group tests retain
real children to verify cancellation and cleanup independently of that logic.
"""

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
                        {
                            "name": "WORKING_DIRECTORY",
                            "value": "/workspaces/llaminar",
                        },
                    ],
                    "command": [
                        "v2_integration_parity_fake_single_device_matrix",
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


def unit_ctest_document(*names: str) -> str:
    """Build a complete model-free Unit registration document."""

    return json.dumps(
        {
            "tests": [
                {
                    "name": name,
                    "properties": [
                        {
                            "name": "LABELS",
                            "value": ["V2", campaigns.PRODUCTION_PARITY_UNIT_LABEL],
                        },
                        {"name": "TIMEOUT", "value": 30.0},
                    ],
                    "command": ["fake_unit_test"],
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

    def test_prefix_restore_schema_requires_cross_epoch_gdn_evidence(self) -> None:
        """Keep the validator synchronized with typed GDN state comparison."""

        columns = next(
            campaigns.csv.reader(
                [campaigns.REQUIRED_CSV_HEADERS["prefix_restore.csv"]]
            )
        )

        self.assertEqual(
            columns[
                columns.index("main_kv_numerically_passed") + 1 :
                columns.index("terminal_hidden_policy")
            ],
            [
                "gdn_state_policy",
                "gdn_numerically_compared",
                "gdn_numerical_payloads",
                "gdn_numerical_elements",
                "gdn_minimum_cosine",
                "gdn_maximum_relative_l2",
                "gdn_maximum_abs",
                "gdn_numerically_passed",
            ],
        )

    @mock.patch.object(campaigns.subprocess, "run")
    def test_complete_unit_inventory_is_discovered_from_ctest(
        self,
        run: mock.Mock,
    ) -> None:
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=unit_ctest_document(
                "V2_Unit_PhysicalMemoryAuthority",
                "V2_Unit_MemoryPlanner",
            ),
            stderr="",
        )

        discovered = campaigns.discover_production_parity_unit_tests(
            Path("build")
        )

        self.assertEqual(
            discovered,
            (
                "V2_Unit_MemoryPlanner",
                "V2_Unit_PhysicalMemoryAuthority",
            ),
        )
        command = run.call_args.args[0]
        self.assertIn("--show-only=json-v1", command)

    @mock.patch.object(campaigns.subprocess, "run")
    def test_unit_inventory_rejects_model_fixture_dependency(
        self,
        run: mock.Mock,
    ) -> None:
        document = json.loads(unit_ctest_document("V2_Unit_Invalid"))
        document["tests"][0]["properties"].append(
            {"name": "FIXTURES_REQUIRED", "value": ["V2_Models"]}
        )
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(document), stderr=""
        )

        with self.assertRaisesRegex(RuntimeError, "requires a fixture"):
            campaigns.discover_production_parity_unit_tests(Path("build"))

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
    @mock.patch.object(
        campaigns,
        "discover_production_parity_unit_tests",
        return_value=("V2_Unit_PhysicalMemoryAuthority",),
    )
    def test_preflight_executes_the_ctest_label_before_campaign_admission(
        self,
        discover_units: mock.Mock,
        discover: mock.Mock,
        run_process: mock.Mock,
    ) -> None:
        return_code, _, tests = campaigns.run_production_parity_preflight(
            Path("build"), 60.0
        )

        self.assertEqual(return_code, 0)
        self.assertEqual(
            tests,
            (
                "V2_Unit_PhysicalMemoryAuthority",
                "V2_Integration_ParityCellLifecycle_MPI1",
            ),
        )
        discover_units.assert_called_once_with(Path("build"))
        discover.assert_called_once_with(Path("build"))
        self.assertEqual(run_process.call_count, 3)
        build_command = run_process.call_args_list[0].args[0]
        unit_command = run_process.call_args_list[1].args[0]
        command = run_process.call_args_list[2].args[0]
        self.assertIn(campaigns.PRODUCTION_PARITY_UNIT_BUILD_TARGET, build_command)
        self.assertIn("--parallel", unit_command)
        self.assertIn("--no-tests=error", unit_command)
        self.assertIn(f"^{campaigns.PRODUCTION_PARITY_UNIT_PREFIX}", unit_command)
        self.assertIn(
            f"^{campaigns.PRODUCTION_PARITY_PREFLIGHT_LABEL}$",
            command,
        )

    @mock.patch.object(
        campaigns,
        "discover_production_parity_preflight_tests",
        return_value=("V2_Integration_ParityCellLifecycle_MPI1",),
    )
    @mock.patch.object(
        campaigns,
        "discover_production_parity_unit_tests",
        return_value=("V2_Unit_PhysicalMemoryAuthority",),
    )
    def test_individual_preflight_receipt_requires_unchanged_build_identity(
        self,
        discover_units: mock.Mock,
        discover: mock.Mock,
    ) -> None:
        """Amortize preflight without allowing an unchecked skip switch."""

        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            build = root / "build"
            build.mkdir()
            boundaries = (
                build / ".ninja_log",
                build / "build.ninja",
                build / "CTestTestfile.cmake",
            )
            for boundary in boundaries:
                boundary.write_text("build identity\n", encoding="utf-8")
                os.utime(boundary, ns=(1_000_000_000, 1_000_000_000))

            report = root / "individual.json"
            report.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            campaigns.INDIVIDUAL_PROGRESS_REPORT_SCHEMA_VERSION
                        ),
                        "mode": "sequential_unseen_exact_cells",
                        "certification_eligible": False,
                        "preflight_return_code": 0,
                        "preflight_test_count": 2,
                        "preflight_tests": [
                            "V2_Unit_PhysicalMemoryAuthority",
                            "V2_Integration_ParityCellLifecycle_MPI1",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            os.utime(report, ns=(2_000_000_000, 2_000_000_000))

            return_code, elapsed, tests = (
                campaigns.reuse_unchanged_production_parity_preflight(
                    build,
                    report,
                )
            )

            self.assertEqual(return_code, 0)
            self.assertEqual(elapsed, 0.0)
            self.assertEqual(
                tests,
                (
                    "V2_Unit_PhysicalMemoryAuthority",
                    "V2_Integration_ParityCellLifecycle_MPI1",
                ),
            )
            discover_units.assert_called_once_with(build.resolve())
            discover.assert_called_once_with(build.resolve())

            os.utime(
                boundaries[0],
                ns=(3_000_000_000, 3_000_000_000),
            )
            with self.assertRaisesRegex(ValueError, "build or CTest"):
                campaigns.reuse_unchanged_production_parity_preflight(
                    build,
                    report,
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
        """Every compiled fixture family must use the typed matrix expander."""

        parity_root = REPO_ROOT / "tests" / "v2" / "integration" / "parity"
        cmake_root = REPO_ROOT / "tests" / "v2"
        cmake = (cmake_root / "CMakeLists.txt").read_text(encoding="utf-8")
        # Compilation boundaries are not configuration authorities. Follow the
        # declared object-library links so a thin registration can use a shared
        # typed expander without copying it into every implementation shard.
        sources_by_target = {
            target: {
                (cmake_root / source).resolve()
                for source in re.findall(r"(?<!\S)[\w./-]+\.cpp(?!\S)", body)
            }
            for target, body in re.findall(
                r"add_(?:executable|library)\(\s*(\w+)\s+(.*?)\)",
                cmake, re.DOTALL | re.IGNORECASE,
            )
        }
        links_by_target: dict[str, set[str]] = {}
        for target, body in re.findall(
            r"target_link_libraries\(\s*(\w+)\s+(.*?)\)",
            cmake, re.DOTALL | re.IGNORECASE,
        ):
            links_by_target.setdefault(target, set()).update(body.split())

        def source_closure(target: str) -> set[Path]:
            """Collect declared source ownership, visiting shared objects once."""
            pending = [target]
            seen: set[str] = set()
            sources: set[Path] = set()
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                sources.update(sources_by_target.get(current, set()))
                pending.extend(links_by_target.get(current, set()) - seen)
            return sources

        contents_by_path: dict[Path, str] = {}
        def contents_for(path: Path) -> str:
            """Read each shared source once, without interpreting model files."""
            if path not in contents_by_path:
                contents_by_path[path] = path.read_text(encoding="utf-8")
            return contents_by_path[path]

        contracts_by_target = {
            target: "\n".join(contents_for(path) for path in source_closure(target))
            for target in sources_by_target
            if any(parity_root in path.parents for path in sources_by_target[target])
        }
        offenders: list[str] = []
        for source in sorted(parity_root.rglob("*.cpp")):
            contents = contents_for(source)
            if "ProductionParity" not in contents:
                continue
            contracts = [
                contracts_by_target[target]
                for target, sources in sources_by_target.items()
                if source.resolve() in sources
            ]
            if not any(
                "ModelParityDefinition" in contract and
                "expandModelParityDefinition" in contract
                for contract in contracts
            ):
                offenders.append(str(source.relative_to(REPO_ROOT)))

        self.assertEqual(
            offenders,
            [],
            "ProductionParity source families must declare and expand the canonical "
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
                "Qwen35MoE_35B_NodeExpertOverlay_ProductionCampaign_CUDA_ALL_PRECISIONS",
                campaigns.CampaignGroup("CUDA", "ALL"),
            ),
        ]

        selected = campaigns.filter_campaigns(
            cells,
            ".*",
            ".*",
            ".*",
            ".*ExpertOverlay.*",
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

    def test_individual_command_narrows_only_registered_exact_filter(self) -> None:
        cases = (
            "MatrixSuite.ProductionParity/Ordinal",
            "MatrixSuite.ProductionParity/Random",
        )
        cell = campaigns.CampaignCell(
            "V2_Parity_ProductionCampaign_CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "ALL"),
            gtest_cases=cases,
            command=(
                "/usr/bin/mpirun",
                "-np",
                "2",
                "v2_integration_parity_fake_single_device_matrix",
                "--flag=retained",
                "--gtest_filter=" + ":".join(cases),
            ),
            environment=("OMP_NUM_THREADS=28",),
            working_directory="/workspaces/llaminar",
        )

        exact = campaigns._exact_registered_command(cell, cases[1])

        self.assertEqual(exact[:-1], cell.command[:-1])
        self.assertEqual(exact[-1], f"--gtest_filter={cases[1]}")
        self.assertEqual(
            campaigns._single_exact_cell(cell, cases[1]).gtest_cases,
            (cases[1],),
        )

    def test_individual_command_rejects_unregistered_or_ambiguous_filter(self) -> None:
        case = "MatrixSuite.ProductionParity/Ordinal"
        missing_filter = campaigns.CampaignCell(
            "V2_Parity_ProductionCampaign_CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "ALL"),
            gtest_cases=(case,),
            command=("v2_integration_parity_fake_single_device_matrix",),
        )
        duplicate_filter = campaigns.CampaignCell(
            missing_filter.name,
            missing_filter.group,
            gtest_cases=(case,),
            command=(
                "v2_integration_parity_fake_single_device_matrix",
                f"--gtest_filter={case}",
                f"--gtest_filter={case}",
            ),
        )

        with self.assertRaisesRegex(ValueError, "exactly one"):
            campaigns._exact_registered_command(missing_filter, case)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            campaigns._exact_registered_command(duplicate_filter, case)
        with self.assertRaisesRegex(ValueError, "is not registered"):
            campaigns._exact_registered_command(missing_filter, case + "X")

    @mock.patch.object(campaigns, "validate_campaign_artifacts")
    @mock.patch.object(campaigns, "_run_process", return_value=0)
    def test_individual_cell_uses_registered_process_contract(
        self,
        run_process: mock.Mock,
        validate: mock.Mock,
    ) -> None:
        case = "MatrixSuite.ProductionParity/Ordinal_KVFP16_MTPOff"
        cell = campaigns.CampaignCell(
            "V2_Parity_ProductionCampaign_CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "ALL"),
            gtest_cases=(case,),
            command=("v2_integration_parity_fake_single_device_matrix", f"--gtest_filter={case}"),
            environment=("OMP_NUM_THREADS=28", "MODE=registered"),
            working_directory="/workspaces/llaminar",
        )
        artifact_directory = "/artifacts/cell"
        validate.return_value = (
            len(campaigns.REQUIRED_CSV_HEADERS),
            (artifact_directory,),
            (),
        )

        result = campaigns.run_individual_cell(
            cell,
            case,
            600.0,
            environment_overrides={"MODE": "override", "EXTRA": "1"},
            artifact_results_root=Path("/artifacts"),
        )

        self.assertEqual(result.return_code, 0)
        self.assertEqual(result.gtest_cases, (case,))
        self.assertEqual(
            run_process.call_args.args[0],
            ["v2_integration_parity_fake_single_device_matrix", f"--gtest_filter={case}"],
        )
        self.assertEqual(
            run_process.call_args.kwargs["working_directory"],
            Path("/workspaces/llaminar"),
        )
        self.assertEqual(
            run_process.call_args.kwargs["environment_overrides"],
            {"OMP_NUM_THREADS": "28", "MODE": "override", "EXTRA": "1"},
        )

    @mock.patch.object(campaigns, "_current_git_short_hash", return_value="abc123")
    def test_green_ledger_requires_exit_zero_and_fresh_artifact_contract(
        self,
        _revision: mock.Mock,
    ) -> None:
        cases = (
            "MatrixSuite.ProductionParity/Ordinal_KVFP16_MTPOff",
            "MatrixSuite.ProductionParity/Random_KVFP16_MTPOff",
        )
        cell = campaigns.CampaignCell(
            "V2_Parity_ProductionCampaign_CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "ALL"),
            gtest_cases=cases,
            command=("v2_integration_parity_fake_single_device_matrix", "--gtest_filter=" + ":".join(cases)),
            environment=("MODE=production",),
            working_directory="/workspaces/llaminar",
        )
        expected_count = len(campaigns.REQUIRED_CSV_HEADERS)
        green = campaigns.CampaignResult(
            campaign=cell.name,
            test_type=cell.test_type,
            backends="CPU",
            precision_set="ALL",
            precision_types=("FP16",),
            gtest_cases=(cases[0],),
            elapsed_seconds=1.0,
            target_seconds=4500.0,
            return_code=0,
            target_met=True,
            outcome="completed",
            artifact_contract_passed=True,
            validated_artifact_file_count=expected_count,
            artifact_directories=("/fresh/ordinal",),
        )
        failed_but_complete = campaigns.CampaignResult(
            **{
                **green.__dict__,
                "gtest_cases": (cases[1],),
                "return_code": 8,
                "artifact_directories": ("/fresh/random",),
            }
        )

        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "green-ledger.json"
            ledger = campaigns.record_individual_green(path, cell, green)
            self.assertEqual(ledger.green_cases, frozenset((cases[0],)))
            self.assertEqual(
                campaigns.unseen_individual_cells((cell,), ledger),
                ((cell, cases[1]),),
            )
            with self.assertRaisesRegex(ValueError, "refusing to record"):
                campaigns.record_individual_green(
                    path,
                    cell,
                    failed_but_complete,
                )
            reloaded = campaigns.load_individual_green_ledger(path)

        self.assertEqual(reloaded.green_cases, frozenset((cases[0],)))
        self.assertFalse(reloaded.certification_eligible)

    def test_green_ledger_rejects_changed_registered_execution_contract(self) -> None:
        case = "MatrixSuite.ProductionParity/Ordinal_KVFP16_MTPOff"
        original = campaigns.CampaignCell(
            "V2_Parity_ProductionCampaign_CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "ALL"),
            gtest_cases=(case,),
            command=("v2_integration_parity_fake_single_device_matrix", f"--gtest_filter={case}"),
            environment=("MODE=old",),
            working_directory="/workspaces/llaminar",
        )
        entry = campaigns.IndividualGreenEvidence(
            gtest_case=case,
            campaign=original.name,
            git_revision="abc123",
            passed_wall_time_ns=1,
            elapsed_seconds=1.0,
            validated_artifact_file_count=len(campaigns.REQUIRED_CSV_HEADERS),
            artifact_directory="/artifacts/cell",
            execution_contract_sha256=(
                campaigns._exact_execution_contract_sha256(original, case)
            ),
        )
        ledger = campaigns.IndividualGreenLedger(
            schema_version=campaigns.INDIVIDUAL_GREEN_LEDGER_SCHEMA_VERSION,
            certification_eligible=False,
            entries=(entry,),
        )
        changed = campaigns.CampaignCell(
            original.name,
            original.group,
            gtest_cases=(case,),
            command=original.command,
            environment=("MODE=new",),
            working_directory=original.working_directory,
        )

        with self.assertRaisesRegex(ValueError, "execution contract changed"):
            campaigns.validate_green_ledger_selection((changed,), ledger)

    @mock.patch.object(campaigns, "_current_git_short_hash", return_value="abc123")
    def test_report_seed_imports_only_green_aggregate_and_declared_fail_fast_prefix(
        self,
        _revision: mock.Mock,
    ) -> None:
        prefix = "FailSuite.ProductionParity/Prefix_KVFP16_MTPOff"
        first_red = "FailSuite.ProductionParity/Red_KVFP16_MTPOff"
        unrun = "FailSuite.ProductionParity/Unrun_KVFP16_MTPOff"
        failed = campaigns.CampaignCell(
            "Failed_ProductionCampaign_CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "ALL"),
            gtest_cases=(prefix, first_red, unrun),
            command=(
                "v2_integration_parity_fake_single_device_matrix",
                "--gtest_filter=" + ":".join((prefix, first_red, unrun)),
            ),
            environment=("GTEST_FAIL_FAST=1",),
            working_directory="/workspaces/llaminar",
        )
        aggregate_green = "GreenSuite.ProductionParity/Only_KVFP16_MTPOff"
        green = campaigns.CampaignCell(
            "Green_ProductionCampaign_CUDA_ALL_PRECISIONS",
            campaigns.CampaignGroup("CUDA", "ALL"),
            gtest_cases=(aggregate_green,),
            command=(
                "v2_integration_parity_fake_single_device_matrix",
                f"--gtest_filter={aggregate_green}",
            ),
            environment=("GTEST_FAIL_FAST=1",),
            working_directory="/workspaces/llaminar",
        )

        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            artifacts = root / "artifacts"
            for case in (prefix, first_red, aggregate_green):
                self.write_canonical_artifacts(
                    artifacts / campaigns._gtest_artifact_directory_name(case)
                )
            report = root / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "schema_version": 12,
                        "artifact_root": str(artifacts),
                        "campaigns": [
                            {
                                "campaign": failed.name,
                                "gtest_cases": list(failed.gtest_cases),
                                "return_code": 8,
                                "outcome": "completed",
                                "artifact_contract_passed": False,
                                "validated_artifact_file_count": 0,
                                "artifact_errors": ["first red"],
                            },
                            {
                                "campaign": green.name,
                                "gtest_cases": list(green.gtest_cases),
                                "return_code": 0,
                                "outcome": "completed",
                                "artifact_contract_passed": True,
                                "validated_artifact_file_count": len(
                                    campaigns.REQUIRED_CSV_HEADERS
                                ),
                                "artifact_errors": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            ledger_path = root / "ledger.json"

            ledger = campaigns.seed_individual_green_ledger_from_campaign_report(
                report,
                (failed, green),
                ledger_path,
                declared_first_reds=((failed.name, first_red),),
            )

        self.assertEqual(
            ledger.green_cases,
            frozenset((prefix, aggregate_green)),
        )
        provenance = {
            entry.gtest_case: entry.provenance for entry in ledger.entries
        }
        self.assertEqual(
            provenance[prefix],
            "aggregate_gtest_fail_fast_prefix_before_declared_first_red",
        )
        self.assertEqual(
            provenance[aggregate_green],
            "aggregate_exit_zero_and_fresh_artifact_contract",
        )
        self.assertNotIn(first_red, ledger.green_cases)
        self.assertNotIn(unrun, ledger.green_cases)

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
            evidence = campaigns.ProcessTerminationEvidence()
            started = campaigns.time.monotonic()

            return_code = campaigns._run_process(
                [sys.executable, "-c", script],
                5.0,
                exact_cell_watch=campaigns.ExactCellTimeoutWatch(
                    timeout_seconds=0.2,
                    progress_files=((case, progress),),
                    not_before_wall_time_ns=campaigns.time.time_ns(),
                ),
                termination_evidence=evidence,
            )

        self.assertEqual(return_code, 124)
        self.assertLess(campaigns.time.monotonic() - started, 2.0)
        self.assertEqual(
            evidence.kind,
            campaigns.ProcessTerminationKind.EXACT_CELL_TIMEOUT,
        )
        self.assertEqual(evidence.exact_gtest_case, case)

    def test_exact_cell_watch_renews_deadline_only_on_next_cell(self) -> None:
        """Cell transitions renew virtual time, independent of CPU scheduling.

        A real sleeping child made this state-machine proof flaky under the
        full parallel Unit gate: scheduler delays became fake cell timeouts.
        The separate stuck-process test still exercises actual process-group
        termination. Here only the observed cell transitions advance the clock.
        """

        with tempfile.TemporaryDirectory() as raw_directory:
            first = Path(raw_directory) / "first" / "test_log.txt"
            second = Path(raw_directory) / "second" / "test_log.txt"
            first.parent.mkdir(parents=True)
            first.write_text("started", encoding="utf-8")
            clock = [0.0]
            process = mock.Mock(spec=subprocess.Popen)

            def wait(timeout: float) -> int:
                """Advance the synthetic child and publish its second cell."""

                clock[0] = min(0.6, clock[0] + timeout)
                if clock[0] >= 0.3 and not second.exists():
                    second.parent.mkdir(parents=True)
                    second.write_text("started", encoding="utf-8")
                if clock[0] >= 0.6:
                    return 0
                raise subprocess.TimeoutExpired("virtual parity child", timeout)

            process.wait.side_effect = wait
            process.poll.side_effect = lambda: 0 if clock[0] >= 0.6 else None
            evidence = campaigns.ProcessTerminationEvidence()
            with (
                mock.patch.object(campaigns.subprocess, "Popen", return_value=process),
                mock.patch.object(campaigns.time, "monotonic", side_effect=lambda: clock[0]),
                mock.patch.object(campaigns, "_terminate_process_group") as terminate,
            ):
                return_code = campaigns._run_process(
                    ["virtual-parity-child"],
                    0.2,
                    exact_cell_watch=campaigns.ExactCellTimeoutWatch(
                        timeout_seconds=0.5,
                        progress_files=(
                            ("MatrixSuite.ProductionParity/First", first),
                            ("MatrixSuite.ProductionParity/Second", second),
                        ),
                        not_before_wall_time_ns=0,
                    ),
                    termination_evidence=evidence,
                )
                terminate.assert_not_called()

        self.assertEqual(return_code, 0)
        self.assertGreater(clock[0], 0.5)
        self.assertEqual(
            evidence.kind,
            campaigns.ProcessTerminationKind.NONE,
        )

    def test_campaign_cancellation_terminates_the_active_process_group(
        self,
    ) -> None:
        """A sibling red must stop both CTest and its MPI-style children."""

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            child_ready = directory / "child-ready"
            child_terminated = directory / "child-terminated"
            parent_ready = directory / "parent-ready"
            child_script = f"""
import signal
import time
from pathlib import Path

ready = Path({str(child_ready)!r})
terminated = Path({str(child_terminated)!r})

def stop(_signum, _frame):
    terminated.write_text("terminated")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
ready.write_text("ready")
while True:
    time.sleep(1)
"""
            parent_script = f"""
import subprocess
import sys
import time
from pathlib import Path

child_ready = Path({str(child_ready)!r})
subprocess.Popen([sys.executable, "-c", {child_script!r}])
deadline = time.monotonic() + 5.0
while not child_ready.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
Path({str(parent_ready)!r}).write_text("ready")
time.sleep(30)
"""
            cancellation = campaigns.CampaignCancellation()
            evidence = campaigns.ProcessTerminationEvidence()

            def cancel_active_group() -> None:
                deadline = campaigns.time.monotonic() + 5.0
                while (
                    not parent_ready.exists()
                    and campaigns.time.monotonic() < deadline
                ):
                    campaigns.time.sleep(0.01)
                cancellation.request_after_failure("first-red-campaign")

            publisher = campaigns.threading.Thread(target=cancel_active_group)
            publisher.start()
            started = campaigns.time.monotonic()
            return_code = campaigns._run_process(
                [sys.executable, "-c", parent_script],
                10.0,
                cancellation=cancellation,
                termination_evidence=evidence,
            )
            publisher.join(timeout=1.0)

            terminated_deadline = campaigns.time.monotonic() + 1.0
            while (
                not child_terminated.exists()
                and campaigns.time.monotonic() < terminated_deadline
            ):
                campaigns.time.sleep(0.01)

            self.assertTrue(parent_ready.exists())
            self.assertTrue(child_ready.exists())
            self.assertTrue(child_terminated.exists())

        self.assertEqual(return_code, 130)
        self.assertLess(campaigns.time.monotonic() - started, 2.0)
        self.assertEqual(
            evidence.kind,
            campaigns.ProcessTerminationKind.CAMPAIGN_CANCELLED,
        )
        self.assertEqual(evidence.cancelling_campaign, "first-red-campaign")

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

        environment = {"LLAMINAR_TEST_ENVIRONMENT_PASSTHROUGH": "proof"}
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
            run_process.call_args.kwargs["termination_evidence"],
            campaigns.ProcessTerminationEvidence,
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
            self.assertIn("acceptance_witness_executed", columns)
            self.assertIn(
                "acceptance_witness_accepted_token_delta", columns
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
            evidence = kwargs["termination_evidence"]
            assert isinstance(evidence, campaigns.ProcessTerminationEvidence)
            evidence.kind = campaigns.ProcessTerminationKind.EXACT_CELL_TIMEOUT
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

    def test_scheduler_prioritizes_unseen_then_partial_then_complete(
        self,
    ) -> None:
        complete_cases = (
            "CompleteSuite.ProductionParity/Ordinal",
            "CompleteSuite.ProductionParity/Random",
        )
        partial_cases = (
            "PartialSuite.ProductionParity/Ordinal",
            "PartialSuite.ProductionParity/Random",
        )
        unseen_cases = (
            "UnseenSuite.ProductionParity/Ordinal",
            "UnseenSuite.ProductionParity/Random",
        )
        complete = campaigns.CampaignCell(
            "a_ProductionCampaign_CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "ALL"),
            gtest_cases=complete_cases,
        )
        partial = campaigns.CampaignCell(
            "b_ProductionCampaign_CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "ALL"),
            gtest_cases=partial_cases,
        )
        unseen = campaigns.CampaignCell(
            "c_ProductionCampaign_CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "ALL"),
            gtest_cases=unseen_cases,
        )
        selected = [complete, partial, unseen]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for gtest_case in (*complete_cases, partial_cases[0]):
                self.write_canonical_artifacts(
                    root
                    / campaigns._gtest_artifact_directory_name(gtest_case)
                )

            evidence = campaigns.inspect_prior_artifact_evidence(
                selected,
                (root,),
            )
            ordered = campaigns.scheduling_order(
                selected,
                prior_evidence=evidence,
            )

        self.assertEqual(ordered, [unseen, partial, complete])
        self.assertEqual(
            campaigns.scheduling_order(selected),
            [complete, partial, unseen],
        )
        self.assertEqual(len(evidence.observed_gtest_cases), 3)
        self.assertEqual(
            evidence.coverage_for(unseen).kind,
            campaigns.PriorEvidenceCoverageKind.UNSEEN,
        )
        self.assertEqual(
            evidence.coverage_for(partial).kind,
            campaigns.PriorEvidenceCoverageKind.PARTIAL,
        )
        self.assertEqual(
            evidence.coverage_for(complete).kind,
            campaigns.PriorEvidenceCoverageKind.COMPLETE,
        )
        self.assertEqual(
            campaigns.prior_evidence_campaign_counts(selected, evidence),
            {
                campaigns.PriorEvidenceCoverageKind.UNSEEN: 1,
                campaigns.PriorEvidenceCoverageKind.PARTIAL: 1,
                campaigns.PriorEvidenceCoverageKind.COMPLETE: 1,
            },
        )
        self.assertEqual(
            {
                (cell.name, cell.gtest_cases)
                for cell in ordered
            },
            {
                (cell.name, cell.gtest_cases)
                for cell in selected
            },
        )

    def test_prior_artifact_roots_are_repeatable_explicit_directories(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            arguments = campaigns.parse_args(
                [
                    "--prioritize-unseen-from-artifact-root",
                    str(root),
                    "--prioritize-unseen-from-artifact-root",
                    str(root),
                ]
            )

        self.assertEqual(arguments.prior_artifact_roots, (root,))

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
            _completion_timeout_seconds: float | None,
            **_kwargs: object,
        ) -> campaigns.CampaignResult:
            self.assertIsNone(_completion_timeout_seconds)
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
            global_started_at=campaigns.time.monotonic() - 31.0,
        )

        self.assertEqual(run_campaign.call_count, 2)
        self.assertEqual([result.campaign for result in results], [c.name for c in cells])
        self.assertTrue(all(result.return_code == 0 for result in results))
        self.assertTrue(all(not result.target_met for result in results))

    @mock.patch.object(campaigns, "run_campaign")
    def test_matrix_first_failure_cancels_sibling_and_stops_admission(
        self,
        run_campaign: mock.Mock,
    ) -> None:
        """Only the initially admitted disjoint set may run after one red."""

        first = campaigns.CampaignCell(
            "a_ProductionCampaign_CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "ALL"),
        )
        pending = campaigns.CampaignCell(
            "b_ProductionCampaign_CPU_ALL_PRECISIONS",
            campaigns.CampaignGroup("CPU", "ALL"),
        )
        sibling = campaigns.CampaignCell(
            "c_ProductionCampaign_CUDA_ALL_PRECISIONS",
            campaigns.CampaignGroup("CUDA", "ALL"),
        )
        sibling_started = campaigns.threading.Event()
        called: list[str] = []

        def result_for(
            _build_dir: Path,
            cell: campaigns.CampaignCell,
            _completion_timeout_seconds: float,
            **kwargs: object,
        ) -> campaigns.CampaignResult:
            called.append(cell.name)
            cancellation = kwargs["cancellation"]
            assert isinstance(cancellation, campaigns.CampaignCancellation)
            if cell == first:
                self.assertTrue(sibling_started.wait(timeout=1.0))
                return_code = 17
                outcome = "completed"
                cancelled_by = ""
            elif cell == sibling:
                sibling_started.set()
                deadline = campaigns.time.monotonic() + 1.0
                while (
                    not cancellation.requested
                    and campaigns.time.monotonic() < deadline
                ):
                    campaigns.time.sleep(0.01)
                self.assertTrue(cancellation.requested)
                return_code = 130
                outcome = "cancelled_after_campaign_failure"
                cancelled_by = cancellation.failing_campaign
            else:  # pragma: no cover - admission is the invariant under test
                self.fail("pending campaign was admitted after the first red")

            return campaigns.CampaignResult(
                campaign=cell.name,
                test_type=cell.test_type,
                backends=cell.group.backends,
                precision_set=cell.group.kv_precision,
                precision_types=cell.precision_types,
                gtest_cases=cell.gtest_cases,
                elapsed_seconds=0.1,
                target_seconds=30.0,
                return_code=return_code,
                target_met=True,
                cancelled_by_campaign=cancelled_by,
                outcome=outcome,
                artifact_contract_passed=(return_code == 0),
            )

        run_campaign.side_effect = result_for
        results, _ = campaigns.run_campaign_matrix(
            Path("build"),
            [first, pending, sibling],
            target_seconds=30.0,
        )

        by_campaign = {result.campaign: result for result in results}
        self.assertCountEqual(called, [first.name, sibling.name])
        self.assertEqual(by_campaign[first.name].return_code, 17)
        self.assertEqual(
            by_campaign[pending.name].outcome,
            "not_started_after_campaign_failure",
        )
        self.assertEqual(
            by_campaign[sibling.name].outcome,
            "cancelled_after_campaign_failure",
        )
        self.assertEqual(
            by_campaign[pending.name].cancelled_by_campaign,
            first.name,
        )
        self.assertEqual(
            by_campaign[sibling.name].cancelled_by_campaign,
            first.name,
        )

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
            executable = directory / "v2_integration_parity_fake_single_device_matrix"
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

            # The same production inventory must not be published under a
            # cell-shaped or legacy binary identity. Exercise real discovery,
            # not a source scan of the target declarations.
            stale_executable = directory / "v2_integration_parity_fake_fp16"
            executable.rename(stale_executable)
            stale_command = list(completed.args)
            stale_command[1] = f"-DTEST_EXECUTABLE={stale_executable}"
            rejected = subprocess.run(
                stale_command, check=False, capture_output=True, text=True
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("configuration axes belong in generated cell names",
                          " ".join(rejected.stderr.split()))

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
            executable = directory / "v2_integration_parity_fake_single_device_matrix"
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
            executable = directory / "v2_integration_parity_fake_single_device_matrix"
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
            executable = directory / "v2_integration_parity_fake_single_device_matrix"
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
            executable = directory / "v2_integration_parity_fake_single_device_matrix"
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
    def test_ramdisk_staging_records_exact_bytes_and_publishes_read_only_models(
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
                evidence[0].source_identity,
                campaigns._source_identity(source.stat()),
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
            ) as stage_one:
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
            self.assertEqual(first[0].cache_status, "copied")
            self.assertEqual(second[0].cache_status, "reused")
            self.assertEqual(second[0].source_identity, first[0].source_identity)
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
            self.assertNotIn("sha256", manifest["models"][source.name])

    @mock.patch.object(campaigns, "_filesystem_type", return_value="tmpfs")
    def test_schema_one_cache_migrates_without_rereading_model_payload(
        self, _: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = directory / "model.gguf"
            source.write_bytes(b"persistent-real-weights")
            staging = directory / "cache" / "models"
            staging.mkdir(parents=True)
            destination = staging / source.name
            destination.write_bytes(source.read_bytes())
            destination.chmod(0o400)
            manifest_path = staging.parent / "model-cache-manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": {
                            source.name: {
                                "source_path": str(source),
                                "source_identity": list(
                                    campaigns._source_identity(source.stat())
                                ),
                                "cached_identity": list(
                                    campaigns._source_identity(destination.stat())
                                ),
                                "size_bytes": source.stat().st_size,
                                "sha256": "legacy-content-digest",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            cell = campaigns.CampaignCell(
                "cell_ProductionCampaign_CPU_ALL_PRECISIONS",
                campaigns.CampaignGroup("CPU", "ALL"),
                model_files=(str(source),),
            )

            with mock.patch.object(
                campaigns, "_stage_one_model", wraps=campaigns._stage_one_model
            ) as stage_one:
                evidence, _ = campaigns.stage_models_in_ramdisk(
                    [cell],
                    staging,
                    campaigns.time.monotonic() + 30.0,
                    persistent=True,
                )

            self.assertEqual(stage_one.call_count, 0)
            self.assertEqual(evidence[0].cache_status, "reused")
            migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                migrated["schema_version"],
                campaigns.PERSISTENT_MODEL_CACHE_SCHEMA_VERSION,
            )
            self.assertNotIn("sha256", migrated["models"][source.name])

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
    def test_persistent_workspace_seals_all_cache_owned_paths(
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
                    published_model = workspace.models / "model.gguf"
                    published_model.write_bytes(b"published-model")

                    workspace.protect_published_models()

                    self.assertEqual(workspace.root.stat().st_mode & 0o777, 0o500)
                    self.assertEqual(workspace.models.stat().st_mode & 0o777, 0o500)
                    if os.geteuid() != 0:
                        with self.assertRaises(PermissionError):
                            published_model.unlink()

                self.assertEqual(root.stat().st_mode & 0o777, 0o500)
                self.assertEqual((root / "models").stat().st_mode & 0o777, 0o500)
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

    def test_individual_cell_admission_limit_is_explicit_and_positive(
        self,
    ) -> None:
        args = campaigns.parse_args(
            [
                "--run-unseen-cells-individually",
                "--green-ledger",
                "green.json",
                "--max-unseen-cells",
                "1",
            ]
        )

        self.assertEqual(args.max_unseen_cells, 1)
        with self.assertRaises(SystemExit):
            campaigns.parse_args(["--max-unseen-cells", "1"])
        with self.assertRaises(SystemExit):
            campaigns.parse_args(
                [
                    "--run-unseen-cells-individually",
                    "--green-ledger",
                    "green.json",
                    "--max-unseen-cells",
                    "0",
                ]
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
