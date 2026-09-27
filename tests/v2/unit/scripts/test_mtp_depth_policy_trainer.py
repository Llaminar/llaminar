#!/usr/bin/env python3
"""Regression tests for the generated MTP depth-policy trainer."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
TRAINER = REPO_ROOT / "scripts" / "train_mtp_depth_policy.py"

# Import the same implementation used by the CLI; small synthetic predicates
# exercise the evidence boundary without a model, GPU, or timing noise.
SPEC = importlib.util.spec_from_file_location("mtp_policy_trainer_test_subject", TRAINER)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MTPDepthPolicyTrainerTest(unittest.TestCase):
    def test_near_best_startup_is_stable_but_real_regressions_keep_their_label(self) -> None:
        """Economic ties do not erase a genuine held-out throughput deficit."""
        startup = MODULE.LearnedStartup("cpu", "moe", "stochastic", 3, 4, 0.99)
        for rate, expected in ((95.0, 3), (94.99, 4)):
            rows = [MODULE.FixedDepthRow(
                ("held",), "cpu", "moe", "stochastic", depth,
                rate if depth == 3 else 100 if depth == 4 else 10,
                0.8, True,
            ) for depth in (1, 2, 3, 4)]
            examples = MODULE.label_examples(rows, [startup], 0.05)
            self.assertEqual({row.target_depth for row in examples}, {expected})
            self.assertTrue(all(row.held_out for row in examples))
            self.assertEqual({row.target_depth for row in MODULE.label_examples(rows)}, {4})

    def test_startup_uses_training_economics_not_deepest_local_winner(self) -> None:
        """A small d4 win must not outweigh a large d3 win on another request."""
        rows = []
        for case, speeds in (("repeat", (20, 30, 44, 38)),
                             ("prose", (20, 28, 30, 29)),
                             ("code", (20, 28, 30, 31)),
                             ("long", (20, 28, 30, 31))):
            rows.extend(MODULE.FixedDepthRow(
                group_key=(case,), backend="cpu", model_class="moe",
                mode="stochastic", depth=depth, decode_tps=speed,
                acceptance_rate=0.8, held_out=False,
            ) for depth, speed in enumerate(speeds, 1))
        labels = MODULE.label_examples(rows)
        self.assertEqual({row.target_depth for row in labels}, {3, 4})
        startup, = MODULE.learn_startups(rows, 0, 0)
        self.assertEqual(startup.depth, 3)
        self.assertEqual(startup.train_groups, 4)

        # A held-out domain is evidence, not an input to startup selection.
        rows.extend(MODULE.FixedDepthRow(
            group_key=("holdout",), backend="cpu", model_class="moe",
            mode="stochastic", depth=depth, decode_tps=1000 if depth == 4 else 1,
            acceptance_rate=0.9, held_out=True,
        ) for depth in (1, 2, 3, 4))
        self.assertEqual(MODULE.learn_startups(rows, 0, 0), [startup])

    def test_startup_normalizes_requests_and_requires_common_measured_depths(self) -> None:
        """Prompt speed units and one missing deep lane cannot bias admission."""
        rows = []
        for case, scale in (("fast", 1000), ("slow", 1)):
            rows.extend(MODULE.FixedDepthRow(
                group_key=(case,), backend="cpu", model_class="moe",
                mode="stochastic", depth=depth, decode_tps=scale * (20 + depth),
                acceptance_rate=0.8, held_out=False,
            ) for depth in (1, 2, 3))
        rows.append(MODULE.FixedDepthRow(
            ("fast",), "cpu", "moe", "stochastic", 15, 1e9, 0.9, False))
        startup, = MODULE.learn_startups(rows, 0, 0)
        self.assertEqual(startup.depth, 3)
        with self.assertRaisesRegex(ValueError, "training"):
            MODULE.learn_startups([
                MODULE.FixedDepthRow(("held",), "cpu", "moe", "stochastic",
                                     depth, 100, 0.9, True)
                for depth in (1, 2, 3)
            ], 0, 0)

    def test_every_supported_depth_can_be_the_learned_startup(self) -> None:
        """Admission may start at any measured winner, including depth fifteen."""
        for winner in range(1, 16):
            rows = [MODULE.FixedDepthRow(
                group_key=("request",), backend="cpu", model_class="moe",
                mode="stochastic", depth=depth, decode_tps=200 if depth == winner else 100,
                acceptance_rate=0.9, held_out=False,
            ) for depth in range(1, 16)]
            startup, = MODULE.learn_startups(rows, 0, 0)
            self.assertEqual(startup.depth, winner)

    def test_every_supported_measured_depth_can_win(self) -> None:
        """The learner must not discard d4..d15 after the sweep measured them."""
        for winner in range(1, 16):
            rows = [MODULE.FixedDepthRow(
                group_key=("same-request",), backend="rocm", model_class="moe",
                mode="stochastic", depth=depth, decode_tps=200 if depth == winner else 100,
                acceptance_rate=0.9,
            ) for depth in range(1, 16)]
            examples = MODULE.label_examples(rows)
            self.assertEqual(len(examples), 15)
            self.assertEqual({example.target_depth for example in examples}, {winner})

    def test_explicit_holdout_never_changes_the_training_winner(self) -> None:
        """Independent prompt lanes must not leak into installed decisions."""
        rows = []
        for held_out, winner in ((False, 3), (True, 1)):
            rows.extend(MODULE.FixedDepthRow(
                group_key=(str(held_out),), backend="rocm", model_class="moe",
                mode="stochastic", depth=depth, decode_tps=200 if depth == winner else 100,
                acceptance_rate=0.9, held_out=held_out,
            ) for depth in (1, 2, 3))
        rules = MODULE.learn_rules(MODULE.label_examples(rows), 0, 0, 0.75, 0.05)
        self.assertEqual({rule.target_depth for rule in rules}, {3})
        self.assertTrue(all(rule.holdout_total == 1 for rule in rules))
        self.assertTrue(all(rule.holdout_correct == 0 for rule in rules))
        with self.assertRaisesRegex(ValueError, "must never be used for fitting"):
            MODULE.learn_rules(MODULE.label_examples(rows[3:]), 0, 0, 0.75, 0.05)

    def test_automatic_holdout_cannot_be_relabelled_as_training(self) -> None:
        rows = [MODULE.FixedDepthRow(
            group_key=("only-group",), backend="rocm", model_class="moe",
            mode="stochastic", depth=depth, decode_tps=100 + depth,
            acceptance_rate=0.9,
        ) for depth in (1, 2, 3)]
        with self.assertRaisesRegex(ValueError, "must never be used for fitting"):
            MODULE.learn_rules(MODULE.label_examples(rows), 1, 0, 0.75, 0.05)

    def test_hold_margin_is_included_in_the_reported_accuracy(self) -> None:
        examples = [
            MODULE.LabeledExample("rocm", "moe", "stochastic", 3, 3, 0.80,
                                  "hold", ("train",), False),
            MODULE.LabeledExample("rocm", "moe", "stochastic", 3, 1, 0.77,
                                  "demote", ("holdout",), True),
        ]
        rule, = MODULE.learn_rules(examples, 0, 0, 0.75, 0.05)
        self.assertAlmostEqual(rule.min_acceptance, 0.75)
        self.assertEqual((rule.holdout_correct, rule.holdout_total), (0, 1))

    def test_relearning_preserves_only_unmeasured_domains(self) -> None:
        """Narrow retraining replaces a whole domain, never all installed rows."""
        def rule(backend, mode):
            return MODULE.LearnedRule(backend, "moe", mode, 1, 3,
                0.7, 1.0, 1.0, 0.0, 2, f"trained_{backend}_{mode}", 1, 1, 1, 1)
        original = [rule("cuda", "stochastic"), rule("rocm", "greedy"),
                    rule("rocm", "stochastic")]
        fresh = [rule("rocm", "stochastic")]
        startups = [MODULE.LearnedStartup(row.backend, row.model_class, row.mode, 3, 1, 1.0)
                    for row in original]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.inc"
            MODULE.write_include(path, original, 3, startups)
            kept = MODULE.preserve_untrained_domains(path, fresh)
            self.assertEqual([row.label for row in kept], [row.label for row in original[:2]])
            self.assertTrue(all(row.train_total == 0 for row in kept))
            kept_startups = MODULE.preserve_untrained_startups(path, fresh)
            self.assertEqual([(row.backend, row.mode, row.depth) for row in kept_startups],
                             [(row.backend, row.mode, row.depth) for row in startups[:2]])
            self.assertTrue(all(row.train_groups == 0 for row in kept_startups))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                MODULE.write_include(path, original, 3, [*startups, startups[0]])
            with self.assertRaisesRegex(ValueError, "missing measured startup"):
                MODULE.write_include(path, original, 3, startups[:1])
            # Corrupted installed metadata must fail at ingestion as well.
            contents = path.read_text()
            marker = "kMTPGeneratedDepthPolicyStartups[] = {\n"
            first = contents.split(marker, 1)[1].splitlines()[0]
            path.write_text(contents.replace(marker, marker + first + "\n"))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                MODULE.preserve_untrained_startups(path, fresh)
            path.write_text('{MTPVerifyMode::bad},\n')
            with self.assertRaisesRegex(ValueError, "malformed"):
                MODULE.preserve_untrained_domains(path, fresh)
            with self.assertRaisesRegex(ValueError, "malformed"):
                MODULE.preserve_untrained_startups(path, fresh)

    def test_duplicate_depth_or_invalid_measurement_is_rejected(self) -> None:
        row = MODULE.FixedDepthRow(("request",), "rocm", "moe", "stochastic", 1, 100, 0.9)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.label_examples([row, row])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.tsv"
            for throughput, acceptance in (("nan", "90"), ("0", "90"), ("100", "101")):
                path.write_text("device\tmodel\tmode\tvariant\tdecode_tps\tacceptance_pct\n"
                    f"rocm:0\tmoe\tstochastic\tfixed_d1\t{throughput}\t{acceptance}\n")
                with self.assertRaisesRegex(ValueError, "invalid fixed-depth"):
                    MODULE.load_fixed_rows([path])

    def test_mode_aliases_replace_the_same_policy_domain(self) -> None:
        """An alternate spelling must not leave an older first-match rule live."""
        for mode in ("stochastic", "speculative-sampling", "sampling"):
            self.assertEqual(MODULE._canonical_mode(mode), "stochastic")
            self.assertEqual(MODULE.cpp_verify_mode(mode), "MTPVerifyMode::SpeculativeSampling")
        with self.assertRaisesRegex(ValueError, "unknown MTP"):
            MODULE.cpp_verify_mode("typo")

    def write_summary(self, path: Path) -> None:
        path.write_text(
            textwrap.dedent(
                """\
                device\tmodel\tmode\tvariant\tsuccess\tdecode_tps\tacceptance_pct
                cuda:0\tdense\tgreedy\tfixed_d1\ttrue\t10.0\t90.0
                cuda:0\tdense\tgreedy\tfixed_d2\ttrue\t14.0\t82.0
                cuda:0\tdense\tgreedy\tfixed_d3\ttrue\t18.0\t75.0
                rocm:0\tdense\tgreedy\tfixed_d1\ttrue\t10.0\t80.0
                rocm:0\tdense\tgreedy\tfixed_d2\ttrue\t13.0\t70.0
                rocm:0\tdense\tgreedy\tfixed_d3\ttrue\t12.0\t40.0
                cpu:0\tdense\tstochastic\tfixed_d1\ttrue\t10.0\t55.0
                cpu:0\tdense\tstochastic\tfixed_d2\ttrue\t9.0\t30.0
                cpu:0\tdense\tstochastic\tfixed_d3\ttrue\t8.0\t20.0
                cuda:0\tdense\tstochastic\tfixed_d1\ttrue\t10.0\t65.0
                cuda:0\tdense\tstochastic\tfixed_d2\ttrue\t11.0\t76.0
                cuda:0\tdense\tstochastic\tfixed_d3\ttrue\t12.0\t81.0
                rocm:0\tdense\tstochastic\tfixed_d1\ttrue\t10.0\t84.0
                rocm:0\tdense\tstochastic\tfixed_d2\ttrue\t9.0\t67.0
                rocm:0\tdense\tstochastic\tfixed_d3\ttrue\t8.0\t65.0
                """
            ),
            encoding="utf-8",
        )

    def test_trainer_generates_policy_include_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary = tmp / "summary.tsv"
            output = tmp / "MTPDepthPolicyGenerated.inc"
            report = tmp / "report.txt"
            self.write_summary(summary)

            result = subprocess.run(
                [
                    "python3",
                    str(TRAINER),
                    "--input",
                    str(summary),
                    "--output",
                    str(output),
                    "--summary",
                    str(report),
                    "--holdout-modulus",
                    "0",
                    "--startup-tie-tolerance",
                    "0",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            include_text = output.read_text(encoding="utf-8")
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("kMTPGeneratedDepthPolicyRules", include_text)
            self.assertIn("kMTPGeneratedDepthPolicyStartups", include_text)
            self.assertIn("startups (training requests only):", report_text)
            self.assertIn("MTPVerifyMode::Greedy", include_text)
            self.assertIn("MTPVerifyMode::SpeculativeSampling", include_text)
            self.assertIn("MTPDepthPolicyBackend::CUDA", include_text)
            self.assertIn("MTPDepthPolicyBackend::ROCm", include_text)
            self.assertIn("MTPDepthPolicyModelClass::Dense", include_text)
            self.assertIn("trained_cuda_dense_greedy_promote_d1_to_d3", include_text)
            self.assertIn("trained_cuda_dense_greedy_hold_d3_to_d3", include_text)
            self.assertIn("trained_rocm_dense_greedy_demote_d3_to_d2", include_text)
            self.assertIn("trained_cuda_dense_stochastic_promote_d1_to_d3", include_text)
            self.assertIn("trained_cuda_dense_stochastic_promote_d2_to_d3", include_text)
            self.assertIn("trained_rocm_dense_stochastic_demote_d2_to_d1", include_text)
            self.assertIn("trained_rocm_dense_stochastic_demote_d3_to_d1", include_text)
            self.assertIn(", +2, \"trained_cuda_dense_greedy_promote_d1_to_d3\"", include_text)
            self.assertIn("examples=15", report_text)

    def test_holdout_accuracy_gate_can_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary = tmp / "summary.tsv"
            output = tmp / "MTPDepthPolicyGenerated.inc"
            report = tmp / "report.txt"
            self.write_summary(summary)

            result = subprocess.run(
                [
                    "python3",
                    str(TRAINER),
                    "--input",
                    str(summary),
                    "--output",
                    str(output),
                    "--summary",
                    str(report),
                    "--holdout-modulus",
                    "1",
                    "--holdout-bucket",
                    "0",
                    "--min-holdout-accuracy",
                    "1.1",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("holdout", result.stderr)

    def test_cross_backend_stochastic_policy_keeps_backend_specific_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary = tmp / "summary.tsv"
            output = tmp / "MTPDepthPolicyGenerated.inc"
            report = tmp / "report.txt"
            summary.write_text(
                textwrap.dedent(
                    """\
                    device\tmodel\tmode\tcase\tvariant\tsuccess\tdecode_tps\tacceptance_pct
                    cuda:0\tdense\tstochastic\tdefault\tfixed_d1\ttrue\t10.0\t65.0
                    cuda:0\tdense\tstochastic\tdefault\tfixed_d2\ttrue\t11.0\t76.0
                    cuda:0\tdense\tstochastic\tdefault\tfixed_d3\ttrue\t12.0\t81.0
                    rocm:0\tdense\tstochastic\tdefault\tfixed_d1\ttrue\t10.0\t84.0
                    rocm:0\tdense\tstochastic\tdefault\tfixed_d2\ttrue\t9.0\t67.0
                    rocm:0\tdense\tstochastic\tdefault\tfixed_d3\ttrue\t8.0\t65.0
                    """
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "python3",
                    str(TRAINER),
                    "--input",
                    str(summary),
                    "--output",
                    str(output),
                    "--summary",
                    str(report),
                    "--holdout-modulus",
                    "0",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            include_text = output.read_text(encoding="utf-8")
            self.assertIn("trained_cuda_dense_stochastic_promote_d1_to_d3", include_text)
            self.assertIn("trained_cuda_dense_stochastic_promote_d2_to_d3", include_text)
            self.assertIn("trained_rocm_dense_stochastic_demote_d2_to_d1", include_text)
            self.assertIn("trained_rocm_dense_stochastic_demote_d3_to_d1", include_text)
            self.assertNotIn("trained_rocm_dense_stochastic_promote_d1", include_text)
            self.assertNotIn("trained_cuda_dense_stochastic_demote_d2", include_text)

    def test_model_class_policy_keeps_dense_and_moe_rows_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary = tmp / "summary.tsv"
            output = tmp / "MTPDepthPolicyGenerated.inc"
            report = tmp / "report.txt"
            summary.write_text(
                textwrap.dedent(
                    """\
                    device\tmodel\tmode\tcase\tvariant\tsuccess\tdecode_tps\tacceptance_pct
                    rocm:0\tdense\tgreedy\tdefault\tfixed_d1\ttrue\t10.0\t90.0
                    rocm:0\tdense\tgreedy\tdefault\tfixed_d2\ttrue\t12.0\t92.0
                    rocm:0\tdense\tgreedy\tdefault\tfixed_d3\ttrue\t14.0\t98.0
                    rocm:0\tmoe\tgreedy\tdefault\tfixed_d1\ttrue\t10.0\t82.0
                    rocm:0\tmoe\tgreedy\tdefault\tfixed_d2\ttrue\t12.0\t76.0
                    rocm:0\tmoe\tgreedy\tdefault\tfixed_d3\ttrue\t16.0\t81.0
                    """
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "python3",
                    str(TRAINER),
                    "--input",
                    str(summary),
                    "--output",
                    str(output),
                    "--summary",
                    str(report),
                    "--holdout-modulus",
                    "0",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            include_text = output.read_text(encoding="utf-8")
            self.assertIn("trained_rocm_dense_greedy_promote_d1_to_d3", include_text)
            self.assertIn("trained_rocm_dense_greedy_promote_d2_to_d3", include_text)
            self.assertIn("trained_rocm_dense_greedy_hold_d3_to_d3", include_text)
            self.assertIn("trained_rocm_moe_greedy_promote_d1_to_d3", include_text)
            self.assertIn("trained_rocm_moe_greedy_promote_d2_to_d3", include_text)
            self.assertIn("trained_rocm_moe_greedy_hold_d3_to_d3", include_text)
            self.assertIn("MTPDepthPolicyModelClass::Dense, 1, 0.900000", include_text)
            self.assertIn("MTPDepthPolicyModelClass::MoE, 1, 0.820000", include_text)
            self.assertIn("MTPDepthPolicyModelClass::Dense, 2, 0.920000", include_text)
            self.assertIn("MTPDepthPolicyModelClass::MoE, 2, 0.760000", include_text)
            self.assertIn("MTPDepthPolicyModelClass::MoE, 3, 0.760000", include_text)
            self.assertIn(", +2, \"trained_rocm_moe_greedy_promote_d1_to_d3\"", include_text)

    def test_multiple_summaries_do_not_overwrite_same_lane_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            short_summary = tmp / "short.tsv"
            long_summary = tmp / "long.tsv"
            output = tmp / "MTPDepthPolicyGenerated.inc"
            report = tmp / "report.txt"
            header = (
                "device\tmodel\tmode\tvariant\tsuccess\tdecode_tps\t"
                "acceptance_pct\tdecode_tokens\trequest_batch\n"
            )
            short_summary.write_text(
                header
                + textwrap.dedent(
                    """\
                    rocm:0\tmoe\tstochastic\tfixed_d1\ttrue\t53.0\t75.0\t16\t1
                    rocm:0\tmoe\tstochastic\tfixed_d2\ttrue\t51.0\t70.0\t16\t1
                    rocm:0\tmoe\tstochastic\tfixed_d3\ttrue\t52.0\t75.0\t16\t1
                    """
                ),
                encoding="utf-8",
            )
            long_summary.write_text(
                header
                + textwrap.dedent(
                    """\
                    rocm:0\tmoe\tstochastic\tfixed_d1\ttrue\t60.0\t46.0\t64\t1
                    rocm:0\tmoe\tstochastic\tfixed_d2\ttrue\t77.0\t83.0\t64\t1
                    rocm:0\tmoe\tstochastic\tfixed_d3\ttrue\t74.0\t79.0\t64\t1
                    """
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "python3",
                    str(TRAINER),
                    "--input",
                    str(short_summary),
                    "--input",
                    str(long_summary),
                    "--startup-tie-tolerance",
                    "0",
                    "--output",
                    str(output),
                    "--summary",
                    str(report),
                    "--holdout-modulus",
                    "0",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            include_text = output.read_text(encoding="utf-8")
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("examples=6", report_text)
            self.assertIn("trained_rocm_moe_stochastic_hold_d1_to_d1", include_text)
            self.assertIn("trained_rocm_moe_stochastic_promote_d1_to_d2", include_text)
            self.assertIn(
                "MTPDepthPolicyModelClass::MoE, 1, 0.460000, 0.740000",
                include_text,
            )

    def test_holdout_split_is_invariant_to_corpus_relocation(self) -> None:
        """Moving identical evidence must not change generated policy rules."""

        with tempfile.TemporaryDirectory() as first_dir:
            with tempfile.TemporaryDirectory() as second_dir:
                generated: list[tuple[str, str]] = []
                for directory in (first_dir, second_dir):
                    root = Path(directory)
                    summary = root / "summary.tsv"
                    output = root / "MTPDepthPolicyGenerated.inc"
                    report = root / "report.txt"
                    self.write_summary(summary)

                    result = subprocess.run(
                        [
                            "python3",
                            str(TRAINER),
                            "--input",
                            str(summary),
                            "--output",
                            str(output),
                            "--summary",
                            str(report),
                            "--holdout-modulus",
                            "4",
                            "--holdout-bucket",
                            "0",
                        ],
                        cwd=REPO_ROOT,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )

                    self.assertEqual(result.returncode, 0, result.stderr)
                    generated.append(
                        (
                            output.read_text(encoding="utf-8"),
                            report.read_text(encoding="utf-8"),
                        )
                    )

                self.assertEqual(generated[0], generated[1])


if __name__ == "__main__":
    unittest.main()
