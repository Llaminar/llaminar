#!/usr/bin/env python3
"""Regression tests for the generated MTP depth-policy trainer."""

from __future__ import annotations

import importlib.util
import csv
import io
from fractions import Fraction
from dataclasses import replace
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


def fixed_fixture(*args, **kwargs):
    """Author one full-prefix observation for positive synthetic counter fixtures."""
    rate = kwargs.get("acceptance_rate", args[6] if len(args) > 6 else 0)
    kwargs.setdefault("full_accepts_lower_bound", 1 if rate > 0 else 0)
    return MODULE.FixedDepthRow(*args, **kwargs)


def labeled_fixture(*args, **kwargs):
    """Keep synthetic promotion evidence explicit, independent of its rate label."""
    rate = kwargs.get("acceptance_rate", args[5] if len(args) > 5 else 0)
    kwargs.setdefault("full_accepts_lower_bound", 1 if rate > 0 else 0)
    return MODULE.LabeledExample(*args, **kwargs)


def write_summary_fixture(path, text, **kwargs):
    """Author exact eligible integer histograms for synthetic TSV rate examples.

    These are fabricated test observations, not evidence inferred for real
    legacy inputs. Existing explicit controller fixtures remain byte-for-byte.
    """
    table = csv.DictReader(io.StringIO(text), delimiter="\t")
    if table.fieldnames and "acceptance_pct" in table.fieldnames and "controller_full_accepts" not in table.fieldnames:
        rows = list(table)
        for row in rows:
            try:
                depth = MODULE.DEPTH_VARIANTS[row["variant"]]
                ratio = Fraction(row["acceptance_pct"]) / 100
                if not 0 <= ratio <= 1: continue
                runs = ratio.denominator
                accepted = depth * ratio.numerator
                factor = max(1, (depth + accepted - 1) // accepted) if accepted else 1
                runs *= factor; accepted *= factor
                full = max(1 if accepted else 0, accepted - (depth - 1) * runs)
                remaining_runs = runs - full
                remaining_accepted = accepted - depth * full
                zero = max(0, remaining_runs - remaining_accepted)
                row.update(controller_accepted=str(accepted), controller_attempted=str(depth * runs),
                    controller_runs=str(runs), controller_full_accepts=str(full), controller_zero_accepts=str(zero),
                    controller_rollbacks=str(runs-full), controller_observation="synthetic-fixture.json",
                    controller_observation_semantics="eligible_unflushed_observe_v1", controller_measurement_iterations="1")
            except (KeyError, ValueError, ZeroDivisionError): pass # Invalid fixtures still reach the production rejection.
        output = io.StringIO(); writer = csv.DictWriter(output, fieldnames=[*table.fieldnames, *MODULE.CONTROLLER_OBSERVATION_COLUMNS], delimiter="\t")
        writer.writeheader(); writer.writerows(rows); text = output.getvalue()
    return path.write_text(text, **kwargs)


class MTPDepthPolicyTrainerTest(unittest.TestCase):
    @staticmethod
    def observed_row(**changes):
        """One eligible window plus separately retained physical work evidence."""
        return dict({
            "device": "rocm:0", "model": "fixture.gguf", "model_class": "moe",
            "mode": "stochastic", "case": "independent-window", "variant": "fixed_d5",
            "decode_tps": "123.5", "acceptance_pct": str(100 * 12 / 55),
            "accepted": "12", "drafted": "55",
            "controller_accepted": "12", "controller_attempted": "50", "controller_runs": "10",
            "controller_full_accepts": "2", "controller_zero_accepts": "6", "controller_rollbacks": "8",
            "controller_observation": "fixture-observe.json",
            "controller_observation_semantics": "eligible_unflushed_observe_v1",
            "controller_measurement_iterations": "1",
        }, **changes)

    def test_explicit_controller_observation_uses_eligible_counts_and_preserves_physical_costs(self):
        row = self.observed_row()
        original = dict(row)
        self.assertEqual(MODULE.acceptance_from_summary(row, 5), (.24, "eligible_unflushed_observe_v1"))
        self.assertEqual(row, original)
        self.assertEqual(MODULE.acceptance_from_summary({"acceptance_pct": "21"}, 5), (.21, "physical_aggregate"))

    def test_controller_observation_partial_or_unknown_evidence_is_fatal(self):
        for missing in MODULE.CONTROLLER_OBSERVATION_COLUMNS:
            row = self.observed_row()
            del row[missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "incomplete"):
                MODULE.acceptance_from_summary(row, 5)
        for change in ({"controller_observation_semantics": "consumed_verifier_rows"},
                       {"controller_observation": ""}, {"controller_accepted": "NaN"},
                       {"controller_runs": "1.5"}, {"controller_zero_accepts": "-1"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                MODULE.acceptance_from_summary(self.observed_row(**change), 5)

    def test_controller_observation_counter_algebra_rejects_impossible_windows(self):
        for change in ({"controller_attempted": "55"}, {"controller_runs": "0"},
                       {"controller_measurement_iterations": "0"}, {"controller_accepted": "51"},
                       {"controller_accepted": "11"}, {"controller_accepted": "19"},
                       {"controller_full_accepts": "5", "controller_zero_accepts": "6"},
                       {"controller_rollbacks": "9"}, {"controller_rollbacks": "11"}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "inconsistent"):
                MODULE.acceptance_from_summary(self.observed_row(**change), 5)
        depth_one = self.observed_row(controller_runs="133", controller_attempted="133",
            controller_accepted="123", controller_full_accepts="123", controller_zero_accepts="10")
        self.assertEqual(MODULE.acceptance_from_summary(depth_one, 1)[0], 123 / 133)
        with self.assertRaises(ValueError):
            MODULE.acceptance_from_summary(dict(depth_one, controller_zero_accepts="9"), 1)
        full = self.observed_row(controller_accepted="50", controller_full_accepts="10",
            controller_zero_accepts="0", controller_rollbacks="0")
        self.assertEqual(MODULE.acceptance_from_summary(full, 5)[0], 1.)
        with self.assertRaises(ValueError):
            MODULE.acceptance_from_summary(dict(full, controller_rollbacks="1"), 5)

    def test_controller_observation_loader_preserves_cost_split_and_feature_source(self):
        row = self.observed_row()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired.tsv"
            write_summary_fixture(path, "\t".join(row) + "\n" + "\t".join(row.values()) + "\n")
            measured, = MODULE.load_fixed_rows([path], True)
        self.assertEqual(measured.decode_tps, 123.5)
        self.assertEqual(measured.acceptance_rate, .24)
        self.assertEqual(measured.acceptance_source, "eligible_unflushed_observe_v1")
        self.assertTrue(measured.held_out)
        self.assertTrue(any("split=holdout" == part for part in measured.group_key))

    def test_controller_observation_domain_cannot_mix_feature_semantics_across_splits(self):
        row = fixed_fixture(("fixture",), "rocm", "moe", "greedy", 5, 100., .24,
            acceptance_source="eligible_unflushed_observe_v1")
        legacy_holdout = replace(row, held_out=True, acceptance_source="physical_aggregate")
        with self.assertRaisesRegex(ValueError, "mixed acceptance"):
            MODULE.validate_acceptance_sources([row, legacy_holdout])
        MODULE.validate_acceptance_sources([row, replace(legacy_holdout, backend="cuda")])

    @staticmethod
    def exact_scope(**changes):
        """Frozen geometry fixture; no filename or CLI labels identify this domain."""
        return MODULE.PolicyScope(**dict(
            verifier_width_policy="exact_logical_rows", tp_width=4, pp_stages=1,
            retained_depth=15, hidden_dim=2048, main_layers=40, expert_count=256,
            experts_per_token=8, compute_capability_major=90, compute_capability_minor=6,
            **changes))

    def test_scoped_summary_requires_complete_loaded_identity(self):
        scope = self.exact_scope()
        row = {name: str(getattr(scope, name)) for name in MODULE.SCOPE_COLUMNS}
        row.update(device="rocm:0", model_class="moe")
        self.assertEqual(MODULE.scope_from_summary(row), scope)
        self.assertEqual(MODULE.scope_from_summary({}), MODULE.PolicyScope())
        for key in MODULE.SCOPE_COLUMNS:
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "partial"):
                MODULE.scope_from_summary({k: v for k, v in row.items() if k != key})
        for bad in ({"model_class": ""}, {"model_class": "dense"}, {"device": "unknown"},
                    {"verifier_width_policy": "backend_auto"}, {"compute_capability_major": "0"},
                    {"experts_per_token": "257"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MODULE.scope_from_summary(dict(row, **bad))

    def test_scoped_loader_groups_and_holdout_fit_are_separate(self):
        scope = self.exact_scope()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.tsv"
            columns = ["device", "model", "model_class", "mode", "case", "variant", "decode_tps", "acceptance_pct", *MODULE.SCOPE_COLUMNS]
            lines = ["\t".join(columns)]
            for item in (scope, replace(scope, tp_width=2)):
                for depth in (1, 2, 3, 5):
                    row = ["rocm", "opaque-model.gguf", "moe", "greedy", "same-prompt", f"fixed_d{depth}",
                           str(100 if depth == 5 else 20), "90", *[str(getattr(item, name)) for name in MODULE.SCOPE_COLUMNS]]
                    lines.append("\t".join(row))
            write_summary_fixture(path, "\n".join(lines) + "\n")
            rows = MODULE.load_fixed_rows([path], False)
        self.assertEqual(len({row.group_key for row in rows}), 2)
        choices = MODULE.learn_startups(rows, 0, 0)
        holdout = [replace(row, group_key=("independent-held-out", *row.group_key), held_out=True,
                           decode_tps=10000 if row.depth == 1 else 1) for row in rows]
        self.assertEqual(MODULE.learn_startups(rows + holdout, 0, 0), choices)
        labels = MODULE.label_examples(rows + holdout)
        rules = MODULE.learn_rules(labels, 0, 0, .75, .05)
        self.assertTrue(all(rule.target_depth == 5 for rule in rules))

    def test_scoped_startups_and_labels_never_merge_tp_model_or_architecture(self):
        scope = self.exact_scope()
        scopes = [scope, replace(scope, tp_width=2), replace(scope, hidden_dim=4096),
                  replace(scope, compute_capability_major=94),
                  replace(scope, verifier_width_policy="dynamic_device_envelope")]
        rows = []
        for index, item in enumerate(scopes):
            for depth in range(1, 7):
                rows.append(fixed_fixture((str(index),), "rocm", "moe", "greedy", depth,
                    100 if depth == index + 2 else 10, .9, False, item))
        choices = MODULE.learn_startups(rows, 0, 0)
        self.assertEqual({choice.scope: choice.depth for choice in choices},
                         {item: index + 2 for index, item in enumerate(scopes)})
        labels = MODULE.label_examples(rows, choices, .05)
        self.assertEqual({(example.scope, example.target_depth) for example in labels},
                         {(item, index + 2) for index, item in enumerate(scopes)})

    @staticmethod
    def rule(scope, depth=3, target=5, minimum=.9, maximum=1.0, label="measured"):
        return MODULE.LearnedRule("rocm", "moe", "greedy", depth, target, minimum, maximum,
                                  1.0, 0.0, target-depth, label, 1, 1, 1, 1, scope)

    def test_preservation_replaces_only_complete_measured_scope(self):
        scope = self.exact_scope()
        other = replace(scope, compute_capability_major=94)
        generic = MODULE.PolicyScope()
        rules = [self.rule(generic, target=3, minimum=.74, label="old"),
                 self.rule(scope, label="old_scoped"), self.rule(other, label="other_hardware")]
        startups = [MODULE.LearnedStartup("rocm", "moe", "greedy", 3, 0, 0, generic),
                    MODULE.LearnedStartup("rocm", "moe", "greedy", 5, 1, 1, scope),
                    MODULE.LearnedStartup("rocm", "moe", "greedy", 6, 1, 1, other)]
        fresh = self.rule(scope, target=6, label="fresh")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.inc"
            MODULE.write_include(path, rules, 0, startups)
            preserved = MODULE.preserve_untrained_domains(path, [fresh])
            preserved_startups = MODULE.preserve_untrained_startups(path, [fresh])
            self.assertEqual([row.label for row in preserved], ["old", "other_hardware"])
            self.assertEqual({row.scope for row in preserved_startups}, {generic, other})
            MODULE.write_include(path, preserved + [fresh], 0,
                preserved_startups + [replace(startups[1], depth=6)])
            self.assertEqual([row.label for row in MODULE.preserve_untrained_domains(path, [])],
                             ["old", "other_hardware", "fresh"])

    def test_domain_selected_before_first_match_predicates_not_row_order(self):
        scope = self.exact_scope()
        generic = MODULE.PolicyScope()
        rules = [self.rule(generic, target=3, minimum=.74, label="old_hold"), self.rule(scope)]
        startups = [MODULE.LearnedStartup("rocm", "moe", "greedy", 3, 0, 0, generic),
                    MODULE.LearnedStartup("rocm", "moe", "greedy", 5, 1, 1, scope)]
        rows = [fixed_fixture(("held",), "rocm", "moe", "greedy", depth,
                                     100 if depth == 5 else 50, .95, True, scope)
                for depth in (1, 2, 3, 5)]
        examples = MODULE.label_examples(rows)
        depth3 = [example for example in examples if example.depth == 3]
        report = MODULE.ordered_policy_report(rules, startups, depth3, rows, 0, 0, True)
        self.assertEqual((report["matched"], report["correct"], report["max_regret"]), (1, 1, 0))
        report = MODULE.ordered_policy_report(rules, startups,
                    [replace(depth3[0], acceptance_rate=.8)], rows, 0, 0, True)
        self.assertEqual((report["matched"], report["unmatched"], report["max_regret"]), (0, 1, None))
        self.assertEqual(MODULE.selected_policy_scope(replace(depth3[0], scope=replace(scope, tp_width=2)), startups), generic)

    def test_negative_majority_cannot_certify_zero_support_promotion(self):
        examples = [labeled_fixture("rocm", "moe", "greedy", 3, 3, rate,
                                         "hold", (str(rate),), False)
                    for rate in (.8, .85, .9, .95)]
        examples.append(labeled_fixture("rocm", "moe", "greedy", 3, 5, .9,
                                              "promote", ("promote",), False))
        examples = [replace(example, relative_tps=((3, 1.0), (5, .1))) for example in examples[:-1]] + [
            replace(examples[-1], relative_tps=((3, .1), (5, 1.0)))]
        with self.assertRaisesRegex(ValueError, "inseparable"):
            MODULE.learn_rules(examples, 0, 0, .75, .05)

    def test_ordered_regret_gate_includes_exact_five_percent_only(self):
        scope = self.exact_scope()
        rules = [self.rule(scope, target=3, minimum=.7)]
        startups = [MODULE.LearnedStartup("rocm", "moe", "greedy", 5, 1, 1, scope)]
        for selected_tps, passes in ((95.0, True), (94.999, False)):
            rows = [fixed_fixture(("held",), "rocm", "moe", "greedy", depth,
                    100 if depth == 5 else selected_tps, .95, True, scope)
                    for depth in (1, 2, 3, 5)]
            examples = [item for item in MODULE.label_examples(rows) if item.depth == 3]
            report = MODULE.ordered_policy_report(rules, startups, examples, rows, 0, 0, True)
            if passes:
                self.assertEqual(report["max_regret"], .05)
                MODULE.validate_ordered_holdout(report, 1, 0, .05)
            else:
                with self.assertRaisesRegex(ValueError, "regret"):
                    MODULE.validate_ordered_holdout(report, 1, 0, .05)

    def test_ordered_report_exposes_shadowing_misses_and_selected_depth_regret(self):
        scope = self.exact_scope()
        rules = [self.rule(scope, target=3, minimum=.7), self.rule(scope)]
        startups = [MODULE.LearnedStartup("rocm", "moe", "greedy", 5, 1, 1, scope)]
        rows = [fixed_fixture(("held",), "rocm", "moe", "greedy", depth,
                                     100 if depth == 5 else 50, .95, True, scope)
                for depth in (1, 2, 3, 5)]
        examples = [item for item in MODULE.label_examples(rows) if item.depth == 3]
        report = MODULE.ordered_policy_report(rules, startups, examples, rows, 0, 0, True)
        self.assertEqual(report["overshadowed_rule_matches"], 1)
        self.assertEqual(report["accuracy"], 0)
        self.assertEqual(report["max_regret"], .5)
        with self.assertRaisesRegex(ValueError, "regret"):
            MODULE.validate_ordered_holdout(report, 1, 0, .05)
        empty = MODULE.ordered_policy_report([], startups, examples, rows, 0, 0, True)
        self.assertIsNone(empty["mean_regret"])
        with self.assertRaisesRegex(ValueError, "coverage"):
            MODULE.validate_ordered_holdout(empty, 1, .75, .05)
        with self.assertRaisesRegex(ValueError, "required"):
            MODULE.validate_ordered_holdout(dict(empty, total=0), 1, .75, .05)


    @staticmethod
    def economic_example(acceptance, rates, name, depth=4, startup=5, held_out=False):
        best = max(rates, key=lambda item: (rates[item], -item))
        preferred = startup if startup in rates and rates[startup] >= .95 else best
        action = "promote" if preferred > depth else "demote" if preferred < depth else "hold"
        return labeled_fixture("rocm", "moe", "greedy", depth, preferred, acceptance,
            action, (name,), held_out, relative_tps=tuple(sorted(rates.items())), startup_depth=startup)

    @staticmethod
    def selected(rules, acceptance, depth=4):
        from fractions import Fraction
        ratio = Fraction(str(acceptance))
        for rule in rules:
            minimum, maximum = round(rule.min_acceptance*1_000_000), round(rule.max_acceptance*1_000_000)
            if rule.depth == depth and minimum*ratio.denominator <= ratio.numerator*1_000_000 <= maximum*ratio.denominator:
                return rule.target_depth
        return None

    def test_economic_partition_uses_intersections_before_startup_coverage(self):
        # Preferred labels5/4/5 do not require three regions when4 is within5%
        # everywhere. Region count precedes startup coverage in the objective.
        examples = [self.economic_example(.2, {4: .98, 5: 1.0}, "a"),
                    self.economic_example(.5, {4: 1.0, 5: .8}, "b"),
                    self.economic_example(.8, {4: .98, 5: 1.0}, "c")]
        self.assertEqual([example.target_depth for example in examples], [5, 4, 5])
        rules = MODULE.learn_rules(examples, 0, 0, .75, .05)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].target_depth, 4)
        self.assertEqual(rules[0].positive_support, 3)

    def test_economic_partition_allows_disconnected_target_regions(self):
        examples = [self.economic_example(.2, {4: .5, 5: 1.0}, "a"),
                    self.economic_example(.5, {4: 1.0, 5: .5}, "b"),
                    self.economic_example(.8, {4: .5, 5: 1.0}, "c")]
        rules = MODULE.learn_rules(examples, 0, 0, .75, .05)
        self.assertEqual([rule.target_depth for rule in rules], [5, 4, 5])
        self.assertEqual([self.selected(rules, x) for x in (.2, .5, .8)], [5, 4, 5])

    def test_economic_partition_midpoint_and_startup_preference_prevent_wrong_side_hold(self):
        examples = [self.economic_example(.66, {4: 1.0, 5: .8}, "low"),
                    self.economic_example(.76, {4: .98, 5: 1.0}, "equivalent"),
                    self.economic_example(.85, {4: .8, 5: 1.0}, "high")]
        rules = MODULE.learn_rules(examples, 0, 0, .75, .05)
        self.assertEqual([rule.target_depth for rule in rules], [4, 5])
        self.assertEqual((rules[0].max_acceptance, rules[1].min_acceptance), (.71, .71))
        self.assertEqual(self.selected(rules, .71), 4)
        self.assertEqual(self.selected(rules, .7100005), 5)
        self.assertEqual(self.selected(rules, .75), 5)
        self.assertEqual(self.selected(rules, 1), 5)
        self.assertIsNone(self.selected(rules, .60))
        self.assertIsNone(self.selected(rules, 0))
        reversed_rules = MODULE.learn_rules(list(reversed(examples)), 0, 0, .75, .05)
        self.assertEqual(rules, reversed_rules)

    def test_economic_partition_positive_hold_and_promotion_support_never_invents_epsilon(self):
        for minimum in (.03, .05, .0500000001):
            example = self.economic_example(minimum, {4: 1.0}, "support", startup=4)
            rule, = MODULE.learn_rules([example], 0, 0, .75, .05)
            self.assertEqual(rule.min_acceptance, int(minimum*1_000_000)/1_000_000)
            self.assertIsNone(self.selected([rule], 0))
            self.assertIsNone(self.selected([rule], .01))
            self.assertEqual(self.selected([rule], minimum), 4)
        for minimum in (0.0, .0000003):
            with self.assertRaisesRegex(ValueError, "invalid positive support"):
                MODULE.learn_rules([self.economic_example(minimum, {4: 1.0}, "unsupported", startup=4)], 0, 0, .75, .05)
        promote, = MODULE.learn_rules([self.economic_example(.03, {5: 1.0}, "positive")], 0, 0, .75, .05)
        self.assertEqual(promote.min_acceptance, .03)
        self.assertIsNone(self.selected([promote], .01))
        self.assertEqual(self.selected([promote], 1), 5)
        demote, = MODULE.learn_rules([self.economic_example(.03, {3: 1.0}, "demote")], 0, 0, .75, .05)
        self.assertEqual(self.selected([demote], 0), 3)

    def test_economic_partition_quantized_shared_edges_and_inseparable_support(self):
        self.assertIsNone(MODULE._cut_ppm(.50000001, .50000002))
        self.assertEqual(MODULE._cut_ppm(.5, .5000001), 500000)
        points = [self.economic_example(.5, {1: 1.0}, "left", depth=3),
                  self.economic_example(.5000001, {3: 1.0}, "right", depth=3)]
        rules = MODULE.learn_rules(points, 0, 0, .75, .05)
        self.assertEqual([self.selected(rules, ratio, 3) for ratio in (0, .5, .50000005, 1)], [1, 1, 3, 3])
        for left, right, cut in ((.000123, .0001231, 123), (.000246, .0002461, 246), (.000253, .0002531, 253)):
            self.assertEqual(MODULE._cut_ppm(left, right), cut)
            tiny = MODULE.learn_rules([replace(points[0], acceptance_rate=left),
                                       replace(points[1], acceptance_rate=right)], 0, 0, .75, .05)
            self.assertEqual(self.selected(tiny, left, 3), 1)
            self.assertEqual(self.selected(tiny, right, 3), 3)
            self.assertEqual(round(tiny[0].max_acceptance*1_000_000), cut)
        for left, right in ((.5, .5), (.50000001, .50000002)):
            with self.assertRaisesRegex(ValueError, "inseparable"):
                MODULE.learn_rules([replace(points[0], acceptance_rate=left), replace(points[1], acceptance_rate=right)], 0, 0, .75, .05)
        with self.assertRaisesRegex(ValueError, "measured relative throughput support"):
            MODULE.learn_rules([replace(points[0], relative_tps=())], 0, 0, .75, .05)

    def test_economic_partition_heldout_changes_only_scores_and_true_regret_stays_red(self):
        train = [self.economic_example(.66, {4: 1.0, 5: .8}, "low"),
                 self.economic_example(.76, {4: .98, 5: 1.0}, "equivalent"),
                 self.economic_example(.85, {4: .8, 5: 1.0}, "high")]
        held = self.economic_example(.68, {4: .9, 5: 1.0}, "held", held_out=True)
        first = MODULE.learn_rules(train, 0, 0, .75, .05)
        with_holdout = MODULE.learn_rules(train+[held], 0, 0, .75, .05)
        self.assertEqual(first, [replace(rule, holdout_correct=0, holdout_total=0) for rule in with_holdout])
        rows = [fixed_fixture(example.group_key, example.backend, example.model_class, example.mode,
                    depth, rate*100, example.acceptance_rate, example.held_out)
                for example in train+[held] for depth, rate in example.relative_tps]
        report = MODULE.ordered_policy_report(with_holdout, [], train+[held], rows, 0, 0, True)
        self.assertEqual(report["coverage"], 1)
        self.assertEqual(report["accuracy"], 0)
        self.assertAlmostEqual(report["max_regret"], .1)
        with self.assertRaisesRegex(ValueError, "regret"):
            MODULE.validate_ordered_holdout(report, 1, 0, .05)
        # A genuine unobserved low-support hole must also remain a failed gate.
        held = replace(held, acceptance_rate=.01)
        report = MODULE.ordered_policy_report(first, [], [held], rows, 0, 0, True)
        self.assertEqual(report["coverage"], 0)
        with self.assertRaisesRegex(ValueError, "coverage"):
            MODULE.validate_ordered_holdout(report, 1, .75, .05)

    def test_economic_holdout_gate_requires_evidence_for_every_fresh_domain(self):
        train = [self.economic_example(.66, {4: 1., 5: .8}, "low"),
                 self.economic_example(.85, {4: .8, 5: 1.}, "high")]
        rules = MODULE.learn_rules(train, 0, 0, .75, .05)
        rows = [fixed_fixture(example.group_key, example.backend, example.model_class, example.mode,
                    depth, rate*100, example.acceptance_rate, example.held_out)
                for example in train for depth, rate in example.relative_tps]
        with self.assertRaisesRegex(ValueError, "heldout evidence required"):
            MODULE.validate_holdout_domains(rules, [], train, rows, 0, 0, .75, 0, 0, None, .05)
        held = replace(train[1], group_key=("held",), held_out=True)
        other_train = [replace(example, backend="cuda", group_key=("other", *example.group_key)) for example in train]
        all_examples = train + other_train + [held]
        all_rules = MODULE.learn_rules(all_examples, 0, 0, .75, .05)
        all_rows = [fixed_fixture(example.group_key, example.backend, example.model_class, example.mode,
                    depth, rate*100, example.acceptance_rate, example.held_out)
                for example in all_examples for depth, rate in example.relative_tps]
        with self.assertRaisesRegex(ValueError, "heldout evidence required"):
            MODULE.validate_holdout_domains(all_rules, [], all_examples, all_rows, 0, 0, .75, 0, 0, None, .05)
        # One observed supported region suffices for the requested per-region
        # precision gate; the other region is unvisited, never credited.
        observed = train+[held]
        observed_rows = [fixed_fixture(example.group_key, example.backend, example.model_class, example.mode,
                    depth, rate*100, example.acceptance_rate, example.held_out)
                for example in observed for depth, rate in example.relative_tps]
        supported_rules = MODULE.learn_rules(observed, 0, 0, .75, .05)
        self.assertTrue(any(rule.holdout_total == 0 for rule in supported_rules))
        MODULE.validate_holdout_domains(supported_rules, [], observed, observed_rows, 0, 0, .75, 1, 1, .05, .05)
        # A heavily populated good domain cannot mask a missed smaller one.
        missed = replace(held, backend="cuda", group_key=("missed",), acceptance_rate=.01)
        bad_examples = observed + other_train + [missed]
        bad_rows = observed_rows + [fixed_fixture(example.group_key, example.backend, example.model_class, example.mode,
                    depth, rate*100, example.acceptance_rate, example.held_out)
                for example in other_train+[missed] for depth, rate in example.relative_tps]
        bad_rules = MODULE.learn_rules(bad_examples, 0, 0, .75, .05)
        with self.assertRaisesRegex(ValueError, "selected heldout evidence required"):
            MODULE.validate_holdout_domains(bad_rules, [], bad_examples, bad_rows, 0, 0, .75, 0, 0, None, .05)
        with self.assertRaisesRegex(ValueError, "coverage"):
            MODULE.validate_holdout_domains(bad_rules, [], bad_examples, bad_rows, 0, 0, 0, 1, .75, .05, .05)

    def test_economic_cli_minimum_holdout_accuracy_cannot_emit_without_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "train.tsv"
            self.write_summary(summary)
            output, report = root / "candidate.inc", root / "report.txt"
            result = subprocess.run([sys.executable, str(TRAINER), "--input", str(summary),
                "--output", str(output), "--summary", str(report), "--holdout-modulus", "0",
                "--min-holdout-accuracy", ".75"], text=True, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("heldout evidence required", result.stderr)
            self.assertFalse(output.exists())

    def test_near_best_startup_is_stable_but_real_regressions_keep_their_label(self) -> None:
        """Economic ties do not erase a genuine held-out throughput deficit."""
        startup = MODULE.LearnedStartup("cpu", "moe", "stochastic", 3, 4, 0.99)
        for rate, expected in ((95.0, 3), (94.99, 4)):
            rows = [fixed_fixture(
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
            rows.extend(fixed_fixture(
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
        rows.extend(fixed_fixture(
            group_key=("holdout",), backend="cpu", model_class="moe",
            mode="stochastic", depth=depth, decode_tps=1000 if depth == 4 else 1,
            acceptance_rate=0.9, held_out=True,
        ) for depth in (1, 2, 3, 4))
        self.assertEqual(MODULE.learn_startups(rows, 0, 0), [startup])

    def test_startup_normalizes_requests_and_requires_common_measured_depths(self) -> None:
        """Prompt speed units and one missing deep lane cannot bias admission."""
        rows = []
        for case, scale in (("fast", 1000), ("slow", 1)):
            rows.extend(fixed_fixture(
                group_key=(case,), backend="cpu", model_class="moe",
                mode="stochastic", depth=depth, decode_tps=scale * (20 + depth),
                acceptance_rate=0.8, held_out=False,
            ) for depth in (1, 2, 3))
        rows.append(fixed_fixture(
            ("fast",), "cpu", "moe", "stochastic", 15, 1e9, 0.9, False))
        startup, = MODULE.learn_startups(rows, 0, 0)
        self.assertEqual(startup.depth, 3)
        with self.assertRaisesRegex(ValueError, "training"):
            MODULE.learn_startups([
                fixed_fixture(("held",), "cpu", "moe", "stochastic",
                                     depth, 100, 0.9, True)
                for depth in (1, 2, 3)
            ], 0, 0)

    def test_every_supported_depth_can_be_the_learned_startup(self) -> None:
        """Admission may start at any measured winner, including depth fifteen."""
        for winner in range(1, 16):
            rows = [fixed_fixture(
                group_key=("request",), backend="cpu", model_class="moe",
                mode="stochastic", depth=depth, decode_tps=200 if depth == winner else 100,
                acceptance_rate=0.9, held_out=False,
            ) for depth in range(1, 16)]
            startup, = MODULE.learn_startups(rows, 0, 0)
            self.assertEqual(startup.depth, winner)

    def test_every_supported_measured_depth_can_win(self) -> None:
        """The learner must not discard d4..d15 after the sweep measured them."""
        for winner in range(1, 16):
            rows = [fixed_fixture(
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
            rows.extend(fixed_fixture(
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
        rows = [fixed_fixture(
            group_key=("only-group",), backend="rocm", model_class="moe",
            mode="stochastic", depth=depth, decode_tps=100 + depth,
            acceptance_rate=0.9,
        ) for depth in (1, 2, 3)]
        with self.assertRaisesRegex(ValueError, "must never be used for fitting"):
            MODULE.learn_rules(MODULE.label_examples(rows), 1, 0, 0.75, 0.05)

    def test_hold_margin_is_included_in_the_reported_accuracy(self) -> None:
        examples = [
            labeled_fixture("rocm", "moe", "stochastic", 3, 3, 0.80,
                                  "hold", ("train",), False),
            labeled_fixture("rocm", "moe", "stochastic", 3, 1, 0.77,
                                  "demote", ("holdout",), True),
        ]
        examples = [replace(examples[0], relative_tps=((1, .1), (3, 1.0))),
                    replace(examples[1], relative_tps=((1, 1.0), (3, .1)))]
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
            write_summary_fixture(path, contents.replace(marker, marker + first + "\n"))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                MODULE.preserve_untrained_startups(path, fresh)
            write_summary_fixture(path, '{MTPVerifyMode::bad},\n')
            with self.assertRaisesRegex(ValueError, "malformed"):
                MODULE.preserve_untrained_domains(path, fresh)
            with self.assertRaisesRegex(ValueError, "malformed"):
                MODULE.preserve_untrained_startups(path, fresh)

    def test_duplicate_depth_or_invalid_measurement_is_rejected(self) -> None:
        row = fixed_fixture(("request",), "rocm", "moe", "stochastic", 1, 100, 0.9)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.label_examples([row, row])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.tsv"
            for throughput, acceptance in (("nan", "90"), ("0", "90"), ("100", "101")):
                write_summary_fixture(path, "device\tmodel\tmode\tvariant\tdecode_tps\tacceptance_pct\n"
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
        write_summary_fixture(path,
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
            self.assertIn(", +2, \"trained_cuda_dense_greedy_promote_d1_to_d3_region0\"", include_text)
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
            write_summary_fixture(summary,
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
            write_summary_fixture(summary,
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
            self.assertIn(", +2, \"trained_rocm_moe_greedy_promote_d1_to_d3_region0\"", include_text)

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
            write_summary_fixture(short_summary,
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
            write_summary_fixture(long_summary,
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
                "MTPDepthPolicyModelClass::MoE, 1, 0.460000, 0.605000",
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


    def test_full_prefix_evidence_guards_fit_and_ordered_promotion_matches(self):
        example = self.economic_example(2 / 3, {3: .5, 4: 1.}, "guard", depth=3, startup=4)
        rule = MODULE.LearnedRule("rocm", "moe", "greedy", 3, 4, .6, .75,
            1., 0., 1, "promotion", 1, 1, 0, 0)
        rows = [fixed_fixture(example.group_key, "rocm", "moe", "greedy", depth,
                    100 if depth == 4 else 50, 2 / 3, False) for depth in (3, 4)]
        for evidence in (0, None, 1):
            observed = replace(example, full_accepts_lower_bound=evidence)
            self.assertEqual(MODULE._interval_matches(rule, observed), evidence == 1)
            report = MODULE.ordered_policy_report([rule], [], [observed], rows, 0, 0, False)
            self.assertEqual(report["matched"], int(evidence == 1))
            if evidence == 1:
                fitted = MODULE.learn_rules([observed], 0, 0, .75, .05)
                self.assertEqual([r.target_depth for r in fitted], [4])
            else:
                with self.assertRaisesRegex(ValueError, "no supported economic partition"):
                    MODULE.learn_rules([observed], 0, 0, .75, .05)
            for target in (2, 3):
                self.assertTrue(MODULE._interval_matches(replace(rule, target_depth=target, delta=target-3), observed))
        for malformed in (True, 1.0, -1):
            with self.assertRaisesRegex(ValueError, "full-prefix"):
                MODULE.promotion_supported(4, replace(example, full_accepts_lower_bound=malformed))

    def test_full_prefix_lower_bound_retains_unknown_legacy_and_exact_observe_counts(self):
        # Even exact physical counters need not describe the eligible window.
        for physical in ({"acceptance_pct": "66.6667"},
                         {"acceptance_pct": "66.6667", "accepted": "8", "drafted": "12", "verifier_runs": "4"},
                         {"acceptance_pct": "75", "accepted": "9", "drafted": "12", "verifier_runs": "4"}):
            self.assertIsNone(MODULE.full_accepts_lower_bound_from_summary(physical, 3, "physical_aggregate"))
        fallback = {"variant": "fixed_d1", "decode_tps": "1", "accepted": "1", "drafted": "2", "verifier_runs": "2", "fixture_identity": "same"}
        self.assertEqual(MODULE._group_key(fallback), MODULE._group_key(dict(fallback, variant="fixed_d3", accepted="9", drafted="12", verifier_runs="4")))
        observed = self.observed_row()
        self.assertEqual(MODULE.full_accepts_lower_bound_from_summary(observed, 5, "eligible_unflushed_observe_v1"), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observed.tsv"
            write_summary_fixture(path, "\t".join(observed)+"\n"+"\t".join(observed.values())+"\n")
            row, = MODULE.load_fixed_rows([path], False)
        self.assertEqual(row.full_accepts_lower_bound, 2)
        self.assertEqual(row.acceptance_source, "eligible_unflushed_observe_v1")
        # Complete fixed1..3 rows propagate exact zero as zero, not unknown.
        rows = [replace(row, depth=d, group_key=("same",), acceptance_rate=.5,
                        full_accepts_lower_bound=0) for d in (1, 2, 3)]
        self.assertTrue(all(e.full_accepts_lower_bound == 0 for e in MODULE.label_examples(rows)))
        rows = [replace(row, full_accepts_lower_bound=None) for row in rows]
        self.assertTrue(all(e.full_accepts_lower_bound is None for e in MODULE.label_examples(rows)))

if __name__ == "__main__":
    unittest.main()
