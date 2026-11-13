#!/usr/bin/env python3
"""Device-free actual-program SDK receipt lifecycle/negative replay."""
import importlib.util
from pathlib import Path
import sys
import unittest
import copy
import json

PATH = Path(sys.argv.pop(1)) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else Path(__file__).resolve().parents[4] / 'scripts/diagnostics/inspect_pooled_native_program_observation.py'
spec = importlib.util.spec_from_file_location('pooled_observation', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def trace(backend='CUDA', failed=False):
    rows = []
    operations = [('StreamCreateWithFlags', '(nil)', '0x10'), ('EventCreateWithFlags', '(nil)', '0x20'),
                  ('StreamBeginCapture', '0x10', '0x10'), ('StreamEndCapture', '0x10', '0x30'),
                  ('GraphInstantiateWithFlags' if backend == 'CUDA' else 'GraphInstantiate', '0x30', '(nil)' if failed else '0x40')]
    if not failed:
        operations += [('GraphLaunch', '0x40', '0x40'), ('GraphExecDestroy', '0x40', '0x40')]
    operations += [('GraphDestroy', '0x30', '0x30'), ('EventDestroy', '0x20', '0x20'), ('StreamDestroy', '0x10', '0x10'), ('DeviceReset', '(nil)', '(nil)')]
    for operation, before, after in operations:
        for phase, handle in [('before', before), ('after', after)]:
            rows.append(dict(schema=2, pid=7, thread=9, parent_backend="", parent_operation="", parent_ordinal=0, qualified=False, backend=backend, operation=operation, phase=phase, ordinal=1, handle=handle, caller='0x123', stream='0x10' if operation in ('StreamBeginCapture','StreamEndCapture','GraphLaunch','StreamDestroy') else '(nil)', flags=0, result=2 if failed and phase == 'after' and operation.startswith('GraphInstantiate') else 0,
                             **{name: len(rows) + 1 for name in module.FIELDS}))
    return rows


def pair(operation, before, after, backend="CUDA", ordinal=1, parent=None, thread=9):
    """Recorded API results, not native calls or certified resource numbers."""
    rows = []
    for phase, handle in (("before", before), ("after", after)):
        rows.append(dict(schema=2, pid=7, thread=thread, qualified=False, backend=backend,
            parent_backend=backend if parent else "", parent_operation=parent[0] if parent else "",
            parent_ordinal=parent[1] if parent else 0, operation=operation, phase=phase, ordinal=ordinal,
            handle=handle, caller="0x123", stream="(nil)", flags=0, result=0,
            **{name: 1 for name in module.FIELDS}))
    return rows


def stamp(rows):
    """Retain ordered host observation times around this recorded call interleaving."""
    for i, row in enumerate(rows):
        for field in module.FIELDS:
            row[field] = i + 1
    return rows


def inventory_trace(backend="CUDA", failed=False):
    """Preserve one real-call-shaped flat snapshot between instantiate entry and result."""
    rows = trace(backend, failed)
    at = next(i for i, row in enumerate(rows) if row["operation"].startswith("GraphInstantiate"))
    original = rows[at]
    base = dict(schema=1, pid=original["pid"], thread=original["thread"], backend=backend,
                operation=original["operation"], ordinal=1, graph=original["handle"], nodes=2, edges=1,
                index=0, node="(nil)", other="(nil)", native_kind=-1, error="", qualified=False)
    inventory = [dict(base, observation="begin"),
                 dict(base, observation="node", node="0x100", native_kind=0),
                 dict(base, observation="node", node="0x200", native_kind=7, index=1),
                 dict(base, observation="edge", node="0x100", other="0x200"),
                 dict(base, observation="complete")]
    return [(module.PREFIX, row) for row in rows[:at + 1]] + [(module.INVENTORY_PREFIX, row) for row in inventory] + [(module.PREFIX, row) for row in rows[at + 1:]]


def inventory_validate(records):
    """Use the same explicit offline completeness gate as native diagnostics."""
    return module.validate_graph_inventory([prefix + json.dumps(row) for prefix, row in records])


class NativeGraphInventory(unittest.TestCase):
    def test_every_actual_instantiate_including_failed_prefix_needs_its_dag(self):
        for backend in ("CUDA", "ROCm"):
            for failed in (False, True):
                for _ in range(20):
                    result = inventory_validate(inventory_trace(backend, failed))
                    self.assertFalse(result["qualified"])
                    self.assertFalse(result["resource_limits_installed"])
                    row = result["inventories"][0]
                    self.assertEqual((row["nodes"], row["edges"], row["child_graphs"]), (2, 1, 0))
                    self.assertEqual(row["native_kind_counts"], {0: 1, 7: 1})

    def test_predecessor_node_only_observations_do_not_prove_completeness(self):
        with self.assertRaisesRegex(ValueError, "complete graph inventory"):
            inventory_validate([(module.PREFIX, row) for row in trace()])

    def test_every_missing_snapshot_record_rejects(self):
        rows = inventory_trace()
        for index, (prefix, _) in enumerate(rows):
            if prefix == module.INVENTORY_PREFIX:
                with self.subTest(index=index), self.assertRaises(ValueError):
                    inventory_validate(rows[:index] + rows[index + 1:])

    def test_foreign_graph_thread_process_or_instantiate_rejects(self):
        for field, value in (("graph", "0x99"), ("thread", 99), ("pid", 99), ("ordinal", 99)):
            rows = inventory_trace()
            row = next(row for prefix, row in rows if prefix == module.INVENTORY_PREFIX)
            row[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "outside its original"):
                inventory_validate(rows)

    def test_child_host_conditional_and_unknown_kinds_cannot_be_flat(self):
        for kind in (3, 4, 8, 10, 13, 99):
            rows = inventory_trace()
            node = next(row for prefix, row in rows if prefix == module.INVENTORY_PREFIX and row["observation"] == "node")
            node["native_kind"] = kind
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "outside the closed flat"):
                inventory_validate(rows)

    def test_duplicate_nodes_edges_or_foreign_endpoints_reject(self):
        for mutation in ("node", "edge", "endpoint"):
            rows = inventory_trace()
            at = next(i for i, (prefix, row) in enumerate(rows) if prefix == module.INVENTORY_PREFIX and row["observation"] == mutation.replace("endpoint", "edge"))
            if mutation == "endpoint": rows[at][1]["other"] = "0x99"
            else: rows.insert(at + 1, copy.deepcopy(rows[at]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                inventory_validate(rows)

    def test_a_cycle_cannot_be_mislabeled_as_a_valid_captured_dag(self):
        rows = inventory_trace()
        at = next(i for i, (prefix, row) in enumerate(rows) if prefix == module.INVENTORY_PREFIX and row["observation"] == "edge")
        reverse = copy.deepcopy(rows[at][1])
        reverse.update(index=1, node="0x200", other="0x100")
        rows.insert(at + 1, (module.INVENTORY_PREFIX, reverse))
        for prefix, row in rows:
            if prefix == module.INVENTORY_PREFIX: row["edges"] = 2
        with self.assertRaisesRegex(ValueError, "dependency cycle"):
            inventory_validate(rows)

    def test_failed_native_query_is_preserved_and_rejected(self):
        rows = inventory_trace()
        at = next(i for i, (prefix, _) in enumerate(rows) if prefix == module.INVENTORY_PREFIX)
        rows[at][1].update(observation="failed", error="node_kind_query")
        with self.assertRaisesRegex(ValueError, "Native flat graph inspection failed: node_kind_query"):
            inventory_validate(rows)

    def test_successful_inventory_cannot_be_reused_after_the_api_result(self):
        rows = inventory_trace()
        snapshot = [(prefix, row) for prefix, row in rows if prefix == module.INVENTORY_PREFIX]
        rows = [(prefix, row) for prefix, row in rows if prefix != module.INVENTORY_PREFIX]
        rows.extend(snapshot)
        with self.assertRaisesRegex(ValueError, "complete graph inventory"):
            inventory_validate(rows)

    def test_inventory_schema_and_observation_fields_are_exact(self):
        for event, field, value in (("begin", "schema", True), ("begin", "nodes", True),
                                    ("begin", "node", "0x10"), ("complete", "index", 1),
                                    ("node", "other", "0x200"), ("edge", "native_kind", 0)):
            rows = inventory_trace()
            row = next(row for prefix, row in rows if prefix == module.INVENTORY_PREFIX and row["observation"] == event)
            row[field] = value
            with self.subTest(event=event, field=field), self.assertRaises(ValueError):
                inventory_validate(rows)


class ActualProgramObservation(unittest.TestCase):
    def test_actual_success_reset_and_repeat_both_backends(self):
        for backend in ('CUDA', 'ROCm'):
            for _ in range(20):
                result = module.validate_records(trace(backend))
                self.assertFalse(result['qualified'])
                self.assertFalse(result['resource_limits_installed'])
                self.assertTrue(result['validated_original_reset_receipt_required'])

    def test_original_instantiate_error_requires_complete_prefix_cleanup(self):
        for backend in ('CUDA', 'ROCm'):
            rows = trace(backend, True)
            module.validate_records(rows, 1)
            with self.assertRaises(ValueError): module.validate_records(rows)
            with self.assertRaises(ValueError): module.validate_records(rows[:-4], 1)

    def test_pairs_and_original_caller_are_mandatory(self):
        rows = trace()
        with self.assertRaises(ValueError): module.validate_records(rows[1:])
        rows[1]['caller'] = '0x456'
        with self.assertRaises(ValueError): module.validate_records(rows)

    def test_duplicate_skipped_and_live_handle_alias_reject(self):
        rows = trace()
        rows[0]['ordinal'] = 2
        with self.assertRaises(ValueError): module.validate_records(rows)
        rows = trace()
        rows[4:4] = copy.deepcopy(rows[:2])
        with self.assertRaises(ValueError): module.validate_records(rows)

    def test_reset_cannot_hide_live_graph_or_event(self):
        rows = trace()
        rows[-2:] = copy.deepcopy(rows[-2:])
        del rows[12:14]  # The successful executable still owns a live handle.
        with self.assertRaises(ValueError): module.validate_records(rows)

    def test_numeric_or_certificate_role_override_rejects(self):
        rows = trace()
        rows[0]['qualified'] = True
        with self.assertRaises(ValueError): module.validate_records(rows)
        rows = trace()
        rows[0]['certified_limit'] = 1024
        with self.assertRaises(ValueError): module.validate_records(rows)
        rows = trace()
        del rows[0]['rss_bytes']
        with self.assertRaises(ValueError): module.validate_records(rows)

    def test_plain_event_and_successful_graph_clone_lineage(self):
        for backend in ("CUDA", "ROCm"):
            rows = trace(backend)
            for row in rows:
                if row["operation"] == "EventCreateWithFlags": row["operation"] = "EventCreate"
                if row["operation"].startswith("GraphInstantiate") and row["phase"] == "before": row["handle"] = "0x32"
            rows[8:8] = pair("GraphCreate", "(nil)", "0x31", backend) + pair("GraphClone", "0x30", "0x32", backend)
            at = next(i for i, row in enumerate(rows) if row["operation"] == "GraphDestroy")
            rows[at:at] = pair("GraphDestroy", "0x32", "0x32", backend, 2) + pair("GraphDestroy", "0x31", "0x31", backend, 3)
            result = module.validate_records(stamp(rows))
            self.assertEqual(result["native_handle_creations"]["graph"], 3)
            clone = next(row for row in result["creator_lineage"] if row["creator"][1] == "GraphClone")
            self.assertEqual(clone["source_graph"], (backend, "StreamEndCapture", 1))
            executable = next(row for row in result["creator_lineage"] if row["kind"] == "executable")
            self.assertEqual(executable["source_graph"], (backend, "GraphClone", 1))
            self.assertTrue(result["observer_overhead_included"])

    def test_nested_exact_sdk_forwarding_is_one_native_handle(self):
        for backend in ("CUDA", "ROCm"):
            rows = trace(backend)
            outer = pair("EventCreate", "(nil)", "0x20", backend)
            for row in rows[2:4]:
                row.update(parent_backend=backend, parent_operation="EventCreate", parent_ordinal=1)
            rows[2:4] = [outer[0], *rows[2:4], outer[1]]
            at = next(i for i, row in enumerate(rows) if row["operation"].startswith("GraphInstantiate"))
            clone = pair("GraphClone", "0x30", "0x32", backend)
            child = pair("GraphCreate", "(nil)", "0x32", backend, parent=("GraphClone", 1))
            rows[at:at] = [clone[0], *child, clone[1]]
            for row in rows:
                if row["operation"].startswith("GraphInstantiate") and row["phase"] == "before": row["handle"] = "0x32"
            at = next(i for i, row in enumerate(rows) if row["operation"] == "GraphDestroy")
            rows[at:at] = pair("GraphDestroy", "0x32", "0x32", backend, 2)
            result = module.validate_records(stamp(rows))
            self.assertEqual(result["native_handle_creations"]["event"], 1)
            self.assertEqual(result["native_handle_creations"]["graph"], 2)

    def test_foreign_clone_and_instantiate_graph_reject(self):
        for operation in ("GraphClone", "GraphInstantiateWithFlags"):
            rows = trace()
            if operation == "GraphClone": rows[8:8] = pair(operation, "0xdead", "0x32")
            else: rows[8]["handle"] = "0xdead"
            with self.assertRaisesRegex(ValueError, "original live graph"):
                module.validate_records(stamp(rows))

    def test_unrelated_alias_or_forged_parent_thread_reject(self):
        rows = trace()
        rows[4:4] = pair("EventCreateWithFlags", "(nil)", "0x20", ordinal=2)
        with self.assertRaisesRegex(ValueError, "unrelated live handle"):
            module.validate_records(stamp(rows))
        for wrong in ("thread", "parent"):
            rows = trace();outer = pair("EventCreate", "(nil)", "0x20")
            for row in rows[2:4]: row.update(parent_backend="CUDA", parent_operation="EventCreate", parent_ordinal=1)
            if wrong == "thread":
                for row in rows[2:4]: row["thread"] = 10
            else:
                for row in rows[2:4]: row["parent_ordinal"] = 2
            rows[2:4] = [outer[0], *rows[2:4], outer[1]]
            with self.assertRaisesRegex(ValueError, "parent/thread"):
                module.validate_records(stamp(rows))

    def test_atomic_ordinals_can_publish_out_of_order_on_distinct_threads(self):
        rows = trace()
        rows[2:2] = pair("EventCreateWithFlags", "(nil)", "0x21", ordinal=2, thread=10)
        rows[-2:-2] = pair("EventDestroy", "0x21", "0x21", ordinal=2, thread=10)
        self.assertEqual(module.validate_records(stamp(rows))["native_handle_creations"]["event"], 2)
        rows[2]["ordinal"] = rows[3]["ordinal"] = 3
        with self.assertRaisesRegex(ValueError, "Skipped"):
            module.validate_records(stamp(rows))

    def test_old_schema_and_detached_capture_reject(self):
        rows = trace();rows[0]["schema"] = 1
        with self.assertRaisesRegex(ValueError, "schema"): module.validate_records(rows)
        rows = trace();del rows[4:6]
        with self.assertRaisesRegex(ValueError, "capture begin"): module.validate_records(stamp(rows))

    def test_labeled_malformed_json_fails_and_unrelated_logs_are_preserved_elsewhere(self):
        with self.assertRaises(ValueError): module.parse_records(['POOLED_NATIVE_API {broken'])
        self.assertEqual(module.parse_records(['ordinary logger line']), [])


if __name__ == '__main__': unittest.main()
