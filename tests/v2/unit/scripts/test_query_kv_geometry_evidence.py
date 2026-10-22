"""Device-free artifact qualification negatives; no model/native/payload access."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integration" / "parity"))
from query_kv_geometry_evidence import validate_geometry


def fixture():
    """Authored original metadata and independent literal TP4 coverage tuples."""
    expected = dict(schema="query_kv_geometry_expectation_v1", world_rank=0,
                    original_q=24, original_kv=2, head_dim=256, k=2560, main_layer_count=8, attention_layers=[3,7],
                    participants=[dict(device=f"ROCm:{d}", q_first=q, q_count=6)
                                  for d, q in ((19, 0), (7, 6), (11, 12), (3, 18))],
                    sources=[dict(layer=layer, role=role, name=f"blk.{layer}.attn_{role.lower()}.weight",
                                  codebook=19, superblock=False)
                             for layer in (3, 7) for role in ("K", "V")])
    records = []
    for participant, (device, q_first, kv_first) in enumerate(((19, 0, 0), (7, 6, 0), (11, 12, 1), (3, 18, 1))):
        name = f"ROCm:{device}"
        base = dict(schema="query_kv_geometry_v1", construction_only="true", per_instance_replay_proven="false",
                    admission_identity="99")
        assignment = dict(coverage="QueryInterval", participant=str(participant), degree="4", world_rank="0",
                          original_q="24", original_kv="2", head_dim="256", k="2560", q_first=str(q_first), q_count="6",
                          kv_first=str(kv_first), kv_count="1", kv_planned_bytes="1024", kv_resident_bytes="0")
        def record(tags):
            records.append(dict(domain="gpu_graph_inventory", name="query_kv_geometry", device=name,
                                kind="counter", phase={"selected_admission_input":"admission_installed", "graph_cache_pma":"cache_constructed",
                                                       "prepared_projection":"cold_binding"}[tags["kind"]],
                                count=1, value=1, tags=dict(base, **tags)))
        record(dict(assignment, kind="selected_admission_input"))
        record(dict(assignment, kind="graph_cache_pma", cache_first=str(kv_first), cache_count="1",
                    payload_first=str(kv_first), payload_count="1", kv_materialized_bytes="1024",
                    kv_committed_new_bytes="1024"))
        for source in expected["sources"]:
            record(dict(kind="prepared_projection", role=source["role"], source_name=source["name"],
                        source_layer=str(source["layer"]), source_codebook="19", source_superblock="false",
                        binding_id=str(source["layer"]*2+(source["role"]=="V")), model_id="0",
                        physical_n="256", policy_n="512", first_column=str(kv_first*256), k="2560",
                        stage_variant_identity="456"))
        records.append(dict(domain="forward_graph", name="retained_parent_replays", device=name, kind="counter", phase="decode", count=20, value=20, tags={}))
    return dict(schema="llaminar.perf_stats.v1", records=records), expected


class QueryKVGeometryEvidenceTest(unittest.TestCase):
    """Source/owner completeness is independent of equal final output tensors."""
    def test_complete_nonordinal_metadata_and_replay_limit(self):
        """Complete nonordinal metadata and replay limit."""
        artifact, expected = fixture()
        result = validate_geometry(artifact, expected)
        self.assertTrue(result["passed"])
        self.assertFalse(result["per_instance_replay_proven"])
        self.assertEqual(result["original_source_roles"], 4)

    def test_original_source_codebook_zero_is_not_missing(self):
        """Original source codebook zero is not missing."""
        artifact, expected = fixture()
        for source in expected["sources"]:
            source["codebook"] = 0
        for record in artifact["records"]:
            if "source_codebook" in record.get("tags", {}):
                record["tags"]["source_codebook"] = "0"
        self.assertTrue(validate_geometry(artifact, expected)["passed"])
        next(r["tags"] for r in artifact["records"] if "source_codebook" in r.get("tags", {}))["source_codebook"] = "19"
        with self.assertRaises(ValueError):
            validate_geometry(artifact, expected)

    def test_wrong_case_device_identity_fails(self):
        """Wrong case device identity fails."""
        artifact, expected = fixture()
        expected["participants"][0]["device"] = "rocm:19"
        with self.assertRaises(ValueError):
            validate_geometry(artifact, expected)

    def test_actual_schema_counter_and_publication_phase_are_required(self):
        """Actual schema counter and publication phase are required."""
        for field, value in (("kind", "timer"), ("phase", "graph_setup"), ("count", 0), ("value", float("nan")),
                             ("value", 0), ("value", 2)):
            artifact, expected = fixture()
            artifact["records"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_geometry(artifact, expected)
        artifact, expected = fixture()
        artifact["schema"] = "foreign"
        with self.assertRaises(ValueError):
            validate_geometry(artifact, expected)

    def test_missing_each_authority_fails(self):
        """Missing each authority fails."""
        for kind in ("selected_admission_input", "graph_cache_pma", "prepared_projection"):
            artifact, expected = fixture()
            artifact["records"] = [r for r in artifact["records"] if r.get("tags", {}).get("kind") != kind]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_geometry(artifact, expected)

    def test_legacy_and_full_bank_cannot_masquerade_as_query_local(self):
        """Legacy and full bank cannot masquerade as query local."""
        for field, value in (("coverage", "Legacy"), ("physical_n", "512"), ("policy_n", "256")):
            artifact, expected = fixture()
            for record in artifact["records"]:
                if field in record.get("tags", {}):
                    record["tags"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_geometry(artifact, expected)

    def test_all_origin_zero_even_when_all_authorities_agree_fails(self):
        """All origin zero even when all authorities agree fails."""
        artifact, expected = fixture()
        for record in artifact["records"]:
            for field in ("kv_first", "cache_first", "payload_first", "first_column"):
                if field in record.get("tags", {}):
                    record["tags"][field] = "0"
        with self.assertRaises(ValueError):
            validate_geometry(artifact, expected)

    def test_conflicting_assignment_source_cache_or_pma_fails(self):
        """Conflicting assignment source cache or pma fails."""
        for field, value in (("q_first", "6"), ("participant", "2"), ("payload_count", "2"),
                             ("source_name", "blk.0.attn_k.weight"), ("source_codebook", "5"),
                             ("admission_identity", "101"), ("kv_planned_bytes", "2048")):
            artifact, expected = fixture()
            next(r["tags"] for r in artifact["records"] if field in r.get("tags", {}))[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_geometry(artifact, expected)

    def test_no_instance_claim_or_unjoined_replay(self):
        """No instance claim or unjoined replay."""
        artifact, expected = fixture()
        artifact["records"][0]["tags"]["per_instance_replay_proven"] = "true"
        with self.assertRaises(ValueError):
            validate_geometry(artifact, expected)
        artifact, expected = fixture()
        artifact["records"] = [r for r in artifact["records"] if r["domain"] != "forward_graph"]
        with self.assertRaises(ValueError):
            validate_geometry(artifact, expected)

    def test_stale_expected_membership_and_missing_source_role_fail(self):
        """Stale expected membership and missing source role fail."""
        artifact, expected = fixture()
        expected["participants"].reverse()
        with self.assertRaises(ValueError):
            validate_geometry(artifact, expected)
        artifact, expected = fixture()
        expected["sources"] = [source for source in expected["sources"] if source["layer"] != 7]
        with self.assertRaises(ValueError):
            validate_geometry(artifact, expected)
        artifact, expected = fixture()
        expected["sources"].pop()
        with self.assertRaises(ValueError):
            validate_geometry(artifact, expected)


if __name__ == "__main__":
    unittest.main()
