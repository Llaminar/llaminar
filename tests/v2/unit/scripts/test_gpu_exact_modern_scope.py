"""Device-free scope, resume and additive GPU exact-key publication regressions.

These tests execute only Python policy composition with authored observations.
They neither discover hardware nor compile generated selectors or run trainers.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
KERNELS = ROOT / "tests/v2/performance/kernels"
sys.path.insert(0, str(KERNELS))
from native_quantised_dispatch.corpus import GenericDomain
from native_quantised_dispatch.profiles import MeasurementProfile
from native_quantised_dispatch.schema import AspectBucket, Backend, ExecutionMode, SemanticContract
from native_quantised_dispatch.segmented_policy import GenericDispatchRule
from native_quantised_dispatch.shape_manifest import load_shape_manifest


def load(name, relative):
    """Load the actual owned Python entrypoint, without executing its CLI."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


public = load("modern_scope_public", "scripts/native_quantised/exact_refresh.py")
cuda = load("modern_scope_cuda", "tests/v2/performance/kernels/cuda/gemm/analyze_cuda_native_quantised_decode_trainer.py")
rocm = load("modern_scope_rocm", "tests/v2/performance/kernels/rocm/analyze_rocm_native_quantised_decode_trainer.py")


class GPUExactModernScopeTest(unittest.TestCase):
    def test_new_public_jobs_are_modern_and_do_not_start_parked_dense(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(public.main(["--backend", "all", "--output-dir", "/unused", "--dry-run"]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(set(plan["model_scope"].values()), {"modern"})
        self.assertEqual(set(plan["production_shapes"].values()), {105})
        text = json.dumps(plan["commands"])
        self.assertNotIn("production_dense_prefill_sweep", text)
        self.assertIn("production_moe_prefill_sweep", text)
        cpu_command = plan["commands"][0]
        self.assertEqual(cpu_command[cpu_command.index("--model-scope") + 1], "modern")

    def test_saved_historical_scope_is_preserved_and_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = public.parser().parse_args(["--backend", "all", "--output-dir", directory, "--resume"])
            root = Path(directory)
            for backend, relative in (("cpu", "cpu-exact-plan.json"),
                                      ("cuda", "decode/m1/collection-contract.json"),
                                      ("rocm", "decode/rows/collection-contract.json")):
                path = root / backend / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
                self.assertEqual(public.model_scope(args, backend), "all-retained")
                self.assertEqual(len(public.decode_shape_names(public.model_scope(args, backend))), 131)
            args.model_scope = "modern"
            with self.assertRaisesRegex(ValueError, "new output directory"):
                public.model_scope(args, "cuda")

    def test_modern_contract_and_conflicting_resumed_scopes_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            args = public.parser().parse_args(["--backend", "cuda", "--output-dir", directory, "--resume"])
            root = Path(directory) / "cuda/decode"
            for phase in ("m1", "rows"):
                path = root / phase / "collection-contract.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"model_scope": "modern"}))
            self.assertEqual(public.model_scope(args, "cuda"), "modern")
            (root / "rows/collection-contract.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "disagree"):
                public.model_scope(args, "cuda")

    def test_moe_original_manifest_is_not_relabelled_or_mistaken_for_old_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            args = public.parser().parse_args(["--backend", "rocm", "--output-dir", directory, "--resume"])
            corpus = Path(directory) / "rocm/moe-prefill/corpus.json"
            corpus.parent.mkdir(parents=True)
            original = '{"trainer_provenance":{"original":"unchanged"}}'
            corpus.write_text(original)
            self.assertEqual(public.model_scope(args, "rocm"), "all-retained")
            public.write_gpu_selection(args, "rocm", "modern")
            self.assertEqual(public.model_scope(args, "rocm"), "modern")
            self.assertEqual(corpus.read_text(), original)
            contract = Path(directory) / "rocm/decode/m1/collection-contract.json"
            contract.parent.mkdir(parents=True)
            contract.write_text('{"model_scope":"modern"}')
            self.assertEqual(public.model_scope(args, "rocm"), "modern")
            contract.write_text('{}')
            with self.assertRaisesRegex(ValueError, "disagree"):
                public.model_scope(args, "rocm")

    def test_current_routed_moe_cases_all_have_modern_source_owners(self):
        from native_quantised_dispatch.modern_sweep_scope import is_modern_qwen_family
        from native_quantised_dispatch.qwen_moe_gguf_patterns import qwen_moe_prefill_mixtures, qwen_moe_routed_prefill_cases
        mixtures = qwen_moe_prefill_mixtures()
        self.assertTrue(mixtures)
        self.assertTrue(all(is_modern_qwen_family(use.release_id) for row in mixtures for use in row.uses))
        modern_keys = {row.overlay_key for row in mixtures if any(is_modern_qwen_family(use.release_id) for use in row.uses)}
        cases = qwen_moe_routed_prefill_cases()
        self.assertTrue(cases)
        self.assertTrue(all(set(case.mixture_overlay_keys) <= modern_keys for case in cases))

    def test_cpu_reuse_root_forwards_and_unprovable_gpu_moe_reuse_rejects(self):
        args = public.parser().parse_args(["--backend", "cpu", "--output-dir", "/unused",
            "--reuse-completed-root", "/original/provenance", "--dry-run"])
        command = public.cpu_command(args)
        self.assertEqual(command[command.index("--reuse-completed-root") + 1], "/original/provenance")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(public.main(["--backend", "cuda", "--surface", "decode",
                "--output-dir", "/unused", "--dry-run", "--reuse-completed-root", "/original/provenance"]), 0)
        with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                public.main(["--backend", "cuda", "--output-dir", "/unused", "--dry-run",
                             "--reuse-completed-root", "/original/provenance"])

    def test_cuda_m1_refresh_retains_unrelated_exact_keys_and_original_bytes(self):
        manifest = load_shape_manifest()
        shapes = [row for row in manifest.shapes if row.exact_overlay][:2]
        first = cuda.FastEntry(codebook=0, execution_mode=ExecutionMode.GRAPH_CAPTURED,
            n=shapes[0].n, k=shapes[0].k, family="direct", tile_n=32, cpt=1,
            target_waves=0, min_kgroups_per_cta=0, max_kb=0, force_two_phase=0, exact_kb=0,
            candidate_id="cuda.native_quantised.decode.fast_m1.direct.tn32.cpt1",
            shape_name=shapes[0].name, max_surface_regret=0., max_cv=0.)
        unrelated = replace(first, n=shapes[1].n, k=shapes[1].k, shape_name=shapes[1].name)
        rule = GenericDispatchRule(domain=GenericDomain(
            backend=Backend.CUDA, architecture_class="sm_86", semantic_contract=SemanticContract.FAST,
            operation_kind="NativeQuantisedDecodeProjection", bundle_signature="single",
            prepared_family_id="NativeQuantised_cuda_CB0", packing_abi="native-quantised-cuda-cb0-v1",
            runtime_codebook_id=0, execution_mode=ExecutionMode.GRAPH_CAPTURED, m=1,
            aspect_bucket=AspectBucket.BALANCED, all_aspects=True), predicates=(),
            candidate_id=first.candidate_id, arithmetic_fingerprint="sha256:fixture",
            development_shape_groups=("a",), development_max_regret=0.,
            development_p95_regret=0., development_mean_regret=0.)
        base = cuda.generate_include([first, unrelated], [replace(rule,
            domain=replace(rule.domain, aspect_bucket=aspect)) for aspect in AspectBucket],
            grouped_entries=[cuda.GroupedEntry(codebook=0, execution_mode=ExecutionMode.GRAPH_CAPTURED,
                m=2, n=first.n, k=first.k, kernel="dp4a_rows", grouped_rows=2,
                candidate_id="cuda.native_quantised.verifier.dp4a_rows.r2",
                shape_name=first.shape_name, max_surface_regret=0., max_cv=0.)],
            corpus_digest="sha256:original", registry_digest="sha256:registry",
            profile=MeasurementProfile.PRODUCTION)
        fresh = replace(first, family="wide", tile_n=128,
                        candidate_id="cuda.native_quantised.decode.fast_m1.wide.tn128.cpt1")
        refreshed = cuda.refresh_exact_m1_dispatch(base, [fresh], corpus_digest="sha256:new",
            registry_digest="sha256:registry", manifest=manifest)
        self.assertIn(cuda._entry_line(unrelated), refreshed)
        self.assertNotIn(cuda._entry_line(first), refreshed)
        self.assertIn(cuda._entry_line(fresh), refreshed)
        generic = "\n    const long long work_items ="
        self.assertEqual(refreshed[refreshed.index(generic):], base[base.index(generic):])
        with self.assertRaisesRegex(ValueError, "canonical|sorted"):
            cuda.refresh_exact_m1_dispatch(base.replace(cuda._entry_line(first),
                cuda._entry_line(first) + "\n" + cuda._entry_line(first)), [fresh],
                corpus_digest="sha256:new", registry_digest="sha256:registry", manifest=manifest)

    def test_rocm_completeness_uses_declared_subset_without_weakening_matrix_gates(self):
        manifest = load_shape_manifest()
        shape = next(row for row in manifest.shapes if row.exact_overlay)
        spec, mode = SimpleNamespace(label="inert-format"), ExecutionMode.EAGER
        fast = [SimpleNamespace(shape_name=shape.name, aggregate_n=shape.n, k=shape.k,
            source_format=spec.label, execution_mode=mode, semantic_contract=SemanticContract.FAST, m=1)]
        gate_names = ("require_canonical_alias_coverage", "require_candidate_matrix_complete",
                      "require_registry_candidate_coverage")
        with ExitStack() as stack:
            gates = [stack.enter_context(patch.object(rocm, name)) for name in gate_names]
            stack.enter_context(patch.object(rocm, "FORMAT_SPECS", (spec,)))
            stack.enter_context(patch.object(rocm, "ExecutionMode", (mode,)))
            rocm.validate_fast_exact_complete(fast, manifest, shape_names=(shape.name,))
            for gate in gates: gate.assert_called_once()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                rocm.validate_fast_exact_complete(fast, manifest)
        verifier = [SimpleNamespace(**(vars(fast[0]) | {
            "m": 2, "semantic_contract": SemanticContract.VERIFIER_SERIAL_M1_BITWISE}))]
        surface = lambda name: (SemanticContract.VERIFIER_SERIAL_M1_BITWISE, 2, name, spec.label, mode)
        plan = SimpleNamespace(expected_decode_surfaces=lambda *a, **kw: (surface(shape.name), surface("other")))
        with ExitStack() as stack:
            for name in ("require_canonical_alias_coverage", "require_verifier_m_matrix", "require_candidate_matrix_complete"):
                stack.enter_context(patch.object(rocm, name))
            rocm.validate_verifier_complete(verifier, manifest, plan, shape_names=(shape.name,))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                rocm.validate_verifier_complete(verifier, manifest, plan)


if __name__ == "__main__":
    unittest.main()
