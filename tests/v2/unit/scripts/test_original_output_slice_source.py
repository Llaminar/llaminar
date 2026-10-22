#!/usr/bin/env python3
"""Guard native projection-local source identity without running a backend.

These architecture tests complement the real CPU byte sweep. They do not
certify CUDA/ROCm execution or replace captured all-format/model qualification.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[4]


def source(relative):
    """Read one code-owned source file; no model/runtime payload is opened."""
    return (ROOT / relative).read_text()


def method(text, signature):
    """Return the exact braced method for bounded ownership-order assertions."""
    first = text.index(signature)
    begin = text.index("{", first)
    depth = 1
    cursor = begin + 1
    while depth:
        depth += (text[cursor] == "{") - (text[cursor] == "}")
        cursor += 1
    return text[first:cursor]


def require_rocm_group(test, text):
    """Require the real verifier gate and projection-local explicit source query."""
    test.assertIn("g_rocm_native_vnni_decode_equivalent_scope &&\n                all_projections_native_vnni && m >= 2", text)
    begin = text.index("struct GeneratedGroupPolicy")
    end = text.index("if (!generated_grouping_available)", begin)
    grouping = text[begin:end]
    test.assertIn("auto original_scope = projections[i].beginOriginalOutputSliceScope();", grouping)
    test.assertIn("projections[i].original_output_slice->sourceIdentity().codebook_id", grouping)
    test.assertIn("rocmGemv_native_vnni_query_serial_m1_config_with_policy(", grouping)
    test.assertIn("Ns[i], k, &policy_kb, &policy_target_waves", grouping)
    test.assertIn("groups[static_cast<size_t>(group_index)].push_back(static_cast<int>(i))", grouping)


class OriginalOutputSliceSource(unittest.TestCase):
    """Check the actual backend entry/loop closures and seeded missing-edge negatives."""

    def test_existing_two_int_direction_is_preserved(self):
        for backend, name in (("cpu", "CPUNativeVNNIGemmKernel.h"),
                              ("cuda", "CUDAQuantisedGemmKernel.cpp"),
                              ("rocm", "ROCmQuantisedGemmKernel.cpp")):
            text = source(f"src/v2/kernels/{backend}/gemm/{name}")
            body = method(text, "beginOutputPartitionEquivalenceScope(")
            self.assertIn("serial_partition_columns > actual_output_columns", body)
            self.assertNotIn("OriginalOutputSlice", body)
        floating = source("src/v2/kernels/common/FloatingOutputPartitionScope.h")
        self.assertIn("serial_columns > actual_columns", floating)

    def test_store_is_actual_source_and_prepared_authority(self):
        body = method(source("src/v2/loaders/PreparedWeightStore.cpp"),
                      "OriginalOutputSlice PreparedWeightStore::originalOutputSlice(")
        for fact in ("samePreparedRef", "!binding.immutable", "binding.tensor != &source",
                     "handle->tensor != &source", "handle->device_id != ref.device",
                     "binding.tensor_owner.get() != &source",
                     "handle->prepared_weights->owned_kernel.get() != kernel",
                     "slice.row_start > slice.source_rows - slice.row_count",
                     "slice.col_start != 0", "slice.col_count != slice.source_cols",
                     "slice.expert_ids.empty()", "kernel->get_n()", "kernel->get_k()",
                     "IINT8Unpackable::fromTensor", "exportNativeVNNISourceIdentity",
                     "prepared_source.codebook_id != format->codebook_id",
                     "prepared_source.is_superblock != format->is_superblock"):
            self.assertIn(fact, body)
        for forbidden in ("cuda", "hip", "TransferEngine::", "exportNativeVNNIMatrixDesc(", "raw_data("):
            self.assertNotIn(forbidden, body)
        header = source("src/v2/kernels/common/OriginalOutputSlice.h")
        self.assertLess(header.index("private:"), header.index("OriginalOutputSlice(std::shared_ptr<ITensorGemm>"))
        self.assertIn("friend class PreparedWeightStore", header)
        stage = source("src/v2/execution/compute_stages/stages/FusedQKVGEMMStage.cpp")
        self.assertIn("originalOutputSlice(*refs[member], *tensor) != *facts[member]", stage)
        self.assertIn("mix(slice->recordingIdentity())", stage)
        self.assertIn("facts[member]->retainedEngine()", stage)
        validation = method(stage, "bool FusedQKVGEMMStage::validatePreparedWeights(")
        self.assertIn("cache_resolved_individual_ && fact", validation)
        self.assertIn("kernel == fact->retainedEngine()", validation)
        self.assertIn("ref->kind == fact->preparedKind()", validation)
        self.assertIn("std::shared_ptr<ITensorGemm> engine_owner_", header)
        self.assertIn("std::shared_ptr<TensorBase> source_owner_", header)
        self.assertNotIn("PreparedGemmHandle", header[header.index("private:"):].replace(
            "mutable PreparedGemmHandle/PreparedGemmWeights publication wrapper.", ""))

    def test_cpu_fused_members_keep_explicit_policy_and_physical_extent(self):
        body = source("src/v2/kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h")
        self.assertIn(".policy_n = projections[i].original_output_slice", body)
        self.assertIn("fused_descs[i].policy_n = proj.original_output_slice", body)
        self.assertIn("descriptors[i].policy_n = projection.original_output_slice", body)
        self.assertIn("auto original_scope = projections[i].beginOriginalOutputSliceScope();", body)
        implementation = source("src/v2/kernels/cpu/gemm/CPUNativeVNNIGemv.h")
        self.assertIn("const int policy_n = d.policy_n > 0 ? d.policy_n", implementation)
        self.assertIn("plan.n_chunks = (d.packed->N + 63) / 64", implementation)
        self.assertIn(".policy_n = d.policy_n", implementation)

    def test_cuda_binds_each_retained_plan_and_recorded_leaf(self):
        text = source("src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel.cpp")
        entry = method(text, "bool CUDAQuantisedGemmKernel::multiply_fused_tensor(")
        self.assertLess(entry.index("projection.requireOriginalOutputSlice(k)"), entry.index("DeviceWorkspaceManager *effective_ws"))
        self.assertIn("auto original_scope = binding.projection->beginOriginalOutputSliceScope();", text)
        self.assertEqual(text.count("auto original_scope = binding.projection->beginOriginalOutputSliceScope();"), 2)
        for signature in ("auto &binding = bindings[pi];", "const auto &binding = bindings[static_cast<size_t>(pi)];"):
            tail = text[text.index(signature):]
            self.assertLess(tail.index("beginOriginalOutputSliceScope"), tail.index("binding.n"))
        self.assertIn("proj.name, &proj", text)

    def test_rocm_keeps_verifier_grouping_and_rejects_missing_scope_controls(self):
        text = source("src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp")
        require_rocm_group(self, text)
        for bad in (text.replace("auto original_scope = projections[i].beginOriginalOutputSliceScope();", ""),
                    text.replace("rocmGemv_native_vnni_query_serial_m1_config_with_policy(", "wrong_query(")):
            with self.assertRaises(AssertionError):
                require_rocm_group(self, bad)
        entry = method(text, "bool ROCmQuantisedGemmKernel::multiply_fused_tensor(")
        self.assertLess(entry.index("projection.requireOriginalOutputSlice(k)"), entry.index("requireGPUStream()"))
        self.assertIn("gemv_ok = launch_group(all_indices);", entry)
        self.assertIn("group_Ns.data()", entry)

    def test_floating_declarations_fail_before_any_path_selection(self):
        for backend, filename, prefix in (
            ("cpu", "FloatingPointGemmKernel.h", ""),
            ("cuda", "CUDAFloatingPointGemmKernel.cpp", "CUDAFloatingPointGemmKernel::"),
            ("rocm", "ROCmFloatingPointGemmKernel.cpp", "ROCmFloatingPointGemmKernel::")):
            text = source(f"src/v2/kernels/{backend}/gemm/{filename}")
            for name in ("multiply_fused_tensor", "multiply_fused_verifier_rows_decode_equivalent"):
                entry = method(text, f"bool {prefix}{name}(")
                first = entry[entry.index("{")+1:].lstrip()
                self.assertTrue(first.startswith("for (const auto &projection : projections)"))
                self.assertIn("Floating fused original-output-slice arithmetic is not implemented", first)
                self.assertNotIn("beginOutputPartitionEquivalenceScope(", first)


if __name__ == "__main__":
    unittest.main()
