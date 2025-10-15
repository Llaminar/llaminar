# CMakeLists.txt Reorganization Plan
**Date**: 2025-01-15
**Author**: Code Cleanup Task
**Status**: Planning Phase

## Objective
Reorganize CMakeLists.txt test definitions into clearly labeled sections to facilitate removal of 36 obsolete tests.

## Current State
- 1462 lines total
- Lines 1-315: Project setup, dependencies, core library (NO CHANGES NEEDED)
- Lines 316-1462: Test definitions scattered without clear organization
- 36 obsolete tests mixed throughout, hard to identify

## New Organization Structure

### Section 1: Project Setup & Core Library (Lines 1-315)
**NO CHANGES** - Keep existing project configuration, dependencies, and core library definition.

### Section 2: Test Utilities & Helpers (Lines 316-360)
- `add_test_with_debug_logging` function
- `add_llaminar_mpi_test` function
- `test_logging_bootstrap` object library

### Section 3: CORE UNIT TESTS (Non-MPI, Passing)
**Status**: ✅ ALL PASSING
Tests that validate fundamental functionality without MPI:
- `test_weight_role_classification` ✅
- `test_rmsnorm_core_correctness` ✅
- `test_softmax_core_correctness` ✅
- `test_basic` ✅
- `test_pipeline_factory` ✅
- `test_weight_contracts` ✅
- `test_weight_validation_integration` ✅
- `test_large_matmul_plan` ✅
- `test_numa` ✅
- `test_contract_loading` ✅
- `test_bias_contracts` ✅
- `test_cosma` ✅
- `test_tp_partition` ✅
- `test_dequant` ✅
- `test_dequant_extra` ✅
- `test_dequant_golden` ✅

### Section 4: MPI KERNEL TESTS (Distributed, Passing)
**Status**: ✅ CORE MPI TESTS PASSING
- `test_mpi_rmsnorm_kernel` ✅
- `test_mpi_softmax_correctness` ✅
- `test_mpi_attention_kernel` ✅ (standalone)
- `test_mpi_attention_kernel_clean` ✅

### Section 5: PARITY FRAMEWORK TESTS ⭐ CRITICAL END-TO-END
**Status**: ✅ 100% PASSING - PRODUCTION VALIDATION
These are the MOST IMPORTANT tests - they validate production code against PyTorch/llama.cpp:
- `test_parity_framework` ✅ (OpenBLAS, COSMA, TrueIncremental)
- `test_incremental_snapshot_helper` ✅
- `test_weight_passthrough` ✅
- `test_prefill_providers` ✅

### Section 6: ATTENTION & ROPE TESTS (Passing)
**Status**: ✅ PASSING
- `test_attention_regression` ✅
- `test_attention_real_weights` (Single & Multi) ✅
- `test_attention_bias_validation` ✅
- `test_attention_output_mode_validation` ✅
- `test_attention_golden` ✅

### Section 7: TENSOR PARALLELISM TESTS (Passing)
**Status**: ✅ PASSING
- `test_tp_gemm_correctness` ✅ (all variants: Col/Row, 2/3/4-way)
- `TPGemmCorrectness_Matrix` ✅

### Section 8: MODEL LOADING & QUANTIZATION TESTS (Passing)
**Status**: ✅ PASSING
- `test_model_loader_golden` ✅
- `test_model_loader_alignment` ✅
- `test_model_loader_quant_shard_regression` ✅
- `test_role_tagging_logs` ✅
- `test_model_load_role_tags` ✅
- `test_integration_q4k` ✅
- `test_quant_dequant_correctness` ✅
- `test_quant_shard_cache_stats` ✅
- `test_partial_decode_q4_0` ✅
- `test_partial_decode_multi_formats` ✅
- `test_partial_decode_aggregated` ✅
- `test_partial_decode_k_formats` ✅

### Section 9: BENCHMARK EXECUTABLES (Commented Out)
- `bench_tp_output_projection`
- `bench_rmsnorm`
- `bench_attention`
- `bench_softmax`
- `bench_softmax_distributed`
- `bench_rope`

### Section 10: FIXTURE TESTS & MODEL DOWNLOADS
- `FetchTestModels` fixture
- `mark_requires_models()` function

### Section 11: OUTPUT DIRECTORIES & CUSTOM TARGETS
- Output directory configuration
- `run_tests` custom target
- Global test environment injection

### Section 12: CONFIGURATION SUMMARY
- Build configuration summary printout

---

## OBSOLETE TESTS SECTION (TO BE REMOVED IN NEXT PASS)

### ❌ DEPRECATED TESTS - FAILED DUE TO ARCHITECTURE REFACTORING
**Total**: 36 tests
**Reason**: Superseded by parity framework, deprecated APIs, wrong snapshot methodology

#### Attention Tests (19 tests)
1. `test_weight_slice_correctness` - Old slicing API
2. `test_attention_shard_correctness` + variants (5 variants) - Old sharding API
3. `test_attention_missing_reduction_negative` - Old negative test
4. `test_attention_stage_contracts` + variants (7 variants) - Contract framework deprecated
5. `test_attention_micro` - Superseded by parity framework
6. `test_attention_primitives` - Standalone primitives now in parity
7. `test_k_gathering_and_rope_gqa` (Single + Multi) - Integrated into kernel tests
8. `test_rope_recurrence_correctness` - Standalone RoPE test deprecated
9. `test_rope_corruption` - Specific edge case, now covered

#### Integration Tests (3 tests)
10. `test_qwen_integration` (QwenIntegrationTests) - Wrong snapshot methodology
11. `test_incremental_generation` - Old incremental methodology
12. `test_incremental_generation_multi` - Old multi-rank incremental

#### COSMA Tests (5 tests)
13. `test_cosma_precision` - Precision edge cases superseded
14. `test_cosma_minimal` - Minimal test superseded by parity
15. `test_cosma_prefill_manager` - Manager test superseded
16. `test_cosma_prefill_attention_integration` - Integration superseded
17. `test_cosma_prefill_manager_orientation_autofix` - Autofix test deprecated
18. `test_cosma_prefill_manager_stats` - Stats test deprecated
19. `test_cosma_fused_rmsnorm_qkv` - Fused op test superseded

#### Linear/Embedding Tests (4 tests)
20. `test_embedding_standalone` - Standalone embedding deprecated
21. `test_embedding_orientation` - Orientation test superseded
22. `test_mpi_linear_kernel` - Old linear kernel API
23. `test_linear_orientation_correctness` - Old orientation API
24. `test_mpi_embedding_kernel` (Normal + Extended) - Old embedding kernel

#### Incremental Decode Tests (4 tests)
25. `test_incremental_decode_correctness` (Single + Multi) - Old methodology
26. `test_kv_cache_growth` - Old KV cache test

#### TP Tests (3 tests)
27. `test_tp_splitters` - Splitter tests integrated
28. `test_attention_tp_correctness` - Old TP attention API
29. `test_tp_generic_gemm_executor` - Old generic executor
30. `test_attention_tp_sim_correctness` - Old TP simulation
31. `test_tp_output_projection_executor` - Old output projection

#### Pipeline Tests (2 tests)
32. `test_mpi_transformer_pipeline` (QwenPipelineTest) - Old pipeline API
33. `test_adaptive_matmul` - Old adaptive matmul

#### RMSNorm Tests (2 tests)
34. `test_rmsnorm_shard_stats` - Shard stats superseded
35. `test_rmsnorm_sharded_gamma` - Sharded gamma superseded

#### MPI Probe Tests (1 test)
36. `test_mpi_finalize_probe` - Diagnostic probe no longer needed

---

## Implementation Strategy

### Phase 1: Reorganize (This Pass) ✅
1. Read full CMakeLists.txt
2. Extract test definitions into organized sections
3. Add clear comment headers for each section
4. Keep all tests (even obsolete ones) for now
5. Mark obsolete tests with clear `❌ OBSOLETE` comments

### Phase 2: Remove Obsolete Tests (Next Pass)
1. Delete all test definitions marked `❌ OBSOLETE`
2. Delete corresponding test source files from `tests/` directory
3. Rebuild and verify no regressions
4. Re-run test suite to confirm ~85 tests pass

### Phase 3: Verify (Final Pass)
1. Ensure 3 core parity tests still pass 100%
2. Update test count documentation
3. Commit changes with detailed changelog

---

## Success Criteria

✅ CMakeLists.txt is readable with clear section labels
✅ Obsolete tests are clearly marked for easy removal
✅ No functional changes in Phase 1 (all tests still registered)
✅ User can easily identify which tests to remove in Phase 2
