/**
 * @file ROCmNativeVNNIGemmShard.h
 * @brief Internal ABI between the ROCm NativeVNNI GEMM dispatcher and shards.
 *
 * The production NativeVNNI prefill implementation instantiates a substantial
 * template surface for every quantization codebook. Keeping all formats in one
 * HIP translation unit serialized several minutes of LLVM optimization. The
 * eight physical shards declared here each own two codebooks and can therefore
 * compile concurrently. The public backend ABI remains centralized in
 * ROCmNativeVNNIGemmDispatch.cpp.
 *
 * These functions are internal build artifacts. Callers outside the dispatcher
 * must continue to use rocmGemm_native_vnni_fp32_with_policy() (or the
 * self-describing compatibility wrapper) and rocmInitIQGridTables_gemm().
 */

#pragma once

#include <cstddef>
#include <cstdint>

using ROCmNativeVNNIGemmShardFn = bool (*)(
    const int8_t *d_A_int8,
    const uint8_t *d_payload,
    const void *d_block_scales,
    const void *d_block_mins,
    const void *d_block_emins,
    float *d_output,
    const float *d_scales_A,
    const float *d_scales_A_blockwise,
    int M,
    int N,
    int K,
    uint8_t codebook_id,
    uint8_t arithmetic_policy_codebook_id,
    int device_id,
    void *stream);

using ROCmNativeVNNIGridInitShardFn = bool (*)(
    int device_id,
    const void *h_iq3s_grid,
    const void *h_iq3xxs_grid,
    const void *h_iq2s_grid,
    const void *h_iq2xs_grid,
    const void *h_iq2xxs_grid,
    const void *h_iq1s_grid);

/**
 * @brief Publish the concrete prefill template selected by one host launch.
 *
 * The dispatcher stores this capture-time diagnostic in thread-local host
 * state. It never participates in graph replay or device execution; its sole
 * purpose is to let trainer and PerfStats tooling turn an AUTO decision into a
 * concrete, reproducible launch tuple without reconstructing policy logic.
 */
extern "C" void rocmNativeVNNIPrefill_recordLastLaunchSelection(
    uint8_t codebook_id,
    int n_tile,
    int m_tile,
    int min_blocks,
    int unroll,
    bool full_tiles,
    const void *function,
    int block_threads);

/**
 * @brief Query the last concrete NativeVNNI prefill launch on this host thread.
 *
 * @param codebook_id Receives the selected runtime codebook.
 * @param n_tile Receives the output-column tile width.
 * @param m_tile Receives the row tile height.
 * @param min_blocks Receives the compile-time launch-bounds occupancy target.
 * @param unroll Receives the compile-time K-group unroll factor.
 * @return True after at least one launch was published on the calling thread.
 */
extern "C" bool rocmNativeVNNIPrefill_getLastLaunchSelection(
    uint8_t *codebook_id,
    int *n_tile,
    int *m_tile,
    int *min_blocks,
    int *unroll,
    bool *full_tiles);

/**
 * @brief Query immutable compiler resources for the last selected kernel.
 *
 * The launch publisher stores only the function identity and block geometry.
 * Resource inspection occurs when trainer or diagnostic code asks for it, so
 * ordinary graph construction does not pay a `hipFuncGetAttributes` or
 * occupancy-query cost for every projection.
 *
 * @param registers_per_thread Receives allocated VGPR count.
 * @param local_memory_bytes_per_thread Receives compiler scratch bytes; a
 *        non-zero value makes a tuning candidate ineligible for promotion.
 * @param static_shared_memory_bytes Receives static LDS bytes.
 * @param max_threads_per_block Receives the compiler block-size ceiling.
 * @param max_active_blocks_per_sm Receives theoretical active blocks for the
 *        exact captured block geometry.
 * @return True when launch identity exists and both HIP queries succeed.
 */
extern "C" bool rocmNativeVNNIPrefill_getLastLaunchResources(
    int *registers_per_thread,
    size_t *local_memory_bytes_per_thread,
    size_t *static_shared_memory_bytes,
    int *max_threads_per_block,
    int *max_active_blocks_per_sm);

#define LLAMINAR_DECLARE_ROCM_NVNNI_GEMM_SHARD(INDEX)                   \
    extern "C" bool rocmGemm_native_vnni_fp32_shard_##INDEX(           \
        const int8_t *, const uint8_t *, const void *, const void *,    \
        const void *, float *, const float *, const float *, int, int,  \
        int, uint8_t, uint8_t, int, void *);                            \
    extern "C" bool rocmInitIQGridTables_gemm_shard_##INDEX(            \
        int, const void *, const void *, const void *, const void *,     \
        const void *, const void *)

LLAMINAR_DECLARE_ROCM_NVNNI_GEMM_SHARD(0);
LLAMINAR_DECLARE_ROCM_NVNNI_GEMM_SHARD(1);
LLAMINAR_DECLARE_ROCM_NVNNI_GEMM_SHARD(2);
LLAMINAR_DECLARE_ROCM_NVNNI_GEMM_SHARD(3);
LLAMINAR_DECLARE_ROCM_NVNNI_GEMM_SHARD(4);
LLAMINAR_DECLARE_ROCM_NVNNI_GEMM_SHARD(5);
LLAMINAR_DECLARE_ROCM_NVNNI_GEMM_SHARD(6);
LLAMINAR_DECLARE_ROCM_NVNNI_GEMM_SHARD(7);

#undef LLAMINAR_DECLARE_ROCM_NVNNI_GEMM_SHARD
