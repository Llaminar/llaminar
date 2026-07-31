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
 * must continue to use rocmGemm_native_vnni_fp32() and
 * rocmInitIQGridTables_gemm().
 */

#pragma once

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

#define LLAMINAR_DECLARE_ROCM_NVNNI_GEMM_SHARD(INDEX)                   \
    extern "C" bool rocmGemm_native_vnni_fp32_shard_##INDEX(           \
        const int8_t *, const uint8_t *, const void *, const void *,    \
        const void *, float *, const float *, const float *, int, int,  \
        int, uint8_t, int, void *);                                     \
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
