/**
 * @file CUDANativeQuantisedPrefillShard.h
 * @brief Internal ABI between CUDA NativeQuantised prefill dispatch and its eight shards.
 *
 * Physical template ownership is build topology. Eight fixed source shards
 * each own the codebook group named by prefillShardIndexForCodebook() and are
 * compiled as complete, deterministic modules without nvcc --split-compile.
 * They export disjoint, index-suffixed symbols; the dispatcher in
 * CUDANativeQuantisedPrefillDispatch.cu is the sole owner of the public names.
 * Callers outside the dispatcher must use the public CUDA prefill entrypoints.
 */

#pragma once

#include "CUDANativeQuantisedPrefillDiagnostics.h"
#include "CUDANativeQuantisedPrefillSchedule.h"
#include "kernels/cuda/gemm/CUDADeviceWorkspace.h"

#include <cuda_runtime.h>
#include <cstdint>

#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_SYMBOL_IMPL(NAME, INDEX) \
    NAME##_prefill_shard_##INDEX
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(NAME, INDEX) \
    LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_SYMBOL_IMPL(NAME, INDEX)

/** @brief Declare the complete internal ABI of one physical prefill shard. */
#define LLAMINAR_DECLARE_CUDA_NATIVE_QUANTISED_PREFILL_SHARD(INDEX)                    \
    extern "C" bool LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(       \
        cudaNativeQuantisedPrefill_launch, INDEX)(                                     \
        const int8_t *, const uint8_t *, const uint16_t *, const uint16_t *,          \
        const uint32_t *, float *, const float *, const int32_t *, int, int, int,     \
        float, float, const float *, const float *, uint8_t, int, int,                \
        cudaStream_t, CUDAPrefillContext *);                                          \
    extern "C" bool LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(       \
        cudaNativeQuantisedPrefill_queryLastLaunchResources, INDEX)(                   \
        uint8_t, int, int, int, CUDADensePrefillKernelResources *,                    \
        CUDADensePrefillKernelResources *);                                           \
    extern "C" bool LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(       \
        cudaNativeQuantisedPrefill_queryCandidateResources, INDEX)(                    \
        uint8_t, int, int, int, int, CUDADensePrefillKernelResources *,               \
        CUDADensePrefillKernelResources *,                                            \
        llaminar2::cuda::prefill::PrefillStagingSchedule)

LLAMINAR_DECLARE_CUDA_NATIVE_QUANTISED_PREFILL_SHARD(0);
LLAMINAR_DECLARE_CUDA_NATIVE_QUANTISED_PREFILL_SHARD(1);
LLAMINAR_DECLARE_CUDA_NATIVE_QUANTISED_PREFILL_SHARD(2);
LLAMINAR_DECLARE_CUDA_NATIVE_QUANTISED_PREFILL_SHARD(3);
LLAMINAR_DECLARE_CUDA_NATIVE_QUANTISED_PREFILL_SHARD(4);
LLAMINAR_DECLARE_CUDA_NATIVE_QUANTISED_PREFILL_SHARD(5);
LLAMINAR_DECLARE_CUDA_NATIVE_QUANTISED_PREFILL_SHARD(6);
LLAMINAR_DECLARE_CUDA_NATIVE_QUANTISED_PREFILL_SHARD(7);
