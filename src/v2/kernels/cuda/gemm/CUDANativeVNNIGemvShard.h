/**
 * @file CUDANativeVNNIGemvShard.h
 * @brief Internal ABI between CUDA NativeVNNI GEMV dispatch and eight shards.
 *
 * The generated M=1/grouped policy is data, while physical template ownership
 * is build topology.  This ABI keeps those concerns separate: policy refreshes
 * replace one generated include, and eight fixed source shards continue to own
 * exactly two codebooks apiece.  Callers outside the dispatcher must use the
 * ordinary public CUDA NativeVNNI entrypoints.
 */

#pragma once

#include "CUDADeviceWorkspace.h"

#include <cstddef>
#include <cstdint>

#define LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL_IMPL(NAME, INDEX) \
    NAME##_shard_##INDEX
#define LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(NAME, INDEX) \
    LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL_IMPL(NAME, INDEX)

#define LLAMINAR_DECLARE_CUDA_NVNNI_GEMV_SHARD(INDEX)                   \
    extern "C" bool LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaNativeVNNIGemvTuned_supportsCodebook, INDEX)(uint8_t);      \
    extern "C" bool LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaNativeVNNIGemvTuned_queryGeneratedDispatch, INDEX)(        \
        uint8_t, int, int, int, int, int *, int *, int *, int *);       \
    extern "C" bool LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaNativeVNNIGemvTuned_queryCanonicalM1Schedule, INDEX)(      \
        uint8_t, int, int, int, int *, int *);                          \
    extern "C" double LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(               \
        cudaNativeVNNIGemvTuned_measureGeneratedDispatchNs, INDEX)(    \
        uint8_t, int, int, const int *, const int *, int, int,          \
        uint64_t *);                                                    \
    extern "C" bool LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaNativeVNNIGemvTuned_fp32, INDEX)(                           \
        const int8_t *, const uint8_t *, const uint16_t *,              \
        const uint16_t *, const uint32_t *, float *, const float *,     \
        int, int, float, float, const float *, const float *, uint8_t,  \
        int, void *, CUDAGemvContext *, CUDARowMajorWeights **);        \
    extern "C" bool LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaNativeVNNIGemvTuned_small_m_fp32, INDEX)(                   \
        const int8_t *, const uint8_t *, const uint16_t *,              \
        const uint16_t *, const uint32_t *, float *, const float *,     \
        int, int, int, float, float, const float *, const float *,      \
        uint8_t, int, void *, CUDAGemvContext *,                        \
        CUDARowMajorWeights **);                                        \
    extern "C" bool LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaNativeVNNIInitIQGridTables_tuned, INDEX)();                 \
    extern "C" void LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaNativeVNNIGemvSweep_setConfig, INDEX)(                      \
        int, int, int, int, int, int, int, int);                        \
    extern "C" void LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaNativeVNNIGemvSweep_clearConfig, INDEX)();                  \
    extern "C" void LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaNativeVNNIGemvSweep_setGroupedRows, INDEX)(int);            \
    extern "C" void LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaNativeVNNIGroupedVerifier_setTensorCoreOverride, INDEX)(    \
        int);                                                           \
    extern "C" bool LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaNativeVNNIGemvSweep_isActive, INDEX)();                     \
    extern "C" void LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaNativeVNNIGemvTuned_clearStaticState, INDEX)();             \
    extern "C" CUDAGemvContext *LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(     \
        cudaGemvContext_create, INDEX)(int);                            \
    extern "C" void LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaGemvContext_destroy, INDEX)(CUDAGemvContext *);             \
    extern "C" void LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaGemvContext_bindWorkspace, INDEX)(                          \
        CUDAGemvContext *, float *, size_t);                            \
    extern "C" CUDARowMajorWeights *                                  \
        LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                               \
            cudaRowMajorWeights_create, INDEX)(                         \
            const uint8_t *, const uint16_t *, const uint16_t *,        \
            const uint32_t *, int, int, int, int, void *);              \
    extern "C" CUDARowMajorWeights *                                  \
        LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                               \
            cudaRowMajorWeights_createForCodebook, INDEX)(              \
            const uint8_t *, const uint16_t *, const uint16_t *,        \
            const uint32_t *, int, int, uint8_t, int, void *);          \
    extern "C" bool LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaNativeVNNIGemvTuned_policyRequiresRowMajor, INDEX)(         \
        uint8_t, int, int);                                             \
    extern "C" void LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaRowMajorWeights_destroy, INDEX)(CUDARowMajorWeights *);     \
    extern "C" void LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config, INDEX)(   \
        int);                                                           \
    extern "C" int LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                   \
        cudaNativeVNNIGemvTuned_getDecodeEquivalentM1Config, INDEX)(); \
    extern "C" void LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                  \
        cudaNativeVNNIGemvTuned_setSerialPartitionN, INDEX)(int);       \
    extern "C" int LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(                   \
        cudaNativeVNNIGemvTuned_getSerialPartitionN, INDEX)()

LLAMINAR_DECLARE_CUDA_NVNNI_GEMV_SHARD(0);
LLAMINAR_DECLARE_CUDA_NVNNI_GEMV_SHARD(1);
LLAMINAR_DECLARE_CUDA_NVNNI_GEMV_SHARD(2);
LLAMINAR_DECLARE_CUDA_NVNNI_GEMV_SHARD(3);
LLAMINAR_DECLARE_CUDA_NVNNI_GEMV_SHARD(4);
LLAMINAR_DECLARE_CUDA_NVNNI_GEMV_SHARD(5);
LLAMINAR_DECLARE_CUDA_NVNNI_GEMV_SHARD(6);
LLAMINAR_DECLARE_CUDA_NVNNI_GEMV_SHARD(7);

#undef LLAMINAR_DECLARE_CUDA_NVNNI_GEMV_SHARD
