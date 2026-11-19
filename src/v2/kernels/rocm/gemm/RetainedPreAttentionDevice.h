/**
 * @file RetainedPreAttentionDevice.h
 * @brief QFN gfx906 pre-attention POST/HC and prepared Q8 projection ABI.
 *
 * The caller retains every original buffer and pre-resolved numerical source.
 * The alternate operation borrows them on a single exact native stream.
 */
#pragma once
#include "kernels/rocm/gemm/ROCmFullNativeHCFFN.h"
#include <hip/hip_runtime_api.h>

namespace llaminar2::rocm::pre_attention_probe {
/** @brief Explicit observed source role program for the local early Q8 cohort. */
enum class EarlyQ8Geometry : std::uint32_t { GDN };
/** @brief A previously published trunk or a POST write owned by this retained operation. */
enum class TrunkSource : std::uint32_t { PreviousWrite, PublishedTrunk };
/** @brief Retain the original floating subgroup unless the prepared half leaf is selected explicitly. */
enum class HalfGDNPolicy : std::uint32_t { SeparateOriginalSubgroup, RetainedPreparedFP16RNE };

/** @brief Borrow original, fully prepared row-major FP16 weights and their raw FP32 publications. */
struct PreparedHalfMatrix
{
    const std::uint16_t* words = nullptr;
    std::size_t bytes = 0;
};

/** @brief Exact native FP16 GDN H12 projections and their raw FP32 destinations. */
struct PreparedHalfGDN
{
    PreparedHalfMatrix alpha;
    PreparedHalfMatrix beta;
    float* raw_alpha = nullptr;
    float* raw_beta = nullptr;
};

/** @brief Immutable borrowed prepared matrix and original public FP32 destination. */
struct Projection {
    DeviceNativeQuantisedMatrixDesc weights;
    float* output;
};

/** @brief POST input, exact original HC prepared layout, and typed early projection tuple. */
struct Invocation {
    FullNativeHCFFNPrefixArgs prefix;
    Projection projections[3];
    EarlyQ8Geometry geometry;
    void* scratch;
    std::size_t scratch_bytes;
    TrunkSource trunk_source = TrunkSource::PreviousWrite;
    HalfGDNPolicy half_policy = HalfGDNPolicy::SeparateOriginalSubgroup;
    PreparedHalfGDN half{};
};

/** @brief Original captured operation's lowered instance identity and observed physical resources. */
struct Admission {
    Invocation invocation;
    const void* function;
    hipFuncAttributes resources;
    int device;
};

/** @return The exact compact original HC state and weighted join byte extent. */
std::size_t scratchBytes() noexcept;
/** @brief Resolve the physical gfx906 kernel and validate only original declared projections. */
hipError_t prepare(const Invocation&,int device,Admission&);
/** @brief Submit this cold-bound invocation on its exact nondefault stream. */
hipError_t launch(const Admission&,hipStream_t) noexcept;
}
