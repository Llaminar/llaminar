/**
 * @file ROCmMoEPrefillArithmetic.h
 * @brief Shared serial reduction authority for grouped Dot4 and native WMMA.
 *
 * Capture resolves source-format partitions from the original full projection.
 * Every physical decoder and down slice borrows that same immutable policy;
 * choosing another dot engine must never change floating-point fold boundaries.
 */
#pragma once
#include "kernels/common/DeviceNativeQuantisedMatrixDesc.h"
#include <cstdint>
#if defined(__HIPCC__)
#define LLAMINAR_MOE_ARITHMETIC_HD __host__ __device__
#else
#define LLAMINAR_MOE_ARITHMETIC_HD
#endif

/**
 * @brief Capture-time serial-M1 split-K policy carried into grouped prefill.
 *
 * A NativeQuantised codebook and matrix geometry select a generated serial-decode
 * K-partition count. That count changes FP32 arithmetic because each partition
 * starts from zero and an ordered reducer then adds the independently rounded
 * partials. Grouped prefill therefore carries the complete codebook-indexed
 * policy by value in its graph-captured kernel arguments. Mixed-codebook down
 * tables can select the correct entry on device without consulting the host.
 */
struct MoEPrefillSerialM1KPartitionPolicy
{
    uint8_t by_codebook[32] = {};

    /** Return the captured partition count, or zero for an invalid codebook. */
    LLAMINAR_MOE_ARITHMETIC_HD int forCodebook(uint8_t codebook_id) const
    {
        return codebook_id < 32 ? static_cast<int>(by_codebook[codebook_id]) : 0;
    }

    /**
     * @brief Select the source-format reduction tree for one execution matrix.
     *
     * Promoted CPU-native weights use an expanded device decoder but must keep
     * the compact source format's serial-M1 partition boundaries. Canonical
     * device formats without explicit provenance retain their execution id.
     */
    LLAMINAR_MOE_ARITHMETIC_HD int forDescriptor(
        const llaminar2::DeviceNativeQuantisedMatrixDesc& desc) const
    {
        uint8_t policy_codebook = desc.source_identity_present
                                      ? desc.source_codebook_id
                                      : desc.codebook_id;
        // Q8_1 and Q8_K prepare to the same signed-INT8 arithmetic as Q8_0.
        if (policy_codebook == 20 || policy_codebook == 21)
            policy_codebook = 19;
        return forCodebook(policy_codebook);
    }
};


#undef LLAMINAR_MOE_ARITHMETIC_HD
