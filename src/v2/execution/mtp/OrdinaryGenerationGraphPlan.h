/**
 * @file OrdinaryGenerationGraphPlan.h
 * @brief Exact retained ordinary-generation owners shared by setup and PMA input.
 *
 * Sampler recordings imported into a parent own no native executable. CUDA
 * needs the resident-input forward and one conditional parent; HIP additionally
 * instantiates initial sampling and the complete forward/sampler transaction.
 * A retained speculative family already accounts for the shared parent/ticket
 * owner, so it must not be charged twice. This is a BOM, never a live ledger.
 */
#pragma once
#include "backends/DeviceId.h"
#include "kernels/common/SparseQueryExecutionProgram.h"
#include <cstddef>
#include <stdexcept>

namespace llaminar2
{
/** @brief Immutable native-owner geometry for scalar ordinary generation. */
struct OrdinaryGenerationGraphPlan
{
    static constexpr std::size_t prefill_sampler = 0;
    static constexpr std::size_t decode_sampler = 1;
    static constexpr std::size_t hosted_transaction = 2;
    static constexpr std::size_t child_recordings = 3;

    /** @return Complete scalar no-sampling transaction owners for a native singleton PP stage.
     * Child model/transport recordings are imported, never independently instantiated. */
    static std::size_t pipelineForwardOnlyExecutableCount(std::size_t query_programs = 1u)
    {
        if (query_programs == 0 || query_programs > kSparseQueryExecutionProgramCount)
            throw std::invalid_argument("Pipeline forced graph inventory has an unknown query-program family");
        return query_programs;
    }

    /** @return Extra general executable owners beyond the existing forward/MTP BOM.
     *  @throws std::invalid_argument for a device without native GPU graphs. */
    static std::size_t additionalExecutableCount(DeviceId device, bool retained_mtp,
        std::size_t query_programs = 1u)
    {
        if (!device.is_gpu())
            throw std::invalid_argument("Ordinary generation graph inventory requires a GPU");
        if (query_programs == 0 || query_programs > kSparseQueryExecutionProgramCount)
            throw std::invalid_argument("Ordinary graph inventory has an unknown query-program family");
        // Each program owns its resident-input forward, parent (shared with
        // MTP when retained), and HIP initial/complete hosted transaction.
        return query_programs * (1u + (retained_mtp ? 0u : 1u) + (device.is_rocm() ? 2u : 0u));
    }
};
}
