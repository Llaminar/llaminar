/**
 * @file CUDARetainedHCPrefix.h
 * @brief Prepared CUDA operands for the post-attention part of the resident QFN FFN.
 *
 * The graph supplies the original tensor and workspace bindings. The prefix
 * borrows them until its enclosing native program retires. Its mixed output
 * is the original local router and shared expert input.
 */
#pragma once

#include "kernels/common/HCDownOutputPublicationArgs.h"
#include "kernels/common/Q8ResidualCorrection.h"
#include "kernels/cuda/gemm/CUDANativeQuantisedPrefillSchedule.h"
#include "kernels/qwen4exp/Qwen4ExpKernels.h"

#include <type_traits>

namespace llaminar2::cuda::moe
{
    /** @brief Mixed HC channels covered by one warp across all four branches. */
    inline constexpr unsigned retained_hc_read_warp_channels = 8;

    /** @brief Original public serial CUDA reduction mode for HC Down. */
    enum class HCDownFold : uint8_t { Direct, OrderedPartitions };

    /** @brief Original post-attention HC operands sealed before recording the resident kernel. */
    struct RetainedHCPrefix
    {
        ::llaminar2::qwen4exp::ResidualWriteArgs write; ///< Original attention output, injection and four-branch residual.
        ::llaminar2::qwen4exp::BranchRMSNormArgs norm; ///< Original fixed 256-lane normalization and retained output.
        DeviceNativeQuantisedMatrixDesc down; ///< Original prepared HC Down weight, with K equal to 4H.
        prefill::CanonicalM1PartitionGeometry down_partitions; ///< Exact CUDA serial M1 FP32 fold.
        HCDownFold down_fold; ///< Original CUDA serial publication mode.
        int8_t* norm_codes; ///< Original persistent Q8 input bank for Down.
        float* norm_scales; ///< Original FP32 K32 scales for Down.
        float* down_partials; ///< Borrowed canonical GEMV partition bank, retired before the FFN.
        float* raw_down; ///< Original public Down values, before the checkpoint activation.
        GPUHCDownOutputView down_publication; ///< Original activated values and paired Q8 Up input.
        GPUQ8ResidualCorrectionArgs up; ///< Original Up weights and independently rounded Q8 terms.
        float* mixed; ///< Original H-wide read for the local router and shared FFN.
        const float* injection_weights = nullptr; ///< Optional original FP32 [4, 4H] injection weights.
        float* injection_output = nullptr; ///< Optional original FP32 [4] injection logits.
        bool has_previous_write = true; ///< True when write publishes norm.input; false for PublishedTrunk.
    };

    static_assert(std::is_trivially_copyable_v<RetainedHCPrefix>);
}
