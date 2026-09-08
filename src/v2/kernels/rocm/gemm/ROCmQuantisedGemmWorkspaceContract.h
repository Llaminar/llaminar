/**
 * @file ROCmQuantisedGemmWorkspaceContract.h
 * @brief Defines the allocation-free ROCm quantized-GEMM workspace ABI.
 *
 * Runtime graph construction and metadata-only memory admission happen at
 * different lifecycle points, but they must price exactly the same persistent
 * buffer names.  This header is intentionally free of HIP runtime types so the
 * model planner can consume the production kernel contract before weights or a
 * device context exist.
 */

#pragma once

#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "utils/PrefillGraphBucketDefaults.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace llaminar2::rocm::quantized_gemm_workspace
{

/** Maximum split-K partitions addressable by the retained scatter reducer. */
inline constexpr int kScatterKBlockCapacity = 64;

/** Maximum side streams used by one fused small-M projection transaction. */
inline constexpr std::size_t kFusedProjectionStreamCapacity = 4u;

/** Output-column tile used by the device-resident self-reduction counter bank. */
inline constexpr int kSelfReduceOutputTileColumns = 128;

/**
 * @brief Multiply positive workspace dimensions with overflow checking.
 * @param left First cardinality.
 * @param right Second cardinality.
 * @param contribution Human-readable allocation name for diagnostics.
 * @return Exact product.
 * @throws std::invalid_argument when either cardinality is zero.
 * @throws std::overflow_error when the product cannot fit in `size_t`.
 */
inline std::size_t checkedProduct(
    std::size_t left,
    std::size_t right,
    std::string_view contribution)
{
    if (left == 0u || right == 0u)
    {
        throw std::invalid_argument(
            "ROCm quantized-GEMM workspace requires positive " +
            std::string(contribution));
    }
    if (right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::overflow_error(
            "ROCm quantized-GEMM workspace overflow for " +
            std::string(contribution));
    }
    return left * right;
}

/**
 * @brief Declare the stable buffers used by one quantized projection.
 *
 * All GGUF quantization codebooks are repacked to the same NativeVNNI runtime
 * representation, so this contract is intentionally codebook-independent.
 * Buffer names, reduction capacity, verifier-row tiling, and alignment are the
 * ABI consumed by captured graphs; changing any of them changes memory
 * admission as well as kernel binding.
 *
 * @param rows Captured logical row capacity for this projection.
 * @param output_columns Matrix N dimension.
 * @param input_columns Matrix K dimension.
 * @return Exact mergeable production workspace requirements.
 */
inline WorkspaceRequirements projectionRequirements(
    int rows,
    int output_columns,
    int input_columns)
{
    if (rows <= 0 || output_columns <= 0 || input_columns <= 0)
    {
        throw std::invalid_argument(
            "ROCm quantized-GEMM workspace dimensions must all be positive");
    }

    const std::size_t m = static_cast<std::size_t>(rows);
    const std::size_t n = static_cast<std::size_t>(output_columns);
    const std::size_t k = static_cast<std::size_t>(input_columns);
    const std::size_t blocks_per_row = (k + 31u) / 32u;
    const std::size_t matrix_a = checkedProduct(m, k, "activation matrix");
    const std::size_t matrix_c = checkedProduct(m, n, "output matrix");
    const std::size_t block_metadata = checkedProduct(
        m, blocks_per_row, "activation block metadata");

    WorkspaceRequirements requirements;
    requirements.buffers.push_back({
        GemmWorkspaceBuffers::QUANT_A,
        matrix_a * sizeof(std::int8_t), 256, true});
    requirements.buffers.push_back({
        GemmWorkspaceBuffers::SCALES_A,
        m * sizeof(float), 256, true});
    requirements.buffers.push_back({
        GemmWorkspaceBuffers::SCALES_A_BLOCKWISE,
        block_metadata * sizeof(float), 256, true});
    requirements.buffers.push_back({
        GemmWorkspaceBuffers::SUMS_A_BLOCKWISE,
        block_metadata * sizeof(std::int32_t), 256, true});
    requirements.buffers.push_back({
        GemmWorkspaceBuffers::ACC_INT32,
        matrix_c * sizeof(std::int32_t), 256, true});
    requirements.buffers.push_back({
        GemmWorkspaceBuffers::TEMP_A_FP32,
        matrix_a * sizeof(float), 256, true});
    requirements.buffers.push_back({
        GemmWorkspaceBuffers::TEMP_C_FP32,
        matrix_c * sizeof(float), 256, true});

    /*
     * Prompt-M GEMM and decode-equivalent GEMV share one kernel object, but
     * only the latter consumes scatter partials.  Retain one certified tile;
     * larger grouped calls execute several tiles against this same address.
     */
    const std::size_t scatter_rows = static_cast<std::size_t>(
        std::clamp(rows, 1, kDefaultNativeVNNIVerifierRowCapacity));
    std::size_t scatter_elements = checkedProduct(
        static_cast<std::size_t>(kScatterKBlockCapacity),
        scatter_rows,
        "split-K partitions and verifier rows");
    scatter_elements = checkedProduct(
        scatter_elements, n, "split-K output columns");
    requirements.buffers.push_back({
        GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL,
        scatter_elements * sizeof(float), 256, true});

    const std::size_t counter_count =
        (n + static_cast<std::size_t>(kSelfReduceOutputTileColumns) - 1u) /
        static_cast<std::size_t>(kSelfReduceOutputTileColumns);
    requirements.buffers.push_back({
        GemmWorkspaceBuffers::ROCM_SELFREDUCE_COUNTERS,
        counter_count * sizeof(std::int32_t), 256, true});
    return requirements;
}

/**
 * @brief Add simultaneous split-K storage for one fused projection bundle.
 *
 * The four side streams each retain only the widest projection assigned to
 * that stream.  Logical projections beyond four reuse a stream after its
 * preceding member completes, matching the runtime launch schedule without
 * reserving one full partial bank per expert.
 *
 * @param requirements Aggregate stage requirements to extend.
 * @param rows Captured row capacity of the fused transaction.
 * @param projection_columns Ordered N dimension for every logical projection.
 * @throws std::invalid_argument for empty/non-positive geometry.
 */
inline void appendFusedProjectionRequirements(
    WorkspaceRequirements& requirements,
    int rows,
    std::span<const int> projection_columns)
{
    if (rows <= 0 || projection_columns.empty())
    {
        throw std::invalid_argument(
            "ROCm fused quantized-GEMM workspace requires positive rows and projections");
    }

    const std::size_t stream_count = std::min(
        projection_columns.size(), kFusedProjectionStreamCapacity);
    std::array<std::size_t, kFusedProjectionStreamCapacity> stream_widths{};
    for (std::size_t projection = 0;
         projection < projection_columns.size();
         ++projection)
    {
        if (projection_columns[projection] <= 0)
        {
            throw std::invalid_argument(
                "ROCm fused quantized-GEMM output widths must be positive");
        }
        const std::size_t stream = projection % stream_count;
        stream_widths[stream] = std::max(
            stream_widths[stream],
            static_cast<std::size_t>(projection_columns[projection]));
    }

    std::size_t simultaneous_columns = 0u;
    for (std::size_t stream = 0; stream < stream_count; ++stream)
    {
        if (stream_widths[stream] >
            std::numeric_limits<std::size_t>::max() - simultaneous_columns)
        {
            throw std::overflow_error(
                "ROCm fused quantized-GEMM stream widths overflow size_t");
        }
        simultaneous_columns += stream_widths[stream];
    }

    const std::size_t verifier_rows = static_cast<std::size_t>(
        std::clamp(rows, 1, kDefaultNativeVNNIVerifierRowCapacity));
    std::size_t elements = checkedProduct(
        static_cast<std::size_t>(kScatterKBlockCapacity),
        verifier_rows,
        "fused split-K partitions and verifier rows");
    elements = checkedProduct(
        elements, simultaneous_columns, "fused output columns");
    requirements.buffers.push_back({
        GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED,
        elements * sizeof(float), 256, true});
}

} // namespace llaminar2::rocm::quantized_gemm_workspace
