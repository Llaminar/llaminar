/**
 * @file ROCmTPRankOrderedReductionKernels.hip
 * @brief ROCm fixed-rank reduction for batch-invariant TP publication.
 *
 * The native allgather preceding this kernel transports participant partials
 * without changing their bits.  One HIP thread then owns each output element
 * and folds rank banks in ascending order with explicit round-to-nearest
 * additions. The caller supplies all storage and the exact graph stream.
 * Device-prefix folds read only live values, retain capacity rank strides and
 * leave every inactive output byte unchanged.
 */

#include "../../common/TPRankOrderedReductionKernels.h"

#include <hip/hip_runtime.h>

namespace llaminar2
{
    namespace
    {
        constexpr int kThreadsPerBlock = 256;

        /**
         * @brief Fold one fixed-bank value column in ascending rank order.
         * @param rank_banks Disjoint immutable participant banks.
         * @param output Complete fixed in-place FP32 destination bank.
         * @param element_count Exact immutable scalar width of each participant bank.
         * @param rank_count Physical participants, folded from rank zero upward.
         */
        __global__ void rankOrderedSumFP32Kernel(
            const float *__restrict__ rank_banks,
            float *__restrict__ output,
            std::size_t element_count,
            int rank_count)
        {
            const std::size_t first =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const std::size_t stride =
                static_cast<std::size_t>(blockDim.x) * gridDim.x;
            for (std::size_t element = first;
                 element < element_count;
                 element += stride)
            {
                float sum = rank_banks[element];
                for (int rank = 1; rank < rank_count; ++rank)
                {
                    sum = __fadd_rn(
                        sum,
                        rank_banks[static_cast<std::size_t>(rank) *
                                       element_count +
                                   element]);
                }
                output[element] = sum;
            }
        }

        /**
         * @brief Fold only live values while retaining the original rank-bank stride.
         * @param rank_banks Immutable rank-major partials published by native allgather.
         * @param output Original in-place FP32 bank; inactive bytes are preserved.
         * @param rows Whole-prefix geometry and its ordered device count authority.
         * @param columns Exact number of FP32 values in one row.
         * @param rank_count Positive participant membership, in canonical rank order.
         *
         * The scalar additions are identical to the fixed-bank implementation.
         * A shorter replay must neither read stale scratch nor republish it into
         * output capacity that is not an operand of this transaction.
         */
        __global__ void rankOrderedLiveSumFP32Kernel(
            const float *__restrict__ rank_banks, float *__restrict__ output,
            DeviceRowRange rows, std::size_t columns, int rank_count)
        {
            const std::size_t bank_stride = static_cast<std::size_t>(rows.capacity()) * columns;
            const std::size_t live_elements = static_cast<std::size_t>(rows.activeRows()) * columns;
            const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
            for (std::size_t element = first; element < live_elements; element += stride)
            {
                float sum = rank_banks[element];
                for (int rank = 1; rank < rank_count; ++rank)
                    sum = __fadd_rn(sum, rank_banks[static_cast<std::size_t>(rank) * bank_stride + element]);
                output[element] = sum;
            }
        }

        /**
         * @brief Replace only the original local-rank load, preserving every add.
         * @param local Original immutable local partial, optionally the output.
         * @param peers Source-indexed peer banks; own slot is never read.
         * @param output Exact fixed-width destination, disjoint from peer banks.
         * @param elements Exact fixed-message scalar extent.
         * @param bank_stride Admitted peer-slot stride, not transmitted capacity.
         * @param rank_count Original ordered physical membership size.
         * @param local_rank Original communicator position, never a device ordinal.
         *
         * Each thread reads its own local column before writing that column, so
         * exact local/output alias is safe after every local send completed.
         * There is no restrict promise between local and output. FP32 addition
         * order and round-to-nearest semantics match the original full-bank fold.
         */
        __global__ void rankOrderedLocalSumFP32Kernel(
            const float *local, const float *__restrict__ peers, float *output,
            std::size_t elements, std::size_t bank_stride,
            int rank_count, int local_rank)
        {
            const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
            for (std::size_t element = first; element < elements; element += stride)
            {
                float sum = local_rank == 0 ? local[element] : peers[element];
                for (int rank = 1; rank < rank_count; ++rank)
                {
                    const float contribution = rank == local_rank ? local[element]
                        : peers[static_cast<std::size_t>(rank) * bank_stride + element];
                    sum = __fadd_rn(sum, contribution);
                }
                output[element] = sum;
            }
        }
    } // namespace

    /**
     * @brief Enqueue a fixed-bank ROCm canonical reduction on its exact stream.
     * @param rank_banks Stable disjoint immutable participant banks.
     * @param output Complete fixed FP32 output bank.
     * @param element_count Exact immutable scalar bank width.
     * @param rank_count Number of physical participants, at least two.
     * @param device_index Exact device ordinal owning both banks.
     * @param stream Non-null ordered native stream.
     * @return Whether validation and native enqueue succeeded.
     */
    bool launchROCmTPRankOrderedSumFP32(
        const float *rank_banks,
        float *output,
        std::size_t element_count,
        int rank_count,
        int device_index,
        void *stream)
    {
        if (!rank_banks || !output || element_count == 0 ||
            rank_count < 2 || device_index < 0 || !stream)
        {
            return false;
        }
        if (hipSetDevice(device_index) != hipSuccess)
            return false;

        const std::size_t blocks_needed =
            (element_count + kThreadsPerBlock - 1u) /
            static_cast<std::size_t>(kThreadsPerBlock);
        const unsigned int blocks = static_cast<unsigned int>(
            blocks_needed > 65535u ? 65535u : blocks_needed);
        hipLaunchKernelGGL(
            rankOrderedSumFP32Kernel,
            dim3(blocks),
            dim3(kThreadsPerBlock),
            0,
            static_cast<hipStream_t>(stream),
            rank_banks,
            output,
            element_count,
            rank_count);
        return hipPeekAtLastError() == hipSuccess;
    }

    /**
     * @brief Enqueue a retained canonical fold over the native collective's live prefix.
     * @param rank_banks Stable disjoint rank-major input banks with capacity strides.
     * @param output Original in-place FP32 participant bank.
     * @param rows Validated whole-prefix geometry shared with the preceding allgather.
     * @param rank_count Number of physical participants, at least two.
     * @param device_index Exact ordinal owning the banks and non-null stream.
     * @param stream Ordered native stream; no count readback or synchronization occurs.
     * @return Whether geometry/device admission and the kernel enqueue succeeded.
     */
    bool launchROCmTPRankOrderedSumFP32(
        const float *rank_banks, float *output, const NativeCollectiveRows &rows,
        int rank_count, int device_index, void *stream)
    {
        if (!rank_banks || !output || rank_count < 2 || device_index < 0 || !stream ||
            !rows.byteGeometryValid(sizeof(float), rank_count)) return false;
        if (hipSetDevice(device_index) != hipSuccess) return false;
        const std::size_t blocks_needed = 1 + (rows.bankElements() - 1) / kThreadsPerBlock;
        const unsigned int blocks = static_cast<unsigned int>(blocks_needed > 65535u ? 65535u : blocks_needed);
        hipLaunchKernelGGL(rankOrderedLiveSumFP32Kernel, dim3(blocks), dim3(kThreadsPerBlock), 0,
            static_cast<hipStream_t>(stream), rank_banks, output, rows.rows(), rows.elementsPerRow(), rank_count);
        return hipPeekAtLastError() == hipSuccess;
    }

    /**
     * @brief Submit only a previously validated immutable local-aware operand set.
     * @param launch Complete fixed-message geometry with explicit communicator rank.
     * @param device_index Actual ROCm ordinal owning all borrowed pointers.
     * @param stream Exact non-null ordered graph/execution stream.
     * @return Whether device selection and native enqueue succeeded, without retry.
     */
    bool launchROCmTPRankOrderedLocalSumFP32(
        const TPRankOrderedLocalSumFP32 &launch, int device_index, void *stream)
    {
        if (!tpRankOrderedLocalLaunchValid(device_index, stream)) return false;
        if (hipSetDevice(device_index) != hipSuccess) return false;
        const std::size_t needed = 1 + (launch.elements() - 1) / kThreadsPerBlock;
        const unsigned int blocks = static_cast<unsigned int>(needed > 65535u ? 65535u : needed);
        hipLaunchKernelGGL(rankOrderedLocalSumFP32Kernel, dim3(blocks), dim3(kThreadsPerBlock), 0,
            static_cast<hipStream_t>(stream), launch.local().data, launch.peerBanks(),
            launch.output(), launch.elements(), launch.bankStride(), launch.rankCount(),
            launch.local().participant_index);
        return hipPeekAtLastError() == hipSuccess;
    }
} // namespace llaminar2
