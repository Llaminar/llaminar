/**
 * @file CPUW4A4Arithmetic.h
 * @brief Allocation-free native CPU W4A4/H preparation and grouped projection.
 *
 * The stage lends persistent, independently admitted scratch to each invocation.
 * Preparation owns the whole-token A4 unit and the ordered Hadamard transform;
 * projection consumes a sealed activation view, not an arbitrary packed pointer.
 * Output tiles share weight decoding across tokens without changing per-group
 * integer shifts or the final FP32 multiplication order. No floating weight
 * representation or serial-row replay path is involved.
 */
#pragma once

#include "tensors/W4A4ActivationFormat.h"
#include "tensors/W4A4SourceView.h"
#include "kernels/common/ProjectionRowWork.h"

namespace llaminar2::w4a4
{
    /** @brief Clipped zero-inclusive extrema for one activation group. */
    struct ActivationRange
    {
        float lower; ///< Group minimum after clipping.
        float upper; ///< Group maximum after clipping.
    };

    /** @brief Typed BOM element counts; allocation remains with the arena/PMA authority. */
    struct CPUWorkspaceCounts
    {
        std::size_t blocks;            ///< Packed A4 blocks for the admitted token envelope.
        std::size_t ranges;            ///< Group extrema used to choose whole-token units.
        std::size_t row_ranges;        ///< Exact token maxima; never reconstructed from rounded units.
        std::size_t butterfly_floats;  ///< Sylvester bank; absent for identity transforms.
        std::size_t transformed_floats;///< Normalized result bank; absent for identity transforms.
        std::size_t active_rows;       ///< Operand-validity bytes for counted/noncontiguous work.
    };

    /** @brief Borrowed invocation-local banks; capacities may exceed the live prefix. */
    struct CPUActivationBuffers
    {
        std::span<ActivationBlock> blocks; ///< Exact A4 storage, separate from input and weights.
        std::span<ActivationRange> ranges; ///< Extrema live through A4 quantization.
        std::span<float> row_ranges;       ///< Original token maxima across the producer barrier.
        std::span<float> butterfly;        ///< Ordered Sylvester intermediate.
        std::span<float> transformed;      ///< Ordered Paley and normalization result.
        std::span<std::uint8_t> active_rows{}; ///< Operand validity, independent of later controller writes.
    };

    class CPUActivations;
    /**
     * @brief Price the source-declared online transform before payload materialization.
     * @param transform Original format's identity or ordered Hadamard policy.
     * @param columns Complete original reduction width, with native extent checks.
     * @param tokens Simultaneous physical rows; zero retains no invocation storage.
     * @return The same typed counts consumed by admitted runtime matrices.
     * @throws AssertionError For unknown policy or invalid/overflowing dimensions.
     */
    [[nodiscard]] CPUWorkspaceCounts cpuWorkspaceCounts(Transform transform, int columns, int tokens);

    /**
     * @brief Compute checked workspace element counts, without allocating or reserving.
     * @param contract Admitted complete-row numerical policy.
     * @param tokens Maximum simultaneously live tokens; zero is an empty invocation.
     * @return Typed counts contributed to the stage's canonical workspace BOM.
     * @throws AssertionError on negative/overflowing invocation geometry.
     */
    [[nodiscard]] CPUWorkspaceCounts cpuWorkspaceCounts(const MatrixContract &contract, int tokens);

    /**
     * @brief Prepare complete A4 rows with the exact source-declared transform and clip.
     * @param contract Admitted weight policy selecting the activation arithmetic.
     * @param input Borrowed FP32 inputs; the function never mutates them.
     * @param tokens Original physical rows of this invocation; unused arena tail is excluded.
     * @param input_stride FP32 elements between complete input rows.
     * @param workspace Invocation-owned admitted banks; inactive operand rows remain untouched.
     * @param row_work Optional original-coordinate CPU publication; never a GPU count readback.
     * @return Sealed borrowed A4 view valid until the workspace's next writer.
     * @throws AssertionError on invalid extents, overlap, rounding mode or numeric range.
     *
     * Every member of an existing OpenMP team must participate. Otherwise the
     * existing nested-safe workshare authority supplies the team. Producer
     * barriers publish complete operands before any consumer is admitted.
     */
    [[nodiscard]] CPUActivations prepareCPUActivations(
        const MatrixContract &contract, std::span<const float> input, int tokens,
        std::size_t input_stride, CPUActivationBuffers workspace,
        std::optional<ProjectionRowWork> row_work = std::nullopt);

    /** @brief A completed native A4 batch; callers cannot forge its arithmetic identity. */
    class CPUActivations final
    {
    public:
        /** @return Original invocation row extent, excluding the unused arena tail. */
        [[nodiscard]] int rows() const noexcept { return rows_; }
        /** @return Whether this original row owns a completed operand, not its current controller count. */
        [[nodiscard]] bool active(int row) const noexcept
        {
            return row >= 0 && row < rows_ && (active_rows_.empty() || active_rows_[row] != 0);
        }
        /** @return Original-coordinate storage; only rows accepted by active() contain operands. */
        [[nodiscard]] std::span<const ActivationBlock> blocks() const noexcept { return blocks_; }
        /** @return Whether another projection consumes exactly the same operand policy. */
        [[nodiscard]] bool matches(const MatrixContract &contract) const noexcept
        {
            return columns_ == contract.columns() && transform_ == contract.transform()
                && rotation_ == contract.rotationLength() && clip_bits_ == contract.encodedClipBits();
        }

    private:
        friend void projectCPU(const SourceView &, const CPUActivations &, std::span<float>, std::size_t);
        friend CPUActivations prepareCPUActivations(const MatrixContract &, std::span<const float>,
            int, std::size_t, CPUActivationBuffers, std::optional<ProjectionRowWork>);
        /** @brief Publish completed operands and their validity after the final producer barrier. */
        CPUActivations(const MatrixContract &contract, int rows,
                       std::span<const ActivationBlock> blocks, std::span<const std::uint8_t> active_rows) noexcept
            : rows_(rows), columns_(contract.columns()), transform_(contract.transform()),
              rotation_(contract.rotationLength()), clip_bits_(contract.encodedClipBits()),
              blocks_(blocks), active_rows_(active_rows) {}
        int rows_;                                   ///< Original physical M.
        int columns_;                                ///< Complete K used to choose token units.
        Transform transform_;                        ///< Exact transform policy.
        int rotation_;                               ///< Source-declared transform-block extent.
        std::uint32_t clip_bits_;                     ///< Exact source clipping encoding.
        std::span<const ActivationBlock> blocks_;     ///< Borrowed published operands, not a shadow.
        std::span<const std::uint8_t> active_rows_;    ///< Validity of those operands; empty means all rows.
    };

    /**
     * @brief Project native operands with weight-sharing token tiles and exact group terms.
     * @param weights Admitted immutable native matrix; no independent raw pointer is accepted.
     * @param activations Completed operands with matching whole-row arithmetic identity.
     * @param output Borrowed FP32 destination, disjoint from all consumed operands.
     * @param output_stride FP32 elements between output rows; padding remains untouched.
     * @throws AssertionError on policy mismatch, incomplete output, overlap or rounding mode.
     *
     * Each tile owns complete output values. No partition-dependent floating
     * reduction or atomics enter the arithmetic; the int32 range was proved by
     * matrix admission. Every positive team size uses the same group contract.
     */
    void projectCPU(const SourceView &weights, const CPUActivations &activations,
                    std::span<float> output, std::size_t output_stride);
} // namespace llaminar2::w4a4
