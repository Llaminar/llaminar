/**
 * @file W4A4SourceView.h
 * @brief Immutable borrowed native weights coupled to their admitted matrix policy.
 *
 * A policy cannot authorize an unrelated raw pointer. This view is constructed
 * only by admitting its exact source span and retains both together. The tensor
 * or PreparedWeightStore owns the physical lifetime; this descriptor neither
 * allocates storage nor introduces a second memory ledger. The immutable borrow
 * must remain inside its owning weight handle's execution lifetime.
 */
#pragma once

#include "W4A4MatrixContract.h"
#include <bit>

namespace llaminar2::w4a4
{
    /** @brief An admitted complete native matrix, never a detached K fragment. */
    class SourceView final
    {
    public:
        /**
         * @brief Seal exact immutable source bytes together with their numerical policy.
         * @param transform Transform declared by the source tensor type.
         * @param rows Complete output rows in the borrowed view.
         * @param columns Complete reduction width of every row.
         * @param source Borrowed native blocks owned by the tensor/weight handle.
         * @return An admitted view or the canonical structured source diagnostic.
         */
        [[nodiscard]] static std::expected<SourceView, AdmissionError> admit(
            Transform transform, std::size_t rows, std::size_t columns,
            std::span<const SourceBlock> source) noexcept
        {
            static_assert(std::endian::native == std::endian::little,
                          "W4A4 source views require little-endian native blocks");
            const auto contract = MatrixContract::admit(transform, rows, columns, source);
            if (!contract) return std::unexpected(contract.error());
            return SourceView(*contract, source);
        }

        /**
         * @brief Borrow a checked subset of complete output rows without rescanning weights.
         * @param first First output row relative to this view.
         * @param count Positive output-row count.
         * @return Exact subspan and derived policy, or the canonical extent diagnostic.
         */
        [[nodiscard]] std::expected<SourceView, AdmissionError> rowSlice(
            std::size_t first, std::size_t count) const noexcept
        {
            const auto subset = contract_.rowSlice(first, count);
            if (!subset) return std::unexpected(subset.error());
            const auto row_blocks = static_cast<std::size_t>(contract_.blocksPerRow());
            return SourceView(*subset, source_.subspan(first * row_blocks, count * row_blocks));
        }

        /** @return The immutable policy admitted from these exact blocks. */
        [[nodiscard]] const MatrixContract &contract() const noexcept { return contract_; }
        /** @return Complete native blocks; the physical owner retains their lifetime. */
        [[nodiscard]] std::span<const SourceBlock> blocks() const noexcept { return source_; }

    private:
        /** @brief Bind only an already validated policy/source pair. */
        SourceView(MatrixContract contract, std::span<const SourceBlock> source) noexcept
            : contract_(contract), source_(source) {}
        MatrixContract contract_;              ///< Immutable admission of this exact span.
        std::span<const SourceBlock> source_;   ///< Borrowed stable bytes, never a floating mirror.
    };
} // namespace llaminar2::w4a4
