/**
 * @file CPUProjectionOutputRows.h
 * @brief Checked FP32 destinations for projections with an independent row order.
 *
 * Expert-major work can publish into its original route slots without changing
 * the address order of weights or activations. The routing owner retains the
 * immutable index span and exclusive destination storage for the invocation.
 */
#pragma once
#include "CPUProjectionStorageProof.h"
#include <functional>
#include <limits>
#include <span>

namespace llaminar2::cpu::native_quantised
{
    /** @brief An optional increasing subset of rows in a retained strided bank. */
    class CPUProjectionOutputRows final
    {
    public:
        /** @brief Use ordinary strided addresses when no index span is supplied. */
        CPUProjectionOutputRows() = default;

        /**
         * @brief Bind a subset from a stable expert bucket to its output bank.
         * @param bank Retained, exclusive destination with optional inter-row padding.
         * @param stride Physical distance between output rows in floats.
         * @param columns Live values in each row.
         * @param indices Strictly increasing original row identities for this projection.
         * @return Borrowed immutable row addresses with checked capacity and geometry.
         * @throws std::invalid_argument If rows are absent, repeated, or outside the bank.
         */
        static CPUProjectionOutputRows indexed(std::span<float> bank, int stride, int columns,
            std::span<const int> indices)
        {
            if (indices.empty() || !bank.data() || columns <= 0 || stride < columns ||
                bank.size() < static_cast<std::size_t>(columns) ||
                bank.size() > std::numeric_limits<std::size_t>::max() / sizeof(float))
                throw std::invalid_argument("Indexed CPU projection requires a complete output bank");
            const auto last = (bank.size() - static_cast<std::size_t>(columns)) / stride;
            int prior = -1;
            for (const int index : indices)
            {
                if (index <= prior || static_cast<std::size_t>(index) > last)
                    throw std::invalid_argument("Indexed CPU projection requires distinct increasing live rows");
                prior = index;
            }
            storage_proof::requireDisjoint({bank.data(), bank.size_bytes()},
                {indices.data(), indices.size_bytes()});
            return CPUProjectionOutputRows(bank.data(), stride, columns, indices);
        }

        /** @return Whether the caller bound indexed destinations. */
        [[nodiscard]] explicit operator bool() const noexcept { return !indices_.empty(); }

        /** @return Whether these indexed destinations belong to the calling projection. */
        [[nodiscard]] bool matches(float *base, int stride, int columns, int count) const noexcept
        {
            return !*this || (base == base_ && stride == stride_ && columns == columns_ &&
                static_cast<std::size_t>(count) == indices_.size());
        }

        /**
         * @brief Resolve a tile's final output address before arithmetic.
         * @param base Ordinary or indexed bank base retained by the projection.
         * @param stride Physical distance between output rows in floats.
         * @param row Valid projection-local row index.
         * @return Exclusive beginning of the row's live values.
         */
        [[nodiscard]] float *rowAddress(float *base, int stride, std::size_t row) const noexcept
        {
            return base + (indices_.empty() ? row : static_cast<std::size_t>(indices_[row])) * stride;
        }

        /** @return Original read-only row index bytes for alias validation. */
        [[nodiscard]] storage_proof::Bytes indexBytes() const noexcept
        {
            return {indices_.data(), indices_.size_bytes()};
        }

    private:
        /**
         * @brief Retain geometry after validating its byte extent and immutable indices.
         * @param base Exclusive destination bank start.
         * @param stride Physical float stride.
         * @param columns Logical live float width.
         * @param indices Validated increasing row identities.
         */
        CPUProjectionOutputRows(float *base, int stride, int columns, std::span<const int> indices)
            : base_(base), stride_(stride), columns_(columns), indices_(indices) {}

        float *base_ = nullptr;
        int stride_ = 0;
        int columns_ = 0;
        std::span<const int> indices_;
    };
}
