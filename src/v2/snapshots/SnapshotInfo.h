/**
 * @file SnapshotInfo.h
 * @brief Lifetime-retained typed diagnostic view shared by inference runner interfaces.
 *
 * Float APIs retain their existing numerical contract. Exact integer identities
 * use a distinct view into the same immutable owner and cannot round through a
 * floating pointer. No live execution state is mirrored by this interface.
 */
#pragma once
#include "SnapshotValues.h"
#include "SnapshotPublication.h"
#include "SnapshotRowPartition.h"
#include "SnapshotSemanticAxis.h"
#include "SnapshotAuthoredTokenRowsObservation.h"
#include <optional>
#include <memory>

namespace llaminar2
{
    /**
     * @brief Lightweight view of a captured snapshot with 2D shape metadata
     *
     * Returned by getSnapshotWithShape() to provide shape information
     * alongside an immutable FP32 or INT64 view. The shape (rows, cols) comes
     * from the stage's getDumpInfo() at capture time, so stages own their
     * own dimension reporting. Production snapshot providers also attach a
     * type-erased lifetime token: code that retains SnapshotInfo across a
     * later graph callback, reset, or map replacement keeps the exact
     * captured tensor alive through that token.
     */
    struct SnapshotInfo
    {
        const float *data = nullptr; ///< Pointer to immutable FP32 snapshot data.
        size_t size = 0;             ///< Total element count (rows * cols)
        size_t rows = 0;             ///< Logical rows (e.g. seq_len)
        size_t cols = 0;             ///< Logical cols (e.g. hidden_dim, kv_dim, d_ff)

        /**
         * @brief Retains the publication backing data when the provider has ownership.
         *
         * SnapshotInfo remains a lightweight view, so this type-erased token
         * avoids exposing SnapshotCapture storage through every runner
         * interface. A null token is permitted for older mock/static providers
         * whose runner itself owns the data for the caller's whole operation.
         */
        std::shared_ptr<const void> lifetime_owner;

        /// Producer completeness belongs to these exact bytes, not to the key.
        SnapshotPublication publication = SnapshotPublication::SchemaPartition;
        SnapshotOwnedRows row_ownership; ///< Immutable observed row ownership; no inference authority.

        SnapshotValueView typed_values; ///< Exact integer view or floating diagnostic view, sharing the same owner.

        SnapshotSemanticAxis semantic_axis=SnapshotSemanticAxis::Legacy; ///< Immutable acquired mathematical axis.
        std::optional<SnapshotAuthoredTokenRowsObservation> authored_token_rows; ///< Acquired original coordinates; no live authority.

        /** @return Whether this publication contains exact signed integer identities. */
        [[nodiscard]] bool isInteger() const noexcept
        {
            return std::holds_alternative<std::span<const std::int64_t>>(typed_values);
        }
        /** @return Exact integer view; rejects floating reinterpretation. */
        [[nodiscard]] std::span<const std::int64_t> integerValues() const
        {
            if (!isInteger()) throw std::logic_error("snapshot is not an exact integer publication");
            return std::get<std::span<const std::int64_t>>(typed_values);
        }
        /** @return Whether either typed payload has live values. */
        explicit operator bool() const { return (size > 0 && (data != nullptr || (isInteger() && !integerValues().empty()))) || (size==0 && rows==0 && cols>0 && lifetime_owner); }
    };

} // namespace llaminar2
