/**
 * @file MoEExpertSourceView.h
 * @brief Checked expert/source coordinates for zero-copy preparation views.
 *
 * A frozen GGUF parent uses [K,N,E] storage, whereas a GEMM view is [N,K].
 * Logical expert IDs may be noncontiguous and need not equal physical E slots.
 * This immutable value resolves that mapping once and composes an optional
 * projection-ownership interval without allocating, repacking or changing K.
 * It owns no weight bytes, placement decisions, or physical-memory accounting.
 */
#pragma once

#include "WeightIdentity.h"
#include "execution/moe/MoEExpertProjectionOwnership.h"

#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llaminar2
{
    /** @brief Immutable mapping from model expert/projection identity to raw source rows. */
    class MoEExpertSourceView final
    {
    public:
        /** @brief A contiguous full-K matrix plus its original logical identity. */
        struct Matrix
        {
            size_t rows;
            size_t columns;
            size_t element_offset;
            WeightSliceSpec source_identity;
        };

        /**
         * @brief Authenticate a frozen raw parent and its expert selection.
         * @param shape Physical GGUF [K,N,E] dimensions, after any declared slicing.
         * @param slice Original GGUF-axis coordinates: row fields describe K,
         *        column fields describe N, and expert fields describe E.
         * @throws std::invalid_argument for malformed, truncated or ambiguous selections.
         * @throws std::overflow_error when element or expert coordinates cannot be represented.
         */
        MoEExpertSourceView(std::span<const size_t> shape, WeightSliceSpec slice)
            : slice_(std::move(slice))
        {
            if (shape.size() != 3 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0)
                throw std::invalid_argument("MoE source view requires nonempty GGUF [K,N,E] geometry");
            columns_ = shape[0];
            rows_ = shape[1];
            experts_ = shape[2];
            if (rows_ > std::numeric_limits<size_t>::max() / columns_ ||
                rows_ * columns_ > std::numeric_limits<size_t>::max() / experts_)
                throw std::overflow_error("MoE source view element extent overflows size_t");
            if (!slice_.expert_ids.empty())
            {
                ids_ = slice_.expert_ids;
                if (ids_.front() < 0 || !std::is_sorted(ids_.begin(), ids_.end()) ||
                    std::adjacent_find(ids_.begin(), ids_.end()) != ids_.end() ||
                    (slice_.expert_count != 0 && slice_.expert_count != ids_.size()))
                    throw std::invalid_argument("MoE source view has inconsistent explicit expert IDs");
                if (slice_.inner_is_presliced ? ids_.size() != experts_
                    : static_cast<size_t>(ids_.back()) >= experts_)
                    throw std::invalid_argument("MoE source view expert IDs exceed the physical parent");
            }
            else
            {
                const size_t start = slice_.expert_start;
                const size_t count = slice_.expert_count != 0 ? slice_.expert_count : experts_;
                if ((slice_.expert_count == 0 && start != 0) ||
                    (slice_.inner_is_presliced ? count != experts_
                        : start > experts_ || count > experts_ - start))
                    throw std::invalid_argument("MoE source view has an incomplete contiguous expert interval");
                if (start > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    count - 1 > static_cast<size_t>(std::numeric_limits<int>::max()) - start)
                    throw std::overflow_error("MoE source view expert identity exceeds int");
                ids_.resize(count);
                for (size_t index = 0; index < count; ++index)
                    ids_[index] = static_cast<int>(start + index);
            }
        }

        /** @return Exact sorted logical experts represented by this source, not other roles. */
        const std::vector<int> &experts() const noexcept { return ids_; }

        /**
         * @brief Resolve one matrix without copying or re-quantizing source weights.
         * @param expert Logical global expert ID, not an index in a packed parent.
         * @param role Semantic routed gate, up or down role.
         * @param ownership Optional exact projection contract; null retains the physical matrix.
         * @return Checked element offset, physical dimensions and original slice identity.
         * @throws std::invalid_argument for absent experts or incompatible source/projection geometry.
         */
        Matrix matrix(int expert, WeightRole role,
                      const MoEExpertProjectionOwnership *ownership = nullptr) const
        {
            if (role != WeightRole::MoEExpertGate && role != WeightRole::MoEExpertUp &&
                role != WeightRole::MoEExpertDown)
                throw std::invalid_argument("MoE source view requires a routed projection role");
            const auto found = std::lower_bound(ids_.begin(), ids_.end(), expert);
            if (found == ids_.end() || *found != expert)
                throw std::invalid_argument("MoE source view does not contain the requested global expert");
            const size_t physical_expert = slice_.inner_is_presliced
                ? static_cast<size_t>(found - ids_.begin()) : static_cast<size_t>(expert);
            // Construct the one-expert identity directly; copying the parent's
            // entire expert-ID vector for every matrix makes setup quadratic.
            Matrix result{rows_, columns_, physical_expert * rows_ * columns_, {}};
            auto &identity = result.source_identity;
            // The parent binding describes GGUF axes [K,N,E], not a GEMM.
            // Translate once at this boundary; square matrices otherwise hide
            // the transposition and permit incorrect partition identities.
            identity.source_rows = slice_.source_cols != 0 ? slice_.source_cols : rows_;
            identity.source_cols = slice_.source_rows != 0 ? slice_.source_rows : columns_;
            identity.row_start = slice_.col_start;
            identity.col_start = slice_.row_start;
            identity.row_count = rows_;
            identity.col_count = columns_;
            identity.expert_start = static_cast<size_t>(expert);
            identity.expert_count = 1;
            identity.expert_ids = {expert};
            identity.inner_is_presliced = true;
            if (!ownership)
                return result;

            const auto projection = ownership->projection(role);
            if (ids_.back() >= ownership->geometry().experts ||
                (!slice_.inner_is_presliced && experts_ != static_cast<size_t>(ownership->geometry().experts)) ||
                columns_ != static_cast<size_t>(projection.source_columns) || slice_.row_start != 0 ||
                (slice_.source_cols != 0 && slice_.source_cols != static_cast<size_t>(projection.source_rows)) ||
                (slice_.source_rows != 0 && slice_.source_rows != columns_) ||
                (slice_.row_count != 0 && slice_.row_count != columns_))
                throw std::invalid_argument("MoE source view disagrees with its projection source identity");

            if (rows_ == static_cast<size_t>(projection.source_rows) && slice_.col_start == 0 &&
                (slice_.col_count == 0 || slice_.col_count == rows_))
            {
                // GGUF keeps each complete K row contiguous. The expert's N
                // interval can therefore be viewed before any native repack.
                result.element_offset += static_cast<size_t>(projection.first_row) * columns_;
            }
            else if (!(slice_.inner_is_presliced &&
                       slice_.source_cols == static_cast<size_t>(projection.source_rows) &&
                       slice_.col_start == static_cast<size_t>(projection.first_row) &&
                       slice_.col_count == static_cast<size_t>(projection.rows) &&
                       rows_ == static_cast<size_t>(projection.rows)))
            {
                // A pre-sliced parent is legal only when its original interval
                // authenticates this exact partition. Matching N alone is not
                // enough: a neighbour's slice has the same physical dimensions.
                throw std::invalid_argument("MoE source view contains a different down-output partition");
            }
            result.rows = static_cast<size_t>(projection.rows);
            identity.source_rows = static_cast<size_t>(projection.source_rows);
            identity.source_cols = static_cast<size_t>(projection.source_columns);
            identity.row_start = static_cast<size_t>(projection.first_row);
            identity.row_count = result.rows;
            identity.col_start = 0;
            return result;
        }

    private:
        WeightSliceSpec slice_;
        std::vector<int> ids_;
        size_t rows_ = 0;
        size_t columns_ = 0;
        size_t experts_ = 0;
    };
} // namespace llaminar2
