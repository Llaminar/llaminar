/**
 * @file MoEActivationBindings.h
 * @brief Checked, non-owning graph-family views of policy-selected MoE storage.
 *
 * Main and MTP graphs share the admitted MoE arena because they do not execute
 * concurrently. Whole-expert publication and projection-column publication
 * have different buffer inventories. This sealed value carries exactly the
 * selected inventory across graph-family boundaries; it never allocates tensor
 * storage or requires the unused inventory merely to satisfy a legacy field.
 */
#pragma once

#include "execution/config/RoutedExpertPolicy.h"
#include "memory/BufferId.h"
#include "tensors/TensorClasses.h"

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace llaminar2
{
    /** @brief Complete, immutable tensor borrows for one physical MoE buffer layout. */
    class MoEActivationBindings final
    {
    public:
        /**
         * @brief Bind the required arena roles without allocating or moving tensor bytes.
         * @param policy Resolved physical expert-compute policy.
         * @param rows Positive invocation row count, not context capacity.
         * @param lookup Returns the arena-owned tensor for an exact BufferId.
         * @return A complete view whose required tensors all cover this invocation.
         * @throws std::invalid_argument On an unknown policy, missing buffer or short extent.
         *
         * This fixed-size value is safe to construct at an existing host request
         * boundary. The graph schema remains the allocation/byte-size authority;
         * this only authenticates its borrowed tensors and row envelope.
         */
        template <typename Lookup>
        [[nodiscard]] static MoEActivationBindings bind(
            RoutedExpertComputePolicy policy, std::size_t rows, Lookup &&lookup)
        {
            MoEActivationBindings result(policy);
            for (const auto id : common_ids_) result.add(id, lookup(id));
            if (policy == RoutedExpertComputePolicy::GateUpOwnedDownColumns)
                for (const auto id : projection_ids_) result.add(id, lookup(id));
            else
                result.add(BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS,
                           lookup(BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS));
            result.require(policy, rows);
            return result;
        }

        /**
         * @brief Authenticate an existing borrow for a graph's exact policy and rows.
         * @throws std::invalid_argument If its physical mode or row envelope differs.
         */
        void require(RoutedExpertComputePolicy policy, std::size_t rows) const
        {
            if (policy != policy_ || rows == 0)
                throw std::invalid_argument("MoE activation binding policy/row identity differs");
            for (const auto &[id, tensor] : entries())
                if (!tensor || tensor->native_type() != TensorType::FP32 || tensor->rows() < rows)
                    throw std::invalid_argument("MoE activation binding " + std::string(bufferIdName(id)) +
                        " requires FP32 arena storage for " + std::to_string(rows) + " rows");
        }

        /** @return Read-only bindings to project into the graph's extension-buffer view. */
        [[nodiscard]] std::span<const std::pair<BufferId, TensorBase *>> entries() const noexcept
        { return {bindings_.data(), count_}; }

    private:
        /** @brief Reject undeclared compute modes before resolving any arena pointers. */
        explicit MoEActivationBindings(RoutedExpertComputePolicy policy) : policy_(policy)
        {
            switch (policy)
            {
            case RoutedExpertComputePolicy::Replicated:
            case RoutedExpertComputePolicy::Apportioned:
            case RoutedExpertComputePolicy::TensorSharded:
            case RoutedExpertComputePolicy::GateUpOwnedDownColumns: return;
            case RoutedExpertComputePolicy::Unspecified: break;
            }
            throw std::invalid_argument("MoE activation bindings require a resolved physical compute policy");
        }

        /** @brief Assemble only the fixed, compile-time inventory owned by this class. */
        void add(BufferId id, TensorBase *tensor) noexcept
        { bindings_[count_++] = {id, tensor}; }

        static constexpr std::array common_ids_{
            BufferId::MOE_EXPERT_INDICES, BufferId::MOE_EXPERT_WEIGHTS,
            BufferId::MOE_COMBINED_OUTPUT, BufferId::MOE_SHARED_EXPERT_OUTPUT,
            BufferId::MOE_GATE_SCRATCH, BufferId::MOE_UP_SCRATCH};
        static constexpr std::array projection_ids_{
            BufferId::MOE_PROJECTION_LOCAL_PACKET, BufferId::MOE_PROJECTION_GATHERED_PACKETS,
            BufferId::MOE_PROJECTION_ROUTE_COLUMNS, BufferId::MOE_PROJECTION_LOCAL_COLUMNS,
            BufferId::MOE_PROJECTION_GATHERED_COLUMNS};
        RoutedExpertComputePolicy policy_;
        std::array<std::pair<BufferId, TensorBase *>, common_ids_.size() + projection_ids_.size()> bindings_{};
        std::size_t count_ = 0;
    };
} // namespace llaminar2
