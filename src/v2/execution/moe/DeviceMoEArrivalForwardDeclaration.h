/**
 * @file DeviceMoEArrivalForwardDeclaration.h
 * @brief The graph-local payload-arrival sites of a routed forward layer.
 *
 * Native arrival count is a property of the retained graph's role and row
 * geometry, not of the command contents or the number of destination slots.
 */
#pragma once

#include "execution/config/RoutedExpertPolicy.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Immutable phase and transport requirements for one routed layer. */
    struct DeviceMoEArrivalForwardDeclaration
    {
        enum class Phase : uint8_t { OrdinaryPrefill, DecodeOrVerifier, Predictor };
        enum class PrefixPayload : uint8_t { Absent, Attached };
        enum class Transport : uint8_t { None, NativeLocalTP };

        struct Request
        {
            Phase phase;
            PrefixPayload prefix_payload;
            RoutedExpertRowExecutionPolicy row_execution;
            RoutedExpertAssignmentPolicy prefill_assignment;
            uint64_t row_count;
            uint64_t top_k;
            uint64_t minimum_routed_rows;

            /** @return Whether this is ordinary participant-assigned LLEP work. */
            [[nodiscard]] constexpr bool requestsCurrentBatch() const noexcept
            {
                return phase == Phase::OrdinaryPrefill && row_count > 1 &&
                    row_execution == RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
                    prefill_assignment == RoutedExpertAssignmentPolicy::LeastLoadedResident;
            }

            /** @return Whether the live row geometry meets the authored LLEP economy gate. */
            [[nodiscard]] constexpr bool passesCurrentBatchCostGate() const noexcept
            {
                // Ceiling division avoids overflowing a future row geometry.
                return !requestsCurrentBatch() || minimum_routed_rows == 0 ||
                    (top_k != 0 && row_count >= minimum_routed_rows / top_k +
                        (minimum_routed_rows % top_k != 0));
            }

            /** @return Whether the exact forward actually borrows the prefix payload program. */
            [[nodiscard]] constexpr bool requestsPrefix() const noexcept
            {
                return phase != Phase::Predictor && prefix_payload == PrefixPayload::Attached;
            }
        };

        /**
         * @brief Resolve the two independent arrival sites from admitted native transport.
         * @throws std::invalid_argument when a requested native site has no transport.
         */
        [[nodiscard]] static constexpr DeviceMoEArrivalForwardDeclaration
        declare(const Request &request, Transport transport)
        {
            const bool prefix = request.requestsPrefix();
            const bool batch = request.requestsCurrentBatch() && request.passesCurrentBatchCostGate();
            if (transport != Transport::NativeLocalTP && (prefix || batch))
                throw std::invalid_argument("MoE forward arrivals require an admitted native LocalTP transport");
            return {prefix, batch};
        }

        /** @return Whether to lower the prefix payload arrival before routing. */
        [[nodiscard]] constexpr bool prefix() const noexcept { return prefix_; }
        /** @return Whether to lower the graph-phased current-batch arrival. */
        [[nodiscard]] constexpr bool currentBatch() const noexcept { return current_batch_; }
        /** @return Number of final updatable sites when this layer belongs to a CUDA owner. */
        [[nodiscard]] constexpr std::size_t sites() const noexcept { return prefix_ + current_batch_; }

    private:
        constexpr DeviceMoEArrivalForwardDeclaration(bool prefix, bool batch) noexcept
            : prefix_(prefix), current_batch_(batch) {}

        bool prefix_;
        bool current_batch_;
    };
} // namespace llaminar2
