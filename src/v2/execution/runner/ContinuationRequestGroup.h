/**
 * @file ContinuationRequestGroup.h
 * @brief Immutable request-payload membership shared by token and prefix admission.
 *
 * Outer commands still reach every execution rank. Prompt tokens and prefix/KV
 * consensus belong only to ranks with continuation graphs; expert-only ranks
 * consume sparse transaction tickets instead. One retained group owns that
 * distinction, including nonzero command roots and multi-rank continuation.
 * Group construction and retirement occur outside request execution.
 */
#pragma once

#include "interfaces/IMPIContext.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace llaminar2
{
    /** @brief Canonical continuation-state owners for one execution membership. */
    class ContinuationRequestGroup final
    {
    public:
        /** MPI extent needed by the frozen continuation membership. */
        enum class Scope : std::uint8_t
        {
            OrchestrationWorld,
            ProcessLocalContinuation,
            ContinuationRankGroup,
        };

        /** Exclusive request-data role derived from the resolved execution plan. */
        enum class Role : std::uint8_t
        {
            TokenConsumer,
            ExpertOnly,
        };

        /**
         * @brief Retain exactly the resolved continuation ranks.
         * @param world Execution context, not a discovery-rank namespace.
         * @param consumer_ranks Sorted, unique ranks owning continuation state.
         * @throws std::invalid_argument for invalid membership.
         * @throws std::runtime_error if subgroup admission fails.
         *
         * All execution ranks construct the same membership during startup.
         * A proper multi-rank subset creates one persistent MPI communicator;
         * a single consumer needs no token transport or subgroup collective.
         */
        ContinuationRequestGroup(
            std::shared_ptr<IMPIContext> world,
            std::vector<int> consumer_ranks);

        /** @brief Release the owned subgroup after request/worker retirement. */
        ~ContinuationRequestGroup();

        ContinuationRequestGroup(const ContinuationRequestGroup &) = delete;
        ContinuationRequestGroup &operator=(const ContinuationRequestGroup &) = delete;

        /** @return This rank's immutable request-data role. */
        [[nodiscard]] Role role() const noexcept { return role_; }
        /** @return The minimal collective scope for continuation state. */
        [[nodiscard]] Scope scope() const noexcept { return scope_; }
        /** @return Number of actual token consumers, including the command root. */
        [[nodiscard]] std::size_t consumerCount() const noexcept { return consumer_ranks_.size(); }
        /** @return Stable observation label for the retained scope. */
        [[nodiscard]] const char *scopeName() const noexcept;

        /**
         * @brief Borrow the same group for prefix/KV consensus.
         * @return Communicator for consumers, or null for a process-local owner.
         * @throws std::logic_error if an expert-only rank requests prefix state.
         */
        [[nodiscard]] MPI_Comm coordinationCommunicator() const;

        /**
         * @brief Publish only live tokens to other continuation consumers.
         * @param tokens Exact nonempty prompt, never allocation capacity.
         * @param root_rank Current command authority in the execution namespace.
         * @throws std::logic_error if the caller is not the consumer root.
         * @throws std::invalid_argument for an invalid extent or nonconsumer root.
         * @throws std::runtime_error on transport failure.
         */
        void publishPrompt(std::span<const std::int32_t> tokens, int root_rank) const;

        /**
         * @brief Receive live prompt bytes only on a continuation follower.
         * @param token_count Validated count from the world command header.
         * @param root_rank Current command authority in the execution namespace.
         * @return Exact tokens for a consumer; an empty vector for an expert-only
         *         rank, with no prompt allocation or payload collective.
         * @throws std::logic_error if the root attempts to receive its own prompt.
         * @throws std::invalid_argument for invalid authority or extent.
         * @throws std::runtime_error on transport failure.
         */
        [[nodiscard]] std::vector<std::int32_t> receivePrompt(
            std::size_t token_count, int root_rank) const;

    private:
        /**
         * @brief Validate the live wire extent and root's recipient membership.
         * @param token_count Exact positive prompt extent.
         * @param root_rank Command authority in the execution namespace.
         * @return Root's index in the sorted consumer subgroup.
         * @throws std::invalid_argument for invalid extent or root membership.
         */
        [[nodiscard]] int validatePrompt(std::size_t token_count, int root_rank) const;
        /**
         * @brief Exchange an exact prompt extent on the already-retained scope.
         * @param tokens Consumer-owned prompt storage; root storage is not modified.
         * @param count Validated live element count.
         * @param root_rank World-context root for a full consumer world.
         * @param group_root Remapped root for a proper consumer subset.
         * @throws std::runtime_error if native subgroup transport fails.
         */
        void broadcastPrompt(std::int32_t *tokens, std::size_t count, int root_rank,
                             int group_root) const;
        /**
         * @brief Observe logical endpoint bytes without becoming protocol authority.
         * @param count Exact prompt extent just published or received.
         * @param root_rank Authority determining send versus receive observation.
         *
         * Tags describe stable topology only; request lengths are values, never
         * aggregation keys that grow the collector with every distinct prompt.
         */
        void recordPromptExtent(std::size_t count, int root_rank) const;

        std::shared_ptr<IMPIContext> world_; ///< Keeps the borrowed execution context alive.
        std::vector<int> consumer_ranks_; ///< Only the resolved plan supplies membership.
        Role role_; ///< Exclusive local data role; never inferred from runtime tickets.
        Scope scope_; ///< Frozen before any request can enter.
        MPI_Comm group_comm_{MPI_COMM_NULL}; ///< Owned only for a proper multi-rank subset.
    };
}
