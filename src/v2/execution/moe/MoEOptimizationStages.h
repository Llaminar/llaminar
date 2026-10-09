/**
 * @file MoEOptimizationStages.h
 * @brief Immutable namespaces for independent pipeline movement publishers.
 *
 * Transaction and epoch integers belong to one stage controller. A pipeline
 * retains ordered, disjoint scopes instead of concatenating those integers
 * into a fictitious model-wide publication stream. These are passive metadata
 * snapshots; they neither own device state nor admit physical memory.
 */
#pragma once

#include "backends/GlobalDeviceAddress.h"
#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llaminar2
{
    /** @brief Checked identity of one stage's main and auxiliary routed layers. */
    class MoEOptimizationStageIdentity final
    {
    public:
        /**
         * @brief Bind a publication namespace to the admitted stage geometry.
         * @param index Position in the authored pipeline, beginning at zero.
         * @param first First global main layer, inclusive.
         * @param main_end Last global main layer, exclusive.
         * @param routed_end Last owned routed layer, including retained MTP.
         * @param participants Exact ordered continuation domain, never capacity.
         * @param terminal Whether this stage owns the global model output boundary.
         * @throws std::invalid_argument for empty or contradictory geometry.
         */
        MoEOptimizationStageIdentity(std::size_t index, int first, int main_end,
            int routed_end, std::vector<GlobalDeviceAddress> participants, bool terminal)
            : index_(index), first_(first), main_end_(main_end), routed_end_(routed_end),
              participants_(std::move(participants)), terminal_(terminal)
        {
            if (first < 0 || main_end <= first || routed_end < main_end || participants_.empty())
                throw std::invalid_argument("Pipeline movement scope has invalid layer or participant geometry");
            for (std::size_t i = 0; i < participants_.size(); ++i)
                if (!participants_[i].toLocalDeviceId().is_valid() ||
                    std::find(participants_.begin(), participants_.begin() + i, participants_[i]) != participants_.begin() + i)
                    throw std::invalid_argument("Pipeline movement scope has an invalid or duplicate participant");
        }
        /** @return Authored pipeline order, independent of controller epochs. */
        std::size_t index() const noexcept { return index_; }
        /** @return First global main and routed layer, inclusive. */
        int firstLayer() const noexcept { return first_; }
        /** @return Main-model stage boundary, exclusive. */
        int mainLastLayer() const noexcept { return main_end_; }
        /** @return Routed boundary, including this stage's retained MTP layers. */
        int routedLastLayer() const noexcept { return routed_end_; }
        /** @return True only for the declared global output stage. */
        bool terminal() const noexcept { return terminal_; }
        /** @return Exact continuation participants in canonical order. */
        const std::vector<GlobalDeviceAddress> &participants() const & noexcept { return participants_; }
        /** @brief A temporary identity cannot lend a participant view past its lifetime. */
        const std::vector<GlobalDeviceAddress> &participants() const && = delete;
        /** @brief Compare every namespace coordinate across immutable observations. */
        bool operator==(const MoEOptimizationStageIdentity &) const = default;

    private:
        std::size_t index_;
        int first_, main_end_, routed_end_;
        std::vector<GlobalDeviceAddress> participants_;
        bool terminal_;
    };

    /**
     * @brief Complete ordered stage observations, sealed against partial geometry.
     * @tparam Snapshot Passive status, progress, topology, or movement value.
     *
     * An empty set belongs to a single authority. A nonempty set can only be
     * created in one transaction with contiguous main-layer ownership. Child
     * snapshots must be leaves: nested TP elects its own single publisher.
     */
    template<class Snapshot>
    class MoEOptimizationStages final
    {
    public:
        /** @brief One immutable value paired with its exact publication namespace. */
        struct Entry
        {
            MoEOptimizationStageIdentity identity;
            Snapshot value;
            /** @brief Compare scope and value without dropping subordinate identities. */
            bool operator==(const Entry &) const = default;
        };

        /** @brief A leaf observation has no subordinate stage authorities. */
        MoEOptimizationStages() = default;

        /**
         * @brief Seal all stage observations, preserving their publication order.
         * @param entries Complete pipeline, with no omitted or nested publisher.
         * @return Immutable stage set; rejection cannot publish a partial set.
         * @throws std::invalid_argument for missing, overlapping, reordered or nested stages.
         */
        static MoEOptimizationStages seal(std::vector<Entry> entries)
        {
            if (entries.empty())
                throw std::invalid_argument("Pipeline movement publication requires its complete stage set");
            int expected_first = 0;
            for (std::size_t i = 0; i < entries.size(); ++i)
            {
                const auto &entry = entries[i];
                if (entry.identity.index() != i || entry.identity.firstLayer() != expected_first ||
                    !entry.value.stages.empty() || entry.identity.terminal() != (i + 1 == entries.size()) ||
                    (i + 1 < entries.size() &&
                        entry.identity.routedLastLayer() != entry.identity.mainLastLayer()))
                    throw std::invalid_argument("Pipeline movement publication has foreign or incomplete stage ownership");
                expected_first = entry.identity.mainLastLayer();
            }
            MoEOptimizationStages result;
            result.entries_ = std::move(entries);
            return result;
        }
        /** @return True for a single-authority observation. */
        bool empty() const noexcept { return entries_.empty(); }
        /** @return Complete immutable stage observations in authored order. */
        std::span<const Entry> entries() const & noexcept { return entries_; }
        /** @brief Retain the observation before borrowing its stage entries. */
        std::span<const Entry> entries() const && = delete;
        /** @brief Preserve every stage coordinate when comparing immutable geometry. */
        bool operator==(const MoEOptimizationStages &) const = default;

    private:
        std::vector<Entry> entries_;
    };
}
