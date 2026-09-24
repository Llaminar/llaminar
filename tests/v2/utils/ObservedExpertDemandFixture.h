/**
 * @file ObservedExpertDemandFixture.h
 * @brief Model-free admission and ingress helpers for explicit routed invocations.
 *
 * These helpers fabricate test workloads, never production evidence. Every batch
 * has named phase and row geometry, enters the real histogram, and is retained by
 * its normal RCU/PMA lifecycle. Marginals are derived by that ingress rather than
 * injected separately from an unrelated token count.
 */
#pragma once

#include "execution/moe/DecodeExpertHistogram.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
    /**
     * @brief Admit the specified mutable banks and bounded concurrently retained samples.
     * @param config Histogram geometry, ownership and token-boundary policy.
     * @param capacity Explicit test sample and maximum actual invocation dimensions.
     * @param snapshots Maximum live immutable samples, including transient replacement.
     * @param rank World rank owning these host allocations.
     * @param mailbox_bytes Optional existing MPI mailbox allocation on the same ledger.
     */
    inline void admitObservedExpertDemand(
        DecodeExpertHistogramConfig &config,
        moe_overlay_economy::TransactionDemandCapacity capacity,
        std::size_t snapshots, int rank = 0, std::size_t mailbox_bytes = 0)
    {
        const unsigned __int128 bytes =
            static_cast<unsigned __int128>(2) * config.num_layers * capacity.allocationBytes() +
            static_cast<unsigned __int128>(snapshots) *
                DecodeExpertTransactionWindow::maximumAllocationBytes(
                    capacity, config.num_layers, config.num_experts) + mailbox_bytes;
        if (snapshots == 0 || bytes > std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("Observed test demand BOM is invalid");
        const auto total = static_cast<std::size_t>(bytes);
        PhysicalMemoryBOMBuilder bom({.world_rank = rank, .device = DeviceId::cpu(),
            .total_bytes = total, .admission_available_bytes = total});
        bom.add(PhysicalMemoryOwner::ExecutionWorkspace, total);
        PhysicalMemoryPlanBuilder plan;
        plan.add(bom.build());
        config.transaction_demand = ExpertHistogramTransactionConfig{
            capacity, std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()), rank)};
    }

    /**
     * @brief Publish one complete, compact invocation through production ingress.
     * @throws On invalid geometry or failed admission; no partial count injection.
     */
    inline void recordObservedExpertBatch(
        DecodeExpertHistogram &histogram, int layer, ExpertHistogramSource phase,
        std::span<const int> routes)
    {
        const int top_k = histogram.config().top_k;
        if (top_k <= 0 || routes.empty() || routes.size() % top_k != 0 ||
            routes.size() / top_k > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Observed test invocation has invalid row geometry");
        const int rows = static_cast<int>(routes.size() / top_k);
        std::vector<std::uint64_t> scratch(histogram.config().num_experts);
        const auto result = histogram.mergeRoutedExpertRows(routes.data(), {
            .source = phase, .layer_idx = layer, .real_token_count = rows,
            .bucket_token_count = rows, .top_k = top_k, .route_stride = top_k,
            .count_window_tokens = true}, scratch);
        if (!result)
            throw std::logic_error("Observed test invocation publication: " + result.error);
    }

    /**
     * @brief Specify row multiplicities inside exactly one top-one prefill invocation.
     * @throws If the declared geometry is not top-one or the batch is empty.
     *
     * This is an explicit synthetic parallel workload, not reconstruction of
     * unknown production co-occurrence. Decode fixtures supply individual rows.
     */
    inline void recordObservedPrefillBatch(
        DecodeExpertHistogram &histogram, int layer,
        std::span<const std::uint64_t> row_multiplicities)
    {
        if (histogram.config().top_k != 1 ||
            row_multiplicities.size() != static_cast<std::size_t>(histogram.config().num_experts))
            throw std::invalid_argument("Observed prefill fixture has invalid expert geometry");
        std::vector<int> routes;
        for (std::size_t expert = 0; expert < row_multiplicities.size(); ++expert)
        {
            if (row_multiplicities[expert] >
                static_cast<std::size_t>(std::numeric_limits<int>::max()) - routes.size())
                throw std::overflow_error("Observed prefill fixture exceeds ingress geometry");
            routes.insert(routes.end(), row_multiplicities[expert], static_cast<int>(expert));
        }
        recordObservedExpertBatch(histogram, layer, ExpertHistogramSource::PrefillChunk, routes);
    }
}
