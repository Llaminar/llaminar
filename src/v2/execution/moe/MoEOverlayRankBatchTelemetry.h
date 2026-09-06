/**
 * @file MoEOverlayRankBatchTelemetry.h
 * @brief Bounded observations for both physical rank-batch transports.
 *
 * Transport validates and executes the protocol; this observer owns no runtime
 * state or placement decision. Only topology and retained graph geometry are
 * aggregation keys. Bytes are measurements, while generation/step/sequence
 * enter the existing ordered witness, never an ever-growing per-token map.
 * The observer borrows one synchronous call's key and allocates no tag strings
 * unless the corresponding PerfStats domain was requested.
 */
#pragma once

#include "MoEOverlayRankBatchTransport.h"
#include "utils/PerfStatsCollector.h"
#include <stdexcept>

namespace llaminar2
{
    /** @brief Which immutable rank-pair endpoint observed this exchange. */
    enum class MoEOverlayRankBatchEndpoint { Source, Target };

    /** @brief Shared, diagnostic-only metric contract for MPI and shared pages. */
    class MoEOverlayRankBatchTelemetry final
    {
    public:
        /**
         * @brief Borrow a key already authenticated by the transport protocol.
         * @param key Complete exchange identity, alive through all observations.
         * @param kind Physical transport actually selected by topology admission.
         * @param endpoint Observing endpoint, independent of dispatch/return direction.
         */
        MoEOverlayRankBatchTelemetry(
            const MoEOverlayRankBatchKey &key,
            MoEOverlayRankBatchTransportKind kind,
            MoEOverlayRankBatchEndpoint endpoint) noexcept
            : key_(key), kind_(kind), endpoint_(endpoint) {}

        /**
         * @brief Retain transaction count, exact logical bytes and ordered identity.
         * @param bytes Live payload bytes, not the workspace's full capacity.
         * @param participant_count Immutable participant count in this envelope.
         *
         * Sender and receiver each publish their own observation. Their rows
         * remain separate, rather than claiming to be a physical-traffic ledger.
         */
        void recordTransaction(std::size_t bytes, std::size_t participant_count) const
        {
            if (!PerfStatsCollector::isDomainEnabled("forward_graph"))
                return;
            auto tags = topologyTags();
            tags.emplace("participant_count", std::to_string(participant_count));
            PerfStatsCollector::addCounter(
                "forward_graph",
                key_.direction == MoEOverlayCollectiveDirection::Dispatch
                    ? "moe_overlay_rank_batch_dispatch_transactions"
                    : "moe_overlay_rank_batch_return_transactions",
                1.0, "moe_overlay", transportName(), tags);
            PerfStatsCollector::addCounter(
                "forward_graph", "moe_overlay_rank_batch_payload_bytes",
                static_cast<double>(bytes), "moe_overlay", transportName(), tags);
            // Preserve ordering without making a separate map entry for every
            // request/token. Runtime key validation remains the actual authority.
            PerfStatsCollector::recordOrderedSequenceStep(
                "forward_graph", "moe_overlay_rank_batch_sequence",
                {key_.generation_id, key_.step_id, key_.sequence, bytes},
                phase(), transportName(), tags);
        }

        /**
         * @brief Aggregate completed wire/codec/total intervals by the same geometry.
         * @param codec_ns MPI encode/decode time; zero for in-place shared rows.
         * @param wait_ns Peer publication/wire wait, including peer computation.
         * @param total_ns Complete local exchange duration in nanoseconds.
         */
        void recordTimings(uint64_t codec_ns, uint64_t wait_ns, uint64_t total_ns) const
        {
            if (!PerfStatsCollector::isDomainEnabled("moe_overlay_transport"))
                return;
            const auto tags = topologyTags();
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_transport", "rank_batch_total", total_ns,
                phase(), transportName(), tags);
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_transport", "rank_batch_wire_wait", wait_ns,
                phase(), transportName(), tags);
            if (kind_ == MoEOverlayRankBatchTransportKind::MPI)
                PerfStatsCollector::recordTimingNs(
                    "moe_overlay_transport", "rank_batch_codec", codec_ns,
                    phase(), transportName(), tags);
            else
                PerfStatsCollector::addCounter(
                    "moe_overlay_transport", "rank_batch_zero_copy_publications",
                    1.0, phase(), transportName(), tags);
        }

        /**
         * @brief Count an actual nonblocking MPI submission and its retained ring size.
         * @throws std::logic_error If called for a shared-page exchange with no MPI send.
         */
        void recordAsyncSendSubmission(std::size_t slot_count) const
        {
            if (kind_ != MoEOverlayRankBatchTransportKind::MPI)
                throw std::logic_error("Shared rank-batch rows cannot claim an MPI submission");
            if (!PerfStatsCollector::isDomainEnabled("moe_overlay_transport"))
                return;
            auto tags = topologyTags();
            tags.emplace("slots", std::to_string(slot_count));
            PerfStatsCollector::addCounter(
                "moe_overlay_transport", "rank_batch_async_send_submissions",
                1.0, phase(), transportName(), tags);
        }

    private:
        /** @return Stable mathematical phase, never a logical token/chunk identity. */
        const char *phase() const noexcept
        {
            return key_.histogram_source == ExpertHistogramSource::PrefillChunk
                ? "prefill" : (key_.histogram_source == ExpertHistogramSource::GroupedVerifier
                    ? "grouped_verifier" : "decode");
        }

        /** @return The selected physical transport's existing diagnostic spelling. */
        const char *transportName() const noexcept
        {
            return kind_ == MoEOverlayRankBatchTransportKind::MPI
                ? "mpi" : "node_local_shared_rows";
        }

        /** @return Only topology/graph-family dimensions, bounded over request lifetime. */
        PerfStatsCollector::Tags topologyTags() const
        {
            return {
                {"direction", llaminar2::toString(key_.direction)},
                {"domain_ordinal", std::to_string(key_.domain_ordinal)},
                {"endpoint_role", endpoint_ == MoEOverlayRankBatchEndpoint::Source ? "source" : "target"},
                {"layer", std::to_string(key_.layer_idx)},
                {"mtp_depth", std::to_string(key_.mtp_depth)},
                {"source_world_rank", std::to_string(key_.source_world_rank)},
                {"target_world_rank", std::to_string(key_.target_world_rank)},
                {"tier", std::to_string(key_.tier_idx)},
                {"transport", transportName()},
            };
        }

        const MoEOverlayRankBatchKey &key_; ///< Borrowed authenticated call identity.
        MoEOverlayRankBatchTransportKind kind_; ///< Immutable selected transport.
        MoEOverlayRankBatchEndpoint endpoint_; ///< Immutable observing rank role.
    };
}
