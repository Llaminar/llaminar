/**
 * @file PlanningInterconnectCostMatrix.h
 * @brief Pure directed physical peer-copy evidence for workload-aware GPU group selection.
 *
 * The matrix retains completed destination-owned direct peer-copy observations in their original
 * discovery namespace. Its scores compare placements on one common measured
 * basis; they are not predictions of an unobserved NCCL/RCCL collective. Memory
 * feasibility and compute cost remain the candidate admission/cost authorities.
 */
#pragma once
#include "PlanningLocalTPMeasurement.h"
#include <cstdint>
#include <map>
#include <string>

namespace llaminar2
{
    /** @brief Enclosing service-owned physical identity; no remote ordinal aliases. */
    struct PlanningInterconnectIdentity
    {
        int discovery_rank = -1, physical_node = -1;
        DeviceType backend = DeviceType::CPU;
        std::vector<std::string> uuids;
        bool operator==(const PlanningInterconnectIdentity &) const = default;
    };

    /** @brief One actual directed live payload, excluding arena padding. */
    struct PlanningInterconnectEdge
    {
        std::string source_uuid, destination_uuid;
        size_t payload_bytes = 0;
    };

    /** @brief Explicit concurrent edges; each sender/receiver occurs at most once. */
    using PlanningInterconnectRound = std::vector<PlanningInterconnectEdge>;

    /** @brief Caller-owned graph schedule and its workload occurrence count. */
    struct PlanningInterconnectPhaseDemand
    {
        std::vector<PlanningInterconnectRound> rounds;
        uint64_t occurrences = 0;
    };

    /** @brief Distinct live prefill/decode traffic; no inferred all-to-all graph. */
    struct PlanningInterconnectWorkload
    {
        PlanningInterconnectPhaseDemand prefill, decode;
    };

    /** @brief Median-observed directed curve value and exact interpolation brackets. */
    struct PlanningInterconnectService
    {
        double seconds = 0;
        size_t lower_payload_bytes = 0, upper_payload_bytes = 0;
    };

    /** @brief Placement-selection proxy only, separate from measured collective latency. */
    struct PlanningInterconnectWorkloadScore
    {
        double prefill_seconds = 0, decode_seconds = 0;
        /** @return Sum of explicit phase proxies, without compute or capacity decisions. */
        double seconds() const noexcept { return prefill_seconds + decode_seconds; }
    };

    /** @brief Immutable matrix; queries perform no device work or capacity accounting. */
    class PlanningInterconnectCostMatrix final
    {
    public:
        /**
         * @brief Validate complete physical peer-copy coverage on exactly the explicit enabled edges.
         * @param identity Original observer, physical node, homogeneous GPU backend and UUID order.
         * @param observations Exact sampler receipts, retaining all three raw event samples.
         * @param direct_peer_access Source-major NxN mask: destination can read source; self is zero.
         * @throws std::invalid_argument for missing/duplicate edges, invalid samples or CPU identity.
         *
         * A singleton has no transfers and requires an empty receipt. The enclosing
         * service authenticates this identity against discovery; no UUID/ordinal
         * substitution or cross-rank evidence reuse happens inside this class.
         */
        PlanningInterconnectCostMatrix(PlanningInterconnectIdentity identity,
            std::span<const PlanningLocalTPPairObservation> observations,
            std::span<const uint8_t> direct_peer_access);

        /** @return Whether the original physical destination can directly read the source allocation. */
        bool hasDirectPeer(const std::string &source_uuid, const std::string &destination_uuid) const;

        /** @return Original immutable enclosing identity. */
        const PlanningInterconnectIdentity &identity() const noexcept { return identity_; }
        /** @return Unmodified directed raw observations in original sample order. */
        const std::vector<PlanningLocalTPPairObservation> &observations() const noexcept { return observations_; }

        /**
         * @brief Price one directed live payload from exact or bounded piecewise-linear observations.
         * @param expected Same discovery identity expected by the candidate's evidence owner.
         * @param source_uuid Original physical sender UUID.
         * @param destination_uuid Original physical receiver UUID; reverse direction is independent.
         * @param payload_bytes Positive live bytes within measured anchors; never allocation extent.
         * @throws std::invalid_argument for stale identity, absent endpoints or out-of-range payload.
         *
         * Each anchor uses the median observed service of its three completed raw
         * intervals. No extrapolated bandwidth, monotonicity repair, reverse-edge
         * substitution or assumed zero-byte latency is introduced.
         */
        PlanningInterconnectService directedService(const PlanningInterconnectIdentity &expected,
            const std::string &source_uuid, const std::string &destination_uuid, size_t payload_bytes) const;

        /**
         * @brief Score explicit live graph rounds on a common pair-observation basis.
         * @param expected Exact enclosing identity, checked even for a singleton.
         * @param group Selected physical UUIDs, in candidate order; any positive size is supported.
         * @param workload Actual prefill/decode schedules and occurrence counts.
         * @return Sum of per-round slowest-edge curves, multiplied by phase occurrences.
         * @throws std::invalid_argument for incomplete/incompatible schedules or identity drift.
         *
         * A round permits full-duplex send/receive but not two sends or two
         * receives on one endpoint. Concurrent contention remains unmeasured:
         * this is selection evidence, not an exact collective timing. Callers
         * must supply a known schedule or label their schedule a proxy; this
         * API never invents ring/tree edges from the participant count.
         */
        PlanningInterconnectWorkloadScore groupWorkloadScore(const PlanningInterconnectIdentity &expected,
            std::span<const std::string> group, const PlanningInterconnectWorkload &workload) const;

    private:
        /** @brief Reject a changed discovery namespace before using any receipt. */
        void requireIdentity(const PlanningInterconnectIdentity &expected) const;
        /** @return Exact original UUID index, rejecting absent endpoints. */
        size_t indexOf(const std::string &uuid) const;
        PlanningInterconnectIdentity identity_;
        std::vector<PlanningLocalTPPairObservation> observations_;
        std::vector<uint8_t> direct_peer_access_;
        /** Immutable indexes avoid rescanning the full matrix for every candidate edge. */
        std::map<std::string, size_t> indices_;
        std::map<std::pair<size_t, size_t>, std::vector<size_t>> curves_;
    };
}
