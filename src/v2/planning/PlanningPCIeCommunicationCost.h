/**
 * @file PlanningPCIeCommunicationCost.h
 * @brief Pure directed PCIe-cut wire bounds from original negotiated link observations.
 *
 * This helper owns immutable inventory facts, never probes devices, chooses a
 * collective algorithm or reserves memory. A negotiated wire capacity is an
 * upper bound on payload throughput, not a measured payload service. Its byte
 * cut quotient is therefore a lower bound only for the declared traffic.
 */
#pragma once
#include "backends/DeviceId.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace llaminar2 {
/** @brief One authored directed live transfer; padding/allocation capacity is not traffic. */
struct PlanningPCIeDirectedTransfer {
    DeviceId source;
    DeviceId destination;
    size_t live_bytes;
};
/** @brief One active observed cut and its independent full-duplex directional loads. */
struct PlanningPCIeCutObservation {
    std::string bdf;
    size_t outbound_bytes=0; ///< Descendants to endpoints outside this cut.
    size_t inbound_bytes=0; ///< Outside endpoints to descendants.
    std::optional<double> capacity_bytes_per_second;
    double seconds=0; ///< Zero means no qualified bound when capacity is absent.
    std::string qualification;
};
/** @brief Largest qualified cut floor plus explicit unresolved coverage. */
struct PlanningPCIeCutBound {
    double seconds=0;
    size_t known_cuts=0,unknown_cuts=0; ///< Active candidate cuts, not all PCI bridges.
    bool topology_complete=true; ///< False when a used endpoint lacks original path/link identity.
    std::vector<PlanningPCIeCutObservation> cuts;
    std::vector<std::string> evidence;
};
/** @brief Original-rank local GPU cut calculations; no remote CPU execution or inferred ring/tree. */
class PlanningPCIeCommunicationCost final {
public:
    /** @brief Copy the original inventory facts; later caller mutations cannot alter this observation.
     * @param inventory Original discovery rank with exact device ordinals and NUMA ownership.
     * @throws std::invalid_argument For ambiguous endpoint identities or an invalid original rank. */
    explicit PlanningPCIeCommunicationCost(const RankInventory &inventory);
    /** @brief Bound an algorithm-independent allreduce's necessary information across each nontrivial cut.
     * @param devices Nonempty unique homogeneous original-rank GPU participants.
     * @param live_bytes One live reduction payload, required once in each cut direction.
     * @return Zero for a singleton; otherwise maximum qualified full-duplex cut floor.
     * @throws std::invalid_argument For zero bytes, foreign/mixed/duplicate/CPU endpoints.
     * No all-to-all multiplicity, chunk schedule or collective latency is inferred. */
    PlanningPCIeCutBound allreduce(std::span<const DeviceId> devices,size_t live_bytes) const;
    /** @brief Bound unique-owner allgather information across each nontrivial cut.
     * @param devices Ordered unique homogeneous original-rank participants.
     * @param owner_live_bytes One exact fragment per corresponding owner; empty owners contribute zero.
     * @return Each owner's fragment crosses a cut once if any consumer lies beyond it.
     * @throws std::invalid_argument For empty/mismatched groups or an entirely empty payload.
     * @throws std::overflow_error For unrepresentable total live bytes.
     * Consumer count never multiplies a unique fragment; a singleton needs no transfer. */
    PlanningPCIeCutBound allgather(std::span<const DeviceId> devices,
        std::span<const size_t> owner_live_bytes) const;
    /** @brief Accumulate exactly the declared directed transfers on cuts they cross.
     * @param edges Authored sends; omit empty work rather than supplying zero/self edges.
     * @return Empty demand has zero cost; shared cuts sum each direction separately.
     * @throws std::invalid_argument For foreign, mixed, self or zero-byte edges.
     * @throws std::overflow_error If exact live byte sums cannot be represented. */
    PlanningPCIeCutBound transfers(std::span<const PlanningPCIeDirectedTransfer> edges) const;
private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};
}
