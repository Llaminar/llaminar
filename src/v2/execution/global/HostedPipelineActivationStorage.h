/**
 * @file HostedPipelineActivationStorage.h
 * @brief One cold-admitted CPU activation bank for serial hosted pipeline boundaries.
 *
 * Maximum retained rows size storage only. Every original MPI handoff still
 * transmits exactly current live rows, including partial/shrinking replays.
 * Source and destination stage graphs borrow this stable parent-owned tensor;
 * no inference-time resize, replacement allocation or CPU graph capture occurs.
 */
#pragma once
#include "memory/CPUStorageRoot.h"
#include "execution/global_pp/GlobalPPRankPlan.h"
#include "execution/global_pp/GlobalPPTopology.h"
#include <algorithm>
#include <limits>
namespace llaminar2 {
/** @brief Shared immutable declaration consumed by both BOM and parent materialization. */
struct HostedPipelineActivationStorage {
    /**
     * @brief Resolve one original rank's actual hosted boundary-bank incidence.
     * @param topology Complete original stage/device ownership declaration.
     * @param rank Original rank's existing compiled transfer actions.
     * @return True for receives, local handoffs or GPU sends requiring CPU staging.
     * CPU sends already own their host-readable source. A GPU send must stage
     * into a separately admitted bank so reading a live prefix never publishes
     * its complete source host mirror. All serial edges on one rank share it.
     */
    static bool requiresBank(const GlobalPPTopology &topology, const GlobalPPRankPlan &rank) {
        const auto transfers = rank.transferActions();
        return std::any_of(transfers.begin(), transfers.end(), [&](const auto *action) {
            if (action->direction == RankTransferAction::Direction::RECV ||
                action->direction == RankTransferAction::Direction::LOCAL_HANDOFF)
                return true;
            if (action->direction != RankTransferAction::Direction::SEND) return false;
            const auto &source = topology.stages.at(static_cast<std::size_t>(action->from_stage));
            const auto &members = source.is_global_tp ? source.per_rank_devices : source.devices;
            return std::any_of(members.begin(), members.end(), [](const auto &member) {
                return member.toLocalDeviceId().is_gpu();
            });
        });
    }
    /**
     * @brief Declare the exact shared serial-boundary root.
     * @param rows Admitted largest actual stage invocation, never communication extent.
     * @param width Full replicated FP32 model activation width.
     * @return Original CPU allocator declaration with the canonical physical geometry.
     * @throws std::invalid_argument for absent/overflowed source geometry.
     */
    static CPUStorageRoot<float> forRows(int rows,int width) {
        if(rows<=0 || width<=0 || std::size_t(rows)>std::numeric_limits<std::size_t>::max()/std::size_t(width))
            throw std::invalid_argument("Hosted pipeline activation requires its admitted row/width envelope");
        return CPUStorageRoot<float>(std::size_t(rows)*std::size_t(width),
            PhysicalMemoryOwner::ActivationTransportStaging,DeviceId::cpu());
    }
};
}
