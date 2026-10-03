/**
 * @file MoERoutingPipeline.h
 * @brief Declarative complete-row distribution of ordinary GPU prefill routing.
 *
 * Models provide their existing router declaration and, for a proven no-P2P
 * projection domain, its admitted packet banks. The reusable builder lowers
 * local math, an explicit exact-count collective and complete publication.
 * Small/decode/verifier graphs retain their separate replicated contract.
 */
#pragma once
#include "MoERoutingStage.h"
#include "MoEProjectionPipeline.h"

namespace llaminar2
{
    /** @brief Optional distributed-prefill resources, not another ownership controller. */
    struct MoERoutingDistribution
    {
        std::shared_ptr<DeviceCountedAllGather> fabric;
        int participant = -1;
        MoEProjectionTensorBinding local_packet, received_packets;
        /** @brief Borrow a domain's installed no-P2P fabric and exact arena roles.
         * @param domain Canonical ordered TP membership; enabled P2P remains native.
         * @param device This graph's exact physical participant.
         * @param buffers Admitted graph-local activation view.
         * @return No distribution for a native-P2P domain; otherwise authenticated bindings.
         * @throws std::invalid_argument For foreign/inconsistent membership. */
        static MoERoutingDistribution fromDomain(ILocalTPContext &domain,
            DeviceId device, const ActivationBuffers &buffers);
    };

    /** @brief Lower one router frontier with no backend work or new allocations.
     * @param graph Participant-local model graph.
     * @param route Complete ordinary router declaration; its kernel/workspace owner is retained.
     * @param node Semantic complete-route frontier used by existing downstream consumers.
     * @param input_node Graph producer of normalized hidden rows.
     * @param distribution Optional admitted no-P2P resources, supplied only for ordinary prefill.
     * @throws std::invalid_argument For inconsistent declarations or aliased buffers.
     *
     * Bulk buckets >=512 amortize the measured exchange cost on CUDA and ROCm.
     * This is an immutable bucket policy, never a host observation of live rows.
     * Packet arena lifetimes end at publication before later projection packing;
     * both bank identities occur explicitly in all three node contracts.
     */
    void appendMoERoutingPipeline(ComputeGraph &graph, MoERoutingStage::Params route,
        std::string node, const std::string &input_node, MoERoutingDistribution distribution = {});
}
