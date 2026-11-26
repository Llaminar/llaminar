/**
 * @file RetainedGDNPreAttentionStage.h
 * @brief Participant-local original POST/HC plus GDN projection declaration.
 */
#pragma once
#include "execution/compute_stages/stages/GDNProjectionStage.h"

namespace llaminar2
{
    class ComputeGraph;
    class Qwen4ExpHCReadPhaseOwner;

    /** @return Shared production workspace quote for graph materialization and PMA admission. */
    [[nodiscard]] WorkspaceRequirements retainedGDNPreAttentionWorkspace();

    /**
     * @brief Own the exact local ordinary M1 ROCm GDN projection boundary.
     * @param projection Unchanged prepared GDN projection declarations.
     * @param hc Original POST/HC read owner with the independent injection remaining outside.
     * @return Cold preparation-only stage borrowing both original declarations.
     */
    [[nodiscard]] std::unique_ptr<IComputeStage> makeRetainedGDNPreAttentionStage(
        const GDNProjectionStage::Params &projection,
        std::shared_ptr<Qwen4ExpHCReadPhaseOwner> hc);

    /**
     * @brief Contract an original participant-local GDN projection and its HC producer chain.
     * @param graph Merged cold participant DAG.
     * @param gdnProjectionName Original exact GDN projection node to retain.
     * @param previousInputBoundary Name of the merged prefix frontier, or empty at graph entry.
     * @details A direct typed write with the exact norm input is retained.
     * Other frontiers remain outside and publish the original HC trunk.
     */
    void lowerRetainedQwenGDNPreAttention(ComputeGraph &graph, const std::string &gdnProjectionName,
                                        const std::string &previousInputBoundary);

#ifdef HAVE_CUDA
    /** @return Shared production workspace quote for CUDA retained full-GDN graph materialization and PMA admission. */
    [[nodiscard]] WorkspaceRequirements retainedCUDAGDNWorkspace();

    /**
     * @brief Contract a participant-local CUDA GDN attention block (HC prefix + GDN proj + conv + rec + ssm_out)
     *        into one single cooperative megakernel across any CUDA topology.
     * @param graph Merged cold participant DAG.
     * @param gdnProjectionName Original exact GDN projection node name.
     * @param previousInputBoundary Name of the merged prefix frontier, or empty at graph entry (Layer 0).
     */
    void lowerRetainedCUDAGDN(ComputeGraph &graph, const std::string &gdnProjectionName,
                              const std::string &previousInputBoundary);
#endif
}
