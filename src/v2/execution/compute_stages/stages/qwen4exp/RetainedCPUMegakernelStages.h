/**
 * @file RetainedCPUMegakernelStages.h
 * @brief Participant-local CPU single-OMP two-megakernel-per-layer lowering for QFN.
 *
 * Contracts each QFN layer on CPU into two single-OMP megakernels bounded by
 * explicit TP collectives (no nested multi-device graphs, no hardcoded device
 * counts, general across all 18 NativeQuantised codebooks + floating weights
 * and Scalar / AVX2 / AVX-512 runtime ISAs via `activeISALevel()`):
 *   1. Attention Megakernel (`RetainedCPUGDNStage` for GDN layers;
 *      `RetainedCPUQSAAttentionStage` for QSA layers): fuses HC write/norm/down/
 *      inject/up/read, attention projections with shared Q8_1 quantization,
 *      ShortConv1d + GDNRecurrence + GatedRMSNorm (or QGateSplit + QKNorm + RoPE
 *      + QSA PrepareReplay + KV Append + QSA Indexer + SelectedAttention +
 *      CommitPartial + AttentionOutputGate), and output projection into ONE
 *      `OMP_LAYER_PARALLEL` region.
 *   2. FFN/MoE Megakernel (`RetainedCPUFFNStage`): fuses HC attention-write +
 *      FFN branch-norm/down/inject/up/read, MoE routing + router Q8_1
 *      publication, Shared Expert FFN, and (on singleton CPU topologies without
 *      intervening collectives) Routed Expert Standalone MoE + Shared Expert
 *      Gate into ONE `OMP_LAYER_PARALLEL` region.
 */
#pragma once

#include <string>

namespace llaminar2
{
    class ComputeGraph;

    /**
     * @brief Contract a participant-local CPU GDN attention block into one single-OMP megakernel.
     * @param graph Cold participant-local ComputeGraph.
     * @param gdnProjectionName Original GDN projection node name (e.g. "layer0_gdn_proj").
     * @param previousInputBoundary Preceding layer terminal node name, or empty at graph entry.
     */
    void lowerRetainedCPUGDN(ComputeGraph &graph,
                             const std::string &gdnProjectionName,
                             const std::string &previousInputBoundary);

    /**
     * @brief Contract a participant-local CPU QSA attention block into one single-OMP megakernel.
     * @param graph Cold participant-local ComputeGraph.
     * @param layerPrefix Exact layer node prefix (e.g. "layer3_").
     * @param previousInputBoundary Preceding layer terminal node name, or empty at graph entry.
     */
    void lowerRetainedCPUQSAAttention(ComputeGraph &graph,
                                      const std::string &layerPrefix,
                                      const std::string &previousInputBoundary);

    /**
     * @brief Contract a participant-local CPU FFN/MoE block into one single-OMP megakernel.
     * @param graph Cold participant-local ComputeGraph.
     * @param layerPrefix Exact layer node prefix (e.g. "layer0_").
     * @param previousAttentionBoundary Attention terminal node name feeding the FFN HC prefix.
     */
    void lowerRetainedCPUFFN(ComputeGraph &graph,
                             const std::string &layerPrefix,
                             const std::string &previousAttentionBoundary);

} // namespace llaminar2
