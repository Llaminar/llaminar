/**
 * @file RetainedQSAAttentionStage.h
 * @brief Participant-local CUDA and ROCm QSA Pre-Attention and Post-Attention megakernel lowering.
 *
 * Contracts each QFN Sparse Attention (QSA) layer from ~35 kernels into 4:
 *   1. RetainedQSAPreAttention (1 cooperative kernel: HC prefix + replay/row_params +
 *      Wq[split Q/fa_gate] + Wk + Wv + qsa_key + optional ranking-guarded qsa_query +
 *      QK RMSNorm + 64-dim RoPE + FP16 KV ring append + ring_head/cached_tokens advance)
 *   2. Standalone qsa_indexer
 *   3. Standalone selectedAttentionFP16Tiles (launched inside RetainedQSAPostAttention)
 *   4. RetainedQSAPostAttention (1 cooperative kernel: foldAttentionPrefix + commitPartial +
 *      sigmoid fa_gate + blockwise Q8 staging + Wo projection -> attn_proj)
 *
 * General across all GPU topologies, TP widths, MoE placements, and all 18 NativeQuantised
 * codebooks plus BF16/FP16/FP32 indexer weights.
 */
#pragma once

#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include <string>

namespace llaminar2
{
    class ComputeGraph;

#ifdef HAVE_CUDA
    /** @return Shared production workspace quote for CUDA retained QSA megakernels. */
    [[nodiscard]] WorkspaceRequirements retainedCUDAQSAAttentionWorkspace();

    /**
     * @brief Contract a participant-local CUDA QSA layer's Pre-Attention and Post-Attention
     *        subgraphs into cooperative megakernels around the standalone indexer.
     * @param graph Merged cold participant DAG.
     * @param layerPrefix Exact layer node prefix, e.g. "layer3_".
     * @param previousInputBoundary Name of the merged prefix frontier, or empty at graph entry.
     */
    void lowerRetainedCUDAQSAAttention(ComputeGraph &graph,
                                       const std::string &layerPrefix,
                                       const std::string &previousInputBoundary);
#endif

#ifdef HAVE_ROCM
    /** @return Shared production workspace quote for ROCm retained QSA megakernels. */
    [[nodiscard]] WorkspaceRequirements retainedROCmQSAAttentionWorkspace();

    /**
     * @brief Contract a participant-local ROCm QSA layer's Pre-Attention and Post-Attention
     *        subgraphs into cooperative megakernels around the standalone indexer.
     * @param graph Merged cold participant DAG.
     * @param layerPrefix Exact layer node prefix, e.g. "layer3_".
     * @param previousInputBoundary Name of the merged prefix frontier, or empty at graph entry.
     */
    void lowerRetainedROCmQSAAttention(ComputeGraph &graph,
                                       const std::string &layerPrefix,
                                       const std::string &previousInputBoundary);
#endif
} // namespace llaminar2
