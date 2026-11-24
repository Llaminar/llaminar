/**
 * @file DeepSeekV4Graph.h
 * @brief Participant-local DeepSeek-V4 graph over shared MoE stages and DeepSeek-V4 primitives.
 *
 * Every main layer is
 *   Sinkhorn HC read -> compressed sparse MQA attention -> HC write ->
 *   Sinkhorn HC read -> sqrt-softplus MoE (+ shared expert) -> HC write
 * with the residual trunk (hc x d) carried in HIDDEN_STATE. Embedding expansion,
 * the block read/write pairing and the terminal head read reuse the
 * QwenGraphBase residual-trunk hooks introduced for Qwen4Exp. Projections use
 * ordinary prepared GEMM stages; DeepSeek-V4 arithmetic and position-indexed
 * state banks use DeepSeekV4PrimitiveStage. Raw kv is also appended to the
 * canonical KV cache so the request frontier, positions and reset lifecycle are
 * owned by the same authority as every other model.
 *
 * Scope of this first implementation: one participant (CPU or CUDA), one
 * request per graph, no prefix-cache state archive and no MTP/DSpark drafter.
 */
#pragma once

#include "../qwen35moe/Qwen35MoEGraph.h"
#include "DeepSeekV4BufferLayout.h"
#include "../../execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include "../../kernels/deepseek4/DeepSeekV4Kernels.h"

#include <optional>

namespace llaminar2
{
/** @brief DeepSeek-V4 (deepseek4 GGUF) graph builder. */
class DeepSeekV4Graph final : public Qwen35MoEGraph
{
public:
    /** @brief Construct a declaration; the production factory injects its model context next. */
    explicit DeepSeekV4Graph(const GraphConfig &config, std::shared_ptr<IMPIContext> mpi = nullptr);

    std::string architectureName() const override { return "deepseek4"; }
    GraphSchema getSchema() const override;
    GraphResolverConfig getResolverConfig(int seq_len) const override;
    void setArena(BufferArena *arena) override;
    /**
     * @brief Validate the frontier-only cache and publish the carried state banks.
     * @details The raw window ring and FP32 compressor state are read by a
     * graph before it overwrites them (they carry the previous chunk's rows),
     * so they are graph-external inputs that must be device-valid before the
     * first transaction. They are zeroed and published once here, before any
     * capture; their bytes are never read for positions this request has not
     * written, so request reset needs no further publication. The pools are
     * always produced in-graph before their readers.
     */
    void initializeCacheAuxiliaryState(IKVCache &cache, void *publicationStream) override;

    /** @brief DeepSeek-V4 compressed sparse attention block (one participant). */
    ComputeGraph buildAttentionGraph(
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int batch_size,
        IKVCache *kv_cache,
        const int *position_ids,
        DeviceId device,
        const std::vector<int> *sequence_lengths = nullptr,
        const void *position_ids_device = nullptr,
        const int32_t *sequence_lengths_device = nullptr) override;

private:
    /** @brief State-bank geometry derived from the admitted raw-ring allocation. */
    DeepSeekV4StateLayout stateLayout() const;
    /** @return The vision image bank and position map of a vision checkpoint. */
    std::optional<PromptEmbeddingBank> promptEmbeddingBank() const override;
    /** @return Vision sentinel ids of a vision checkpoint, or the disabled set for text-only. */
    deepseek4::ImageSentinelIds imageSentinelIds() const;
    /** @return Vision: this forward's execution-device token row; nullopt for text-only checkpoints. */
    std::optional<Qwen4ExpIntInput> imageTokens(int rows, DeviceId device) const;
};
} // namespace llaminar2
