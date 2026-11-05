/**
 * @file DeepSeekV4PrimitiveStage.h
 * @brief Declarative backend-neutral DeepSeek-V4 stages with mandatory arena bindings.
 *
 * One closed operation variant per DeepSeek-V4 primitive. The binding
 * vocabulary (typed arena views, borrowed execution-owner INT32 inputs,
 * immutable FP32 model weights and row extents) is shared with the Qwen4Exp
 * primitive stages so both hyper-connection families compose through the same
 * graph contracts. Projections and ordinary norms stay in their existing
 * prepared GEMM/RMSNorm stages; this stage owns only model-specific arithmetic
 * and position-indexed state banks (raw window ring, compressor rings and
 * compressed pools, stored as packed records in INT32 arena words whose layout
 * is owned by DeepSeekV4KVFormat.h). All views and policies are sealed at construction;
 * execute verifies prepared storage and submits on the exact executor stream.
 */
#pragma once
#include "../qwen4exp/Qwen4ExpPrimitiveStage.h"
#include "../../../../kernels/deepseek4/DeepSeekV4Kernels.h"
#include "../../../../kernels/IKVCache.h"

#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace llaminar2
{
/** @brief Immutable model weight in any native format (Q8_0/BF16/FP32/INT32 payload). */
struct DeepSeekV4NativeWeightBinding
{
    const TensorBase *tensor; ///< Frozen model-owned tensor; its native type selects decoding.
    /** @brief Bind the exact native model tensor without taking ownership. */
    explicit DeepSeekV4NativeWeightBinding(const TensorBase &source) : tensor(&source) {}
};

/** @brief One semantic DeepSeek-V4 operation selected declaratively by its variant. */
class DeepSeekV4PrimitiveStage final : public IComputeStage
{
public:
    /** @brief Hyper-connection read: Sinkhorn split, mixed read and optional output RMSNorm. */
    struct HCRead
    {
        int hc, hidden;
        bool head;                                  ///< Terminal head (pre only).
        float normEps, hcEps, outputEps;
        int sinkhornIters;
        Qwen4ExpFloatBinding trunk;                 ///< [M, hc*H].
        Qwen4ExpFloatBinding mixes;                 ///< [M, mixWidth] GEMM output.
        Qwen4ExpImmutableWeightBinding scale;       ///< [3] or [1].
        Qwen4ExpImmutableWeightBinding base;        ///< [mixWidth].
        std::optional<Qwen4ExpImmutableWeightBinding> gain; ///< [H] output norm gain.
        std::optional<Qwen4ExpFloatBinding> mixed;  ///< Optional [M, H] pre-norm read (diagnostic).
        Qwen4ExpFloatBinding output;                ///< [M, H].
        std::optional<Qwen4ExpFloatBinding> post;   ///< [M, hc] (block sites).
        std::optional<Qwen4ExpFloatBinding> comb;   ///< [M, hc*hc] (block sites).
    };
    /** @brief Hyper-connection write; output may alias the trunk exactly. */
    struct HCWrite
    {
        int hc, hidden;
        Qwen4ExpFloatBinding block, post, comb, trunk, output;
    };
    /** @brief Expand embeddings into the trunk. */
    struct HCExpand
    {
        int hc, hidden;
        Qwen4ExpFloatBinding input, output;
    };
    /** @brief In-place per-head RMS (optional gain), partial RoPE and optional QAT rounding. */
    struct HeadNormRope
    {
        int heads, headDim;
        bool normalize;
        float eps;
        std::optional<Qwen4ExpImmutableWeightBinding> gain;
        deepseek4::RopePolicy rope;
        bool inverse;
        deepseek4::ActivationQuant quant;           ///< Official QAT post-operation.
        std::optional<Qwen4ExpIntInput> positions;  ///< Required when rope.ropeDim > 0.
        Qwen4ExpFloatBinding data;                  ///< [M, heads*headDim], in place.
    };
    /** @brief Carry the chunk's last stateRows() projections (score + ape) into the FP32 state. */
    struct CompressorStore
    {
        deepseek4::CompressorGeometry geometry;
        Qwen4ExpIntInput positions;
        Qwen4ExpFloatBinding kv, score;             ///< Chunk scratch [M, projectionWidth].
        Qwen4ExpImmutableWeightBinding ape;         ///< [ratio, projectionWidth] FP32.
        Qwen4ExpFloatBinding stateKV, stateScore;   ///< Persistent [stateRows, projectionWidth].
    };
    /** @brief Pool completed blocks (chunk scratch + carried state) into the packed pool. */
    struct CompressorPool
    {
        deepseek4::CompressorGeometry geometry;
        Qwen4ExpIntInput positions;
        Qwen4ExpFloatBinding kv, score;             ///< Chunk scratch [M, projectionWidth] (score without ape).
        Qwen4ExpImmutableWeightBinding ape;         ///< [ratio, projectionWidth] FP32.
        Qwen4ExpFloatBinding stateKV, stateScore;   ///< Persistent [stateRows, projectionWidth].
        Qwen4ExpImmutableWeightBinding gain;
        float eps;
        deepseek4::RopePolicy rope;
        Qwen4ExpIntBinding pool;                    ///< Persistent packed [poolRows, entryWords] INT32 words.
    };
    /** @brief Pack the chunk's last @c window kv rows into the persistent window ring. */
    struct RawStore
    {
        deepseek4::PackedKVEntryFormat format;
        int window;
        Qwen4ExpIntInput positions;
        Qwen4ExpFloatBinding kv;                    ///< [M, headDim] QAT kv.
        Qwen4ExpIntBinding ring;                    ///< Persistent packed [window, entryWords].
    };
    /** @brief Lightning-indexer scoring and top-k selection over packed MXFP4 keys. */
    struct IndexerTopK
    {
        int heads;
        deepseek4::PackedIndexerKeyFormat format;
        int ratio, topK, poolRows, scoreRows;
        float weightScale;
        Qwen4ExpIntInput positions;
        Qwen4ExpFloatBinding query, weights;
        Qwen4ExpIntBinding pool;                    ///< Packed [poolRows, entryWords].
        std::optional<Qwen4ExpFloatBinding> scores; ///< [scoreRows, poolRows] bounded scratch; required on GPU.
        Qwen4ExpIntBinding selected;                ///< [M, topK].
    };
    /** @brief Sparse sink attention with output de-rotation over chunk kv and packed records. */
    struct SparseAttention
    {
        int heads;
        deepseek4::PackedKVEntryFormat format;
        int window, ratio, topK, poolRows;
        deepseek4::CompressedAccess access;
        float scale;
        deepseek4::RopePolicy rope;
        Qwen4ExpIntInput positions;
        Qwen4ExpFloatBinding query, chunkKV;        ///< [M, heads*headDim], [M, headDim].
        Qwen4ExpIntBinding rawRing;                 ///< Packed [window, entryWords].
        std::optional<Qwen4ExpIntBinding> pool;     ///< Packed [poolRows, entryWords].
        std::optional<Qwen4ExpIntBinding> selected;
        Qwen4ExpImmutableWeightBinding sinks;
        Qwen4ExpFloatBinding output;
        std::optional<Qwen4ExpIntInput> tokens;     ///< Vision: chunk token ids (image spans widen the window).
        deepseek4::ImageSentinelIds imageIds{};
    };
    /**
     * @brief Vision: overwrite image-sentinel embedding rows from the request bank.
     * @details The bank and its position map are request-admission inputs; the
     * embedding is updated in place before the hyper-connection expansion.
     */
    struct ImageMerge
    {
        int width, bankRows, mapRows;
        deepseek4::ImageSentinelIds imageIds;
        Qwen4ExpIntInput tokens, positions;
        Qwen4ExpFloatBinding bank;                  ///< [bankRows, width].
        Qwen4ExpIntBinding map;                     ///< [mapRows] bank row + 1 (0 = token embedding).
        Qwen4ExpFloatBinding embedding;             ///< [M, width], in place.
    };
    /**
     * @brief Publish the forward's real rows to the canonical cache sequence frontier (GPU).
     * @details DeepSeek-V4 attention state lives in model-owned banks, so the
     * cache holds no attention layers and no append advances its frontier.
     * The terminal layer advances it exactly once per forward on the exact
     * stream (IKVCache::advanceRecurrentSequenceState) by the stage's own row
     * extent: the device live-row count when bound (padded prefill, including
     * zero live rows), else every admitted row. This is the same row authority
     * every DeepSeek-V4 bank kernel consumes, so the frontier and the banks
     * cannot disagree.
     */
    struct AdvanceSequenceFrontier
    {
        IKVCache *cache;                            ///< Canonical main cache (model_state layers only).
        int layer;                                  ///< Global terminal layer of this graph.
        Qwen4ExpIntInput positions;                 ///< Ordering anchor on the row positions.
    };
    /** @brief Grouped low-rank output projection (wo_a). */
    struct GroupedProjection
    {
        int groups, rank, groupWidth;
        DeepSeekV4NativeWeightBinding weight;       ///< [G*R, groupWidth] Q8_0/BF16/FP32.
        Qwen4ExpFloatBinding input, output;
    };
    /** @brief sqrt(softplus) score routing or token-id hash routing. */
    struct Route
    {
        int experts, topK;
        float routeScale;
        Qwen4ExpFloatBinding logits;
        std::optional<Qwen4ExpImmutableWeightBinding> bias;      ///< Score routing.
        std::optional<DeepSeekV4NativeWeightBinding> hashTable;  ///< INT32 [vocab, topK].
        std::optional<Qwen4ExpIntInput> tokens;                  ///< Hash routing token ids.
        Qwen4ExpIntBinding ids;
        Qwen4ExpFloatBinding weights;
    };
    /** @brief Closed operation set; absent operands cannot be represented. */
    using Operation = std::variant<HCRead, HCWrite, HCExpand, HeadNormRope, CompressorStore, CompressorPool,
                                   RawStore, IndexerTopK, SparseAttention, GroupedProjection, Route,
                                   AdvanceSequenceFrontier, ImageMerge>;
    /** @brief Immutable graph configuration. */
    struct Params
    {
        STAGE_PARAMS_COMMON_FIELDS;
        Qwen4ExpStageRows rows; ///< Admitted geometry and optional count subview.
        Operation operation;    ///< Fully typed semantic operation.
    };
    static_assert(StageParamsRequired<Params>);

    /**
     * @brief Validate and seal one operation and construct its backend bundle.
     * @throws std::invalid_argument On type, geometry, capacity, alignment or alias violations,
     *         and with "DeepSeek V4: ROCm unimplemented" for ROCm participants.
     */
    explicit DeepSeekV4PrimitiveStage(Params params);
    /** @brief Submit the selected arithmetic on already-prepared storage. */
    bool execute(IDeviceContext *context) override;
    /** @brief Semantic operation tag. */
    ComputeStageType type() const override;
    /**
     * @brief Position-indexed state banks are legitimately all-zero before their
     *        first completed block (e.g. a short prompt on a ratio-128 layer), and
     *        the indexer selection is an integer row; only arithmetic outputs are
     *        subject to the uninitialized-zero diagnostic.
     */
    bool allowsZeroOutput() const override;
    /** @brief Arithmetic at the admitted row bound. */
    std::size_t estimatedFlops() const override;
    /** @brief Bound operand bytes, excluding spare capacity. */
    std::size_t estimatedMemoryBytes() const override;
    /** @brief Support for the constructed backend only. */
    bool supportsBackend(ComputeBackendType backend) const override;
    /** @brief Bound views for diagnostics, without reading device counts. */
    StageDumpInfo buildDumpInfoImpl() const override;
    /** @brief No owned scratch. */
    StageBufferRequirements getBufferRequirements() const override { return {}; }
    /** @brief Complete read/write arena contracts and immutable weights. */
    StageBufferContract bufferContract() const override;
    /** @brief Sealed configuration. */
    const Params &getParams() const noexcept { return params_; }

private:
    enum class Access { Input, Output };
    enum class AliasPolicy { Disjoint, ExactOutputAliasAllowed };
    struct Operand
    {
        TensorBase *tensor;
        BufferId buffer;
        TensorType dtype;
        std::size_t byteOffset, rows, columns;
        Access access;
        const char *name;
        AliasPolicy alias;
        std::size_t bytes() const noexcept { return rows * columns * 4; }
    };
    struct WeightOperand
    {
        const TensorBase *tensor;
        const char *name;
    };
    template <TensorType Type> void add(Qwen4ExpArenaBinding<Type> binding, std::size_t rows, std::size_t columns,
                                        Access access, const char *name, AliasPolicy alias = AliasPolicy::Disjoint);
    void addIntInput(const Qwen4ExpIntInput &binding, std::size_t count, const char *name);
    const std::int32_t *intInput(const Qwen4ExpIntInput &binding) const;
    void addFP32Weight(Qwen4ExpImmutableWeightBinding binding, std::size_t elements, const char *name);
    void addNativeWeight(DeepSeekV4NativeWeightBinding binding, const char *name);
    const void *weightPointer(const TensorBase *tensor) const;
    void validateAliases() const;
    template <typename T, TensorType Type> T *pointer(Qwen4ExpArenaBinding<Type> binding, Access access) const;
    deepseek4::RowExtent rowExtent() const;

    Params params_;
    std::vector<Operand> operands_;
    std::vector<WeightOperand> weights_;
    std::vector<Qwen4ExpNativeIntInput> native_inputs_;
    std::unique_ptr<deepseek4::IDeepSeekV4Kernels> kernels_;
};
} // namespace llaminar2
