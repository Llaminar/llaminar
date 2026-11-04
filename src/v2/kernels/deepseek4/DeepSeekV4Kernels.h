/**
 * @file DeepSeekV4Kernels.h
 * @brief Allocation-free FP32 DeepSeek-V4 primitive contracts (CPU / CUDA).
 *
 * DeepSeek-V4 adds five model-specific operation families on top of the
 * ordinary prepared GEMM/RMSNorm bridges:
 *
 *  - Sinkhorn hyper-connections: an hc-way residual trunk read through
 *    sigmoid "pre" weights and written back through sigmoid "post" weights
 *    plus a Sinkhorn-normalized hc x hc combination of the previous copies.
 *  - Partial RoPE on the trailing rope dims with per-layer YaRN, including the
 *    inverse rotation DeepSeek applies to the attention output.
 *  - Gated-pooling KV compression (ratio 4 with overlapping windows, ratio
 *    128 without), writing completed blocks to an append-only pool.
 *  - Lightning-indexer scoring over compressed indexer keys with learned
 *    per-token head weights and bounded top-k selection.
 *  - Sparse MQA attention (K == V) over a raw sliding window plus selected or
 *    all visible compressed entries, with per-head attention sinks.
 *  - A grouped low-rank output projection (wo_a) and sqrt(softplus) MoE routing
 *    with selection bias, route scaling and token-id hash routing.
 *
 * Every row carries its absolute position, so prefill chunks, decode and
 * restored prefixes share one arithmetic definition: the value at position p
 * is a function of tokens [0, p] only. All position/count pointers live in the
 * execution device's address space (host for CPU, device for GPU) and are
 * never read by GPU launch code. Matrices are contiguous and row-major.
 *
 * Mathematical source: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
 * inference/model.py @ 6821d6ad, including its QAT activation rounding
 * (FP8 window/compressed kv, Hadamard + FP4 indexer q/k). The persistent
 * banks store exactly those values in compact records (DeepSeekV4KVFormat.h):
 * the raw window ring and compressed pools are 584-byte fp8_ds_mla entries,
 * the indexer pool holds 68-byte MXFP4 keys, and only the O(ratio)
 * compressor state stays FP32. The Python oracle is
 * python/reference/deepseek4.py (QAT on by default).
 */
#pragma once

#include "DeepSeekV4KVFormat.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DSV4_HD __host__ __device__ inline
#else
#define DSV4_HD inline
#endif

namespace llaminar2::deepseek4
{
/** @brief Admitted row bound and optional execution-owner live row count. */
struct RowExtent
{
    int rows = 0;                  ///< Admitted positive row capacity; part of capture identity.
    const int *liveRows = nullptr; ///< Optional authoritative count in [0, rows]; null = all rows.
};

/**
 * @brief Partial-RoPE frequency policy for one layer.
 * @details Rotates adjacent (even, odd) pairs of the trailing @c ropeDim
 * channels. With YaRN enabled, pair i uses
 * f_i = b^(-2i/d) * (s_i + (1 - s_i) / factor), s_i = 1 - clamp((i - low) / (high - low), 0, 1).
 * DeepSeek-V4 applies no YaRN magnitude correction.
 */
struct RopePolicy
{
    int ropeDim = 64;        ///< Rotated trailing channels (even); 0 disables rotation.
    float base = 10000.0f;   ///< Frequency base.
    bool yarn = false;       ///< Enable the YaRN frequency ramp.
    float factor = 1.0f;     ///< YaRN interpolation factor.
    float rampLow = 0.0f;    ///< First ramp pair index (floor of the beta_fast correction dim).
    float rampHigh = 1.0f;   ///< Last ramp pair index (ceil of the beta_slow correction dim), > rampLow.

    /**
     * @brief Build the DeepSeek YaRN policy from checkpoint metadata.
     * @param ropeDim Rotated channels.
     * @param base Frequency base.
     * @param originalContext Pre-extension context; 0 disables YaRN.
     * @param factor Interpolation factor.
     * @param betaFast Fast correction rotations (32).
     * @param betaSlow Slow correction rotations (1).
     * @return Immutable policy with host-resolved integer ramp bounds.
     */
    static RopePolicy make(int ropeDim, float base, int originalContext, float factor,
                           float betaFast, float betaSlow);
};

/** @brief Read side of one hyper-connection site (attention, FFN or terminal head). */
struct HCReadArgs
{
    RowExtent extent;               ///< Live token rows.
    int hc = 4;                     ///< Residual copies.
    int hidden = 0;                 ///< Per-copy width H.
    bool head = false;              ///< Terminal head: mixes has hc columns and only "pre" is produced.
    float normEps = 1e-6f;          ///< RMS epsilon of the flattened trunk (model norm epsilon).
    float hcEps = 1e-6f;            ///< Hyper-connection epsilon.
    int sinkhornIters = 20;         ///< Sinkhorn iterations (>= 1).
    const float *trunk = nullptr;   ///< [rows, hc, H] residual copies.
    const float *mixes = nullptr;   ///< [rows, mixWidth] = flat(trunk) . fn^T (unnormalized).
    const float *scale = nullptr;   ///< [3] (block) or [1] (head) learned scales.
    const float *base = nullptr;    ///< [mixWidth] learned biases.
    const float *gain = nullptr;    ///< Optional [H] RMSNorm gain applied to the mixed read.
    float outputEps = 1e-6f;        ///< RMS epsilon of the optional output norm.
    float *mixed = nullptr;         ///< Optional [rows, H] un-normalized mixed read.
    float *output = nullptr;        ///< [rows, H] mixed read, RMS-normalized with @c gain when given.
    float *post = nullptr;          ///< [rows, hc] block write weights (ignored for head).
    float *comb = nullptr;          ///< [rows, hc(src), hc(dst)] Sinkhorn combination (ignored for head).
};

/** @brief Write side: trunk[k] = post[k] * block + sum_j comb[j, k] * trunk[j]. */
struct HCWriteArgs
{
    RowExtent extent;               ///< Live token rows.
    int hc = 4;                     ///< Residual copies.
    int hidden = 0;                 ///< Per-copy width H.
    const float *block = nullptr;   ///< [rows, H] attention or FFN output.
    const float *post = nullptr;    ///< [rows, hc] from the paired read.
    const float *comb = nullptr;    ///< [rows, hc, hc] from the paired read.
    const float *trunk = nullptr;   ///< [rows, hc, H] trunk before the block.
    float *output = nullptr;        ///< [rows, hc, H]; may alias @c trunk exactly.
};

/** @brief Broadcast an H-wide embedding into the hc-copy trunk. */
struct HCExpandArgs
{
    RowExtent extent;               ///< Live token rows.
    int hc = 4;                     ///< Residual copies.
    int hidden = 0;                 ///< Width H.
    const float *input = nullptr;   ///< [rows, H].
    float *output = nullptr;        ///< [rows, hc, H], distinct from input.
};

/**
 * @brief Per-head RMS normalization, partial RoPE and optional QAT rounding, in place.
 * @details y = rms(x) * gain (gain optional), then rotate trailing channels at
 * the row's absolute position (inverse rotation when @c inverse is set), then
 * apply @c quant to every head vector (see DeepSeekV4KVFormat.h). With
 * @c normalize false only the rotation (and quantization) is applied.
 */
struct HeadNormRopeArgs
{
    RowExtent extent;                      ///< Live token rows.
    int heads = 1;                         ///< Heads per row.
    int headDim = 0;                       ///< Channels per head (>= ropeDim, <= kMaxHeadDim).
    bool normalize = true;                 ///< Apply RMS normalization before rotation.
    float eps = 1e-6f;                     ///< RMS epsilon.
    const float *gain = nullptr;           ///< Optional [headDim] gain shared by all heads.
    RopePolicy rope;                       ///< Layer rotation policy.
    bool inverse = false;                  ///< Rotate by -angle (attention output de-rotation).
    ActivationQuant quant = ActivationQuant::None; ///< Official QAT post-operation.
    const std::int32_t *positions = nullptr; ///< [rows] absolute positions.
    float *data = nullptr;                 ///< [rows, heads, headDim], updated in place.
};

/** @brief Largest head width any DeepSeek-V4 primitive accepts (GPU shared-memory staging). */
inline constexpr int kMaxHeadDim = 512;

/** @brief Which compressed stream a compressor produces, and therefore its packed record. */
enum class CompressedStream : std::uint8_t
{
    Attention, ///< Main KV: FP8 nope + BF16 rope (PackedKVEntryFormat).
    Indexer,   ///< Indexer keys: Hadamard + MXFP4 (PackedIndexerKeyFormat).
};

/**
 * @brief Compressor geometry for one layer and one compressed stream.
 * @details Ratio 4 uses overlapping windows: projections have 2*headDim
 * channels, block b pools the first half of block b-1 and the second half of
 * block b (8 entries per channel). Ratio 128 pools headDim channels of its own
 * 128 tokens. Pooling is a per-channel softmax over entries of
 * (score + ape[position % ratio]) weighting the projected kv.
 *
 * Projections of the current chunk are read directly from the chunk's
 * projection scratch; earlier positions come from the carried state of the
 * last stateRows() positions (the official kv_state/score_state), slot
 * position % stateRows(). Rows of one call carry contiguous positions.
 */
struct CompressorGeometry
{
    int ratio = 4;          ///< Tokens per block (4 or 128).
    int headDim = 0;        ///< Compressed entry width.
    int poolRows = 0;       ///< Compressed pool capacity (max context / ratio).
    CompressedStream stream = CompressedStream::Attention; ///< Output record format.
    int ropeDim = 0;        ///< Rope channels of the record (Attention stream).
    DSV4_HD bool overlap() const noexcept { return ratio == 4; }
    DSV4_HD int projectionWidth() const noexcept { return overlap() ? 2 * headDim : headDim; }
    DSV4_HD int stateRows() const noexcept { return 2 * ratio; }
    /** @brief Packed record bytes of one pool row. */
    DSV4_HD int entryBytes() const noexcept
    {
        return stream == CompressedStream::Attention ? PackedKVEntryFormat{headDim, ropeDim}.entryBytes()
                                                     : PackedIndexerKeyFormat{headDim}.entryBytes();
    }
};

/**
 * @brief Carry the last stateRows() live positions' projections into the state ring.
 * @details Runs after the paired pool has read the previous state. Only rows
 * with position > lastLivePosition - stateRows() are written, so every slot
 * has at most one writer.
 */
struct CompressorStoreArgs
{
    RowExtent extent;                        ///< Live token rows.
    CompressorGeometry geometry;             ///< Layer stream geometry.
    const std::int32_t *positions = nullptr; ///< [rows] contiguous absolute positions.
    const float *kv = nullptr;               ///< [rows, projectionWidth] kv projection.
    const float *score = nullptr;            ///< [rows, projectionWidth] gate projection.
    const float *ape = nullptr;              ///< [ratio, projectionWidth] learned position bias (FP32).
    float *stateKV = nullptr;                ///< [stateRows, projectionWidth].
    float *stateScore = nullptr;             ///< [stateRows, projectionWidth] score + ape.
};

/**
 * @brief Pool every block completed by a live row, QAT-round it and pack it into the pool.
 * @details Entry e of the block at position p reads the chunk scratch when
 * p >= positions[0], else the carried state. The pooled vector is
 * RMS-normalized with @c gain, rotated at the block's first position, then
 * quantized: Attention -> FP8 nope + BF16 rope; Indexer -> Hadamard + MXFP4.
 */
struct CompressorPoolArgs
{
    RowExtent extent;                        ///< Live token rows.
    CompressorGeometry geometry;             ///< Layer stream geometry.
    const std::int32_t *positions = nullptr; ///< [rows] contiguous absolute positions.
    const float *kv = nullptr;               ///< [rows, projectionWidth] chunk kv projection.
    const float *score = nullptr;            ///< [rows, projectionWidth] chunk gate projection (no ape).
    const float *ape = nullptr;              ///< [ratio, projectionWidth].
    const float *stateKV = nullptr;          ///< [stateRows, projectionWidth] carried earlier positions.
    const float *stateScore = nullptr;       ///< [stateRows, projectionWidth] carried score + ape.
    const float *gain = nullptr;             ///< [headDim] compressor RMSNorm gain.
    float eps = 1e-6f;                       ///< RMS epsilon.
    RopePolicy rope;                         ///< Rotation at block position b * ratio.
    std::uint8_t *pool = nullptr;            ///< [poolRows, entryBytes] packed completed entries.
};

/**
 * @brief Pack the last @c window live kv rows into the persistent raw window ring.
 * @details Runs after every reader of the previous window (sparse attention).
 * Slot = position % window; only rows with position > lastLivePosition - window
 * are written.
 */
struct RawStoreArgs
{
    RowExtent extent;                        ///< Live token rows.
    PackedKVEntryFormat format;              ///< Record layout.
    int window = 0;                          ///< Ring rows (the attention window).
    const std::int32_t *positions = nullptr; ///< [rows] contiguous absolute positions.
    const float *kv = nullptr;               ///< [rows, headDim] QAT-rounded kv.
    std::uint8_t *ring = nullptr;            ///< [window, entryBytes].
};

/**
 * @brief Lightning-indexer scoring and bounded top-k over visible blocks.
 * @details score(p, b) = sum_h relu(q[p, h] . k[b]) * (w[p, h] * weightScale),
 * accumulated over heads with fused multiply-add in ascending head order; each
 * head dot is qat::indexerDot (exact per 32-block). The min(topK, visible)
 * best blocks (score descending, ascending block id on exact ties) are
 * written as a set in ascending block order; remaining slots are -1.
 * GPU backends score at most @c scoreRows rows per pass into @c scores, so
 * scratch is bounded independently of the chunk size.
 */
struct IndexerTopKArgs
{
    RowExtent extent;                        ///< Live token rows.
    int heads = 0;                           ///< Indexer heads.
    PackedIndexerKeyFormat format;           ///< Key record layout (headDim).
    int ratio = 4;                           ///< Compression ratio of the indexed pool.
    int topK = 0;                            ///< Selection bound.
    int poolRows = 0;                        ///< Pool capacity.
    int scoreRows = 0;                       ///< Rows per scoring pass (rows of @c scores).
    const std::int32_t *positions = nullptr; ///< [rows] absolute positions.
    const float *query = nullptr;            ///< [rows, heads, headDim] rotated FP4-QAT queries.
    const float *weights = nullptr;          ///< [rows, heads] head weights (weights_proj output).
    float weightScale = 1.0f;                ///< Multiplier on @c weights (headDim^-0.5 * heads^-0.5).
    const std::uint8_t *pool = nullptr;      ///< [poolRows, entryBytes] packed indexer keys.
    float *scores = nullptr;                 ///< [scoreRows, poolRows] scratch (GPU; unused on CPU).
    std::int32_t *selected = nullptr;        ///< [rows, topK] selected block ids or -1.
};

/** @brief Which compressed entries a query attends to. */
enum class CompressedAccess : std::uint8_t
{
    None,     ///< Raw window only (ratio 0).
    All,      ///< Every visible compressed entry (ratio 128).
    Selected, ///< Indexer-selected entries (ratio 4).
};

/**
 * @brief Sparse sink attention for one layer.
 * @details For query row p: keys are the raw kv of positions
 * [max(0, p - window + 1), p] (the chunk's QAT kv for positions >= positions[0],
 * else the packed window ring) plus compressed entries per @c access. K == V.
 * out = sum softmax(q . k * scale) k with the softmax denominator augmented by
 * exp(sink[h]). The output is then de-rotated (inverse RoPE at p).
 */
struct SparseAttentionArgs
{
    RowExtent extent;                        ///< Live token rows.
    int heads = 0;                           ///< Query heads.
    PackedKVEntryFormat format;              ///< Ring/pool record layout; format.headDim is the head width.
    int window = 0;                          ///< Raw sliding window (= ring rows).
    int ratio = 0;                           ///< Compression ratio (0 = none).
    int topK = 0;                            ///< Row stride of @c selected.
    CompressedAccess access = CompressedAccess::None;
    float scale = 1.0f;                      ///< Softmax scale (headDim^-0.5).
    RopePolicy rope;                         ///< Output de-rotation policy.
    const std::int32_t *positions = nullptr; ///< [rows] contiguous absolute positions.
    const float *query = nullptr;            ///< [rows, heads, headDim] normalized rotated queries.
    const float *chunkKV = nullptr;          ///< [rows, headDim] QAT kv of this chunk.
    const std::uint8_t *rawRing = nullptr;   ///< [window, entryBytes] packed earlier positions.
    const std::uint8_t *pool = nullptr;      ///< [poolRows, entryBytes] packed compressed kv (None: null).
    const std::int32_t *selected = nullptr;  ///< [rows, topK] for Selected.
    const float *sinks = nullptr;            ///< [heads] attention sink logits.
    float *output = nullptr;                 ///< [rows, heads, headDim] de-rotated output.
};

/** @brief Storage format of a grouped projection weight. */
enum class WeightFormat : std::uint8_t
{
    FP32,
    BF16,
    Q8_0,
};

/**
 * @brief Grouped low-rank projection: out[r, g*R + j] = in[r, g, :] . W[g*R + j, :].
 * @details W is [groups * rank, groupWidth] in GGUF row-major order (one GGUF
 * row per output); Q8_0 rows are groupWidth / 32 blocks of {fp16 d, int8 q[32]}.
 */
struct GroupedProjectionArgs
{
    RowExtent extent;            ///< Live token rows.
    int groups = 0;              ///< Output groups G.
    int rank = 0;                ///< Outputs per group R.
    int groupWidth = 0;          ///< Input width per group.
    WeightFormat format = WeightFormat::Q8_0;
    const void *weight = nullptr;  ///< Raw native weight payload.
    const float *input = nullptr;  ///< [rows, G, groupWidth].
    float *output = nullptr;       ///< [rows, G * R].
};

/**
 * @brief sqrt(softplus) MoE routing.
 * @details s = sqrt(softplus(logits)). Score routing selects the top-k of
 * s + bias (descending, ascending expert id on ties); hash routing reads
 * table[token]. Weights are the unbiased s of the selected experts,
 * normalized to sum 1 and multiplied by @c routeScale.
 */
struct SqrtSoftplusRouteArgs
{
    RowExtent extent;                        ///< Live token rows.
    int experts = 0;                         ///< Routed experts.
    int topK = 0;                            ///< Selected experts per row.
    float routeScale = 1.0f;                 ///< Scale applied after normalization.
    const float *logits = nullptr;           ///< [rows, experts] router GEMM output.
    const float *bias = nullptr;             ///< [experts] selection bias (score routing).
    const std::int32_t *hashTable = nullptr; ///< [vocab, topK] (hash routing) or null.
    int vocab = 0;                           ///< Rows of @c hashTable.
    const std::int32_t *tokens = nullptr;    ///< [rows] token ids (hash routing).
    std::int32_t *ids = nullptr;             ///< [rows, topK] selected expert ids.
    float *weights = nullptr;                ///< [rows, topK] routing weights.
};

/**
 * @brief Backend-specific stateless DeepSeek-V4 primitive bundle.
 * @details Every GPU call requires a non-null, non-default exact stream; CPU
 * ignores it. The graph owns pointers, banks and ordering events.
 */
class IDeepSeekV4Kernels
{
public:
    /** @brief Destroy the stateless implementation. */
    virtual ~IDeepSeekV4Kernels() = default;
    /** @brief Hyper-connection read (Sinkhorn split + mixed read + optional norm). */
    virtual void hcRead(const HCReadArgs &args, void *stream) const = 0;
    /** @brief Hyper-connection write back into the trunk. */
    virtual void hcWrite(const HCWriteArgs &args, void *stream) const = 0;
    /** @brief Expand embeddings into the hc-copy trunk. */
    virtual void hcExpand(const HCExpandArgs &args, void *stream) const = 0;
    /** @brief Per-head RMS normalization and partial RoPE, in place. */
    virtual void headNormRope(const HeadNormRopeArgs &args, void *stream) const = 0;
    /** @brief Store compressor projections in the ring. */
    virtual void compressorStore(const CompressorStoreArgs &args, void *stream) const = 0;
    /** @brief Pool completed blocks into the compressed pool. */
    virtual void compressorPool(const CompressorPoolArgs &args, void *stream) const = 0;
    /** @brief Store raw window rows in the ring. */
    virtual void rawStore(const RawStoreArgs &args, void *stream) const = 0;
    /** @brief Indexer scoring and top-k selection. */
    virtual void indexerTopK(const IndexerTopKArgs &args, void *stream) const = 0;
    /** @brief Sparse sink attention with output de-rotation. */
    virtual void sparseAttention(const SparseAttentionArgs &args, void *stream) const = 0;
    /** @brief Grouped low-rank output projection. */
    virtual void groupedProjection(const GroupedProjectionArgs &args, void *stream) const = 0;
    /** @brief sqrt(softplus) score or hash MoE routing. */
    virtual void route(const SqrtSoftplusRouteArgs &args, void *stream) const = 0;
    /**
     * @brief Zero a persistent state bank once at setup, before any capture.
     * @param destination Execution-device bytes (4-byte aligned).
     * @param bytes Multiple of 4.
     */
    virtual void zeroBank(void *destination, std::size_t bytes, void *stream) const = 0;
};

/** @brief Construct the CPU bundle outside captured execution. */
std::unique_ptr<IDeepSeekV4Kernels> createCPUDeepSeekV4Kernels();
/** @brief Construct the CUDA bundle outside captured execution. */
std::unique_ptr<IDeepSeekV4Kernels> createCUDADeepSeekV4Kernels();

/** @brief Diagnostic used whenever a ROCm participant is requested. */
inline constexpr const char *kROCmUnimplemented = "DeepSeek V4: ROCm unimplemented";
} // namespace llaminar2::deepseek4
