/**
 * @file QwenSparseAttention.h
 * @brief Caller-owned Qwen4Exp sparse-indexer and selected-KV execution contracts.
 *
 * The transaction base remains immutable until acceptance. Newly completed
 * pools may be written speculatively: a row can only address its own complete
 * prefix, and a later transaction overwrites any rejected pool before reuse.
 * Partial raw keys are committed separately using the accepted row count.
 * Storage, admission, streams, and publication events belong to the graph.
 */
#pragma once
#include "Qwen4ExpStateIO.h"
#include "QwenSparseAttentionArithmetic.h"
#include "QwenSparseTransactionPrior.h"
#include "../common/AcceptedRequestPrefixes.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include "kernels/attention/AttentionDeviceParams.h"
#include "kernels/attention/AttentionExecutionPolicy.h"

namespace llaminar2::qwen4exp
{
inline constexpr int QwenSparseIndexHeads = 4; ///< Indexer query heads.
inline constexpr int QwenSparseIndexDim = 128; ///< Indexer coordinates per head.
inline constexpr int QwenSparsePool = 4; ///< Visible tokens per complete pool.
inline constexpr int QwenSparseBlocks = 512; ///< Maximum selected complete pools.
inline constexpr int QwenSparseFirstRankingLogicalRow = QwenSparsePool*(QwenSparseBlocks+1)-1; ///< First zero-based row that needs pool ranking.
inline constexpr int QwenSparseSelected = 2051; ///< 2048 complete tokens plus tail.
inline constexpr int QwenSparseQueryHeads = 24; ///< Core attention query heads.
inline constexpr int QwenSparseKVHeads = 2; ///< Core attention key/value heads.
inline constexpr int QwenSparseHeadDim = 256; ///< Core attention coordinates.
inline constexpr int QwenSparsePartitions = 8; ///< Fixed selected-index arithmetic partitions.
inline constexpr int QwenSparsePartitionTokens = 257; ///< Covers 2051 indices in eight partitions.

/** @brief Explicit query authorship; keys and accepted history always remain live. */
enum class QwenSparseQueryDemand : std::uint8_t { AllLive, RankingOnly };

/** @brief Explicit representation; unsupported codecs cannot enter a native reader. */
enum class QwenSparseScalar : std::uint8_t { FP32, FP16, BF16 };

/** @brief Explicit rotary arithmetic authority; indexer and core phases have different invariant producers. */
enum class QwenSparseRotarySource : std::uint8_t { TablePlanes, AnalyticTheta, CanonicalEngineTheta };
/** @brief Setup-owned HF FP32 reciprocal-power frequencies for the exact 64-wide QFN rotary prefix. */
using QwenSparsePowerFrequencies=std::array<float,32>;
static_assert(sizeof(QwenSparsePowerFrequencies)==32*sizeof(float) && alignof(QwenSparsePowerFrequencies)==alignof(float));
/** @brief Construct immutable indexer frequency scalars once before capture, without allocation. */
QwenSparsePowerFrequencies makeQwenSparsePowerFrequencies(float theta,int rotaryDimension);

/**
 * @brief Immutable backend-neutral pool selection policy; Installed preserves current dispatch.
 * @details Every bounded geometry has an explicit native implementation. Declarative
 * graph callers must select it before construction; the default remains Installed.
 */
enum class QwenSparsePoolSelection : std::uint8_t
{
    Installed,
    Bounded256_512,
    Bounded512_512,
    Bounded1024_1024
};

/**
 * @brief Reject an unknown immutable selector policy before graph construction or enqueue.
 * @param policy Explicit backend-neutral selector implementation.
 * @throws std::invalid_argument If the value is outside the closed supported enum.
 * @details This check reads no execution-owned state and performs no device operation.
 */
void validateQwenSparsePoolSelection(QwenSparsePoolSelection policy);
/**
 * @brief Admit an explicit selector against immutable invocation row capacity.
 * @param policy Closed native selection policy, independent of live replay bytes.
 * @param rowCapacity Positive admitted rows, not the device-owned live count.
 * @throws std::invalid_argument For unknown policy or nonpositive admitted capacity.
 */
void validateQwenSparsePoolSelection(QwenSparsePoolSelection policy, int rowCapacity);

/**
 * @brief Optional row-local diagnostic work, published only after selection.
 * @details These are source-level element/pass counts, NOT physical HBM traffic.
 * Timing observations are collected separately outside canonical timed replay.
 */
struct QwenSparseSelectionObservation
{
    std::uint64_t score_elements; ///< Main scan scores; excludes CPU validation reads.
    std::uint64_t compactions; ///< Actual bounded-bank compaction calls.
    std::uint64_t compacted_keys; ///< Sum of each pre-compaction live bank size.
    std::uint64_t radix_passes; ///< Actual bounded histogram digits after common-prefix skips.
};
static_assert(sizeof(QwenSparseSelectionObservation) == 32);

/**
 * @brief Persistent indexer state and exclusive transaction scratch.
 *
 * Indexer projections and stored pools are FP32 containers whose values are
 * rounded to representation at the reference boundaries. Norm weights already
 * include the zero-centered +1 conversion performed by the GGUF importer.
 * RoPE planes contain one cosine/sine per split-half pair (rotary_dim/2).
 */
struct QwenSparseIndexer
{
    const QwenSparseReplay *replay = nullptr; ///< Ordered immutable transaction base.
    const float *query = nullptr; ///< Raw Q projection [row_capacity,4,128].
    const float *raw_keys = nullptr; ///< Raw K projection [row_capacity,128].
    const float *partial_keys = nullptr; ///< Prior raw tail [4,128], first prefix%4 rows live.
    float *pooled_keys = nullptr; ///< Persistent prepared complete pools [block_capacity,128].
    const float *query_norm = nullptr; ///< Converted RMSNorm weights [128].
    const float *key_norm = nullptr; ///< Converted RMSNorm weights [128].
    const float *cosine = nullptr; ///< Absolute-position plane [position_capacity,rotary_dim/2].
    const float *sine = nullptr; ///< Same shape as cosine.
    float *prepared_query = nullptr; ///< Exclusive prepared Q scratch [row_capacity,4,128].
    float *scores = nullptr; ///< Exclusive complete-pool scores [row_capacity,block_capacity].
    std::int32_t *selected = nullptr; ///< Exclusive sorted logical indices [row_capacity,2051].
    std::int32_t *selected_count = nullptr; ///< Live count for every admitted row.
    int row_capacity = 0; ///< Capture geometry, independent of live_rows.
    int token_capacity = 0;
    QwenSparseWorkCounterRow work_counters; ///< Optional canonical terminal observations for actual accepted writes. ///< Admitted logical token capacity.
    int block_capacity = 0; ///< At least ceil(token_capacity/4).
    int position_capacity = 0; ///< Admitted absolute-position RoPE plane extent.
    int rotary_dim = 64; ///< Even partial split-half RoPE width, at most 128.
    float norm_epsilon = 1.0e-6f; ///< RMSNorm epsilon.
    QwenSparseScalar representation = QwenSparseScalar::FP32; ///< Indexer cast boundaries.
    QwenSparseRotarySource rotary_source = QwenSparseRotarySource::TablePlanes; ///< Explicit factor source.
    float rope_theta = 10000.0f; ///< Analytic base with exponent denominator rotary_dim.
    QwenSparsePowerFrequencies inverse_frequencies{}; ///< Setup-derived FP32 power invariants embedded by value in capture arguments.
    const QwenSparseTransactionPrior* transaction_prior=nullptr; ///< Optional typed immutable original replay/tail codec.
    QwenSparsePoolSelection pool_selection = QwenSparsePoolSelection::Installed; ///< Explicit selector candidate; scores/arithmetic are unchanged.
    QwenSparseSelectionObservation *selection_observations = nullptr; ///< Optional original [row_capacity] diagnostics; no host mirror.
    QwenSparseQueryDemand query_demand = QwenSparseQueryDemand::AllLive; ///< Explicit decode query work; ordinary prefill retains complete query authorship.
};

/**
 * @brief Acceptance-only publication of the incomplete raw indexer block.
 *
 * The original tail is read before any output write, allowing in-place commit.
 * No prefix count is mutated: the generation controller publishes its canonical
 * frontier after committing all recurrent/cache components on the same DAG.
 */
struct QwenSparsePartialCommit
{
    const QwenSparseReplay *replay = nullptr; ///< Original transaction base.
    const float *prior_partial = nullptr; ///< Original [4,128] raw tail.
    const float *raw_keys = nullptr; ///< Current raw projections [row_capacity,128].
    AcceptedRequestPrefixes acceptance; ///< One canonical request, with counts or flat accepted-row encoding.
    float *partial_keys = nullptr; ///< Persistent output [4,128].
    const QwenSparseTransactionPrior* transaction_prior=nullptr; ///< Typed immutable evidence for repeated verifier acceptance.
    int token_capacity=0; ///< Mandatory original cache geometry when transaction_prior is present.
    QwenSparseWorkCounterRow work_counters; ///< Optional original request terminal counters for actual accepted writes.

};

/** @brief Physical ordering of non-overlapping native token/head slices. */
enum class QwenSparseNativeLayout : std::uint8_t { TokenMajor, HeadMajor };

/** @brief Immutable native bank layout; all codec tables are setup-owned persistent values. */
struct QwenSparseNativeBank
{
    const void *data = nullptr; ///< Native [physical_token,kv_head,coordinate/block] bank.
    std::size_t token_stride_bytes = 0; ///< Native byte distance between token rows.
    std::size_t head_stride_bytes = 0; ///< Native byte distance between KV heads.
    TensorType codec = TensorType::FP32; ///< FP32/FP16/BF16/Q8_1/Q16_1/AQ8/TQ4/TQ8 only.
    int q16_block_size = 32; ///< Native Q16 block coordinates: 32, 64, or 128.
    const float *centroids = nullptr; ///< Canonical TQ4[16] or TQ8[256] codebook.
    const float *rotations[2]{}; ///< Canonical TQ encoding rotation [256,256] per local KV head, row-major.
    const float *anchor = nullptr; ///< Mandatory AQ8 request-local anchor [local_heads,256].
    QwenSparseNativeLayout layout = QwenSparseNativeLayout::TokenMajor; ///< Explicit native stride interpretation.
};

/** @brief Explicit canonical logical-row mapping source. */
enum class QwenSparseKVAddressing : std::uint8_t { Contiguous, FixedLogicalView, RowParameters };

/** @brief Separate native K/V codecs with canonical ring and key-transform policies. */
struct QwenSparseKV
{
    QwenSparseNativeBank keys; ///< Native key bank, consumed in its declared basis.
    QwenSparseNativeBank values; ///< Native value bank, independent of the key codec.
    QwenSparseKVAddressing addressing = QwenSparseKVAddressing::Contiguous; ///< Sole mapping selector.
    attention::AttentionKVLogicalView logical_view{}; ///< Fixed CPU/native mapping when explicitly selected.
    const attention::AttentionDeviceParams *row_parameters = nullptr; ///< Execution-owned per-row dynamic ring mapping.
    attention::AttentionKeyCachePolicy key_policy{}; ///< Explicit pre/post-RoPE cache contract.
    const float *key_cosine = nullptr; ///< Pre-RoPE unrotated keys: [absolute_position,rotary_dim/2].
    const float *key_sine = nullptr; ///< Same extent as key_cosine.
    int key_rotary_dim = 64; ///< Even pre-RoPE partial rotation width.
    int key_position_capacity = 0; ///< Absolute-position RoPE plane extent.
    QwenSparseRotarySource key_rotary_source = QwenSparseRotarySource::TablePlanes; ///< Explicit selected-key factor source.
    float key_negative_log_theta = 0.0f; ///< Setup-derived -std::log(theta), matching ordinary RoPE's native invariant producer.
};

/** @brief Per-row/head native reads published by the actual selected-index loops. */
struct QwenSparseAttentionObservation
{
    std::uint64_t key_elements = 0; ///< Native key coordinates actually decoded.
    std::uint64_t value_elements = 0; ///< Native value coordinates actually decoded.
    std::uint32_t selected_tokens = 0; ///< Live selected rows consumed by this head.
};

/** @brief Exclusive GPU partial for one fixed attention partition; reused across layers. */
struct QwenSparseAttentionPartial
{
    float maximum = 0.0f; ///< Partition logit maximum; negative infinity for an empty partition.
    float denominator = 0.0f; ///< Sum of exponentials relative to maximum.
    float weighted[QwenSparseHeadDim]; ///< Unnormalized weighted values in native attention basis.
    std::uint32_t key_elements = 0; ///< Actual key coordinate reads in this partition.
    std::uint32_t value_elements = 0; ///< Actual value coordinate reads in this partition.
};

/**
 * @brief True selected-index core attention; only selected native K/V rows are read.
 *
 * Q has already received the core Q norm and rotary transform. K has received
 * its own norm and rotary transform before cache publication. The optional
 * sigmoid gate is applied after attention, before the graph's output projection.
 */
struct QwenSparseAttention
{
    const QwenSparseReplay *replay = nullptr; ///< Current device/host row cardinality.
    const float *query = nullptr; ///< Prepared Q [row_capacity,24,256].
    QwenSparseKV kv; ///< Native cache; no full-bank conversion is permitted.
    const std::int32_t *selected = nullptr; ///< Sorted original token indices [row_capacity,2051].
    const std::int32_t *selected_count = nullptr; ///< Exact live attention extent per row.
    const float *gate = nullptr; ///< Optional raw sigmoid gate [row_capacity,24,256].
    float *output = nullptr; ///< Complete output [row_capacity,24,256].
    QwenSparseAttentionPartial *gpu_partials = nullptr; ///< Mandatory exclusive GPU scratch [row_capacity,24,8]; unused on CPU.
    QwenSparseAttentionObservation *observations = nullptr; ///< Optional execution-owned [row_capacity,24] traffic proof.
    int row_capacity = 0; ///< Admitted captured row geometry.
    int token_capacity = 0;
    QwenSparseWorkCounterRow work_counters; ///< Optional canonical terminal observations for actual accepted writes. ///< Admitted native cache token extent.
    int query_heads = 24; ///< Participant-local Q heads, retaining global GQA ownership.
    int kv_heads = 2; ///< Participant-local native KV heads.
    int query_head_start = 0; ///< First global Q head in this participant's contiguous shard.
    int kv_head_start = 0; ///< First global KV head stored in the native banks.
    QwenSparseAttentionArithmetic arithmetic = kDefaultQwenSparseAttentionArithmetic; ///< Sealed arithmetic, independent of row/context geometry.
};

/** @brief Immutable captured preparation programs, selected only by admitted row capacity. */
enum class QwenSparsePreparationProgram : std::uint8_t
{
    FusedGrouped,
    SeparatePrefill
};
/** @brief Checked original CTA counts and immutable launch program; no CTA is omitted. */
struct QwenSparsePreparationGrid
{
    int pool_blocks, query_blocks, total_blocks;
    QwenSparsePreparationProgram program;
};
/**
 * @brief Derive representable CTA extents and the retained row-regime program without live-state reads.
 * @param rows Immutable physical row capacity, independent of current live rows.
 * @return Original pool/query CTA counts, their exact sum, and the grouped/prefill launch program.
 * @throws std::invalid_argument For nonpositive rows or native signed-index overflow.
 */
[[nodiscard]] QwenSparsePreparationGrid qwenSparsePreparationGrid(int rows);

/** @brief Validate static pointer/geometry bindings without reading device state.
 * @param params Complete arena/cache indexer bindings.
 * @throws std::invalid_argument For an incomplete binding or unsupported immutable policy.
 */
void validateQwenSparseIndexer(const QwenSparseIndexer &params);
/** @brief Validate static native-cache/selected-index bindings.
 * @param params Native banks, sharding, mapping, and selected-output bindings.
 * @throws std::invalid_argument For incomplete codecs, ambiguous mapping, or invalid ownership.
 */
void validateQwenSparseAttention(const QwenSparseAttention &params);
/** @brief Validate static acceptance/partial-state bindings.
 * @param params Complete original-tail/candidate/acceptance bindings.
 * @throws std::invalid_argument For an incomplete binding or nonpositive row capacity.
 */
void validateQwenSparsePartialCommit(const QwenSparsePartialCommit &params);
/**
 * @brief Validate score-selection bindings without reading execution-owned state.
 * @param params Original replay/scores/selected/count geometry and explicit selector.
 * @throws std::invalid_argument For missing/overflowing/aliased or unsupported bindings.
 * @details Other indexer operands are unused; this is a score-only view of the
 * same typed contract, not an alternative score producer or request authority.
 */
void validateQwenSparseSelection(const QwenSparseIndexer &params);
/**
 * @brief Select from original finite nonnegative FP32 scores on CPU.
 * @param params Original replay and output bindings; inactive output suffix stays untouched.
 * @throws std::invalid_argument For malformed geometry/replay or consumed nonfinite scores.
 */
void cpuQwenSparseSelectPools(const QwenSparseIndexer &params);

/** @brief CPU indexer pooling, normalization, scoring, deterministic top-k, and expansion.
 * @param params Host-owned replay state and admitted persistent cache/scratch bindings.
 * @throws std::invalid_argument For invalid dynamic replay or unavailable native CPU ISA.
 */
void cpuQwenSparseIndexer(const QwenSparseIndexer &params);
/** @brief CPU native selected-index attention with fixed row-local arithmetic order.
 * @param params Host-owned selected indices, native post-RoPE banks, and output bindings.
 * @throws std::invalid_argument For invalid mapping, selected indices, codecs, or CPU ISA.
 */
void cpuQwenSparseAttention(const QwenSparseAttention &params);
/** @brief Commit the accepted incomplete block on CPU without changing its frontier.
 * @param params Original host replay, immutable candidates, and accepted-prefix decision.
 * @throws std::invalid_argument When acceptance exceeds the authenticated live extent.
 */
void cpuQwenSparseCommitPartial(const QwenSparsePartialCommit &params);

/** @brief Exact verifier raw-key retention; persistent pools/native KV are absent by construction. */
struct QwenSparseCandidateRetention
{
    const float *source;
    float *destination;
    const std::int32_t *liveRows;
    std::uint64_t *copiedBytes;
    int rowCapacity;
    QwenSparseWorkCounterRow work_counters; ///< Optional canonical terminal observations for live retained payload.
};
/** @brief Validate one intentional candidate transaction and expose its bounded native transfer. */
QwenStateTransfer qwenSparseCandidateTransfer(const QwenSparseCandidateRetention &params);

/** @brief Backend-independent sparse primitive authority used by compute stages. */
struct QwenSparsePrepareReplay;

class IQwenSparseAttentionKernels
{
public:
    /** @brief Retire a setup-owned backend bundle after its graphs are retired. */
    virtual ~IQwenSparseAttentionKernels() = default;
    /** @brief Publish exact pre-append replay and row parameters from execution-owned metadata.
     * @param params Canonical original cache cursor/count, live rows and absolute positions.
     * @param stream Exact non-default GPU stream; unused by CPU.
     */
    virtual void prepareReplay(const QwenSparsePrepareReplay &params,void *stream) const = 0;
    /** @brief Freeze exact original replay/tail evidence before native KV append. */
    virtual void snapshotPrior(const QwenSparsePriorSnapshot& params,void* stream) const = 0;
    /**
     * @brief Prepare causal indexer pools and publish sorted selected indices.
     * @param params Immutable bindings and execution-owned live state.
     * @param stream Exact non-default native GPU stream; unused by CPU.
     */
    virtual void indexer(const QwenSparseIndexer &params, void *stream) const = 0;
    /**
     * @brief Consume native selected K/V rows and publish the gated core result.
     * @param params Immutable native codecs, canonical mapping, and persistent scratch.
     * @param stream Exact non-default native GPU stream; unused by CPU.
     */
    virtual void attention(const QwenSparseAttention &params, void *stream) const = 0;
    /**
     * @brief Publish the accepted partial raw key block without changing its frontier.
     * @param params Execution-owned accepted-prefix decision and immutable candidate input.
     * @param stream Exact non-default native GPU stream; unused by CPU.
     */
    virtual void commitPartial(const QwenSparsePartialCommit &params, void *stream) const = 0;
    /** @brief Retain only live M-by128 raw indexer candidates on the exact execution stream.
     * @param params Shared raw projection, admitted transaction bank and execution-owned live count.
     * @param stream Exact non-default native GPU stream; unused by CPU.
     */
    virtual void retainCandidates(const QwenSparseCandidateRetention &params,void *stream) const = 0;
};
/** @brief Construct the allocation-free CPU sparse primitive bundle during setup. */
std::unique_ptr<IQwenSparseAttentionKernels> createCPUQwenSparseAttentionKernels();
/** @brief Construct the allocation-free CUDA sparse primitive bundle during setup. */
std::unique_ptr<IQwenSparseAttentionKernels> createCUDAQwenSparseAttentionKernels();
/** @brief Construct the allocation-free ROCm sparse primitive bundle during setup. */
std::unique_ptr<IQwenSparseAttentionKernels> createROCmQwenSparseAttentionKernels();
} // namespace llaminar2::qwen4exp
