/**
 * @file Qwen4ExpKernels.h
 * @brief Allocation-free FP32 gated-residual and PLE execution contracts.
 *
 * Large projections and branch RMSNorm use the ordinary prepared GEMM/norm
 * bridges. These primitives own only the model-specific elementwise arithmetic,
 * exact integer hashing and dilated history. Candidate evaluation never mutates
 * committed history: the caller supplies distinct persistent input/output banks
 * and explicitly publishes the accepted prefix. All matrices are contiguous,
 * row-major, with checkpoint-native [row, branch, channel] trunk layout.
 */
#pragma once

#include <array>
#include <cstdint>
#include <memory>

namespace llaminar2::qwen4exp
{
/** @brief Checkpoint-native residual branch count. */
inline constexpr int kResidualBranches = 4;
/** @brief Eight bigram heads followed by eight trigram heads. */
inline constexpr int kNgramHeads = 16;
/** @brief Number of preceding normalized values retained by dilation-three convolution. */
inline constexpr int kConvHistoryRows = 9;

/** @brief Admitted row bound and optional execution-owner live row count.
 * @details A null liveRows means all rows are live. On CPU this pointer is host
 * memory; on GPU it is device memory and must never be read by launch code.
 * Inactive output rows are untouched, including large-to-small captured replay.
 */
struct RowExtent
{
    int rows = 0; ///< Admitted positive row capacity; part of capture identity.
    const int *liveRows = nullptr; ///< Optional authoritative count in [0, rows].
};

/** @brief Checkpoint-provided integer hash policy, copied unchanged into launches.
 * @details Multiplications and XOR use exact unsigned 64-bit arithmetic. The
 * checkpoint token/multiplier contract keeps products below INT64_MAX, making
 * unsigned remainder identical to its signed reference. Unsigned wrap is also
 * specified for adversarial tests; no float conversion is permitted.
 */
struct NgramHashPolicy
{
    std::array<std::uint64_t, 3> multipliers{}; ///< Current, preceding, second-preceding multipliers.
    std::array<std::uint64_t, kNgramHeads> moduli{}; ///< Positive head partition sizes.
    std::array<std::uint64_t, kNgramHeads> offsets{}; ///< Absolute row base of each partition.
    std::int64_t eosToken = 248044; ///< Hash reset/padding token, distinct from stop policy.

    /** @brief Reject missing partitions and row-index overflow before capture.
     * @throws std::invalid_argument For invalid policy or signed output overflow.
     */
    void validate() const;
};

/** @brief FP32 RMSNorm with a distinct authenticated gain for each residual branch.
 * @details Gains are exported actual multipliers, so no zero-centered +1 is
 * applied here. Each branch reduces H values through the same 256 arithmetic
 * lanes and binary tree as PLE similarity. Output may alias input exactly.
 */
struct BranchRMSNormArgs
{
    RowExtent extent; ///< Live token rows.
    int hidden = 0; ///< Per-branch width H.
    float epsilon = 1e-6f; ///< Positive finite checkpoint RMS epsilon.
    const float *input = nullptr; ///< [rows,4,H] unnormalized branch values.
    const float *gain = nullptr; ///< [4,H] actual learned gain, distinct per branch.
    float *output = nullptr; ///< [rows,4,H], optionally equal to input.
};

/** @brief Broadcast one H-wide embedding into four independent residual branches. */
struct ResidualExpandArgs
{
    RowExtent extent; ///< Live token rows.
    int hidden = 0; ///< Compute width H, independent of trunk width 4H.
    const float *input = nullptr; ///< [rows,H] ordinary embedding.
    float *output = nullptr; ///< [rows,4,H] distinct trunk storage.
};

/** @brief Scale the GR low-rank down projection by four and apply SiLU. */
struct ResidualDownArgs
{
    RowExtent extent; ///< Live token rows.
    int lowRank = 0; ///< Low-rank width (320 in the released checkpoint).
    const float *input = nullptr; ///< [rows,lowRank] existing GEMM output.
    float *output = nullptr; ///< [rows,lowRank], may alias input.
};

/** @brief Collapse normalized branches with sigmoid read gates; also the final mixer. */
struct ResidualReadArgs
{
    RowExtent extent; ///< Live token rows.
    int hidden = 0; ///< Per-branch compute width.
    const float *normalized = nullptr; ///< [rows,4,H] branch RMSNorm output.
    const float *gateLogits = nullptr; ///< [rows,4,H] low-rank up projection.
    float *output = nullptr; ///< [rows,H] gated mean, distinct from inputs.
};

/** @brief Inject a block output through 2*sigmoid(injectionLogit/4). */
struct ResidualWriteArgs
{
    RowExtent extent; ///< Live token rows.
    int hidden = 0; ///< Per-branch compute width.
    const float *residual = nullptr; ///< [rows,4,H] original unnormalized branches.
    const float *block = nullptr; ///< [rows,H] attention or MoE output.
    const float *gateLogits = nullptr; ///< [rows,4] existing injection GEMM output.
    float *output = nullptr; ///< [rows,4,H], may alias residual.
};

/** @brief Exact PLE suffix hash, without mutation of committed token history. */
struct NgramHashArgs
{
    RowExtent extent; ///< Live token rows in one request.
    const std::int32_t *tokens = nullptr; ///< [rows] raw token IDs on execution device.
    const std::int32_t *priorTokens = nullptr; ///< [2] oldest then newest; initialized to EOS.
    std::int64_t *output = nullptr; ///< [rows,16] absolute immutable-table row IDs.
    NgramHashPolicy policy; ///< Authenticated checkpoint constants.
};

/** @brief PLE similarity and shared-value gating after existing branch norm bridges.
 * @details The 256 fixed arithmetic lanes and ascending binary reduction tree
 * are independent of row count, scheduling, CPU worker count and GPU warp size.
 * Norm inputs have already applied authenticated checkpoint gain semantics.
 */
struct PLEGateArgs
{
    RowExtent extent; ///< Live token rows.
    int hidden = 0; ///< Per-branch width H.
    const float *normalizedKey = nullptr; ///< [rows,4,H] normalized projected key.
    const float *normalizedQuery = nullptr; ///< [rows,4,H] normalized original trunk.
    const float *value = nullptr; ///< [rows,H] shared value projection.
    float *gated = nullptr; ///< [rows,4,H], subsequently normalized by ordinary RMSNorm.
};

/** @brief Four-tap dilation-three convolution with independent candidate rows.
 * @details For row t, tap j consumes t-9+3*j; negative rows read priorHistory.
 * State is chronological [oldest,...,newest], never a mutable circular buffer.
 * Each row/channel is independent, so prefill and verifier work is parallel
 * without row replay. Output is residual + (gated + SiLU(convolution)), retaining
 * the reference's two residual addition boundaries. History remains unchanged.
 */
struct PLEConvArgs
{
    RowExtent extent; ///< Live token rows in one request.
    int channels = 0; ///< Trunk channels, exactly 4H.
    const float *normalized = nullptr; ///< [rows,channels] normalized gated values.
    const float *priorHistory = nullptr; ///< [9,channels] immutable admitted history.
    const float *weights = nullptr; ///< [channels,4], oldest-to-current tap order.
    const float *gated = nullptr; ///< [rows,channels] unnormalized gated values.
    const float *residual = nullptr; ///< [rows,channels] pre-PLE trunk.
    float *output = nullptr; ///< [rows,channels], may alias residual or gated.
};

/** @brief Accepted-prefix publication controls shared by both history families.
 * @details GPU acceptedRows is device-owned. Null selects hostAcceptedRows,
 * which is a captured policy scalar, not a host mirror of device decisions.
 */
struct AcceptedPrefix
{
    RowExtent candidates; ///< Retained candidates and their current live count.
    const int *acceptedRows = nullptr; ///< Optional authoritative accepted-prefix length.
    int hostAcceptedRows = 0; ///< Fixed CPU/setup count if acceptedRows is null.
};

/** @brief Publish the exact last nine normalized rows after an accepted prefix. */
struct ConvHistoryCommitArgs
{
    AcceptedPrefix prefix; ///< Accept/reject decision from execution owner.
    int channels = 0; ///< Trunk width 4H.
    const float *priorHistory = nullptr; ///< [9,channels], immutable and distinct from output.
    const float *candidates = nullptr; ///< [capacity,channels] evaluated normalized values.
    float *outputHistory = nullptr; ///< [9,channels] next history bank, including acceptance zero.
};

/** @brief Publish the last two raw token IDs after an accepted prefix. */
struct TokenHistoryCommitArgs
{
    AcceptedPrefix prefix; ///< Accept/reject decision from execution owner.
    const std::int32_t *priorTokens = nullptr; ///< [2] immutable prior raw IDs.
    const std::int32_t *candidates = nullptr; ///< [capacity] raw candidate IDs.
    std::int32_t *outputTokens = nullptr; ///< [2] distinct next token-history bank.
};

/** @brief Backend-specific allocation-free primitive bundle.
 * @details Every GPU call requires a non-null, non-default exact stream.
 * CPU ignores stream. The graph owns pointers, generations, reset/publication
 * events and bank selection; this object owns no mutable inference state.
 */
class IQwen4ExpKernels
{
public:
    /** @brief Destroy the stateless implementation. */
    virtual ~IQwen4ExpKernels() = default;
    /** @brief Normalize each branch with its own actual gain and fixed reduction tree.
     * @param args Complete immutable operand and admitted-geometry binding.
     * @param stream Exact non-default GPU stream; ignored by the CPU implementation.
     * @throws std::invalid_argument For missing operands or invalid static geometry.
     */
    virtual void branchRMSNorm(const BranchRMSNormArgs &args, void *stream) const = 0;
    /** @brief Expand ordinary embeddings; @throws std::invalid_argument for invalid bindings.
     * @param args Complete immutable operand and admitted-geometry binding.
     * @param stream Exact non-default GPU stream; ignored by the CPU implementation.
     * @throws std::invalid_argument For missing operands or invalid static geometry.
     */
    virtual void residualExpand(const ResidualExpandArgs &args, void *stream) const = 0;
    /** @brief Apply the GR bottleneck activation; rejects invalid bindings.
     * @param args Complete immutable operand and admitted-geometry binding.
     * @param stream Exact non-default GPU stream; ignored by the CPU implementation.
     * @throws std::invalid_argument For missing operands or invalid static geometry.
     */
    virtual void residualDown(const ResidualDownArgs &args, void *stream) const = 0;
    /** @brief Mix the four normalized branches, including final-mixer use.
     * @param args Complete immutable operand and admitted-geometry binding.
     * @param stream Exact non-default GPU stream; ignored by the CPU implementation.
     * @throws std::invalid_argument For missing operands or invalid static geometry.
     */
    virtual void residualRead(const ResidualReadArgs &args, void *stream) const = 0;
    /** @brief Publish four independently gated residual branches.
     * @param args Complete immutable operand and admitted-geometry binding.
     * @param stream Exact non-default GPU stream; ignored by the CPU implementation.
     * @throws std::invalid_argument For missing operands or invalid static geometry.
     */
    virtual void residualWrite(const ResidualWriteArgs &args, void *stream) const = 0;
    /** @brief Produce sixteen exact ngram IDs per live row without changing history.
     * @param args Complete immutable operand and admitted-geometry binding.
     * @param stream Exact non-default GPU stream; ignored by the CPU implementation.
     * @throws std::invalid_argument For missing operands or invalid static geometry.
     */
    virtual void ngramHash(const NgramHashArgs &args, void *stream) const = 0;
    /** @brief Compute signed-square-root similarity gates with a fixed reduction order.
     * @param args Complete immutable operand and admitted-geometry binding.
     * @param stream Exact non-default GPU stream; ignored by the CPU implementation.
     * @throws std::invalid_argument For missing operands or invalid static geometry.
     */
    virtual void pleGate(const PLEGateArgs &args, void *stream) const = 0;
    /** @brief Add the live dilated-convolution PLE contribution without changing history.
     * @param args Complete immutable operand and admitted-geometry binding.
     * @param stream Exact non-default GPU stream; ignored by the CPU implementation.
     * @throws std::invalid_argument For missing operands or invalid static geometry.
     */
    virtual void pleConv(const PLEConvArgs &args, void *stream) const = 0;
    /** @brief Publish convolution history after explicit candidate-prefix acceptance.
     * @param args Complete immutable operand and admitted-geometry binding.
     * @param stream Exact non-default GPU stream; ignored by the CPU implementation.
     * @throws std::invalid_argument For missing operands or invalid static geometry.
     */
    virtual void commitConvHistory(const ConvHistoryCommitArgs &args, void *stream) const = 0;
    /** @brief Publish raw token history after explicit candidate-prefix acceptance.
     * @param args Complete immutable operand and admitted-geometry binding.
     * @param stream Exact non-default GPU stream; ignored by the CPU implementation.
     * @throws std::invalid_argument For missing operands or invalid static geometry.
     */
    virtual void commitTokenHistory(const TokenHistoryCommitArgs &args, void *stream) const = 0;
};

/** @brief Construct the CPU bundle outside captured execution. @return Stateless bundle. */
std::unique_ptr<IQwen4ExpKernels> createCPUQwen4ExpKernels();
/** @brief Construct the CUDA bundle outside captured execution. @return Stateless bundle. */
std::unique_ptr<IQwen4ExpKernels> createCUDAQwen4ExpKernels();
/** @brief Construct the ROCm bundle outside captured execution. @return Stateless bundle. */
std::unique_ptr<IQwen4ExpKernels> createROCmQwen4ExpKernels();
} // namespace llaminar2::qwen4exp
