/**
 * @file CPUFlashAttentionKernelT.h
 * @brief Flash Attention kernel for CPU, using tiled online softmax.
 *
 * ## What is Flash Attention?
 *
 * Standard attention computes `softmax(Q·K^T / √d) · V` by first materialising
 * the full `seq_len × kv_len` score matrix, running softmax over every row, and
 * then multiplying by V.  For long sequences this score matrix can be enormous
 * (e.g. 4096² × sizeof(float) ≈ 64 MB per head) and thrashes the CPU cache.
 *
 * Flash Attention avoids ever materialising the full score matrix.  Instead it
 * processes K/V in *tiles* of `kv_tile` positions at a time, computing a partial
 * softmax that is corrected "on the fly" as new tiles arrive.  This is called
 * **online softmax** (see Milakov & Gimelshein 2018).  Only a small tile of
 * scores lives in L1/L2 at a time, giving a dramatic cache-locality win.
 *
 * ## Template parameter
 *
 * The kernel is templated on the graph activation precision. The production
 * tensor API currently requires FP32 Q/output rows and consumes native FP32,
 * FP16, BF16, Q8_1, Q16_1, TQ4, and TQ8 KV storage directly. Each storage
 * format has a vectorized inner loop; none requires an FP32 cache shadow.
 *
 * ## How this file is organised
 *
 * 1. `cpu::fa2_policy::selectCPUFA2KVTile()` chooses the KV tile size from
 *    detected private-cache capacity and the exact K/V storage format.
 * 2. `detail::FlashAttentionPrecisionToTensor` – maps `ActivationPrecision`
 *    enum values to concrete tensor classes (FP32Tensor, BF16Tensor, …).
 * 3. `CPUFlashAttentionKernelT<Precision>` – the main kernel class.
 *    - Public API: `compute()`, `compute_batch()`, `compute_decode()`,
 *      `compute_tensor()` – front-end methods matching the `ITensorAttention`
 *      interface.
 *    - Private helpers: SIMD dot-products, quantisation routines, and the
 *      core `compute_flash_fp32()` tiled-softmax implementation.
 *
 * @see ITensorAttention     The polymorphic interface this class implements.
 * @see cpu::fa2_policy::CPUFA2KVTileGeometry Cache and storage launch model.
 */

#pragma once

#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/config/RuntimeConfig.h"
#include "../../../tensors/FP16Utils.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TQ8Tensor.h"
#include "../../../tensors/TQ4Tensor.h"
#include "../../../utils/CPUFeatures.h"
#include "../../../utils/Assertions.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/OpenMPUtils.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../primitives/ActivationTraits.h"
#include "../CPUKernelBase.h"
#include "../turboquant/TurboQuantRotation.h"
#include "../turboquant/TurboQuantContext.h"
#include "CPUFlashAttentionLaunchPolicy.h"
#include "TQFusedAttentionPrimitives.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include "../simd/AVX2Helpers.h"
#endif

// Forward declaration for friend access from parity tests
class AVX2Q16DotParityTest;

namespace llaminar2
{
    namespace cpu::AttentionWorkspaceBuffers
    {
        /** Persistent FP32 numerator summaries for CPU K/V-context attention. */
        inline constexpr const char *PARTIAL_OUTPUT = "attn_partial_output";
        /** Persistent per-summary online-softmax maxima. */
        inline constexpr const char *PARTIAL_M = "attn_partial_m";
        /** Persistent per-summary online-softmax denominators. */
        inline constexpr const char *PARTIAL_L = "attn_partial_l";
    } // namespace cpu::AttentionWorkspaceBuffers

    namespace detail
    {
        /**
         * @brief Select one authenticated K/V tile from exact storage geometry.
         *
         * The cache detector reports capacities per physical core. The caller
         * supplies one stored K head row and one stored V head row so FP32,
         * FP16, Q16, Q8, and TurboQuant all use the same normalized policy
         * without lying about memory traffic. Debug tournament controls are
         * exact: an uncompiled positive override is a fatal configuration error
         * rather than being rounded to a different experiment.
         *
         * @param storage_pair Exact native K/V codecs executed by this path.
         * @param head_dim Logical elements in one attention head.
         * @param kv_len Number of addressable K/V rows.
         * @param key_head_row_bytes Stored bytes read for one K head row.
         * @param value_head_row_bytes Stored bytes read for one V head row.
         * @param explicit_tile Stable per-kernel tournament override, or zero.
         * @return A compiled tile in `cpu::fa2_policy::kCompiledKVTiles`.
         */
        [[nodiscard]] inline int selectCPUFlashKVTile(
            cpu::fa2_policy::CPUFA2KVStoragePair storage_pair,
            int head_dim,
            int kv_len,
            std::size_t key_head_row_bytes,
            std::size_t value_head_row_bytes,
            int explicit_tile)
        {
            const CacheInfo &cache = cache_info();
            const auto vector_isa = []
            {
                switch (activeISALevel())
                {
                case ISALevel::Scalar:
                    return cpu::fa2_policy::CPUFA2VectorISA::Scalar;
                case ISALevel::AVX2:
                    return cpu::fa2_policy::CPUFA2VectorISA::AVX2;
                case ISALevel::AVX512:
                    return cpu::fa2_policy::CPUFA2VectorISA::AVX512;
                }
                return cpu::fa2_policy::CPUFA2VectorISA::Invalid;
            }();
            const int tile = cpu::fa2_policy::selectCPUFA2KVTile(
                {
                    .storage_pair = storage_pair,
                    .vector_isa = vector_isa,
                    .codegen_isa =
                        cpu::fa2_policy::compiledCPUFA2CodegenISA(),
                    .head_dim = head_dim,
                    .kv_rows = kv_len,
                    .key_head_row_bytes = key_head_row_bytes,
                    .value_head_row_bytes = value_head_row_bytes,
                    .cache = {
                        .private_l1d_bytes = cache.l1_size,
                        .private_l2_bytes = cache.l2_size,
                        .shared_l3_bytes = cache.l3_size,
                        .cache_line_bytes = cache.cache_line,
                    },
                },
                explicit_tile);
            if (tile == 0)
            {
                LLAMINAR_UNREACHABLE(
                    "invalid CPU FA2 K/V tile request: override="
                    << explicit_tile << " storage="
                    << cpu::fa2_policy::cpuFA2KVStoragePairName(storage_pair)
                    << " isa="
                    << cpu::fa2_policy::cpuFA2VectorISAName(vector_isa)
                    << " codegen="
                    << cpu::fa2_policy::cpuFA2CodegenISAName(
                           cpu::fa2_policy::compiledCPUFA2CodegenISA())
                    << " head_dim=" << head_dim
                    << " kv_len=" << kv_len << " K-row-bytes="
                    << key_head_row_bytes << " V-row-bytes="
                    << value_head_row_bytes);
            }
            return tile;
        }

        /// Maximum KV tile size across all flash attention policies.
        /// Used to size stack-allocated score buffers instead of heap vectors.
        static constexpr int kMaxKVTile = cpu::fa2_policy::kMaximumKVTile;

        /// Maximum I16 row stride for quantized Q buffers.
        /// Covers head_dim up to 256 with 32-element alignment: ((256+31)/32)*32 = 256.
        static constexpr int kMaxI16RowStride = 256;

        /**
         * @brief Authenticated logical-row addressing for one Q16 K/V tensor.
         *
         * Q16 cache tensors have two supported physical layouts. Position-major
         * tensors store every KV head in one physical row, while head-major
         * tensors store one `[head][physical-row]` pair per tensor row. The
         * attention arithmetic must always consume oldest-to-newest logical
         * rows, regardless of both that layout choice and the ring origin.
         *
         * Centralising the formula here is intentional: score, value, prefetch,
         * prefill, and decode loops must not reconstruct subtly different byte
         * offsets. Callers validate the instance once, then use `blockIndex()`
         * for every raw block access.
         */
        struct Q16KVLogicalLayout
        {
            std::size_t blocks_per_tensor_row = 0;
            std::size_t blocks_per_head = 0;
            std::size_t physical_rows_per_head = 0;
            int kv_heads = 0;
            bool head_major = false;
            attention::AttentionKVLogicalView logical_view{};

            /**
             * @brief Validate layout and ring geometry against a tensor pair.
             * @param tensor_rows Number of physical rows in either Q16 tensor.
             * @param logical_rows Number of oldest-to-newest rows to consume.
             * @return True only when every translated block lies in the tensor.
             */
            [[nodiscard]] constexpr bool valid(
                std::size_t tensor_rows,
                int logical_rows) const noexcept
            {
                if (blocks_per_tensor_row == 0 || blocks_per_head == 0 ||
                    physical_rows_per_head == 0 || kv_heads <= 0 ||
                    !logical_view.validFor(logical_rows))
                {
                    return false;
                }

                const std::size_t expected_rows = head_major
                                                      ? physical_rows_per_head *
                                                            static_cast<std::size_t>(kv_heads)
                                                      : physical_rows_per_head;
                if (expected_rows != tensor_rows)
                    return false;

                if (logical_view.isContiguous())
                {
                    return static_cast<std::size_t>(logical_rows) <=
                           physical_rows_per_head;
                }
                return static_cast<std::size_t>(
                           logical_view.physical_row_capacity) <=
                       physical_rows_per_head;
            }

            /**
             * @brief Map one logical KV element block into the raw Q16 array.
             * @param logical_row Oldest-to-newest cache row.
             * @param kv_head KV head stored in that row.
             * @param head_block Block offset within the selected head.
             * @return Flat Q16 block index in the backing tensor.
             */
            [[nodiscard]] constexpr std::size_t blockIndex(
                int logical_row,
                int kv_head,
                std::size_t head_block) const noexcept
            {
                const std::size_t physical_row = static_cast<std::size_t>(
                    logical_view.physicalRow(logical_row));
                if (head_major)
                {
                    const std::size_t tensor_row =
                        static_cast<std::size_t>(kv_head) *
                            physical_rows_per_head +
                        physical_row;
                    return tensor_row * blocks_per_head + head_block;
                }
                return physical_row * blocks_per_tensor_row +
                       static_cast<std::size_t>(kv_head) * blocks_per_head +
                       head_block;
            }
        };

        /**
         * @brief Compile-time map from ActivationPrecision → concrete tensor type.
         *
         * The flash attention kernel needs to know which tensor class
         * corresponds to each precision enum value so it can declare typed
         * aliases.  Each specialisation below simply defines a `Type` member.
         *
         * Example:
         * @code
         * using TensorT = FlashAttentionPrecisionToTensor<ActivationPrecision::FP32>::Type;
         * // TensorT == FP32Tensor
         * @endcode
         */
        template <ActivationPrecision P>
        struct FlashAttentionPrecisionToTensor;

        template <>
        struct FlashAttentionPrecisionToTensor<ActivationPrecision::FP32>
        {
            using Type = FP32Tensor;
        };

        template <>
        struct FlashAttentionPrecisionToTensor<ActivationPrecision::BF16>
        {
            using Type = BF16Tensor;
        };

        template <>
        struct FlashAttentionPrecisionToTensor<ActivationPrecision::FP16>
        {
            using Type = FP16Tensor;
        };
    }

    /**
     * @brief CPU Flash Attention kernel with tiled online softmax.
     *
     * This class implements the `ITensorAttention` interface using an
     * *online-softmax flash-attention* algorithm on the CPU.  It is the
     * primary attention kernel used during CPU inference in Llaminar V2.
     *
     * ## How online softmax works (simplified)
     *
     * Traditional softmax requires two passes over the score vector:
     *   1. Find the maximum (for numerical stability).
     *   2. Compute `exp(s - max)` and sum, then normalise.
     *
     * Online softmax fuses both passes into a single streaming pass by
     * maintaining a *running maximum* (`running_m`) and a *running
     * denominator* (`running_l`).  When a new tile of KV positions arrives
     * with a new local maximum, the previously accumulated output and
     * denominator are *rescaled* by `exp(old_max − new_max)` before the new
     * tile's contributions are added.  At the end, the accumulated output is
     * divided by `running_l` to produce the final normalised result.
     *
     * ## Precision handling
     *
     * When `Precision == FP32`, all work is done directly in this kernel.
     * Non-FP32 precisions are not supported and will return false.
     * Precision dispatch is resolved entirely at compile time via `if constexpr`.
     *
     * ## Thread safety
     *
     * All parallel work is wrapped in `OMP_WORKSHARE_REGION()` so the kernel
     * can participate in an outer parallel region without redundant thread
     * fork/join overhead (see copilot-instructions.md § OpenMP).
     *
     * @tparam Precision  The activation storage format.  Currently FP32 is
     *                    fully optimised; non-FP32 precisions are unsupported.
     *
     * @see cpu::fa2_policy::selectCPUFA2KVTile Tile-size selection logic.
     */
    template <ActivationPrecision Precision>
    class CPUFlashAttentionKernelT : public ITensorAttention,
                                     public CPUKernelBase
    {
    private:
        /**
         * @brief Physical query-row layout consumed by the direct Q16 kernel.
         *
         * Ordinary prefill stores Q as `[row][head][dim]`. Hybrid Q16 RoPE
         * deliberately leaves compact verifier queries as `[head][row][dim]`
         * so serial decode can consume one contiguous head. Teaching the
         * grouped kernel both layouts keeps that producer contract intact and
         * avoids a heap-backed transpose in the verifier hot path.
         */
        enum class Q16QueryLayout : std::uint8_t
        {
            RowMajor,
            HeadMajor,
        };

    public:
        /** @brief The concrete tensor class for this precision (e.g. FP32Tensor). */
        using TensorT = typename detail::FlashAttentionPrecisionToTensor<Precision>::Type;
        /** @brief The scalar element type for this precision (e.g. float). */
        using ElementType = typename primitives::ActivationTraits<TensorT>::ElementType;

        /**
         * @brief Construct a kernel with launch policy snapshotted from config.
         *
         * The debug override is read once here, never from a compute call. This
         * keeps launch behavior stable for the lifetime of a captured or
         * prepared kernel and prevents mutable process state from changing a
         * live request's cache-access pattern.
         */
        CPUFlashAttentionKernelT()
        {
            const int configured_tile = debugEnv().attention.flash_kv_tile;
            configureLaunchPolicy({
                .explicit_kv_tile = configured_tile > 0 ? configured_tile : 0,
            });
        }
        ~CPUFlashAttentionKernelT() override = default;

        /**
         * @brief Install an authenticated launch policy before execution.
         *
         * Performance tournaments use this typed setter between synchronous
         * CPU launches. Production graph/stage setup calls it, if needed,
         * before exposing the kernel to execution. Invalid candidates fail
         * immediately instead of being rounded or deferred to a hot call.
         *
         * @param policy Complete per-instance CPU FA2 launch policy.
         */
        void configureLaunchPolicy(
            const cpu::fa2_policy::CPUFA2KernelLaunchPolicy &policy)
        {
            if (!policy.valid())
            {
                LLAMINAR_UNREACHABLE(
                    "invalid CPU FA2 per-kernel K/V tile override: "
                    << policy.explicit_kv_tile);
            }
            launch_policy_ = policy;
        }

        /**
         * @brief Declare persistent summaries for the largest grouped CPU row set.
         *
         * `m` is the compact query-row envelope supplied by the attention
         * stage, `n` is local query heads, and `k` is head dimension. The split
         * count is deliberately derived from heads and physical workers rather
         * than M, matching the batch-invariant production launch policy.
         *
         * @param m Maximum compact query rows sharing this graph family.
         * @param n Maximum local query heads.
         * @param k Maximum head dimension.
         * @return Required named CPU workspace buffers.
         */
        WorkspaceRequirements getWorkspaceRequirements(
            int m,
            int n = 0,
            int k = 0) const override
        {
            const std::size_t rows = static_cast<std::size_t>(std::max(1, m));
            const std::size_t heads = static_cast<std::size_t>(std::max(1, n));
            const std::size_t head_dim = static_cast<std::size_t>(std::max(1, k));
            const std::size_t workers =
                static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
            const std::size_t producer_slots =
                std::max<std::size_t>(1, (workers + heads - 1) / heads);
            const std::size_t slots_per_row = producer_slots + 1;

            const auto checked_product = [](std::size_t lhs,
                                            std::size_t rhs,
                                            const char *description)
            {
                if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
                {
                    throw std::overflow_error(
                        std::string("CPU FA2 workspace overflow for ") +
                        description);
                }
                return lhs * rhs;
            };

            const std::size_t partial_slots = checked_product(
                checked_product(rows, heads, "rows x heads"),
                slots_per_row,
                "rows x heads x merged/producer slots");
            const std::size_t padded_head_dim = (head_dim + 15U) & ~15U;
            const std::size_t partial_output_elements = checked_product(
                partial_slots,
                padded_head_dim,
                "partial slots x padded head dimension");
            const std::size_t partial_output_bytes = checked_product(
                partial_output_elements,
                sizeof(float),
                "partial output bytes");
            const std::size_t partial_meta_bytes = checked_product(
                partial_slots,
                sizeof(float),
                "partial metadata bytes");

            WorkspaceRequirements requirements;
            requirements.buffers.push_back({
                cpu::AttentionWorkspaceBuffers::PARTIAL_OUTPUT,
                partial_output_bytes,
                64,
                true});
            requirements.buffers.push_back({
                cpu::AttentionWorkspaceBuffers::PARTIAL_M,
                partial_meta_bytes,
                64,
                true});
            requirements.buffers.push_back({
                cpu::AttentionWorkspaceBuffers::PARTIAL_L,
                partial_meta_bytes,
                64,
                true});
            return requirements;
        }

        /**
         * @brief Bind setup-owned CPU attention summaries before execution.
         *
         * Binding caches both addresses and capacities. Hot attention calls do
         * not perform map lookups, allocation, growth, or ownership changes.
         * A context-parallel call with an absent or undersized binding fails
         * instead of entering an internal-allocation path.
         *
         * @param workspace Manager whose named buffers are already allocated.
         */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override
        {
            CPUKernelBase::bindWorkspace(workspace);
            partial_output_ = workspace
                                  ? static_cast<float *>(workspace->getBuffer(
                                        cpu::AttentionWorkspaceBuffers::PARTIAL_OUTPUT))
                                  : nullptr;
            partial_m_ = workspace
                             ? static_cast<float *>(workspace->getBuffer(
                                   cpu::AttentionWorkspaceBuffers::PARTIAL_M))
                             : nullptr;
            partial_l_ = workspace
                             ? static_cast<float *>(workspace->getBuffer(
                                   cpu::AttentionWorkspaceBuffers::PARTIAL_L))
                             : nullptr;
            partial_output_capacity_ = workspace
                                           ? workspace->getBufferSize(
                                                 cpu::AttentionWorkspaceBuffers::PARTIAL_OUTPUT) /
                                                 sizeof(float)
                                           : 0;
            partial_m_capacity_ = workspace
                                      ? workspace->getBufferSize(
                                            cpu::AttentionWorkspaceBuffers::PARTIAL_M) /
                                            sizeof(float)
                                      : 0;
            partial_l_capacity_ = workspace
                                      ? workspace->getBufferSize(
                                            cpu::AttentionWorkspaceBuffers::PARTIAL_L) /
                                            sizeof(float)
                                      : 0;
        }

        /**
         * @brief Reports whether this kernel can run on a given device.
         *
         * Flash attention is a CPU-only kernel, so it only supports device -1
         * (the host / CPU device sentinel).
         *
         * @param device_idx  Device index to check (-1 = CPU).
         * @return true if `device_idx == -1`, false otherwise.
         */
        bool supports_device(int device_idx) const override
        {
            return device_idx == -1;
        }

        /**
         * @brief Compute attention for a single-batch, equal-length Q/KV input.
         *
         * This is the simplest entry point: `seq_len == kv_len`, batch size 1.
         * Raw float pointers are expected in row-major layout:
         *   - Q, K, V: `[seq_len, n_heads/n_kv_heads * head_dim]`
         *   - output:   `[seq_len, n_heads * head_dim]`
         *
         * If the precision is not FP32, the call returns false.
         *
         * @param Q               Query matrix, shape [seq_len, n_heads * head_dim].
         * @param K               Key matrix,   shape [seq_len, n_kv_heads * head_dim].
         * @param V               Value matrix, shape [seq_len, n_kv_heads * head_dim].
         * @param output          Output buffer, same shape as Q.
         * @param seq_len         Number of query (and key/value) positions.
         * @param n_heads         Number of query attention heads.
         * @param n_kv_heads      Number of key/value heads (≤ n_heads; GQA when < n_heads).
         * @param head_dim        Dimension per head (e.g. 64 or 128).
         * @param causal          If true, positions can only attend to earlier positions.
         * @param window_size     Sliding-window size (-1 = unlimited).
         * @param workspace_scores  (unused) Pre-allocated score workspace.
         * @param workspace_buffer  (unused) Pre-allocated scratch buffer.
         * @param workspace_context (unused) Pre-allocated context buffer.
         * @param workspace_mask  Optional additive mask [seq_len, kv_len].
         * @param use_bf16        (unused) BF16 hint – precision is selected at compile time.
         * @param mpi_ctx         (unused) MPI context.
         * @param device_idx      (unused) Device index.
         * @return true on success, false on failure.
         */
        bool compute(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)workspace_scores;
            (void)workspace_buffer;
            (void)workspace_context;
            (void)use_bf16;
            (void)mpi_ctx;
            (void)device_idx;

            if constexpr (!std::is_same_v<ElementType, float>)
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] compute() not supported for non-FP32 precision");
                return false;
            }

            const float *mask = workspace_mask ? workspace_mask->data() : nullptr;
            return compute_flash_fp32(Q, K, V, output,
                                      seq_len, seq_len,
                                      n_heads, n_kv_heads, head_dim,
                                      causal, window_size, 0,
                                      mask);
        }

        /**
         * @brief Compute attention for multiple independent sequences (batch > 1).
         *
         * Each batch element is processed sequentially by calling
         * `compute_flash_fp32()` with the appropriate pointer offsets.
         * Parallelism happens *within* each batch element (over heads).
         *
         * @param Q           Query tensor, shape   [batch, seq_len, n_heads * head_dim].
         * @param K           Key tensor,   shape   [batch, seq_len, n_kv_heads * head_dim].
         * @param V           Value tensor, shape   [batch, seq_len, n_kv_heads * head_dim].
         * @param output      Output tensor, shape  [batch, seq_len, n_heads * head_dim].
         * @param batch_size  Number of independent sequences.
         * @param seq_len     Positions per sequence (same for Q and KV here).
         * @param n_heads     Number of query heads.
         * @param n_kv_heads  Number of key/value heads.
         * @param head_dim    Dimension per head.
         * @param causal      Apply causal masking.
         * @param window_size Sliding-window size (-1 = unlimited).
         * @return true on success, false if any batch element fails.
         */
        bool compute_batch(
            const float *Q, const float *K, const float *V, float *output,
            int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            int head_start = 0,
            int gqa_n_rep = 0)
        {
            (void)workspace_buffer;
            (void)workspace_context;

            if constexpr (!std::is_same_v<ElementType, float>)
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] compute_batch() not supported for non-FP32 precision");
                return false;
            }

            const size_t q_stride = static_cast<size_t>(seq_len) * static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
            const size_t kv_stride = static_cast<size_t>(seq_len) * static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);
            const float *mask = workspace_mask ? workspace_mask->data() : nullptr;

            for (int b = 0; b < batch_size; ++b)
            {
                const float *Q_b = Q + static_cast<size_t>(b) * q_stride;
                const float *K_b = K + static_cast<size_t>(b) * kv_stride;
                const float *V_b = V + static_cast<size_t>(b) * kv_stride;
                float *O_b = output + static_cast<size_t>(b) * q_stride;
                if (!compute_flash_fp32(Q_b, K_b, V_b, O_b,
                                        seq_len, seq_len,
                                        n_heads, n_kv_heads, head_dim,
                                        causal, window_size, 0,
                                        mask,
                                        head_start, gqa_n_rep))
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Compute attention for autoregressive decode (seq_len ≠ kv_len).
         *
         * During decode, typically `seq_len == 1` (the single new token) while
         * `kv_len` is the accumulated context length from the KV cache.  The
         * `position_offset` parameter tells the causal-mask logic what absolute
         * position the first query token sits at (so the mask is
         * `q_abs = position_offset + q_pos`).
         *
         * @param Q               Query, shape [seq_len, n_heads * head_dim].
         * @param K               Keys from KV cache, shape [kv_len, n_kv_heads * head_dim].
         * @param V               Values from KV cache, shape [kv_len, n_kv_heads * head_dim].
         * @param output          Output, shape [seq_len, n_heads * head_dim].
         * @param seq_len         Number of query positions (usually 1 in decode).
         * @param kv_len          Number of cached key/value positions.
         * @param n_heads         Number of query heads.
         * @param n_kv_heads      Number of KV heads.
         * @param head_dim        Dimension per head.
         * @param causal          Apply causal masking (default true for decode).
         * @param position_offset Absolute position of the first query token.
         * @return true on success.
         */
        bool compute_decode(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = true,
            int position_offset = 0)
        {
            if constexpr (!std::is_same_v<ElementType, float>)
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] compute_decode() not supported for non-FP32 precision");
                return false;
            }

            return compute_flash_fp32(Q, K, V, output,
                                      seq_len, kv_len,
                                      n_heads, n_kv_heads, head_dim,
                                      causal, -1, position_offset,
                                      nullptr);
        }

        /**
         * @brief High-level tensor-based entry point used by compute stages.
         *
         * This method accepts polymorphic `ITensor` pointers (the common
         * interface used by `DeviceGraphExecutor` and the compute-stage
         * framework) and routes to the appropriate raw-pointer method.
         *
         * ### Native execution contract
         *
         * Q and output are FP32 activation rows. K/V may remain in any native
         * cache format implemented above; the dispatch enters that format's
         * direct SIMD implementation and returns false for an invalid pairing
         * rather than converting it behind the caller's back. Both declared
         * physical axes reduce the same fixed 256-row summaries in ascending
         * order, making physical scheduling byte-invariant.
         *
         * @param Q                 Query tensor.
         * @param K                 Key tensor.
         * @param V                 Value tensor.
         * @param output            Output tensor (written in-place).
         * @param batch_size        Batch dimension.
         * @param seq_len           Query sequence length.
         * @param kv_len            Key/Value sequence length.
         * @param n_heads           Total number of query heads.
         * @param n_kv_heads        Total number of KV heads.
         * @param head_dim          Dimension per head.
         * @param causal            Apply causal masking.
         * @param window_size       Sliding-window size (-1 = unlimited).
         * @param workspace_scores  Optional pre-allocated score workspace.
         * @param workspace_mask    Optional additive mask tensor.
         * @param mpi_ctx           MPI context (unused on this path).
         * @param device_idx        Device index (unused, CPU only).
         * @param head_start        First head index when sharded (0 = unsharded).
         * @param local_n_heads     Local query heads (-1 = all).
         * @param local_n_kv_heads  Local KV heads (-1 = all).
         * @param execution_policy  Typed physical scheduling policy. Query-row
         *        and K/V-context modes are both native implementations.
         * @return true on success.
         */
        bool compute_tensor(
            const ITensor *Q,
            const ITensor *K,
            const ITensor *V,
            ITensor *output,
            int batch_size,
            int seq_len,
            int kv_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal = false,
            int window_size = -1,
            ITensor *workspace_scores = nullptr,
            ITensor *workspace_mask = nullptr,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            int head_start = 0,
            int local_n_heads = -1,
            int local_n_kv_heads = -1,
            int gqa_n_rep = 0,
            const attention::AttentionExecutionPolicy &execution_policy = {},
            const attention::AttentionKVLogicalView &kv_logical_view = {}) override
        {
            if constexpr (!std::is_same_v<ElementType, float>)
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] compute_tensor() not supported for non-FP32 precision");
                return false;
            }

            if (Q->native_type() != TensorType::FP32 ||
                output->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] compute_tensor() requires FP32 Q and output tensors");
                return false;
            }

            const cpu::fa2_policy::CPUFA2ParallelPlan invocation_plan =
                cpu::fa2_policy::selectCPUFA2ParallelPlan({
                    .batch_size = batch_size,
                    .query_rows = seq_len,
                    .local_query_heads = n_heads,
                    .kv_rows = kv_len,
                    .physical_workers = std::max(1, omp_get_max_threads()),
                    .requested_axis =
                        execution_policy.prefill_parallel_axis,
                });
            if (!invocation_plan.valid)
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] Invalid declared CPU FA2 geometry");
                return false;
            }
            if (!kv_logical_view.validFor(kv_len))
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] Invalid logical KV view: origin="
                          << kv_logical_view.logical_row_origin
                          << " capacity="
                          << kv_logical_view.physical_row_capacity
                          << " logical_rows=" << kv_len);
                return false;
            }

            const auto *Q_base = dynamic_cast<const TensorBase *>(Q);
            const auto *K_base = dynamic_cast<const TensorBase *>(K);
            const auto *V_base = dynamic_cast<const TensorBase *>(V);
            auto *O_base = dynamic_cast<TensorBase *>(output);
            auto *scores_base = dynamic_cast<TensorBase *>(workspace_scores);
            auto *mask_base = dynamic_cast<TensorBase *>(workspace_mask);

            if (!Q_base || !K_base || !V_base || !O_base)
            {
                return false;
            }

            // ---------------------------------------------------------------
            // FP16/BF16 native K/V path: convert in-register for every query
            // count, without allocating or materializing an FP32 shadow cache.
            // ---------------------------------------------------------------
            if (batch_size == 1 &&
                K->native_type() == V->native_type() &&
                (K->native_type() == TensorType::FP16 ||
                 K->native_type() == TensorType::BF16))
            {
                const float *Q_ptr = Q_base->fp32_data();
                float *O_ptr = O_base->mutable_data();
                if (!Q_ptr || !O_ptr)
                    return false;

                const int base_position_offset = (kv_len > seq_len)
                                                     ? (kv_len - seq_len)
                                                     : 0;
                const auto launch = [&]<Native16BitKVFormat Format,
                                        typename Tensor>(
                                        const Tensor *key_tensor,
                                        const Tensor *value_tensor)
                {
                    if (!key_tensor || !value_tensor)
                        return false;
                    return compute_native16kv<Format>(
                        Q_ptr,
                        key_tensor->typed_data(),
                        value_tensor->typed_data(),
                        O_ptr,
                        seq_len,
                        kv_len,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        causal,
                        window_size,
                        base_position_offset,
                        head_start,
                        gqa_n_rep,
                        execution_policy,
                        kv_logical_view);
                };

                if (K->native_type() == TensorType::FP16)
                {
                    return launch.template operator()<
                        Native16BitKVFormat::FP16>(
                        dynamic_cast<const FP16Tensor *>(K),
                        dynamic_cast<const FP16Tensor *>(V));
                }
                return launch.template operator()<
                    Native16BitKVFormat::BF16>(
                    dynamic_cast<const BF16Tensor *>(K),
                    dynamic_cast<const BF16Tensor *>(V));
            }

            // ---------------------------------------------------------------
            // Native Q16_1 VNNI attention for decode, grouped verification,
            // and prefill. One implementation owns every M so all regimes use
            // identical canonical summaries and bounded wave workspace.
            // ---------------------------------------------------------------
#if (defined(__AVX512F__) && defined(__AVX512VNNI__)) || defined(__AVX2__)
            if (K->native_type() == TensorType::Q16_1 &&
                V->native_type() == TensorType::Q16_1 &&
                batch_size == 1)
            {
                const auto *K_q16 = dynamic_cast<const Q16_1Tensor *>(K);
                const auto *V_q16 = dynamic_cast<const Q16_1Tensor *>(V);
                if (K_q16 && V_q16)
                {
                    const float *Q_ptr = Q_base->fp32_data();
                    float *O_ptr = O_base->mutable_data();
                    if (Q_ptr && O_ptr)
                    {
                        const int position_offset = (kv_len > seq_len)
                                                        ? (kv_len - seq_len)
                                                        : 0;
                        return compute_prefill_q16kv(
                            Q_ptr,
                            K_q16,
                            V_q16,
                            O_ptr,
                            seq_len,
                            kv_len,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            causal,
                            window_size,
                            position_offset,
                            head_start,
                            gqa_n_rep,
                            execution_policy,
                            kv_logical_view);
                    }
                }
            }
#endif

            // ---------------------------------------------------------------
            // TurboQuant TQ4/TQ8 native paths: zero shadow buffers for decode,
            // grouped verification, and prefill. Q is rotated once per logical
            // output row, then each K/V row costs O(D). Both symmetric cache
            // modes and the production TQ8-K/TQ4-V asymmetric mode enter this
            // same scheduler and therefore share one arithmetic contract.
            // ---------------------------------------------------------------
#if defined(__AVX512F__) || defined(__AVX2__)
            if ((K->native_type() == TensorType::TQ4 ||
                 K->native_type() == TensorType::TQ8) &&
                (V->native_type() == TensorType::TQ4 ||
                 V->native_type() == TensorType::TQ8) &&
                batch_size == 1)
            {
                const float *Q_ptr = Q_base->fp32_data();
                float *O_ptr = O_base->mutable_data();
                if (!Q_ptr || !O_ptr)
                    return false;

                const int base_position_offset = (kv_len > seq_len)
                                                     ? (kv_len - seq_len)
                                                     : 0;
                const auto launch = [&]<typename KeyTensor,
                                        typename ValueTensor>(
                                        const KeyTensor *key_tensor,
                                        const ValueTensor *value_tensor)
                {
                    if (!key_tensor || !value_tensor ||
                        !key_tensor->turboquant_context() ||
                        !value_tensor->turboquant_context())
                        return false;
                    return compute_tqkv(
                        Q_ptr,
                        key_tensor,
                        value_tensor,
                        O_ptr,
                        seq_len,
                        kv_len,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        causal,
                        window_size,
                        base_position_offset,
                        head_start,
                        gqa_n_rep,
                        execution_policy,
                        kv_logical_view);
                };

                const auto launch_for_key = [&]<typename KeyTensor>(
                                                const KeyTensor *key_tensor)
                {
                    if (V->native_type() == TensorType::TQ4)
                    {
                        return launch(
                            key_tensor,
                            dynamic_cast<const TQ4Tensor *>(V));
                    }
                    return launch(
                        key_tensor,
                        dynamic_cast<const TQ8Tensor *>(V));
                };

                if (K->native_type() == TensorType::TQ4)
                {
                    return launch_for_key(
                        dynamic_cast<const TQ4Tensor *>(K));
                }
                return launch_for_key(
                    dynamic_cast<const TQ8Tensor *>(K));
            }
#endif

            // ---------------------------------------------------------------
            // Q8_1 native K/V path: inline int8-to-float dequantization for
            // decode, grouped verification, and prefill. No FP32 shadow cache
            // is materialized for any query-row count.
            // ---------------------------------------------------------------
#if defined(__AVX512F__) || defined(__AVX2__)
            if (K->native_type() == TensorType::Q8_1 &&
                V->native_type() == TensorType::Q8_1 &&
                batch_size == 1)
            {
                const auto *K_q8 = dynamic_cast<const Q8_1Tensor *>(K);
                const auto *V_q8 = dynamic_cast<const Q8_1Tensor *>(V);
                if (K_q8 && V_q8)
                {
                    const float *Q_ptr = Q_base->fp32_data();
                    float *O_ptr = O_base->mutable_data();
                    if (Q_ptr && O_ptr)
                    {
                        const int position_offset = (kv_len > seq_len)
                                                        ? (kv_len - seq_len)
                                                        : 0;
                        return compute_q8kv(
                            Q_ptr,
                            K_q8,
                            V_q8,
                            O_ptr,
                            seq_len,
                            kv_len,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            causal,
                            window_size,
                            position_offset,
                            head_start,
                            gqa_n_rep,
                            execution_policy,
                            kv_logical_view);
                    }
                }
            }
#endif

            if (K->native_type() != TensorType::FP32 ||
                V->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] No direct CPU attention implementation for K="
                          << static_cast<int>(K->native_type())
                          << " V=" << static_cast<int>(V->native_type()));
                return false;
            }
            if (batch_size > 1 && !kv_logical_view.isContiguous())
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] Request-batched ring views require per-request descriptors");
                return false;
            }

            const float *Q_ptr = Q_base->data();
            const float *K_ptr = K_base->data();
            const float *V_ptr = V_base->data();
            float *O_ptr = O_base->mutable_data();

            if (!Q_ptr || !K_ptr || !V_ptr || !O_ptr)
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] compute_tensor() null data pointer");
                return false;
            }

            if (batch_size > 1)
            {
                return compute_batch(Q_ptr, K_ptr, V_ptr, O_ptr,
                                     batch_size, seq_len, n_heads, n_kv_heads, head_dim,
                                     causal, window_size,
                                     scores_base, nullptr, nullptr, mask_base,
                                     false, mpi_ctx, device_idx,
                                     head_start, gqa_n_rep);
            }

            if (kv_len != seq_len)
            {
                const int position_offset = (kv_len > seq_len) ? (kv_len - seq_len) : 0;
                const float *mask = mask_base ? mask_base->data() : nullptr;
                return compute_flash_fp32(Q_ptr, K_ptr, V_ptr, O_ptr,
                                          seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                                          causal, window_size, position_offset, mask,
                                          head_start, gqa_n_rep,
                                          execution_policy,
                                          kv_logical_view);
            }

            {
                const float *mask = mask_base ? mask_base->data() : nullptr;
                return compute_flash_fp32(Q_ptr, K_ptr, V_ptr, O_ptr,
                                          seq_len, seq_len, n_heads, n_kv_heads, head_dim,
                                          causal, window_size, 0, mask,
                                          head_start, gqa_n_rep,
                                          execution_policy,
                                          kv_logical_view);
            }
        }

        /**
         * @brief Group independent CPU request histories without stage-level replay.
         *
         * Descriptor authentication and storage-format dispatch happen once for
         * the request group. Each independent history then enters the matching
         * native tiled primitive directly; the method never calls the public
         * polymorphic entry point per request and never asks a tensor to expose
         * an FP32 conversion shadow. Long histories retain full-team K/V-context
         * parallelism, so processing requests in stable admission order does not
         * strand cores merely to manufacture request-axis concurrency.
         */
        bool compute_request_batch_decode_equivalent(
            const ITensor *Q,
            const ITensor *const *K_by_request,
            const ITensor *const *V_by_request,
            const int *kv_lens,
            ITensor *output,
            int request_count,
            int query_rows,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal,
            int window_size = -1,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            int head_start = 0,
            int gqa_n_rep = 0,
            const attention::AttentionExecutionPolicy &execution_policy = {},
            const attention::AttentionKVLogicalView *kv_logical_views = nullptr) override
        {
            (void)mpi_ctx;
            (void)device_idx;

            if constexpr (!std::is_same_v<ElementType, float>)
            {
                return false;
            }

            if (!Q || !K_by_request || !V_by_request || !kv_lens ||
                !kv_logical_views || !output ||
                request_count <= 1 || query_rows <= 0 ||
                n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0)
            {
                return false;
            }

            const auto *q_tensor = dynamic_cast<const TensorBase *>(Q);
            auto *output_tensor = dynamic_cast<TensorBase *>(output);
            const float *q = q_tensor ? q_tensor->data() : nullptr;
            float *out = output_tensor ? output_tensor->mutable_data() : nullptr;
            if (!q || !out || Q->native_type() != TensorType::FP32 ||
                output->native_type() != TensorType::FP32)
            {
                return false;
            }

            if (!K_by_request[0] || !V_by_request[0])
                return false;
            const TensorType key_type = K_by_request[0]->native_type();
            const TensorType value_type = V_by_request[0]->native_type();
            for (int request = 0; request < request_count; ++request)
            {
                if (!K_by_request[request] || !V_by_request[request] ||
                    K_by_request[request]->native_type() != key_type ||
                    V_by_request[request]->native_type() != value_type ||
                    kv_lens[request] < query_rows ||
                    !kv_logical_views[request].validFor(kv_lens[request]))
                {
                    LOG_ERROR("[CPUFlashAttentionKernelT] Invalid or heterogeneous native request-batch descriptor"
                              << " request=" << request);
                    return false;
                }
            }

            const size_t query_stride =
                static_cast<size_t>(query_rows) *
                static_cast<size_t>(n_heads) *
                static_cast<size_t>(head_dim);
            const auto run_requests = [&](auto &&launch_request)
            {
                for (int request = 0; request < request_count; ++request)
                {
                    if (!launch_request(
                            request,
                            q + static_cast<size_t>(request) * query_stride,
                            out + static_cast<size_t>(request) * query_stride,
                            kv_lens[request],
                            kv_logical_views[request]))
                    {
                        return false;
                    }
                }
                return true;
            };

            bool success = false;
            if (key_type == TensorType::FP32 &&
                value_type == TensorType::FP32)
            {
                success = run_requests(
                    [&](int request,
                        const float *request_q,
                        float *request_output,
                        int kv_len,
                        const attention::AttentionKVLogicalView &view)
                    {
                        const auto *key = dynamic_cast<const FP32Tensor *>(
                            K_by_request[request]);
                        const auto *value = dynamic_cast<const FP32Tensor *>(
                            V_by_request[request]);
                        return key && value && compute_flash_fp32(
                            request_q,
                            key->data(),
                            value->data(),
                            request_output,
                            query_rows,
                            kv_len,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            causal,
                            window_size,
                            std::max(0, kv_len - query_rows),
                            /*mask=*/nullptr,
                            head_start,
                            gqa_n_rep,
                            execution_policy,
                            view);
                    });
            }
            else if (key_type == TensorType::FP16 &&
                     value_type == TensorType::FP16)
            {
                success = run_requests(
                    [&](int request,
                        const float *request_q,
                        float *request_output,
                        int kv_len,
                        const attention::AttentionKVLogicalView &view)
                    {
                        const auto *key = dynamic_cast<const FP16Tensor *>(
                            K_by_request[request]);
                        const auto *value = dynamic_cast<const FP16Tensor *>(
                            V_by_request[request]);
                        return key && value && compute_native16kv<
                            Native16BitKVFormat::FP16>(
                            request_q,
                            key->typed_data(),
                            value->typed_data(),
                            request_output,
                            query_rows,
                            kv_len,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            causal,
                            window_size,
                            std::max(0, kv_len - query_rows),
                            head_start,
                            gqa_n_rep,
                            execution_policy,
                            view);
                    });
            }
            else if (key_type == TensorType::BF16 &&
                     value_type == TensorType::BF16)
            {
                success = run_requests(
                    [&](int request,
                        const float *request_q,
                        float *request_output,
                        int kv_len,
                        const attention::AttentionKVLogicalView &view)
                    {
                        const auto *key = dynamic_cast<const BF16Tensor *>(
                            K_by_request[request]);
                        const auto *value = dynamic_cast<const BF16Tensor *>(
                            V_by_request[request]);
                        return key && value && compute_native16kv<
                            Native16BitKVFormat::BF16>(
                            request_q,
                            key->typed_data(),
                            value->typed_data(),
                            request_output,
                            query_rows,
                            kv_len,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            causal,
                            window_size,
                            std::max(0, kv_len - query_rows),
                            head_start,
                            gqa_n_rep,
                            execution_policy,
                            view);
                    });
            }
#if defined(__AVX512F__) || defined(__AVX2__)
            else if (key_type == TensorType::Q8_1 &&
                     value_type == TensorType::Q8_1)
            {
                success = run_requests(
                    [&](int request,
                        const float *request_q,
                        float *request_output,
                        int kv_len,
                        const attention::AttentionKVLogicalView &view)
                    {
                        const auto *key = dynamic_cast<const Q8_1Tensor *>(
                            K_by_request[request]);
                        const auto *value = dynamic_cast<const Q8_1Tensor *>(
                            V_by_request[request]);
                        return key && value && compute_q8kv(
                            request_q,
                            key,
                            value,
                            request_output,
                            query_rows,
                            kv_len,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            causal,
                            window_size,
                            std::max(0, kv_len - query_rows),
                            head_start,
                            gqa_n_rep,
                            execution_policy,
                            view);
                    });
            }
            else if ((key_type == TensorType::TQ4 ||
                      key_type == TensorType::TQ8) &&
                     (value_type == TensorType::TQ4 ||
                      value_type == TensorType::TQ8))
            {
                const auto run_tq_pair = [&]<typename KeyTensor,
                                              typename ValueTensor>()
                {
                    return run_requests(
                        [&](int request,
                            const float *request_q,
                            float *request_output,
                            int kv_len,
                            const attention::AttentionKVLogicalView &view)
                        {
                            const auto *key = dynamic_cast<const KeyTensor *>(
                                K_by_request[request]);
                            const auto *value = dynamic_cast<const ValueTensor *>(
                                V_by_request[request]);
                            return key && value &&
                                   key->turboquant_context() &&
                                   value->turboquant_context() ==
                                       key->turboquant_context() &&
                                   compute_tqkv(
                                       request_q,
                                       key,
                                       value,
                                       request_output,
                                       query_rows,
                                       kv_len,
                                       n_heads,
                                       n_kv_heads,
                                       head_dim,
                                       causal,
                                       window_size,
                                       std::max(0, kv_len - query_rows),
                                       head_start,
                                       gqa_n_rep,
                                       execution_policy,
                                       view);
                        });
                };

                if (key_type == TensorType::TQ4)
                {
                    success = value_type == TensorType::TQ4
                                  ? run_tq_pair.template operator()<
                                        TQ4Tensor, TQ4Tensor>()
                                  : run_tq_pair.template operator()<
                                        TQ4Tensor, TQ8Tensor>();
                }
                else
                {
                    success = value_type == TensorType::TQ4
                                  ? run_tq_pair.template operator()<
                                        TQ8Tensor, TQ4Tensor>()
                                  : run_tq_pair.template operator()<
                                        TQ8Tensor, TQ8Tensor>();
                }
            }
#endif
#if (defined(__AVX512F__) && defined(__AVX512VNNI__)) || defined(__AVX2__)
            else if (key_type == TensorType::Q16_1 &&
                     value_type == TensorType::Q16_1)
            {
                success = run_requests(
                    [&](int request,
                        const float *request_q,
                        float *request_output,
                        int kv_len,
                        const attention::AttentionKVLogicalView &view)
                    {
                        const auto *key = dynamic_cast<const Q16_1Tensor *>(
                            K_by_request[request]);
                        const auto *value = dynamic_cast<const Q16_1Tensor *>(
                            V_by_request[request]);
                        if (!key || !value)
                            return false;
                        const int position_offset =
                            std::max(0, kv_len - query_rows);
                        return compute_prefill_q16kv(
                            request_q,
                            key,
                            value,
                            request_output,
                            query_rows,
                            kv_len,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            causal,
                            window_size,
                            position_offset,
                            head_start,
                            gqa_n_rep,
                            execution_policy,
                            view);
                    });
            }
#endif

            if (!success)
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] No direct native request-batch implementation for K="
                          << static_cast<int>(key_type)
                          << " V=" << static_cast<int>(value_type));
                return false;
            }

            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_attention_grouped_request_decode_calls",
                1.0,
                "decode",
                "cpu",
                {{"requests", std::to_string(request_count)},
                 {"query_rows", std::to_string(query_rows)},
                 {"cache_view", "per_request_native_ring"},
                 {"key_type", std::to_string(static_cast<int>(key_type))},
                 {"value_type", std::to_string(static_cast<int>(value_type))},
                 {"math", "serial_decode_equivalent"}});
            return true;
        }

        /**
         * @brief Grouped CPU verifier attention for compact MTP rows.
         *
         * This is the CPU implementation of the Phase 9.8 verifier contract:
         * compute all compact verifier rows in one grouped attention call while
         * preserving the causal visibility of serial one-token decode. Q16_1
         * retains its dedicated grouped VNNI implementation. Every other
         * native CPU cache format enters `compute_tensor()` once for the entire
         * verifier group, so reduced-precision storage cannot drift into an
         * FP32 shadow-materialization path. Unsupported combinations fail
         * closed instead of silently replaying serial rows.
         */
        bool compute_verifier_rows_decode_equivalent(
            const ITensor *Q,
            const ITensor *K,
            const ITensor *V,
            ITensor *output,
            int verifier_rows,
            int kv_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal,
            int window_size = -1,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            int head_start = 0,
            int gqa_n_rep = 0,
            const attention::AttentionKVLogicalView &kv_logical_view = {},
            const attention::AttentionExecutionPolicy &execution_policy = {}) override
        {
            (void)mpi_ctx;
            (void)device_idx;

            if constexpr (!std::is_same_v<ElementType, float>)
            {
                return false;
            }

            if (!Q || !K || !V || !output ||
                verifier_rows < 2 ||
                kv_len <= verifier_rows ||
                n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 ||
                !causal || !kv_logical_view.validFor(kv_len))
            {
                return false;
            }

            const auto *Q_base = dynamic_cast<const TensorBase *>(Q);
            const auto *K_base = dynamic_cast<const TensorBase *>(K);
            const auto *V_base = dynamic_cast<const TensorBase *>(V);
            const auto *K_q16 = dynamic_cast<const Q16_1Tensor *>(K);
            const auto *V_q16 = dynamic_cast<const Q16_1Tensor *>(V);
            auto *O_base = dynamic_cast<TensorBase *>(output);
            if (!Q_base || !K_base || !V_base || !O_base ||
                output->native_type() != TensorType::FP32)
            {
                return false;
            }

            const float *q_src = Q_base->data();
            float *out = O_base->mutable_data();
            if (!q_src || !out)
            {
                return false;
            }

            /*
             * Hybrid Q16 CPU RoPE may leave Q in [head][row][dim] order so the
             * single-row integer attention kernel can consume one head block.
             * The grouped Q16 kernel accepts that physical layout directly;
             * transposing it here would allocate and copy in the verifier hot
             * path.
             */
            const auto &q_shape = Q_base->shape();
            const bool head_major_q =
                Q_base->native_type() == TensorType::Q16_1 &&
                q_shape.size() >= 2 &&
                q_shape[0] == static_cast<size_t>(n_heads * verifier_rows) &&
                q_shape[1] == static_cast<size_t>(head_dim);

            const int base_position_offset = kv_len - verifier_rows;
            if (K_q16 && V_q16)
            {
                const bool success = compute_prefill_q16kv(
                    q_src,
                    K_q16,
                    V_q16,
                    out,
                    verifier_rows,
                    kv_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    causal,
                    window_size,
                    base_position_offset,
                    head_start,
                    gqa_n_rep,
                    execution_policy,
                    kv_logical_view,
                    head_major_q
                        ? Q16QueryLayout::HeadMajor
                        : Q16QueryLayout::RowMajor);
                if (success)
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "cpu_attention_grouped_verifier_rows_calls",
                        1.0,
                        "verifier",
                        "cpu",
                        {{"cache_format", "q16_1"},
                         {"verifier_rows", std::to_string(verifier_rows)},
                         {"kv_len", std::to_string(kv_len)},
                         {"n_heads", std::to_string(n_heads)},
                         {"n_kv_heads", std::to_string(n_kv_heads)},
                         {"head_dim", std::to_string(head_dim)},
                         {"tile_policy", "serial_decode_equivalent"}});
                }
                return success;
            }

            if (Q->native_type() != TensorType::FP32)
                return false;

            const bool success = compute_tensor(
                Q,
                K,
                V,
                output,
                /*batch_size=*/1,
                verifier_rows,
                kv_len,
                n_heads,
                n_kv_heads,
                head_dim,
                causal,
                window_size,
                /*workspace_scores=*/nullptr,
                /*workspace_mask=*/nullptr,
                mpi_ctx,
                device_idx,
                head_start,
                /*local_n_heads=*/-1,
                /*local_n_kv_heads=*/-1,
                gqa_n_rep,
                execution_policy,
                kv_logical_view);
            if (success)
            {
                const auto cache_format = [&]() -> const char *
                {
                    if (K->native_type() == TensorType::FP32 &&
                        V->native_type() == TensorType::FP32)
                        return "fp32";
                    if (K->native_type() == TensorType::FP16 &&
                        V->native_type() == TensorType::FP16)
                        return "fp16";
                    if (K->native_type() == TensorType::BF16 &&
                        V->native_type() == TensorType::BF16)
                        return "bf16";
                    if (K->native_type() == TensorType::Q8_1 &&
                        V->native_type() == TensorType::Q8_1)
                        return "q8_1";
                    if (K->native_type() == TensorType::TQ4 &&
                        V->native_type() == TensorType::TQ4)
                        return "tq4_k_tq4_v";
                    if (K->native_type() == TensorType::TQ8 &&
                        V->native_type() == TensorType::TQ4)
                        return "tq8_k_tq4_v";
                    if (K->native_type() == TensorType::TQ8 &&
                        V->native_type() == TensorType::TQ8)
                        return "tq8_k_tq8_v";
                    return "invalid";
                }();
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_attention_grouped_verifier_rows_calls",
                    1.0,
                    "verifier",
                    "cpu",
                    {{"cache_format", cache_format},
                     {"verifier_rows", std::to_string(verifier_rows)},
                     {"kv_len", std::to_string(kv_len)},
                     {"n_heads", std::to_string(n_heads)},
                     {"n_kv_heads", std::to_string(n_kv_heads)},
                     {"head_dim", std::to_string(head_dim)},
                     {"tile_policy", "serial_decode_equivalent"}});
            }
            return success;
        }

        /**
         * @brief Return metadata describing this kernel's I/O for the snapshot framework.
         *
         * The snapshot/dump system uses this to know what tensors to capture
         * (Q, K, V, output) and what scalar parameters exist (seq_len, etc.)
         * when `LLAMINAR_STAGE_DUMP_ENABLED=1`.
         */
        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::attention()
                .withInput("Q", "query tensor [seq_len, n_heads * head_dim]", KernelBufferDtype::FP32)
                .withInput("K", "key tensor [kv_len, n_kv_heads * head_dim]", KernelBufferDtype::FP32)
                .withInput("V", "value tensor [kv_len, n_kv_heads * head_dim]", KernelBufferDtype::FP32)
                .withOutput("output", "attention output [seq_len, n_heads * head_dim]", KernelBufferDtype::FP32)
                .withScalar("seq_len", "query sequence length", KernelBufferDtype::INT32)
                .withScalar("kv_len", "key/value sequence length", KernelBufferDtype::INT32)
                .withScalar("n_heads", "number of query heads", KernelBufferDtype::INT32)
                .withScalar("n_kv_heads", "number of key/value heads", KernelBufferDtype::INT32)
                .withScalar("head_dim", "dimension per head", KernelBufferDtype::INT32)
                .withScalar("causal", "apply causal masking", KernelBufferDtype::INT32);
        }

        // Allow parity tests to call private static helpers directly
        friend class ::AVX2Q16DotParityTest;

    private:
        // -----------------------------------------------------------------
        // Dot-product helpers
        // -----------------------------------------------------------------

        /**
         * @brief Scalar (no-SIMD) dot product of two FP32 vectors.
         *
         * Used as a fallback when AVX-512 is not available at compile time.
         *
         * @param a  First input vector.
         * @param b  Second input vector.
         * @param n  Number of elements.
         * @return   The dot product `Σ a[i]*b[i]`.
         */
        static float dot_fp32_scalar(const float *a, const float *b, int n)
        {
            float sum = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                sum += a[i] * b[i];
            }
            return sum;
        }

        /**
         * @brief AVX-512 vectorised dot product of two FP32 vectors.
         *
         * Processes 16 floats per iteration using `_mm512_fmadd_ps`, with a
         * scalar tail loop for the remaining elements.  Falls back to the
         * scalar version if AVX-512 is not available at compile time.
         *
         * @param a  First input vector.
         * @param b  Second input vector.
         * @param n  Number of elements.
         * @return   The dot product `Σ a[i]*b[i]`.
         */
        static float dot_fp32_avx512(const float *a, const float *b, int n)
        {
#if defined(__AVX512F__)
            __m512 acc = _mm512_setzero_ps();
            int i = 0;
            for (; i + 15 < n; i += 16)
            {
                __m512 va = _mm512_loadu_ps(a + i);
                __m512 vb = _mm512_loadu_ps(b + i);
                acc = _mm512_fmadd_ps(va, vb, acc);
            }
            float sum = _mm512_reduce_add_ps(acc);
            for (; i < n; ++i)
            {
                sum += a[i] * b[i];
            }
            return sum;
#else
            return dot_fp32_scalar(a, b, n);
#endif
        }

        /** @brief Runtime dispatch: FP32 dot product. */
        static float dot_fp32(const float *a, const float *b, int n)
        {
            switch (activeISALevel())
            {
#if defined(__AVX512F__)
            case ISALevel::AVX512:
                return dot_fp32_avx512(a, b, n);
#endif
            case ISALevel::AVX2:
                return dot_fp32_avx2(a, b, n);
            default:
                return dot_fp32_scalar(a, b, n);
            }
        }

        /**
         * @brief AVX2 vectorised dot product of two FP32 vectors.
         *
         * Processes 8 floats per iteration using `_mm256_fmadd_ps`.
         */
        static float dot_fp32_avx2(const float *a, const float *b, int n)
        {
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for (; i + 7 < n; i += 8)
            {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                acc = _mm256_fmadd_ps(va, vb, acc);
            }
            // Horizontal sum: 8 → 4 → 2 → 1
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 sum4 = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(sum4);
            sum4 = _mm_add_ps(sum4, shuf);
            shuf = _mm_movehl_ps(shuf, sum4);
            sum4 = _mm_add_ss(sum4, shuf);
            float sum = _mm_cvtss_f32(sum4);
            for (; i < n; ++i)
                sum += a[i] * b[i];
            return sum;
        }

#if defined(__AVX512F__)
        /**
         * @brief Fast vectorised exp() for 16 FP32 values using AVX-512.
         *
         * Uses range reduction (x = n*ln2 + f) and a 5th-order polynomial
         * approximation of exp(f) on [-ln2/2, ln2/2].  Inputs below -88.7
         * are clamped to produce 0 (avoids NaN from -inf).
         * Accuracy: <1 ULP for |x| < 87.
         */
        static inline __m512 fast_exp_avx512(__m512 x)
        {
            // Clamp to avoid NaN from -inf inputs (exp(-88.7) ≈ 0)
            x = _mm512_max_ps(x, _mm512_set1_ps(-88.722839f));

            const __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
            const __m512 ln2_hi = _mm512_set1_ps(0.693145751953125f);
            const __m512 ln2_lo = _mm512_set1_ps(1.42860682030941723212e-06f);

            // Range reduction: n = round(x / ln2), f = x - n*ln2
            __m512 t = _mm512_mul_ps(x, log2e);
            __m512 n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512 f = _mm512_fnmadd_ps(n, ln2_hi, x);
            f = _mm512_fnmadd_ps(n, ln2_lo, f);

            // Horner evaluation of exp(f) ≈ 1 + f + f²/2 + f³/6 + f⁴/24 + f⁵/120
            __m512 p = _mm512_set1_ps(8.36564774e-03f);                 // ≈ 1/120
            p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(4.16689515e-02f)); // ≈ 1/24
            p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.66666716e-01f)); // ≈ 1/6
            p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(4.99999851e-01f)); // ≈ 1/2
            p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.00000000e+00f)); // 1
            p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.00000000e+00f)); // 1

            // Construct 2^n by setting the IEEE754 exponent field
            // Clamp ni >= 0 so that extreme negative inputs (masked -inf
            // positions clamped to -88.7) produce 0 rather than NaN.
            __m512i ni = _mm512_cvtps_epi32(n);
            ni = _mm512_add_epi32(ni, _mm512_set1_epi32(127));
            ni = _mm512_max_epi32(ni, _mm512_setzero_si512());
            ni = _mm512_slli_epi32(ni, 23);
            return _mm512_mul_ps(p, _mm512_castsi512_ps(ni));
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief Fast vectorised exp() for 8 FP32 values using AVX2.
         *
         * Same polynomial and range reduction as fast_exp_avx512 but
         * processes 8 floats (YMM) instead of 16 (ZMM).
         */
        static inline __m256 fast_exp_avx2(__m256 x)
        {
            // Clamp to avoid NaN from -inf inputs
            x = _mm256_max_ps(x, _mm256_set1_ps(-88.722839f));

            const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
            const __m256 ln2_hi = _mm256_set1_ps(0.693145751953125f);
            const __m256 ln2_lo = _mm256_set1_ps(1.42860682030941723212e-06f);

            // Range reduction: n = round(x / ln2), f = x - n*ln2
            __m256 t = _mm256_mul_ps(x, log2e);
            __m256 n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 f = _mm256_fnmadd_ps(n, ln2_hi, x);
            f = _mm256_fnmadd_ps(n, ln2_lo, f);

            // Horner polynomial for exp(f)
            __m256 p = _mm256_set1_ps(8.36564774e-03f);
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(4.16689515e-02f));
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.66666716e-01f));
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(4.99999851e-01f));
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.00000000e+00f));
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.00000000e+00f));

            // Construct 2^n via IEEE754 exponent field
            __m256i ni = _mm256_cvtps_epi32(n);
            ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
            ni = _mm256_max_epi32(ni, _mm256_setzero_si256());
            ni = _mm256_slli_epi32(ni, 23);
            return _mm256_mul_ps(p, _mm256_castsi256_ps(ni));
        }
#endif

        /**
         * @brief Compute 4 dot products Q·K[0..3] simultaneously for ILP.
         *
         * By running 4 independent FMA accumulator chains, we keep both
         * FMA ports busy on Cascade Lake (2 FMA/cycle, 4-cycle latency).
         * Throughput: ~13 cycles per K-row vs ~32 for the single-accumulator
         * dot_fp32_avx512 path (limited by FMA latency).
         */
#if defined(__AVX512F__)
        static void dot_fp32_avx512_4row(
            const float *q,
            const float *k0, const float *k1, const float *k2, const float *k3,
            int head_dim,
            float &s0, float &s1, float &s2, float &s3)
        {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            int d = 0;
            for (; d + 15 < head_dim; d += 16)
            {
                __m512 vq = _mm512_loadu_ps(q + d);
                acc0 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(k0 + d), acc0);
                acc1 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(k1 + d), acc1);
                acc2 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(k2 + d), acc2);
                acc3 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(k3 + d), acc3);
            }

            s0 = _mm512_reduce_add_ps(acc0);
            s1 = _mm512_reduce_add_ps(acc1);
            s2 = _mm512_reduce_add_ps(acc2);
            s3 = _mm512_reduce_add_ps(acc3);

            for (; d < head_dim; ++d)
            {
                float qd = q[d];
                s0 += qd * k0[d];
                s1 += qd * k1[d];
                s2 += qd * k2[d];
                s3 += qd * k3[d];
            }
        }

#endif

        /**
         * @brief Native two-byte KV encodings consumed without an FP32 shadow.
         *
         * FP16 and BF16 have identical storage width but different bit layouts.
         * Carrying the distinction as a template argument lets one attention
         * scheduler share its arithmetic order while each ISA emits the
         * correct in-register conversion instruction sequence.
         */
        enum class Native16BitKVFormat : std::uint8_t
        {
            FP16,
            BF16,
        };

        /** @brief Decode one native 16-bit KV element for scalar execution. */
        template <Native16BitKVFormat Format>
        static float decode_native16_scalar(std::uint16_t value)
        {
            if constexpr (Format == Native16BitKVFormat::FP16)
                return fp16_to_fp32(value);
            return simd::bf16_to_fp32(value);
        }

        /** @brief Scalar FP32-query dot product against one native 16-bit row. */
        template <Native16BitKVFormat Format>
        static float dot_native16_scalar(
            const float *query,
            const std::uint16_t *key,
            int head_dim)
        {
            float sum = 0.0f;
            for (int d = 0; d < head_dim; ++d)
                sum += query[d] * decode_native16_scalar<Format>(key[d]);
            return sum;
        }

        /** @brief Scalar four-row native 16-bit dot product with independent accumulators. */
        template <Native16BitKVFormat Format>
        static void dot_native16_4row_scalar(
            const float *query,
            const std::uint16_t *key0,
            const std::uint16_t *key1,
            const std::uint16_t *key2,
            const std::uint16_t *key3,
            int head_dim,
            float &score0,
            float &score1,
            float &score2,
            float &score3)
        {
            score0 = 0.0f;
            score1 = 0.0f;
            score2 = 0.0f;
            score3 = 0.0f;
            for (int d = 0; d < head_dim; ++d)
            {
                const float q = query[d];
                score0 += q * decode_native16_scalar<Format>(key0[d]);
                score1 += q * decode_native16_scalar<Format>(key1[d]);
                score2 += q * decode_native16_scalar<Format>(key2[d]);
                score3 += q * decode_native16_scalar<Format>(key3[d]);
            }
        }

        /** @brief Scalar weighted accumulation from one native 16-bit V row. */
        template <Native16BitKVFormat Format>
        static void accum_native16_scalar(
            float *output,
            const std::uint16_t *value,
            float weight,
            int head_dim)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                output[d] +=
                    weight * decode_native16_scalar<Format>(value[d]);
            }
        }

        /** @brief Scalar weighted accumulation from four native 16-bit V rows. */
        template <Native16BitKVFormat Format>
        static void accum_native16_4row_scalar(
            float *output,
            const std::uint16_t *value0,
            float weight0,
            const std::uint16_t *value1,
            float weight1,
            const std::uint16_t *value2,
            float weight2,
            const std::uint16_t *value3,
            float weight3,
            int head_dim)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                output[d] +=
                    weight0 * decode_native16_scalar<Format>(value0[d]) +
                    weight1 * decode_native16_scalar<Format>(value1[d]) +
                    weight2 * decode_native16_scalar<Format>(value2[d]) +
                    weight3 * decode_native16_scalar<Format>(value3[d]);
            }
        }

#if defined(__AVX2__) && defined(__F16C__)
        /** @brief Convert eight FP16 or BF16 elements to one AVX2 FP32 vector. */
        template <Native16BitKVFormat Format>
        static __m256 load_native16_avx2(const std::uint16_t *source)
        {
            const __m128i packed = _mm_loadu_si128(
                reinterpret_cast<const __m128i *>(source));
            if constexpr (Format == Native16BitKVFormat::FP16)
                return _mm256_cvtph_ps(packed);

            const __m256i expanded = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(packed),
                16);
            return _mm256_castsi256_ps(expanded);
        }

        /** @brief AVX2 FP32-query dot product against one native 16-bit row. */
        template <Native16BitKVFormat Format>
        static float dot_native16_avx2(
            const float *query,
            const std::uint16_t *key,
            int head_dim)
        {
            __m256 accumulator = _mm256_setzero_ps();
            int d = 0;
            for (; d + 8 <= head_dim; d += 8)
            {
                accumulator = _mm256_fmadd_ps(
                    _mm256_loadu_ps(query + d),
                    load_native16_avx2<Format>(key + d),
                    accumulator);
            }
            float sum = avx2::hsum_ps(accumulator);
            for (; d < head_dim; ++d)
                sum += query[d] * decode_native16_scalar<Format>(key[d]);
            return sum;
        }

        /** @brief AVX2 four-row dot product for native 16-bit K storage. */
        template <Native16BitKVFormat Format>
        static void dot_native16_4row_avx2(
            const float *query,
            const std::uint16_t *key0,
            const std::uint16_t *key1,
            const std::uint16_t *key2,
            const std::uint16_t *key3,
            int head_dim,
            float &score0,
            float &score1,
            float &score2,
            float &score3)
        {
            __m256 accumulator0 = _mm256_setzero_ps();
            __m256 accumulator1 = _mm256_setzero_ps();
            __m256 accumulator2 = _mm256_setzero_ps();
            __m256 accumulator3 = _mm256_setzero_ps();
            int d = 0;
            for (; d + 8 <= head_dim; d += 8)
            {
                const __m256 q = _mm256_loadu_ps(query + d);
                accumulator0 = _mm256_fmadd_ps(
                    q, load_native16_avx2<Format>(key0 + d), accumulator0);
                accumulator1 = _mm256_fmadd_ps(
                    q, load_native16_avx2<Format>(key1 + d), accumulator1);
                accumulator2 = _mm256_fmadd_ps(
                    q, load_native16_avx2<Format>(key2 + d), accumulator2);
                accumulator3 = _mm256_fmadd_ps(
                    q, load_native16_avx2<Format>(key3 + d), accumulator3);
            }
            score0 = avx2::hsum_ps(accumulator0);
            score1 = avx2::hsum_ps(accumulator1);
            score2 = avx2::hsum_ps(accumulator2);
            score3 = avx2::hsum_ps(accumulator3);
            for (; d < head_dim; ++d)
            {
                const float q = query[d];
                score0 += q * decode_native16_scalar<Format>(key0[d]);
                score1 += q * decode_native16_scalar<Format>(key1[d]);
                score2 += q * decode_native16_scalar<Format>(key2[d]);
                score3 += q * decode_native16_scalar<Format>(key3[d]);
            }
        }

        /** @brief AVX2 weighted accumulation from one native 16-bit V row. */
        template <Native16BitKVFormat Format>
        static void accum_native16_avx2(
            float *output,
            const std::uint16_t *value,
            float weight,
            int head_dim)
        {
            const __m256 vector_weight = _mm256_set1_ps(weight);
            int d = 0;
            for (; d + 8 <= head_dim; d += 8)
            {
                const __m256 previous = _mm256_loadu_ps(output + d);
                _mm256_storeu_ps(
                    output + d,
                    _mm256_fmadd_ps(
                        load_native16_avx2<Format>(value + d),
                        vector_weight,
                        previous));
            }
            for (; d < head_dim; ++d)
            {
                output[d] +=
                    weight * decode_native16_scalar<Format>(value[d]);
            }
        }

        /** @brief AVX2 weighted accumulation from four native 16-bit V rows. */
        template <Native16BitKVFormat Format>
        static void accum_native16_4row_avx2(
            float *output,
            const std::uint16_t *value0,
            float weight0,
            const std::uint16_t *value1,
            float weight1,
            const std::uint16_t *value2,
            float weight2,
            const std::uint16_t *value3,
            float weight3,
            int head_dim)
        {
            const __m256 vector_weight0 = _mm256_set1_ps(weight0);
            const __m256 vector_weight1 = _mm256_set1_ps(weight1);
            const __m256 vector_weight2 = _mm256_set1_ps(weight2);
            const __m256 vector_weight3 = _mm256_set1_ps(weight3);
            int d = 0;
            for (; d + 8 <= head_dim; d += 8)
            {
                __m256 accumulated = _mm256_loadu_ps(output + d);
                accumulated = _mm256_fmadd_ps(
                    load_native16_avx2<Format>(value0 + d),
                    vector_weight0,
                    accumulated);
                accumulated = _mm256_fmadd_ps(
                    load_native16_avx2<Format>(value1 + d),
                    vector_weight1,
                    accumulated);
                accumulated = _mm256_fmadd_ps(
                    load_native16_avx2<Format>(value2 + d),
                    vector_weight2,
                    accumulated);
                accumulated = _mm256_fmadd_ps(
                    load_native16_avx2<Format>(value3 + d),
                    vector_weight3,
                    accumulated);
                _mm256_storeu_ps(output + d, accumulated);
            }
            for (; d < head_dim; ++d)
            {
                output[d] +=
                    weight0 * decode_native16_scalar<Format>(value0[d]) +
                    weight1 * decode_native16_scalar<Format>(value1[d]) +
                    weight2 * decode_native16_scalar<Format>(value2[d]) +
                    weight3 * decode_native16_scalar<Format>(value3[d]);
            }
        }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__F16C__)
        /** @brief Convert sixteen FP16 or BF16 elements to one AVX-512 FP32 vector. */
        template <Native16BitKVFormat Format>
        static __m512 load_native16_avx512(const std::uint16_t *source)
        {
            const __m256i packed = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(source));
            if constexpr (Format == Native16BitKVFormat::FP16)
                return _mm512_cvtph_ps(packed);

            const __m512i expanded = _mm512_slli_epi32(
                _mm512_cvtepu16_epi32(packed),
                16);
            return _mm512_castsi512_ps(expanded);
        }

        /** @brief AVX-512 FP32-query dot product against one native 16-bit row. */
        template <Native16BitKVFormat Format>
        static float dot_native16_avx512(
            const float *query,
            const std::uint16_t *key,
            int head_dim)
        {
            __m512 accumulator = _mm512_setzero_ps();
            int d = 0;
            for (; d + 16 <= head_dim; d += 16)
            {
                accumulator = _mm512_fmadd_ps(
                    _mm512_loadu_ps(query + d),
                    load_native16_avx512<Format>(key + d),
                    accumulator);
            }
            float sum = _mm512_reduce_add_ps(accumulator);
            for (; d < head_dim; ++d)
                sum += query[d] * decode_native16_scalar<Format>(key[d]);
            return sum;
        }

        /** @brief AVX-512 four-row dot product for native 16-bit K storage. */
        template <Native16BitKVFormat Format>
        static void dot_native16_4row_avx512(
            const float *query,
            const std::uint16_t *key0,
            const std::uint16_t *key1,
            const std::uint16_t *key2,
            const std::uint16_t *key3,
            int head_dim,
            float &score0,
            float &score1,
            float &score2,
            float &score3)
        {
            __m512 accumulator0 = _mm512_setzero_ps();
            __m512 accumulator1 = _mm512_setzero_ps();
            __m512 accumulator2 = _mm512_setzero_ps();
            __m512 accumulator3 = _mm512_setzero_ps();
            int d = 0;
            for (; d + 16 <= head_dim; d += 16)
            {
                const __m512 q = _mm512_loadu_ps(query + d);
                accumulator0 = _mm512_fmadd_ps(
                    q, load_native16_avx512<Format>(key0 + d), accumulator0);
                accumulator1 = _mm512_fmadd_ps(
                    q, load_native16_avx512<Format>(key1 + d), accumulator1);
                accumulator2 = _mm512_fmadd_ps(
                    q, load_native16_avx512<Format>(key2 + d), accumulator2);
                accumulator3 = _mm512_fmadd_ps(
                    q, load_native16_avx512<Format>(key3 + d), accumulator3);
            }
            score0 = _mm512_reduce_add_ps(accumulator0);
            score1 = _mm512_reduce_add_ps(accumulator1);
            score2 = _mm512_reduce_add_ps(accumulator2);
            score3 = _mm512_reduce_add_ps(accumulator3);
            for (; d < head_dim; ++d)
            {
                const float q = query[d];
                score0 += q * decode_native16_scalar<Format>(key0[d]);
                score1 += q * decode_native16_scalar<Format>(key1[d]);
                score2 += q * decode_native16_scalar<Format>(key2[d]);
                score3 += q * decode_native16_scalar<Format>(key3[d]);
            }
        }

        /** @brief AVX-512 weighted accumulation from one native 16-bit V row. */
        template <Native16BitKVFormat Format>
        static void accum_native16_avx512(
            float *output,
            const std::uint16_t *value,
            float weight,
            int head_dim)
        {
            const __m512 vector_weight = _mm512_set1_ps(weight);
            int d = 0;
            for (; d + 16 <= head_dim; d += 16)
            {
                const __m512 previous = _mm512_loadu_ps(output + d);
                _mm512_storeu_ps(
                    output + d,
                    _mm512_fmadd_ps(
                        load_native16_avx512<Format>(value + d),
                        vector_weight,
                        previous));
            }
            for (; d < head_dim; ++d)
            {
                output[d] +=
                    weight * decode_native16_scalar<Format>(value[d]);
            }
        }

        /** @brief AVX-512 weighted accumulation from four native 16-bit V rows. */
        template <Native16BitKVFormat Format>
        static void accum_native16_4row_avx512(
            float *output,
            const std::uint16_t *value0,
            float weight0,
            const std::uint16_t *value1,
            float weight1,
            const std::uint16_t *value2,
            float weight2,
            const std::uint16_t *value3,
            float weight3,
            int head_dim)
        {
            const __m512 vector_weight0 = _mm512_set1_ps(weight0);
            const __m512 vector_weight1 = _mm512_set1_ps(weight1);
            const __m512 vector_weight2 = _mm512_set1_ps(weight2);
            const __m512 vector_weight3 = _mm512_set1_ps(weight3);
            int d = 0;
            for (; d + 16 <= head_dim; d += 16)
            {
                __m512 accumulated = _mm512_loadu_ps(output + d);
                accumulated = _mm512_fmadd_ps(
                    load_native16_avx512<Format>(value0 + d),
                    vector_weight0,
                    accumulated);
                accumulated = _mm512_fmadd_ps(
                    load_native16_avx512<Format>(value1 + d),
                    vector_weight1,
                    accumulated);
                accumulated = _mm512_fmadd_ps(
                    load_native16_avx512<Format>(value2 + d),
                    vector_weight2,
                    accumulated);
                accumulated = _mm512_fmadd_ps(
                    load_native16_avx512<Format>(value3 + d),
                    vector_weight3,
                    accumulated);
                _mm512_storeu_ps(output + d, accumulated);
            }
            for (; d < head_dim; ++d)
            {
                output[d] +=
                    weight0 * decode_native16_scalar<Format>(value0[d]) +
                    weight1 * decode_native16_scalar<Format>(value1[d]) +
                    weight2 * decode_native16_scalar<Format>(value2[d]) +
                    weight3 * decode_native16_scalar<Format>(value3[d]);
            }
        }
#endif

        /** @brief Runtime-ISA dispatch for one native 16-bit K dot product. */
        template <Native16BitKVFormat Format>
        static float dot_native16(
            const float *query,
            const std::uint16_t *key,
            int head_dim)
        {
            switch (activeISALevel())
            {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__F16C__)
            case ISALevel::AVX512:
                return dot_native16_avx512<Format>(query, key, head_dim);
#endif
#if defined(__AVX2__) && defined(__F16C__)
            case ISALevel::AVX2:
                return dot_native16_avx2<Format>(query, key, head_dim);
#endif
            default:
                return dot_native16_scalar<Format>(query, key, head_dim);
            }
        }

        /** @brief Runtime-ISA dispatch for four native 16-bit K dot products. */
        template <Native16BitKVFormat Format>
        static void dot_native16_4row(
            const float *query,
            const std::uint16_t *key0,
            const std::uint16_t *key1,
            const std::uint16_t *key2,
            const std::uint16_t *key3,
            int head_dim,
            float &score0,
            float &score1,
            float &score2,
            float &score3)
        {
            switch (activeISALevel())
            {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__F16C__)
            case ISALevel::AVX512:
                dot_native16_4row_avx512<Format>(
                    query, key0, key1, key2, key3, head_dim,
                    score0, score1, score2, score3);
                return;
#endif
#if defined(__AVX2__) && defined(__F16C__)
            case ISALevel::AVX2:
                dot_native16_4row_avx2<Format>(
                    query, key0, key1, key2, key3, head_dim,
                    score0, score1, score2, score3);
                return;
#endif
            default:
                dot_native16_4row_scalar<Format>(
                    query, key0, key1, key2, key3, head_dim,
                    score0, score1, score2, score3);
            }
        }

        /** @brief Runtime-ISA dispatch for one native 16-bit V accumulation. */
        template <Native16BitKVFormat Format>
        static void accum_native16(
            float *output,
            const std::uint16_t *value,
            float weight,
            int head_dim)
        {
            switch (activeISALevel())
            {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__F16C__)
            case ISALevel::AVX512:
                accum_native16_avx512<Format>(
                    output, value, weight, head_dim);
                return;
#endif
#if defined(__AVX2__) && defined(__F16C__)
            case ISALevel::AVX2:
                accum_native16_avx2<Format>(
                    output, value, weight, head_dim);
                return;
#endif
            default:
                accum_native16_scalar<Format>(
                    output, value, weight, head_dim);
            }
        }

        /** @brief Runtime-ISA dispatch for four native 16-bit V accumulations. */
        template <Native16BitKVFormat Format>
        static void accum_native16_4row(
            float *output,
            const std::uint16_t *value0,
            float weight0,
            const std::uint16_t *value1,
            float weight1,
            const std::uint16_t *value2,
            float weight2,
            const std::uint16_t *value3,
            float weight3,
            int head_dim)
        {
            switch (activeISALevel())
            {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__F16C__)
            case ISALevel::AVX512:
                accum_native16_4row_avx512<Format>(
                    output,
                    value0, weight0,
                    value1, weight1,
                    value2, weight2,
                    value3, weight3,
                    head_dim);
                return;
#endif
#if defined(__AVX2__) && defined(__F16C__)
            case ISALevel::AVX2:
                accum_native16_4row_avx2<Format>(
                    output,
                    value0, weight0,
                    value1, weight1,
                    value2, weight2,
                    value3, weight3,
                    head_dim);
                return;
#endif
            default:
                accum_native16_4row_scalar<Format>(
                    output,
                    value0, weight0,
                    value1, weight1,
                    value2, weight2,
                    value3, weight3,
                    head_dim);
            }
        }

#if defined(__AVX2__)
        /**
         * @brief AVX2 4-row dot product: Q·K[0..3] with 4 FMA accumulators.
         */
        static void dot_fp32_avx2_4row(
            const float *q,
            const float *k0, const float *k1, const float *k2, const float *k3,
            int head_dim,
            float &s0, float &s1, float &s2, float &s3)
        {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            int d = 0;
            for (; d + 7 < head_dim; d += 8)
            {
                __m256 vq = _mm256_loadu_ps(q + d);
                acc0 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(k0 + d), acc0);
                acc1 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(k1 + d), acc1);
                acc2 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(k2 + d), acc2);
                acc3 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(k3 + d), acc3);
            }

            s0 = avx2::hsum_ps(acc0);
            s1 = avx2::hsum_ps(acc1);
            s2 = avx2::hsum_ps(acc2);
            s3 = avx2::hsum_ps(acc3);

            for (; d < head_dim; ++d)
            {
                float qd = q[d];
                s0 += qd * k0[d];
                s1 += qd * k1[d];
                s2 += qd * k2[d];
                s3 += qd * k3[d];
            }
        }
#endif

        static inline void dot_fp32_4row_scalar(
            const float *q,
            const float *k0, const float *k1, const float *k2, const float *k3,
            int head_dim,
            float &s0, float &s1, float &s2, float &s3)
        {
            s0 = 0.0f;
            s1 = 0.0f;
            s2 = 0.0f;
            s3 = 0.0f;
            for (int d = 0; d < head_dim; ++d)
            {
                float qd = q[d];
                s0 += qd * k0[d];
                s1 += qd * k1[d];
                s2 += qd * k2[d];
                s3 += qd * k3[d];
            }
        }

        /** @brief Dispatch: 4-row FP32 dot product. */
        static void dot_fp32_4row(
            const float *q,
            const float *k0, const float *k1, const float *k2, const float *k3,
            int head_dim,
            float &s0, float &s1, float &s2, float &s3)
        {
            switch (activeISALevel())
            {
#if defined(__AVX512F__)
            case ISALevel::AVX512:
                dot_fp32_avx512_4row(q, k0, k1, k2, k3, head_dim, s0, s1, s2, s3);
                break;
#endif
            case ISALevel::AVX2:
                dot_fp32_avx2_4row(q, k0, k1, k2, k3, head_dim, s0, s1, s2, s3);
                break;
            default:
                dot_fp32_4row_scalar(q, k0, k1, k2, k3, head_dim, s0, s1, s2, s3);
                break;
            }
        }

        // ---- Batched exp(score - max) for 4 values: named implementations ----

        static inline void batch_exp_4_scalar(
            float s0, float s1, float s2, float s3, float max_val,
            float &p0, float &p1, float &p2, float &p3)
        {
            p0 = std::exp(s0 - max_val);
            p1 = std::exp(s1 - max_val);
            p2 = std::exp(s2 - max_val);
            p3 = std::exp(s3 - max_val);
        }

#if defined(__AVX2__)
        static inline void batch_exp_4_avx2(
            float s0, float s1, float s2, float s3, float max_val,
            float &p0, float &p1, float &p2, float &p3)
        {
            alignas(32) float scores8[8] = {s0, s1, s2, s3, 0.0f, 0.0f, 0.0f, 0.0f};
            __m256 exp_in = _mm256_sub_ps(
                _mm256_load_ps(scores8), _mm256_set1_ps(max_val));
            __m256 exp_out = fast_exp_avx2(exp_in);
            alignas(32) float pp[8];
            _mm256_store_ps(pp, exp_out);
            p0 = pp[0];
            p1 = pp[1];
            p2 = pp[2];
            p3 = pp[3];
        }
#endif

#if defined(__AVX512F__)
        static inline void batch_exp_4_avx512(
            float s0, float s1, float s2, float s3, float max_val,
            float &p0, float &p1, float &p2, float &p3)
        {
            const __m128 scores4 = _mm_set_ps(s3, s2, s1, s0);
            const __m128 nm4 = _mm_set1_ps(max_val);
            const __m512 exp_in = _mm512_castps128_ps512(_mm_sub_ps(scores4, nm4));
            const __m512 exp_out = fast_exp_avx512(exp_in);
            const __m128 probs = _mm512_castps512_ps128(exp_out);
            alignas(16) float pp[4];
            _mm_store_ps(pp, probs);
            p0 = pp[0];
            p1 = pp[1];
            p2 = pp[2];
            p3 = pp[3];
        }
#endif

        /** @brief Dispatch: compute 4 × exp(score - max). */
        static void batch_exp_4(
            float s0, float s1, float s2, float s3, float max_val,
            float &p0, float &p1, float &p2, float &p3)
        {
            switch (activeISALevel())
            {
#if defined(__AVX512F__)
            case ISALevel::AVX512:
                batch_exp_4_avx512(s0, s1, s2, s3, max_val, p0, p1, p2, p3);
                break;
#endif
            case ISALevel::AVX2:
                batch_exp_4_avx2(s0, s1, s2, s3, max_val, p0, p1, p2, p3);
                break;
            default:
                batch_exp_4_scalar(s0, s1, s2, s3, max_val, p0, p1, p2, p3);
                break;
            }
        }

        /**
         * @brief Compute optimal thread count for attention based on work size.
         *
         * Balances parallelism against OpenMP fork/join overhead (~30-50µs)
         * by estimating total FLOPs and ensuring each thread has enough work
         * to amortise the cost of entering a parallel region.
         *
         * At ~25 GFLOP/s effective FP32 throughput per core, 50µs = 1.25M FLOPs.
         * We use 2M as the threshold to ensure clear amortisation.
         */
        static int computeOptimalAttentionThreads(
            int n_heads, int seq_len, int kv_len, int head_dim, bool causal)
        {
            const int max_threads = omp_get_max_threads();

            // Work per head: QK dots + V accumulation + exp overhead
            // Causal masking halves the average KV positions per query.
            const int64_t avg_kv = causal ? (static_cast<int64_t>(kv_len) + 1) / 2
                                          : static_cast<int64_t>(kv_len);
            const int64_t flops_per_head =
                static_cast<int64_t>(seq_len) * avg_kv * static_cast<int64_t>(head_dim) * 4 + static_cast<int64_t>(seq_len) * avg_kv * 10; // exp + softmax overhead
            const int64_t total_flops = flops_per_head * n_heads;

            // Minimum FLOPs per thread to justify OMP overhead
            constexpr int64_t MIN_FLOPS_PER_THREAD = 2000000; // ~80µs at 25 GFLOP/s

            if (total_flops <= MIN_FLOPS_PER_THREAD)
            {
                return 1;
            }

            const int64_t independent_output_rows =
                static_cast<int64_t>(n_heads) *
                static_cast<int64_t>(std::max(1, seq_len));
            int desired = static_cast<int>(total_flops / MIN_FLOPS_PER_THREAD);
            return std::max(
                1,
                std::min(
                    {desired,
                     static_cast<int>(std::min<int64_t>(
                         independent_output_rows,
                         std::numeric_limits<int>::max())),
                     max_threads}));
        }

        // -----------------------------------------------------------------
        // Vector accumulation and scaling helpers
        // -----------------------------------------------------------------

        /**
         * @brief Accumulate `weight * v[d]` into `out[d]` for each dimension.
         *
         * This is the inner loop of the V accumulation step:
         * `out[d] += weight * V[k, d]` for one KV position k.
         *
         * Uses AVX-512 FMA when available (16 floats/cycle).
         *
         * @param out       Running output accumulator, length `head_dim`.
         * @param v         Value row for this KV position, length `head_dim`.
         * @param weight    Softmax probability for this KV position.
         * @param head_dim  Number of dimensions per head.
         * @param use_avx512 True if AVX-512 is available at runtime.
         */
        static void accum_weighted_v_scalar(float *out, const float *v, float weight, int head_dim)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                out[d] += weight * v[d];
            }
        }

#if defined(__AVX2__)
        static void accum_weighted_v_avx2(float *out, const float *v, float weight, int head_dim)
        {
            __m256 w = _mm256_set1_ps(weight);
            int d = 0;
            for (; d + 7 < head_dim; d += 8)
            {
                __m256 o = _mm256_loadu_ps(out + d);
                __m256 vv = _mm256_loadu_ps(v + d);
                o = _mm256_fmadd_ps(vv, w, o);
                _mm256_storeu_ps(out + d, o);
            }
            for (; d < head_dim; ++d)
                out[d] += weight * v[d];
        }
#endif

#if defined(__AVX512F__)
        static void accum_weighted_v_avx512(float *out, const float *v, float weight, int head_dim)
        {
            __m512 w = _mm512_set1_ps(weight);
            int d = 0;
            for (; d + 15 < head_dim; d += 16)
            {
                __m512 o = _mm512_loadu_ps(out + d);
                __m512 vv = _mm512_loadu_ps(v + d);
                o = _mm512_fmadd_ps(vv, w, o);
                _mm512_storeu_ps(out + d, o);
            }
            for (; d < head_dim; ++d)
            {
                out[d] += weight * v[d];
            }
        }
#endif

        static void accum_weighted_v(float *out, const float *v, float weight, int head_dim, bool use_avx512)
        {
            (void)use_avx512;
            switch (activeISALevel())
            {
#if defined(__AVX512F__)
            case ISALevel::AVX512:
                accum_weighted_v_avx512(out, v, weight, head_dim);
                break;
#endif
            case ISALevel::AVX2:
                accum_weighted_v_avx2(out, v, weight, head_dim);
                break;
            default:
                accum_weighted_v_scalar(out, v, weight, head_dim);
                break;
            }
        }

        /**
         * @brief Batched 4-row FP32 V accumulation with single output load/store.
         *
         * Accumulates 4 weighted V vectors into out in one pass, saving ~60%
         * of L1 traffic vs calling accum_weighted_v four times.
         */
        // ---- FP32 4-row V-accumulation: named implementations ----

        static inline void accum_weighted_v_4row_scalar(
            float *__restrict out,
            const float *v0, float w0,
            const float *v1, float w1,
            const float *v2, float w2,
            const float *v3, float w3,
            int head_dim)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                out[d] += w0 * v0[d] + w1 * v1[d] + w2 * v2[d] + w3 * v3[d];
            }
        }

#if defined(__AVX2__)
        static inline void accum_weighted_v_4row_avx2(
            float *__restrict out,
            const float *v0, float w0,
            const float *v1, float w1,
            const float *v2, float w2,
            const float *v3, float w3,
            int head_dim)
        {
            const __m256 vw0 = _mm256_set1_ps(w0);
            const __m256 vw1 = _mm256_set1_ps(w1);
            const __m256 vw2 = _mm256_set1_ps(w2);
            const __m256 vw3 = _mm256_set1_ps(w3);
            int d = 0;
            for (; d + 15 < head_dim; d += 16)
            {
                __m256 oA = _mm256_loadu_ps(out + d);
                __m256 oB = _mm256_loadu_ps(out + d + 8);
                oA = _mm256_fmadd_ps(_mm256_loadu_ps(v0 + d), vw0, oA);
                oB = _mm256_fmadd_ps(_mm256_loadu_ps(v0 + d + 8), vw0, oB);
                oA = _mm256_fmadd_ps(_mm256_loadu_ps(v1 + d), vw1, oA);
                oB = _mm256_fmadd_ps(_mm256_loadu_ps(v1 + d + 8), vw1, oB);
                oA = _mm256_fmadd_ps(_mm256_loadu_ps(v2 + d), vw2, oA);
                oB = _mm256_fmadd_ps(_mm256_loadu_ps(v2 + d + 8), vw2, oB);
                oA = _mm256_fmadd_ps(_mm256_loadu_ps(v3 + d), vw3, oA);
                oB = _mm256_fmadd_ps(_mm256_loadu_ps(v3 + d + 8), vw3, oB);
                _mm256_storeu_ps(out + d, oA);
                _mm256_storeu_ps(out + d + 8, oB);
            }
            for (; d + 7 < head_dim; d += 8)
            {
                __m256 o = _mm256_loadu_ps(out + d);
                o = _mm256_fmadd_ps(_mm256_loadu_ps(v0 + d), vw0, o);
                o = _mm256_fmadd_ps(_mm256_loadu_ps(v1 + d), vw1, o);
                o = _mm256_fmadd_ps(_mm256_loadu_ps(v2 + d), vw2, o);
                o = _mm256_fmadd_ps(_mm256_loadu_ps(v3 + d), vw3, o);
                _mm256_storeu_ps(out + d, o);
            }
            for (; d < head_dim; ++d)
                out[d] += w0 * v0[d] + w1 * v1[d] + w2 * v2[d] + w3 * v3[d];
        }
#endif

#if defined(__AVX512F__)
        static inline void accum_weighted_v_4row_avx512(
            float *__restrict out,
            const float *v0, float w0,
            const float *v1, float w1,
            const float *v2, float w2,
            const float *v3, float w3,
            int head_dim)
        {
            const __m512 vw0 = _mm512_set1_ps(w0);
            const __m512 vw1 = _mm512_set1_ps(w1);
            const __m512 vw2 = _mm512_set1_ps(w2);
            const __m512 vw3 = _mm512_set1_ps(w3);
            int d = 0;
            for (; d + 31 < head_dim; d += 32)
            {
                __m512 oA = _mm512_loadu_ps(out + d);
                __m512 oB = _mm512_loadu_ps(out + d + 16);
                oA = _mm512_fmadd_ps(_mm512_loadu_ps(v0 + d), vw0, oA);
                oB = _mm512_fmadd_ps(_mm512_loadu_ps(v0 + d + 16), vw0, oB);
                oA = _mm512_fmadd_ps(_mm512_loadu_ps(v1 + d), vw1, oA);
                oB = _mm512_fmadd_ps(_mm512_loadu_ps(v1 + d + 16), vw1, oB);
                oA = _mm512_fmadd_ps(_mm512_loadu_ps(v2 + d), vw2, oA);
                oB = _mm512_fmadd_ps(_mm512_loadu_ps(v2 + d + 16), vw2, oB);
                oA = _mm512_fmadd_ps(_mm512_loadu_ps(v3 + d), vw3, oA);
                oB = _mm512_fmadd_ps(_mm512_loadu_ps(v3 + d + 16), vw3, oB);
                _mm512_storeu_ps(out + d, oA);
                _mm512_storeu_ps(out + d + 16, oB);
            }
            for (; d + 15 < head_dim; d += 16)
            {
                __m512 o = _mm512_loadu_ps(out + d);
                o = _mm512_fmadd_ps(_mm512_loadu_ps(v0 + d), vw0, o);
                o = _mm512_fmadd_ps(_mm512_loadu_ps(v1 + d), vw1, o);
                o = _mm512_fmadd_ps(_mm512_loadu_ps(v2 + d), vw2, o);
                o = _mm512_fmadd_ps(_mm512_loadu_ps(v3 + d), vw3, o);
                _mm512_storeu_ps(out + d, o);
            }
            for (; d < head_dim; ++d)
            {
                out[d] += w0 * v0[d] + w1 * v1[d] + w2 * v2[d] + w3 * v3[d];
            }
        }
#endif

        static void accum_weighted_v_4row(
            float *__restrict out,
            const float *v0, float w0,
            const float *v1, float w1,
            const float *v2, float w2,
            const float *v3, float w3,
            int head_dim)
        {
            switch (activeISALevel())
            {
#if defined(__AVX512F__)
            case ISALevel::AVX512:
                accum_weighted_v_4row_avx512(out, v0, w0, v1, w1, v2, w2, v3, w3, head_dim);
                break;
#endif
            case ISALevel::AVX2:
                accum_weighted_v_4row_avx2(out, v0, w0, v1, w1, v2, w2, v3, w3, head_dim);
                break;
            default:
                accum_weighted_v_4row_scalar(out, v0, w0, v1, w1, v2, w2, v3, w3, head_dim);
                break;
            }
        }

        /**
         * @brief Multiply every element of `out` by a scalar `alpha`.
         *
         * Used during online softmax to rescale the running accumulator when a
         * new tile's maximum exceeds the previous running maximum:
         *   `out[d] *= exp(old_max − new_max)`
         *
         * @param out        Vector to scale in-place, length `head_dim`.
         * @param alpha      Scalar multiplier.
         * @param head_dim   Vector length.
         * @param use_avx512 True if AVX-512 is available at runtime.
         */
        static void scale_vec_scalar(float *out, float alpha, int head_dim)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                out[d] *= alpha;
            }
        }

#if defined(__AVX2__)
        static void scale_vec_avx2(float *out, float alpha, int head_dim)
        {
            __m256 a = _mm256_set1_ps(alpha);
            int d = 0;
            for (; d + 7 < head_dim; d += 8)
            {
                __m256 o = _mm256_loadu_ps(out + d);
                o = _mm256_mul_ps(o, a);
                _mm256_storeu_ps(out + d, o);
            }
            for (; d < head_dim; ++d)
                out[d] *= alpha;
        }
#endif

#if defined(__AVX512F__)
        static void scale_vec_avx512(float *out, float alpha, int head_dim)
        {
            __m512 a = _mm512_set1_ps(alpha);
            int d = 0;
            for (; d + 15 < head_dim; d += 16)
            {
                __m512 o = _mm512_loadu_ps(out + d);
                o = _mm512_mul_ps(o, a);
                _mm512_storeu_ps(out + d, o);
            }
            for (; d < head_dim; ++d)
            {
                out[d] *= alpha;
            }
        }
#endif

        static void scale_vec(float *out, float alpha, int head_dim, bool use_avx512)
        {
            (void)use_avx512;
            switch (activeISALevel())
            {
#if defined(__AVX512F__)
            case ISALevel::AVX512:
                scale_vec_avx512(out, alpha, head_dim);
                break;
#endif
            case ISALevel::AVX2:
                scale_vec_avx2(out, alpha, head_dim);
                break;
            default:
                scale_vec_scalar(out, alpha, head_dim);
                break;
            }
        }

        /**
         * @brief Divide every element of `out` by `denom`.  No-op if denom ≤ 0.
         *
         * Called once at the end of the online softmax for each (head, query)
         * pair to normalise the accumulated output by the softmax denominator.
         *
         * @param out        Vector to normalise in-place.
         * @param denom      Softmax denominator (sum of exp(scores − max)).
         * @param head_dim   Vector length.
         * @param use_avx512 True if AVX-512 is available at runtime.
         */
        static void div_vec(float *out, float denom, int head_dim, bool use_avx512)
        {
            // Guard against division by zero or negative denominators which
            // can occur when all scores were -infinity (completely masked).
            if (denom <= 0.0f)
            {
                return;
            }
            scale_vec(out, 1.0f / denom, head_dim, use_avx512);
        }

        // -----------------------------------------------------------------
        // I16/I12 query quantisation for native Q16 K/V attention
        // -----------------------------------------------------------------
        // These routines quantise FP32 rows into 16-bit integers using an
        // absmax scheme with a configurable `qmax` (typically 2047, giving
        // ~12 effective bits).  The resulting integers can be fed into the
        // native Q16 cache rows below for high-throughput QK scoring. K/V are
        // never repacked: decode, grouped verification, and prefill all read
        // the same cache bytes and use the same fixed arithmetic order.
        // -----------------------------------------------------------------

        /**
         * @brief Quantise a single FP32 row to int16 using absmax scaling.
         *
         * 1. Find the absolute maximum of the row.
         * 2. Compute `scale = max_abs / qmax`.
         * 3. Quantise each element: `dst[i] = round(src[i] / scale)`.
         *
         * @param src   Source FP32 vector, length `n`.
         * @param dst   Destination int16 vector, length `n`.
         * @param n     Number of elements.
         * @param qmax  Maximum quantised value (e.g. 2047 for "I12").
         * @return The absmax scale factor.  Multiply `(int_dot * scale_q * scale_k)`
         *         to recover the approximate FP32 dot product.
         */
        static float quantize_row_i16_i12_scalar(const float *src, int16_t *dst, int n, int qmax)
        {
            float max_abs = 0.0f;
            for (int i = 0; i < n; ++i)
                max_abs = std::max(max_abs, std::abs(src[i]));

            if (max_abs <= 1e-12f)
            {
                std::memset(dst, 0, static_cast<size_t>(n) * sizeof(int16_t));
                return 0.0f;
            }

            const float scale = max_abs / static_cast<float>(qmax);
            const float inv_scale = 1.0f / scale;
            for (int i = 0; i < n; ++i)
            {
                const int q = static_cast<int>(std::lrint(src[i] * inv_scale));
                dst[i] = static_cast<int16_t>(std::max(-qmax, std::min(q, qmax)));
            }
            return scale;
        }

#if defined(__AVX2__)
        static float quantize_row_i16_i12_avx2(const float *src, int16_t *dst, int n, int qmax)
        {
            // AVX2 Pass 1: absmax using 8-wide vectors
            const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            __m256 vmax = _mm256_setzero_ps();
            int i = 0;
            for (; i + 7 < n; i += 8)
            {
                const __m256 v = _mm256_loadu_ps(src + i);
                vmax = _mm256_max_ps(vmax, _mm256_and_ps(v, abs_mask));
            }
            // Horizontal max: 8 → 4 → 2 → 1
            __m128 hi4 = _mm256_extractf128_ps(vmax, 1);
            __m128 lo4 = _mm256_castps256_ps128(vmax);
            lo4 = _mm_max_ps(lo4, hi4);
            __m128 shuf = _mm_movehdup_ps(lo4);
            lo4 = _mm_max_ps(lo4, shuf);
            shuf = _mm_movehl_ps(shuf, lo4);
            lo4 = _mm_max_ss(lo4, shuf);
            float max_abs = _mm_cvtss_f32(lo4);
            for (; i < n; ++i)
                max_abs = std::max(max_abs, std::abs(src[i]));

            if (max_abs <= 1e-12f)
            {
                std::memset(dst, 0, static_cast<size_t>(n) * sizeof(int16_t));
                return 0.0f;
            }

            // AVX2 Pass 2: quantize 8 at a time, pack i32→i16
            const float scale = max_abs / static_cast<float>(qmax);
            const float inv_scale = 1.0f / scale;
            const __m256 v_inv = _mm256_set1_ps(inv_scale);
            const __m256i v_hi = _mm256_set1_epi32(qmax);
            const __m256i v_lo = _mm256_set1_epi32(-qmax);
            i = 0;
            for (; i + 7 < n; i += 8)
            {
                const __m256 v = _mm256_loadu_ps(src + i);
                __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(v, v_inv));
                i32 = _mm256_max_epi32(v_lo, _mm256_min_epi32(v_hi, i32));
                const __m128i packed = _mm_packs_epi32(
                    _mm256_castsi256_si128(i32),
                    _mm256_extracti128_si256(i32, 1));
                _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i),
                                 packed);
            }
            for (; i < n; ++i)
            {
                const int q = static_cast<int>(std::lrint(src[i] * inv_scale));
                dst[i] = static_cast<int16_t>(std::max(-qmax, std::min(q, qmax)));
            }
            return scale;
        }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
        static float quantize_row_i16_i12_avx512(const float *src, int16_t *dst, int n, int qmax)
        {
            // Pass 1: vectorized absmax
            const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
            __m512 vmax = _mm512_setzero_ps();
            int i = 0;
            for (; i + 15 < n; i += 16)
            {
                const __m512 v = _mm512_loadu_ps(src + i);
                const __m512 va = _mm512_castsi512_ps(
                    _mm512_and_epi32(_mm512_castps_si512(v), abs_mask));
                vmax = _mm512_max_ps(vmax, va);
            }
            float max_abs = _mm512_reduce_max_ps(vmax);
            for (; i < n; ++i)
                max_abs = std::max(max_abs, std::abs(src[i]));

            if (max_abs <= 1e-12f)
            {
                std::memset(dst, 0, static_cast<size_t>(n) * sizeof(int16_t));
                return 0.0f;
            }

            // Pass 2: vectorized quantize (cvtps_epi32 uses round-to-nearest-even)
            const float scale = max_abs / static_cast<float>(qmax);
            const float inv_scale = 1.0f / scale;
            const __m512 v_inv = _mm512_set1_ps(inv_scale);
            const __m512i v_hi = _mm512_set1_epi32(qmax);
            const __m512i v_lo = _mm512_set1_epi32(-qmax);
            i = 0;
            for (; i + 15 < n; i += 16)
            {
                const __m512 v = _mm512_loadu_ps(src + i);
                __m512i i32 = _mm512_cvtps_epi32(_mm512_mul_ps(v, v_inv));
                i32 = _mm512_max_epi32(v_lo, _mm512_min_epi32(v_hi, i32));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i),
                                    _mm512_cvtsepi32_epi16(i32));
            }
            for (; i < n; ++i)
            {
                const int q = static_cast<int>(std::lrint(src[i] * inv_scale));
                dst[i] = static_cast<int16_t>(std::max(-qmax, std::min(q, qmax)));
            }
            return scale;
        }
#endif

        static float quantize_row_i16_i12(const float *src, int16_t *dst, int n, int qmax)
        {
            switch (activeISALevel())
            {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            case ISALevel::AVX512:
                return quantize_row_i16_i12_avx512(src, dst, n, qmax);
#endif
            case ISALevel::AVX2:
                return quantize_row_i16_i12_avx2(src, dst, n, qmax);
            default:
                return quantize_row_i16_i12_scalar(src, dst, n, qmax);
            }
        }

        // -----------------------------------------------------------------
        // Integer dot-product routines (I16 × I16 → I32)
        // -----------------------------------------------------------------
        // These compute `Σ a[i]*b[i]` entirely in integer arithmetic.
        // The VNNI variants use the AVX-512 VPDPWSSD instruction which
        // performs 32 × (int16 × int16) → int32 accumulate per cycle.
        // -----------------------------------------------------------------

        /**
         * @brief Scalar int16 dot product (fallback, no SIMD).
         */
        static int32_t dot_i16_i16_i32_scalar(const int16_t *a, const int16_t *b, int n)
        {
            int32_t sum = 0;
            for (int i = 0; i < n; ++i)
            {
                sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
            }
            return sum;
        }

        /**
         * @brief VNNI-accelerated int16 dot product.
         *
         * Uses `_mm512_dpwssd_epi32` which computes 32 int16 multiply-adds
         * per instruction.  Tail elements are handled by scalar code.
         */
        static int32_t dot_i16_i16_i32_vnni(const int16_t *a, const int16_t *b, int n)
        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            __m512i acc = _mm512_setzero_si512();
            int i = 0;
            for (; i + 31 < n; i += 32)
            {
                const __m512i va = _mm512_loadu_si512(reinterpret_cast<const void *>(a + i));
                const __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void *>(b + i));
                acc = _mm512_dpwssd_epi32(acc, va, vb);
            }

            alignas(64) int32_t lanes[16];
            _mm512_store_si512(reinterpret_cast<void *>(lanes), acc);
            int32_t sum = 0;
            for (int lane = 0; lane < 16; ++lane)
            {
                sum += lanes[lane];
            }

            for (; i < n; ++i)
            {
                sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
            }
            return sum;
#else
            return dot_i16_i16_i32_scalar(a, b, n);
#endif
        }

        /** @brief Runtime dispatch: i16 dot product. */
        static int32_t dot_i16_i16_i32(const int16_t *a, const int16_t *b, int n)
        {
            switch (activeISALevel())
            {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            case ISALevel::AVX512:
                return dot_i16_i16_i32_vnni(a, b, n);
#endif
            case ISALevel::AVX2:
                return dot_i16_i16_i32_avx2(a, b, n);
            default:
                return dot_i16_i16_i32_scalar(a, b, n);
            }
        }

        /**
         * @brief AVX2 int16 dot product using madd (i16×i16→i32 pairs).
         *
         * Uses `_mm256_madd_epi16` which computes adjacent pairs:
         * acc[k] += a[2k]*b[2k] + a[2k+1]*b[2k+1] for 8 output i32s.
         */
        static int32_t dot_i16_i16_i32_avx2(const int16_t *a, const int16_t *b, int n)
        {
            __m256i acc = _mm256_setzero_si256();
            int i = 0;
            for (; i + 15 < n; i += 16)
            {
                const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
                const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
            }
            // Horizontal sum
            __m128i lo = _mm256_castsi256_si128(acc);
            __m128i hi = _mm256_extracti128_si256(acc, 1);
            lo = _mm_add_epi32(lo, hi);
            lo = _mm_hadd_epi32(lo, lo);
            lo = _mm_hadd_epi32(lo, lo);
            int32_t sum = _mm_extract_epi32(lo, 0);
            for (; i < n; ++i)
                sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
            return sum;
        }

        /**
         * @brief Dual-row VNNI dot product:  `out0 = q·k0` and `out1 = q·k1`.
         *
         * Computes two dot-products against two separate K rows in a single
         * pass over the Q vector, exploiting instruction-level parallelism.
         * The inner loop is unrolled 2× (64 elements / iteration) for ILP.
         */
        static void dot_i16_i16_i32_vnni_2row_scalar(
            const int16_t *q,
            const int16_t *k0,
            const int16_t *k1,
            int n,
            int32_t &out0,
            int32_t &out1)
        {
            out0 = dot_i16_i16_i32_scalar(q, k0, n);
            out1 = dot_i16_i16_i32_scalar(q, k1, n);
        }

#if defined(__AVX2__)
        static void dot_i16_i16_i32_vnni_2row_avx2(
            const int16_t *q,
            const int16_t *k0,
            const int16_t *k1,
            int n,
            int32_t &out0,
            int32_t &out1)
        {
            __m256i a0 = _mm256_setzero_si256();
            __m256i a1 = _mm256_setzero_si256();
            int i = 0;
            for (; i + 15 < n; i += 16)
            {
                const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + i));
                const __m256i k0v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k0 + i));
                const __m256i k1v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k1 + i));
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(qv, k0v));
                a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(qv, k1v));
            }
            // Horizontal sums
            auto hsum = [](const __m256i &v) -> int32_t
            {
                __m128i lo = _mm256_castsi256_si128(v);
                __m128i hi = _mm256_extracti128_si256(v, 1);
                lo = _mm_add_epi32(lo, hi);
                lo = _mm_hadd_epi32(lo, lo);
                lo = _mm_hadd_epi32(lo, lo);
                return _mm_extract_epi32(lo, 0);
            };
            int32_t sum0 = hsum(a0);
            int32_t sum1 = hsum(a1);
            for (; i < n; ++i)
            {
                const int32_t qv = static_cast<int32_t>(q[i]);
                sum0 += qv * static_cast<int32_t>(k0[i]);
                sum1 += qv * static_cast<int32_t>(k1[i]);
            }
            out0 = sum0;
            out1 = sum1;
        }
#endif

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
        static void dot_i16_i16_i32_vnni_2row_avx512(
            const int16_t *q,
            const int16_t *k0,
            const int16_t *k1,
            int n,
            int32_t &out0,
            int32_t &out1)
        {
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();

            int i = 0;
            for (; i + 63 < n; i += 64)
            {
                const __m512i q0 = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                const __m512i q1 = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i + 32));

                const __m512i k00 = _mm512_loadu_si512(reinterpret_cast<const void *>(k0 + i));
                const __m512i k01 = _mm512_loadu_si512(reinterpret_cast<const void *>(k0 + i + 32));
                const __m512i k10 = _mm512_loadu_si512(reinterpret_cast<const void *>(k1 + i));
                const __m512i k11 = _mm512_loadu_si512(reinterpret_cast<const void *>(k1 + i + 32));

                acc0 = _mm512_dpwssd_epi32(acc0, q0, k00);
                acc0 = _mm512_dpwssd_epi32(acc0, q1, k01);
                acc1 = _mm512_dpwssd_epi32(acc1, q0, k10);
                acc1 = _mm512_dpwssd_epi32(acc1, q1, k11);
            }

            for (; i + 31 < n; i += 32)
            {
                const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                const __m512i k0v = _mm512_loadu_si512(reinterpret_cast<const void *>(k0 + i));
                const __m512i k1v = _mm512_loadu_si512(reinterpret_cast<const void *>(k1 + i));
                acc0 = _mm512_dpwssd_epi32(acc0, qv, k0v);
                acc1 = _mm512_dpwssd_epi32(acc1, qv, k1v);
            }

            alignas(64) int32_t lanes0[16];
            alignas(64) int32_t lanes1[16];
            _mm512_store_si512(reinterpret_cast<void *>(lanes0), acc0);
            _mm512_store_si512(reinterpret_cast<void *>(lanes1), acc1);

            int32_t sum0 = 0;
            int32_t sum1 = 0;
            for (int lane = 0; lane < 16; ++lane)
            {
                sum0 += lanes0[lane];
                sum1 += lanes1[lane];
            }

            for (; i < n; ++i)
            {
                const int32_t qv = static_cast<int32_t>(q[i]);
                sum0 += qv * static_cast<int32_t>(k0[i]);
                sum1 += qv * static_cast<int32_t>(k1[i]);
            }

            out0 = sum0;
            out1 = sum1;
        }
#endif

        static void dot_i16_i16_i32_vnni_2row(
            const int16_t *q,
            const int16_t *k0,
            const int16_t *k1,
            int n,
            int32_t &out0,
            int32_t &out1)
        {
            switch (activeISALevel())
            {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            case ISALevel::AVX512:
                dot_i16_i16_i32_vnni_2row_avx512(q, k0, k1, n, out0, out1);
                break;
#endif
            case ISALevel::AVX2:
                dot_i16_i16_i32_vnni_2row_avx2(q, k0, k1, n, out0, out1);
                break;
            default:
                dot_i16_i16_i32_vnni_2row_scalar(q, k0, k1, n, out0, out1);
                break;
            }
        }

    public:
        // Named implementations are public for direct parity testing.
        // Use the dispatch functions (dot_i16_i16_i32_vnni_*) for production code.

        /**
         * @brief 4-row VNNI dot product against separate (non-packed) K rows.
         *
         * Computes four independent int16 dot products in a single pass over
         * the Q vector, exploiting ILP across four accumulator chains.
         * Each K row remains at its native cache address. The Q16_1 decode and
         * grouped-verifier paths use this layout directly, avoiding a
         * transient repack while retaining four independent accumulators.
         *
         * @param q   Quantised query vector, length `n`.
         * @param k0  First K row (int16), length `n`.
         * @param k1  Second K row (int16), length `n`.
         * @param k2  Third K row (int16), length `n`.
         * @param k3  Fourth K row (int16), length `n`.
         * @param n   Number of elements.
         * @param out0-out3  (out) Int32 dot products.
         */
        // ---- 4-row separate: named implementations ----

        static inline void dot_4row_separate_scalar(
            const int16_t *q, const int16_t *k0, const int16_t *k1,
            const int16_t *k2, const int16_t *k3, int n,
            int32_t &out0, int32_t &out1, int32_t &out2, int32_t &out3)
        {
            out0 = dot_i16_i16_i32_scalar(q, k0, n);
            out1 = dot_i16_i16_i32_scalar(q, k1, n);
            out2 = dot_i16_i16_i32_scalar(q, k2, n);
            out3 = dot_i16_i16_i32_scalar(q, k3, n);
        }

#if defined(__AVX2__)
        static inline void dot_4row_separate_avx2(
            const int16_t *q, const int16_t *k0, const int16_t *k1,
            const int16_t *k2, const int16_t *k3, int n,
            int32_t &out0, int32_t &out1, int32_t &out2, int32_t &out3)
        {
            __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
            __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
            int i = 0;
            for (; i + 15 < n; i += 16)
            {
                const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + i));
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(qv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k0 + i))));
                a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(qv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k1 + i))));
                a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(qv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k2 + i))));
                a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(qv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k3 + i))));
            }
            auto hsum = [](const __m256i &v) -> int32_t
            {
                __m128i lo = _mm256_castsi256_si128(v);
                __m128i hi = _mm256_extracti128_si256(v, 1);
                lo = _mm_add_epi32(lo, hi);
                lo = _mm_hadd_epi32(lo, lo);
                lo = _mm_hadd_epi32(lo, lo);
                return _mm_extract_epi32(lo, 0);
            };
            out0 = hsum(a0);
            out1 = hsum(a1);
            out2 = hsum(a2);
            out3 = hsum(a3);
            for (; i < n; ++i)
            {
                const int32_t qv = static_cast<int32_t>(q[i]);
                out0 += qv * static_cast<int32_t>(k0[i]);
                out1 += qv * static_cast<int32_t>(k1[i]);
                out2 += qv * static_cast<int32_t>(k2[i]);
                out3 += qv * static_cast<int32_t>(k3[i]);
            }
        }
#endif

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
        static inline void dot_4row_separate_avx512(
            const int16_t *q, const int16_t *k0, const int16_t *k1,
            const int16_t *k2, const int16_t *k3, int n,
            int32_t &out0, int32_t &out1, int32_t &out2, int32_t &out3)
        {
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();

            int i = 0;
            for (; i + 63 < n; i += 64)
            {
                const __m512i q0 = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                const __m512i q1 = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i + 32));
                acc0 = _mm512_dpwssd_epi32(acc0, q0, _mm512_loadu_si512(reinterpret_cast<const void *>(k0 + i)));
                acc0 = _mm512_dpwssd_epi32(acc0, q1, _mm512_loadu_si512(reinterpret_cast<const void *>(k0 + i + 32)));
                acc1 = _mm512_dpwssd_epi32(acc1, q0, _mm512_loadu_si512(reinterpret_cast<const void *>(k1 + i)));
                acc1 = _mm512_dpwssd_epi32(acc1, q1, _mm512_loadu_si512(reinterpret_cast<const void *>(k1 + i + 32)));
                acc2 = _mm512_dpwssd_epi32(acc2, q0, _mm512_loadu_si512(reinterpret_cast<const void *>(k2 + i)));
                acc2 = _mm512_dpwssd_epi32(acc2, q1, _mm512_loadu_si512(reinterpret_cast<const void *>(k2 + i + 32)));
                acc3 = _mm512_dpwssd_epi32(acc3, q0, _mm512_loadu_si512(reinterpret_cast<const void *>(k3 + i)));
                acc3 = _mm512_dpwssd_epi32(acc3, q1, _mm512_loadu_si512(reinterpret_cast<const void *>(k3 + i + 32)));
            }

            for (; i + 31 < n; i += 32)
            {
                const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                acc0 = _mm512_dpwssd_epi32(acc0, qv, _mm512_loadu_si512(reinterpret_cast<const void *>(k0 + i)));
                acc1 = _mm512_dpwssd_epi32(acc1, qv, _mm512_loadu_si512(reinterpret_cast<const void *>(k1 + i)));
                acc2 = _mm512_dpwssd_epi32(acc2, qv, _mm512_loadu_si512(reinterpret_cast<const void *>(k2 + i)));
                acc3 = _mm512_dpwssd_epi32(acc3, qv, _mm512_loadu_si512(reinterpret_cast<const void *>(k3 + i)));
            }

            out0 = _mm512_reduce_add_epi32(acc0);
            out1 = _mm512_reduce_add_epi32(acc1);
            out2 = _mm512_reduce_add_epi32(acc2);
            out3 = _mm512_reduce_add_epi32(acc3);

            for (; i < n; ++i)
            {
                const int32_t qv = static_cast<int32_t>(q[i]);
                out0 += qv * static_cast<int32_t>(k0[i]);
                out1 += qv * static_cast<int32_t>(k1[i]);
                out2 += qv * static_cast<int32_t>(k2[i]);
                out3 += qv * static_cast<int32_t>(k3[i]);
            }
        }
#endif

        /** @brief Dispatch: 4-row dot against separate K rows. */
        static void dot_i16_i16_i32_vnni_4row(
            const int16_t *q, const int16_t *k0, const int16_t *k1,
            const int16_t *k2, const int16_t *k3, int n,
            int32_t &out0, int32_t &out1, int32_t &out2, int32_t &out3)
        {
            switch (activeISALevel())
            {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            case ISALevel::AVX512:
                dot_4row_separate_avx512(q, k0, k1, k2, k3, n, out0, out1, out2, out3);
                break;
#endif
            case ISALevel::AVX2:
                dot_4row_separate_avx2(q, k0, k1, k2, k3, n, out0, out1, out2, out3);
                break;
            default:
                dot_4row_separate_scalar(q, k0, k1, k2, k3, n, out0, out1, out2, out3);
                break;
            }
        }

    private:
        /**
         * @brief Maximum head width accepted by the stack-resident CPU FA2 scheduler.
         *
         * Every currently supported Qwen attention geometry is at most 256
         * elements wide. Keeping the row summaries in worker-local aligned
         * storage avoids allocation and false sharing in both physical modes.
         * A wider model must extend this compile-time contract and its totality
         * tests deliberately rather than entering a hidden heap path.
         */
        static constexpr int kMaximumAttentionHeadDim = 256;

        /** Visible K/V interval for one query row inside one physical tile. */
        struct CPUFA2VisibleTile
        {
            int begin = 0; ///< First visible K/V row, inclusive.
            int end = 0;   ///< Last visible K/V row, exclusive.

            /** @return True when this tile contributes to the attention row. */
            [[nodiscard]] constexpr bool empty() const noexcept
            {
                return begin >= end;
            }
        };

        /**
         * @brief Merge one unnormalised online-softmax summary in fixed order.
         *
         * Query-sequence execution and K/V-context execution both call this
         * exact routine for the same canonical partitions in ascending K/V
         * order. Their only difference is which worker produces a partition.
         * Consequently thread scheduling cannot alter reduction order or the
         * resulting bytes.
         *
         * @param destination Accumulated FP32 numerator, initialized to zero.
         * @param merged_m Running maximum for all prior partitions.
         * @param merged_l Running denominator for all prior partitions.
         * @param partial Numerator produced by the next partition.
         * @param partial_m Maximum produced by the next partition.
         * @param partial_l Denominator produced by the next partition.
         * @param head_dim Number of valid FP32 elements in each numerator.
         */
        static void mergeAttentionSummary(
            float *destination,
            float &merged_m,
            float &merged_l,
            const float *partial,
            float partial_m,
            float partial_l,
            int head_dim)
        {
            if (partial_l == 0.0f)
                return;

            const float new_m = std::max(merged_m, partial_m);
            const float destination_scale = std::isfinite(merged_m)
                                                ? std::exp(merged_m - new_m)
                                                : 0.0f;
            const float partial_scale = std::exp(partial_m - new_m);
#if defined(__AVX512F__)
            const __m512 vd = _mm512_set1_ps(destination_scale);
            const __m512 vp = _mm512_set1_ps(partial_scale);
            int d = 0;
            for (; d + 15 < head_dim; d += 16)
            {
                _mm512_storeu_ps(
                    destination + d,
                    _mm512_fmadd_ps(
                        vp,
                        _mm512_loadu_ps(partial + d),
                        _mm512_mul_ps(
                            vd,
                            _mm512_loadu_ps(destination + d))));
            }
            for (; d < head_dim; ++d)
            {
                destination[d] = destination[d] * destination_scale +
                                 partial[d] * partial_scale;
            }
#elif defined(__AVX2__)
            const __m256 vd = _mm256_set1_ps(destination_scale);
            const __m256 vp = _mm256_set1_ps(partial_scale);
            int d = 0;
            for (; d + 7 < head_dim; d += 8)
            {
                _mm256_storeu_ps(
                    destination + d,
                    _mm256_fmadd_ps(
                        vp,
                        _mm256_loadu_ps(partial + d),
                        _mm256_mul_ps(
                            vd,
                            _mm256_loadu_ps(destination + d))));
            }
            for (; d < head_dim; ++d)
            {
                destination[d] = destination[d] * destination_scale +
                                 partial[d] * partial_scale;
            }
#else
            for (int d = 0; d < head_dim; ++d)
            {
                destination[d] = destination[d] * destination_scale +
                                 partial[d] * partial_scale;
            }
#endif
            merged_l = merged_l * destination_scale +
                       partial_l * partial_scale;
            merged_m = new_m;
        }

        /**
         * @brief Evaluate one canonical summary through cache-sized K/V chunks.
         *
         * Physical tiling and floating-point arithmetic are intentionally
         * separate concepts. `arithmetic_begin` identifies the immutable
         * canonical score-array origin. `visible` is resolved once for the whole
         * canonical summary, then both QK and P@V visit that interval in chunks
         * of `physical_kv_tile` rows. Every compiled tile is a multiple of four,
         * so the format callbacks retain the same four-row vector groups and the
         * same scalar tail for every physical tile choice.
         *
         * The maximum is reduced over the complete canonical score array before
         * the numerator is scaled, and `partial_m`/`partial_l` are published only
         * after the complete P@V phase. Thus changing cache geometry cannot alter
         * online-softmax boundaries, reduction order, or result bytes.
         *
         * @tparam Scratch Format-specific worker-local row state.
         * @tparam ScoreTile Callback that writes scores and extends `block_max`.
         * @tparam AccumulateTile Callback that consumes scores and accumulates V.
         * @param row Logical `(query, head)` output-row index.
         * @param scratch Prepared format-specific row state.
         * @param arithmetic_begin First row of the canonical score array.
         * @param visible Complete visible interval inside this canonical summary.
         * @param head_dim Number of elements in the output accumulator.
         * @param physical_kv_tile Rows streamed through cache per callback.
         * @param score_tile Format-specific QK callback.
         * @param accumulate_tile Format-specific P@V callback.
         * @param scores Canonical score array with capacity for 256 rows.
         * @param partial Unnormalised canonical output numerator.
         * @param partial_m Canonical maximum, initialized to negative infinity.
         * @param partial_l Canonical denominator, initialized to zero.
         * @param profiling_enabled Whether phase timing is active.
         * @param qk_duration_ns Thread-local QK profiler accumulator.
         * @param v_duration_ns Thread-local P@V profiler accumulator.
         */
        template <typename Scratch,
                  typename ScoreTile,
                  typename AccumulateTile>
        static void evaluateCanonicalAttentionSummary(
            int row,
            const Scratch &scratch,
            int arithmetic_begin,
            CPUFA2VisibleTile visible,
            int head_dim,
            int physical_kv_tile,
            ScoreTile &score_tile,
            AccumulateTile &accumulate_tile,
            float *scores,
            float *partial,
            float &partial_m,
            float &partial_l,
            bool profiling_enabled,
            std::uint64_t &qk_duration_ns,
            std::uint64_t &v_duration_ns)
        {
            if (visible.empty())
                return;

            float block_max = -std::numeric_limits<float>::infinity();
            const auto qk_start = profiling_enabled
                                      ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
            for (int physical_begin = visible.begin;
                 physical_begin < visible.end;
                 physical_begin += physical_kv_tile)
            {
                const CPUFA2VisibleTile physical_visible{
                    .begin = physical_begin,
                    .end = std::min(
                        physical_begin + physical_kv_tile,
                        visible.end),
                };
                score_tile(
                    row,
                    scratch,
                    arithmetic_begin,
                    physical_visible,
                    scores,
                    block_max);
            }
            if (profiling_enabled)
            {
                qk_duration_ns += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - qk_start)
                        .count());
            }

            const float new_m = std::max(partial_m, block_max);
            const float alpha = std::isfinite(partial_m)
                                    ? std::exp(partial_m - new_m)
                                    : 0.0f;
            scale_vec(
                partial,
                alpha,
                head_dim,
                cpu_supports_avx512());
            float new_l = partial_l * alpha;

            const auto v_start = profiling_enabled
                                     ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
            for (int physical_begin = visible.begin;
                 physical_begin < visible.end;
                 physical_begin += physical_kv_tile)
            {
                const CPUFA2VisibleTile physical_visible{
                    .begin = physical_begin,
                    .end = std::min(
                        physical_begin + physical_kv_tile,
                        visible.end),
                };
                accumulate_tile(
                    row,
                    scratch,
                    arithmetic_begin,
                    physical_visible,
                    scores,
                    new_m,
                    partial,
                    new_l);
            }
            if (profiling_enabled)
            {
                v_duration_ns += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - v_start)
                        .count());
            }

            partial_m = new_m;
            partial_l = new_l;
        }

        /**
         * @brief Produce canonical K/V summaries in bounded parallel waves.
         *
         * A long context can contain thousands of canonical 256-row summaries,
         * but only enough producer slots to occupy the physical worker team are
         * useful at once. Slot zero of every output row retains the merged
         * numerator/max/denominator. Slots `[1, P]` hold one wave of at most P
         * independently produced summaries. After the producer barrier, a
         * row-parallel reducer merges those slots in ascending K/V order before
         * the next wave overwrites them.
         *
         * This design bounds persistent workspace by physical concurrency while
         * preserving exactly the same canonical arithmetic sequence used by
         * query-owned execution. It performs no allocation and introduces no
         * dependence on OpenMP scheduling order.
         */
        template <typename Scratch,
                  typename PrepareRow,
                  typename VisibleTile,
                  typename ScoreTile,
                  typename AccumulateTile,
                  typename FinalizeRow>
        bool executeContextParallelAttentionWaves(
            int query_rows,
            int n_heads,
            int kv_len,
            int head_dim,
            int kv_tile,
            const cpu::fa2_policy::CPUFA2ParallelPlan &plan,
            PrepareRow &prepare_row,
            VisibleTile &visible_tile,
            ScoreTile &score_tile,
            AccumulateTile &accumulate_tile,
            FinalizeRow &finalize_row)
        {
            const int output_rows = query_rows * n_heads;
            const int producer_slots = plan.context_partitions;
            const int slots_per_row = producer_slots + 1;
            const int canonical_summaries = plan.arithmetic_partitions;
            const int padded_head_dim = (head_dim + 15) & ~15;
            const std::size_t workspace_slots =
                static_cast<std::size_t>(output_rows) * slots_per_row;
            if (!hasContextSummaryCapacity(
                    workspace_slots,
                    static_cast<std::size_t>(padded_head_dim)))
            {
                return false;
            }

            const bool profiling_enabled = KernelProfiler::isEnabled();
            std::uint64_t qk_duration_ns = 0;
            std::uint64_t v_duration_ns = 0;
            const auto slot_index = [slots_per_row](int row, int slot)
            {
                return static_cast<std::size_t>(row) * slots_per_row + slot;
            };

            auto work = [&]()
            {
                Scratch scratch{};
                alignas(64) float scores[detail::kMaxKVTile];

#pragma omp for schedule(static)
                for (int row = 0; row < output_rows; ++row)
                {
                    const std::size_t merged_slot = slot_index(row, 0);
                    std::fill(
                        partial_output_ + merged_slot * padded_head_dim,
                        partial_output_ + merged_slot * padded_head_dim +
                            head_dim,
                        0.0f);
                    partial_m_[merged_slot] =
                        -std::numeric_limits<float>::infinity();
                    partial_l_[merged_slot] = 0.0f;
                }

                for (int wave_begin = 0;
                     wave_begin < canonical_summaries;
                     wave_begin += producer_slots)
                {
                    const int active_slots = std::min(
                        producer_slots,
                        canonical_summaries - wave_begin);
                    const int wave_items = output_rows * active_slots;

#pragma omp for schedule(static) reduction(+ : qk_duration_ns, v_duration_ns)
                    for (int item = 0; item < wave_items; ++item)
                    {
                        const int row = item / active_slots;
                        const int producer_slot = item % active_slots;
                        const int summary = wave_begin + producer_slot;
                        const int range_begin =
                            summary * plan.context_partition_rows;
                        const int range_end = std::min(
                            range_begin + plan.context_partition_rows,
                            kv_len);
                        const std::size_t workspace_slot =
                            slot_index(row, producer_slot + 1);
                        float *partial =
                            partial_output_ + workspace_slot * padded_head_dim;
                        std::fill(partial, partial + head_dim, 0.0f);
                        float partial_m =
                            -std::numeric_limits<float>::infinity();
                        float partial_l = 0.0f;
                        prepare_row(row, scratch);

                        const CPUFA2VisibleTile visible =
                            visible_tile(row, range_begin, range_end);
                        evaluateCanonicalAttentionSummary(
                            row,
                            scratch,
                            range_begin,
                            visible,
                            head_dim,
                            kv_tile,
                            score_tile,
                            accumulate_tile,
                            scores,
                            partial,
                            partial_m,
                            partial_l,
                            profiling_enabled,
                            qk_duration_ns,
                            v_duration_ns);
                        partial_m_[workspace_slot] = partial_m;
                        partial_l_[workspace_slot] = partial_l;
                    }

                    /* Producer `omp for` completion is the wave publication. */
#pragma omp for schedule(static)
                    for (int row = 0; row < output_rows; ++row)
                    {
                        const std::size_t merged_slot = slot_index(row, 0);
                        float *merged = partial_output_ +
                                        merged_slot * padded_head_dim;
                        float merged_m = partial_m_[merged_slot];
                        float merged_l = partial_l_[merged_slot];
                        for (int producer_slot = 0;
                             producer_slot < active_slots;
                             ++producer_slot)
                        {
                            const std::size_t workspace_slot =
                                slot_index(row, producer_slot + 1);
                            mergeAttentionSummary(
                                merged,
                                merged_m,
                                merged_l,
                                partial_output_ +
                                    workspace_slot * padded_head_dim,
                                partial_m_[workspace_slot],
                                partial_l_[workspace_slot],
                                head_dim);
                        }
                        partial_m_[merged_slot] = merged_m;
                        partial_l_[merged_slot] = merged_l;
                    }
                }

#pragma omp for schedule(static)
                for (int row = 0; row < output_rows; ++row)
                {
                    const std::size_t merged_slot = slot_index(row, 0);
                    finalize_row(
                        row,
                        partial_output_ + merged_slot * padded_head_dim,
                        partial_l_[merged_slot]);
                }
            };

            OMP_WORKSHARE_REGION(work);
            if (profiling_enabled)
            {
                KernelProfiler::recordParallel(
                    KernelType::ATTENTION_QK,
                    qk_duration_ns,
                    omp_get_max_threads());
                KernelProfiler::recordParallel(
                    KernelType::ATTENTION_V,
                    v_duration_ns,
                    omp_get_max_threads());
            }
            return true;
        }

        /**
         * @brief Execute canonical K/V summaries under either physical schedule.
         *
         * The format-specific callbacks contain only storage decoding and
         * vector math. This scheduler owns partition boundaries, OpenMP work
         * ownership, persistent context workspace, ordered reduction, and
         * profiler accounting once for every CPU K/V representation.
         *
         * `PrepareRow` runs before a worker evaluates one logical output row.
         * `VisibleTile` returns causal/window bounds for one canonical summary.
         * `ScoreTile` fills scores relative to the canonical summary origin and
         * updates its maximum while visiting a physical cache chunk.
         * `AccumulateTile` adds weighted V rows from one cache chunk to an
         * unnormalised numerator. `FinalizeRow` normalizes or transforms the
         * merged numerator into the caller's output layout.
         *
         * @tparam Scratch Worker-local, default-constructible format scratch.
         * @param query_rows Number of logical query rows in this invocation.
         * @param n_heads Number of local query heads per query row.
         * @param kv_len Number of addressable K/V rows.
         * @param head_dim Elements in one attention head.
         * @param kv_tile Cache-derived physical K/V tile width.
         * @param causal Whether the invocation applies causal visibility.
         * @param execution_policy Declarative physical scheduling policy.
         * @param prepare_row Format-specific row preparation callback.
         * @param visible_tile Query-specific K/V visibility callback.
         * @param score_tile Format-specific QK callback.
         * @param accumulate_tile Format-specific P@V callback.
         * @param finalize_row Format-specific final-output callback.
         * @return True after all rows are produced and finalized.
         */
        template <typename Scratch,
                  typename PrepareRow,
                  typename VisibleTile,
                  typename ScoreTile,
                  typename AccumulateTile,
                  typename FinalizeRow>
        bool executePartitionedAttention(
            int query_rows,
            int n_heads,
            int kv_len,
            int head_dim,
            int kv_tile,
            bool causal,
            const attention::AttentionExecutionPolicy &execution_policy,
            PrepareRow &&prepare_row,
            VisibleTile &&visible_tile,
            ScoreTile &&score_tile,
            AccumulateTile &&accumulate_tile,
            FinalizeRow &&finalize_row)
        {
            if (query_rows <= 0 || n_heads <= 0 || kv_len <= 0 ||
                head_dim <= 0 || head_dim > kMaximumAttentionHeadDim ||
                kv_tile <= 0 || kv_tile > detail::kMaxKVTile)
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] Invalid partitioned attention geometry");
                return false;
            }

            const int physical_workers = std::max(1, omp_get_max_threads());
            const cpu::fa2_policy::CPUFA2ParallelPlan plan =
                cpu::fa2_policy::selectCPUFA2ParallelPlan({
                    .batch_size = 1,
                    .query_rows = query_rows,
                    .local_query_heads = n_heads,
                    .kv_rows = kv_len,
                    .physical_workers = physical_workers,
                    .requested_axis =
                        execution_policy.prefill_parallel_axis,
                });
            if (!plan.valid)
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] Invalid partitioned attention plan");
                return false;
            }

            if (PerfStatsCollector::isDomainEnabled("kernel"))
            {
                /*
                 * Report the branch that will actually execute, rather than
                 * merely echoing the requested policy. In particular, an
                 * explicitly requested context plan with only one available
                 * partition enters the query-row implementation below. The
                 * E2E contract must authenticate physical execution, not an
                 * intent that the runtime geometry could not realize.
                 *
                 * `kv_len` is intentionally absent from the key. Canonical
                 * arithmetic/context partition counts expose every meaningful
                 * mode transition while keeping the number of aggregated
                 * records bounded during long decode runs.
                 */
                const auto executed_mode =
                    plan.usesContextParallelism()
                        ? cpu::fa2_policy::CPUFA2PhysicalMode::KeyValueContext
                        : cpu::fa2_policy::CPUFA2PhysicalMode::QuerySequence;
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_fa2_parallel_plan_executions",
                    1.0,
                    "execute",
                    "cpu",
                    {
                        {"requested_axis",
                         attention::attentionPrefillParallelAxisName(
                             execution_policy.prefill_parallel_axis)},
                        {"selected_mode",
                         cpu::fa2_policy::cpuFA2PhysicalModeName(executed_mode)},
                        {"query_rows", std::to_string(query_rows)},
                        {"local_query_heads", std::to_string(n_heads)},
                        {"head_dim", std::to_string(head_dim)},
                        {"physical_workers", std::to_string(physical_workers)},
                        {"arithmetic_partitions",
                         std::to_string(plan.arithmetic_partitions)},
                        {"context_partitions",
                         std::to_string(plan.context_partitions)},
                        {"context_partition_rows",
                         std::to_string(plan.context_partition_rows)},
                        {"physical_kv_tile", std::to_string(kv_tile)},
                    });
            }

            if (plan.usesContextParallelism())
            {
                return executeContextParallelAttentionWaves<Scratch>(
                    query_rows,
                    n_heads,
                    kv_len,
                    head_dim,
                    kv_tile,
                    plan,
                    prepare_row,
                    visible_tile,
                    score_tile,
                    accumulate_tile,
                    finalize_row);
            }

            const int output_rows = query_rows * n_heads;
            const int partitions = plan.arithmetic_partitions;
            const int partition_rows = plan.context_partition_rows;
            const int work_items = output_rows;

            const bool profiling_enabled = KernelProfiler::isEnabled();
            std::uint64_t qk_duration_ns = 0;
            std::uint64_t v_duration_ns = 0;
            const int requested_threads = computeOptimalAttentionThreads(
                n_heads, query_rows, kv_len, head_dim, causal);

            auto work = [&]()
            {
                Scratch scratch{};
                alignas(64) float local_partial[kMaximumAttentionHeadDim];
                alignas(64) float merged[kMaximumAttentionHeadDim];
                alignas(64) float scores[detail::kMaxKVTile];

#pragma omp for schedule(static) reduction(+ : qk_duration_ns, v_duration_ns)
                for (int item = 0; item < work_items; ++item)
                {
                    const int row = item;
                    prepare_row(row, scratch);

                    std::fill(merged, merged + head_dim, 0.0f);
                    float merged_m =
                        -std::numeric_limits<float>::infinity();
                    float merged_l = 0.0f;

                    const int first_partition = 0;
                    const int partition_end = partitions;
                    for (int partition = first_partition;
                         partition < partition_end;
                         ++partition)
                    {
                        const int range_begin = partition * partition_rows;
                        const int range_end =
                            std::min(range_begin + partition_rows, kv_len);
                        if (range_begin >= range_end)
                            continue;

                        float *partial = local_partial;
                        std::fill(partial, partial + head_dim, 0.0f);
                        float partial_m =
                            -std::numeric_limits<float>::infinity();
                        float partial_l = 0.0f;

                        const CPUFA2VisibleTile visible =
                            visible_tile(row, range_begin, range_end);
                        evaluateCanonicalAttentionSummary(
                            row,
                            scratch,
                            range_begin,
                            visible,
                            head_dim,
                            kv_tile,
                            score_tile,
                            accumulate_tile,
                            scores,
                            partial,
                            partial_m,
                            partial_l,
                            profiling_enabled,
                            qk_duration_ns,
                            v_duration_ns);

                        mergeAttentionSummary(
                            merged,
                            merged_m,
                            merged_l,
                            partial,
                            partial_m,
                            partial_l,
                            head_dim);
                    }

                    finalize_row(row, merged, merged_l);
                }

            };

            const bool use_full_team =
                kv_len > 100 || output_rows >= physical_workers;
            int actual_threads = requested_threads;
            if (use_full_team)
            {
                OMP_WORKSHARE_REGION(work);
                actual_threads = physical_workers;
            }
            else
            {
#pragma omp parallel num_threads(requested_threads)
                {
                    work();
                }
            }

            if (profiling_enabled)
            {
                KernelProfiler::recordParallel(
                    KernelType::ATTENTION_QK,
                    qk_duration_ns,
                    actual_threads);
                KernelProfiler::recordParallel(
                    KernelType::ATTENTION_V,
                    v_duration_ns,
                    actual_threads);
            }
            return true;
        }

        /**
         * @brief Weighted V accumulation from Q16_1 block: out[d] += weight * (scale * qs[d]).
         *
         * Loads int16 values from a Q16_1 block's qs[] array, sign-extends to
         * int32 via _mm512_cvtepi16_epi32, converts to FP32, multiplies by the
         * combined weight (softmax_weight * block_scale), and FMA-accumulates
         * into the FP32 output vector.
         *
         * @param out       FP32 output vector (accumulated in-place).
         * @param v_qs      Pointer to int16_t qs[] data from Q16_1 block.
         * @param combined  Pre-computed (softmax_weight * v_block.d).
         * @param head_dim  Number of elements.
         */
        // ---- Q16 single-row V-accumulation: named implementations ----

        static inline void accum_weighted_v_q16_scalar(
            float *out, const int16_t *v_qs, float combined, int head_dim)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                out[d] += combined * static_cast<float>(v_qs[d]);
            }
        }

#if defined(__AVX2__)
        static inline void accum_weighted_v_q16_avx2(
            float *out, const int16_t *v_qs, float combined, int head_dim)
        {
            const __m256 w = _mm256_set1_ps(combined);
            int d = 0;
            for (; d + 7 < head_dim; d += 8)
            {
                __m256 o = _mm256_loadu_ps(out + d);
                const __m128i vi16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(v_qs + d));
                const __m256i vi32 = _mm256_cvtepi16_epi32(vi16);
                const __m256 vf = _mm256_cvtepi32_ps(vi32);
                o = _mm256_fmadd_ps(vf, w, o);
                _mm256_storeu_ps(out + d, o);
            }
            for (; d < head_dim; ++d)
                out[d] += combined * static_cast<float>(v_qs[d]);
        }
#endif

#if defined(__AVX512F__)
        static inline void accum_weighted_v_q16_avx512(
            float *out, const int16_t *v_qs, float combined, int head_dim)
        {
            const __m512 w = _mm512_set1_ps(combined);
            int d = 0;
            for (; d + 31 < head_dim; d += 32)
            {
                __m512 o0 = _mm512_loadu_ps(out + d);
                __m512 o1 = _mm512_loadu_ps(out + d + 16);
                const __m512 vf0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v_qs + d))));
                const __m512 vf1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v_qs + d + 16))));
                o0 = _mm512_fmadd_ps(vf0, w, o0);
                o1 = _mm512_fmadd_ps(vf1, w, o1);
                _mm512_storeu_ps(out + d, o0);
                _mm512_storeu_ps(out + d + 16, o1);
            }
            for (; d + 15 < head_dim; d += 16)
            {
                __m512 o = _mm512_loadu_ps(out + d);
                const __m512 vf = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v_qs + d))));
                o = _mm512_fmadd_ps(vf, w, o);
                _mm512_storeu_ps(out + d, o);
            }
            for (; d < head_dim; ++d)
            {
                out[d] += combined * static_cast<float>(v_qs[d]);
            }
        }
#endif

        static void accum_weighted_v_q16(
            float *out, const int16_t *v_qs, float combined, int head_dim)
        {
            switch (activeISALevel())
            {
#if defined(__AVX512F__)
            case ISALevel::AVX512:
                accum_weighted_v_q16_avx512(out, v_qs, combined, head_dim);
                break;
#endif
            case ISALevel::AVX2:
                accum_weighted_v_q16_avx2(out, v_qs, combined, head_dim);
                break;
            default:
                accum_weighted_v_q16_scalar(out, v_qs, combined, head_dim);
                break;
            }
        }

        /**
         * @brief Batched 4-row V accumulation: out[d] += Σ_i w_i * v_i_qs[d].
         *
         * Loads the output vector once, accumulates 4 dequantised V rows with
         * their softmax weights, then stores once.  Saves ~60% of L1 traffic
         * vs calling accum_weighted_v_q16 four separate times.
         */
        // ---- Q16 4-row V-accumulation: named implementations ----

        static inline void accum_weighted_v_q16_4row_scalar(
            float *__restrict out,
            const int16_t *v0, float w0,
            const int16_t *v1, float w1,
            const int16_t *v2, float w2,
            const int16_t *v3, float w3,
            int head_dim)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                out[d] += w0 * static_cast<float>(v0[d]) + w1 * static_cast<float>(v1[d]) + w2 * static_cast<float>(v2[d]) + w3 * static_cast<float>(v3[d]);
            }
        }

#if defined(__AVX2__)
        static inline void accum_weighted_v_q16_4row_avx2(
            float *__restrict out,
            const int16_t *v0, float w0,
            const int16_t *v1, float w1,
            const int16_t *v2, float w2,
            const int16_t *v3, float w3,
            int head_dim)
        {
            const __m256 vw0 = _mm256_set1_ps(w0);
            const __m256 vw1 = _mm256_set1_ps(w1);
            const __m256 vw2 = _mm256_set1_ps(w2);
            const __m256 vw3 = _mm256_set1_ps(w3);
            int d = 0;
            for (; d + 7 < head_dim; d += 8)
            {
                __m256 o = _mm256_loadu_ps(out + d);
                auto cvt8 = [](const int16_t *p) -> __m256
                {
                    return _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(p))));
                };
                o = _mm256_fmadd_ps(cvt8(v0 + d), vw0, o);
                o = _mm256_fmadd_ps(cvt8(v1 + d), vw1, o);
                o = _mm256_fmadd_ps(cvt8(v2 + d), vw2, o);
                o = _mm256_fmadd_ps(cvt8(v3 + d), vw3, o);
                _mm256_storeu_ps(out + d, o);
            }
            for (; d < head_dim; ++d)
                out[d] += w0 * static_cast<float>(v0[d]) + w1 * static_cast<float>(v1[d]) + w2 * static_cast<float>(v2[d]) + w3 * static_cast<float>(v3[d]);
        }
#endif

#if defined(__AVX512F__)
        static inline void accum_weighted_v_q16_4row_avx512(
            float *__restrict out,
            const int16_t *v0, float w0,
            const int16_t *v1, float w1,
            const int16_t *v2, float w2,
            const int16_t *v3, float w3,
            int head_dim)
        {
            const __m512 vw0 = _mm512_set1_ps(w0);
            const __m512 vw1 = _mm512_set1_ps(w1);
            const __m512 vw2 = _mm512_set1_ps(w2);
            const __m512 vw3 = _mm512_set1_ps(w3);
            int d = 0;
            for (; d + 31 < head_dim; d += 32)
            {
                __m512 oA = _mm512_loadu_ps(out + d);
                __m512 oB = _mm512_loadu_ps(out + d + 16);
                oA = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v0 + d)))),
                    vw0, oA);
                oB = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v0 + d + 16)))),
                    vw0, oB);
                oA = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v1 + d)))),
                    vw1, oA);
                oB = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v1 + d + 16)))),
                    vw1, oB);
                oA = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v2 + d)))),
                    vw2, oA);
                oB = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v2 + d + 16)))),
                    vw2, oB);
                oA = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v3 + d)))),
                    vw3, oA);
                oB = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v3 + d + 16)))),
                    vw3, oB);
                _mm512_storeu_ps(out + d, oA);
                _mm512_storeu_ps(out + d + 16, oB);
            }
            for (; d + 15 < head_dim; d += 16)
            {
                __m512 o = _mm512_loadu_ps(out + d);
                o = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v0 + d)))),
                    vw0, o);
                o = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v1 + d)))),
                    vw1, o);
                o = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v2 + d)))),
                    vw2, o);
                o = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v3 + d)))),
                    vw3, o);
                _mm512_storeu_ps(out + d, o);
            }
            for (; d < head_dim; ++d)
            {
                out[d] += w0 * static_cast<float>(v0[d]) + w1 * static_cast<float>(v1[d]) + w2 * static_cast<float>(v2[d]) + w3 * static_cast<float>(v3[d]);
            }
        }
#endif

        static void accum_weighted_v_q16_4row(
            float *__restrict out,
            const int16_t *v0, float w0,
            const int16_t *v1, float w1,
            const int16_t *v2, float w2,
            const int16_t *v3, float w3,
            int head_dim)
        {
            switch (activeISALevel())
            {
#if defined(__AVX512F__)
            case ISALevel::AVX512:
                accum_weighted_v_q16_4row_avx512(out, v0, w0, v1, w1, v2, w2, v3, w3, head_dim);
                break;
#endif
            case ISALevel::AVX2:
                accum_weighted_v_q16_4row_avx2(out, v0, w0, v1, w1, v2, w2, v3, w3, head_dim);
                break;
            default:
                accum_weighted_v_q16_4row_scalar(out, v0, w0, v1, w1, v2, w2, v3, w3, head_dim);
                break;
            }
        }

        // =================================================================
        // Core flash-attention implementation
        // =================================================================

        /**
         * @brief Core tiled flash-attention with online softmax (FP32 path).
         *
         * This is the heart of the flash attention kernel.  It computes:
         *
         *     output[q][h] = softmax( Q[q,h] · K[:,kv_h]^T / √d ) · V[:,kv_h]
         *
         * …without ever materialising the full `[seq_len × kv_len]` score matrix.
         *
         * ### Algorithm outline
         *
         * For each query head `h` and query position `q`:
         *   1. Initialise `running_m = -∞` (running max) and `running_l = 0`
         *      (running softmax denominator).
         *   2. Iterate over KV-position tiles of size `kv_tile`.
         *      a. Compute `score[k] = Q[q,h] · K[k, kv_h] / √d` for each
         *         position k in the tile with the exact FP32 dot-product
         *         arithmetic used by serial decode.
         *      b. Find `block_max` = max score in this tile.
         *      c. Compute `new_m = max(running_m, block_max)`.
         *      d. **Rescale** the running output and denominator:
         *         `out *= exp(running_m − new_m)` and
         *         `running_l *= exp(running_m − new_m)`.
         *      e. For each k in tile: `p = exp(score[k] − new_m)`,
         *         `running_l += p`, `out += p * V[k, kv_h]`.
         *      f. Update `running_m = new_m`.
         *   3. Normalise: `out /= running_l`.
         *
         * ### Parallelism
         *
         * The outer loop over heads (`h`) is parallelised with OpenMP via
         * `OMP_WORKSHARE_REGION`.  Each head is fully independent, so this
         * achieves near-linear scaling across CPU cores.
         *
         * @param Q               Query data, row-major [seq_len, n_heads * head_dim].
         * @param K               Key data,   row-major [kv_len, n_kv_heads * head_dim].
         * @param V               Value data, row-major [kv_len, n_kv_heads * head_dim].
         * @param output          Output buffer,       [seq_len, n_heads * head_dim].
         * @param seq_len         Number of query positions.
         * @param kv_len          Number of key/value positions.
         * @param n_heads         Total query heads.
         * @param n_kv_heads      Total KV heads (GQA: n_heads / n_kv_heads > 1).
         * @param head_dim        Elements per head.
         * @param causal          If true, position k is masked when k > q_abs.
         * @param window_size     Sliding-window limit (-1 = unlimited).
         * @param position_offset Absolute position of the first query token
         *                        (non-zero during decode where q_pos 0 is not
         *                        the beginning of the sequence).
         * @param mask            Optional additive mask [seq_len, kv_len] or nullptr.
         * @return true on success, false if pointers are null or dimensions invalid.
         */
        bool compute_flash_fp32(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset,
            const float *mask,
            int head_start = 0,
            int gqa_n_rep = 0,
            const attention::AttentionExecutionPolicy &execution_policy = {},
            const attention::AttentionKVLogicalView &kv_logical_view = {})
        {
            KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);

            // --- Input validation ---
            if (!Q || !K || !V || !output)
            {
                return false;
            }
            if (seq_len <= 0 || kv_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0)
            {
                return false;
            }
            // GQA requires n_heads to be a multiple of n_kv_heads (e.g. 32 Q heads / 8 KV heads = 4 Q heads per KV head).
            // In TP with replicated KV, gqa_n_rep carries the global ratio so
            // local n_heads need not divide n_kv_heads evenly (e.g. 7 / 2).
            if (gqa_n_rep <= 0 && n_heads % n_kv_heads != 0)
            {
                return false;
            }
            if (!kv_logical_view.validFor(kv_len))
                return false;

            const auto physical_kv_row = [&](int logical_row) -> std::size_t
            {
                return static_cast<std::size_t>(
                    kv_logical_view.physicalRow(logical_row));
            };

            // --- Pre-compute constants ---
            const bool use_avx512 = cpu_supports_avx512();
            // Tile choice is invariant over positive K/V length. Grouped rows
            // can therefore use the same physical tile as their serial-decode
            // counterparts while causal bounds hide later speculative rows.
            const std::size_t fp32_head_row_bytes =
                static_cast<std::size_t>(head_dim) * sizeof(float);
            const int kv_tile = detail::selectCPUFlashKVTile(
                cpu::fa2_policy::CPUFA2KVStoragePair::FP32,
                head_dim,
                kv_len,
                fp32_head_row_bytes,
                fp32_head_row_bytes,
                launch_policy_.explicit_kv_tile);

            // For Grouped Query Attention: how many Q heads share one KV head.
            // TP-aware: use global GQA ratio when gqa_n_rep is provided.
            const int heads_per_kv = (gqa_n_rep > 0) ? gqa_n_rep : (n_heads / n_kv_heads);

            // Attention score scaling factor:  1/√d
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // Row strides in the flat [seq_len, heads*head_dim] layout.
            const int q_stride = n_heads * head_dim;     // Q row = all Q heads concatenated
            const int kv_stride = n_kv_heads * head_dim; // K/V row = all KV heads concatenated

            // Profiling: track QK dot-product and V accumulation times separately.
            const bool profiling_enabled = KernelProfiler::isEnabled();
            uint64_t qk_duration_ns = 0;
            uint64_t v_duration_ns = 0;

            const cpu::fa2_policy::CPUFA2ParallelPlan partition_plan =
                cpu::fa2_policy::selectCPUFA2ParallelPlan({
                    .batch_size = 1,
                    .query_rows = seq_len,
                    .local_query_heads = n_heads,
                    .kv_rows = kv_len,
                    .physical_workers = std::max(1, omp_get_max_threads()),
                    .requested_axis =
                        execution_policy.prefill_parallel_axis,
                });
            if (!partition_plan.valid)
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] Invalid FP32 attention partition plan");
                return false;
            }

            /*
             * The one-summary case below remains the lowest-overhead physical
             * implementation. Once a local head shard requires more than one
             * canonical summary, both query ownership and context ownership
             * enter the common producer/reducer so their arithmetic bytes are
             * identical. Grouped verifier rows retain their row-local decode
             * tiles; that separately proven path compares against serial rows
             * with differing visible cache lengths.
             */
            if (partition_plan.arithmetic_partitions > 1)
            {
                /** Per-worker state for one FP32 output row. */
                struct FP32RowScratch
                {
                    const float *query = nullptr;
                    const float *key = nullptr;
                    const float *value = nullptr;
                    const float *mask = nullptr;
                    int query_position = 0;
                    int visible_kv_rows = 0;
                };

                const auto prepare_row = [&](int row,
                                             FP32RowScratch &scratch)
                {
                    const int query_position = row / n_heads;
                    const int head = row % n_heads;
                    const int kv_head = (gqa_n_rep > 0)
                                            ? (head_start + head) / heads_per_kv
                                            : head / heads_per_kv;
                    scratch.query =
                        Q + static_cast<std::size_t>(query_position) * q_stride +
                        static_cast<std::size_t>(head) * head_dim;
                    scratch.key =
                        K + static_cast<std::size_t>(kv_head) * head_dim;
                    scratch.value =
                        V + static_cast<std::size_t>(kv_head) * head_dim;
                    scratch.mask = mask
                                       ? mask +
                                             static_cast<std::size_t>(query_position) *
                                                 kv_len
                                       : nullptr;
                    scratch.query_position = position_offset + query_position;
                    scratch.visible_kv_rows = kv_len;
                };

                const auto visible_tile = [&](int row,
                                              int tile_begin,
                                              int tile_end)
                {
                    const int query_position = row / n_heads;
                    const int absolute_position =
                        position_offset + query_position;
                    int visible_begin = tile_begin;
                    int visible_end = tile_end;
                    if (window_size > 0)
                    {
                        visible_begin = std::max(
                            visible_begin,
                            absolute_position - window_size + 1);
                    }
                    if (causal)
                        visible_end = std::min(visible_end, absolute_position + 1);
                    return CPUFA2VisibleTile{
                        .begin = std::max(
                            tile_begin,
                            std::min(visible_begin, tile_end)),
                        .end = std::max(
                            tile_begin,
                            std::min(visible_end, tile_end)),
                    };
                };

                const auto score_tile = [&](int,
                                            const FP32RowScratch &scratch,
                                            int tile_begin,
                                            CPUFA2VisibleTile visible,
                                            float *scores,
                                            float &block_max)
                {
                    int kv_row = visible.begin;
                    while (kv_row < visible.end)
                    {
                        if (kv_row + 3 < visible.end)
                        {
                            float score0 = 0.0f;
                            float score1 = 0.0f;
                            float score2 = 0.0f;
                            float score3 = 0.0f;
                            dot_fp32_4row(
                                scratch.query,
                                scratch.key +
                                    physical_kv_row(kv_row + 0) * kv_stride,
                                scratch.key +
                                    physical_kv_row(kv_row + 1) * kv_stride,
                                scratch.key +
                                    physical_kv_row(kv_row + 2) * kv_stride,
                                scratch.key +
                                    physical_kv_row(kv_row + 3) * kv_stride,
                                head_dim,
                                score0,
                                score1,
                                score2,
                                score3);
                            float values[4]{
                                score0 * scale,
                                score1 * scale,
                                score2 * scale,
                                score3 * scale,
                            };
                            for (int lane = 0; lane < 4; ++lane)
                            {
                                if (scratch.mask)
                                    values[lane] += scratch.mask[kv_row + lane];
                                scores[static_cast<std::size_t>(
                                    kv_row + lane - tile_begin)] = values[lane];
                                block_max = std::max(block_max, values[lane]);
                            }
                            kv_row += 4;
                            continue;
                        }

                        float score = dot_fp32(
                                          scratch.query,
                                          scratch.key +
                                              physical_kv_row(kv_row) * kv_stride,
                                          head_dim) *
                                      scale;
                        if (scratch.mask)
                            score += scratch.mask[kv_row];
                        scores[static_cast<std::size_t>(kv_row - tile_begin)] =
                            score;
                        block_max = std::max(block_max, score);
                        ++kv_row;
                    }
                };

                const auto accumulate_tile = [&](int,
                                                 const FP32RowScratch &scratch,
                                                 int tile_begin,
                                                 CPUFA2VisibleTile visible,
                                                 const float *scores,
                                                 float new_m,
                                                 float *partial,
                                                 float &new_l)
                {
                    int kv_row = visible.begin;
                    for (; kv_row + 3 < visible.end; kv_row += 4)
                    {
                        float probabilities[4];
                        batch_exp_4(
                            scores[static_cast<std::size_t>(kv_row - tile_begin + 0)],
                            scores[static_cast<std::size_t>(kv_row - tile_begin + 1)],
                            scores[static_cast<std::size_t>(kv_row - tile_begin + 2)],
                            scores[static_cast<std::size_t>(kv_row - tile_begin + 3)],
                            new_m,
                            probabilities[0],
                            probabilities[1],
                            probabilities[2],
                            probabilities[3]);
                        new_l += probabilities[0] + probabilities[1] +
                                 probabilities[2] + probabilities[3];
                        accum_weighted_v_4row(
                            partial,
                            scratch.value +
                                physical_kv_row(kv_row + 0) * kv_stride,
                            probabilities[0],
                            scratch.value +
                                physical_kv_row(kv_row + 1) * kv_stride,
                            probabilities[1],
                            scratch.value +
                                physical_kv_row(kv_row + 2) * kv_stride,
                            probabilities[2],
                            scratch.value +
                                physical_kv_row(kv_row + 3) * kv_stride,
                            probabilities[3],
                            head_dim);
                    }
                    for (; kv_row < visible.end; ++kv_row)
                    {
                        const float probability = std::exp(
                            scores[static_cast<std::size_t>(kv_row - tile_begin)] -
                            new_m);
                        new_l += probability;
                        accum_weighted_v(
                            partial,
                            scratch.value +
                                physical_kv_row(kv_row) * kv_stride,
                            probability,
                            head_dim,
                            use_avx512);
                    }
                };

                const auto finalize_row = [&](int row,
                                              float *merged,
                                              float merged_l)
                {
                    if (merged_l > 0.0f)
                        div_vec(merged, merged_l, head_dim, use_avx512);
                    std::memcpy(
                        output + static_cast<std::size_t>(row / n_heads) * q_stride +
                            static_cast<std::size_t>(row % n_heads) * head_dim,
                        merged,
                        static_cast<std::size_t>(head_dim) * sizeof(float));
                };

                return executePartitionedAttention<FP32RowScratch>(
                    seq_len,
                    n_heads,
                    kv_len,
                    head_dim,
                    kv_tile,
                    causal,
                    execution_policy,
                    prepare_row,
                    visible_tile,
                    score_tile,
                    accumulate_tile,
                    finalize_row);
            }

            // ---------------------------------------------------------------
            // Phase 2: Main flash-attention loop.
            // ---------------------------------------------------------------
            // Compute optimal thread count.  For small problems (short
            // sequences, decode), the per-head work may be too small to
            // amortise OMP fork/join overhead (~30-50µs).  The heuristic
            // estimates total FLOPs and scales threads accordingly.
            const int attn_threads = computeOptimalAttentionThreads(
                n_heads, seq_len, kv_len, head_dim, causal);

            /*
             * This flag controls only OpenMP ownership. Physical K/V tiling is
             * resolved independently from detected private-cache geometry; the
             * canonical scheduler keeps arithmetic byte-invariant under either
             * ownership or tile choice.
             */
            const bool decode_schedule = kv_len != seq_len && kv_len >= 1;
            const bool partition_query_rows =
                !decode_schedule && seq_len > 1;
            const int attention_tasks =
                partition_query_rows ? n_heads * seq_len : n_heads;

            auto work = [&]()
            {
#pragma omp for schedule(static) reduction(+ : qk_duration_ns, v_duration_ns)
                for (int task = 0; task < attention_tasks; ++task)
                {
                    /*
                     * Every prefill output row is independent once Q/K/V and
                     * its causal horizon are fixed.  Scheduling complete
                     * (head,row) tasks therefore exposes the query-sequence
                     * policy promised by the public interface while preserving
                     * the exact dot-product, tile, softmax, and V-accumulation
                     * order within each row.  Decode remains head-partitioned;
                     * grouped verifier rows use the same row decomposition so
                     * small-M work can occupy surplus cores without row replay.
                     */
                    const int h = partition_query_rows
                                      ? (task / seq_len)
                                      : task;
                    const int q_begin = partition_query_rows
                                            ? (task % seq_len)
                                            : 0;
                    const int q_end = partition_query_rows
                                          ? q_begin + 1
                                          : seq_len;

                    // GQA mapping: which KV head does this Q head read from?
                    // When gqa_n_rep > 0, KV heads are REPLICATED (each rank has all KV heads),
                    // so we need the global head position (head_start + h) to index into the
                    // full KV tensor. When gqa_n_rep == 0, KV heads are SHARDED (each rank
                    // has only local_n_kv_heads), so we use local indexing (h only).
                    const int kv_h = (gqa_n_rep > 0)
                                         ? (head_start + h) / heads_per_kv
                                         : h / heads_per_kv;

                    // --- Loop over query positions for this head ---
                    for (int q_pos = q_begin; q_pos < q_end; ++q_pos)
                    {
                        // Pointer to this (q_pos, head) slice of the output buffer.
                        float *out = output + static_cast<size_t>(q_pos) * q_stride + static_cast<size_t>(h) * head_dim;
                        std::fill(out, out + head_dim, 0.0f); // Zero-init the accumulator

                        // Online softmax running state:
                        //   running_m = max score seen so far (for numerical stability)
                        //   running_l = sum of exp(score - running_m) seen so far (denominator)
                        float running_m = -std::numeric_limits<float>::infinity();
                        float running_l = 0.0f;

                        // Raw Q pointer for this (q_pos, head) pair.
                        const float *q_ptr = Q + static_cast<size_t>(q_pos) * q_stride + static_cast<size_t>(h) * head_dim;

                        // Absolute position of this query token in the full sequence
                        // (needed for causal masking during decode).
                        const int q_abs = position_offset + q_pos;

                        // Optional additive mask row for this query position.
                        const float *mask_row = mask ? (mask + static_cast<size_t>(q_pos) * kv_len) : nullptr;

                        const int query_kv_len = kv_len;
                        const int query_kv_tile = kv_tile;

                        // ===================================================
                        // KV tile loop — the heart of flash attention
                        // ===================================================
                        // We iterate over KV positions in tiles of `kv_tile`.
                        // For each tile we:
                        //   (a) compute QK scores (the "QK phase")
                        //   (b) run the online softmax correction
                        //   (c) accumulate weighted V into the output (the "V phase")
                        for (int k0 = 0; k0 < query_kv_len; k0 += query_kv_tile)
                        {
                            const int k1 = std::min(k0 + query_kv_tile, query_kv_len); // End of this tile
                            float block_max = -std::numeric_limits<float>::infinity();

                            // Stack-allocated score buffer (max tile size bounded by policy).
                            float block_scores[detail::kMaxKVTile];
                            const int blk = k1 - k0; // Number of KV positions in this tile

                            const auto qk_start = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

                            // --- Compute the valid attention window within this tile ---
                            // Positions outside [valid_start, valid_end) are masked to -inf
                            // (they must not contribute to the softmax).
                            int valid_start = k0;
                            int valid_end = k1;
                            if (window_size > 0)
                            {
                                // Sliding window: only attend to the most recent `window_size` positions.
                                valid_start = std::max(valid_start, q_abs - window_size + 1);
                            }
                            if (causal)
                            {
                                // Causal mask: cannot attend to future positions.
                                valid_end = std::min(valid_end, q_abs + 1);
                            }
                            // Clamp to tile boundaries.
                            valid_start = std::max(k0, std::min(valid_start, k1));
                            valid_end = std::max(k0, std::min(valid_end, k1));

                            // Fill masked positions with -inf so exp(-inf) = 0.
                            for (int k = k0; k < valid_start; ++k)
                            {
                                block_scores[static_cast<size_t>(k - k0)] = -std::numeric_limits<float>::infinity();
                            }
                            for (int k = valid_end; k < k1; ++k)
                            {
                                block_scores[static_cast<size_t>(k - k0)] = -std::numeric_limits<float>::infinity();
                            }

                            // --- QK Phase: compute score[k] = Q · K[k] / √d ---
                            // Four-row FP32 dot products expose independent
                            // accumulators for instruction-level parallelism;
                            // the scalar loop handles only the final tail.
                            for (int k = valid_start; k < valid_end;)
                            {
                                // Batched 4-row dot products for ILP.
                                if (k + 3 < valid_end)
                                {
                                    const float *kbase = K + static_cast<size_t>(kv_h) * head_dim;
                                    float s0, s1, s2, s3;
                                    dot_fp32_4row(
                                        q_ptr,
                                        kbase + physical_kv_row(k + 0) * kv_stride,
                                        kbase + physical_kv_row(k + 1) * kv_stride,
                                        kbase + physical_kv_row(k + 2) * kv_stride,
                                        kbase + physical_kv_row(k + 3) * kv_stride,
                                        head_dim, s0, s1, s2, s3);
                                    s0 *= scale;
                                    s1 *= scale;
                                    s2 *= scale;
                                    s3 *= scale;
                                    if (mask_row)
                                    {
                                        s0 += mask_row[k + 0];
                                        s1 += mask_row[k + 1];
                                        s2 += mask_row[k + 2];
                                        s3 += mask_row[k + 3];
                                    }
                                    block_scores[static_cast<size_t>(k - k0 + 0)] = s0;
                                    block_scores[static_cast<size_t>(k - k0 + 1)] = s1;
                                    block_scores[static_cast<size_t>(k - k0 + 2)] = s2;
                                    block_scores[static_cast<size_t>(k - k0 + 3)] = s3;
                                    block_max = std::max(block_max, std::max(std::max(s0, s1), std::max(s2, s3)));
                                    k += 4;
                                    continue;
                                }
                                // Scalar / tail path
                                float s = dot_fp32(
                                    q_ptr,
                                    K + physical_kv_row(k) * kv_stride +
                                        static_cast<size_t>(kv_h) * head_dim,
                                    head_dim);
                                s *= scale;

                                if (mask_row)
                                {
                                    s += mask_row[k];
                                }

                                block_scores[static_cast<size_t>(k - k0)] = s;
                                block_max = std::max(block_max, s);
                                ++k;
                            }

                            if (profiling_enabled)
                            {
                                qk_duration_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - qk_start).count());
                            }

                            // --- Online softmax correction ---
                            // The key insight: if a new tile has a higher max than what
                            // we've seen, all previous contributions must be rescaled.
                            //
                            //   new_m = max(running_m, block_max)
                            //   alpha = exp(running_m - new_m)      ← correction factor
                            //   out   *= alpha                       ← rescale accumulated output
                            //   running_l *= alpha                   ← rescale denominator
                            //
                            // If running_m was -inf (first tile), alpha = 0, which correctly
                            // zeroes out any stale data in the accumulator.
                            const float new_m = std::max(running_m, block_max);
                            const float alpha = std::isfinite(running_m) ? std::exp(running_m - new_m) : 0.0f;
                            scale_vec(out, alpha, head_dim, use_avx512);
                            float new_l = running_l * alpha;

                            // --- V Phase: accumulate weighted values ---
                            // For each KV position in this tile, compute the unnormalised
                            // softmax probability p = exp(score - new_m) and add p*V to out.
                            // Positions in [valid_start, valid_end) are guaranteed finite.
                            // Positions outside that range are -inf (masked) — skip them.
                            const auto v_start = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

                            {
                                const float *v_base = V + static_cast<size_t>(kv_h) * head_dim;
                                int k = valid_start;
                                // 4-wide batched: vectorized exp + batched V accumulation
                                for (; k + 3 < valid_end; k += 4)
                                {
                                    float pp[4];
                                    batch_exp_4(
                                        block_scores[static_cast<size_t>(k - k0 + 0)],
                                        block_scores[static_cast<size_t>(k - k0 + 1)],
                                        block_scores[static_cast<size_t>(k - k0 + 2)],
                                        block_scores[static_cast<size_t>(k - k0 + 3)],
                                        new_m, pp[0], pp[1], pp[2], pp[3]);
                                    new_l += pp[0] + pp[1] + pp[2] + pp[3];

                                    accum_weighted_v_4row(
                                        out,
                                        v_base + physical_kv_row(k + 0) * kv_stride, pp[0],
                                        v_base + physical_kv_row(k + 1) * kv_stride, pp[1],
                                        v_base + physical_kv_row(k + 2) * kv_stride, pp[2],
                                        v_base + physical_kv_row(k + 3) * kv_stride, pp[3],
                                        head_dim);
                                }
                                // Scalar tail: remaining valid positions
                                for (; k < valid_end; ++k)
                                {
                                    const float p = std::exp(block_scores[static_cast<size_t>(k - k0)] - new_m);
                                    new_l += p;
                                    accum_weighted_v(
                                        out,
                                        v_base + physical_kv_row(k) * kv_stride,
                                        p,
                                        head_dim,
                                        use_avx512);
                                }
                            }

                            if (profiling_enabled)
                            {
                                v_duration_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - v_start).count());
                            }

                            // Advance running state for this query position.
                            running_m = new_m;
                            running_l = new_l;
                        } // end KV tile loop

                        // --- Final normalisation ---
                        // Divide the accumulated output by the softmax denominator
                        // to get the true weighted average:  out = out / Σ exp(s - m).
                        div_vec(out, running_l, head_dim, use_avx512);
                    } // end q_pos loop
                } // end head loop
            };

            // DECODE THREADING STRATEGY: prevent libgomp thread pool sleep.
            //
            // When kv_len is large enough that attention takes >~200μs
            // (roughly kv_len > 100 for this model), the idle gap between the
            // last OMP region (RoPE) and the next one (Wo GEMV) causes libgomp
            // to put worker threads to futex-sleep. Waking 27 threads costs
            // ~5ms via futex(FUTEX_WAKE), which was the root cause of decode
            // degradation from 10→4 tok/s at long context.
            //
            // Fix: for decode at kv_len > 100, use ALL threads in the
            // attention OMP region. This keeps the thread pool alive between
            // RoPE and Wo. Each thread handles ~1 head (28 heads / 28 threads).
            //
            // For short context (kv_len <= 100) or prefill, use the heuristic
            // thread count — threads stay within libgomp's spin window and
            // Wo GEMV fork is fast regardless.
            //
            // Note: OMP_WAIT_POLICY=active and GOMP_SPINCOUNT=infinite do NOT
            // prevent the inter-region sleep; libgomp's team-end idle path has
            // a fixed spin count independent of these settings.
            const bool is_decode_phase = (kv_len != seq_len);
            const bool force_full_pool = is_decode_phase && kv_len > 100;

            // Use OMP_WORKSHARE_REGION for decode so the kernel is
            // compatible with a persistent outer parallel region (the
            // macro detects omp_in_parallel() and skips creating a new
            // team, eliminating ~16 µs fork/join overhead per call).
            int actual_threads;
            if (force_full_pool)
            {
                OMP_WORKSHARE_REGION(work);
                actual_threads = omp_get_max_threads();
            }
            else
            {
#pragma omp parallel num_threads(attn_threads)
                {
                    work();
                }
                actual_threads = attn_threads;
            }

            // Report profiling breakdown (QK vs V phase) if enabled.
            // qk_duration_ns and v_duration_ns are thread-time sums from
            // the OMP reduction — divide by active thread count to get
            // wall-clock estimates that are consistent with other kernel
            // timings in the profiler output.
            if (profiling_enabled)
            {
                KernelProfiler::recordParallel(KernelType::ATTENTION_QK, qk_duration_ns, actual_threads);
                KernelProfiler::recordParallel(KernelType::ATTENTION_V, v_duration_ns, actual_threads);
            }
            return true;
        }

#if (defined(__AVX512F__) && defined(__AVX512VNNI__)) || defined(__AVX2__)
        /**
         * @brief Prefill flash attention with Q16_1 KV cache (VNNI accelerated).
         *
         * Handles seq_len >= 1 (including prefill where seq_len == kv_len).
         * Uses the same VNNI QK dot products and Q16_1 V accumulation as
         * the decode path, but loops over multiple query positions.
         *
         * QK Phase: Q is quantized per-head to int16, then dotted against
         * K block qs[] data using VPDPWSSD. For single-block-per-head layouts
         * (block_size >= head_dim), a 4-row batched VNNI dot is used.
         *
         * V Phase: 4-wide vectorized exp via fast_exp_avx512 + batched
         * accum_weighted_v_q16_4row for maximum throughput.
         */
        bool compute_prefill_q16kv(
            const float *Q,
            const Q16_1Tensor *K_q16, const Q16_1Tensor *V_q16,
            float *output,
            int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset,
            int head_start = 0,
            int gqa_n_rep = 0,
            const attention::AttentionExecutionPolicy &execution_policy = {},
            const attention::AttentionKVLogicalView &kv_logical_view = {},
            Q16QueryLayout query_layout = Q16QueryLayout::RowMajor)
        {
            KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);

            if (!Q || !K_q16 || !V_q16 || !output)
                return false;
            if (seq_len <= 0 || kv_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0)
                return false;
            if (gqa_n_rep <= 0 && n_heads % n_kv_heads != 0)
                return false;
            if (!kv_logical_view.validFor(kv_len))
                return false;

            const int heads_per_kv = (gqa_n_rep > 0) ? gqa_n_rep : (n_heads / n_kv_heads);
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // Q16_1 block layout
            const Q16BlockSize blk_size = K_q16->q16_block_size();
            const size_t block_bytes = q16_block_size_bytes(blk_size);
            const size_t block_elems = q16_block_size_elements(blk_size);
            const size_t blocks_per_kv_row = K_q16->blocks_per_row();
            constexpr size_t QS_OFFSET = sizeof(float) + sizeof(int32_t); // = 8 bytes

            const size_t blocks_per_head_detect =
                (static_cast<size_t>(head_dim) + block_elems - 1) / block_elems;
            const bool is_head_major =
                (blocks_per_kv_row == blocks_per_head_detect && n_kv_heads > 1);
            const size_t rows_per_head = is_head_major
                                             ? (K_q16->rows() / static_cast<size_t>(n_kv_heads))
                                             : 0;

            const detail::Q16KVLogicalLayout q16_layout{
                .blocks_per_tensor_row = blocks_per_kv_row,
                .blocks_per_head = blocks_per_head_detect,
                .physical_rows_per_head = is_head_major
                                              ? rows_per_head
                                              : K_q16->rows(),
                .kv_heads = n_kv_heads,
                .head_major = is_head_major,
                .logical_view = kv_logical_view,
            };
            if (K_q16->q16_block_size() != V_q16->q16_block_size() ||
                K_q16->rows() != V_q16->rows() ||
                K_q16->blocks_per_row() != V_q16->blocks_per_row() ||
                !q16_layout.valid(K_q16->rows(), kv_len))
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] Invalid Q16 prefill K/V layout or logical ring view");
                return false;
            }

            const uint8_t *k_raw = static_cast<const uint8_t *>(K_q16->raw_data());
            const uint8_t *v_raw = static_cast<const uint8_t *>(V_q16->raw_data());
            if (!k_raw || !v_raw)
                return false;

            const int q_stride = n_heads * head_dim;
            const auto query_head = [&](int query_position,
                                        int head) -> const float *
            {
                if (query_layout == Q16QueryLayout::HeadMajor)
                {
                    return Q +
                           (static_cast<std::size_t>(head) * seq_len +
                            static_cast<std::size_t>(query_position)) *
                               head_dim;
                }
                return Q +
                       static_cast<std::size_t>(query_position) * q_stride +
                       static_cast<std::size_t>(head) * head_dim;
            };

            // Physical cache tiling is independent of canonical arithmetic.
            const std::size_t q16_head_row_bytes =
                blocks_per_head_detect * block_bytes;
            const int kv_tile = detail::selectCPUFlashKVTile(
                cpu::fa2_policy::CPUFA2KVStoragePair::Q16_1,
                head_dim,
                kv_len,
                q16_head_row_bytes,
                q16_head_row_bytes,
                launch_policy_.explicit_kv_tile);

            const bool profiling_enabled = KernelProfiler::isEnabled();
            uint64_t qk_duration_ns = 0;
            uint64_t v_duration_ns = 0;

            const int attn_threads = computeOptimalAttentionThreads(
                n_heads, seq_len, kv_len, head_dim, causal);

            // For VNNI QK: Q is quantized per-head to int16
            constexpr int QMAX = 2047;

            const cpu::fa2_policy::CPUFA2ParallelPlan partition_plan =
                cpu::fa2_policy::selectCPUFA2ParallelPlan({
                    .batch_size = 1,
                    .query_rows = seq_len,
                    .local_query_heads = n_heads,
                    .kv_rows = kv_len,
                    .physical_workers = std::max(1, omp_get_max_threads()),
                    .requested_axis =
                        execution_policy.prefill_parallel_axis,
                });
            if (!partition_plan.valid)
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] Invalid Q16 prefill partition plan");
                return false;
            }

            if (partition_plan.arithmetic_partitions > 1)
            {
                /** Per-worker state for one Q16 output row. */
                struct Q16RowScratch
                {
                    alignas(64) std::int16_t query[
                        detail::kMaxI16RowStride]{};
                    float combined_query_scale = 0.0f;
                    int kv_head = 0;
                    int absolute_position = 0;
                };

                const auto prepare_row = [&](int row,
                                             Q16RowScratch &scratch)
                {
                    const int query_position = row / n_heads;
                    const int head = row % n_heads;
                    const int kv_head = (gqa_n_rep > 0)
                                            ? (head_start + head) / heads_per_kv
                                            : head / heads_per_kv;
                    const float *query = query_head(query_position, head);
                    scratch.combined_query_scale = quantize_row_i16_i12(
                                                       query,
                                                       scratch.query,
                                                       head_dim,
                                                       QMAX) *
                                                   scale;
                    scratch.kv_head = kv_head;
                    scratch.absolute_position =
                        position_offset + query_position;
                };

                const auto visible_tile = [&](int row,
                                              int tile_begin,
                                              int tile_end)
                {
                    const int absolute_position =
                        position_offset + row / n_heads;
                    int visible_begin = tile_begin;
                    int visible_end = tile_end;
                    if (window_size > 0)
                    {
                        visible_begin = std::max(
                            visible_begin,
                            absolute_position - window_size + 1);
                    }
                    if (causal)
                        visible_end = std::min(visible_end, absolute_position + 1);
                    return CPUFA2VisibleTile{
                        .begin = std::max(
                            tile_begin,
                            std::min(visible_begin, tile_end)),
                        .end = std::max(
                            tile_begin,
                            std::min(visible_end, tile_end)),
                    };
                };

                const auto score_tile = [&](int,
                                            const Q16RowScratch &scratch,
                                            int tile_begin,
                                            CPUFA2VisibleTile visible,
                                            float *scores,
                                            float &block_max)
                {
                    int kv_row = visible.begin;
                    while (kv_row < visible.end)
                    {
                        if (q16_layout.blocks_per_head == 1 &&
                            kv_row + 3 < visible.end)
                        {
                            const std::uint8_t *block0 =
                                k_raw + q16_layout.blockIndex(
                                            kv_row + 0, scratch.kv_head, 0) *
                                            block_bytes;
                            const std::uint8_t *block1 =
                                k_raw + q16_layout.blockIndex(
                                            kv_row + 1, scratch.kv_head, 0) *
                                            block_bytes;
                            const std::uint8_t *block2 =
                                k_raw + q16_layout.blockIndex(
                                            kv_row + 2, scratch.kv_head, 0) *
                                            block_bytes;
                            const std::uint8_t *block3 =
                                k_raw + q16_layout.blockIndex(
                                            kv_row + 3, scratch.kv_head, 0) *
                                            block_bytes;
                            float scale0 = 0.0f;
                            float scale1 = 0.0f;
                            float scale2 = 0.0f;
                            float scale3 = 0.0f;
                            std::memcpy(&scale0, block0, sizeof(float));
                            std::memcpy(&scale1, block1, sizeof(float));
                            std::memcpy(&scale2, block2, sizeof(float));
                            std::memcpy(&scale3, block3, sizeof(float));
                            std::int32_t dot0 = 0;
                            std::int32_t dot1 = 0;
                            std::int32_t dot2 = 0;
                            std::int32_t dot3 = 0;
                            dot_i16_i16_i32_vnni_4row(
                                scratch.query,
                                reinterpret_cast<const std::int16_t *>(
                                    block0 + QS_OFFSET),
                                reinterpret_cast<const std::int16_t *>(
                                    block1 + QS_OFFSET),
                                reinterpret_cast<const std::int16_t *>(
                                    block2 + QS_OFFSET),
                                reinterpret_cast<const std::int16_t *>(
                                    block3 + QS_OFFSET),
                                head_dim,
                                dot0,
                                dot1,
                                dot2,
                                dot3);
                            const float values[4]{
                                static_cast<float>(dot0) *
                                    scratch.combined_query_scale * scale0,
                                static_cast<float>(dot1) *
                                    scratch.combined_query_scale * scale1,
                                static_cast<float>(dot2) *
                                    scratch.combined_query_scale * scale2,
                                static_cast<float>(dot3) *
                                    scratch.combined_query_scale * scale3,
                            };
                            for (int lane = 0; lane < 4; ++lane)
                            {
                                scores[static_cast<std::size_t>(
                                    kv_row + lane - tile_begin)] = values[lane];
                                block_max = std::max(block_max, values[lane]);
                            }
                            kv_row += 4;
                            continue;
                        }

                        float score = 0.0f;
                        for (std::size_t block = 0;
                             block < q16_layout.blocks_per_head;
                             ++block)
                        {
                            const std::size_t block_index =
                                q16_layout.blockIndex(
                                    kv_row, scratch.kv_head, block);
                            const std::uint8_t *stored =
                                k_raw + block_index * block_bytes;
                            float stored_scale = 0.0f;
                            std::memcpy(
                                &stored_scale,
                                stored,
                                sizeof(float));
                            const int elements = static_cast<int>(std::min(
                                block_elems,
                                static_cast<std::size_t>(head_dim) -
                                    block * block_elems));
                            const std::int32_t dot = dot_i16_i16_i32(
                                scratch.query + block * block_elems,
                                reinterpret_cast<const std::int16_t *>(
                                    stored + QS_OFFSET),
                                elements);
                            score += static_cast<float>(dot) *
                                     scratch.combined_query_scale *
                                     stored_scale;
                        }
                        scores[static_cast<std::size_t>(kv_row - tile_begin)] =
                            score;
                        block_max = std::max(block_max, score);
                        ++kv_row;
                    }
                };

                const auto accumulate_tile = [&](int,
                                                 const Q16RowScratch &scratch,
                                                 int tile_begin,
                                                 CPUFA2VisibleTile visible,
                                                 const float *scores,
                                                 float new_m,
                                                 float *partial,
                                                 float &new_l)
                {
                    int kv_row = visible.begin;
                    if (q16_layout.blocks_per_head == 1)
                    {
                        for (; kv_row + 3 < visible.end; kv_row += 4)
                        {
                            float probabilities[4];
                            batch_exp_4(
                                scores[static_cast<std::size_t>(kv_row - tile_begin + 0)],
                                scores[static_cast<std::size_t>(kv_row - tile_begin + 1)],
                                scores[static_cast<std::size_t>(kv_row - tile_begin + 2)],
                                scores[static_cast<std::size_t>(kv_row - tile_begin + 3)],
                                new_m,
                                probabilities[0],
                                probabilities[1],
                                probabilities[2],
                                probabilities[3]);
                            new_l += probabilities[0] + probabilities[1] +
                                     probabilities[2] + probabilities[3];
                            const std::uint8_t *block0 =
                                v_raw + q16_layout.blockIndex(
                                            kv_row + 0, scratch.kv_head, 0) *
                                            block_bytes;
                            const std::uint8_t *block1 =
                                v_raw + q16_layout.blockIndex(
                                            kv_row + 1, scratch.kv_head, 0) *
                                            block_bytes;
                            const std::uint8_t *block2 =
                                v_raw + q16_layout.blockIndex(
                                            kv_row + 2, scratch.kv_head, 0) *
                                            block_bytes;
                            const std::uint8_t *block3 =
                                v_raw + q16_layout.blockIndex(
                                            kv_row + 3, scratch.kv_head, 0) *
                                            block_bytes;
                            float scale0 = 0.0f;
                            float scale1 = 0.0f;
                            float scale2 = 0.0f;
                            float scale3 = 0.0f;
                            std::memcpy(&scale0, block0, sizeof(float));
                            std::memcpy(&scale1, block1, sizeof(float));
                            std::memcpy(&scale2, block2, sizeof(float));
                            std::memcpy(&scale3, block3, sizeof(float));
                            accum_weighted_v_q16_4row(
                                partial,
                                reinterpret_cast<const std::int16_t *>(
                                    block0 + QS_OFFSET),
                                probabilities[0] * scale0,
                                reinterpret_cast<const std::int16_t *>(
                                    block1 + QS_OFFSET),
                                probabilities[1] * scale1,
                                reinterpret_cast<const std::int16_t *>(
                                    block2 + QS_OFFSET),
                                probabilities[2] * scale2,
                                reinterpret_cast<const std::int16_t *>(
                                    block3 + QS_OFFSET),
                                probabilities[3] * scale3,
                                head_dim);
                        }
                    }

                    for (; kv_row < visible.end; ++kv_row)
                    {
                        const float probability = std::exp(
                            scores[static_cast<std::size_t>(kv_row - tile_begin)] -
                            new_m);
                        new_l += probability;
                        for (std::size_t block = 0;
                             block < q16_layout.blocks_per_head;
                             ++block)
                        {
                            const std::size_t block_index =
                                q16_layout.blockIndex(
                                    kv_row, scratch.kv_head, block);
                            const std::uint8_t *stored =
                                v_raw + block_index * block_bytes;
                            float stored_scale = 0.0f;
                            std::memcpy(
                                &stored_scale,
                                stored,
                                sizeof(float));
                            const int elements = static_cast<int>(std::min(
                                block_elems,
                                static_cast<std::size_t>(head_dim) -
                                    block * block_elems));
                            accum_weighted_v_q16(
                                partial + block * block_elems,
                                reinterpret_cast<const std::int16_t *>(
                                    stored + QS_OFFSET),
                                probability * stored_scale,
                                elements);
                        }
                    }
                };

                const auto finalize_row = [&](int row,
                                              float *merged,
                                              float merged_l)
                {
                    if (merged_l > 0.0f)
                    {
                        scale_vec(
                            merged,
                            1.0f / merged_l,
                            head_dim,
                            cpu_supports_avx512());
                    }
                    std::memcpy(
                        output + static_cast<std::size_t>(row / n_heads) * q_stride +
                            static_cast<std::size_t>(row % n_heads) * head_dim,
                        merged,
                        static_cast<std::size_t>(head_dim) * sizeof(float));
                };

                return executePartitionedAttention<Q16RowScratch>(
                    seq_len,
                    n_heads,
                    kv_len,
                    head_dim,
                    kv_tile,
                    causal,
                    execution_policy,
                    prepare_row,
                    visible_tile,
                    score_tile,
                    accumulate_tile,
                    finalize_row);
            }

            const bool partition_query_rows = seq_len > 1;
            const int attention_tasks =
                partition_query_rows ? n_heads * seq_len : n_heads;

            auto work = [&]()
            {
                alignas(64) int16_t q_i16_buf[detail::kMaxI16RowStride];

#pragma omp for schedule(static) reduction(+ : qk_duration_ns, v_duration_ns)
                for (int task = 0; task < attention_tasks; ++task)
                {
                    /*
                     * Quantized prefill follows the same physical query policy
                     * as FP32: one task owns one complete output row.  Keeping
                     * the entire row inside a task preserves VNNI dot-product
                     * and online-softmax order while allowing TP shards with
                     * fewer heads than cores to use their full worker team.
                     */
                    const int h = partition_query_rows
                                      ? task / seq_len
                                      : task;
                    const int q_begin = partition_query_rows
                                            ? task % seq_len
                                            : 0;
                    const int q_end = partition_query_rows
                                          ? q_begin + 1
                                          : seq_len;

                    // GQA mapping: replicated KV uses global head position;
                    // sharded KV uses local indexing (see compute_flash_fp32 comment).
                    const int kv_h = (gqa_n_rep > 0)
                                         ? (head_start + h) / heads_per_kv
                                         : h / heads_per_kv;

                    // Hoist per-head block layout invariants. Q16_1 KV caches
                    // may be either [position][head][dim] or [head][position][dim].
                    // The grouped verifier must read the same logical cache
                    // rows as serial decode, so both layouts share the same
                    // address formula as every native Q16 attention regime.
                    const size_t blocks_per_head = q16_layout.blocks_per_head;

                    float block_scores[detail::kMaxKVTile];

                    for (int q_pos = q_begin; q_pos < q_end; ++q_pos)
                    {
                        float *out = output + static_cast<size_t>(q_pos) * q_stride + static_cast<size_t>(h) * head_dim;
                        std::fill(out, out + head_dim, 0.0f);

                        float running_m = -std::numeric_limits<float>::infinity();
                        float running_l = 0.0f;

                        const float *q_ptr = query_head(q_pos, h);
                        const int q_abs = position_offset + q_pos;

                        // Quantize Q to int16 once per (head, q_pos)
                        const float q_scale = quantize_row_i16_i12(
                            q_ptr, q_i16_buf, head_dim, QMAX);
                        const float qk_combined_scale = q_scale * scale;

                        // KV tile loop
                        for (int k0 = 0; k0 < kv_len; k0 += kv_tile)
                        {
                            const int k1 = std::min(k0 + kv_tile, kv_len);
                            float block_max = -std::numeric_limits<float>::infinity();

                            const int blk = k1 - k0;

                            // Valid window for causal/sliding-window masking
                            int valid_start = k0;
                            int valid_end = k1;
                            if (window_size > 0)
                                valid_start = std::max(valid_start, q_abs - window_size + 1);
                            if (causal)
                                valid_end = std::min(valid_end, q_abs + 1);
                            valid_start = std::max(k0, std::min(valid_start, k1));
                            valid_end = std::max(k0, std::min(valid_end, k1));

                            // Fill masked positions
                            for (int k = k0; k < valid_start; ++k)
                                block_scores[static_cast<size_t>(k - k0)] = -std::numeric_limits<float>::infinity();
                            for (int k = valid_end; k < k1; ++k)
                                block_scores[static_cast<size_t>(k - k0)] = -std::numeric_limits<float>::infinity();

                            const auto qk_start = profiling_enabled
                                                      ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point();

                            // --- QK Phase: VNNI int16 dot products ---
                            for (int k = valid_start; k < valid_end;)
                            {
                                if (blocks_per_head == 1)
                                {
                                    // Fast path: 1 block per head
                                    if (k + 3 < valid_end)
                                    {
                                        const uint8_t *b0 = k_raw + q16_layout.blockIndex(k + 0, kv_h, 0) * block_bytes;
                                        const uint8_t *b1 = k_raw + q16_layout.blockIndex(k + 1, kv_h, 0) * block_bytes;
                                        const uint8_t *b2 = k_raw + q16_layout.blockIndex(k + 2, kv_h, 0) * block_bytes;
                                        const uint8_t *b3 = k_raw + q16_layout.blockIndex(k + 3, kv_h, 0) * block_bytes;

                                        if (k + 7 < valid_end)
                                        {
                                            _mm_prefetch(reinterpret_cast<const char *>(
                                                             k_raw + q16_layout.blockIndex(k + 4, kv_h, 0) * block_bytes),
                                                         _MM_HINT_T0);
                                            _mm_prefetch(reinterpret_cast<const char *>(
                                                             k_raw + q16_layout.blockIndex(k + 5, kv_h, 0) * block_bytes),
                                                         _MM_HINT_T0);
                                            _mm_prefetch(reinterpret_cast<const char *>(
                                                             k_raw + q16_layout.blockIndex(k + 6, kv_h, 0) * block_bytes),
                                                         _MM_HINT_T0);
                                            _mm_prefetch(reinterpret_cast<const char *>(
                                                             k_raw + q16_layout.blockIndex(k + 7, kv_h, 0) * block_bytes),
                                                         _MM_HINT_T0);
                                        }

                                        float kd0, kd1, kd2, kd3;
                                        std::memcpy(&kd0, b0, sizeof(float));
                                        std::memcpy(&kd1, b1, sizeof(float));
                                        std::memcpy(&kd2, b2, sizeof(float));
                                        std::memcpy(&kd3, b3, sizeof(float));

                                        const int16_t *k_qs0 = reinterpret_cast<const int16_t *>(b0 + QS_OFFSET);
                                        const int16_t *k_qs1 = reinterpret_cast<const int16_t *>(b1 + QS_OFFSET);
                                        const int16_t *k_qs2 = reinterpret_cast<const int16_t *>(b2 + QS_OFFSET);
                                        const int16_t *k_qs3 = reinterpret_cast<const int16_t *>(b3 + QS_OFFSET);

                                        int32_t dot0, dot1, dot2, dot3;
                                        dot_i16_i16_i32_vnni_4row(
                                            q_i16_buf, k_qs0, k_qs1, k_qs2, k_qs3,
                                            head_dim, dot0, dot1, dot2, dot3);

                                        float s0 = static_cast<float>(dot0) * qk_combined_scale * kd0;
                                        float s1 = static_cast<float>(dot1) * qk_combined_scale * kd1;
                                        float s2 = static_cast<float>(dot2) * qk_combined_scale * kd2;
                                        float s3 = static_cast<float>(dot3) * qk_combined_scale * kd3;

                                        block_scores[static_cast<size_t>(k - k0 + 0)] = s0;
                                        block_scores[static_cast<size_t>(k - k0 + 1)] = s1;
                                        block_scores[static_cast<size_t>(k - k0 + 2)] = s2;
                                        block_scores[static_cast<size_t>(k - k0 + 3)] = s3;
                                        block_max = std::max(block_max, std::max(std::max(s0, s1), std::max(s2, s3)));
                                        k += 4;
                                        continue;
                                    }

                                    // Scalar tail
                                    const uint8_t *blk_s = k_raw + q16_layout.blockIndex(k, kv_h, 0) * block_bytes;
                                    float kd;
                                    std::memcpy(&kd, blk_s, sizeof(float));
                                    const int16_t *k_qs = reinterpret_cast<const int16_t *>(blk_s + QS_OFFSET);
                                    int32_t dot = dot_i16_i16_i32(q_i16_buf, k_qs, head_dim);
                                    float s = static_cast<float>(dot) * qk_combined_scale * kd;
                                    block_scores[static_cast<size_t>(k - k0)] = s;
                                    block_max = std::max(block_max, s);
                                    ++k;
                                }
                                else
                                {
                                    // Multi-block-per-head fallback
                                    float s = 0.0f;
                                    for (size_t bi = 0; bi < blocks_per_head; ++bi)
                                    {
                                        const size_t blk_idx = q16_layout.blockIndex(k, kv_h, bi);
                                        const uint8_t *blk_ptr = k_raw + blk_idx * block_bytes;
                                        float kd;
                                        std::memcpy(&kd, blk_ptr, sizeof(float));
                                        const int16_t *k_qs = reinterpret_cast<const int16_t *>(blk_ptr + QS_OFFSET);
                                        const int elem_count = static_cast<int>(
                                            std::min(block_elems, static_cast<size_t>(head_dim) - bi * block_elems));
                                        int32_t dot = dot_i16_i16_i32(
                                            q_i16_buf + bi * block_elems, k_qs, elem_count);
                                        s += static_cast<float>(dot) * q_scale * kd;
                                    }
                                    s *= scale;
                                    block_scores[static_cast<size_t>(k - k0)] = s;
                                    block_max = std::max(block_max, s);
                                    ++k;
                                }
                            }

                            if (profiling_enabled)
                                qk_duration_ns += static_cast<uint64_t>(
                                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - qk_start)
                                        .count());

                            // --- Online softmax correction ---
                            const float new_m = std::max(running_m, block_max);
                            const float alpha = std::isfinite(running_m)
                                                    ? std::exp(running_m - new_m)
                                                    : 0.0f;
                            {
#if defined(__AVX512F__)
                                const __m512 va = _mm512_set1_ps(alpha);
                                int sd = 0;
                                for (; sd + 15 < head_dim; sd += 16)
                                    _mm512_storeu_ps(out + sd, _mm512_mul_ps(va, _mm512_loadu_ps(out + sd)));
                                for (; sd < head_dim; ++sd)
                                    out[sd] *= alpha;
#else
                                const __m256 va = _mm256_set1_ps(alpha);
                                int sd = 0;
                                for (; sd + 7 < head_dim; sd += 8)
                                    _mm256_storeu_ps(out + sd, _mm256_mul_ps(va, _mm256_loadu_ps(out + sd)));
                                for (; sd < head_dim; ++sd)
                                    out[sd] *= alpha;
#endif
                            }
                            float new_l = running_l * alpha;

                            // --- V Phase: Q16_1 V weighted accumulation ---
                            const auto v_start = profiling_enabled
                                                     ? std::chrono::steady_clock::now()
                                                     : std::chrono::steady_clock::time_point();

                            if (blocks_per_head == 1)
                            {
                                int k = valid_start;
                                for (; k + 3 < valid_end; k += 4)
                                {
                                    if (k + 7 < valid_end)
                                    {
                                        _mm_prefetch(reinterpret_cast<const char *>(
                                                         v_raw + q16_layout.blockIndex(k + 4, kv_h, 0) * block_bytes),
                                                     _MM_HINT_T0);
                                        _mm_prefetch(reinterpret_cast<const char *>(
                                                         v_raw + q16_layout.blockIndex(k + 5, kv_h, 0) * block_bytes),
                                                     _MM_HINT_T0);
                                        _mm_prefetch(reinterpret_cast<const char *>(
                                                         v_raw + q16_layout.blockIndex(k + 6, kv_h, 0) * block_bytes),
                                                     _MM_HINT_T0);
                                        _mm_prefetch(reinterpret_cast<const char *>(
                                                         v_raw + q16_layout.blockIndex(k + 7, kv_h, 0) * block_bytes),
                                                     _MM_HINT_T0);
                                    }

                                    alignas(16) float pp[4];
                                    batch_exp_4(
                                        block_scores[static_cast<size_t>(k - k0 + 0)],
                                        block_scores[static_cast<size_t>(k - k0 + 1)],
                                        block_scores[static_cast<size_t>(k - k0 + 2)],
                                        block_scores[static_cast<size_t>(k - k0 + 3)],
                                        new_m,
                                        pp[0], pp[1], pp[2], pp[3]);
                                    new_l += pp[0] + pp[1] + pp[2] + pp[3];

                                    const uint8_t *vb0 = v_raw + q16_layout.blockIndex(k + 0, kv_h, 0) * block_bytes;
                                    const uint8_t *vb1 = v_raw + q16_layout.blockIndex(k + 1, kv_h, 0) * block_bytes;
                                    const uint8_t *vb2 = v_raw + q16_layout.blockIndex(k + 2, kv_h, 0) * block_bytes;
                                    const uint8_t *vb3 = v_raw + q16_layout.blockIndex(k + 3, kv_h, 0) * block_bytes;

                                    float vd0, vd1, vd2, vd3;
                                    std::memcpy(&vd0, vb0, sizeof(float));
                                    std::memcpy(&vd1, vb1, sizeof(float));
                                    std::memcpy(&vd2, vb2, sizeof(float));
                                    std::memcpy(&vd3, vb3, sizeof(float));

                                    accum_weighted_v_q16_4row(
                                        out,
                                        reinterpret_cast<const int16_t *>(vb0 + QS_OFFSET), pp[0] * vd0,
                                        reinterpret_cast<const int16_t *>(vb1 + QS_OFFSET), pp[1] * vd1,
                                        reinterpret_cast<const int16_t *>(vb2 + QS_OFFSET), pp[2] * vd2,
                                        reinterpret_cast<const int16_t *>(vb3 + QS_OFFSET), pp[3] * vd3,
                                        head_dim);
                                }

                                // Scalar tail
                                for (; k < valid_end; ++k)
                                {
                                    const float p = std::exp(block_scores[static_cast<size_t>(k - k0)] - new_m);
                                    new_l += p;
                                    const uint8_t *blk_ptr = v_raw + q16_layout.blockIndex(k, kv_h, 0) * block_bytes;
                                    float vd;
                                    std::memcpy(&vd, blk_ptr, sizeof(float));
                                    accum_weighted_v_q16(out,
                                                         reinterpret_cast<const int16_t *>(blk_ptr + QS_OFFSET),
                                                         p * vd, head_dim);
                                }
                            }
                            else
                            {
                                // Native multi-block implementation.
                                for (int k = valid_start; k < valid_end; ++k)
                                {
                                    const float p = std::exp(block_scores[static_cast<size_t>(k - k0)] - new_m);
                                    new_l += p;
                                    for (size_t bi = 0; bi < blocks_per_head; ++bi)
                                    {
                                        const size_t blk_idx = q16_layout.blockIndex(k, kv_h, bi);
                                        const uint8_t *blk_ptr = v_raw + blk_idx * block_bytes;
                                        float vd;
                                        std::memcpy(&vd, blk_ptr, sizeof(float));
                                        const int16_t *v_qs = reinterpret_cast<const int16_t *>(blk_ptr + QS_OFFSET);
                                        const int elem_count = static_cast<int>(
                                            std::min(block_elems, static_cast<size_t>(head_dim) - bi * block_elems));
                                        accum_weighted_v_q16(
                                            out + bi * block_elems, v_qs, p * vd, elem_count);
                                    }
                                }
                            }

                            if (profiling_enabled)
                                v_duration_ns += static_cast<uint64_t>(
                                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - v_start)
                                        .count());

                            running_m = new_m;
                            running_l = new_l;
                        } // end KV tile loop

                        // Final normalisation
                        if (running_l > 0.0f)
                        {
                            const float inv_l = 1.0f / running_l;
#if defined(__AVX512F__)
                            const __m512 vi = _mm512_set1_ps(inv_l);
                            int sd = 0;
                            for (; sd + 15 < head_dim; sd += 16)
                                _mm512_storeu_ps(out + sd, _mm512_mul_ps(vi, _mm512_loadu_ps(out + sd)));
                            for (; sd < head_dim; ++sd)
                                out[sd] *= inv_l;
#else
                            const __m256 vi = _mm256_set1_ps(inv_l);
                            int sd = 0;
                            for (; sd + 7 < head_dim; sd += 8)
                                _mm256_storeu_ps(out + sd, _mm256_mul_ps(vi, _mm256_loadu_ps(out + sd)));
                            for (; sd < head_dim; ++sd)
                                out[sd] *= inv_l;
#endif
                        }
                    } // end q_pos loop
                } // end head loop
            };

            // Threading: prefill has enough work for all threads
            OMP_WORKSHARE_REGION(work);
            const int actual_threads = omp_get_max_threads();

            if (profiling_enabled)
            {
                KernelProfiler::recordParallel(KernelType::ATTENTION_QK, qk_duration_ns, actual_threads);
                KernelProfiler::recordParallel(KernelType::ATTENTION_V, v_duration_ns, actual_threads);
            }
            return true;
        }
#endif // (__AVX512F__ && __AVX512VNNI__) || __AVX2__


        // =================================================================
        // Q8_1 inline-dequant helpers for fused attention
        // =================================================================

        /**
         * @brief Inline Q8_1 dot product: dot(Q_fp32, dequant(K_q8_1_block))
         *
         * For each of the 32 int8 values in K_block: float_k = qs[i] * scale
         * Then computes dot(Q_slice, float_k) using AVX-512.
         *
         * @param q_fp32  Pointer to 32 FP32 Q elements (aligned to current block offset)
         * @param k_block Pointer to Q8_1Block
         * @return FP32 partial dot product for this block
         */
        static inline float dot_q_fp32_k_q8_1_block(
            const float *q_fp32, const Q8_1Block *k_block)
        {
#if defined(__AVX512F__) && defined(__F16C__)
            // Scale: uint16_t d is FP16-encoded
            const float k_scale = _cvtsh_ss(k_block->d);
            const __m512 vscale = _mm512_set1_ps(k_scale);

            // Load 32 int8 → 2×16 int32 → 2×16 FP32
            const __m256i qs_raw = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(k_block->qs));
            // Lower 16 bytes → 16×int32
            const __m512i lo_i32 = _mm512_cvtepi8_epi32(_mm256_castsi256_si128(qs_raw));
            // Upper 16 bytes → 16×int32
            const __m512i hi_i32 = _mm512_cvtepi8_epi32(_mm256_extracti128_si256(qs_raw, 1));

            const __m512 lo_f = _mm512_mul_ps(_mm512_cvtepi32_ps(lo_i32), vscale);
            const __m512 hi_f = _mm512_mul_ps(_mm512_cvtepi32_ps(hi_i32), vscale);

            // Dot product with Q
            const __m512 q0 = _mm512_loadu_ps(q_fp32);
            const __m512 q1 = _mm512_loadu_ps(q_fp32 + 16);
            const __m512 prod0 = _mm512_mul_ps(q0, lo_f);
            const __m512 prod1 = _mm512_fmadd_ps(q1, hi_f, prod0);

            return _mm512_reduce_add_ps(prod1);
#else
            const float k_scale = fp16_to_fp32(k_block->d);
            float dot = 0.0f;
            for (int i = 0; i < 32; ++i)
                dot += q_fp32[i] * (static_cast<float>(k_block->qs[i]) * k_scale);
            return dot;
#endif
        }

        /**
         * @brief Accumulate weighted dequantized Q8_1 V block into output.
         *
         * out[d] += weight * (qs[d] * scale)  for d in [0, 32)
         *
         * @param out     Output accumulator (32 floats at current block offset)
         * @param v_block Pointer to Q8_1Block for V
         * @param weight  Softmax probability × block scale already combined? No — just prob
         */
        static inline void accum_weighted_v_q8_1_block(
            float *out, const Q8_1Block *v_block, float weight)
        {
#if defined(__AVX512F__) && defined(__F16C__)
            const float v_scale = _cvtsh_ss(v_block->d);
            const float combined = weight * v_scale;
            const __m512 vw = _mm512_set1_ps(combined);

            const __m256i qs_raw = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(v_block->qs));
            const __m512i lo_i32 = _mm512_cvtepi8_epi32(_mm256_castsi256_si128(qs_raw));
            const __m512i hi_i32 = _mm512_cvtepi8_epi32(_mm256_extracti128_si256(qs_raw, 1));

            const __m512 lo_f = _mm512_cvtepi32_ps(lo_i32);
            const __m512 hi_f = _mm512_cvtepi32_ps(hi_i32);

            __m512 o0 = _mm512_loadu_ps(out);
            __m512 o1 = _mm512_loadu_ps(out + 16);
            o0 = _mm512_fmadd_ps(lo_f, vw, o0);
            o1 = _mm512_fmadd_ps(hi_f, vw, o1);
            _mm512_storeu_ps(out, o0);
            _mm512_storeu_ps(out + 16, o1);
#else
            const float v_scale = fp16_to_fp32(v_block->d);
            const float combined = weight * v_scale;
            for (int i = 0; i < 32; ++i)
                out[i] += combined * static_cast<float>(v_block->qs[i]);
#endif
        }

        /**
         * @brief Native Q8_1 flash attention for every positive query-row count.
         *
         * Reads K and V directly as Q8_1 blocks, performing inline int8→float
         * dequantization in the dot product and V accumulation inner loops.
         * Eliminates FP32 shadow buffers (~440 MB for 7B models with 8K context).
         *
         * Layout support: POSITION_MAJOR [pos][n_kv_heads * blocks_per_head]
         * and HEAD_MAJOR [head][pos][blocks_per_head].
         */
        bool compute_q8kv(
            const float *Q,
            const Q8_1Tensor *K_q8, const Q8_1Tensor *V_q8,
            float *output,
            int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset,
            int head_start = 0,
            int gqa_n_rep = 0,
            const attention::AttentionExecutionPolicy &execution_policy = {},
            const attention::AttentionKVLogicalView &kv_logical_view = {})
        {
            KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);

            if (!Q || !K_q8 || !V_q8 || !output)
                return false;
            if (seq_len <= 0 || kv_len <= 0 || n_heads <= 0 ||
                n_kv_heads <= 0 || head_dim <= 0)
                return false;
            if (gqa_n_rep <= 0 && n_heads % n_kv_heads != 0)
                return false;
            if (!kv_logical_view.validFor(kv_len))
                return false;

            const auto physical_kv_row = [&](int logical_row) -> std::size_t
            {
                return static_cast<std::size_t>(
                    kv_logical_view.physicalRow(logical_row));
            };

            const int heads_per_kv = (gqa_n_rep > 0) ? gqa_n_rep : (n_heads / n_kv_heads);
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            constexpr size_t BLOCK_SIZE = Q8_1Block::BLOCK_SIZE; // 32
            const size_t blocks_per_head = (static_cast<size_t>(head_dim) + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // Detect layout from K tensor shape
            const Q8_1Block *k_blocks = K_q8->typed_data();
            const Q8_1Block *v_blocks = V_q8->typed_data();
            if (!k_blocks || !v_blocks)
                return false;

            // blocks_per_row for the K tensor: cols / BLOCK_SIZE
            const size_t k_cols = K_q8->shape().size() > 1 ? K_q8->shape()[1] : K_q8->shape()[0];
            const size_t blocks_per_kv_row = (k_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // HEAD_MAJOR: blocks_per_row == blocks_per_head (one head per row)
            // POSITION_MAJOR: blocks_per_row == n_kv_heads * blocks_per_head
            const bool is_head_major = (blocks_per_kv_row == blocks_per_head && n_kv_heads > 1);
            const size_t rows_per_head = is_head_major
                                             ? (K_q8->shape()[0] / static_cast<size_t>(n_kv_heads))
                                             : 0;
            const std::size_t physical_rows = is_head_major
                                                  ? rows_per_head
                                                  : K_q8->shape()[0];
            if (!kv_logical_view.isContiguous() &&
                static_cast<std::size_t>(
                    kv_logical_view.physical_row_capacity) != physical_rows)
            {
                return false;
            }

            const std::size_t q8_head_row_bytes =
                blocks_per_head * sizeof(Q8_1Block);
            const int kv_tile = detail::selectCPUFlashKVTile(
                cpu::fa2_policy::CPUFA2KVStoragePair::Q8_1,
                head_dim,
                kv_len,
                q8_head_row_bytes,
                q8_head_row_bytes,
                launch_policy_.explicit_kv_tile);

            /** Per-worker immutable addresses for one Q8 attention row. */
            struct Q8RowScratch
            {
                const float *query = nullptr;
                std::size_t head_base = 0;
            };

            const auto prepare_row = [&](int row, Q8RowScratch &scratch)
            {
                const int query_position = row / n_heads;
                const int h = row % n_heads;
                const int kv_h = (gqa_n_rep > 0)
                                     ? (head_start + h) / heads_per_kv
                                     : h / heads_per_kv;
                scratch.query =
                    Q + static_cast<std::size_t>(query_position) *
                            n_heads * head_dim +
                    static_cast<std::size_t>(h) * head_dim;
                scratch.head_base = is_head_major
                                        ? static_cast<std::size_t>(kv_h) *
                                              rows_per_head * blocks_per_head
                                        : static_cast<std::size_t>(kv_h) *
                                              blocks_per_head;
            };

            const auto visible_tile = [&](int row,
                                          int tile_begin,
                                          int tile_end)
            {
                const int absolute_position =
                    position_offset + row / n_heads;
                int visible_begin = tile_begin;
                if (window_size > 0)
                {
                    visible_begin = std::max(
                        visible_begin,
                        absolute_position - window_size + 1);
                }
                const int visible_end = causal
                                            ? std::min(
                                                  tile_end,
                                                  absolute_position + 1)
                                            : tile_end;
                return CPUFA2VisibleTile{
                    .begin = std::max(
                        tile_begin,
                        std::min(visible_begin, tile_end)),
                    .end = std::max(
                        tile_begin,
                        std::min(visible_end, tile_end)),
                };
            };

            const auto score_tile = [&](int,
                                        const Q8RowScratch &scratch,
                                        int tile_begin,
                                        CPUFA2VisibleTile visible,
                                        float *scores,
                                        float &block_max)
            {
                for (int kv_row = visible.begin;
                     kv_row < visible.end;
                     ++kv_row)
                {
                    float dot = 0.0f;
                    for (std::size_t block = 0;
                         block < blocks_per_head;
                         ++block)
                    {
                        const std::size_t block_index = is_head_major
                                                            ? scratch.head_base +
                                                                  physical_kv_row(kv_row) *
                                                                      blocks_per_head +
                                                                  block
                                                            : physical_kv_row(kv_row) *
                                                                      blocks_per_kv_row +
                                                                  scratch.head_base +
                                                                  block;
                        dot += dot_q_fp32_k_q8_1_block(
                            scratch.query + block * BLOCK_SIZE,
                            &k_blocks[block_index]);
                    }
                    const float score = dot * scale;
                    scores[static_cast<std::size_t>(kv_row - tile_begin)] =
                        score;
                    block_max = std::max(block_max, score);

                    if (kv_row + 1 < visible.end)
                    {
                        const std::size_t next_index = is_head_major
                                                           ? scratch.head_base +
                                                                 physical_kv_row(kv_row + 1) *
                                                                     blocks_per_head
                                                           : physical_kv_row(kv_row + 1) *
                                                                     blocks_per_kv_row +
                                                                 scratch.head_base;
                        _mm_prefetch(
                            reinterpret_cast<const char *>(
                                &k_blocks[next_index]),
                            _MM_HINT_T0);
                    }
                }
            };

            const auto accumulate_tile = [&](int,
                                             const Q8RowScratch &scratch,
                                             int tile_begin,
                                             CPUFA2VisibleTile visible,
                                             const float *scores,
                                             float new_m,
                                             float *partial,
                                             float &new_l)
            {
                for (int kv_row = visible.begin;
                     kv_row < visible.end;
                     ++kv_row)
                {
                    const float probability = std::exp(
                        scores[static_cast<std::size_t>(kv_row - tile_begin)] -
                        new_m);
                    new_l += probability;
                    for (std::size_t block = 0;
                         block < blocks_per_head;
                         ++block)
                    {
                        const std::size_t block_index = is_head_major
                                                            ? scratch.head_base +
                                                                  physical_kv_row(kv_row) *
                                                                      blocks_per_head +
                                                                  block
                                                            : physical_kv_row(kv_row) *
                                                                      blocks_per_kv_row +
                                                                  scratch.head_base +
                                                                  block;
                        accum_weighted_v_q8_1_block(
                            partial + block * BLOCK_SIZE,
                            &v_blocks[block_index],
                            probability);
                    }

                    if (kv_row + 1 < visible.end)
                    {
                        const std::size_t next_index = is_head_major
                                                           ? scratch.head_base +
                                                                 physical_kv_row(kv_row + 1) *
                                                                     blocks_per_head
                                                           : physical_kv_row(kv_row + 1) *
                                                                     blocks_per_kv_row +
                                                                 scratch.head_base;
                        _mm_prefetch(
                            reinterpret_cast<const char *>(
                                &v_blocks[next_index]),
                            _MM_HINT_T0);
                    }
                }
            };

            const auto finalize_row = [&](int row,
                                          float *merged,
                                          float merged_l)
            {
                if (merged_l > 0.0f)
                {
                    scale_vec(
                        merged,
                        1.0f / merged_l,
                        head_dim,
                        activeISALevel() == ISALevel::AVX512);
                }
                std::memcpy(
                    output + static_cast<std::size_t>(row / n_heads) *
                                     n_heads * head_dim +
                        static_cast<std::size_t>(row % n_heads) * head_dim,
                    merged,
                    static_cast<std::size_t>(head_dim) * sizeof(float));
            };

            return executePartitionedAttention<Q8RowScratch>(
                seq_len,
                n_heads,
                kv_len,
                head_dim,
                kv_tile,
                causal,
                execution_policy,
                prepare_row,
                visible_tile,
                score_tile,
                accumulate_tile,
                finalize_row);
        }

        /**
         * @brief Native FP16/BF16 attention for every positive query-row count.
         *
         * Reads K and V directly in their two-byte storage format and converts
         * each vector register in place. The direct path is available to AVX2,
         * AVX-512, and scalar builds; no ISA configuration is permitted to
         * materialize a persistent FP32 cache as an execution substitute.
         *
         * @tparam Format Native FP16 or BF16 bit interpretation.
         */
        template <Native16BitKVFormat Format>
        bool compute_native16kv(
            const float *Q,
            const uint16_t *K_native, const uint16_t *V_native,
            float *output,
            int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset,
            int head_start = 0,
            int gqa_n_rep = 0,
            const attention::AttentionExecutionPolicy &execution_policy = {},
            const attention::AttentionKVLogicalView &kv_logical_view = {})
        {
            KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);

            if (!Q || !K_native || !V_native || !output)
                return false;
            if (seq_len <= 0 || kv_len <= 0 || n_heads <= 0 ||
                n_kv_heads <= 0 || head_dim <= 0)
                return false;
            if (gqa_n_rep <= 0 && n_heads % n_kv_heads != 0)
                return false;
            if (!kv_logical_view.validFor(kv_len))
                return false;

            const auto physical_kv_row = [&](int logical_row) -> std::size_t
            {
                return static_cast<std::size_t>(
                    kv_logical_view.physicalRow(logical_row));
            };

            const int heads_per_kv = (gqa_n_rep > 0) ? gqa_n_rep : (n_heads / n_kv_heads);
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            const int q_stride = n_heads * head_dim;
            const int kv_stride = n_kv_heads * head_dim;

            const std::size_t native_head_row_bytes =
                static_cast<std::size_t>(head_dim) * sizeof(std::uint16_t);
            constexpr auto storage_pair =
                Format == Native16BitKVFormat::FP16
                    ? cpu::fa2_policy::CPUFA2KVStoragePair::FP16
                    : cpu::fa2_policy::CPUFA2KVStoragePair::BF16;
            const int kv_tile = detail::selectCPUFlashKVTile(
                storage_pair,
                head_dim,
                kv_len,
                native_head_row_bytes,
                native_head_row_bytes,
                launch_policy_.explicit_kv_tile);

            /** Per-worker immutable addresses for one native 16-bit row. */
            struct Native16BitRowScratch
            {
                const float *query = nullptr;
                const std::uint16_t *key = nullptr;
                const std::uint16_t *value = nullptr;
            };

            const auto prepare_row = [&](int row,
                                         Native16BitRowScratch &scratch)
            {
                const int query_position = row / n_heads;
                const int h = row % n_heads;
                const int kv_h = (gqa_n_rep > 0)
                                     ? (head_start + h) / heads_per_kv
                                     : h / heads_per_kv;
                scratch.query =
                    Q + static_cast<std::size_t>(query_position) * q_stride +
                    static_cast<std::size_t>(h) * head_dim;
                scratch.key = K_native + static_cast<std::size_t>(kv_h) * head_dim;
                scratch.value = V_native + static_cast<std::size_t>(kv_h) * head_dim;
            };

            const auto visible_tile = [&](int row,
                                          int tile_begin,
                                          int tile_end)
            {
                const int absolute_position =
                    position_offset + row / n_heads;
                int visible_begin = tile_begin;
                if (window_size > 0)
                {
                    visible_begin = std::max(
                        visible_begin,
                        absolute_position - window_size + 1);
                }
                const int visible_end = causal
                                            ? std::min(
                                                  tile_end,
                                                  absolute_position + 1)
                                            : tile_end;
                return CPUFA2VisibleTile{
                    .begin = std::max(
                        tile_begin,
                        std::min(visible_begin, tile_end)),
                    .end = std::max(
                        tile_begin,
                        std::min(visible_end, tile_end)),
                };
            };

            const auto score_tile = [&](int,
                                        const Native16BitRowScratch &scratch,
                                        int tile_begin,
                                        CPUFA2VisibleTile visible,
                                        float *scores,
                                        float &block_max)
            {
                int kv_row = visible.begin;
                for (; kv_row + 3 < visible.end; kv_row += 4)
                {
                    float score0 = 0.0f;
                    float score1 = 0.0f;
                    float score2 = 0.0f;
                    float score3 = 0.0f;
                    dot_native16_4row<Format>(
                        scratch.query,
                        scratch.key + physical_kv_row(kv_row + 0) * kv_stride,
                        scratch.key + physical_kv_row(kv_row + 1) * kv_stride,
                        scratch.key + physical_kv_row(kv_row + 2) * kv_stride,
                        scratch.key + physical_kv_row(kv_row + 3) * kv_stride,
                        head_dim,
                        score0,
                        score1,
                        score2,
                        score3);
                    score0 *= scale;
                    score1 *= scale;
                    score2 *= scale;
                    score3 *= scale;
                    scores[static_cast<std::size_t>(kv_row - tile_begin + 0)] = score0;
                    scores[static_cast<std::size_t>(kv_row - tile_begin + 1)] = score1;
                    scores[static_cast<std::size_t>(kv_row - tile_begin + 2)] = score2;
                    scores[static_cast<std::size_t>(kv_row - tile_begin + 3)] = score3;
                    block_max = std::max(
                        block_max,
                        std::max(
                            std::max(score0, score1),
                            std::max(score2, score3)));
                }
                for (; kv_row < visible.end; ++kv_row)
                {
                    const float score = dot_native16<Format>(
                                            scratch.query,
                                            scratch.key +
                                                physical_kv_row(kv_row) * kv_stride,
                                            head_dim) *
                                        scale;
                    scores[static_cast<std::size_t>(kv_row - tile_begin)] = score;
                    block_max = std::max(block_max, score);
                }
            };

            const auto accumulate_tile = [&](int,
                                             const Native16BitRowScratch &scratch,
                                             int tile_begin,
                                             CPUFA2VisibleTile visible,
                                             const float *scores,
                                             float new_m,
                                             float *partial,
                                             float &new_l)
            {
                int kv_row = visible.begin;
                for (; kv_row + 3 < visible.end; kv_row += 4)
                {
                    float probability0 = 0.0f;
                    float probability1 = 0.0f;
                    float probability2 = 0.0f;
                    float probability3 = 0.0f;
                    batch_exp_4(
                        scores[static_cast<std::size_t>(kv_row - tile_begin + 0)],
                        scores[static_cast<std::size_t>(kv_row - tile_begin + 1)],
                        scores[static_cast<std::size_t>(kv_row - tile_begin + 2)],
                        scores[static_cast<std::size_t>(kv_row - tile_begin + 3)],
                        new_m,
                        probability0,
                        probability1,
                        probability2,
                        probability3);
                    new_l += probability0 + probability1 +
                             probability2 + probability3;
                    accum_native16_4row<Format>(
                        partial,
                        scratch.value + physical_kv_row(kv_row + 0) * kv_stride,
                        probability0,
                        scratch.value + physical_kv_row(kv_row + 1) * kv_stride,
                        probability1,
                        scratch.value + physical_kv_row(kv_row + 2) * kv_stride,
                        probability2,
                        scratch.value + physical_kv_row(kv_row + 3) * kv_stride,
                        probability3,
                        head_dim);
                }
                for (; kv_row < visible.end; ++kv_row)
                {
                    const float probability = std::exp(
                        scores[static_cast<std::size_t>(kv_row - tile_begin)] -
                        new_m);
                    new_l += probability;
                    accum_native16<Format>(
                        partial,
                        scratch.value +
                            physical_kv_row(kv_row) * kv_stride,
                        probability,
                        head_dim);
                }
            };

            const auto finalize_row = [&](int row,
                                          float *merged,
                                          float merged_l)
            {
                if (merged_l > 0.0f)
                {
                    scale_vec(
                        merged,
                        1.0f / merged_l,
                        head_dim,
                        activeISALevel() == ISALevel::AVX512);
                }
                std::memcpy(
                    output + static_cast<std::size_t>(row / n_heads) *
                                     q_stride +
                        static_cast<std::size_t>(row % n_heads) * head_dim,
                    merged,
                    static_cast<std::size_t>(head_dim) * sizeof(float));
            };

            return executePartitionedAttention<Native16BitRowScratch>(
                seq_len,
                n_heads,
                kv_len,
                head_dim,
                kv_tile,
                causal,
                execution_policy,
                prepare_row,
                visible_tile,
                score_tile,
                accumulate_tile,
                finalize_row);
        }

        // =================================================================
        // TurboQuant fused native attention (zero shadow buffers)
        //
        // Exploits the orthogonality of the TQ rotation matrix:
        //   dot(Q, dequant(K)) = (reconstruction_norm/√D) ·
        //                           dot(Π·Q, centroids(K))
        // Pre-rotates Q once per head [O(D²)], then per KV position [O(D)].
        //
        // V accumulation in rotated centroid space, one final Πᵀ at end.
        // Total: O(D² + kv_len·D) instead of O(kv_len·D²).
        // =================================================================
#if defined(__AVX512F__) || defined(__AVX2__)
        /**
         * @brief Native TQ4/TQ8 attention for every supported K/V pairing.
         *
         * `KeyTensor` selects the direct key-score codec and `ValueTensor`
         * selects the direct value-accumulation codec. Online-softmax
         * partitioning and reduction are identical for every combination, so
         * adding a storage mode cannot change grouped verifier ordering.
         *
         * @param Q          Query tensor data [seq_len × n_heads × head_dim], FP32
         * @param K_tq       TQ4 or TQ8 K cache tensor (POSITION_MAJOR layout)
         * @param V_tq       TQ4 or TQ8 V cache tensor (POSITION_MAJOR layout)
         * @param output     Output tensor [seq_len × n_heads × head_dim], FP32
         * @param seq_len    Number of logical query rows
         * @param kv_len     Number of cached KV positions
         * @param n_heads    Number of query heads
         * @param n_kv_heads Number of KV heads (GQA: n_heads / n_kv_heads is group size)
         * @param head_dim   Head dimension (64 or 128)
         * @param causal     Whether to apply causal masking
         * @param position_offset Offset for causal masking (typically kv_len - 1 for decode)
         * @return true on success
         */
        template <typename KeyTensor, typename ValueTensor>
        bool compute_tqkv(
            const float *Q,
            const KeyTensor *K_tq,
            const ValueTensor *V_tq,
            float *output,
            int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset,
            int head_start = 0,
            int gqa_n_rep = 0,
            const attention::AttentionExecutionPolicy &execution_policy = {},
            const attention::AttentionKVLogicalView &kv_logical_view = {})
        {
            static_assert(
                std::is_same_v<KeyTensor, TQ4Tensor> ||
                    std::is_same_v<KeyTensor, TQ8Tensor>,
                "TurboQuant attention supports only native TQ4 or TQ8 K");
            static_assert(
                std::is_same_v<ValueTensor, TQ4Tensor> ||
                    std::is_same_v<ValueTensor, TQ8Tensor>,
                "TurboQuant attention supports only native TQ4 or TQ8 V");
            KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);

            if (!Q || !K_tq || !V_tq || !output)
                return false;
            if (seq_len <= 0 || kv_len <= 0 || n_heads <= 0 ||
                n_kv_heads <= 0 || head_dim <= 0)
                return false;
            if (gqa_n_rep <= 0 && n_heads % n_kv_heads != 0)
                return false;
            if (!kv_logical_view.validFor(kv_len))
                return false;

            const auto physical_kv_row = [&](int logical_row) -> std::size_t
            {
                return static_cast<std::size_t>(
                    kv_logical_view.physicalRow(logical_row));
            };

            // Get TQ context (per-layer, set by the stage).
            // Each KV head uses a DIFFERENT derived rotation: ctx.for_layer(kv_h).
            const TurboQuantContext *tq_ctx = K_tq->turboquant_context();
            if (!tq_ctx || V_tq->turboquant_context() != tq_ctx)
                return false;
            if (K_tq->head_dim() != head_dim ||
                V_tq->head_dim() != head_dim ||
                K_tq->blocks_per_row() <
                    static_cast<std::size_t>(n_kv_heads) ||
                V_tq->blocks_per_row() <
                    static_cast<std::size_t>(n_kv_heads))
                return false;

            const int heads_per_kv = (gqa_n_rep > 0) ? gqa_n_rep : (n_heads / n_kv_heads);
            // Combined scale: (1/√D) from TQ dequant descale × (1/√D) from attention = 1/D
            const float combined_scale = 1.0f / static_cast<float>(head_dim);

            // Raw byte access for position-major TQ layout
            const uint8_t *k_raw = K_tq->typed_data();
            const uint8_t *v_raw = V_tq->typed_data();
            if (!k_raw || !v_raw)
                return false;

            const size_t k_block_bytes = K_tq->block_bytes();
            const size_t v_block_bytes = V_tq->block_bytes();
            const size_t k_blocks_per_row = K_tq->blocks_per_row();
            const size_t v_blocks_per_row = V_tq->blocks_per_row();
            const size_t k_row_stride = k_blocks_per_row * k_block_bytes;
            const size_t v_row_stride = v_blocks_per_row * v_block_bytes;

            constexpr auto storage_pair =
                std::is_same_v<KeyTensor, TQ4Tensor>
                    ? (std::is_same_v<ValueTensor, TQ4Tensor>
                           ? cpu::fa2_policy::CPUFA2KVStoragePair::TQ4_TQ4
                           : cpu::fa2_policy::CPUFA2KVStoragePair::TQ4_TQ8)
                    : (std::is_same_v<ValueTensor, TQ4Tensor>
                           ? cpu::fa2_policy::CPUFA2KVStoragePair::TQ8_TQ4
                           : cpu::fa2_policy::CPUFA2KVStoragePair::TQ8_TQ8);
            const int kv_tile = detail::selectCPUFlashKVTile(
                storage_pair,
                head_dim,
                kv_len,
                k_block_bytes,
                v_block_bytes,
                launch_policy_.explicit_kv_tile);

            // V accumulation in rotated space uses 1/√D scaling:
            // rotated_accum accumulates weight × reconstruction_norm × centroids.
            // The 1/√D from TQ descale is deferred to the final inverse rotation step.
            const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));

            /*
             * Rotation is O(D^2), so performing it independently in every
             * context partition would erase the parallel-attention benefit.
             * The caller-owned output row is not observable until completion;
             * use it as the invocation-persistent rotated-Q buffer. All
             * partition producers read it before the reducer overwrites it.
             */
            auto rotate_queries = [&]()
            {
#pragma omp for schedule(static)
                for (int row = 0; row < seq_len * n_heads; ++row)
                {
                    const int query_position = row / n_heads;
                    const int h = row % n_heads;
                    const int kv_h = (gqa_n_rep > 0)
                                         ? (head_start + h) / heads_per_kv
                                         : h / heads_per_kv;
                    const TurboQuantRotation &rotation =
                        tq_ctx->for_layer(kv_h).rotation();
                    apply_rotation(
                        rotation,
                        Q + static_cast<std::size_t>(query_position) *
                                n_heads * head_dim +
                            static_cast<std::size_t>(h) * head_dim,
                        output + static_cast<std::size_t>(query_position) *
                                     n_heads * head_dim +
                            static_cast<std::size_t>(h) * head_dim);
                }
            };
            OMP_WORKSHARE_REGION(rotate_queries);

            /** Per-worker immutable addresses for one TurboQuant row. */
            struct TurboQuantRowScratch
            {
                const float *rotated_query = nullptr;
                std::size_t key_head_offset = 0;
                std::size_t value_head_offset = 0;
            };

            const auto prepare_row = [&](int row,
                                         TurboQuantRowScratch &scratch)
            {
                const int query_position = row / n_heads;
                const int h = row % n_heads;
                const int kv_h = (gqa_n_rep > 0)
                                     ? (head_start + h) / heads_per_kv
                                     : h / heads_per_kv;
                scratch.rotated_query =
                    output + static_cast<std::size_t>(query_position) *
                                 n_heads * head_dim +
                    static_cast<std::size_t>(h) * head_dim;
                scratch.key_head_offset =
                    static_cast<std::size_t>(kv_h) * k_block_bytes;
                scratch.value_head_offset =
                    static_cast<std::size_t>(kv_h) * v_block_bytes;
            };

            const auto visible_tile = [&](int row,
                                          int tile_begin,
                                          int tile_end)
            {
                const int absolute_position =
                    position_offset + row / n_heads;
                int visible_begin = tile_begin;
                if (window_size > 0)
                {
                    visible_begin = std::max(
                        visible_begin,
                        absolute_position - window_size + 1);
                }
                const int visible_end = causal
                                            ? std::min(
                                                  tile_end,
                                                  absolute_position + 1)
                                            : tile_end;
                return CPUFA2VisibleTile{
                    .begin = std::max(
                        tile_begin,
                        std::min(visible_begin, tile_end)),
                    .end = std::max(
                        tile_begin,
                        std::min(visible_end, tile_end)),
                };
            };

            const auto score_tile = [&](int,
                                        const TurboQuantRowScratch &scratch,
                                        int tile_begin,
                                        CPUFA2VisibleTile visible,
                                        float *scores,
                                        float &block_max)
            {
                for (int kv_row = visible.begin;
                     kv_row < visible.end;
                     ++kv_row)
                {
                    if (kv_row + 4 < visible.end)
                    {
                        _mm_prefetch(
                            reinterpret_cast<const char *>(
                                k_raw +
                                physical_kv_row(kv_row + 4) *
                                    k_row_stride +
                                scratch.key_head_offset),
                            _MM_HINT_T0);
                    }
                    const std::uint8_t *key_block =
                        k_raw + physical_kv_row(kv_row) * k_row_stride +
                        scratch.key_head_offset;
                    const float unscaled_score = [&]()
                    {
                        if constexpr (std::is_same_v<KeyTensor, TQ8Tensor>)
                        {
                            return tq8_dot_rotated_q(
                                scratch.rotated_query,
                                key_block,
                                head_dim);
                        }
                        else
                        {
                            return tq4_dot_rotated_q(
                                scratch.rotated_query,
                                key_block,
                                head_dim);
                        }
                    }();
                    const float score = unscaled_score * combined_scale;
                    scores[static_cast<std::size_t>(kv_row - tile_begin)] = score;
                    block_max = std::max(block_max, score);
                }
            };

            const auto accumulate_tile = [&](int,
                                             const TurboQuantRowScratch &scratch,
                                             int tile_begin,
                                             CPUFA2VisibleTile visible,
                                             const float *scores,
                                             float new_m,
                                             float *partial,
                                             float &new_l)
            {
                for (int kv_row = visible.begin;
                     kv_row < visible.end;
                     ++kv_row)
                {
                    if (kv_row + 4 < visible.end)
                    {
                        _mm_prefetch(
                            reinterpret_cast<const char *>(
                                v_raw +
                                physical_kv_row(kv_row + 4) *
                                    v_row_stride +
                                scratch.value_head_offset),
                            _MM_HINT_T0);
                    }
                    const float probability = std::exp(
                        scores[static_cast<std::size_t>(kv_row - tile_begin)] -
                        new_m);
                    new_l += probability;
                    const std::uint8_t *value_block =
                        v_raw + physical_kv_row(kv_row) * v_row_stride +
                        scratch.value_head_offset;
                    if constexpr (std::is_same_v<ValueTensor, TQ8Tensor>)
                    {
                        tq8_accum_weighted(
                            partial,
                            value_block,
                            probability,
                            head_dim);
                    }
                    else
                    {
                        tq4_accum_weighted(
                            partial,
                            value_block,
                            probability,
                            head_dim);
                    }
                }
            };

            const auto finalize_row = [&](int row,
                                          float *merged,
                                          float merged_l)
            {
                if (merged_l > 0.0f)
                {
                    scale_vec(
                        merged,
                        inv_sqrt_d / merged_l,
                        head_dim,
                        false);
                }
                else
                {
                    std::fill(merged, merged + head_dim, 0.0f);
                }

                const int query_position = row / n_heads;
                const int h = row % n_heads;
                const int kv_h = (gqa_n_rep > 0)
                                     ? (head_start + h) / heads_per_kv
                                     : h / heads_per_kv;
                apply_rotation_transpose(
                    tq_ctx->for_layer(kv_h).rotation(),
                    merged,
                    output + static_cast<std::size_t>(query_position) *
                                 n_heads * head_dim +
                        static_cast<std::size_t>(h) * head_dim);
            };

            return executePartitionedAttention<TurboQuantRowScratch>(
                seq_len,
                n_heads,
                kv_len,
                head_dim,
                kv_tile,
                causal,
                execution_policy,
                prepare_row,
                visible_tile,
                score_tile,
                accumulate_tile,
                finalize_row);
        }
#endif // __AVX512F__ || __AVX2__ (TQ fused)

        /**
         * @brief Validate and expose prebound context-summary storage.
         *
         * This method performs arithmetic and pointer checks only. It never
         * allocates, grows, clears, or rebinds workspace. Callers initialize
         * every element they consume inside the producer OpenMP phase.
         *
         * @param partial_slots Number of `(row, head, partition)` summaries.
         * @param padded_head_dim FP32 stride reserved for each numerator.
         * @return True when all three named buffers cover the requested span.
         */
        [[nodiscard]] bool hasContextSummaryCapacity(
            std::size_t partial_slots,
            std::size_t padded_head_dim) const
        {
            if (partial_slots == 0 || padded_head_dim == 0 ||
                partial_slots >
                    std::numeric_limits<std::size_t>::max() /
                        padded_head_dim)
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] Invalid context-summary geometry: slots="
                          << partial_slots << " padded_head_dim="
                          << padded_head_dim);
                return false;
            }

            const std::size_t output_elements =
                partial_slots * padded_head_dim;
            if (!partial_output_ || !partial_m_ || !partial_l_ ||
                partial_output_capacity_ < output_elements ||
                partial_m_capacity_ < partial_slots ||
                partial_l_capacity_ < partial_slots)
            {
                LOG_ERROR("[CPUFlashAttentionKernelT] Context-parallel attention requires complete prebound workspace: "
                          << "required output/meta elements=" << output_elements
                          << "/" << partial_slots
                          << " available output/m/l="
                          << partial_output_capacity_ << "/"
                          << partial_m_capacity_ << "/"
                          << partial_l_capacity_);
                return false;
            }
            return true;
        }

        /** Immutable-by-execution launch controls resolved during setup. */
        cpu::fa2_policy::CPUFA2KernelLaunchPolicy launch_policy_{};
        float *partial_output_ = nullptr; ///< Prebound FP32 partial numerators.
        float *partial_m_ = nullptr;      ///< Prebound partial score maxima.
        float *partial_l_ = nullptr;      ///< Prebound partial softmax sums.
        std::size_t partial_output_capacity_ = 0; ///< Numerator FP32 elements.
        std::size_t partial_m_capacity_ = 0;      ///< Maximum-count elements.
        std::size_t partial_l_capacity_ = 0;      ///< Denominator elements.
    };

    extern template class CPUFlashAttentionKernelT<ActivationPrecision::FP32>;
    extern template class CPUFlashAttentionKernelT<ActivationPrecision::BF16>;
    extern template class CPUFlashAttentionKernelT<ActivationPrecision::FP16>;

} // namespace llaminar2
