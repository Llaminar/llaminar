/**
 * @file RetainedCPUMegakernelStages.cpp
 * @brief Participant-local CPU single-OMP two-megakernel-per-layer stages for QFN.
 *
 * Implements three participant-local CPU megakernel stages contracted via
 * @ref lowerLocalMegakernelRegion during cold graph construction:
 *   1. @c RetainedCPUGDNStage: fuses HC write/norm/down/inject/up/read, GDN
 *      projections (QKV+Z+A+B with shared Q8_1 quantization), ShortConv1d,
 *      GDNRecurrence, GatedRMSNorm, and ssm_out into one @c OMP_LAYER_PARALLEL region.
 *   2. @c RetainedCPUQSAAttentionStage: fuses HC write/norm/down/inject/up/read,
 *      QKV + qsa_key + ranking-guarded qsa_query (with shared Q8_1 quantization),
 *      QGateSplit, Q/K RMSNorm, RoPE, QSA PrepareReplay, KV Append, QSA Indexer,
 *      SelectedAttention, optional CommitPartial, AttentionOutputGate, and Wo
 *      into one @c OMP_LAYER_PARALLEL region.
 *   3. @c RetainedCPUFFNStage: fuses HC write/norm/down/inject/up/read, MoE
 *      routing + router Q8_1 publication, Shared Expert FFN, and (on singleton
 *      CPU topologies without intervening collectives) Standalone Routed MoE +
 *      Shared Expert Gate into one @c OMP_LAYER_PARALLEL region.
 *
 * General across all 18 NativeQuantised codebooks, FP16/BF16/FP32 weights,
 * Scalar / AVX2 / AVX-512 runtime ISAs (`activeISALevel()`), and arbitrary CPU
 * TP/participant counts (zero hardcoded device counts or codebook mixes).
 */
#include "RetainedCPUMegakernelStages.h"

#include "execution/compute_stages/ProjectionRowWorkBinding.h"
#include "execution/compute_stages/stages/AttentionOutputGateStage.h"
#include "execution/compute_stages/stages/FusedQKVGEMMStage.h"
#include "execution/compute_stages/stages/GDNProjectionStage.h"
#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/compute_stages/stages/GatedRMSNormStage.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoERoutingStage.h"
#include "execution/compute_stages/stages/QGateSplitStage.h"
#include "execution/compute_stages/stages/QKNormStage.h"
#include "execution/compute_stages/stages/RoPEStage.h"
#include "execution/compute_stages/stages/ShortConv1dStage.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpHyperconnectionBuilder.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include "execution/compute_stages/stages/qwen4exp/QwenSparseAttentionStage.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/LocalMegakernelLowering.h"
#include "kernels/IKVCache.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include "kernels/cpu/gdn/CPUShortConvolution.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemv.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedScalarReference.h"
#include "kernels/cpu/gemm/CPUProjectionQ8Publication.h"
#include "kernels/cpu/gemm/CPUQ8ResidualCorrection.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "kernels/cpu/moe/CPUMoEKernel.h"
#include "kernels/cpu/primitives/RMSNormPrimitives.h"
#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "kernels/cpu/primitives/VectorPrimitives.h"
#include "kernels/qwen4exp/Qwen4ExpKernels.h"
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "kernels/qwen4exp/QwenSparseReplay.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/Tensors.h"
#include "utils/CPUFeatures.h"
#include "utils/Logger.h"
#include "utils/OpenMPUtils.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{
namespace
{
    using cpu::native_quantised::CPUNativeQuantisedGemmKernel;
    using cpu::native_quantised::CPUNativeQuantisedPackedWeights;
    using cpu::native_quantised::FusedGemvDesc;
    using cpu::native_quantised::NativeQuantisedFusedRowsPlan;

    constexpr size_t kDefaultPartialScratchFloats = 262144u; ///< Fallback partial capacity only if no workspace is bound.
    inline const std::string kCPUMegakernelRoundedA = "cpu_megakernel_rounded_a";
    inline const std::string kCPUMegakernelFFNRoutedScratch = "cpu_megakernel_ffn_routed_scratch";

    inline size_t alignUp64(size_t bytes) noexcept { return (bytes + 63u) & ~size_t{63u}; }

    /**
     * @brief Compute shared FFN routed scratch byte offsets inside one 64-byte-aligned
     *        device workspace slab reused across all FFN megakernel layers (L3-hot!).
     */
    struct CPUMegakernelFFNRoutedLayout
    {
        size_t gate_offset = 0;
        size_t up_offset = 0;
        size_t down_offset = 0;
        size_t swiglu_q8_offset = 0;
        size_t splitk_offset = 0;
        size_t total_bytes = 0;

        static CPUMegakernelFFNRoutedLayout compute(size_t top_k, size_t inter, size_t d_model) noexcept
        {
            CPUMegakernelFFNRoutedLayout l{};
            size_t cur = 0;
            l.gate_offset = cur;
            cur = alignUp64(cur + top_k * inter * sizeof(float));
            l.up_offset = cur;
            cur = alignUp64(cur + top_k * inter * sizeof(float));
            l.down_offset = cur;
            cur = alignUp64(cur + top_k * d_model * sizeof(float));
            l.swiglu_q8_offset = cur;
            // Reserve (top_k + 1) SwiGLU Q8_1 rows so shared_expert can fuse as route `top_k`.
            cur = alignUp64(cur + (top_k + 1u) * ((inter + 31u) / 32u) * sizeof(Q8_1Block));
            l.splitk_offset = cur;
            cur = alignUp64(cur + CPUNativeQuantisedGemmKernel::kStandaloneMoESplitKScratchFloats * sizeof(float));
            l.total_bytes = cur;
            return l;
        }
    };

    struct CPUProjectionHandle;
    inline size_t maxPartialFloatsForHandle(const CPUProjectionHandle &h);

    constexpr int kMaxPrivateActivationFloats = 4096; ///< Stack cap for thread-private L1d M=1 activations (16 KiB).

    /**
     * @brief Round `[0, k)` to binary16-RNE in FP32 into a thread-private stack buffer
     *        using staggered 64-byte cache-line order (zero barrier, zero HitM convoy).
     */
    inline void staggeredRoundRowToFP16RNEPrivate(
        const float *src, float *local_dst, int k, ISALevel isa) noexcept
    {
        const int n_lines = k / 16;
        const int nt = std::max(1, omp_get_num_threads());
        const int tid = omp_get_thread_num();
        const int start = (n_lines > 0) ? ((tid * n_lines) / nt) : 0;
        for (int step = 0; step < n_lines; ++step)
        {
            int idx = start + step;
            if (idx >= n_lines)
                idx -= n_lines;
            const int off = idx * 16;
            gemm::round_fp32_to_fp16_rne_as_fp32_range(src, local_dst, off, off + 16, isa);
        }
        const int tail = n_lines * 16;
        if (tail < k)
            gemm::round_fp32_to_fp16_rne_as_fp32_range(src, local_dst, tail, k, isa);
    }

    /**
     * @brief Copy `[0, k)` FP32 activations into a thread-private stack buffer in staggered
     *        64-byte cache-line order to eliminate multi-core Modified-state snoop storms.
     */
    inline void staggeredCopyFloatsPrivate(const float *src, float *local_dst, int k) noexcept
    {
        const int n_lines = k / 16;
        const int nt = std::max(1, omp_get_num_threads());
        const int tid = omp_get_thread_num();
        const int start = (n_lines > 0) ? ((tid * n_lines) / nt) : 0;
        for (int step = 0; step < n_lines; ++step)
        {
            int idx = start + step;
            if (idx >= n_lines)
                idx -= n_lines;
            const int off = idx * 16;
            std::memcpy(local_dst + off, src + off, 16u * sizeof(float));
        }
        const int tail = n_lines * 16;
        if (tail < k)
            std::memcpy(local_dst + tail, src + tail, static_cast<size_t>(k - tail) * sizeof(float));
    }

    inline void roundRowToFP16RNEInTeam(const float *src, float *dst, int k, ISALevel isa) noexcept
    {
        constexpr int kBlock = 64;
        const int blocks = (k + kBlock - 1) / kBlock;
#pragma omp for schedule(static)
        for (int b = 0; b < blocks; ++b)
        {
            const int start = b * kBlock;
            const int end = std::min(start + kBlock, k);
            gemm::round_fp32_to_fp16_rne_as_fp32_range(src, dst, start, end, isa);
        }
    }

    /** @brief Exact serial-decode std::fma dot matching FloatingPointGemmKernel::run_fp32_skinny_matmul. */
    inline float dotFP32SerialFMA(const float *a, const float *b, int k) noexcept
    {
        float acc = 0.0f;
        for (int kk = 0; kk < k; ++kk)
            acc = std::fma(a[kk], b[kk], acc);
        return acc;
    }

    /** @brief Exact serial-decode std::fma dot matching FloatingPointGemmKernel::run_fp32xbf16_skinny_matmul. */
    inline float dotFP32xBF16SerialFMA(const float *a, const uint16_t *b, int k) noexcept
    {
        float acc = 0.0f;
        for (int kk = 0; kk < k; ++kk)
            acc = std::fma(a[kk], simd::bf16_to_fp32(b[kk]), acc);
        return acc;
    }

    /** @brief Exact serial-decode std::fma dot matching FloatingPointGemmKernel::run_fp32xfp16_skinny_matmul. */
    inline float dotFP32xFP16SerialFMA(const float *a, const uint16_t *b, int k) noexcept
    {
        float acc = 0.0f;
        for (int kk = 0; kk < k; ++kk)
            acc = std::fma(a[kk], fp16_to_fp32(b[kk]), acc);
        return acc;
    }

    void require(bool condition, const char *message)
    {
        if (!condition)
            throw std::invalid_argument(message);
    }

    /** @brief Resolve raw host float pointer for a zero-or-nonzero byte-offset FP32 arena binding. */
    inline float *bindingFloatPtr(const Qwen4ExpFloatBinding &b) noexcept
    {
        return reinterpret_cast<float *>(
            static_cast<std::uint8_t *>(b.tensor->raw_mutable_data()) + b.byteOffset);
    }

    inline const float *bindingConstFloatPtr(const Qwen4ExpFloatBinding &b) noexcept
    {
        return reinterpret_cast<const float *>(
            static_cast<const std::uint8_t *>(b.tensor->raw_data()) + b.byteOffset);
    }

    inline std::int32_t *bindingIntPtr(const Qwen4ExpIntBinding &b) noexcept
    {
        return reinterpret_cast<std::int32_t *>(
            static_cast<std::uint8_t *>(b.tensor->raw_mutable_data()) + b.byteOffset);
    }

    inline const std::int32_t *bindingConstIntPtr(const Qwen4ExpIntBinding &b) noexcept
    {
        return reinterpret_cast<const std::int32_t *>(
            static_cast<const std::uint8_t *>(b.tensor->raw_data()) + b.byteOffset);
    }

    inline const std::int32_t *resolveIntInputPtr(const Qwen4ExpIntInput &input) noexcept
    {
        if (const auto *arena = std::get_if<Qwen4ExpIntBinding>(&input.value))
            return bindingConstIntPtr(*arena);
        return std::get<Qwen4ExpNativeIntInput>(input.value).data;
    }

    /** @brief Infer effective BLAS transpose_B matching GEMMStage::effectiveTransposeBForWeight. */
    inline bool effectiveFloatTransposeB(const TensorBase *weight, int n, int k, bool requested) noexcept
    {
        if (requested || !weight)
            return requested;
        const TensorType t = weight->native_type();
        if (t != TensorType::FP32 && t != TensorType::FP16 && t != TensorType::BF16)
            return requested;
        const auto &shape = weight->shape();
        if (shape.size() >= 2 && shape[0] == static_cast<size_t>(n) && shape[1] == static_cast<size_t>(k))
            return true;
        return requested;
    }

    /**
     * @brief Quantize a single FP32 activation row [k] into Q8_1Block[ceil(k/32)]
     *        in parallel across the caller's active OpenMP team.
     *
     * Uses the exact pair-block quantization order of `gemv_native_quantised`
     * so every produced Q8_1 block is bitwise identical to standalone GEMV.
     */
    inline void quantizeRowToQ8InTeam(const float *src, Q8_1Block *dst, int k,
                                      CPUProjectionNumericalPolicy policy) noexcept
    {
        const int k_blocks = (k + static_cast<int>(Q8_1Block::BLOCK_SIZE) - 1) /
                             static_cast<int>(Q8_1Block::BLOCK_SIZE);
        if ((k % 64) == 0)
        {
            const int pairs = k_blocks / 2;
#pragma omp for schedule(static)
            for (int p = 0; p < pairs; ++p)
            {
                const int kb = p * 2;
                cpu::native_quantised::quantizeTwoActivationBlocks(
                    src + static_cast<size_t>(kb) * 32u, dst[kb], dst[kb + 1], policy);
            }
        }
        else
        {
#pragma omp for schedule(static)
            for (int kb = 0; kb < k_blocks; ++kb)
            {
                const int off = kb * 32;
                cpu::native_quantised::quantizeActivationBlock(
                    src + off, dst[kb], std::min(32, k - off), policy);
            }
        }
    }

    /**
     * @brief Execute 1..N NativeQuantised M=1 projections sharing one pre-quantized Q8_1 row
     *        inside an already-active OpenMP team across all 18 codebooks and Scalar/AVX2/AVX-512.
     */
    inline void runNativeQuantisedBundleInTeam(
        const Q8_1Block *input_q8,
        const FusedGemvDesc *descs,
        int num_descs,
        std::span<float> partials)
    {
        if (num_descs <= 0)
            return;
        // Stage compact Q8_1 rows (<= 128 blocks = 4.6 KiB, e.g. K=2560) into
        // thread-private L1d stack memory in staggered order so workers avoid
        // multi-core HitM snoop convoys without blowing L1d on wide K=10240 rows.
        constexpr int kMaxLocalQ8Blocks = 128;
        alignas(64) Q8_1Block local_q8[kMaxLocalQ8Blocks];
        const Q8_1Block *active_q8 = input_q8;
        const int k_blocks = descs[0].packed ? descs[0].packed->blocks_per_row : 0;
        const int nt = omp_get_num_threads();
        if (nt > 1 && k_blocks > 0 && k_blocks <= kMaxLocalQ8Blocks)
        {
            const int tid = omp_get_thread_num();
            const int start = (tid * k_blocks) / nt;
            for (int step = 0; step < k_blocks; ++step)
            {
                int idx = start + step;
                if (idx >= k_blocks)
                    idx -= k_blocks;
                local_q8[idx] = input_q8[idx];
            }
            active_q8 = local_q8;
        }
        const ISALevel isa = activeISALevel();
        if (isa >= ISALevel::AVX2)
        {
            if (num_descs == 1 && descs[0].bias == nullptr &&
                (descs[0].policy_n == 0 || descs[0].policy_n == descs[0].N))
            {
                cpu::native_quantised::gemv_native_quantised_preq(
                    *descs[0].packed, active_q8, descs[0].output, partials);
                return;
            }
            cpu::native_quantised::gemv_native_quantised_fused_preq(
                active_q8, descs, partials, num_descs);
            return;
        }

        // Scalar ISA fallback: exact ordered K-partition scalar reference across all 18 codebooks.
        const int workers = std::max(1, omp_get_num_threads());
        const auto &cache = CPUExecutionGeometry::local().cache;
        for (int p = 0; p < num_descs; ++p)
        {
            const auto &d = descs[p];
            const auto &packed = *d.packed;
            const int p_k_blocks = packed.blocks_per_row;
            const int n_chunks = (d.N + 63) / 64;
            const int policy_n = d.policy_n > 0 ? d.policy_n : d.N;
            const auto cfg = cpu::native_quantised::serialTileConfigForPreparedProjection(
                packed.preparedFootprint(), packed.codebook_id,
                packed.numerical_policy, policy_n, packed.K, workers, cache);
            const int k_tiles = std::max(1, cfg.k_tiles);
            const int bpt = (p_k_blocks + k_tiles - 1) / k_tiles;
#pragma omp for schedule(static) nowait
            for (int chunk = 0; chunk < n_chunks; ++chunk)
            {
                const int col_start = chunk * 64;
                const int cols = std::min(64, d.N - col_start);
                cpu::native_quantised::gemvScalarOrderedPartitions(
                    packed, active_q8, d.output + col_start, chunk, p_k_blocks, k_tiles, bpt, cols);
                if (d.bias)
                {
                    for (int c = 0; c < cols; ++c)
                        d.output[col_start + c] += d.bias[col_start + c];
                }
            }
        }
#pragma omp barrier
    }

    /** @brief Prepared single M=1 CPU projection handle supporting NativeQuantised and FloatingPoint kernels. */
    struct CPUProjectionHandle
    {
        const TensorBase *weight = nullptr;
        ITensorGemm *engine = nullptr;
        CPUNativeQuantisedGemmKernel *native_engine = nullptr;
        const void *fp_weight_data = nullptr;
        TensorType fp_weight_type = TensorType::FP32;
        int n = 0;
        int k = 0;
        int policy_n = 0;
        bool transpose_b = false;
        QuantizedActivationOperandPrecision precision = QuantizedActivationOperandPrecision::SingleQ8;
        FloatingActivationOperandPrecision fp_operand_precision = FloatingActivationOperandPrecision::Native;

        void resolvePreparedFloatingView()
        {
            fp_weight_data = nullptr;
            fp_weight_type = weight ? weight->native_type() : TensorType::FP32;
            if (!native_engine && engine)
            {
                ContiguousFloatingPointWeightDescriptor desc{};
                if (engine->exportContiguousFloatingPointWeights(desc) && desc.valid())
                {
                    fp_weight_data = desc.data;
                    fp_weight_type = desc.type;
                }
            }
        }

        void bindFromGEMMParams(const GEMMStage::Params &p)
        {
            require(p.prepared_store && p.prepared_ref && p.B,
                    "CPU megakernel projection requires prepared weight reference");
            weight = dynamic_cast<const TensorBase *>(p.B);
            engine = p.prepared_store->gemmKernel(*p.prepared_ref);
            require(weight && engine, "CPU megakernel projection failed to resolve prepared engine");
            native_engine = dynamic_cast<CPUNativeQuantisedGemmKernel *>(engine);
            n = p.n;
            k = p.k;
            policy_n = 0;
            transpose_b = effectiveFloatTransposeB(weight, n, k, p.transpose_B);
            precision = p.quantized_operand_precision;
            fp_operand_precision = engine->floatingActivationOperandPrecision();
            resolvePreparedFloatingView();
        }

        void bindFromRef(const ITensor *w, PreparedWeightStore *store, const PreparedWeightRef &ref,
                         int out_n, int in_k, const std::optional<OriginalOutputSlice> &slice = std::nullopt,
                         QuantizedActivationOperandPrecision prec = QuantizedActivationOperandPrecision::SingleQ8)
        {
            require(store && w, "CPU megakernel projection requires prepared store and weight");
            weight = dynamic_cast<const TensorBase *>(w);
            engine = store->gemmKernel(ref);
            require(weight && engine, "CPU megakernel projection failed to resolve prepared engine");
            native_engine = dynamic_cast<CPUNativeQuantisedGemmKernel *>(engine);
            n = out_n;
            k = in_k;
            policy_n = slice ? slice->policyColumns() : 0;
            transpose_b = effectiveFloatTransposeB(weight, n, k, false);
            precision = prec;
            fp_operand_precision = engine->floatingActivationOperandPrecision();
            resolvePreparedFloatingView();
        }

        bool isSingleQ8Native() const noexcept
        {
            return native_engine != nullptr &&
                   precision == QuantizedActivationOperandPrecision::SingleQ8;
        }

        bool isTwoTermQ8Native() const noexcept
        {
            return native_engine != nullptr &&
                   precision == QuantizedActivationOperandPrecision::TwoTermQ8;
        }
    };

    inline size_t maxPartialFloatsForHandle(const CPUProjectionHandle &h)
    {
        if (!h.isSingleQ8Native())
            return 0;
        return cpu::native_quantised::nativeQuantisedProjectionPartialEnvelopeFloats(
            h.native_engine->packedWeights(), 1);
    }

    /**
     * @brief Nested-safe, zero-race, bit-exact M=1 floating-point GEMV inside an active OpenMP team.
     *
     * Operates directly on immutable prepared floating weight storage resolved at
     * graph construction time, eliminating concurrent `TensorBase::mutable_data()`
     * coherence-state writes, per-thread `dynamic_cast`s, and oneDNN contention while
     * preserving bit-for-bit equivalence with `FloatingPointGemmKernel::multiply_tensor`:
     *   - `FP16RNE` (`ssm_alpha/beta`, etc.): pre-rounds `input_ptr[K]` once into
     *     L1-hot `rounded_a[K]` and runs the exact 256-lane SIMD tree.
     *   - Native `FP32` / `BF16` / `FP16` (`hc_inject`, `qsa_key_proj`, etc.): exact
     *     serial-decode `std::fma` per output column matching `run_fp32*_skinny_matmul`.
     */
    inline bool runFloatGemvInTeam(
        const TensorBase *input_tensor,
        const float *input_ptr,
        const CPUProjectionHandle &proj,
        TensorBase *output_tensor,
        float *output_ptr,
        float *rounded_a,
        float *status_slot,
        DeviceWorkspaceManager *workspace,
        ISALevel isa)
    {
        const int N = proj.n;
        const int K = proj.k;
        if (proj.transpose_b && proj.fp_weight_data != nullptr)
        {
            if (proj.fp_operand_precision == FloatingActivationOperandPrecision::FP16RNE &&
                proj.fp_weight_type == TensorType::FP16 && (rounded_a != nullptr || K <= kMaxPrivateActivationFloats))
            {
                alignas(64) float local_a[kMaxPrivateActivationFloats];
                const float *act_ptr = rounded_a;
                if (K <= kMaxPrivateActivationFloats)
                {
                    staggeredRoundRowToFP16RNEPrivate(input_ptr, local_a, K, isa);
                    act_ptr = local_a;
                }
                else
                {
                    roundRowToFP16RNEInTeam(input_ptr, rounded_a, K, isa);
                }
#pragma omp for schedule(static)
                for (int col = 0; col < N; ++col)
                {
                    output_ptr[col] = gemm::dot_gpu_aligned_expert_one_row<TensorType::FP16>(
                        act_ptr, proj.fp_weight_data,
                        static_cast<std::size_t>(col) * static_cast<std::size_t>(K), K, isa);
                }
                return true;
            }
            if (proj.fp_operand_precision == FloatingActivationOperandPrecision::Native ||
                proj.fp_operand_precision == FloatingActivationOperandPrecision::FP32)
            {
                if (proj.fp_weight_type == TensorType::FP32)
                {
                    const float *w = static_cast<const float *>(proj.fp_weight_data);
#pragma omp for schedule(static)
                    for (int col = 0; col < N; ++col)
                    {
                        output_ptr[col] = dotFP32SerialFMA(
                            input_ptr, w + static_cast<std::size_t>(col) * static_cast<std::size_t>(K), K);
                    }
                    return true;
                }
                if (proj.fp_weight_type == TensorType::BF16)
                {
                    const uint16_t *w = static_cast<const uint16_t *>(proj.fp_weight_data);
#pragma omp for schedule(static)
                    for (int col = 0; col < N; ++col)
                    {
                        output_ptr[col] = dotFP32xBF16SerialFMA(
                            input_ptr, w + static_cast<std::size_t>(col) * static_cast<std::size_t>(K), K);
                    }
                    return true;
                }
                if (proj.fp_weight_type == TensorType::FP16)
                {
                    const uint16_t *w = static_cast<const uint16_t *>(proj.fp_weight_data);
#pragma omp for schedule(static)
                    for (int col = 0; col < N; ++col)
                    {
                        output_ptr[col] = dotFP32xFP16SerialFMA(
                            input_ptr, w + static_cast<std::size_t>(col) * static_cast<std::size_t>(K), K);
                    }
                    return true;
                }
            }
        }
        // Safe serialized fallback for non-transposed or exotic floating engines (never 28 concurrent calls!).
        OMP_SINGLE([&]() {
            const bool ok_call = proj.engine->multiply_tensor(
                input_tensor, output_tensor, 1, N, K, proj.transpose_b,
                1.0f, 0.0f, nullptr, nullptr, -1, workspace);
            if (status_slot)
                *status_slot = ok_call ? 1.0f : -1.0f;
        });
        return !status_slot || *status_slot >= 0.0f;
    }

    /**
     * @brief Fuse two M=1 transposed floating projections sharing one input vector into
     *        a single OpenMP workshare with bit-exact parity across FP16RNE, FP32, BF16, and FP16.
     */
    inline bool runPairedFloatGemvInTeam(
        const float *input_ptr,
        const CPUProjectionHandle &p0,
        float *out0,
        const CPUProjectionHandle &p1,
        float *out1,
        float *rounded_a,
        ISALevel isa)
    {
        if (!p0.transpose_b || !p1.transpose_b || p0.k != p1.k ||
            !p0.fp_weight_data || !p1.fp_weight_data ||
            p0.fp_operand_precision != p1.fp_operand_precision ||
            p0.fp_weight_type != p1.fp_weight_type)
            return false;

        const int K = p0.k;
        const int N0 = p0.n;
        const int total_n = N0 + p1.n;

        if (p0.fp_operand_precision == FloatingActivationOperandPrecision::FP16RNE &&
            p0.fp_weight_type == TensorType::FP16 && (rounded_a != nullptr || K <= kMaxPrivateActivationFloats))
        {
            alignas(64) float local_a[kMaxPrivateActivationFloats];
            const float *act_ptr = rounded_a;
            if (K <= kMaxPrivateActivationFloats)
            {
                staggeredRoundRowToFP16RNEPrivate(input_ptr, local_a, K, isa);
                act_ptr = local_a;
            }
            else
            {
                roundRowToFP16RNEInTeam(input_ptr, rounded_a, K, isa);
            }
#pragma omp for schedule(static)
            for (int idx = 0; idx < total_n; ++idx)
            {
                if (idx < N0)
                {
                    out0[idx] = gemm::dot_gpu_aligned_expert_one_row<TensorType::FP16>(
                        act_ptr, p0.fp_weight_data,
                        static_cast<std::size_t>(idx) * static_cast<std::size_t>(K), K, isa);
                }
                else
                {
                    const int c = idx - N0;
                    out1[c] = gemm::dot_gpu_aligned_expert_one_row<TensorType::FP16>(
                        act_ptr, p1.fp_weight_data,
                        static_cast<std::size_t>(c) * static_cast<std::size_t>(K), K, isa);
                }
            }
            return true;
        }
        if (p0.fp_operand_precision == FloatingActivationOperandPrecision::Native ||
            p0.fp_operand_precision == FloatingActivationOperandPrecision::FP32)
        {
            if (p0.fp_weight_type == TensorType::FP32)
            {
                const float *w0 = static_cast<const float *>(p0.fp_weight_data);
                const float *w1 = static_cast<const float *>(p1.fp_weight_data);
#pragma omp for schedule(static)
                for (int idx = 0; idx < total_n; ++idx)
                {
                    if (idx < N0)
                        out0[idx] = dotFP32SerialFMA(input_ptr, w0 + static_cast<std::size_t>(idx) * static_cast<std::size_t>(K), K);
                    else
                    {
                        const int c = idx - N0;
                        out1[c] = dotFP32SerialFMA(input_ptr, w1 + static_cast<std::size_t>(c) * static_cast<std::size_t>(K), K);
                    }
                }
                return true;
            }
            if (p0.fp_weight_type == TensorType::BF16)
            {
                const uint16_t *w0 = static_cast<const uint16_t *>(p0.fp_weight_data);
                const uint16_t *w1 = static_cast<const uint16_t *>(p1.fp_weight_data);
#pragma omp for schedule(static)
                for (int idx = 0; idx < total_n; ++idx)
                {
                    if (idx < N0)
                        out0[idx] = dotFP32xBF16SerialFMA(input_ptr, w0 + static_cast<std::size_t>(idx) * static_cast<std::size_t>(K), K);
                    else
                    {
                        const int c = idx - N0;
                        out1[c] = dotFP32xBF16SerialFMA(input_ptr, w1 + static_cast<std::size_t>(c) * static_cast<std::size_t>(K), K);
                    }
                }
                return true;
            }
            if (p0.fp_weight_type == TensorType::FP16)
            {
                const uint16_t *w0 = static_cast<const uint16_t *>(p0.fp_weight_data);
                const uint16_t *w1 = static_cast<const uint16_t *>(p1.fp_weight_data);
#pragma omp for schedule(static)
                for (int idx = 0; idx < total_n; ++idx)
                {
                    if (idx < N0)
                        out0[idx] = dotFP32xFP16SerialFMA(input_ptr, w0 + static_cast<std::size_t>(idx) * static_cast<std::size_t>(K), K);
                    else
                    {
                        const int c = idx - N0;
                        out1[c] = dotFP32xFP16SerialFMA(input_ptr, w1 + static_cast<std::size_t>(c) * static_cast<std::size_t>(K), K);
                    }
                }
                return true;
            }
        }
        return false;
    }

    /** @brief Merge StageBufferContract, StageBufferRequirements, and WorkspaceRequirements from contracted nodes. */
    struct ContractedStageMetadata
    {
        StageBufferContract contract;
        StageBufferRequirements buffer_reqs;
        WorkspaceRequirements workspace_reqs;

        void absorb(const ComputeGraph &graph, const std::vector<std::string> &member_names)
        {
            std::unordered_set<BufferId> produced_inside;
            std::unordered_set<BufferId> seen_in;
            std::unordered_set<BufferId> seen_out;
            std::unordered_set<BufferId> seen_inout;
            std::unordered_set<const ITensor *> seen_weights;
            std::unordered_set<uint64_t> seen_prepared_ids;

            for (const auto &name : member_names)
            {
                const auto *node = graph.getNode(name);
                if (!node || !node->stage)
                    continue;
                const auto c = node->stage->bufferContract();
                for (const auto &b : c.inputs)
                {
                    if (!produced_inside.contains(b.id) && seen_in.insert(b.id).second)
                        contract.addInput(b.id);
                }
                for (const auto &b : c.inouts)
                {
                    if (!produced_inside.contains(b.id) && seen_inout.insert(b.id).second)
                        contract.addInOut(b.id);
                    produced_inside.insert(b.id);
                }
                for (const auto &b : c.outputs)
                {
                    produced_inside.insert(b.id);
                    if (seen_out.insert(b.id).second)
                        contract.addOutput(b.id);
                }
                for (auto *w : c.weight_tensors)
                {
                    if (w && seen_weights.insert(w).second)
                        contract.addWeight(w);
                }
                for (const auto &pw : c.prepared_weights)
                {
                    if (pw.ref.binding_id != 0 && seen_prepared_ids.insert(pw.ref.binding_id).second)
                        contract.addPreparedWeight(pw.source_tensor, pw.store, pw.ref);
                }
                if (const auto *ws = dynamic_cast<const IWorkspaceConsumer *>(node->stage.get()))
                    workspace_reqs.merge(ws->getWorkspaceRequirements(1, 0, 0));
            }
        }
    };

    /**
     * @brief Discovered participant-local CPU HyperConnection read region.
     *
     * Handles both fused TwoTermQ8 `hc_up` (with embedded `hc_residual_down`) and
     * unfused `down -> ResidualDown -> up` graphs, plus optional preceding
     * `ResidualWrite` and paired `write_projection` (`hc_inject`).
     */
    struct CPUHCRegionNodes
    {
        std::string previous_write;
        std::string branch_norm;
        std::string down;
        std::string read_activation; ///< Empty when hc_up owns hc_residual_down.
        std::string up;
        std::string read;
        std::string injection; ///< Empty when not safely absorbable or final read.

        std::optional<Qwen4ExpPrimitiveStage::ResidualWrite> write_op;
        std::optional<Qwen4ExpPrimitiveStage::BranchRMSNorm> norm_op;
        GEMMStage::Params down_params;
        std::optional<Qwen4ExpPrimitiveStage::ResidualDown> activation_op;
        GEMMStage::Params up_params;
        std::optional<Qwen4ExpPrimitiveStage::ResidualRead> read_op;
        std::optional<GEMMStage::Params> inject_params;
    };

    bool discoverCPUHCRegion(const ComputeGraph &graph,
                             const std::string &read_name,
                             const std::string &previous_boundary,
                             CPUHCRegionNodes &out)
    {
        using Primitive = Qwen4ExpPrimitiveStage;
        const auto *read_node = graph.getNode(read_name);
        const auto *read_stage = read_node ? dynamic_cast<const Primitive *>(read_node->stage.get()) : nullptr;
        const auto *r = read_stage ? std::get_if<Primitive::ResidualRead>(&read_stage->getParams().operation) : nullptr;
        if (!r || read_stage->getParams().rows.capacity != 1 || read_stage->getParams().rows.liveRows.has_value())
            return false;

        const Primitive *norm_stage = nullptr;
        const GEMMStage *up_stage = nullptr;
        for (const auto &dep : read_node->dependencies)
        {
            const auto *node = graph.getNode(dep);
            if (!node || !node->stage)
                continue;
            if (const auto *prim = dynamic_cast<const Primitive *>(node->stage.get()))
            {
                if (const auto *n = std::get_if<Primitive::BranchRMSNorm>(&prim->getParams().operation))
                {
                    if (n->output.tensor == r->normalized.tensor && n->output.buffer == r->normalized.buffer)
                    {
                        if (norm_stage)
                            return false;
                        norm_stage = prim;
                        out.branch_norm = dep;
                    }
                }
            }
            if (const auto *gemm = dynamic_cast<const GEMMStage *>(node->stage.get()))
            {
                if (gemm->getParams().C == r->gateLogits.tensor)
                {
                    if (up_stage)
                        return false;
                    up_stage = gemm;
                    out.up = dep;
                }
            }
        }
        if (!norm_stage || !up_stage)
            return false;

        const auto &norm_op = std::get<Primitive::BranchRMSNorm>(norm_stage->getParams().operation);
        if (norm_stage->getParams().rows.capacity != 1 || norm_stage->getParams().rows.liveRows.has_value() ||
            norm_op.diagnostic_input_publication != Primitive::DiagnosticInputPublication::None ||
            norm_op.input_role != Primitive::BranchNormInputRole::Ordinary)
            return false;

        // Locate down (either direct predecessor of up or via ResidualDown).
        const GEMMStage *down_stage = nullptr;
        const Primitive *act_stage = nullptr;
        const auto *up_node = graph.getNode(out.up);
        for (const auto &dep : up_node->dependencies)
        {
            const auto *node = graph.getNode(dep);
            if (!node || !node->stage)
                continue;
            if (const auto *gemm = dynamic_cast<const GEMMStage *>(node->stage.get()))
            {
                if (gemm->getParams().C == up_stage->getParams().A)
                {
                    down_stage = gemm;
                    out.down = dep;
                }
            }
            else if (const auto *prim = dynamic_cast<const Primitive *>(node->stage.get()))
            {
                if (const auto *rd = std::get_if<Primitive::ResidualDown>(&prim->getParams().operation))
                {
                    if (rd->output.tensor == up_stage->getParams().A)
                    {
                        act_stage = prim;
                        out.read_activation = dep;
                        for (const auto &adep : node->dependencies)
                        {
                            const auto *anode = graph.getNode(adep);
                            if (const auto *dgemm = anode ? dynamic_cast<const GEMMStage *>(anode->stage.get()) : nullptr)
                            {
                                if (dgemm->getParams().C == rd->input.tensor)
                                {
                                    down_stage = dgemm;
                                    out.down = adep;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (!down_stage)
            return false;

        const auto validPlainGEMM = [](const GEMMStage::Params &p) -> bool {
            return p.m == 1 && !p.needs_allreduce && p.output_range.empty() &&
                   !p.force_decode_equivalent_verifier_prefill && !p.request_rows &&
                   !p.projection_row_work && !p.hc_short_k_read && !p.gated_norm &&
                   !p.do_swiglu && !p.bias && !p.bias_tensor &&
                   p.alpha == 1.0f && p.beta == 0.0f &&
                   p.prepared_store && p.prepared_ref && p.prepared_store->contains(*p.prepared_ref);
        };
        if (!validPlainGEMM(down_stage->getParams()) || !validPlainGEMM(up_stage->getParams()))
            return false;

        // Optional paired injection projection sharing norm_op.output.
        for (const auto &name : graph.getExecutionOrder())
        {
            const auto *node = graph.getNode(name);
            const auto *p = node ? dynamic_cast<const GEMMStage *>(node->stage.get()) : nullptr;
            if (!p || p == down_stage || p == up_stage)
                continue;
            const auto &a = p->getParams();
            if (a.A == norm_op.output.tensor && a.n == 4 && a.k == 4 * norm_op.hidden &&
                validPlainGEMM(a) &&
                node->dependencies == std::vector<std::string>{out.branch_norm} &&
                name != graph.terminalNode())
            {
                if (!out.injection.empty())
                    return false;
                out.injection = name;
                out.inject_params = a;
            }
        }

        // Optional immediately preceding ResidualWrite whose sole consumer is branch_norm.
        if (!previous_boundary.empty())
        {
            const auto *bnode = graph.getNode(previous_boundary);
            const auto *bprim = bnode ? dynamic_cast<const Primitive *>(bnode->stage.get()) : nullptr;
            const auto *bwrite = bprim ? std::get_if<Primitive::ResidualWrite>(&bprim->getParams().operation) : nullptr;
            const auto *norm_node = graph.getNode(out.branch_norm);
            if (bwrite && norm_node &&
                bnode->device == read_node->device &&
                bprim->getParams().rows.capacity == 1 && !bprim->getParams().rows.liveRows.has_value() &&
                bwrite->hidden == norm_op.hidden &&
                bwrite->output.tensor == norm_op.input.tensor &&
                bwrite->output.buffer == norm_op.input.buffer &&
                bwrite->output.byteOffset == norm_op.input.byteOffset &&
                norm_node->dependencies == std::vector<std::string>{previous_boundary} &&
                graph.directConsumersOf(previous_boundary) == std::vector<std::string>{out.branch_norm})
            {
                out.previous_write = previous_boundary;
                out.write_op.emplace(*bwrite);
            }
        }

        out.read = read_name;
        out.norm_op.emplace(norm_op);
        out.down_params = down_stage->getParams();
        if (act_stage)
            out.activation_op.emplace(std::get<Primitive::ResidualDown>(act_stage->getParams().operation));
        out.up_params = up_stage->getParams();
        out.read_op.emplace(*r);
        return true;
    }

    /**
     * @brief Reusable CPU HyperConnection prefix executor operating inside an active OpenMP team.
     *
     * Fuses optional previous ResidualWrite + BranchRMSNorm, shares one Q8_1 quantization
     * of `normalized[4H]` across `hc_down` and `hc_inject` when both are SingleQ8 native,
     * runs TwoTermQ8 or SingleQ8 `hc_up`, and executes `ResidualRead` -> `mixed[H]`.
     */
    class CPUHCLocalPrefix final
    {
    public:
        CPUHCLocalPrefix(DeviceId device, const CPUHCRegionNodes &region)
            : write_op_(region.write_op),
              norm_op_(*region.norm_op),
              down_params_(region.down_params),
              act_op_(region.activation_op),
              up_params_(region.up_params),
              read_op_(*region.read_op),
              inject_params_(region.inject_params),
              qwen4exp_kernels_(llaminar::v2::kernels::KernelFactory::createQwen4Exp(device))
        {
            require(qwen4exp_kernels_ != nullptr, "CPU HC local prefix requires CPU Qwen4Exp kernel bundle");
            down_.bindFromGEMMParams(down_params_);
            up_.bindFromGEMMParams(up_params_);
            if (inject_params_)
                inject_.bindFromGEMMParams(*inject_params_);
            const size_t norm_q8_blocks = static_cast<size_t>((4 * norm_op_.hidden + 31) / 32);
            norm_q8_.resize(norm_q8_blocks);
            const size_t up_q8_blocks = static_cast<size_t>((up_.k + 31) / 32);
            up_primary_q8_.resize(up_q8_blocks);
            up_residual_q8_.resize(up_q8_blocks);
        }

        void appendMemberNames(std::vector<std::string> &members, const CPUHCRegionNodes &r)
        {
            if (!r.previous_write.empty())
                members.push_back(r.previous_write);
            members.push_back(r.branch_norm);
            members.push_back(r.down);
            if (!r.read_activation.empty())
                members.push_back(r.read_activation);
            members.push_back(r.up);
            members.push_back(r.read);
            if (!r.injection.empty())
                members.push_back(r.injection);
        }

        /** @brief Host pointers resolved once on master thread before entering OMP_LAYER_PARALLEL. */
        struct ResolvedPointers
        {
            const float *write_residual = nullptr;
            const float *write_block = nullptr;
            const float *write_gate_logits = nullptr;
            float *write_output = nullptr;
            const float *norm_input = nullptr;
            const float *gain = nullptr;
            float *norm_output = nullptr;
            float *down_out = nullptr;
            float *inject_out = nullptr;
            const float *act_input = nullptr;
            float *act_output = nullptr;
            float *up_out = nullptr;
            float *activated_out = nullptr;
            const float *read_normalized = nullptr;
            const float *read_gate_logits = nullptr;
            float *read_output = nullptr;
        };

        ResolvedPointers resolvePointers() const
        {
            ResolvedPointers p{};
            if (write_op_)
            {
                p.write_residual = bindingConstFloatPtr(write_op_->residual);
                p.write_block = bindingConstFloatPtr(write_op_->block);
                p.write_gate_logits = bindingConstFloatPtr(write_op_->gateLogits);
                p.write_output = bindingFloatPtr(write_op_->output);
            }
            else
            {
                p.norm_input = bindingConstFloatPtr(norm_op_.input);
            }
            p.gain = norm_op_.gain.tensor->data();
            p.norm_output = bindingFloatPtr(norm_op_.output);
            p.down_out = dynamic_cast<TensorBase *>(down_params_.C)->mutable_data();
            p.inject_out = inject_params_
                ? dynamic_cast<TensorBase *>(inject_params_->C)->mutable_data()
                : nullptr;
            if (act_op_)
            {
                p.act_input = bindingConstFloatPtr(act_op_->input);
                p.act_output = bindingFloatPtr(act_op_->output);
            }
            p.up_out = dynamic_cast<TensorBase *>(up_params_.C)->mutable_data();
            p.activated_out = (up_.isTwoTermQ8Native() && up_params_.hc_residual_down)
                ? up_params_.hc_residual_down->activated->mutable_data()
                : nullptr;
            p.read_normalized = bindingConstFloatPtr(read_op_.normalized);
            p.read_gate_logits = bindingConstFloatPtr(read_op_.gateLogits);
            p.read_output = bindingFloatPtr(read_op_.output);
            return p;
        }

        /** @brief Run the entire HC read prefix inside the caller's already-active OpenMP team. */
        bool executeInTeam(const ResolvedPointers &ptrs, std::span<float> partials,
                           float *rounded_a, DeviceWorkspaceManager *workspace, ISALevel isa,
                           std::array<std::chrono::steady_clock::time_point, 5> *hc_ts = nullptr)
        {
            auto hmark = [&](size_t i) noexcept {
                if (hc_ts && omp_get_thread_num() == 0)
                    (*hc_ts)[i] = std::chrono::steady_clock::now();
            };
            hmark(0);
            // 1. Optional fused ResidualWrite + BranchRMSNorm (or standalone BranchRMSNorm).
            if (write_op_)
            {
                qwen4exp::ResidualWriteBranchRMSNormArgs args{
                    .extent = {1, nullptr},
                    .hidden = write_op_->hidden,
                    .epsilon = norm_op_.epsilon,
                    .residual = ptrs.write_residual,
                    .block = ptrs.write_block,
                    .gateLogits = ptrs.write_gate_logits,
                    .output = ptrs.write_output,
                    .gain = ptrs.gain,
                    .normalized = ptrs.norm_output,
                };
                qwen4exp_kernels_->residualWriteBranchRMSNorm(args, nullptr);
            }
            else
            {
                qwen4exp::BranchRMSNormArgs args{
                    .extent = {1, nullptr},
                    .hidden = norm_op_.hidden,
                    .epsilon = norm_op_.epsilon,
                    .input = ptrs.norm_input,
                    .gain = ptrs.gain,
                    .output = ptrs.norm_output,
                };
                qwen4exp_kernels_->branchRMSNorm(args, nullptr);
            }
            hmark(1);

            // 2. hc_down [R, 4H] + optional hc_inject [4, 4H], sharing one Q8_1 quantization of normalized[4H].
            const float *norm_ptr = ptrs.norm_output;
            float *down_out = ptrs.down_out;
            float *inject_out = ptrs.inject_out;

            const bool share_norm_q8 =
                down_.isSingleQ8Native() && inject_params_ && inject_.isSingleQ8Native() &&
                down_.native_engine->packedWeights().numerical_policy ==
                    inject_.native_engine->packedWeights().numerical_policy;

            if (share_norm_q8)
            {
                quantizeRowToQ8InTeam(norm_ptr, norm_q8_.data(), down_.k,
                                      down_.native_engine->packedWeights().numerical_policy);
                std::array<FusedGemvDesc, 2> descs{{
                    {.packed = &down_.native_engine->packedWeights(),
                     .output = down_out,
                     .bias = nullptr,
                     .N = down_.n,
                     .policy_n = down_.policy_n},
                    {.packed = &inject_.native_engine->packedWeights(),
                     .output = inject_out,
                     .bias = nullptr,
                     .N = inject_.n,
                     .policy_n = inject_.policy_n},
                }};
                runNativeQuantisedBundleInTeam(norm_q8_.data(), descs.data(), 2, partials);
                hmark(2);
                hmark(3);
            }
            else
            {
                // Execute down.
                if (down_.isSingleQ8Native())
                {
                    quantizeRowToQ8InTeam(norm_ptr, norm_q8_.data(), down_.k,
                                          down_.native_engine->packedWeights().numerical_policy);
                    FusedGemvDesc desc{
                        .packed = &down_.native_engine->packedWeights(),
                        .output = down_out,
                        .bias = nullptr,
                        .N = down_.n,
                        .policy_n = down_.policy_n};
                    runNativeQuantisedBundleInTeam(norm_q8_.data(), &desc, 1, partials);
                }
                else if (down_.isTwoTermQ8Native())
                {
                    cpu::native_quantised::cpuQ8ResidualCorrectedProjection(
                        down_.native_engine->packedWeights(), norm_ptr, down_out,
                        DeviceRowRange::fullyActive(1),
                        std::span<Q8_1Block>(norm_q8_.data(), norm_q8_.size()),
                        std::span<Q8_1Block>(up_residual_q8_.data(), up_residual_q8_.size()));
                }
                else
                {
                    if (!runFloatGemvInTeam(
                            norm_op_.output.tensor, norm_ptr, down_,
                            dynamic_cast<TensorBase *>(down_params_.C), down_out,
                            rounded_a, &float_status_, workspace, isa))
                        return false;
                }
                hmark(2);

                // Execute optional injection projection.
                if (inject_params_)
                {
                    if (inject_.isSingleQ8Native())
                    {
                        quantizeRowToQ8InTeam(norm_ptr, norm_q8_.data(), inject_.k,
                                              inject_.native_engine->packedWeights().numerical_policy);
                        FusedGemvDesc desc{
                            .packed = &inject_.native_engine->packedWeights(),
                            .output = inject_out,
                            .bias = nullptr,
                            .N = inject_.n,
                            .policy_n = inject_.policy_n};
                        runNativeQuantisedBundleInTeam(norm_q8_.data(), &desc, 1, partials);
                    }
                    else
                    {
                        if (!runFloatGemvInTeam(
                                norm_op_.output.tensor, norm_ptr, inject_,
                                dynamic_cast<TensorBase *>(inject_params_->C), inject_out,
                                rounded_a, &float_status_, workspace, isa))
                            return false;
                    }
                }
                hmark(3);
            }

            // 3. hc_up [4H, R] (either fused TwoTermQ8 + ResidualDown or explicit ResidualDown + up).
            float *up_out = ptrs.up_out;
            if (up_.isTwoTermQ8Native() && up_params_.hc_residual_down)
            {
                float *activated_out = ptrs.activated_out;
                cpu::native_quantised::cpuHCResidualDownQ8Projection(
                    up_.native_engine->packedWeights(),
                    down_out,
                    activated_out,
                    up_out,
                    DeviceRowRange::fullyActive(1),
                    std::span<Q8_1Block>(up_primary_q8_.data(), up_primary_q8_.size()),
                    std::span<Q8_1Block>(up_residual_q8_.data(), up_residual_q8_.size()));
            }
            else
            {
                const float *up_in_ptr = down_out;
                const TensorBase *up_in_tensor = dynamic_cast<const TensorBase *>(up_params_.A);
                if (act_op_)
                {
                    qwen4exp::ResidualDownArgs rd_args{
                        .extent = {1, nullptr},
                        .lowRank = act_op_->lowRank,
                        .input = ptrs.act_input,
                        .output = ptrs.act_output,
                    };
                    qwen4exp_kernels_->residualDown(rd_args, nullptr);
                    up_in_ptr = ptrs.act_output;
                    up_in_tensor = act_op_->output.tensor;
                }
                if (up_.isSingleQ8Native())
                {
                    quantizeRowToQ8InTeam(up_in_ptr, up_primary_q8_.data(), up_.k,
                                          up_.native_engine->packedWeights().numerical_policy);
                    FusedGemvDesc desc{
                        .packed = &up_.native_engine->packedWeights(),
                        .output = up_out,
                        .bias = nullptr,
                        .N = up_.n,
                        .policy_n = up_.policy_n};
                    runNativeQuantisedBundleInTeam(up_primary_q8_.data(), &desc, 1, partials);
                }
                else if (up_.isTwoTermQ8Native())
                {
                    cpu::native_quantised::cpuQ8ResidualCorrectedProjection(
                        up_.native_engine->packedWeights(), up_in_ptr, up_out,
                        DeviceRowRange::fullyActive(1),
                        std::span<Q8_1Block>(up_primary_q8_.data(), up_primary_q8_.size()),
                        std::span<Q8_1Block>(up_residual_q8_.data(), up_residual_q8_.size()));
                }
                else
                {
                    if (!runFloatGemvInTeam(
                            up_in_tensor, up_in_ptr, up_,
                            dynamic_cast<TensorBase *>(up_params_.C), up_out,
                            rounded_a, &float_status_, workspace, isa))
                        return false;
                }
            }

            // 4. ResidualRead: mixed[H] = 0.25 * sum_b sigmoid(readLogits[b]) * normalized[b].
            qwen4exp::ResidualReadArgs rr_args{
                .extent = {1, nullptr},
                .hidden = read_op_.hidden,
                .normalized = ptrs.read_normalized,
                .gateLogits = ptrs.read_gate_logits,
                .output = ptrs.read_output,
            };
            qwen4exp_kernels_->residualRead(rr_args, nullptr);
            hmark(4);
            return true;
        }

        size_t maxPartialFloats() const
        {
            size_t m = std::max(maxPartialFloatsForHandle(down_), maxPartialFloatsForHandle(up_));
            if (inject_params_)
                m = std::max(m, maxPartialFloatsForHandle(inject_));
            return m;
        }

        int maxFloatK() const noexcept
        {
            int mk = std::max(down_.k, up_.k);
            if (inject_params_)
                mk = std::max(mk, inject_.k);
            return mk;
        }

        const TensorBase *mixedTensor() const noexcept { return read_op_.output.tensor; }

    private:
        std::optional<Qwen4ExpPrimitiveStage::ResidualWrite> write_op_;
        Qwen4ExpPrimitiveStage::BranchRMSNorm norm_op_;
        GEMMStage::Params down_params_;
        std::optional<Qwen4ExpPrimitiveStage::ResidualDown> act_op_;
        GEMMStage::Params up_params_;
        Qwen4ExpPrimitiveStage::ResidualRead read_op_;
        std::optional<GEMMStage::Params> inject_params_;
        std::unique_ptr<qwen4exp::IQwen4ExpKernels> qwen4exp_kernels_;
        CPUProjectionHandle down_;
        CPUProjectionHandle up_;
        CPUProjectionHandle inject_;
        AlignedVector<Q8_1Block> norm_q8_;
        AlignedVector<Q8_1Block> up_primary_q8_;
        AlignedVector<Q8_1Block> up_residual_q8_;
        float float_status_ = 1.0f;
    };

    // =========================================================================
    // Megakernel 1: RetainedCPUGDNStage (Single-OMP GDN Attention Layer)
    // =========================================================================

    struct CPUGDNDeclaration
    {
        GDNProjectionStage::Params proj;
        ShortConv1dStage::Params conv;
        GDNRecurrenceStage::Params rec;
        GatedRMSNormStage::Params norm;
        GEMMStage::Params out_proj;
    };

    class RetainedCPUGDNStage final : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        RetainedCPUGDNStage(const CPUHCRegionNodes &hc_region,
                            CPUGDNDeclaration decl,
                            ContractedStageMetadata meta)
            : IComputeStage(decl.proj.device_id),
              hc_(decl.proj.device_id, hc_region),
              decl_(std::move(decl)),
              meta_(std::move(meta)),
              conv_kernel_(dynamic_cast<CPUShortConvolution *>(decl_.conv.kernel)),
              rec_kernel_(dynamic_cast<CPUGatedDeltaNet *>(decl_.rec.kernel))
        {
            require(device().is_cpu() && conv_kernel_ && rec_kernel_,
                    "RetainedCPUGDNStage requires CPU ShortConv and GatedDeltaNet kernels");
            const auto &p = decl_.proj;
            projs_[0].bindFromRef(p.w_qkv, p.prepared_store, *p.prepared_ref_qkv,
                                  p.n_qkv, p.k, std::nullopt, p.qkv_operand_precision);
            projs_[1].bindFromRef(p.w_z, p.prepared_store, *p.prepared_ref_z,
                                  p.n_z, p.k);
            projs_[2].bindFromRef(p.w_a, p.prepared_store, *p.prepared_ref_a, p.n_a, p.k);
            projs_[3].bindFromRef(p.w_b, p.prepared_store, *p.prepared_ref_b, p.n_b, p.k);
            out_proj_.bindFromGEMMParams(decl_.out_proj);

            mixed_q8_.resize(static_cast<size_t>((p.k + 31) / 32));
            mixed_residual_q8_.resize(static_cast<size_t>((p.k + 31) / 32));
            out_proj_q8_.resize(static_cast<size_t>((decl_.out_proj.k + 31) / 32));

            required_partials_floats_ = std::max(hc_.maxPartialFloats(), maxPartialFloatsForHandle(out_proj_));
            for (const auto &pr : projs_)
                required_partials_floats_ = std::max(required_partials_floats_, maxPartialFloatsForHandle(pr));
            if (required_partials_floats_ > 0)
                meta_.workspace_reqs.merge(cpuProjectionPartialWorkspaceRequirements(required_partials_floats_));
            max_rounded_a_floats_ = static_cast<size_t>(std::max({hc_.maxFloatK(), p.k, decl_.out_proj.k}));
            meta_.workspace_reqs.buffers.push_back(
                {kCPUMegakernelRoundedA, max_rounded_a_floats_ * sizeof(float), 64});
        }

        ComputeStageType type() const override { return ComputeStageType::GDN_PROJECTION; }
        std::string name() const override { return "retained_cpu_gdn_megakernel"; }
        bool supportsBackend(ComputeBackendType backend) const override { return backend == ComputeBackendType::CPU; }
        StageBufferContract bufferContract() const override { return meta_.contract; }
        StageBufferRequirements getBufferRequirements() const override { return meta_.buffer_reqs; }
        WorkspaceRequirements getWorkspaceRequirements(int, int = 0, int = 0) const override { return meta_.workspace_reqs; }
        void bindWorkspace(DeviceWorkspaceManager *ws) override { workspace_ = ws; }
        void unbindWorkspace() override { workspace_ = nullptr; }
        bool hasWorkspace() const override { return workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }
        bool hasDynamicParams() const override { return true; }
        void updateDynamicParams(int /*pos_offset*/, int seq_len) override
        {
            if (seq_len != 1)
                throw std::invalid_argument("RetainedCPUGDNStage supports M=1 decode only");
        }
        void resetSessionState() override
        {
            IComputeStage::resetSessionState();
            conv_kernel_->resetGPUState(nullptr);
            rec_kernel_->resetGPUState(nullptr);
        }
        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo info;
            info.addOutput("attn_proj", decl_.out_proj.C, 1, decl_.out_proj.n);
            return info;
        }

        bool execute(IDeviceContext *ctx) override
        {
            if (!ctx || ctx->deviceId() != device())
                return false;

            // Master-thread setup before entering the single OpenMP region.
            rec_kernel_->prepareMergedDecodeScratch(
                decl_.rec.n_k_heads, decl_.rec.n_heads, decl_.rec.d_k);

            std::span<float> partials{};
            if (workspace_)
                partials = CPUInvocationWorkspace::available<float>(workspace_, kCPUProjectionPartials);
            if (partials.size() < required_partials_floats_)
            {
                if (fallback_partials_.size() < std::max(required_partials_floats_, kDefaultPartialScratchFloats))
                    fallback_partials_.resize(std::max(required_partials_floats_, kDefaultPartialScratchFloats));
                partials = std::span<float>(fallback_partials_.data(), fallback_partials_.size());
            }

            float *rounded_a = nullptr;
            if (workspace_)
            {
                auto ws_ra = CPUInvocationWorkspace::available<float>(workspace_, kCPUMegakernelRoundedA);
                if (ws_ra.size() >= max_rounded_a_floats_)
                    rounded_a = ws_ra.data();
            }
            if (!rounded_a)
            {
                if (fallback_rounded_a_.size() < max_rounded_a_floats_)
                    fallback_rounded_a_.resize(max_rounded_a_floats_);
                rounded_a = fallback_rounded_a_.data();
            }

            const auto *mixed_tensor = dynamic_cast<const TensorBase *>(decl_.proj.input);
            const float *mixed_ptr = mixed_tensor->data();
            const std::array<TensorBase *, 4> proj_out_tensors{
                dynamic_cast<TensorBase *>(decl_.proj.output_qkv),
                dynamic_cast<TensorBase *>(decl_.proj.output_z),
                dynamic_cast<TensorBase *>(decl_.proj.output_a),
                dynamic_cast<TensorBase *>(decl_.proj.output_b)};
            const std::array<float *, 4> proj_out_ptrs{
                proj_out_tensors[0]->mutable_data(),
                proj_out_tensors[1]->mutable_data(),
                proj_out_tensors[2]->mutable_data(),
                proj_out_tensors[3]->mutable_data()};

            const float *conv_weight = dynamic_cast<const TensorBase *>(decl_.conv.weight)->data();
            const float *conv_bias = decl_.conv.bias ? dynamic_cast<const TensorBase *>(decl_.conv.bias)->data() : nullptr;
            float *conv_out = dynamic_cast<TensorBase *>(decl_.conv.output)->mutable_data();
            float *conv_state = decl_.conv.conv_state;

            const float *a_log = dynamic_cast<const TensorBase *>(decl_.rec.A_log)->data();
            const float *dt_bias = dynamic_cast<const TensorBase *>(decl_.rec.dt_bias)->data();
            float *rec_out = dynamic_cast<TensorBase *>(decl_.rec.output)->mutable_data();
            float *rec_state = decl_.rec.recurrence_state;

            const auto &gn = decl_.norm;
            const float *gn_gamma = dynamic_cast<const TensorBase *>(gn.gamma)->data();
            const size_t gn_feature_dim = static_cast<size_t>(gn.feature_dim);
            const size_t gn_norm_dim = gn.norm_dim > 0 ? static_cast<size_t>(gn.norm_dim) : gn_feature_dim;
            const size_t gn_gamma_numel = gn.gamma->numel();
            const size_t gn_gamma_period = gn_gamma_numel < gn_norm_dim ? gn_gamma_numel : gn_norm_dim;
            const int gn_groups = static_cast<int>(gn_feature_dim / gn_norm_dim);

            float *out_proj_dst = dynamic_cast<TensorBase *>(decl_.out_proj.C)->mutable_data();
            const TensorBase *out_proj_in_tensor = dynamic_cast<const TensorBase *>(decl_.out_proj.A);

            // Classify the 4 GDN projections once before entering the team.
            std::array<FusedGemvDesc, 4> native_descs{};
            int native_count = 0;
            CPUProjectionNumericalPolicy shared_native_policy = CPUProjectionNumericalPolicy::BackendNative;
            bool shared_native_policy_set = false;
            bool all_native_same_policy = true;
            for (int i = 0; i < 4; ++i)
            {
                if (projs_[i].isSingleQ8Native())
                {
                    const auto pol = projs_[i].native_engine->packedWeights().numerical_policy;
                    if (!shared_native_policy_set)
                    {
                        shared_native_policy = pol;
                        shared_native_policy_set = true;
                    }
                    else if (pol != shared_native_policy)
                    {
                        all_native_same_policy = false;
                    }
                    native_descs[native_count++] = FusedGemvDesc{
                        .packed = &projs_[i].native_engine->packedWeights(),
                        .output = proj_out_ptrs[i],
                        .bias = nullptr,
                        .N = projs_[i].n,
                        .policy_n = projs_[i].policy_n};
                }
            }

            const auto hc_ptrs = hc_.resolvePointers();
            const ISALevel isa = activeISALevel();
            const bool detail_timing = PerfStatsCollector::isDomainEnabled("stage_cpu_detail");
            using Clock = std::chrono::steady_clock;
            std::array<Clock::time_point, 7> ts{};
            auto mark = [&](size_t idx) noexcept {
                if (detail_timing && omp_get_thread_num() == 0)
                    ts[idx] = Clock::now();
            };
            std::atomic<bool> ok{true};
            auto run_team = [&]() {
                mark(0);
                // 1. HC prefix -> mixed[H].
                if (!hc_.executeInTeam(hc_ptrs, partials, rounded_a, workspace_, isa))
                {
                    ok.store(false, std::memory_order_relaxed);
                }
                mark(1);

                if (ok.load(std::memory_order_relaxed))
                {
                    // 2. GDN projections: QKV, Z, A, B.
                    if (projs_[0].isTwoTermQ8Native())
                    {
                        cpu::native_quantised::cpuQ8ResidualCorrectedProjection(
                            projs_[0].native_engine->packedWeights(), mixed_ptr, proj_out_ptrs[0],
                            DeviceRowRange::fullyActive(1),
                            std::span<Q8_1Block>(mixed_q8_.data(), mixed_q8_.size()),
                            std::span<Q8_1Block>(mixed_residual_q8_.data(), mixed_residual_q8_.size()));
                    }

                    if (native_count > 0 && all_native_same_policy)
                    {
                        quantizeRowToQ8InTeam(mixed_ptr, mixed_q8_.data(), decl_.proj.k, shared_native_policy);
                        runNativeQuantisedBundleInTeam(mixed_q8_.data(), native_descs.data(), native_count, partials);
                    }
                    else if (native_count > 0)
                    {
                        for (int i = 0; i < 4; ++i)
                        {
                            if (!projs_[i].isSingleQ8Native())
                                continue;
                            quantizeRowToQ8InTeam(mixed_ptr, mixed_q8_.data(), projs_[i].k,
                                                  projs_[i].native_engine->packedWeights().numerical_policy);
                            FusedGemvDesc d{
                                .packed = &projs_[i].native_engine->packedWeights(),
                                .output = proj_out_ptrs[i],
                                .bias = nullptr,
                                .N = projs_[i].n,
                                .policy_n = projs_[i].policy_n};
                            runNativeQuantisedBundleInTeam(mixed_q8_.data(), &d, 1, partials);
                        }
                    }
                    mark(2);

                    const bool a_is_float = !projs_[2].isSingleQ8Native() && !projs_[2].isTwoTermQ8Native();
                    const bool b_is_float = !projs_[3].isSingleQ8Native() && !projs_[3].isTwoTermQ8Native();
                    bool paired_ab_done = false;
                    if (a_is_float && b_is_float)
                    {
                        paired_ab_done = runPairedFloatGemvInTeam(
                            mixed_ptr, projs_[2], proj_out_ptrs[2], projs_[3], proj_out_ptrs[3], rounded_a, isa);
                    }
                    for (int i = 0; i < 4; ++i)
                    {
                        if (projs_[i].isSingleQ8Native() || projs_[i].isTwoTermQ8Native())
                            continue;
                        if (paired_ab_done && (i == 2 || i == 3))
                            continue;
                        if (!runFloatGemvInTeam(
                                mixed_tensor, mixed_ptr, projs_[i],
                                proj_out_tensors[i], proj_out_ptrs[i],
                                rounded_a, &float_status_, workspace_, isa))
                        {
                            ok.store(false, std::memory_order_relaxed);
                        }
                    }
                }
                mark(3);

                // 3. ShortConv1d (M=1 decode, nested-safe workshare).
                if (ok.load(std::memory_order_relaxed))
                {
                    if (!conv_kernel_->forward(
                            proj_out_ptrs[0], conv_weight, conv_bias,
                            conv_out, conv_state, 1, decl_.conv.channels, decl_.conv.kernel_size, true))
                    {
                        ok.store(false, std::memory_order_relaxed);
                    }
                }
                mark(4);

                // 4. GDN Recurrence (M=1 merged-QKV fixed-epoch recurrence in team).
                if (ok.load(std::memory_order_relaxed))
                {
                    if (!rec_kernel_->executeMergedDecodeInTeam(
                            conv_out, decl_.conv.channels,
                            proj_out_ptrs[2], proj_out_ptrs[3],
                            a_log, dt_bias,
                            rec_out, rec_state,
                            decl_.rec.n_k_heads, decl_.rec.n_heads, decl_.rec.d_k, decl_.rec.d_v,
                            decl_.rec.global_v_head_offset, decl_.rec.use_qk_l2norm))
                    {
                        ok.store(false, std::memory_order_relaxed);
                    }
                }
                mark(5);

                // 5. GatedRMSNorm (in-place on rec_out with gate = Z) + 6. ssm_out projection.
                if (ok.load(std::memory_order_relaxed))
                {
#pragma omp for schedule(static)
                    for (int g = 0; g < gn_groups; ++g)
                    {
                        const size_t offset = static_cast<size_t>(g) * gn_norm_dim;
                        applyCPUGatedRMSNormGroup(
                            rec_out, proj_out_ptrs[1], gn_gamma, rec_out,
                            offset, gn_norm_dim, gn_gamma_period,
                            gn.eps, gn.subtract_one, gn.gate_activation);
                    }
                    // Implicit barrier publishes gated rec_out before ssm_out quantization.

                    if (out_proj_.isSingleQ8Native())
                    {
                        quantizeRowToQ8InTeam(rec_out, out_proj_q8_.data(), out_proj_.k,
                                              out_proj_.native_engine->packedWeights().numerical_policy);
                        FusedGemvDesc d{
                            .packed = &out_proj_.native_engine->packedWeights(),
                            .output = out_proj_dst,
                            .bias = nullptr,
                            .N = out_proj_.n,
                            .policy_n = out_proj_.policy_n};
                        runNativeQuantisedBundleInTeam(out_proj_q8_.data(), &d, 1, partials);
                    }
                    else
                    {
                        if (!runFloatGemvInTeam(
                                out_proj_in_tensor, rec_out, out_proj_,
                                dynamic_cast<TensorBase *>(decl_.out_proj.C), out_proj_dst,
                                rounded_a, &float_status_, workspace_, isa))
                        {
                            ok.store(false, std::memory_order_relaxed);
                        }
                    }
                }
                mark(6);
            };
            OMP_LAYER_PARALLEL(run_team(););

            if (detail_timing && ok.load(std::memory_order_relaxed))
            {
                const auto ns = [&](size_t a, size_t b) {
                    return static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(ts[b] - ts[a]).count());
                };
                PerfStatsCollector::recordTimingNs("mk_sub", "gdn_1_hc", ns(0, 1), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "gdn_2_qkvz", ns(1, 2), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "gdn_3_ab", ns(2, 3), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "gdn_4_conv", ns(3, 4), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "gdn_5_rec", ns(4, 5), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "gdn_6_norm_out", ns(5, 6), "decode", "cpu");
            }

            if (ok.load(std::memory_order_relaxed) && PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel", "cpu_retained_gdn_megakernel_calls", 1.0, "decode", "cpu",
                    {{"n_heads", std::to_string(decl_.rec.n_heads)},
                     {"isa", isaLevelName(isa)}});
            }
            return ok.load(std::memory_order_relaxed);
        }

    private:
        CPUHCLocalPrefix hc_;
        CPUGDNDeclaration decl_;
        ContractedStageMetadata meta_;
        CPUShortConvolution *conv_kernel_ = nullptr;
        CPUGatedDeltaNet *rec_kernel_ = nullptr;
        std::array<CPUProjectionHandle, 4> projs_{};
        CPUProjectionHandle out_proj_;
        AlignedVector<Q8_1Block> mixed_q8_;
        AlignedVector<Q8_1Block> mixed_residual_q8_;
        AlignedVector<Q8_1Block> out_proj_q8_;
        size_t required_partials_floats_ = 0;
        size_t max_rounded_a_floats_ = 0;
        AlignedVector<float> fallback_partials_;
        AlignedVector<float> fallback_rounded_a_;
        float float_status_ = 1.0f;
        DeviceWorkspaceManager *workspace_ = nullptr;
    };

    // =========================================================================
    // Megakernel 2: RetainedCPUQSAAttentionStage (Single-OMP QSA Attention Layer)
    // =========================================================================

    template <typename BlockType>
    inline void quantizeFP32ToQ16BlocksSerial(const float *src_data, void *dst_blocks,
                                              size_t total_elements, size_t n_blocks) noexcept
    {
        constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
        BlockType *blocks = static_cast<BlockType *>(dst_blocks);
        for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx)
        {
            const size_t offset = block_idx * BLOCK_SIZE;
            const size_t count = std::min(BLOCK_SIZE, total_elements - offset);
            BlockType &block = blocks[block_idx];
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
                max_abs = std::max(max_abs, std::abs(src_data[offset + i]));
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d;
            int64_t sum_i64 = 0;
            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                for (size_t i = 0; i < count; ++i)
                {
                    const float scaled = src_data[offset + i] * inv_d;
                    const float clamped = std::max(-32767.0f, std::min(32767.0f, scaled));
                    const int16_t quantized = static_cast<int16_t>(std::round(clamped));
                    block.qs[i] = quantized;
                    sum_i64 += static_cast<int64_t>(quantized);
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                    block.qs[i] = 0;
            }
            for (size_t i = count; i < BLOCK_SIZE; ++i)
                block.qs[i] = 0;
            block.sum_qs = static_cast<int32_t>(sum_i64);
        }
    }

    inline bool quantizeFP32ToQ16ScratchSerial(Q16_1Tensor &dst, void *raw_blocks, const float *src_data) noexcept
    {
        if (!src_data || !raw_blocks)
            return false;
        const size_t total_elements = dst.rows() * dst.cols();
        const auto bs_enum = dst.q16_block_size();
        const size_t bs = static_cast<size_t>(bs_enum);
        const size_t n_blocks = (total_elements + bs - 1) / bs;
        switch (bs_enum)
        {
        case Q16BlockSize::BLOCK_32:
            quantizeFP32ToQ16BlocksSerial<Q16_1Block>(src_data, raw_blocks, total_elements, n_blocks);
            return true;
        case Q16BlockSize::BLOCK_64:
            quantizeFP32ToQ16BlocksSerial<Q16_1Block_64>(src_data, raw_blocks, total_elements, n_blocks);
            return true;
        case Q16BlockSize::BLOCK_128:
            quantizeFP32ToQ16BlocksSerial<Q16_1Block_128>(src_data, raw_blocks, total_elements, n_blocks);
            return true;
        default:
            return false;
        }
    }

    struct CPUQSADeclaration
    {
        FusedQKVGEMMStage::Params qkv;
        QGateSplitStage::Params split;
        std::optional<QKNormStage::Params> q_norm;
        std::optional<QKNormStage::Params> k_norm;
        RoPEStage::Params rope;
        QwenSparseAttentionStage::Params replay_stage;
        KVCacheAppendStage::Params kv_append;
        GEMMStage::Params key_proj;
        std::optional<GEMMStage::Params> query_proj;
        QwenSparseAttentionStage::Params indexer_stage;
        QwenSparseAttentionStage::Params attn_stage;
        std::optional<QwenSparseAttentionStage::Params> commit_stage;
        AttentionOutputGateStage::Params gate;
        GEMMStage::Params wo;
    };

    class RetainedCPUQSAAttentionStage final : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        RetainedCPUQSAAttentionStage(const CPUHCRegionNodes &hc_region,
                                     CPUQSADeclaration decl,
                                     ContractedStageMetadata meta)
            : IComputeStage(decl.qkv.device_id),
              hc_(decl.qkv.device_id, hc_region),
              decl_(std::move(decl)),
              meta_(std::move(meta)),
              qsa_kernels_(qwen4exp::createCPUQwenSparseAttentionKernels()),
              fallback_kv_append_stage_(decl_.kv_append)
        {
            require(device().is_cpu() && qsa_kernels_ != nullptr,
                    "RetainedCPUQSAAttentionStage requires CPU QSA kernels");
            const auto &qkv = decl_.qkv;
            qkv_projs_[0].bindFromRef(qkv.wq, qkv.prepared_store, *qkv.prepared_ref_q,
                                      qkv.n_q, qkv.k, qkv.original_output_slice_q);
            qkv_projs_[1].bindFromRef(qkv.wk, qkv.prepared_store, *qkv.prepared_ref_k,
                                      qkv.n_k, qkv.k, qkv.original_output_slice_k);
            qkv_projs_[2].bindFromRef(qkv.wv, qkv.prepared_store, *qkv.prepared_ref_v,
                                      qkv.n_v, qkv.k, qkv.original_output_slice_v);
            key_proj_.bindFromGEMMParams(decl_.key_proj);
            if (decl_.query_proj)
            {
                query_proj_.emplace().bindFromGEMMParams(*decl_.query_proj);
                if (decl_.query_proj->projection_row_work)
                {
                    query_proj_row_work_ = std::make_unique<ProjectionRowWorkBinding>(
                        *decl_.query_proj->projection_row_work,
                        device(),
                        1,
                        decl_.query_proj->verifier_row_range,
                        decl_.query_proj->request_rows);
                }
            }
            wo_proj_.bindFromGEMMParams(decl_.wo);

            if (decl_.q_norm)
                q_norm_kernel_ = llaminar::v2::kernels::KernelFactory::getOrCreateRMSNorm(
                    dynamic_cast<const TensorBase *>(decl_.q_norm->input), device());
            if (decl_.k_norm)
                k_norm_kernel_ = llaminar::v2::kernels::KernelFactory::getOrCreateRMSNorm(
                    dynamic_cast<const TensorBase *>(decl_.k_norm->input), device());

            const auto &idx = std::get<QwenSparseAttentionStage::Indexer>(decl_.indexer_stage.operation);
            indexer_freqs_ = qwen4exp::makeQwenSparsePowerFrequencies(idx.ropeTheta, idx.rotaryDim);
            key_neg_log_theta_ = -std::log(decl_.rope.theta_base);

            const int eff_rotary = (decl_.rope.partial_rotary_factor > 0.0f &&
                                    decl_.rope.partial_rotary_factor < 1.0f)
                ? static_cast<int>(std::round(decl_.rope.head_dim * decl_.rope.partial_rotary_factor))
                : decl_.rope.head_dim;
            rope_eff_rotary_ = eff_rotary;
            const auto angle_layout = primitives::rope_angle_layout(1, eff_rotary / 2);
            rope_angle_storage_.resize(angle_layout.bytes);

            mixed_q8_.resize(static_cast<size_t>((qkv.k + 31) / 32));
            wo_q8_.resize(static_cast<size_t>((decl_.wo.k + 31) / 32));

            required_partials_floats_ = std::max({
                hc_.maxPartialFloats(),
                maxPartialFloatsForHandle(qkv_projs_[0]),
                maxPartialFloatsForHandle(qkv_projs_[1]),
                maxPartialFloatsForHandle(qkv_projs_[2]),
                maxPartialFloatsForHandle(key_proj_),
                query_proj_ ? maxPartialFloatsForHandle(*query_proj_) : size_t{0},
                maxPartialFloatsForHandle(wo_proj_)});
            if (required_partials_floats_ > 0)
                meta_.workspace_reqs.merge(cpuProjectionPartialWorkspaceRequirements(required_partials_floats_));
            max_rounded_a_floats_ = static_cast<size_t>(std::max({hc_.maxFloatK(), qkv.k, decl_.wo.k}));
            meta_.workspace_reqs.buffers.push_back(
                {kCPUMegakernelRoundedA, max_rounded_a_floats_ * sizeof(float), 64});

            // Pre-allocate persistent 1-row Q16_1 scratch tensors when cache is Q16_1
            // so M=1 decode performs zero heap allocations during KV append.
            const auto k_prec = decl_.kv_append.kv_cache->k_precision();
            const auto v_prec = decl_.kv_append.kv_cache->v_precision();
            if (k_prec == ActivationPrecision::Q16_1 && v_prec == ActivationPrecision::Q16_1)
            {
                const size_t kv_dim = decl_.kv_append.K->cols();
                const auto bs = optimal_q16_block_size(decl_.kv_append.head_dim);
                q16_k_scratch_ = std::make_unique<Q16_1Tensor>(std::vector<size_t>{1u, kv_dim}, bs);
                q16_v_scratch_ = std::make_unique<Q16_1Tensor>(std::vector<size_t>{1u, kv_dim}, bs);
            }
        }

        ComputeStageType type() const override { return ComputeStageType::GEMM_FUSED_QKV; }
        std::string name() const override { return "retained_cpu_qsa_megakernel"; }
        bool supportsBackend(ComputeBackendType backend) const override { return backend == ComputeBackendType::CPU; }
        StageBufferContract bufferContract() const override { return meta_.contract; }
        StageBufferRequirements getBufferRequirements() const override { return meta_.buffer_reqs; }
        WorkspaceRequirements getWorkspaceRequirements(int, int = 0, int = 0) const override { return meta_.workspace_reqs; }
        void bindWorkspace(DeviceWorkspaceManager *ws) override
        {
            workspace_ = ws;
            if (query_proj_row_work_ && ws)
                query_proj_row_work_->prepare(ws, nullptr);
        }
        void unbindWorkspace() override { workspace_ = nullptr; }
        bool hasWorkspace() const override { return workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }
        bool hasDynamicParams() const override { return true; }
        bool supportsDeviceResidentDynamicPositionReplay() const override { return true; }
        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            decl_.rope.pos_offset = pos_offset;
            decl_.rope.seq_len = seq_len;
            decl_.rope.position_ids = nullptr;
            position_ids_cache_.clear();
        }
        void updateDynamicPositionIds(const int *position_ids, int seq_len) override
        {
            decl_.rope.seq_len = seq_len;
            position_ids_cache_.clear();
            if (position_ids && seq_len > 0)
            {
                position_ids_cache_.assign(position_ids, position_ids + seq_len);
                decl_.rope.position_ids = position_ids_cache_.data();
                decl_.rope.pos_offset = position_ids_cache_.front();
            }
            else
            {
                decl_.rope.position_ids = nullptr;
                decl_.rope.pos_offset = 0;
            }
        }
        void resetSessionState() override
        {
            IComputeStage::resetSessionState();
            decl_.rope.pos_offset = 0;
            decl_.rope.position_ids = nullptr;
            position_ids_cache_.clear();
        }
        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo info;
            info.addOutput("attn_proj", decl_.wo.C, 1, decl_.wo.n);
            return info;
        }

        bool execute(IDeviceContext *ctx) override
        {
            if (!ctx || ctx->deviceId() != device())
                return false;

            std::span<float> partials{};
            if (workspace_)
                partials = CPUInvocationWorkspace::available<float>(workspace_, kCPUProjectionPartials);
            if (partials.size() < required_partials_floats_)
            {
                if (fallback_partials_.size() < std::max(required_partials_floats_, kDefaultPartialScratchFloats))
                    fallback_partials_.resize(std::max(required_partials_floats_, kDefaultPartialScratchFloats));
                partials = std::span<float>(fallback_partials_.data(), fallback_partials_.size());
            }

            float *rounded_a = nullptr;
            if (workspace_)
            {
                auto ws_ra = CPUInvocationWorkspace::available<float>(workspace_, kCPUMegakernelRoundedA);
                if (ws_ra.size() >= max_rounded_a_floats_)
                    rounded_a = ws_ra.data();
            }
            if (!rounded_a)
            {
                if (fallback_rounded_a_.size() < max_rounded_a_floats_)
                    fallback_rounded_a_.resize(max_rounded_a_floats_);
                rounded_a = fallback_rounded_a_.data();
            }

            const auto *mixed_tensor = dynamic_cast<const TensorBase *>(decl_.qkv.input);
            const float *mixed_ptr = mixed_tensor->data();
            auto *q_raw_tensor = dynamic_cast<TensorBase *>(decl_.qkv.output_q);
            auto *k_tensor = dynamic_cast<TensorBase *>(decl_.qkv.output_k);
            auto *v_tensor = dynamic_cast<TensorBase *>(decl_.qkv.output_v);
            auto *q_split_tensor = dynamic_cast<TensorBase *>(decl_.split.output_q);
            auto *gate_split_tensor = dynamic_cast<TensorBase *>(decl_.split.output_gate);
            auto *key_out_tensor = dynamic_cast<TensorBase *>(decl_.key_proj.C);
            auto *query_out_tensor = decl_.query_proj ? dynamic_cast<TensorBase *>(decl_.query_proj->C) : nullptr;

            float *q_raw_ptr = q_raw_tensor->mutable_data();
            float *k_ptr = k_tensor->mutable_data();
            float *v_ptr = v_tensor->mutable_data();
            float *q_split_ptr = q_split_tensor->mutable_data();
            float *gate_split_ptr = gate_split_tensor->mutable_data();
            float *key_out_ptr = key_out_tensor->mutable_data();
            float *query_out_ptr = query_out_tensor ? query_out_tensor->mutable_data() : nullptr;

            const auto &prep = std::get<QwenSparseAttentionStage::PrepareReplay>(decl_.replay_stage.operation);
            const auto &idx = std::get<QwenSparseAttentionStage::Indexer>(decl_.indexer_stage.operation);
            const auto &sel = std::get<QwenSparseAttentionStage::SelectedAttention>(decl_.attn_stage.operation);
            const auto *commit_op = decl_.commit_stage
                ? &std::get<QwenSparseAttentionStage::CommitPartial>(decl_.commit_stage->operation)
                : nullptr;

            auto *replay_ptr = reinterpret_cast<qwen4exp::QwenSparseReplay *>(
                bindingIntPtr(decl_.replay_stage.replay));
            const int rope_pos = decl_.rope.position_ids ? decl_.rope.position_ids[0] : decl_.rope.pos_offset;
            const auto angles = primitives::carve_rope_angle_rows(
                std::span<std::byte>(rope_angle_storage_.data(), rope_angle_storage_.size()),
                1, rope_eff_rotary_ / 2);

            // PrepareReplay on master thread BEFORE entering the OpenMP team so `replay`
            // is immediately ready for ranking-only qsa_query_proj gating and QSA indexer!
            {
                int cpu_cursor = prep.cache.ring_head(prep.layer, prep.sequence);
                int cpu_count = prep.cache.get_cached_tokens(prep.layer, prep.sequence);
                qwen4exp::QwenSparsePreAppendSequenceView seq_view{};
                seq_view.capacity = decl_.replay_stage.tokenCapacity;
                seq_view.cursor = &cpu_cursor;
                seq_view.cached_tokens = &cpu_count;
                seq_view.encoding = qwen4exp::QwenSparseRingCursor::OldestRow;
                const auto *pos_input = std::get_if<Qwen4ExpIntInput>(&prep.positionIds);
                const int first_pos = pos_input
                    ? 0
                    : std::get<QwenSparseAttentionStage::PrepareReplay::ContiguousPositions>(prep.positionIds).first;
                qsa_kernels_->prepareReplay(
                    {seq_view,
                     prep.liveRows ? resolveIntInputPtr(*prep.liveRows) : nullptr,
                     pos_input ? resolveIntInputPtr(*pos_input) : nullptr,
                     replay_ptr, nullptr, decl_.replay_stage.rowCapacity, first_pos},
                    nullptr);
            }

            // Evaluate whether optional qsa_query_proj is live for this token's visible prefix.
            bool run_query_proj = false;
            if (decl_.query_proj)
            {
                if (query_proj_row_work_)
                {
                    if (!query_proj_row_work_->prepared())
                        query_proj_row_work_->prepare(workspace_, nullptr);
                    query_proj_row_work_->requirePrepared(workspace_, nullptr);
                    query_proj_row_work_->requirePublishedInputs();
                    const cpu::native_quantised::CPUProjectionRowGrid grid(
                        1, &query_proj_row_work_->work());
                    run_query_proj = grid.active(0);
                }
                else
                {
                    run_query_proj = true;
                }
            }

            // Build fused NativeQuantised bundle sharing mixed_q8 across Q, K, V, qsa_key, [qsa_query].
            std::array<FusedGemvDesc, 5> native_descs{};
            std::array<bool, 5> is_in_native_bundle{false, false, false, false, false};
            int native_count = 0;
            CPUProjectionNumericalPolicy shared_policy = CPUProjectionNumericalPolicy::BackendNative;
            bool shared_policy_set = false;
            auto tryAddToBundle = [&](int slot, const CPUProjectionHandle &h, float *dst)
            {
                if (!h.isSingleQ8Native())
                    return;
                const auto pol = h.native_engine->packedWeights().numerical_policy;
                if (!shared_policy_set)
                {
                    shared_policy = pol;
                    shared_policy_set = true;
                }
                else if (pol != shared_policy)
                {
                    return;
                }
                is_in_native_bundle[slot] = true;
                native_descs[native_count++] = FusedGemvDesc{
                    .packed = &h.native_engine->packedWeights(),
                    .output = dst,
                    .bias = nullptr,
                    .N = h.n,
                    .policy_n = h.policy_n};
            };
            tryAddToBundle(0, qkv_projs_[0], q_raw_ptr);
            tryAddToBundle(1, qkv_projs_[1], k_ptr);
            tryAddToBundle(2, qkv_projs_[2], v_ptr);
            tryAddToBundle(3, key_proj_, key_out_ptr);
            if (run_query_proj && query_proj_)
                tryAddToBundle(4, *query_proj_, query_out_ptr);

            const auto k_prec = decl_.kv_append.kv_cache->k_precision();
            const auto v_prec = decl_.kv_append.kv_cache->v_precision();
            const bool kv_is_q16 = (k_prec == ActivationPrecision::Q16_1 && v_prec == ActivationPrecision::Q16_1);
            const bool kv_is_aq8 = (k_prec == ActivationPrecision::AQ8);
            void *q16_k_raw = (kv_is_q16 && q16_k_scratch_) ? q16_k_scratch_->raw_mutable_data() : nullptr;
            void *q16_v_raw = (kv_is_q16 && q16_v_scratch_) ? q16_v_scratch_->raw_mutable_data() : nullptr;

            const float *q_gamma_ptr = (decl_.q_norm && decl_.q_norm->gamma)
                ? dynamic_cast<const TensorBase *>(decl_.q_norm->gamma)->data()
                : nullptr;
            const float *k_gamma_ptr = (decl_.k_norm && decl_.k_norm->gamma)
                ? dynamic_cast<const TensorBase *>(decl_.k_norm->gamma)->data()
                : nullptr;
            const ISALevel isa = activeISALevel();

            float *attn_out_ptr = bindingFloatPtr(sel.output);
            const float *sel_query_ptr = bindingConstFloatPtr(sel.query);
            const std::int32_t *sel_selected_ptr = bindingConstIntPtr(sel.selected);
            const std::int32_t *sel_selected_count_ptr = bindingConstIntPtr(sel.selectedCount);
            const float *sel_gate_ptr = sel.gate ? bindingConstFloatPtr(*sel.gate) : nullptr;
            const float *commit_prior_ptr = commit_op ? bindingConstFloatPtr(commit_op->priorPartial) : nullptr;
            const float *commit_raw_ptr = commit_op ? bindingConstFloatPtr(commit_op->rawKey) : nullptr;
            const std::int32_t *commit_accepted_ptr = commit_op ? resolveIntInputPtr(commit_op->acceptedRows) : nullptr;
            float *commit_out_ptr = commit_op ? bindingFloatPtr(commit_op->outputPartial) : nullptr;
            const float *gate_in_ptr = dynamic_cast<const TensorBase *>(decl_.gate.gate)->data();
            float *gate_out_ptr = dynamic_cast<TensorBase *>(decl_.gate.output)->mutable_data();
            const int gate_cols = decl_.gate.feature_dim;
            float *wo_dst = dynamic_cast<TensorBase *>(decl_.wo.C)->mutable_data();
            const TensorBase *wo_in_tensor = dynamic_cast<const TensorBase *>(decl_.wo.A);

            // Pre-build immutable QSA indexer and commit structs.
            const bool no_ranking = (idx.queryDemand == qwen4exp::QwenSparseQueryDemand::ProvenNoRanking);
            const qwen4exp::QwenSparseIndexer indexer_args{
                .replay = replay_ptr,
                .query = no_ranking ? nullptr : bindingConstFloatPtr(idx.query),
                .raw_keys = bindingConstFloatPtr(idx.rawKey),
                .partial_keys = bindingConstFloatPtr(idx.priorPartial),
                .pooled_keys = bindingFloatPtr(idx.pooledKeys),
                .query_norm = no_ranking ? nullptr : idx.queryNorm.tensor->data(),
                .key_norm = idx.keyNorm.tensor->data(),
                .cosine = idx.cosine ? idx.cosine->tensor->data() : nullptr,
                .sine = idx.sine ? idx.sine->tensor->data() : nullptr,
                .prepared_query = no_ranking ? nullptr : bindingFloatPtr(idx.preparedQuery),
                .scores = no_ranking ? nullptr : bindingFloatPtr(idx.scores),
                .selected = bindingIntPtr(idx.selected),
                .selected_count = bindingIntPtr(idx.selectedCount),
                .row_capacity = decl_.indexer_stage.rowCapacity,
                .token_capacity = decl_.indexer_stage.tokenCapacity,
                .block_capacity = idx.blockCapacity,
                .position_capacity = idx.positionCapacity,
                .rotary_dim = idx.rotaryDim,
                .norm_epsilon = idx.epsilon,
                .representation = idx.representation,
                .rotary_source = idx.rotarySource,
                .rope_theta = idx.ropeTheta,
                .inverse_frequencies = indexer_freqs_,
                .transaction_prior = nullptr,
                .pool_selection = idx.poolSelection,
                .query_demand = idx.queryDemand,
            };

            const auto hc_ptrs = hc_.resolvePointers();
            qwen4exp::QwenSparseKV shared_kv_view{};
            const bool detail_timing = PerfStatsCollector::isDomainEnabled("stage_cpu_detail");
            using Clock = std::chrono::steady_clock;
            std::array<Clock::time_point, 7> ts{};
            auto mark = [&](size_t idx) noexcept {
                if (detail_timing && omp_get_thread_num() == 0)
                    ts[idx] = Clock::now();
            };
            std::atomic<bool> ok{true};

            auto run_team = [&]() {
                mark(0);
                // 1. HC prefix -> mixed[H].
                if (!hc_.executeInTeam(hc_ptrs, partials, rounded_a, workspace_, isa))
                    ok.store(false, std::memory_order_relaxed);
                mark(1);

                // 2. Fused Q, K, V, qsa_key, [qsa_query] projections sharing ONE mixed_q8 quantization!
                if (ok.load(std::memory_order_relaxed))
                {
                    if (native_count > 0)
                    {
                        quantizeRowToQ8InTeam(mixed_ptr, mixed_q8_.data(), decl_.qkv.k, shared_policy);
                        runNativeQuantisedBundleInTeam(mixed_q8_.data(), native_descs.data(), native_count, partials);
                    }
                    bool paired_kq_done = false;
                    if (!is_in_native_bundle[3] && !key_proj_.isSingleQ8Native() &&
                        run_query_proj && query_proj_ && !is_in_native_bundle[4] && !query_proj_->isSingleQ8Native())
                    {
                        paired_kq_done = runPairedFloatGemvInTeam(
                            mixed_ptr, key_proj_, key_out_ptr, *query_proj_, query_out_ptr, rounded_a, isa);
                    }
                    auto runFallbackProj = [&](int slot, const CPUProjectionHandle &h, TensorBase *dst_tensor, float *dst_ptr)
                    {
                        if (is_in_native_bundle[slot])
                            return;
                        if (h.isSingleQ8Native())
                        {
                            quantizeRowToQ8InTeam(mixed_ptr, mixed_q8_.data(), h.k,
                                                  h.native_engine->packedWeights().numerical_policy);
                            FusedGemvDesc d{
                                .packed = &h.native_engine->packedWeights(),
                                .output = dst_ptr,
                                .bias = nullptr,
                                .N = h.n,
                                .policy_n = h.policy_n};
                            runNativeQuantisedBundleInTeam(mixed_q8_.data(), &d, 1, partials);
                        }
                        else if (!runFloatGemvInTeam(
                                     mixed_tensor, mixed_ptr, h, dst_tensor, dst_ptr,
                                     rounded_a, &float_status_, workspace_, isa))
                        {
                            ok.store(false, std::memory_order_relaxed);
                        }
                    };
                    runFallbackProj(0, qkv_projs_[0], q_raw_tensor, q_raw_ptr);
                    runFallbackProj(1, qkv_projs_[1], k_tensor, k_ptr);
                    runFallbackProj(2, qkv_projs_[2], v_tensor, v_ptr);
                    if (!paired_kq_done)
                    {
                        runFallbackProj(3, key_proj_, key_out_tensor, key_out_ptr);
                        if (run_query_proj && query_proj_)
                            runFallbackProj(4, *query_proj_, query_out_tensor, query_out_ptr);
                    }
                }
                mark(2);

                // 3+4. Fused QGateSplit + Q_RMSNorm + K_RMSNorm in ONE parallel loop (zero races, 1 barrier!).
                if (ok.load(std::memory_order_relaxed))
                {
                    const int nh = decl_.split.n_heads;
                    const int hd = decl_.split.head_dim;
                    const int nkh = (decl_.k_norm && k_gamma_ptr) ? decl_.k_norm->n_heads : 0;
                    const int total_norm_heads = nh + nkh;
                    const float q_eps = decl_.q_norm ? decl_.q_norm->eps : 1e-6f;
                    const float k_eps = decl_.k_norm ? decl_.k_norm->eps : 1e-6f;
#pragma omp for schedule(static)
                    for (int h = 0; h < total_norm_heads; ++h)
                    {
                        if (h < nh)
                        {
                            const float *src = q_raw_ptr + static_cast<size_t>(h) * static_cast<size_t>(2 * hd);
                            float *q_dst = q_split_ptr + static_cast<size_t>(h) * static_cast<size_t>(hd);
                            std::memcpy(gate_split_ptr + static_cast<size_t>(h) * static_cast<size_t>(hd),
                                        src + hd, static_cast<size_t>(hd) * sizeof(float));
                            if (q_gamma_ptr)
                            {
                                primitives::rmsnorm_fused_vectorized_row(
                                    src, q_gamma_ptr, q_dst, static_cast<size_t>(hd), q_eps);
                            }
                            else
                            {
                                std::memcpy(q_dst, src, static_cast<size_t>(hd) * sizeof(float));
                            }
                        }
                        else
                        {
                            const int kh = h - nh;
                            float *k_dst = k_ptr + static_cast<size_t>(kh) * static_cast<size_t>(decl_.k_norm->head_dim);
                            primitives::rmsnorm_fused_vectorized_row(
                                k_dst, k_gamma_ptr, k_dst,
                                static_cast<size_t>(decl_.k_norm->head_dim), k_eps);
                        }
                    }

                    // 5. RoPE (nested-safe via OMP_WORKSHARE_REGION).
                    if (rope_pos >= 0)
                    {
                        float *k_rope = decl_.rope.skip_k ? nullptr : k_ptr;
                        if (rope_eff_rotary_ < decl_.rope.head_dim)
                        {
                            primitives::apply_rope_partial(
                                q_split_ptr, k_rope, 1, decl_.rope.head_dim, rope_eff_rotary_,
                                decl_.rope.n_heads, decl_.rope.n_kv_heads,
                                rope_pos, decl_.rope.theta_base, angles);
                        }
                        else
                        {
                            primitives::apply_rope_vectorized(
                                q_split_ptr, k_rope, 1, decl_.rope.head_dim,
                                decl_.rope.n_heads, decl_.rope.n_kv_heads,
                                rope_pos, decl_.rope.theta_base, angles);
                        }
                    }
                }
                mark(3);

                // 6. KV Cache Append (supports Q16_1, AQ8, FP16, FP32, Q8_1, TQ4, TQ8 with zero extra forks).
                if (ok.load(std::memory_order_relaxed))
                {
                    if (kv_is_q16)
                    {
                        auto append_q16 = [&]() {
                            if (!quantizeFP32ToQ16ScratchSerial(*q16_k_scratch_, q16_k_raw, k_ptr) ||
                                !quantizeFP32ToQ16ScratchSerial(*q16_v_scratch_, q16_v_raw, v_ptr) ||
                                !decl_.kv_append.kv_cache->append(
                                    decl_.kv_append.layer_idx, decl_.kv_append.seq_idx,
                                    q16_k_scratch_.get(), q16_v_scratch_.get(), 1))
                            {
                                ok.store(false, std::memory_order_relaxed);
                            }
                        };
                        OMP_SINGLE(append_q16);
                    }
                    else if (kv_is_aq8)
                    {
                        if (!decl_.kv_append.kv_cache->append(
                                decl_.kv_append.layer_idx, decl_.kv_append.seq_idx,
                                decl_.kv_append.K, decl_.kv_append.V, 1))
                        {
                            ok.store(false, std::memory_order_relaxed);
                        }
                    }
                    else
                    {
                        OMP_SINGLE([&]() {
                            if (!fallback_kv_append_stage_.execute(ctx))
                                ok.store(false, std::memory_order_relaxed);
                        });
                    }
                }
                mark(4);

                // 7. QSA Indexer + 8. SelectedAttention + 9. optional CommitPartial.
                if (ok.load(std::memory_order_relaxed))
                {
                    qsa_kernels_->indexer(indexer_args, nullptr);

                    OMP_SINGLE([&]() {
                        shared_kv_view = sel.native.cache.acquireSparseAttentionNativeView(
                            sel.native.layer, sel.native.sequence, nullptr, sel.native.keyPolicy, nullptr);
                        shared_kv_view.key_rotary_source = sel.keyRotarySource;
                        shared_kv_view.key_rotary_dim = sel.keyRotaryDim;
                        shared_kv_view.key_negative_log_theta = key_neg_log_theta_;
                        if (sel.keyCosine)
                        {
                            shared_kv_view.key_cosine = sel.keyCosine->tensor->data();
                            shared_kv_view.key_sine = sel.keySine->tensor->data();
                            shared_kv_view.key_position_capacity = sel.keyPositionCapacity;
                        }
                    });

                    const qwen4exp::QwenSparseAttention attn_args{
                        .replay = replay_ptr,
                        .query = sel_query_ptr,
                        .kv = shared_kv_view,
                        .selected = sel_selected_ptr,
                        .selected_count = sel_selected_count_ptr,
                        .gate = sel_gate_ptr,
                        .output = attn_out_ptr,
                        .gpu_partials = nullptr,
                        .observations = nullptr,
                        .row_capacity = decl_.attn_stage.rowCapacity,
                        .token_capacity = decl_.attn_stage.tokenCapacity,
                        .query_heads = sel.queryHeads,
                        .kv_heads = sel.kvHeads,
                        .query_head_start = sel.queryHeadStart,
                        .kv_head_start = sel.kvHeadStart,
                        .arithmetic = sel.arithmetic,
                    };
                    qsa_kernels_->attention(attn_args, nullptr);

                    if (commit_op)
                    {
                        qsa_kernels_->commitPartial(
                            {replay_ptr,
                             commit_prior_ptr,
                             commit_raw_ptr,
                             AcceptedRequestPrefixes::counted(
                                 DeviceRequestRowRanges::deviceCounted(
                                     1, decl_.commit_stage->rowCapacity, &replay_ptr->live_rows),
                                 commit_accepted_ptr),
                             commit_out_ptr},
                            nullptr);
                    }
                }
                mark(5);

                // 10. AttentionOutputGate (parallel 256-element SIMD blocks) + 11. Wo projection.
                if (ok.load(std::memory_order_relaxed))
                {
                    constexpr int kGateBlock = 256;
                    const int n_blocks = (gate_cols + kGateBlock - 1) / kGateBlock;
#pragma omp for schedule(static)
                    for (int blk = 0; blk < n_blocks; ++blk)
                    {
                        const int off = blk * kGateBlock;
                        const size_t cnt = static_cast<size_t>(std::min(kGateBlock, gate_cols - off));
                        applyCPUAttentionOutputGateRange(
                            gate_in_ptr + off, attn_out_ptr + off, gate_out_ptr + off, cnt);
                    }
                    // Implicit barrier publishes gated attention output before Wo quantization.

                    if (wo_proj_.isSingleQ8Native())
                    {
                        quantizeRowToQ8InTeam(gate_out_ptr, wo_q8_.data(), wo_proj_.k,
                                              wo_proj_.native_engine->packedWeights().numerical_policy);
                        FusedGemvDesc d{
                            .packed = &wo_proj_.native_engine->packedWeights(),
                            .output = wo_dst,
                            .bias = nullptr,
                            .N = wo_proj_.n,
                            .policy_n = wo_proj_.policy_n};
                        runNativeQuantisedBundleInTeam(wo_q8_.data(), &d, 1, partials);
                    }
                    else if (!runFloatGemvInTeam(
                                 wo_in_tensor, gate_out_ptr, wo_proj_,
                                 dynamic_cast<TensorBase *>(decl_.wo.C), wo_dst,
                                 rounded_a, &float_status_, workspace_, isa))
                    {
                        ok.store(false, std::memory_order_relaxed);
                    }
                }
                mark(6);
            };
            OMP_LAYER_PARALLEL(run_team(););

            if (detail_timing && ok.load(std::memory_order_relaxed))
            {
                const auto ns = [&](size_t a, size_t b) {
                    return static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(ts[b] - ts[a]).count());
                };
                PerfStatsCollector::recordTimingNs("mk_sub", "qsa_1_hc", ns(0, 1), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "qsa_2_qkv_key", ns(1, 2), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "qsa_3_norm_rope", ns(2, 3), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "qsa_4_kv_append", ns(3, 4), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "qsa_5_idx_attn", ns(4, 5), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "qsa_6_gate_wo", ns(5, 6), "decode", "cpu");
            }

            if (ok.load(std::memory_order_relaxed) && PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel", "cpu_retained_qsa_megakernel_calls", 1.0, "decode", "cpu",
                    {{"query_heads", std::to_string(sel.queryHeads)},
                     {"isa", isaLevelName(isa)}});
            }
            return ok.load(std::memory_order_relaxed);
        }

    private:
        CPUHCLocalPrefix hc_;
        CPUQSADeclaration decl_;
        ContractedStageMetadata meta_;
        std::unique_ptr<qwen4exp::IQwenSparseAttentionKernels> qsa_kernels_;
        KVCacheAppendStage fallback_kv_append_stage_;
        ITensorRMSNorm *q_norm_kernel_ = nullptr;
        ITensorRMSNorm *k_norm_kernel_ = nullptr;
        std::unique_ptr<ProjectionRowWorkBinding> query_proj_row_work_;
        std::array<CPUProjectionHandle, 3> qkv_projs_{};
        CPUProjectionHandle key_proj_;
        std::optional<CPUProjectionHandle> query_proj_;
        CPUProjectionHandle wo_proj_;
        qwen4exp::QwenSparsePowerFrequencies indexer_freqs_{};
        float key_neg_log_theta_ = 0.0f;
        int rope_eff_rotary_ = 0;
        AlignedVector<std::byte> rope_angle_storage_;
        AlignedVector<Q8_1Block> mixed_q8_;
        AlignedVector<Q8_1Block> wo_q8_;
        size_t required_partials_floats_ = 0;
        size_t max_rounded_a_floats_ = 0;
        AlignedVector<float> fallback_partials_;
        AlignedVector<float> fallback_rounded_a_;
        float float_status_ = 1.0f;
        std::unique_ptr<Q16_1Tensor> q16_k_scratch_;
        std::unique_ptr<Q16_1Tensor> q16_v_scratch_;
        std::vector<int> position_ids_cache_;
        DeviceWorkspaceManager *workspace_ = nullptr;
    };

    // =========================================================================
    // Megakernel 3: RetainedCPUFFNStage (Single-OMP FFN / MoE Layer)
    // =========================================================================

    struct CPUFFNDeclaration
    {
        MoERoutingStage::Params router;
        SharedExpertFFNStage::Params shared;
        std::optional<MoEExpertComputeStage::Params> routed;
        std::optional<SharedExpertGateStage::Params> shared_gate;
    };

    inline void applySharedGateAddSlice(
        float *shared_row,
        const float *residual_row,
        float *combined_row,
        float gate,
        int count,
        ISALevel isa) noexcept
    {
        int j = 0;
#if defined(__AVX512F__)
        if (isa >= ISALevel::AVX512)
        {
            const __m512 vgate = _mm512_set1_ps(gate);
            for (; j + 16 <= count; j += 16)
            {
                const __m512 gated = _mm512_mul_ps(vgate, _mm512_loadu_ps(shared_row + j));
                _mm512_storeu_ps(shared_row + j, gated);
                _mm512_storeu_ps(combined_row + j,
                                 _mm512_add_ps(_mm512_loadu_ps(residual_row + j), gated));
            }
        }
#endif
#if defined(__AVX2__)
        if (isa >= ISALevel::AVX2)
        {
            const __m256 vgate = _mm256_set1_ps(gate);
            for (; j + 8 <= count; j += 8)
            {
                const __m256 gated = _mm256_mul_ps(vgate, _mm256_loadu_ps(shared_row + j));
                _mm256_storeu_ps(shared_row + j, gated);
                _mm256_storeu_ps(combined_row + j,
                                 _mm256_add_ps(_mm256_loadu_ps(residual_row + j), gated));
            }
        }
#endif
        (void)isa;
        for (; j < count; ++j)
        {
            const float gated = gate * shared_row[j];
            shared_row[j] = gated;
            combined_row[j] = residual_row[j] + gated;
        }
    }

    class RetainedCPUFFNStage final : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        RetainedCPUFFNStage(const CPUHCRegionNodes &hc_region,
                            CPUFFNDeclaration decl,
                            ContractedStageMetadata meta)
            : IComputeStage(decl.router.device_id),
              hc_(decl.router.device_id, hc_region),
              decl_(std::move(decl)),
              meta_(std::move(meta))
        {
            require(device().is_cpu(), "RetainedCPUFFNStage requires CPU participant");
            const auto &s = decl_.shared;
            shared_gate_.bindFromRef(s.gate_w, s.prepared_store, *s.prepared_ref_gate, s.intermediate, s.d_model);
            shared_up_.bindFromRef(s.up_w, s.prepared_store, *s.prepared_ref_up, s.intermediate, s.d_model);
            shared_down_.bindFromRef(s.down_w, s.prepared_store, *s.prepared_ref_down, s.d_model, s.intermediate);

            router_logits_.resize(static_cast<size_t>(decl_.router.num_experts));
            router_sort_idx_.resize(static_cast<size_t>(decl_.router.num_experts));
            router_q8_local_.resize(static_cast<size_t>((s.d_model + 31) / 32));
            shared_in_q8_.resize(static_cast<size_t>((s.d_model + 31) / 32));
            shared_swiglu_q8_.resize(static_cast<size_t>((s.intermediate + 31) / 32));
            shared_gate_buf_.resize(static_cast<size_t>(s.intermediate));
            shared_up_buf_.resize(static_cast<size_t>(s.intermediate));
            shared_swiglu_fp32_.resize(static_cast<size_t>(s.intermediate));

            required_partials_floats_ = std::max({
                hc_.maxPartialFloats(),
                maxPartialFloatsForHandle(shared_gate_),
                maxPartialFloatsForHandle(shared_up_),
                maxPartialFloatsForHandle(shared_down_)});
            if (required_partials_floats_ > 0)
                meta_.workspace_reqs.merge(cpuProjectionPartialWorkspaceRequirements(required_partials_floats_));
            max_rounded_a_floats_ = static_cast<size_t>(std::max({hc_.maxFloatK(), s.d_model, s.intermediate}));
            meta_.workspace_reqs.buffers.push_back(
                {kCPUMegakernelRoundedA, max_rounded_a_floats_ * sizeof(float), 64});

            if (decl_.routed)
            {
                const auto &rt = *decl_.routed;
                const size_t top_k = static_cast<size_t>(rt.top_k);
                const size_t inter = static_cast<size_t>(rt.expert_intermediate);
                const size_t dm = static_cast<size_t>(rt.d_model);
                routed_scratch_layout_ = CPUMegakernelFFNRoutedLayout::compute(top_k, inter, dm);
                meta_.workspace_reqs.buffers.push_back(
                    {kCPUMegakernelFFNRoutedScratch, routed_scratch_layout_.total_bytes, 64});

                // Pre-cast all prepared routed expert engines once at graph construction
                // (eliminates 30 * 28 = 840 dynamic_casts per layer on the hot path!).
                const size_t nexp = static_cast<size_t>(rt.num_experts);
                routed_gate_native_.resize(nexp, nullptr);
                routed_up_native_.resize(nexp, nullptr);
                routed_down_native_.resize(nexp, nullptr);
                for (size_t e = 0; e < nexp; ++e)
                {
                    routed_gate_native_[e] = dynamic_cast<CPUNativeQuantisedGemmKernel *>(rt.prepared_gate_gemm[e]);
                    routed_up_native_[e] = dynamic_cast<CPUNativeQuantisedGemmKernel *>(rt.prepared_up_gemm[e]);
                    routed_down_native_[e] = dynamic_cast<CPUNativeQuantisedGemmKernel *>(rt.prepared_down_gemm[e]);
                }
                rebuildCachedMoERowEntries(std::max(1, omp_get_max_threads()), activeISALevel());
            }
        }

        ComputeStageType type() const override { return ComputeStageType::MOE_SHARED_EXPERT_FFN; }
        std::string name() const override
        {
            return decl_.routed ? "retained_cpu_full_ffn_megakernel" : "retained_cpu_precollective_ffn_megakernel";
        }
        bool supportsBackend(ComputeBackendType backend) const override { return backend == ComputeBackendType::CPU; }
        StageBufferContract bufferContract() const override { return meta_.contract; }
        StageBufferRequirements getBufferRequirements() const override { return meta_.buffer_reqs; }
        WorkspaceRequirements getWorkspaceRequirements(int, int = 0, int = 0) const override { return meta_.workspace_reqs; }
        void bindWorkspace(DeviceWorkspaceManager *ws) override
        {
            if (ws)
            {
                const auto requested = std::make_shared<const CPURouterQ8StorageLayout>(
                    CPURouterQ8StorageLayout::declare(
                        std::max(1, decl_.router.seq_len), decl_.router.d_model, device()));
                ws->requireCPURouterQ8Storage(requested);
                const auto storage = ws->cpuRouterQ8Storage();
                if (!storage)
                    throw std::logic_error("RetainedCPUFFNStage requires declared CPU router Q8 storage");
                if (auto *cpu_k = resolveCPUMoEKernel())
                    cpu_k->bindRouterQ8HiddenStorage(storage);
            }
            workspace_ = ws;
        }
        void unbindWorkspace() override { workspace_ = nullptr; }
        bool hasWorkspace() const override { return workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }
        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo info;
            if (decl_.shared_gate && decl_.shared_gate->combined_output)
                info.addOutput("combined_output", decl_.shared_gate->combined_output, 1, decl_.router.d_model);
            else
                info.addOutput("shared_output", decl_.shared.output, 1, decl_.shared.d_model);
            return info;
        }

        bool execute(IDeviceContext *ctx) override
        {
            if (!ctx || ctx->deviceId() != device())
                return false;

            // Resolve CPUMoEKernel owner before entering the OpenMP team (zero locks in team).
            CPUMoEKernel *cpu_moe_kernel = resolveCPUMoEKernel();
            if (!cpu_moe_kernel)
                return false;
            if (workspace_)
            {
                if (const auto storage = workspace_->cpuRouterQ8Storage())
                    cpu_moe_kernel->bindRouterQ8HiddenStorage(storage);
            }

            std::span<float> partials{};
            if (workspace_)
                partials = CPUInvocationWorkspace::available<float>(workspace_, kCPUProjectionPartials);
            if (partials.size() < required_partials_floats_)
            {
                if (fallback_partials_.size() < std::max(required_partials_floats_, kDefaultPartialScratchFloats))
                    fallback_partials_.resize(std::max(required_partials_floats_, kDefaultPartialScratchFloats));
                partials = std::span<float>(fallback_partials_.data(), fallback_partials_.size());
            }

            float *rounded_a = nullptr;
            if (workspace_)
            {
                auto ws_ra = CPUInvocationWorkspace::available<float>(workspace_, kCPUMegakernelRoundedA);
                if (ws_ra.size() >= max_rounded_a_floats_)
                    rounded_a = ws_ra.data();
            }
            if (!rounded_a)
            {
                if (fallback_rounded_a_.size() < max_rounded_a_floats_)
                    fallback_rounded_a_.resize(max_rounded_a_floats_);
                rounded_a = fallback_rounded_a_.data();
            }

            const auto &rt = decl_.router;
            const auto &sh = decl_.shared;
            const int d_model = rt.d_model;
            const int num_experts = rt.num_experts;
            const int top_k = rt.top_k;
            const int sh_inter = sh.intermediate;

            const float *mixed_ptr = rt.input->data();
            float *out_idx_ptr = rt.output_indices->mutable_data();
            float *out_wt_ptr = rt.output_weights->mutable_data();
            float *shared_out_ptr = sh.output->mutable_data();

            const TensorType gate_type = rt.gate_weights->native_type();
            const void *router_gate_data = nullptr;
            if (gate_type == TensorType::FP16)
                router_gate_data = dynamic_cast<const FP16Tensor *>(rt.gate_weights)->typed_data();
            else if (gate_type == TensorType::BF16)
                router_gate_data = dynamic_cast<const BF16Tensor *>(rt.gate_weights)->typed_data();
            else
                router_gate_data = rt.gate_weights->data();

            const bool shared_all_native =
                shared_gate_.isSingleQ8Native() && shared_up_.isSingleQ8Native() && shared_down_.isSingleQ8Native();

            std::array<int, kDeviceMoEMaxTopK> selected_experts{};
            std::array<float, kDeviceMoEMaxTopK> selected_weights{};
            const Q8_1Block *published_router_q8 = nullptr;
            float shared_gate_scalar = 0.0f;
            const ISALevel isa = activeISALevel();

            // Shared expert scratch pointers (prefer arena scratch if sized, else stage-owned).
            float *sh_gate_ptr = (sh.gate_scratch && sh.gate_scratch->numel() >= static_cast<size_t>(sh_inter))
                ? sh.gate_scratch->mutable_data()
                : shared_gate_buf_.data();
            float *sh_up_ptr = (sh.up_scratch && sh.up_scratch->numel() >= static_cast<size_t>(sh_inter))
                ? sh.up_scratch->mutable_data()
                : shared_up_buf_.data();

            const auto hc_ptrs = hc_.resolvePointers();
            const float *shared_gate_inp_ptr = (decl_.shared_gate && decl_.shared_gate->gate_inp)
                ? decl_.shared_gate->gate_inp->data()
                : nullptr;
            float *routed_out = decl_.routed ? decl_.routed->output->mutable_data() : nullptr;
            float *combined_out = (decl_.shared_gate && decl_.shared_gate->combined_output)
                ? decl_.shared_gate->combined_output->mutable_data()
                : nullptr;

            // Borrow ONE shared L3-hot workspace slab for routed expert scratch across all 48 FFN layers.
            float *routed_gate_ptr = nullptr;
            float *routed_up_ptr = nullptr;
            float *routed_down_ptr = nullptr;
            Q8_1Block *routed_swiglu_q8_ptr = nullptr;
            size_t routed_swiglu_q8_count = 0;
            float *routed_splitk_ptr = nullptr;
            size_t routed_splitk_cap = 0;
            if (decl_.routed)
            {
                std::byte *slab = nullptr;
                if (workspace_)
                {
                    auto ws_slab = CPUInvocationWorkspace::available<std::byte>(
                        workspace_, kCPUMegakernelFFNRoutedScratch);
                    if (ws_slab.size() >= routed_scratch_layout_.total_bytes)
                        slab = ws_slab.data();
                }
                if (!slab)
                {
                    if (fallback_routed_scratch_.size() < routed_scratch_layout_.total_bytes)
                        fallback_routed_scratch_.resize(routed_scratch_layout_.total_bytes);
                    slab = fallback_routed_scratch_.data();
                }
                routed_gate_ptr = reinterpret_cast<float *>(slab + routed_scratch_layout_.gate_offset);
                routed_up_ptr = reinterpret_cast<float *>(slab + routed_scratch_layout_.up_offset);
                routed_down_ptr = reinterpret_cast<float *>(slab + routed_scratch_layout_.down_offset);
                routed_swiglu_q8_ptr = reinterpret_cast<Q8_1Block *>(slab + routed_scratch_layout_.swiglu_q8_offset);
                routed_swiglu_q8_count =
                    static_cast<size_t>(top_k + 1) * ((static_cast<size_t>(decl_.routed->expert_intermediate) + 31u) / 32u);
                routed_splitk_ptr = reinterpret_cast<float *>(slab + routed_scratch_layout_.splitk_offset);
                routed_splitk_cap = CPUNativeQuantisedGemmKernel::kStandaloneMoESplitKScratchFloats;
            }

            const bool fuse_shared_into_routed =
                decl_.routed.has_value() &&
                shared_all_native &&
                sh_inter == decl_.routed->expert_intermediate &&
                (d_model % 64) == 0 &&
                (sh_inter % 64) == 0 &&
                static_cast<size_t>(top_k + 1) <= CPUNativeQuantisedGemmKernel::StandaloneMoEOneRowPlan::kMaxRoutes &&
                CPUNativeQuantisedGemmKernel::supportsStandaloneMoEOneRowInTeam(
                    d_model, sh_inter, static_cast<size_t>(top_k + 1));
            const CPUProjectionNumericalPolicy shared_in_policy = shared_all_native
                ? shared_gate_.native_engine->packedWeights().numerical_policy
                : CPUProjectionNumericalPolicy::BackendNative;
            const bool need_shared_in_q8_pre_router =
                fuse_shared_into_routed &&
                shared_in_policy != CPUProjectionNumericalPolicy::GPUAlignedExpert;
            const bool parallel_router_q8 = decl_.routed.has_value() && ((d_model % 64) == 0);

            const int expected_workers = std::max(1, omp_get_max_threads());
            if (decl_.routed && (cached_moe_workers_ != expected_workers || cached_moe_isa_ != isa))
                rebuildCachedMoERowEntries(expected_workers, isa);

            CPUNativeQuantisedGemmKernel::StandaloneMoEOneRowPlan routed_plan{};
            const bool detail_timing = PerfStatsCollector::isDomainEnabled("stage_cpu_detail");
            using Clock = std::chrono::steady_clock;
            std::array<Clock::time_point, 7> ts{};
            std::array<Clock::time_point, 5> hc_ts{};
            auto mark = [&](size_t idx) noexcept {
                if (detail_timing && omp_get_thread_num() == 0)
                    ts[idx] = Clock::now();
            };
            std::atomic<bool> ok{true};
            auto run_team = [&]() {
                mark(0);
                // 1. HC prefix (fused hc_attn_write + hc_ffn_branch_norm + down/inject/up/read) -> mixed[H].
                if (!hc_.executeInTeam(hc_ptrs, partials, rounded_a, workspace_, isa, detail_timing ? &hc_ts : nullptr))
                    ok.store(false, std::memory_order_relaxed);
                mark(1);

                // 2. MoE Router GEMV + Softmax Top-K + canonical router Q8_1 publication.
                if (ok.load(std::memory_order_relaxed))
                {
                    // Quantize shared_in_q8_ and router_q8_local_ in parallel with `nowait`:
                    // router GEMV reads mixed_ptr directly, and the implicit barrier at the end
                    // of router GEMV publishes both Q8 banks before finalize_routing / MoE.
                    if (need_shared_in_q8_pre_router || parallel_router_q8)
                    {
                        const int pairs = d_model / 64;
#pragma omp for schedule(static) nowait
                        for (int p = 0; p < pairs; ++p)
                        {
                            const int start = p * 64;
                            if (need_shared_in_q8_pre_router)
                            {
                                cpu::native_quantised::quantizeTwoActivationBlocks(
                                    mixed_ptr + start,
                                    shared_in_q8_[static_cast<size_t>(p * 2)],
                                    shared_in_q8_[static_cast<size_t>(p * 2 + 1)],
                                    shared_in_policy);
                            }
                            if (parallel_router_q8)
                            {
#if defined(__AVX512F__)
                                if (isa >= ISALevel::AVX512)
                                {
                                    cpu::gpu_aligned_expert_q8::quantizeTwoBlocks(
                                        mixed_ptr + start,
                                        router_q8_local_[static_cast<size_t>(p * 2)],
                                        router_q8_local_[static_cast<size_t>(p * 2 + 1)]);
                                }
                                else
#endif
                                {
                                    cpu::gpu_aligned_expert_q8::quantizeBlock(
                                        mixed_ptr + start,
                                        router_q8_local_[static_cast<size_t>(p * 2)], 32);
                                    cpu::gpu_aligned_expert_q8::quantizeBlock(
                                        mixed_ptr + start + 32,
                                        router_q8_local_[static_cast<size_t>(p * 2 + 1)], 32);
                                }
                            }
                        }
                    }

                    // Stage activation row into thread-private L1d stack memory in staggered
                    // cache-line order (zero barrier, zero multi-core HitM convoy!).
                    alignas(64) float local_a[kMaxPrivateActivationFloats];
                    if (gate_type == TensorType::FP16)
                    {
                        const float *act_ptr = rounded_a;
                        if (d_model <= kMaxPrivateActivationFloats)
                        {
                            staggeredRoundRowToFP16RNEPrivate(mixed_ptr, local_a, d_model, isa);
                            act_ptr = local_a;
                        }
                        else
                        {
                            roundRowToFP16RNEInTeam(mixed_ptr, rounded_a, d_model, isa);
                        }
#pragma omp for schedule(static)
                        for (int e = 0; e < num_experts; ++e)
                        {
                            router_logits_[static_cast<size_t>(e)] =
                                gemm::dot_gpu_aligned_expert_one_row<TensorType::FP16>(
                                    act_ptr, router_gate_data,
                                    static_cast<std::size_t>(e) * static_cast<std::size_t>(d_model),
                                    d_model, isa);
                        }
                    }
                    else if (gate_type == TensorType::BF16)
                    {
                        const float *act_ptr = mixed_ptr;
                        if (d_model <= kMaxPrivateActivationFloats)
                        {
                            staggeredCopyFloatsPrivate(mixed_ptr, local_a, d_model);
                            act_ptr = local_a;
                        }
#pragma omp for schedule(static)
                        for (int e = 0; e < num_experts; ++e)
                        {
                            router_logits_[static_cast<size_t>(e)] =
                                gemm::dot_gpu_aligned_expert_one_row<TensorType::BF16>(
                                    act_ptr, router_gate_data,
                                    static_cast<std::size_t>(e) * static_cast<std::size_t>(d_model),
                                    d_model, isa);
                        }
                    }
                    else
                    {
                        const float *act_ptr = mixed_ptr;
                        if (d_model <= kMaxPrivateActivationFloats)
                        {
                            staggeredCopyFloatsPrivate(mixed_ptr, local_a, d_model);
                            act_ptr = local_a;
                        }
                        const float *w_fp32 = static_cast<const float *>(router_gate_data);
#pragma omp for schedule(static)
                        for (int e = 0; e < num_experts; ++e)
                        {
                            router_logits_[static_cast<size_t>(e)] = primitives::vec_dot(
                                w_fp32 + static_cast<size_t>(e) * static_cast<std::size_t>(d_model),
                                act_ptr, d_model);
                        }
                    }

                    mark(2);
                    // Single-thread softmax + top-k + router Q8_1 publication + routed(+shared) MoE plan prep
                    // (implicit barrier publishes everything to all workers with ZERO extra barriers!).
                    auto finalize_routing = [&]() {
                        if (!ok.load(std::memory_order_relaxed))
                            return;
                        primitives::softmax_row_fp32(router_logits_.data(), num_experts);
                        std::iota(router_sort_idx_.begin(), router_sort_idx_.end(), 0);
                        const float *probs = router_logits_.data();
                        std::partial_sort(
                            router_sort_idx_.begin(),
                            router_sort_idx_.begin() + top_k,
                            router_sort_idx_.end(),
                            [probs](int a, int b) { return probs[a] > probs[b]; });
                        float topk_sum = 0.0f;
                        for (int k = 0; k < top_k; ++k)
                            topk_sum += probs[router_sort_idx_[static_cast<size_t>(k)]];
                        for (int k = 0; k < top_k; ++k)
                        {
                            const int eid = router_sort_idx_[static_cast<size_t>(k)];
                            const float w = rt.norm_topk_prob ? (probs[eid] / topk_sum) : probs[eid];
                            selected_experts[static_cast<size_t>(k)] = eid;
                            selected_weights[static_cast<size_t>(k)] = w;
                            out_idx_ptr[k] = static_cast<float>(eid);
                            out_wt_ptr[k] = w;
                        }
                        if (parallel_router_q8)
                        {
                            published_router_q8 = router_q8_local_.data();
                        }
                        else
                        {
                            if (!cpu_moe_kernel->publishTransportedRouterQ8Hidden(mixed_ptr, 1, d_model))
                            {
                                ok.store(false, std::memory_order_relaxed);
                                return;
                            }
                            published_router_q8 = cpu_moe_kernel->publishedRouterQ8Hidden(mixed_ptr, 1, d_model);
                        }
                        if (shared_gate_inp_ptr)
                        {
                            const float dot = primitives::vec_dot(shared_gate_inp_ptr, mixed_ptr, d_model);
                            shared_gate_scalar = 1.0f / (1.0f + std::exp(-dot));
                        }
                        if (decl_.routed && published_router_q8)
                        {
                            const auto &rparams = *decl_.routed;
                            const int inter = rparams.expert_intermediate;
                            const size_t act_blocks_per_route = static_cast<size_t>(inter) / Q8_1Block::BLOCK_SIZE;
                            const int cur_workers = std::max(1, omp_get_num_threads());
                            const bool use_cached = cached_moe_valid_ &&
                                                    cached_moe_workers_ == cur_workers &&
                                                    cached_moe_isa_ == isa;
                            if (use_cached)
                            {
                                bool all_ok = true;
                                for (int r = 0; r < top_k; ++r)
                                {
                                    const size_t eid = static_cast<size_t>(selected_experts[static_cast<size_t>(r)]);
                                    if (!cached_gate_entries_[eid].weights || !cached_up_entries_[eid].weights ||
                                        !cached_down_entries_[eid].weights)
                                    {
                                        all_ok = false;
                                        break;
                                    }
                                    auto &ge = routed_plan.gate_plans[static_cast<size_t>(2 * r)];
                                    auto &ue = routed_plan.gate_plans[static_cast<size_t>(2 * r + 1)];
                                    auto &de = routed_plan.down_plans[static_cast<size_t>(r)];
                                    ge = cached_gate_entries_[eid];
                                    ge.input = published_router_q8;
                                    ge.output = routed_gate_ptr + static_cast<size_t>(r) * static_cast<size_t>(inter);
                                    ue = cached_up_entries_[eid];
                                    ue.input = published_router_q8;
                                    ue.output = routed_up_ptr + static_cast<size_t>(r) * static_cast<size_t>(inter);
                                    de = cached_down_entries_[eid];
                                    de.input = routed_swiglu_q8_ptr + static_cast<size_t>(r) * act_blocks_per_route;
                                    de.output = routed_down_ptr + static_cast<size_t>(r) * static_cast<size_t>(d_model);
                                }
                                size_t planned_routes = static_cast<size_t>(top_k);
                                if (all_ok && fuse_shared_into_routed)
                                {
                                    const Q8_1Block *sh_q8 =
                                        (shared_in_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                                            ? published_router_q8
                                            : shared_in_q8_.data();
                                    const size_t sr = static_cast<size_t>(top_k);
                                    auto &ge = routed_plan.gate_plans[2 * sr];
                                    auto &ue = routed_plan.gate_plans[2 * sr + 1];
                                    auto &de = routed_plan.down_plans[sr];
                                    ge = cached_sh_gate_entry_;
                                    ge.input = sh_q8;
                                    ge.output = sh_gate_ptr;
                                    ue = cached_sh_up_entry_;
                                    ue.input = sh_q8;
                                    ue.output = sh_up_ptr;
                                    de = cached_sh_down_entry_;
                                    de.input = routed_swiglu_q8_ptr + sr * act_blocks_per_route;
                                    de.output = shared_out_ptr;
                                    planned_routes = sr + 1u;
                                }
                                if (!all_ok || !CPUNativeQuantisedGemmKernel::finalizeStandaloneMoEOneRowPlan(
                                                   routed_plan, planned_routes, d_model, inter,
                                                   cur_workers, isa, routed_splitk_cap))
                                {
                                    ok.store(false, std::memory_order_relaxed);
                                }
                            }
                            else
                            {
                                using Projection = CPUNativeQuantisedGemmKernel::BatchedPrequantizedProjectionDesc;
                                std::array<Projection, 2 * CPUNativeQuantisedGemmKernel::StandaloneMoEOneRowPlan::kMaxRoutes> gate_up{};
                                std::array<Projection, CPUNativeQuantisedGemmKernel::StandaloneMoEOneRowPlan::kMaxRoutes> down{};
                                bool all_native = true;
                                for (int r = 0; r < top_k; ++r)
                                {
                                    const size_t eid = static_cast<size_t>(selected_experts[static_cast<size_t>(r)]);
                                    auto *g_eng = routed_gate_native_[eid];
                                    auto *u_eng = routed_up_native_[eid];
                                    auto *d_eng = routed_down_native_[eid];
                                    if (!g_eng || !u_eng || !d_eng)
                                    {
                                        all_native = false;
                                        break;
                                    }
                                    gate_up[static_cast<size_t>(2 * r)] = Projection{
                                        .kernel = g_eng,
                                        .input_q8 = published_router_q8,
                                        .output = routed_gate_ptr + static_cast<size_t>(r) * static_cast<size_t>(inter),
                                        .rows = 1, .n = inter, .ldc = inter};
                                    gate_up[static_cast<size_t>(2 * r + 1)] = Projection{
                                        .kernel = u_eng,
                                        .input_q8 = published_router_q8,
                                        .output = routed_up_ptr + static_cast<size_t>(r) * static_cast<size_t>(inter),
                                        .rows = 1, .n = inter, .ldc = inter};
                                    down[static_cast<size_t>(r)] = Projection{
                                        .kernel = d_eng,
                                        .input_q8 = routed_swiglu_q8_ptr + static_cast<size_t>(r) * act_blocks_per_route,
                                        .output = routed_down_ptr + static_cast<size_t>(r) * static_cast<size_t>(d_model),
                                        .rows = 1, .n = d_model, .ldc = d_model};
                                }
                                size_t planned_routes = static_cast<size_t>(top_k);
                                if (all_native && fuse_shared_into_routed)
                                {
                                    const Q8_1Block *sh_q8 =
                                        (shared_in_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                                            ? published_router_q8
                                            : shared_in_q8_.data();
                                    const size_t sr = static_cast<size_t>(top_k);
                                    gate_up[2 * sr] = Projection{
                                        .kernel = shared_gate_.native_engine,
                                        .input_q8 = sh_q8,
                                        .output = sh_gate_ptr,
                                        .rows = 1, .n = inter, .ldc = inter};
                                    gate_up[2 * sr + 1] = Projection{
                                        .kernel = shared_up_.native_engine,
                                        .input_q8 = sh_q8,
                                        .output = sh_up_ptr,
                                        .rows = 1, .n = inter, .ldc = inter};
                                    down[sr] = Projection{
                                        .kernel = shared_down_.native_engine,
                                        .input_q8 = routed_swiglu_q8_ptr + sr * act_blocks_per_route,
                                        .output = shared_out_ptr,
                                        .rows = 1, .n = d_model, .ldc = d_model};
                                    planned_routes = sr + 1u;
                                }
                                if (!all_native ||
                                    !CPUNativeQuantisedGemmKernel::prepareStandaloneMoEOneRowPlan(
                                        routed_plan,
                                        std::span(gate_up).first(2u * planned_routes),
                                        std::span(down).first(planned_routes),
                                        d_model, inter, cur_workers, isa,
                                        routed_splitk_cap))
                                {
                                    ok.store(false, std::memory_order_relaxed);
                                }
                            }
                        }
                    };
                    OMP_SINGLE(finalize_routing);
                }
                mark(3);

                // 3. Shared Expert FFN (when not fused into the unified routed+shared plan below).
                if (ok.load(std::memory_order_relaxed) && !fuse_shared_into_routed)
                {
                    if (shared_all_native)
                    {
                        const auto sh_policy = shared_gate_.native_engine->packedWeights().numerical_policy;
                        const Q8_1Block *sh_q8 = nullptr;
                        if (sh_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert && published_router_q8)
                        {
                            sh_q8 = published_router_q8;
                        }
                        else
                        {
                            quantizeRowToQ8InTeam(mixed_ptr, shared_in_q8_.data(), d_model, sh_policy);
                            sh_q8 = shared_in_q8_.data();
                        }
                        std::array<FusedGemvDesc, 2> gu_descs{{
                            {.packed = &shared_gate_.native_engine->packedWeights(),
                             .output = sh_gate_ptr,
                             .bias = nullptr,
                             .N = sh_inter,
                             .policy_n = shared_gate_.policy_n},
                            {.packed = &shared_up_.native_engine->packedWeights(),
                             .output = sh_up_ptr,
                             .bias = nullptr,
                             .N = sh_inter,
                             .policy_n = shared_up_.policy_n},
                        }};
                        runNativeQuantisedBundleInTeam(sh_q8, gu_descs.data(), 2, partials);

                        // Fused SwiGLU -> Q8_1 in parallel across 64-element chunks (exact arithmetic match).
                        const auto down_policy = shared_down_.native_engine->packedWeights().numerical_policy;
                        const int sh_chunks = (sh_inter + 63) / 64;
#pragma omp for schedule(static)
                        for (int c = 0; c < sh_chunks; ++c)
                        {
                            const int off = c * 64;
                            const int len = std::min(64, sh_inter - off);
                            alignas(64) float sw[64];
                            if (down_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                                primitives::compute_swiglu_gpu_aligned_expert_serial(
                                    sh_gate_ptr + off, sh_up_ptr + off, sw, len);
                            else
                                primitives::compute_swiglu_serial(
                                    sh_gate_ptr + off, sh_up_ptr + off, sw, len);
                            if (len == 64)
                            {
                                cpu::native_quantised::quantizeTwoActivationBlocks(
                                    sw, shared_swiglu_q8_[static_cast<size_t>(c * 2)],
                                    shared_swiglu_q8_[static_cast<size_t>(c * 2 + 1)], down_policy);
                            }
                            else
                            {
                                const int blks = (len + 31) / 32;
                                for (int b = 0; b < blks; ++b)
                                {
                                    cpu::native_quantised::quantizeActivationBlock(
                                        sw + b * 32,
                                        shared_swiglu_q8_[static_cast<size_t>(c * 2 + b)],
                                        std::min(32, len - b * 32), down_policy);
                                }
                            }
                        }

                        FusedGemvDesc d_desc{
                            .packed = &shared_down_.native_engine->packedWeights(),
                            .output = shared_out_ptr,
                            .bias = nullptr,
                            .N = d_model,
                            .policy_n = shared_down_.policy_n};
                        runNativeQuantisedBundleInTeam(shared_swiglu_q8_.data(), &d_desc, 1, partials);
                    }
                    else
                    {
                        // Floating shared expert fallback inside the same team (zero oneDNN contention).
                        if (!runFloatGemvInTeam(
                                sh.input, mixed_ptr, shared_gate_, sh.gate_scratch, sh_gate_ptr,
                                rounded_a, &float_status_, workspace_, isa) ||
                            !runFloatGemvInTeam(
                                sh.input, mixed_ptr, shared_up_, sh.up_scratch, sh_up_ptr,
                                rounded_a, &float_status_, workspace_, isa))
                        {
                            ok.store(false, std::memory_order_relaxed);
                        }
                        else
                        {
                            const int sh_chunks = (sh_inter + 63) / 64;
#pragma omp for schedule(static)
                            for (int c = 0; c < sh_chunks; ++c)
                            {
                                const int off = c * 64;
                                const int len = std::min(64, sh_inter - off);
                                primitives::compute_swiglu_serial(
                                    sh_gate_ptr + off, sh_up_ptr + off, shared_swiglu_fp32_.data() + off, len);
                            }
                            if (!runFloatGemvInTeam(
                                    sh.gate_scratch, shared_swiglu_fp32_.data(), shared_down_,
                                    sh.output, shared_out_ptr, rounded_a, &float_status_, workspace_, isa))
                            {
                                ok.store(false, std::memory_order_relaxed);
                            }
                        }
                    }
                }

                mark(4);
                // 4. Optional singleton-CPU Routed Expert Standalone MoE + SharedExpertGate (all in the same team!).
                if (ok.load(std::memory_order_relaxed) && decl_.routed)
                {
                    if (!CPUNativeQuantisedGemmKernel::executePreparedStandaloneMoEOneRowInTeam(
                            routed_plan,
                            std::span<Q8_1Block>(routed_swiglu_q8_ptr, routed_swiglu_q8_count),
                            routed_splitk_ptr, routed_splitk_cap))
                    {
                        ok.store(false, std::memory_order_relaxed);
                    }
                    else
                    {
                        mark(5);
                        // Parallel weighted route accumulation + fused SharedExpertGate in 64-float strips!
                        constexpr int kStrip = 64;
                        const int n_strips = (d_model + kStrip - 1) / kStrip;
#pragma omp for schedule(static)
                        for (int s = 0; s < n_strips; ++s)
                        {
                            const int off = s * kStrip;
                            const int len = std::min(kStrip, d_model - off);
                            std::fill_n(routed_out + off, len, 0.0f);
                            for (int r = 0; r < top_k; ++r)
                            {
                                const float *src = routed_down_ptr +
                                                   static_cast<size_t>(r) * static_cast<size_t>(d_model) + off;
                                primitives::vec_axpy(routed_out + off, src, selected_weights[static_cast<size_t>(r)], len);
                            }
                            if (combined_out)
                            {
                                applySharedGateAddSlice(
                                    shared_out_ptr + off, routed_out + off, combined_out + off,
                                    shared_gate_scalar, len, isa);
                            }
                        }
                    }
                }
                mark(6);
            };
            OMP_LAYER_PARALLEL(run_team(););

            if (detail_timing && ok.load(std::memory_order_relaxed))
            {
                const auto ns = [&](size_t a, size_t b) {
                    return static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(ts[b] - ts[a]).count());
                };
                PerfStatsCollector::recordTimingNs("mk_sub", "ffn_1_hc", ns(0, 1), "decode", "cpu");
                const auto hns = [&](size_t a, size_t b) {
                    return static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(hc_ts[b] - hc_ts[a]).count());
                };
                PerfStatsCollector::recordTimingNs("mk_sub", "ffn_1a_hc_norm", hns(0, 1), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "ffn_1b_hc_down", hns(1, 2), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "ffn_1c_hc_inject", hns(2, 3), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "ffn_1d_hc_up_read", hns(3, 4), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "ffn_2_router_gemv", ns(1, 2), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "ffn_3_finalize_routing", ns(2, 3), "decode", "cpu");
                PerfStatsCollector::recordTimingNs("mk_sub", "ffn_4_shared_sep", ns(3, 4), "decode", "cpu");
                if (decl_.routed)
                {
                    PerfStatsCollector::recordTimingNs("mk_sub", "ffn_5_moe_plan", ns(4, 5), "decode", "cpu");
                    PerfStatsCollector::recordTimingNs("mk_sub", "ffn_6_accum_gate", ns(5, 6), "decode", "cpu");
                }
            }

            if (ok.load(std::memory_order_relaxed))
            {
                if (rt.decode_histogram && rt.layer_idx >= 0)
                {
                    rt.decode_histogram->record(
                        rt.layer_idx, selected_experts.data(), selected_weights.data(), top_k);
                }
                if (PerfStatsCollector::isDomainEnabled("kernel"))
                {
                    PerfStatsCollector::addCounter(
                        "kernel", "cpu_retained_ffn_megakernel_calls", 1.0, "decode", "cpu",
                        {{"mode", decl_.routed ? "full_singleton_ffn" : "precollective_ffn"},
                         {"isa", isaLevelName(isa)}});
                }
            }
            return ok.load(std::memory_order_relaxed);
        }

    private:
        CPUMoEKernel *resolveCPUMoEKernel()
        {
            if (decl_.router.routed_pipeline_kernel_owner)
            {
                auto &owner = *decl_.router.routed_pipeline_kernel_owner;
                if (!owner.kernel)
                    owner.kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device());
                return dynamic_cast<CPUMoEKernel *>(owner.kernel.get());
            }
            if (!owned_moe_kernel_)
                owned_moe_kernel_ = llaminar::v2::kernels::KernelFactory::createMoEKernel(device());
            return dynamic_cast<CPUMoEKernel *>(owned_moe_kernel_.get());
        }

        void rebuildCachedMoERowEntries(int workers, ISALevel isa)
        {
            cached_moe_valid_ = false;
            cached_moe_workers_ = workers;
            cached_moe_isa_ = isa;
            if (!decl_.routed)
                return;
            const auto &rt = *decl_.routed;
            const size_t nexp = routed_gate_native_.size();
            cached_gate_entries_.assign(nexp, {});
            cached_up_entries_.assign(nexp, {});
            cached_down_entries_.assign(nexp, {});
            bool all_ok = true;
            for (size_t e = 0; e < nexp; ++e)
            {
                if (!routed_gate_native_[e] || !routed_up_native_[e] || !routed_down_native_[e])
                {
                    all_ok = false;
                    break;
                }
                if (!CPUNativeQuantisedGemmKernel::prepareStandaloneMoEOneRowEntry(
                        cached_gate_entries_[e], routed_gate_native_[e], rt.expert_intermediate, rt.d_model, workers, isa) ||
                    !CPUNativeQuantisedGemmKernel::prepareStandaloneMoEOneRowEntry(
                        cached_up_entries_[e], routed_up_native_[e], rt.expert_intermediate, rt.d_model, workers, isa) ||
                    !CPUNativeQuantisedGemmKernel::prepareStandaloneMoEOneRowEntry(
                        cached_down_entries_[e], routed_down_native_[e], rt.d_model, rt.expert_intermediate, workers, isa))
                {
                    all_ok = false;
                    break;
                }
            }
            if (all_ok && shared_gate_.isSingleQ8Native() && shared_up_.isSingleQ8Native() &&
                shared_down_.isSingleQ8Native() && decl_.shared.intermediate == rt.expert_intermediate)
            {
                if (!CPUNativeQuantisedGemmKernel::prepareStandaloneMoEOneRowEntry(
                        cached_sh_gate_entry_, shared_gate_.native_engine, rt.expert_intermediate, rt.d_model, workers, isa) ||
                    !CPUNativeQuantisedGemmKernel::prepareStandaloneMoEOneRowEntry(
                        cached_sh_up_entry_, shared_up_.native_engine, rt.expert_intermediate, rt.d_model, workers, isa) ||
                    !CPUNativeQuantisedGemmKernel::prepareStandaloneMoEOneRowEntry(
                        cached_sh_down_entry_, shared_down_.native_engine, rt.d_model, rt.expert_intermediate, workers, isa))
                {
                    all_ok = false;
                }
            }
            cached_moe_valid_ = all_ok;
        }

        CPUHCLocalPrefix hc_;
        CPUFFNDeclaration decl_;
        ContractedStageMetadata meta_;
        CPUProjectionHandle shared_gate_;
        CPUProjectionHandle shared_up_;
        CPUProjectionHandle shared_down_;
        std::unique_ptr<IMoEKernel> owned_moe_kernel_;
        AlignedVector<float> router_logits_;
        std::vector<int> router_sort_idx_;
        AlignedVector<Q8_1Block> router_q8_local_;
        AlignedVector<Q8_1Block> shared_in_q8_;
        AlignedVector<Q8_1Block> shared_swiglu_q8_;
        AlignedVector<float> shared_gate_buf_;
        AlignedVector<float> shared_up_buf_;
        AlignedVector<float> shared_swiglu_fp32_;
        CPUMegakernelFFNRoutedLayout routed_scratch_layout_{};
        std::vector<CPUNativeQuantisedGemmKernel *> routed_gate_native_;
        std::vector<CPUNativeQuantisedGemmKernel *> routed_up_native_;
        std::vector<CPUNativeQuantisedGemmKernel *> routed_down_native_;
        using CachedRowEntry = CPUNativeQuantisedGemmKernel::StandaloneMoEOneRowPlan::RowEntry;
        std::vector<CachedRowEntry> cached_gate_entries_;
        std::vector<CachedRowEntry> cached_up_entries_;
        std::vector<CachedRowEntry> cached_down_entries_;
        CachedRowEntry cached_sh_gate_entry_{};
        CachedRowEntry cached_sh_up_entry_{};
        CachedRowEntry cached_sh_down_entry_{};
        int cached_moe_workers_ = 0;
        ISALevel cached_moe_isa_ = ISALevel::Scalar;
        bool cached_moe_valid_ = false;
        size_t required_partials_floats_ = 0;
        size_t max_rounded_a_floats_ = 0;
        AlignedVector<std::byte> fallback_routed_scratch_;
        AlignedVector<float> fallback_partials_;
        AlignedVector<float> fallback_rounded_a_;
        float float_status_ = 1.0f;
        DeviceWorkspaceManager *workspace_ = nullptr;
    };

} // namespace

// =============================================================================
// Public Graph Lowerers
// =============================================================================

void lowerRetainedCPUGDN(ComputeGraph &graph,
                         const std::string &gdnProjectionName,
                         const std::string &previousInputBoundary)
{
    using Primitive = Qwen4ExpPrimitiveStage;
    const auto *proj_node = graph.getNode(gdnProjectionName);
    const auto *proj_stage = proj_node ? dynamic_cast<const GDNProjectionStage *>(proj_node->stage.get()) : nullptr;
    if (!proj_stage)
        return;
    const auto &p = proj_stage->getParams();
    if (!p.device_id.is_cpu() || p.m != 1 || p.k <= 0 ||
        p.force_decode_equivalent_verifier_prefill || p.verifier_row_range.has_value() ||
        !p.prepared_store || !p.prepared_ref_qkv || !p.prepared_ref_z ||
        !p.prepared_ref_a || !p.prepared_ref_b || !p.input || !p.input_buffer_id)
        return;

    // Verify closed downstream chain: gdn_proj -> short_conv -> gdn_recurrence -> gated_norm -> gdn_out_proj.
    const auto proj_consumers = graph.directConsumersOf(gdnProjectionName);
    if (proj_consumers.size() != 1u)
        return;
    const auto *conv_node = graph.getNode(proj_consumers[0]);
    const auto *conv_stage = conv_node ? dynamic_cast<const ShortConv1dStage *>(conv_node->stage.get()) : nullptr;
    if (!conv_stage)
        return;
    const auto conv_consumers = graph.directConsumersOf(proj_consumers[0]);
    if (conv_consumers.size() != 1u)
        return;
    const auto *rec_node = graph.getNode(conv_consumers[0]);
    const auto *rec_stage = rec_node ? dynamic_cast<const GDNRecurrenceStage *>(rec_node->stage.get()) : nullptr;
    if (!rec_stage)
        return;
    const auto rec_consumers = graph.directConsumersOf(conv_consumers[0]);
    if (rec_consumers.size() != 1u)
        return;
    const auto *gn_node = graph.getNode(rec_consumers[0]);
    const auto *gn_stage = gn_node ? dynamic_cast<const GatedRMSNormStage *>(gn_node->stage.get()) : nullptr;
    if (!gn_stage)
        return;
    const auto gn_consumers = graph.directConsumersOf(rec_consumers[0]);
    if (gn_consumers.size() != 1u)
        return;
    const std::string out_proj_name = gn_consumers[0];
    const auto *out_node = graph.getNode(out_proj_name);
    const auto *out_stage = out_node ? dynamic_cast<const GEMMStage *>(out_node->stage.get()) : nullptr;
    if (!out_stage)
        return;

    const auto &cp = conv_stage->getParams();
    const auto &rp = rec_stage->getParams();
    const auto &gn = gn_stage->getParams();
    const auto &op = out_stage->getParams();
    const bool out_deps_closed = std::all_of(
        out_node->dependencies.begin(), out_node->dependencies.end(),
        [&](const std::string &dep) {
            return dep == gdnProjectionName || dep == proj_consumers[0] ||
                   dep == conv_consumers[0] || dep == rec_consumers[0];
        });
    if (!out_deps_closed ||
        cp.device_id != p.device_id || rp.device_id != p.device_id ||
        gn.device_id != p.device_id || op.device_id != p.device_id ||
        cp.seq_len != 1 || cp.request_count != 1 || cp.channels != p.n_qkv ||
        cp.input != p.output_qkv || !cp.output || !cp.weight || !cp.conv_state ||
        cp.verifier_state_capture_rows != 0 || cp.speculative_state_slot_rows != 0 ||
        dynamic_cast<const CPUShortConvolution *>(cp.kernel) == nullptr ||
        rp.seq_len != 1 || rp.request_count != 1 || rp.n_heads != p.n_a ||
        rp.n_k_heads <= 0 || rp.Q != cp.output || rp.K != cp.output || rp.V != cp.output ||
        rp.alpha != p.output_a || rp.beta != p.output_b || !rp.A_log || !rp.dt_bias ||
        !rp.output || !rp.recurrence_state ||
        rp.verifier_state_capture_rows != 0 || rp.speculative_state_slot_rows != 0 ||
        dynamic_cast<const CPUGatedDeltaNet *>(rp.kernel) == nullptr ||
        gn.seq_len != 1 || gn.feature_dim != p.n_z || gn.input != rp.output ||
        gn.output != rp.output || gn.gate != p.output_z || !gn.gamma ||
        op.m != 1 || op.k != p.n_z || op.A != gn.output || !op.B || !op.C ||
        op.bias || op.bias_tensor || op.alpha != 1.0f || op.beta != 0.0f ||
        op.needs_allreduce || !op.prepared_store || !op.prepared_ref)
        return;

    // Find direct HC ResidualRead feeding gdn_proj.
    std::string read_name;
    for (const auto &dep : proj_node->dependencies)
    {
        const auto *node = graph.getNode(dep);
        const auto *prim = node ? dynamic_cast<const Primitive *>(node->stage.get()) : nullptr;
        const auto *read = prim ? std::get_if<Primitive::ResidualRead>(&prim->getParams().operation) : nullptr;
        if (read && read->output.tensor == p.input && read->output.buffer == *p.input_buffer_id &&
            read->output.byteOffset == 0)
        {
            if (!read_name.empty())
                return;
            read_name = dep;
        }
    }
    if (read_name.empty())
        return;

    CPUHCRegionNodes hc_region{};
    if (!discoverCPUHCRegion(graph, read_name, previousInputBoundary, hc_region))
        return;

    std::vector<std::string> member_names;
    if (!hc_region.previous_write.empty())
        member_names.push_back(hc_region.previous_write);
    member_names.push_back(hc_region.branch_norm);
    member_names.push_back(hc_region.down);
    if (!hc_region.read_activation.empty())
        member_names.push_back(hc_region.read_activation);
    member_names.push_back(hc_region.up);
    member_names.push_back(hc_region.read);
    if (!hc_region.injection.empty())
        member_names.push_back(hc_region.injection);
    member_names.push_back(gdnProjectionName);
    member_names.push_back(proj_consumers[0]);
    member_names.push_back(conv_consumers[0]);
    member_names.push_back(rec_consumers[0]);
    member_names.push_back(out_proj_name);

    ContractedStageMetadata meta;
    meta.absorb(graph, member_names);

    auto replacement = std::make_unique<RetainedCPUGDNStage>(
        hc_region, CPUGDNDeclaration{p, cp, rp, gn, op}, std::move(meta));

    GraphComposer composer(graph);
    std::vector<GraphNodeRef> members;
    members.reserve(member_names.size());
    for (const auto &m : member_names)
        members.push_back(composer.ref(m));

    (void)lowerLocalMegakernelRegion(graph, LocalMegakernelRegionSpec{
        .members = std::move(members),
        .terminal = composer.ref(out_proj_name),
        .replacement = std::move(replacement),
    });
}

void lowerRetainedCPUQSAAttention(ComputeGraph &graph,
                                  const std::string &layerPrefix,
                                  const std::string &previousInputBoundary)
{
    using Primitive = Qwen4ExpPrimitiveStage;
    const std::string qkv_name = layerPrefix + "qkv_proj";
    const std::string split_name = layerPrefix + "q_gate_split";
    const std::string q_norm_name = layerPrefix + "q_norm";
    const std::string k_norm_name = layerPrefix + "k_norm";
    const std::string rope_name = layerPrefix + "rope";
    const std::string replay_name = layerPrefix + "qsa_prepare_replay";
    const std::string kv_append_name = layerPrefix + "kv_append";
    const std::string key_proj_name = layerPrefix + "qsa_key_proj";
    const std::string query_proj_name = layerPrefix + "qsa_query_proj";
    const std::string indexer_name = layerPrefix + "qsa_indexer";
    const std::string attn_name = layerPrefix + "attention";
    const std::string commit_name = layerPrefix + "qsa_commit_partial";
    const std::string gate_name = layerPrefix + "attn_output_gate";
    const std::string wo_name = layerPrefix + "wo_proj";

    const auto *qkv_node = graph.getNode(qkv_name);
    const auto *split_node = graph.getNode(split_name);
    const auto *rope_node = graph.getNode(rope_name);
    const auto *replay_node = graph.getNode(replay_name);
    const auto *kva_node = graph.getNode(kv_append_name);
    const auto *key_node = graph.getNode(key_proj_name);
    const auto *query_node = graph.getNode(query_proj_name);
    const auto *q_norm_node = graph.getNode(q_norm_name);
    const auto *k_norm_node = graph.getNode(k_norm_name);
    const auto *idx_node = graph.getNode(indexer_name);
    const auto *attn_node = graph.getNode(attn_name);
    const auto *commit_node = graph.getNode(commit_name);
    const auto *gate_node = graph.getNode(gate_name);
    const auto *wo_node = graph.getNode(wo_name);

    const auto *qkv_stage = qkv_node ? dynamic_cast<const FusedQKVGEMMStage *>(qkv_node->stage.get()) : nullptr;
    const auto *split_stage = split_node ? dynamic_cast<const QGateSplitStage *>(split_node->stage.get()) : nullptr;
    const auto *rope_stage = rope_node ? dynamic_cast<const RoPEStage *>(rope_node->stage.get()) : nullptr;
    const auto *replay_stage = replay_node ? dynamic_cast<const QwenSparseAttentionStage *>(replay_node->stage.get()) : nullptr;
    const auto *kva_stage = kva_node ? dynamic_cast<const KVCacheAppendStage *>(kva_node->stage.get()) : nullptr;
    const auto *key_stage = key_node ? dynamic_cast<const GEMMStage *>(key_node->stage.get()) : nullptr;
    const auto *query_stage = query_node ? dynamic_cast<const GEMMStage *>(query_node->stage.get()) : nullptr;
    const auto *q_norm_stage = q_norm_node ? dynamic_cast<const QKNormStage *>(q_norm_node->stage.get()) : nullptr;
    const auto *k_norm_stage = k_norm_node ? dynamic_cast<const QKNormStage *>(k_norm_node->stage.get()) : nullptr;
    const auto *idx_stage = idx_node ? dynamic_cast<const QwenSparseAttentionStage *>(idx_node->stage.get()) : nullptr;
    const auto *attn_stage = attn_node ? dynamic_cast<const QwenSparseAttentionStage *>(attn_node->stage.get()) : nullptr;
    const auto *commit_stage = commit_node ? dynamic_cast<const QwenSparseAttentionStage *>(commit_node->stage.get()) : nullptr;
    const auto *gate_stage = gate_node ? dynamic_cast<const AttentionOutputGateStage *>(gate_node->stage.get()) : nullptr;
    const auto *wo_stage = wo_node ? dynamic_cast<const GEMMStage *>(wo_node->stage.get()) : nullptr;

    if (!qkv_stage || !split_stage || !rope_stage || !replay_stage || !kva_stage ||
        !key_stage || !idx_stage || !attn_stage || !gate_stage || !wo_stage)
        return;

    const auto &qkv = qkv_stage->getParams();
    const auto &sp = split_stage->getParams();
    const auto &rp = rope_stage->getParams();
    const auto &rprep = replay_stage->getParams();
    const auto *prep_op = std::get_if<QwenSparseAttentionStage::PrepareReplay>(&rprep.operation);
    const auto &kva = kva_stage->getParams();
    const auto &kp = key_stage->getParams();
    const auto &idxp = idx_stage->getParams();
    const auto *idx_op = std::get_if<QwenSparseAttentionStage::Indexer>(&idxp.operation);
    const auto &attnp = attn_stage->getParams();
    const auto *sel_op = std::get_if<QwenSparseAttentionStage::SelectedAttention>(&attnp.operation);
    const auto &gp = gate_stage->getParams();
    const auto &wp = wo_stage->getParams();

    if (!qkv.device_id.is_cpu() || qkv.m != 1 || qkv.k <= 0 ||
        qkv.projection_set != AttentionProjectionSet::QueryKeyValue ||
        qkv.bias_q || qkv.bias_k || qkv.bias_v ||
        !qkv.prepared_store || !qkv.prepared_ref_q || !qkv.prepared_ref_k || !qkv.prepared_ref_v ||
        !qkv.input || !qkv.input_buffer_id ||
        sp.seq_len != 1 || sp.head_dim <= 0 || sp.n_heads <= 0 ||
        rp.Q_out || rp.K_out || rp.operand_set != RoPEOperandSet::QueryAndOptionalKey ||
        !prep_op || rprep.rowCapacity != 1 || prep_op->liveRows.has_value() ||
        !kva.kv_cache || kva.num_tokens != 1 || kva.batch_size != 1 ||
        kva.append_semantics != KVCacheAppendSemantics::Standard ||
        kva.kv_rotation != nullptr || kva.V_dequant_out != nullptr ||
        kp.m != 1 || kp.k != qkv.k || kp.A != qkv.input || kp.bias || kp.bias_tensor ||
        !kp.prepared_store || !kp.prepared_ref ||
        !idx_op || idx_op->transactionPrior.has_value() ||
        !sel_op || attnp.rowCapacity != 1 || sel_op->gate.has_value() ||
        gp.seq_len != 1 || gp.input != sel_op->output.tensor || gp.output != sel_op->output.tensor ||
        wp.m != 1 || wp.k != gp.feature_dim || wp.A != gp.output ||
        wp.bias || wp.bias_tensor || wp.alpha != 1.0f || wp.beta != 0.0f ||
        wp.needs_allreduce || !wp.prepared_store || !wp.prepared_ref)
        return;

    if (query_stage)
    {
        const auto &qp = query_stage->getParams();
        if (qp.m != 1 || qp.k != qkv.k || qp.A != qkv.input || qp.bias || qp.bias_tensor ||
            !qp.prepared_store || !qp.prepared_ref)
            return;
    }
    if (commit_stage &&
        !std::holds_alternative<QwenSparseAttentionStage::CommitPartial>(commit_stage->getParams().operation))
        return;

    std::string read_name;
    for (const auto &dep : qkv_node->dependencies)
    {
        const auto *node = graph.getNode(dep);
        const auto *prim = node ? dynamic_cast<const Primitive *>(node->stage.get()) : nullptr;
        const auto *read = prim ? std::get_if<Primitive::ResidualRead>(&prim->getParams().operation) : nullptr;
        if (read && read->output.tensor == qkv.input && read->output.buffer == *qkv.input_buffer_id &&
            read->output.byteOffset == 0)
        {
            if (!read_name.empty())
                return;
            read_name = dep;
        }
    }
    if (read_name.empty())
        return;

    CPUHCRegionNodes hc_region{};
    if (!discoverCPUHCRegion(graph, read_name, previousInputBoundary, hc_region))
        return;

    std::vector<std::string> member_names;
    if (!hc_region.previous_write.empty())
        member_names.push_back(hc_region.previous_write);
    member_names.push_back(hc_region.branch_norm);
    member_names.push_back(hc_region.down);
    if (!hc_region.read_activation.empty())
        member_names.push_back(hc_region.read_activation);
    member_names.push_back(hc_region.up);
    member_names.push_back(hc_region.read);
    if (!hc_region.injection.empty())
        member_names.push_back(hc_region.injection);
    member_names.push_back(qkv_name);
    member_names.push_back(split_name);
    if (q_norm_stage)
        member_names.push_back(q_norm_name);
    if (k_norm_stage)
        member_names.push_back(k_norm_name);
    member_names.push_back(rope_name);
    member_names.push_back(replay_name);
    member_names.push_back(kv_append_name);
    member_names.push_back(key_proj_name);
    if (query_stage)
        member_names.push_back(query_proj_name);
    member_names.push_back(indexer_name);
    member_names.push_back(attn_name);
    if (commit_stage)
        member_names.push_back(commit_name);
    member_names.push_back(gate_name);
    member_names.push_back(wo_name);

    ContractedStageMetadata meta;
    meta.absorb(graph, member_names);

    CPUQSADeclaration decl{
        .qkv = qkv,
        .split = sp,
        .q_norm = q_norm_stage ? std::optional{q_norm_stage->getParams()} : std::nullopt,
        .k_norm = k_norm_stage ? std::optional{k_norm_stage->getParams()} : std::nullopt,
        .rope = rp,
        .replay_stage = rprep,
        .kv_append = kva,
        .key_proj = kp,
        .query_proj = query_stage ? std::optional{query_stage->getParams()} : std::nullopt,
        .indexer_stage = idxp,
        .attn_stage = attnp,
        .commit_stage = commit_stage ? std::optional{commit_stage->getParams()} : std::nullopt,
        .gate = gp,
        .wo = wp,
    };

    auto replacement = std::make_unique<RetainedCPUQSAAttentionStage>(
        hc_region, std::move(decl), std::move(meta));

    GraphComposer composer(graph);
    std::vector<GraphNodeRef> members;
    members.reserve(member_names.size());
    for (const auto &m : member_names)
        members.push_back(composer.ref(m));

    (void)lowerLocalMegakernelRegion(graph, LocalMegakernelRegionSpec{
        .members = std::move(members),
        .terminal = composer.ref(wo_name),
        .replacement = std::move(replacement),
    });
}

void lowerRetainedCPUFFN(ComputeGraph &graph,
                         const std::string &layerPrefix,
                         const std::string &previousAttentionBoundary)
{
    using Primitive = Qwen4ExpPrimitiveStage;
    const std::string router_name = layerPrefix + "moe_routing";
    const std::string shared_name = layerPrefix + "shared_expert_ffn";
    const std::string routed_name = layerPrefix + "moe_expert_ffn";
    const std::string gate_name = layerPrefix + "shared_expert_gate";

    const auto *router_node = graph.getNode(router_name);
    const auto *shared_node = graph.getNode(shared_name);
    const auto *router_stage = router_node ? dynamic_cast<const MoERoutingStage *>(router_node->stage.get()) : nullptr;
    const auto *shared_stage = shared_node ? dynamic_cast<const SharedExpertFFNStage *>(shared_node->stage.get()) : nullptr;
    if (!router_stage || !shared_stage)
        return;

    const auto &rt = router_stage->declaration();
    const auto &sh = shared_stage->declaration();
    if (!rt.device_id.is_cpu() || rt.seq_len != 1 || rt.d_model <= 0 ||
        rt.num_experts <= 0 || rt.top_k <= 0 || rt.top_k > kDeviceMoEMaxTopK ||
        rt.force_decode_equivalent_verifier_prefill || rt.owned_rows.has_value() ||
        !rt.input || !rt.gate_weights || !rt.output_indices || !rt.output_weights ||
        sh.device_id != rt.device_id || sh.seq_len != 1 || sh.d_model != rt.d_model ||
        sh.input != rt.input || sh.force_decode_equivalent_verifier_prefill ||
        !sh.prepared_store || !sh.prepared_ref_gate || !sh.prepared_ref_up || !sh.prepared_ref_down ||
        !sh.output)
        return;

    const TensorType gate_type = rt.gate_weights->native_type();
    if (gate_type != TensorType::FP32 && gate_type != TensorType::FP16 && gate_type != TensorType::BF16)
        return;

    // Locate direct HC ResidualRead feeding moe_routing.
    std::string read_name;
    for (const auto &dep : router_node->dependencies)
    {
        const auto *node = graph.getNode(dep);
        const auto *prim = node ? dynamic_cast<const Primitive *>(node->stage.get()) : nullptr;
        const auto *read = prim ? std::get_if<Primitive::ResidualRead>(&prim->getParams().operation) : nullptr;
        if (read && read->output.tensor == rt.input && read->output.byteOffset == 0)
        {
            if (!read_name.empty())
                return;
            read_name = dep;
        }
    }
    if (read_name.empty())
        return;

    // Ensure shared_expert_ffn has no outside dependencies (e.g. a collective or routed branch).
    for (const auto &dep : shared_node->dependencies)
    {
        if (dep != read_name && dep != router_name)
            return;
    }

    CPUHCRegionNodes hc_region{};
    if (!discoverCPUHCRegion(graph, read_name, previousAttentionBoundary, hc_region))
        return;

    // Check if singleton-CPU local moe_expert_ffn + shared_expert_gate can ALSO be absorbed
    // into the exact same single-OMP FFN megakernel (zero collectives between them).
    std::optional<MoEExpertComputeStage::Params> routed_params;
    std::optional<SharedExpertGateStage::Params> gate_params;
    std::string terminal_name = shared_name;

    const auto *routed_node = graph.getNode(routed_name);
    const auto *gate_node = graph.getNode(gate_name);
    const auto *routed_stage = routed_node ? dynamic_cast<const MoEExpertComputeStage *>(routed_node->stage.get()) : nullptr;
    const auto *gate_stage = gate_node ? dynamic_cast<const SharedExpertGateStage *>(gate_node->stage.get()) : nullptr;
    if (routed_stage && gate_stage)
    {
        const auto &rp = routed_stage->declaration();
        const auto &gp = gate_stage->declaration();
        const bool routed_deps_ok = std::all_of(
            routed_node->dependencies.begin(), routed_node->dependencies.end(),
            [&](const std::string &dep) {
                return dep == read_name || dep == router_name || dep == shared_name;
            });
        const bool gate_deps_ok = std::all_of(
            gate_node->dependencies.begin(), gate_node->dependencies.end(),
            [&](const std::string &dep) {
                return dep == read_name || dep == router_name || dep == shared_name || dep == routed_name;
            });
        const bool all_experts_local_native =
            rp.device_id == rt.device_id && rp.seq_len == 1 && rp.d_model == rt.d_model &&
            rp.num_experts == rt.num_experts && rp.top_k == rt.top_k &&
            rp.input == rt.input && rp.output != nullptr &&
            rp.canonical_route_contributions == nullptr &&
            (rp.local_expert_count == -1 || rp.local_expert_count == rt.num_experts) &&
            rp.expert_mask.empty() && rp.replica_set.num_replicated == 0 &&
            !rp.force_decode_equivalent_verifier_prefill &&
            rp.prepared_gate_gemm.size() == static_cast<size_t>(rt.num_experts) &&
            rp.prepared_up_gemm.size() == static_cast<size_t>(rt.num_experts) &&
            rp.prepared_down_gemm.size() == static_cast<size_t>(rt.num_experts) &&
            CPUNativeQuantisedGemmKernel::supportsStandaloneMoEOneRowInTeam(
                rp.d_model, rp.expert_intermediate, static_cast<size_t>(rt.top_k)) &&
            std::all_of(rp.prepared_gate_gemm.begin(), rp.prepared_gate_gemm.end(),
                [](ITensorGemm *e) { return dynamic_cast<CPUNativeQuantisedGemmKernel *>(e) != nullptr; }) &&
            std::all_of(rp.prepared_up_gemm.begin(), rp.prepared_up_gemm.end(),
                [](ITensorGemm *e) { return dynamic_cast<CPUNativeQuantisedGemmKernel *>(e) != nullptr; }) &&
            std::all_of(rp.prepared_down_gemm.begin(), rp.prepared_down_gemm.end(),
                [](ITensorGemm *e) { return dynamic_cast<CPUNativeQuantisedGemmKernel *>(e) != nullptr; });
        const bool gate_ok =
            gp.device_id == rt.device_id && gp.seq_len == 1 && gp.d_model == rt.d_model &&
            gp.execution_role == SharedExpertGateStage::ExecutionRole::ReplicatedOwner &&
            gp.input == rt.input && gp.gate_inp != nullptr &&
            gp.shared_output == sh.output && gp.routed_residual == rp.output &&
            gp.combined_output != nullptr;

        if (routed_deps_ok && gate_deps_ok && all_experts_local_native && gate_ok)
        {
            routed_params = rp;
            gate_params = gp;
            terminal_name = gate_name;
        }
    }

    std::vector<std::string> member_names;
    if (!hc_region.previous_write.empty())
        member_names.push_back(hc_region.previous_write);
    member_names.push_back(hc_region.branch_norm);
    member_names.push_back(hc_region.down);
    if (!hc_region.read_activation.empty())
        member_names.push_back(hc_region.read_activation);
    member_names.push_back(hc_region.up);
    member_names.push_back(hc_region.read);
    if (!hc_region.injection.empty())
        member_names.push_back(hc_region.injection);
    member_names.push_back(router_name);
    member_names.push_back(shared_name);
    if (routed_params && gate_params)
    {
        member_names.push_back(routed_name);
        member_names.push_back(gate_name);
    }

    ContractedStageMetadata meta;
    meta.absorb(graph, member_names);
    meta.workspace_reqs.cpu_router_q8_storage = std::make_shared<const CPURouterQ8StorageLayout>(
        CPURouterQ8StorageLayout::declare(std::max(1, rt.seq_len), rt.d_model, rt.device_id));

    auto replacement = std::make_unique<RetainedCPUFFNStage>(
        hc_region,
        CPUFFNDeclaration{rt, sh, std::move(routed_params), std::move(gate_params)},
        std::move(meta));

    GraphComposer composer(graph);
    std::vector<GraphNodeRef> members;
    members.reserve(member_names.size());
    for (const auto &m : member_names)
        members.push_back(composer.ref(m));

    (void)lowerLocalMegakernelRegion(graph, LocalMegakernelRegionSpec{
        .members = std::move(members),
        .terminal = composer.ref(terminal_name),
        .replacement = std::move(replacement),
    });
}

} // namespace llaminar2
