/**
 * @file RuntimeConfig.h
 * @brief Runtime configuration for inference
 * @author David Sanftenberg
 * @date 2025-10-25
 *
 * Encapsulates runtime parameters that affect inference behavior but are not
 * part of the model architecture (which comes from GGUF metadata).
 */

#pragma once

#include "../backends/ComputeBackend.h"
#include "../utils/CPUFeatures.h"
#include "../utils/Logger.h"
#include <string>

namespace llaminar2
{

    /**
     * @brief Fused attention execution backend
     *
     * Selects which implementation to use for fused attention + Wo projection.
     */
    enum class FusedAttentionBackend
    {
        JIT,       ///< AVX-512 VNNI JIT (fastest, default)
        REFERENCE, ///< Pure C++ reference (for testing/debugging)
        TILED      ///< Cache-blocked tiled (balanced)
    };

    /**
     * @brief Parse FusedAttentionBackend from string
     * @param s Backend name ("jit", "reference", "tiled")
     * @return Corresponding backend enum, or JIT if unrecognized
     */
    inline FusedAttentionBackend parseFusedAttentionBackend(const std::string &s)
    {
        if (s == "reference" || s == "ref")
            return FusedAttentionBackend::REFERENCE;
        if (s == "tiled")
            return FusedAttentionBackend::TILED;
        return FusedAttentionBackend::JIT; // Default
    }

    /**
     * @brief Convert FusedAttentionBackend to string
     */
    inline const char *fusedAttentionBackendToString(FusedAttentionBackend b)
    {
        switch (b)
        {
        case FusedAttentionBackend::JIT:
            return "JIT";
        case FusedAttentionBackend::REFERENCE:
            return "REFERENCE";
        case FusedAttentionBackend::TILED:
            return "TILED";
        default:
            return "UNKNOWN";
        }
    }

    /**
     * @brief Weight loading strategy
     *
     * Determines how weights are loaded from GGUF and stored in memory.
     * This is independent of the compute precision used during inference.
     *
     * NATIVE (default): Keep weights in original GGUF format
     *   - Memory efficient (weights stay compressed)
     *   - Dequantization happens on-the-fly in GEMM kernels (per-operation)
     *   - Original formats: IQ4_NL, Q4_0, Q6_K, Q8_0, F16, F32, etc.
     *   - Best for memory-constrained environments
     *
     * CONVERT_TO_FP32: Dequantize all weights to FP32 at load time
     *   - Higher memory usage (4 bytes per weight element)
     *   - No runtime dequantization overhead
     *   - Useful for parity testing against reference implementations
     *
     * CONVERT_TO_BF16: Dequantize all weights to BF16 at load time
     *   - Moderate memory usage (2 bytes per weight element)
     *   - Best for Intel Sapphire Rapids+ (AMX BF16 instructions)
     *
     * CONVERT_TO_FP16: Dequantize all weights to FP16 at load time
     *   - Moderate memory usage (2 bytes per weight element)
     *   - Best for ARM/mobile/GPU hardware with native FP16 support
     *
     * CONVERT_TO_INT8: Dequantize all weights to INT8 at load time
     *   - Low memory usage (1 byte per weight element)
     *   - Enables AVX512-VNNI (CPU) and CUDA INT8 Tensor Cores
     *   - Requires scale factors to be stored separately
     */
    enum class WeightPrecision
    {
        NATIVE,          ///< Keep weights in original GGUF format (default, on-the-fly dequant)
        CONVERT_TO_FP32, ///< Dequantize all weights to FP32 at load (high memory, no runtime dequant)
        CONVERT_TO_BF16, ///< Dequantize all weights to BF16 at load (Intel AMX optimization)
        CONVERT_TO_FP16, ///< Dequantize all weights to FP16 at load (ARM/GPU optimization)
        CONVERT_TO_INT8  ///< Dequantize all weights to INT8 at load (AVX512-VNNI, CUDA INT8)
    };

    /**
     * @brief Compute precision for intermediate activations and accumulation
     *
     * Determines the precision used for:
     * - Activation tensors (hidden states between layers)
     * - Accumulation buffers (GEMM output, attention scores, etc.)
     * - Intermediate computations (softmax, RMSNorm, SwiGLU, etc.)
     *
     * This is INDEPENDENT of weight precision - you can have:
     * - Native quantized weights (IQ4_NL) with FP32 activations (most common)
     * - FP32 weights with BF16 activations (memory bandwidth optimization)
     * - Quantized weights with INT32 activations (accumulation buffer for GEMM)
     * - Any combination that makes sense for your use case
     *
     * FP32 (default): All activations and accumulation in 32-bit float
     *   - Highest numerical accuracy
     *   - Standard baseline for correctness validation
     *   - 4 bytes per activation element
     *
     * BF16: All activations and accumulation in bfloat16
     *   - Reduced memory bandwidth (2× faster on Ice Lake+)
     *   - Slightly reduced accuracy (acceptable for most models)
     *   - 2 bytes per activation element
     *   - Requires BF16-aware kernels for RMSNorm, Softmax, etc.
     *
     * FP16: All activations and accumulation in half precision
     *   - Reduced memory bandwidth (faster on ARM/GPU)
     *   - Requires careful handling of numerical stability
     *   - 2 bytes per activation element
     *
     * Q8_1: Block-quantized 8-bit integer with scale and sum
     *   - Extreme memory bandwidth reduction (1.125 bytes per element)
     *   - 36 bytes per 32 elements (Q8_1Block structure)
     *   - Ideal for residual storage (3.5x compression vs FP32)
     *   - Pre-computed sum enables efficient VNNI dot products
     */
    enum class ActivationPrecision
    {
        FP32, ///< 32-bit float activations (default, highest accuracy)
        BF16, ///< bfloat16 activations (Intel AMX, reduced bandwidth)
        FP16, ///< 16-bit float activations (ARM/GPU optimization)
        Q8_1  ///< Block-quantized int8 (36 bytes per 32 elements, 3.5x compression)
    };

    /**
     * @brief Runtime configuration for pipeline initialization
     *
     * This struct separates runtime configuration (user-specified parameters)
     * from model architecture (GGUF metadata) and device placement (WeightPlacementMap).
     *
     * Design rationale:
     * - ModelContext: Model file metadata (immutable, from GGUF)
     * - WeightPlacementMap: Device placement decisions (Phase 4)
     * - RuntimeConfig: Runtime behavior settings (this struct)
     *
     * This separation follows V2's philosophy of clear ownership and composability.
     */
    struct RuntimeConfig
    {
        /**
         * @brief Maximum sequence length for inference
         *
         * Determines buffer allocation size for:
         * - KV cache: n_layers × max_seq_len × n_kv_heads × head_dim
         * - Activation buffers: max_seq_len × d_model (various stages)
         *
         * Typical values:
         * - 512: Short conversations, low memory
         * - 2048: Standard conversations (default)
         * - 4096: Long context
         * - 8192+: Extended context models
         *
         * Note: Actual sequence can be shorter, but cannot exceed this limit.
         */
        int max_seq_len = 2048;

        /**
         * @brief Number of OpenMP threads for CPU operations
         *
         * -1 = auto-detect (use all available cores)
         * 0 = single-threaded
         * >0 = explicit thread count
         *
         * Note: Ignored for GPU operations
         */
        int n_threads = -1;

        /**
         * @brief Batch size for batched inference
         *
         * Future extension for batched processing (not yet implemented in V2).
         * Currently must be 1.
         */
        int batch_size = 1;

        /**
         * @brief Enable memory-mapped file access for weights
         *
         * true = mmap weights (lower memory, slower first access)
         * false = load all weights to RAM (higher memory, faster access)
         */
        bool use_mmap = true;

        /**
         * @brief Random seed for sampling
         *
         * -1 = random seed (time-based)
         * ≥0 = deterministic seed for reproducibility
         */
        int seed = -1;

        /**
         * @brief Weight loading precision (how weights are stored in memory)
         *
         * Determines whether weights are kept in original GGUF format or
         * immediately converted to a different format at load time.
         *
         * Examples:
         * - NATIVE: IQ4_NL weights stay as IQ4_NL (dequantized on-the-fly in kernels)
         * - CONVERT_TO_FP32: IQ4_NL weights converted to FP32 at load (no runtime dequant)
         * - CONVERT_TO_INT8: Q8_0 weights converted to INT8 at load
         *
         * Default: NATIVE (memory-efficient, dequantize on-the-fly)
         */
        WeightPrecision weight_precision = WeightPrecision::NATIVE;

        /**
         * @brief Activation buffer format
         *
         * Determines the format for all activation buffers in DRAM:
         * - Hidden states between layers
         * - KV cache entries
         * - Residual buffers
         * - Intermediate computation results
         *
         * Memory per element:
         * - FP32: 4.0 bytes (highest accuracy)
         * - BF16: 2.0 bytes (Intel AMX optimization)
         * - FP16: 2.0 bytes (ARM/GPU optimization)
         * - Q8_1: 1.125 bytes (36 bytes per 32 elements, maximum compression)
         *
         * Kernels handle format conversion internally. This setting only
         * affects how data is stored in DRAM between kernel invocations.
         *
         * Default: FP32 (standard baseline)
         */
        ActivationPrecision activation_precision = ActivationPrecision::FP32;

        /**
         * @brief Enable fused attention + Wo projection kernel
         *
         * When enabled, replaces the separate attention + Wo GEMM calls with
         * a single fused kernel that:
         * 1. Eliminates the context quantization round-trip (FP32 → Q8_1 → FP32)
         * 2. Improves cache locality (context stays in registers through projection)
         * 3. Reduces memory bandwidth (single pass over V and Wo)
         *
         * Requirements:
         * - Q8_1 activation precision (activation_precision == Q8_1)
         * - Will fall back to unfused path if requirements not met
         *
         * Performance impact:
         * - ~10-15% faster attention computation (fewer memory passes)
         * - ~5% better numerical accuracy (eliminates one quantization step)
         *
         * Default: false (use proven unfused path)
         */
        bool use_fused_attention = false;

        /**
         * @brief Fused attention execution backend
         *
         * Selects which implementation to use when use_fused_attention=true:
         * - JIT (default): AVX-512 VNNI optimized, fastest
         * - REFERENCE: Pure C++ implementation, for testing
         * - TILED: Cache-blocked implementation, good balance
         *
         * Only affects fused attention path; ignored if use_fused_attention=false.
         */
        FusedAttentionBackend fused_attention_backend = FusedAttentionBackend::JIT;

        // ===== Multi-Device Executor Feature Flags (Incremental Rollout) =====

        /**
         * @brief Enable LayerExecutor for FFN RMSNorm operations
         *
         * When enabled, FFN pre-normalization uses the new LayerExecutor framework
         * for multi-device work distribution. Default disabled for safety.
         */
        bool executor_ffn_norm = false;

        /**
         * @brief Enable LayerExecutor for FFN SwiGLU operations
         *
         * When enabled, SwiGLU activation uses the new LayerExecutor framework.
         * Default disabled for safety.
         */
        bool executor_ffn_swiglu = false;

        /**
         * @brief Enable LayerExecutor for FFN residual add operations
         *
         * When enabled, FFN residual connections use the new LayerExecutor framework.
         * Default disabled for safety.
         */
        bool executor_ffn_residual = false;

        /**
         * @brief Enable LayerExecutor for attention RMSNorm operations
         *
         * When enabled, pre-attention RMSNorm uses the new LayerExecutor framework.
         * Default disabled for safety.
         */
        bool executor_attn_norm = false;

        /**
         * @brief Enable LayerExecutor for attention residual add operations
         *
         * When enabled, attention residual connections use the new LayerExecutor framework.
         * Default disabled for safety.
         */
        bool executor_attn_residual = false;

        /**
         * @brief Enable LayerExecutor for RoPE operations
         *
         * When enabled, RoPE positional encoding uses the new LayerExecutor framework.
         * Default disabled for safety.
         */
        bool executor_rope = false;

        /**
         * @brief Default constructor with standard settings
         */
        RuntimeConfig() = default;

        /**
         * @brief Construct with explicit max_seq_len (common case)
         */
        explicit RuntimeConfig(int max_seq_len_) : max_seq_len(max_seq_len_) {}
    };

    /**
     * @brief Auto-select optimal activation precision for a device
     *
     * Selection priority (highest to lowest performance):
     * 1. BF16: If AMX-BF16 or AVX512-BF16 available (Intel Sapphire Rapids+)
     * 2. FP16: If AVX512-FP16 available (Intel Sapphire Rapids+, no BF16)
     * 3. FP32: Fallback (universal compatibility)
     *
     * Hardware requirements:
     * - BF16: Intel Ice Lake+ (AMX-BF16), Cooper Lake+ (AVX512-BF16)
     * - FP16: Intel Sapphire Rapids+ (AVX512-FP16), ARM NEON (future)
     * - FP32: All architectures
     *
     * Note: INT8 is not auto-selected (requires explicit opt-in)
     *
     * @param device Device to query for capabilities
     * @return Recommended activation precision mode
     */
    inline ActivationPrecision selectOptimalActivationPrecision(const ComputeDevice &device)
    {
        switch (device.type)
        {
        case ComputeBackendType::CPU:
        {
            // Priority 1: AMX-BF16 (Intel Sapphire Rapids+, 4th gen Xeon)
            if (cpu_supports_amx_bf16())
            {
                LOG_INFO("AUTO activation precision: Detected AMX-BF16 → selecting BF16");
                LOG_INFO("  Expected: 50% memory bandwidth, 1.5-2× throughput vs FP32");
                return ActivationPrecision::BF16;
            }

            // Priority 2: AVX512-BF16 (Intel Cooper Lake+, 3rd gen Xeon)
            if (cpu_supports_avx512_bf16())
            {
                LOG_INFO("AUTO activation precision: Detected AVX512-BF16 → selecting BF16");
                LOG_INFO("  Expected: 50% memory bandwidth, 1.3-1.8× throughput vs FP32");
                return ActivationPrecision::BF16;
            }

            // Priority 3: AVX512-FP16 (Intel Sapphire Rapids+, but no BF16?)
            if (cpu_supports_avx512_fp16())
            {
                LOG_INFO("AUTO activation precision: Detected AVX512-FP16 → selecting FP16");
                LOG_INFO("  Expected: 50% memory bandwidth, 1.2-1.6× throughput vs FP32");
                return ActivationPrecision::FP16;
            }

            // Fallback: FP32 (universal)
            LOG_INFO("AUTO activation precision: No FP16/BF16 acceleration detected → selecting FP32");
            LOG_INFO("  CPU: " << cpu_vendor());
            LOG_INFO("  AVX512: " << (cpu_supports_avx512() ? "yes" : "no"));
            LOG_INFO("  AVX2: " << (cpu_supports_avx2() ? "yes" : "no"));
            return ActivationPrecision::FP32;
        }

        case ComputeBackendType::GPU_CUDA:
        {
            // CUDA devices: Check hardware capabilities
            if (device.supports_bf16)
            {
                LOG_INFO("AUTO activation precision: CUDA device supports BF16 → selecting BF16");
                LOG_INFO("  Device: " << device.name);
                LOG_INFO("  Expected: 50% memory bandwidth, tensor core acceleration");
                return ActivationPrecision::BF16;
            }
            else if (device.supports_fp16)
            {
                LOG_INFO("AUTO activation precision: CUDA device supports FP16 → selecting FP16");
                LOG_INFO("  Device: " << device.name);
                LOG_INFO("  Expected: 50% memory bandwidth, tensor core acceleration");
                return ActivationPrecision::FP16;
            }
            else
            {
                LOG_INFO("AUTO activation precision: CUDA device, no FP16/BF16 → selecting FP32");
                LOG_INFO("  Device: " << device.name);
                return ActivationPrecision::FP32;
            }
        }

        case ComputeBackendType::GPU_ROCM:
        {
            // ROCm devices: Prefer FP16 (better support than BF16 on AMD)
            if (device.supports_fp16)
            {
                LOG_INFO("AUTO activation precision: ROCm device supports FP16 → selecting FP16");
                LOG_INFO("  Device: " << device.name);
                LOG_INFO("  Expected: 50% memory bandwidth, matrix core acceleration");
                return ActivationPrecision::FP16;
            }
            else if (device.supports_bf16)
            {
                LOG_INFO("AUTO activation precision: ROCm device supports BF16 → selecting BF16");
                LOG_INFO("  Device: " << device.name);
                return ActivationPrecision::BF16;
            }
            else
            {
                LOG_INFO("AUTO activation precision: ROCm device, no FP16/BF16 → selecting FP32");
                LOG_INFO("  Device: " << device.name);
                return ActivationPrecision::FP32;
            }
        }

        case ComputeBackendType::GPU_VULKAN:
        {
            // Vulkan: Conservative FP32 for now (extension-dependent)
            LOG_INFO("AUTO activation precision: Vulkan device → selecting FP32 (conservative)");
            LOG_INFO("  Device: " << device.name);
            LOG_INFO("  Note: FP16/BF16 support depends on extensions (not yet detected)");
            return ActivationPrecision::FP32;
        }

        default:
            LOG_WARN("AUTO activation precision: Unknown device type → defaulting to FP32");
            return ActivationPrecision::FP32;
        }
    }

} // namespace llaminar2
