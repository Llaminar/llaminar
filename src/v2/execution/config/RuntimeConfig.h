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

#include "../../backends/ComputeBackend.h"
#include "../../utils/CPUFeatures.h"
#include "../../utils/Logger.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <optional>
#include <sstream>
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
        JIT,        ///< AVX-512 VNNI JIT (fastest, default)
        REFERENCE,  ///< Pure C++ reference (for testing/debugging)
        TILED,      ///< Cache-blocked tiled (balanced)
        Q16_INTEGER ///< Pure integer Q16_1 reference (experimental)
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
        if (s == "q16_integer" || s == "q16" || s == "q16_int")
            return FusedAttentionBackend::Q16_INTEGER;
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
        case FusedAttentionBackend::Q16_INTEGER:
            return "Q16_INTEGER";
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
        FP32,      ///< 32-bit float activations (default, highest accuracy)
        BF16,      ///< bfloat16 activations (Intel AMX, reduced bandwidth)
        FP16,      ///< 16-bit float activations (ARM/GPU optimization)
        Q8_1,      ///< Block-quantized int8 (36 bytes per 32 elements, 3.5x compression)
        Q16_1,     ///< Block-quantized int16 (72 bytes per 32 elements, 266× better than Q8_1)
        Hybrid,    ///< Mixed precision: FP32 residual, BF16 KV cache, Q8_1 QKV activations
        HybridQ16, ///< Mixed precision: Q16_1 residual, Q8_1 activations (62% memory savings)
        TQ4,       ///< TurboQuant 4-bit KV cache (7.5× compression vs FP32)
        TQ8        ///< TurboQuant 8-bit (K-projection cache, SQNR 38.79 dB)
    };

    /**
     * @brief Convert ActivationPrecision enum to string for logging
     */
    inline const char *activationPrecisionToString(ActivationPrecision prec)
    {
        switch (prec)
        {
        case ActivationPrecision::FP32:
            return "FP32";
        case ActivationPrecision::BF16:
            return "BF16";
        case ActivationPrecision::FP16:
            return "FP16";
        case ActivationPrecision::Q8_1:
            return "Q8_1";
        case ActivationPrecision::Q16_1:
            return "Q16_1";
        case ActivationPrecision::Hybrid:
            return "Hybrid";
        case ActivationPrecision::HybridQ16:
            return "HybridQ16";
        case ActivationPrecision::TQ4:
            return "TQ4";
        case ActivationPrecision::TQ8:
            return "TQ8";
        default:
            return "Unknown";
        }
    }

    /**
     * @brief Explicit KV cache storage precision mode
     *
     * AUTO defaults to FP16 — half the VRAM of FP32 with <2% decode throughput impact
     * (GPU-side conversion via hip_convert_tensor_to_fp32).
     */
    enum class KVCachePrecision
    {
        AUTO,
        FP32,
        FP16,
        Q8_1,
        Q16_1,
        TQ4,
        TQ ///< TurboQuant asymmetric: TQ8 for K, TQ4 for V
    };

    inline const char *kvCachePrecisionToString(KVCachePrecision precision)
    {
        switch (precision)
        {
        case KVCachePrecision::AUTO:
            return "AUTO (Q16_1 on CPU, FP16 on GPU)";
        case KVCachePrecision::FP32:
            return "FP32";
        case KVCachePrecision::FP16:
            return "FP16";
        case KVCachePrecision::Q8_1:
            return "Q8_1";
        case KVCachePrecision::Q16_1:
            return "Q16_1";
        case KVCachePrecision::TQ4:
            return "TQ4";
        case KVCachePrecision::TQ:
            return "TQ (TQ8 K + TQ4 V)";
        default:
            return "UNKNOWN";
        }
    }

    inline KVCachePrecision parseKVCachePrecision(const std::string &value)
    {
        std::string lower = value;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });

        if (lower == "fp32" || lower == "f32")
            return KVCachePrecision::FP32;
        if (lower == "fp16" || lower == "f16")
            return KVCachePrecision::FP16;
        if (lower == "q8_1" || lower == "q8" || lower == "q81")
            return KVCachePrecision::Q8_1;
        if (lower == "q16_1" || lower == "q16" || lower == "q161" || lower == "i16" || lower == "int16")
            return KVCachePrecision::Q16_1;
        if (lower == "tq4")
            return KVCachePrecision::TQ4;
        if (lower == "tq")
            return KVCachePrecision::TQ;
        return KVCachePrecision::AUTO;
    }

    inline ActivationPrecision resolveKVCacheStoragePrecision(
        KVCachePrecision mode, bool is_cpu = false)
    {
        switch (mode)
        {
        case KVCachePrecision::FP32:
            return ActivationPrecision::FP32;
        case KVCachePrecision::FP16:
            return ActivationPrecision::FP16;
        case KVCachePrecision::Q8_1:
            return ActivationPrecision::Q8_1;
        case KVCachePrecision::Q16_1:
            return ActivationPrecision::Q16_1;
        case KVCachePrecision::TQ4:
            return ActivationPrecision::TQ4;
        case KVCachePrecision::TQ:
            return ActivationPrecision::TQ8; // K precision; V uses TQ4 via asymmetric cache
        case KVCachePrecision::AUTO:
        default:
            // CPU: Q16_1 uses VNNI int16 attention — ~1.4x decode speedup, 50% KV memory
            // GPU: FP16 — half the VRAM with <2% decode throughput cost
            return is_cpu ? ActivationPrecision::Q16_1 : ActivationPrecision::FP16;
        }
    }

    /**
     * @brief Get bytes per element for ActivationPrecision
     *
     * Returns the storage size per logical element:
     * - FP32: 4.0 bytes
     * - BF16: 2.0 bytes
     * - FP16: 2.0 bytes
     * - Q8_1: 1.125 bytes (36 bytes per 32 elements)
     * - Q16_1: 2.25 bytes (72 bytes per 32 elements)
     * - Hybrid/HybridQ16: 4.0 bytes (worst-case FP32 for buffer sizing)
     *
     * For buffer allocation, multiply element count by this value.
     * Note: Q8_1/Q16_1 return average bytes/element; actual allocation
     * should round up to block boundaries (32 elements).
     */
    inline float activationPrecisionBytesPerElement(ActivationPrecision prec)
    {
        switch (prec)
        {
        case ActivationPrecision::FP32:
            return 4.0f;
        case ActivationPrecision::BF16:
        case ActivationPrecision::FP16:
            return 2.0f;
        case ActivationPrecision::Q8_1:
            return 36.0f / 32.0f; // 1.125 bytes/element
        case ActivationPrecision::Q16_1:
            return 72.0f / 32.0f; // 2.25 bytes/element
        case ActivationPrecision::TQ4:
            return 68.0f / 128.0f; // ~0.53 bytes/element (head_dim=128)
        case ActivationPrecision::TQ8:
            return 136.0f / 128.0f; // ~1.0625 bytes/element (head_dim=128)
        case ActivationPrecision::Hybrid:
        case ActivationPrecision::HybridQ16:
        default:
            // Use FP32 as worst-case for buffer sizing
            return 4.0f;
        }
    }

    /**
     * @brief Calculate buffer bytes for element count with ActivationPrecision
     *
     * Properly handles block quantization alignment for Q8_1 and Q16_1.
     *
     * @param element_count Number of logical elements
     * @param prec Activation precision format
     * @return Bytes needed for storage (rounded up for block formats)
     */
    inline size_t activationPrecisionBufferBytes(size_t element_count, ActivationPrecision prec)
    {
        switch (prec)
        {
        case ActivationPrecision::FP32:
            return element_count * 4;
        case ActivationPrecision::BF16:
        case ActivationPrecision::FP16:
            return element_count * 2;
        case ActivationPrecision::Q8_1:
        {
            // Q8_1: 36 bytes per 32 elements, round up to block boundary
            size_t blocks = (element_count + 31) / 32;
            return blocks * 36;
        }
        case ActivationPrecision::Q16_1:
        {
            // Q16_1: 72 bytes per 32 elements, round up to block boundary
            size_t blocks = (element_count + 31) / 32;
            return blocks * 72;
        }
        case ActivationPrecision::Hybrid:
        case ActivationPrecision::HybridQ16:
        default:
            // Use FP32 as worst-case for buffer sizing
            return element_count * 4;
        }
    }

    /**
     * @brief Parse ActivationPrecision from string
     * @param value Precision name ("fp32", "bf16", "fp16", "q8_1", "q16_1", "hybrid", "hybridq16")
     * @return Corresponding enum value, or FP32 if unrecognized
     */
    inline ActivationPrecision parseActivationPrecision(const std::string &value)
    {
        std::string lower = value;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });

        if (lower == "bf16")
            return ActivationPrecision::BF16;
        if (lower == "fp16")
            return ActivationPrecision::FP16;
        if (lower == "q8_1")
            return ActivationPrecision::Q8_1;
        if (lower == "q16_1")
            return ActivationPrecision::Q16_1;
        if (lower == "hybrid")
            return ActivationPrecision::Hybrid;
        if (lower == "hybridq16")
            return ActivationPrecision::HybridQ16;
        return ActivationPrecision::FP32;
    }

    /**
     * @brief Routed MoE expert execution mode for the standard Qwen3.5 MoE path.
     */
    enum class MoEExpertMode
    {
        ExpertParallel, ///< Split whole expert ids across TP participants
        TensorParallel, ///< Shard every selected expert internally (not implemented yet)
        Replicated      ///< Keep routed expert tensors/execution fully replicated
    };

    inline const char *moeExpertModeToString(MoEExpertMode mode)
    {
        switch (mode)
        {
        case MoEExpertMode::ExpertParallel:
            return "expert-parallel";
        case MoEExpertMode::TensorParallel:
            return "tensor-parallel";
        case MoEExpertMode::Replicated:
            return "replicated";
        default:
            return "unknown";
        }
    }

    inline std::optional<MoEExpertMode> parseMoEExpertMode(const std::string &value)
    {
        std::string lower = value;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        std::replace(lower.begin(), lower.end(), '_', '-');

        if (lower == "expert-parallel" || lower == "ep")
            return MoEExpertMode::ExpertParallel;
        if (lower == "tensor-parallel" || lower == "tp" || lower == "tensor-parallel-experts")
            return MoEExpertMode::TensorParallel;
        if (lower == "replicated" || lower == "replicate" || lower == "full")
            return MoEExpertMode::Replicated;
        return std::nullopt;
    }

    /**
     * @brief User-facing bounded hot expert cache configuration.
     */
    struct MoEHotExpertCacheConfig
    {
        enum class Kind
        {
            Percent,
            Count,
            Off
        };

        Kind kind = Kind::Percent;
        int count = 0;
        float percent = 10.0f;

        bool enabled() const { return kind != Kind::Off; }

        int resolveCap(int num_experts, bool dynamic_rebalance_enabled) const
        {
            if (kind == Kind::Off || num_experts <= 0)
                return 0;
            if (kind == Kind::Count)
                return std::max(0, std::min(count, num_experts));

            const float clamped = std::max(0.0f, std::min(percent, 100.0f));
            int resolved = static_cast<int>(std::floor(static_cast<float>(num_experts) * clamped / 100.0f));
            if (dynamic_rebalance_enabled && clamped > 0.0f && resolved == 0)
                resolved = 1;
            return std::max(0, std::min(resolved, num_experts));
        }

        std::string toString() const
        {
            if (kind == Kind::Off)
                return "off";
            if (kind == Kind::Count)
                return std::to_string(count);
            std::ostringstream oss;
            oss << percent << "%";
            return oss.str();
        }
    };

    /**
     * @brief MoE decode histogram and dynamic rebalance configuration.
     */
    enum class MoERebalanceRuntimeMode
    {
        Off,
        Observe,
        Dynamic
    };

    inline const char *moeRebalanceRuntimeModeToString(MoERebalanceRuntimeMode mode)
    {
        switch (mode)
        {
        case MoERebalanceRuntimeMode::Off:
            return "off";
        case MoERebalanceRuntimeMode::Observe:
            return "observe";
        case MoERebalanceRuntimeMode::Dynamic:
            return "dynamic";
        default:
            return "unknown";
        }
    }

    inline std::optional<MoERebalanceRuntimeMode> parseMoERebalanceRuntimeMode(const std::string &value)
    {
        std::string lower = value;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        std::replace(lower.begin(), lower.end(), '_', '-');

        if (lower == "off" || lower == "disabled" || lower == "false")
            return MoERebalanceRuntimeMode::Off;
        if (lower == "observe" || lower == "observer")
            return MoERebalanceRuntimeMode::Observe;
        if (lower == "dynamic" || lower == "on" || lower == "true")
            return MoERebalanceRuntimeMode::Dynamic;
        return std::nullopt;
    }

    struct MoERebalanceRuntimeConfig
    {
        MoERebalanceRuntimeMode mode = MoERebalanceRuntimeMode::Dynamic;
        int window_size = 256;
        int max_window_size = 4096;
        float window_growth_factor = 1.5f;
        bool release_raw_expert_weights = false;
    };

    /**
     * @brief Canonical runtime configuration carried through the config chain
     *
     * RuntimeConfig holds pre-parsed runtime parameters that flow from
     * CLI/YAML (OrchestrationConfig) through the execution plan into
     * per-device runner configs. Fields are parsed once during plan
     * building and carried as typed values thereafter.
     *
     * The config chain: OrchestrationConfig (raw strings)
     *   → ExecutionPlanBuilder parses once → RankExecutionPlan.runtime
     *     → MDO::Config / InferenceRunnerConfig read from it
     *       → GraphConfig consumes the values
     */
    struct RuntimeConfig
    {
        /// Maximum sequence length (buffer allocation sizing)
        int max_seq_len = 4096;

        /// Batch size (currently must be 1)
        int batch_size = 1;

        /// Activation buffer precision
        ActivationPrecision activation_precision = ActivationPrecision::FP32;

        /// Fused attention backend selection
        FusedAttentionBackend fused_attention_backend = FusedAttentionBackend::JIT;

        /// Fixed scales for Q16_1 KV cache quantization (K and V separate)
        float kv_cache_scale_k = 256.0f;
        float kv_cache_scale_v = 32.0f;

        /// Explicit KV cache precision (AUTO defaults to FP16)
        KVCachePrecision kv_cache_precision = KVCachePrecision::AUTO;

        /// Routed MoE expert execution mode.
        MoEExpertMode moe_expert_mode = MoEExpertMode::ExpertParallel;

        /// Bounded remote hot-expert cache configuration for dynamic EP.
        MoEHotExpertCacheConfig moe_hot_expert_cache;

        /// MoE rebalance runtime configuration.
        MoERebalanceRuntimeConfig moe_rebalance;

        RuntimeConfig() = default;

        explicit RuntimeConfig(int max_seq_len_) : max_seq_len(max_seq_len_) {}

        /**
         * @brief Create RuntimeConfig by parsing raw strings from OrchestrationConfig
         */
        static RuntimeConfig fromOrchestrationConfig(
            int max_seq_len,
            const std::string &activation_precision_str,
            const std::string &kv_cache_precision_str,
            FusedAttentionBackend fused_backend = FusedAttentionBackend::JIT,
            MoEExpertMode moe_expert_mode = MoEExpertMode::ExpertParallel,
            MoEHotExpertCacheConfig moe_hot_expert_cache = {},
            MoERebalanceRuntimeConfig moe_rebalance = {})
        {
            RuntimeConfig rc;
            rc.max_seq_len = max_seq_len;
            rc.activation_precision = parseActivationPrecision(activation_precision_str);
            rc.kv_cache_precision = parseKVCachePrecision(kv_cache_precision_str);
            rc.fused_attention_backend = fused_backend;
            rc.moe_expert_mode = moe_expert_mode;
            rc.moe_hot_expert_cache = moe_hot_expert_cache;
            rc.moe_rebalance = moe_rebalance;
            return rc;
        }
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
