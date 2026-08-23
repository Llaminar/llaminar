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
#include "../../utils/DebugEnv.h"
#include "../../utils/Logger.h"
#include "RoutedExpertPolicy.h"
#include "../moe/DeviceMoERebalancePolicyShared.h"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
        TQ8,       ///< TurboQuant 8-bit value storage
        AQ8        ///< Cubic-companded int8 attention-key storage
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
        case ActivationPrecision::AQ8:
            return "AQ8";
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

    inline std::string normalizeRuntimeConfigToken(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        std::replace(value.begin(), value.end(), '-', '_');
        return value;
    }

    enum class PrefixCacheStorageMode
    {
        Disabled,
        Ram,
        Device,
        Tiered,
    };

    inline const char *prefixCacheStorageModeToString(PrefixCacheStorageMode mode)
    {
        switch (mode)
        {
        case PrefixCacheStorageMode::Disabled:
            return "disabled";
        case PrefixCacheStorageMode::Ram:
            return "ram";
        case PrefixCacheStorageMode::Device:
            return "device";
        case PrefixCacheStorageMode::Tiered:
            return "tiered";
        default:
            return "unknown";
        }
    }

    inline std::optional<PrefixCacheStorageMode> parsePrefixCacheStorageMode(const std::string &value)
    {
        const std::string normalized = normalizeRuntimeConfigToken(value);
        if (normalized == "disabled" || normalized == "off")
            return PrefixCacheStorageMode::Disabled;
        if (normalized == "ram")
            return PrefixCacheStorageMode::Ram;
        if (normalized == "device" || normalized == "vram")
            return PrefixCacheStorageMode::Device;
        if (normalized == "tiered")
            return PrefixCacheStorageMode::Tiered;
        return std::nullopt;
    }

    enum class PrefixCacheTerminalStateMode
    {
        Off,
        Auto,
        Always,
    };

    inline const char *prefixCacheTerminalStateModeToString(PrefixCacheTerminalStateMode mode)
    {
        switch (mode)
        {
        case PrefixCacheTerminalStateMode::Off:
            return "off";
        case PrefixCacheTerminalStateMode::Auto:
            return "auto";
        case PrefixCacheTerminalStateMode::Always:
            return "always";
        default:
            return "unknown";
        }
    }

    inline std::optional<PrefixCacheTerminalStateMode> parsePrefixCacheTerminalStateMode(const std::string &value)
    {
        const std::string normalized = normalizeRuntimeConfigToken(value);
        if (normalized == "off" || normalized == "disabled")
            return PrefixCacheTerminalStateMode::Off;
        if (normalized == "auto")
            return PrefixCacheTerminalStateMode::Auto;
        if (normalized == "always")
            return PrefixCacheTerminalStateMode::Always;
        return std::nullopt;
    }

    enum class PrefixCacheMoEPolicy
    {
        Disabled,
        PlacementFingerprint,
        InvalidateOnRebalance,
    };

    inline const char *prefixCacheMoEPolicyToString(PrefixCacheMoEPolicy policy)
    {
        switch (policy)
        {
        case PrefixCacheMoEPolicy::Disabled:
            return "disabled";
        case PrefixCacheMoEPolicy::PlacementFingerprint:
            return "placement-fingerprint";
        case PrefixCacheMoEPolicy::InvalidateOnRebalance:
            return "invalidate-on-rebalance";
        default:
            return "unknown";
        }
    }

    inline std::optional<PrefixCacheMoEPolicy> parsePrefixCacheMoEPolicy(const std::string &value)
    {
        const std::string normalized = normalizeRuntimeConfigToken(value);
        if (normalized == "disabled" || normalized == "off")
            return PrefixCacheMoEPolicy::Disabled;
        if (normalized == "placement_fingerprint" || normalized == "fingerprint")
            return PrefixCacheMoEPolicy::PlacementFingerprint;
        if (normalized == "invalidate_on_rebalance" || normalized == "invalidate")
            return PrefixCacheMoEPolicy::InvalidateOnRebalance;
        return std::nullopt;
    }

    /**
     * @brief Resolve the default durable prefix-cache directory.
     *
     * The path is expanded here instead of storing a literal `~`, because C++
     * filesystem APIs do not perform shell expansion. An empty result is a
     * deliberate hard signal that the process has no usable HOME directory;
     * callers enabling a disk budget can then report a configuration error.
     */
    inline std::string defaultPrefixCacheDiskDirectory()
    {
        const char *home = DebugEnv::envValue("HOME");
        if (!home || home[0] == '\0')
            return {};
        return (std::filesystem::path(home) / ".llaminar" / "kvcache").string();
    }

    /** Default bounded host-memory capacity for reusable prefix state. */
    inline constexpr size_t kDefaultPrefixCacheRamBudgetBytes =
        4ull * 1024ull * 1024ull * 1024ull;

    /** Default bounded accelerator-memory capacity for the hottest prefixes. */
    inline constexpr size_t kDefaultPrefixCacheDeviceBudgetBytes =
        256ull * 1024ull * 1024ull;

    /**
     * Default bounded durable capacity for prefixes evicted from RAM.
     *
     * Disk storage is sparse and demand-allocated; this value is a retention
     * ceiling, not startup allocation. Thirty-two GiB retains useful service
     * history without allowing an unconfigured server to consume a volume.
     */
    inline constexpr size_t kDefaultPrefixCacheDiskBudgetBytes =
        32ull * 1024ull * 1024ull * 1024ull;

    struct PrefixCacheRuntimeConfig
    {
        bool enabled = true;
        PrefixCacheStorageMode storage_mode = PrefixCacheStorageMode::Tiered;
        int block_size = 64;
        size_t ram_budget_bytes = kDefaultPrefixCacheRamBudgetBytes;
        size_t device_budget_bytes = kDefaultPrefixCacheDeviceBudgetBytes;
        size_t disk_budget_bytes = kDefaultPrefixCacheDiskBudgetBytes;
        std::string disk_dir = defaultPrefixCacheDiskDirectory();
        PrefixCacheTerminalStateMode terminal_state = PrefixCacheTerminalStateMode::Auto;
        PrefixCacheMoEPolicy moe_policy = PrefixCacheMoEPolicy::PlacementFingerprint;
    };

    enum class MTPVerifyMode
    {
        Greedy,
        SpeculativeSampling,
    };

    inline const char *mtpVerifyModeToString(MTPVerifyMode mode)
    {
        switch (mode)
        {
        case MTPVerifyMode::Greedy:
            return "greedy";
        case MTPVerifyMode::SpeculativeSampling:
            return "speculative-sampling";
        default:
            return "unknown";
        }
    }

    inline std::optional<MTPVerifyMode> parseMTPVerifyMode(const std::string &value)
    {
        const std::string normalized = normalizeRuntimeConfigToken(value);
        if (normalized == "greedy")
            return MTPVerifyMode::Greedy;
        if (normalized == "speculative_sampling" || normalized == "sampling")
            return MTPVerifyMode::SpeculativeSampling;
        return std::nullopt;
    }

    enum class MTPDepthPolicyMode
    {
        Fixed,
        Observe,
        Dynamic,
    };

    inline const char *mtpDepthPolicyModeToString(MTPDepthPolicyMode mode)
    {
        switch (mode)
        {
        case MTPDepthPolicyMode::Fixed:
            return "fixed";
        case MTPDepthPolicyMode::Observe:
            return "observe";
        case MTPDepthPolicyMode::Dynamic:
            return "dynamic";
        default:
            return "unknown";
        }
    }

    inline std::optional<MTPDepthPolicyMode> parseMTPDepthPolicyMode(const std::string &value)
    {
        const std::string normalized = normalizeRuntimeConfigToken(value);
        if (normalized == "fixed" || normalized == "off")
            return MTPDepthPolicyMode::Fixed;
        if (normalized == "observe" || normalized == "profile")
            return MTPDepthPolicyMode::Observe;
        if (normalized == "dynamic" || normalized == "adaptive")
            return MTPDepthPolicyMode::Dynamic;
        return std::nullopt;
    }

    /**
     * @brief Coarse execution backend used by the generated MTP depth policy.
     *
     * The online controller intentionally avoids clocks and backend-specific
     * performance probes.  The offline trainer can still learn that CUDA,
     * ROCm, and CPU prefer different speculative depths by keying generated
     * rules on this small backend class.
     */
    enum class MTPDepthPolicyBackend
    {
        Any,
        CPU,
        CUDA,
        ROCm,
    };

    /**
     * @brief Coarse model family used by the generated MTP depth policy.
     *
     * Dynamic-depth economics differ between dense and MoE graphs even when
     * token acceptance looks similar: MoE verifier cost, routed expert work,
     * and condition-forward replay can make a depth profitable or unprofitable
     * at different acceptance rates.  Keep this intentionally small so the
     * runtime remains deterministic and the offline trainer can learn separate
     * tables without depending on model-specific strings.
     */
    enum class MTPDepthPolicyModelClass
    {
        Any,
        Dense,
        MoE,
    };

    struct MTPDepthPolicyConfig
    {
        MTPDepthPolicyMode mode = MTPDepthPolicyMode::Fixed;
        MTPDepthPolicyBackend backend = MTPDepthPolicyBackend::Any;
        MTPDepthPolicyModelClass model_class = MTPDepthPolicyModelClass::Any;
        int min_depth = 1;
        int max_depth = 0;     ///< 0 derives from MTPRuntimeConfig::draft_tokens.
        int initial_depth = 0; ///< 0 derives from a policy-specific default.
        int window_size = 16;
        int min_samples = 4;
        int cooldown_steps = 8;
        int promote_consecutive_windows = 3;
        /**
         * @brief Use the offline-trained depth policy table for dynamic mode.
         *
         * The generated table is deterministic C++ produced by the benchmark
         * training pipeline. It can make earlier promote/demote decisions from
         * the same rolling-window counters as the handwritten fallback policy.
         * Fixed mode ignores this flag.
         */
        bool use_generated_policy = true;
        double promote_full_accept_rate = 1.0;
        double demote_zero_accept_rate = 0.30;
        /**
         * @brief Draft-token acceptance rate below which dynamic depth demotes.
         *
         * This threshold shrinks deeper speculative drafts toward depth 1.
         * Depth 0 is an explicit bypass mode and is entered only through the
         * zero-acceptance threshold. Keep the default conservative:
         * stochastic requests can have noisy short windows, and over-eager
         * demotion causes depth churn before the verifier path has enough
         * samples to prove that a lower depth is actually faster.
         */
        double demote_acceptance_rate = 0.55;
    };

    /**
     * @brief Resolve the effective initial MTP draft depth.
     *
     * Fixed mode pins to the configured fixed depth.  Greedy dynamic/observe
     * starts at depth 2 when available because recent dense lanes show that as
     * a cheap warm start below the risky deepest lane.  Stochastic
     * dynamic/observe starts at depth 1: rejection sampling cannot legally
     * produce ready logits after residual corrections, so a bad first window
     * at depth 2 has an outsized condition-forward tax.  An explicit
     * depth-zero bypass range still starts at zero so operators can force a
     * conservative adaptive warmup.
     */
    inline int resolveMTPDepthPolicyInitialDepth(
        const MTPDepthPolicyConfig &config,
        int configured_draft_tokens,
        MTPVerifyMode verify_mode = MTPVerifyMode::Greedy)
    {
        const int effective_max_depth =
            config.max_depth > 0 ? config.max_depth : configured_draft_tokens;
        if (config.initial_depth > 0)
            return config.initial_depth;
        if (config.mode == MTPDepthPolicyMode::Fixed)
            return configured_draft_tokens;
        if (config.min_depth == 0)
            return 0;
        if (verify_mode == MTPVerifyMode::SpeculativeSampling)
            return config.min_depth;
        return std::clamp(2, config.min_depth, effective_max_depth);
    }

    /**
     * @enum MTPTerminalHeadPolicy
     * @brief Physical placement of final normalization and vocabulary projection weights.
     *
     * This policy names only the small terminal surface used to turn verifier
     * hidden rows into logits. It does not replicate attention blocks, dense
     * FFNs, shared MoE experts, or routed experts. Keeping that distinction in
     * the type prevents a request for a mirrored MTP head from accidentally
     * selecting the much larger replicated-dense or replicated-expert modes.
     */
    enum class MTPTerminalHeadPolicy
    {
        /**
         * Keep the model's vocabulary-sharded final norm and LM-head layout.
         *
         * This is an explicit diagnostic/economy policy. It is never selected
         * implicitly from the TP scope when mirrored ownership was requested.
         */
        VocabularySharded,

        /**
         * Mirror the complete final norm and full-vocabulary LM head on every
         * TP participant so verifier sampling needs no tiny logits collective.
         * A participant may be an intra-rank device, a node-local MPI rank, or
         * a global MPI rank; scope does not alter the ownership policy.
         */
        MirroredFullVocabulary,
    };

    /**
     * @brief Return the canonical CLI/YAML spelling for an MTP head policy.
     * @param policy Typed terminal-head placement policy.
     * @return Stable lowercase configuration token.
     */
    inline const char *mtpTerminalHeadPolicyToString(
        MTPTerminalHeadPolicy policy)
    {
        switch (policy)
        {
        case MTPTerminalHeadPolicy::VocabularySharded:
            return "vocabulary-sharded";
        case MTPTerminalHeadPolicy::MirroredFullVocabulary:
            return "mirrored-full-vocabulary";
        }
        return "unknown";
    }

    /**
     * @brief Parse one canonical MTP terminal-head placement policy.
     * @param value CLI or YAML token.
     * @return Typed policy, or `std::nullopt` for an unknown spelling.
     */
    inline std::optional<MTPTerminalHeadPolicy> parseMTPTerminalHeadPolicy(
        const std::string &value)
    {
        const std::string normalized = normalizeRoutedExpertPolicyToken(value);
        if (normalized == "vocabulary-sharded")
            return MTPTerminalHeadPolicy::VocabularySharded;
        if (normalized == "mirrored-full-vocabulary")
            return MTPTerminalHeadPolicy::MirroredFullVocabulary;
        return std::nullopt;
    }

    /**
     * @brief Return whether a policy binds a complete terminal head locally.
     * @param policy Typed terminal-head placement policy.
     * @return True only for the full-vocabulary mirrored policy.
     */
    inline bool mtpTerminalHeadIsMirrored(
        MTPTerminalHeadPolicy policy) noexcept
    {
        return policy == MTPTerminalHeadPolicy::MirroredFullVocabulary;
    }

    /**
     * @enum MTPTerminalLogitsLayout
     * @brief Resolved vocabulary ownership of one participant's MTP logits.
     *
     * The configured terminal-head policy and the model's primary LM-head
     * sharding bit jointly determine the tensor written by an MTP projection.
     * Keeping that resolution in one typed value prevents graph builders,
     * orchestrators, and samplers from independently guessing whether
     * `MTP_LOGITS` contains a complete distribution or only one vocabulary
     * shard.
     */
    enum class MTPTerminalLogitsLayout
    {
        /** Every participant writes the complete vocabulary locally. */
        FullVocabularyPerParticipant,

        /** Every participant writes only its assigned vocabulary columns. */
        VocabularyShardPerParticipant,
    };

    /**
     * @brief Resolve the physical MTP logits layout from declarative policy.
     * @param primary_lm_head_column_parallel Whether the model's primary
     *        terminal projection is vocabulary-column sharded.
     * @param policy Explicit MTP terminal-head ownership policy.
     * @return The exact tensor ownership produced by every participant.
     *
     * A model without a column-parallel primary head already owns a complete
     * vocabulary projection, irrespective of the requested MTP policy. When
     * the primary head is column parallel, only the explicit mirrored policy
     * authorizes use of the replicated full-vocabulary weight binding.
     */
    inline MTPTerminalLogitsLayout resolveMTPTerminalLogitsLayout(
        bool primary_lm_head_column_parallel,
        MTPTerminalHeadPolicy policy) noexcept
    {
        return primary_lm_head_column_parallel &&
                       !mtpTerminalHeadIsMirrored(policy)
                   ? MTPTerminalLogitsLayout::VocabularyShardPerParticipant
                   : MTPTerminalLogitsLayout::FullVocabularyPerParticipant;
    }

    /**
     * @enum MTPTerminalLogitsCollective
     * @brief Collective, if any, owned by an MTP terminal projection.
     */
    enum class MTPTerminalLogitsCollective
    {
        /** Participant output is already complete or the graph emits no logits. */
        None,

        /** A multi-rank GlobalTP domain must assemble vocabulary shards. */
        GlobalVocabularyAllGather,
    };

    /**
     * @struct MTPTerminalLogitsCollectiveRequest
     * @brief Complete declarative input to MTP terminal collective planning.
     */
    struct MTPTerminalLogitsCollectiveRequest
    {
        MTPTerminalLogitsLayout layout =
            MTPTerminalLogitsLayout::FullVocabularyPerParticipant;
        bool sidecar_produces_logits = false;
        bool spans_multiple_global_ranks = false;
    };

    /**
     * @brief Resolve the terminal logits collective from complete typed policy.
     * @param request Output ownership and execution-topology facts.
     * @return `GlobalVocabularyAllGather` only when every prerequisite is true.
     *
     * KV-only shifted-prefill graphs set `sidecar_produces_logits=false` and
     * therefore never acquire a dummy terminal collective. Mirrored heads set
     * a full-vocabulary participant layout and likewise resolve to `None` for
     * local, node-local, and global TP.
     */
    inline MTPTerminalLogitsCollective resolveMTPTerminalLogitsCollective(
        const MTPTerminalLogitsCollectiveRequest &request) noexcept
    {
        return request.sidecar_produces_logits &&
                       request.spans_multiple_global_ranks &&
                       request.layout ==
                           MTPTerminalLogitsLayout::VocabularyShardPerParticipant
                   ? MTPTerminalLogitsCollective::GlobalVocabularyAllGather
                   : MTPTerminalLogitsCollective::None;
    }

    /**
     * @struct MTPRuntimeConfig
     * @brief Runtime and graph-layout policy for speculative MTP execution.
     */
    struct MTPRuntimeConfig
    {
        bool enabled = false;
        int draft_tokens = 1;
        /**
         * @brief Retained graph/arena draft capacity, or zero to derive it.
         *
         * This is deliberately independent of the selected execution depth.
         * A service may retain one maximum-capacity graph family while fixed
         * requests execute smaller depths, avoiding weight-placement changes
         * and graph recapture when only the requested speculative width
         * changes.  A positive value must cover the fixed depth or adaptive
         * policy ceiling; it never authorizes the depth controller to select
         * additional drafts.
         */
        int graph_capacity_draft_tokens = 0;
        /**
         * @brief Maximum number of requests to amortize in one speculative transaction.
         *
         * vLLM-style production MTP batches target verification rows across
         * requests so tiny per-request verifier/condition forwards do not
         * dominate MoE decode. Values greater than one are a real runner
         * capacity request: planning must size request-local state and graph
         * buffers for at least this many active verifier rows. Live decode may
         * still hard-fail unsupported batched transaction paths until Phase 8
         * wires the corresponding scheduler execution.
         */
        int max_request_batch = 1;
        MTPVerifyMode verify_mode = MTPVerifyMode::Greedy;
        /**
         * @brief Placement of the verifier's final norm and LM-head weights.
         *
         * Tensor-parallel MTP defaults to a complete mirrored terminal head
         * because a tiny per-draft vocabulary collective is generally less
         * economical than duplicating this terminal projection. The policy has
         * identical meaning for local, node-local, and global TP scopes.
         */
        MTPTerminalHeadPolicy terminal_head_policy =
            MTPTerminalHeadPolicy::MirroredFullVocabulary;
        bool require_terminal_hidden_for_full_hit = true;
        MTPDepthPolicyConfig depth_policy;
    };

    /**
     * @brief Resolve the largest depth the execution policy may select.
     *
     * Fixed mode selects exactly @ref MTPRuntimeConfig::draft_tokens. Dynamic
     * and observe modes may select up to the adaptive policy ceiling. This is
     * an execution-policy bound and deliberately ignores retained graph
     * over-capacity.
     */
    inline int resolveMTPMaximumExecutionDraftDepth(
        const MTPRuntimeConfig &config)
    {
        if (config.depth_policy.mode == MTPDepthPolicyMode::Fixed)
            return std::max(1, config.draft_tokens);
        return std::max(
            1,
            config.depth_policy.max_depth > 0
                ? config.depth_policy.max_depth
                : config.draft_tokens);
    }

    /**
     * @brief Resolve the largest draft depth that runtime planning must own.
     *
     * By default the retained capacity equals the execution-policy ceiling.
     * An explicit @ref MTPRuntimeConfig::graph_capacity_draft_tokens may make
     * the graph, arena, transaction slots, and prepared-weight admission
     * envelope wider without changing the depth selected for execution. The
     * configuration validator rejects an explicit value narrower than the
     * execution-policy ceiling.
     */
    inline int resolveMTPMaximumDraftDepth(const MTPRuntimeConfig &config)
    {
        const int execution_maximum =
            resolveMTPMaximumExecutionDraftDepth(config);
        return config.graph_capacity_draft_tokens > 0
                   ? std::max(execution_maximum,
                              config.graph_capacity_draft_tokens)
                   : execution_maximum;
    }

    /**
     * @brief Resolve compact target-verifier row capacity for MTP graph buffers.
     *
     * A single request needs `maximum draft depth + 1` target rows: one row per
     * draft comparison plus the bonus row. Request-batched MTP flattens those
     * per-request rows into one compact LM-head input tensor, so capacity also
     * scales with `max_request_batch`. The default sixteen-row certification
     * is not a hard limit; larger configured policies produce larger graphs.
     */
    inline int resolveMTPMaxTargetQueryRows(const MTPRuntimeConfig &config)
    {
        const int request_count = std::max(1, config.max_request_batch);
        const int draft_count = resolveMTPMaximumDraftDepth(config);
        return request_count * (draft_count + 1);
    }

    /**
     * @brief Resolve one retained graph's complete activation-row capacity.
     *
     * Captured prefill and MTP verification reuse one resident graph family.
     * The prefill bucket therefore cannot be treated as the whole graph's row
     * capacity: verification may need `maximum draft depth + 1` rows even when
     * the selected prefill segment is smaller. This helper is the single
     * accounting rule shared by admission and preflight. It does not change
     * the selected prefill segment; it only sizes the graph that owns both
     * shapes.
     *
     * @param prefill_rows Positive row capacity selected for captured prefill.
     * @param config Runtime MTP policy and maximum request batch.
     * @return Rows the shared retained graph must materialize.
     */
    inline int resolveRetainedGraphRowCapacity(
        int prefill_rows,
        const MTPRuntimeConfig &config)
    {
        const int base_rows = std::max(1, prefill_rows);
        if (!config.enabled)
            return base_rows;
        return std::max(base_rows, resolveMTPMaxTargetQueryRows(config));
    }

    /**
     * @brief Resolve the runner batch capacity required by MTP request batching.
     *
     * `batch_size` is the general runner capacity knob. `max_request_batch` is
     * the MTP-specific request batching knob. When MTP is enabled, both knobs
     * describe real capacity that must exist before speculative verification
     * can publish per-request KV/GDN/hidden state without racing or
     * over-indexing runner-owned buffers.
     */
    inline int resolveRuntimeBatchSizeForMTP(int configured_batch_size, const MTPRuntimeConfig &config)
    {
        const int base_batch_size = std::max(1, configured_batch_size);
        if (!config.enabled)
            return base_batch_size;
        return std::max(base_batch_size, std::max(1, config.max_request_batch));
    }

    /**
     * @brief Resolve the persistent terminal-hidden archive row capacity.
     *
     * Main prefill archives one terminal row for every admitted request, while
     * grouped verification and accepted-state publication may temporarily
     * archive every flattened target row. The one graph-stable GPU owner must
     * therefore cover both independently configurable dimensions.
     *
     * @param configured_batch_size General runner request capacity.
     * @param config MTP depth and request-batch policy.
     * @return Number of FP32 hidden rows required by the shared archive.
     */
    inline int resolveMTPTerminalHiddenRowCapacity(
        int configured_batch_size,
        const MTPRuntimeConfig &config)
    {
        return std::max(
            resolveRuntimeBatchSizeForMTP(configured_batch_size, config),
            resolveMTPMaxTargetQueryRows(config));
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
     * @brief Physical distribution of dense and shared-always-on model work.
     *
     * Routed-expert distribution is deliberately absent from this enum.  It is
     * described independently by RoutedExpertComputePolicy, preventing a dense
     * policy from silently implying apportioned or tensor-sharded experts.
     */
    enum class DenseParallelPolicy
    {
        /// Dense/shared weights are fully present on each participant.
        Replicated,

        /// Dense/shared weights are tensor-parallel sharded, with collectives
        /// emitted where sharded partials must be combined.
        TensorParallel,

        /// Dense/shared weights remain tensor-parallel, except decode uses a
        /// mirrored full embedding table to avoid a tiny per-token allreduce.
        TensorParallelDecodeMirroredEmbedding,

        /// Prefill uses dense tensor parallelism. Decode uses replicated
        /// dense/shared weights to avoid small per-token dense collectives.
        PrefillTensorParallelDecodeReplicated
    };

    /**
     * @brief Explicit composite of the five independent MoE execution axes.
     *
     * This is a value object rather than a combinatorial enum.  Callers can
     * inspect each policy directly, so adding a dense policy cannot accidentally
     * invent another ambiguous composite abbreviation.
     */
    struct MoEExecutionPolicy
    {
        ///< Distribution of dense and shared-always-on model work.
        DenseParallelPolicy dense = DenseParallelPolicy::Replicated;

        ///< Physical distribution of routed-expert weights and GEMMs.
        RoutedExpertComputePolicy routed_compute =
            RoutedExpertComputePolicy::Apportioned;

        ///< Phase-specific scheduling over physically resident routed experts.
        RoutedExpertPhasePolicy routed_phase =
            RoutedExpertPhasePolicy::Uniform;

        ///< Decode/grouped-verifier scheduling among complete residents.
        RoutedExpertAssignmentPolicy routed_decode_assignment =
            RoutedExpertAssignmentPolicy::StaticOwner;

        ///< Ordinary prefill scheduling among complete residents.
        RoutedExpertAssignmentPolicy routed_prefill_assignment =
            RoutedExpertAssignmentPolicy::StaticOwner;

        /** @brief Compare all five independent policy axes. */
        bool operator==(const MoEExecutionPolicy &other) const
        {
            return dense == other.dense &&
                   routed_compute == other.routed_compute &&
                   routed_phase == other.routed_phase &&
                   routed_decode_assignment == other.routed_decode_assignment &&
                   routed_prefill_assignment == other.routed_prefill_assignment;
        }

        /** @brief Return true when any independent policy axis differs. */
        bool operator!=(const MoEExecutionPolicy &other) const
        {
            return !(*this == other);
        }
    };

    enum class ExpertReplicaPolicy
    {
        None,
        HotExpertReplicaCache
    };

    inline std::string normalizeParallelPolicyToken(const std::string &value)
    {
        std::string lower = value;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        std::replace(lower.begin(), lower.end(), '_', '-');
        return lower;
    }

    inline const char *denseParallelPolicyToString(DenseParallelPolicy policy)
    {
        switch (policy)
        {
        case DenseParallelPolicy::Replicated:
            return "replicated";
        case DenseParallelPolicy::TensorParallel:
            return "tensor-parallel";
        case DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding:
            return "tensor-parallel-decode-mirrored-embedding";
        case DenseParallelPolicy::PrefillTensorParallelDecodeReplicated:
            return "prefill-tensor-parallel-decode-replicated";
        default:
            return "unknown";
        }
    }

    /**
     * @brief Parse one canonical dense/shared-model distribution policy.
     * @param value CLI or YAML token to parse.
     * @return Typed dense policy, or `std::nullopt` for an obsolete alias or
     *         otherwise unknown value.
     *
     * Only the spellings emitted by `denseParallelPolicyToString()` are
     * accepted. In particular, abbreviations such as `tp` and `full` are not
     * retained as compatibility paths because they conceal which model phase
     * or weight family is being distributed.
     */
    inline std::optional<DenseParallelPolicy> parseDenseParallelPolicy(
        const std::string &value)
    {
        const std::string lower = normalizeParallelPolicyToken(value);
        if (lower == "replicated")
            return DenseParallelPolicy::Replicated;
        if (lower == "tensor-parallel")
            return DenseParallelPolicy::TensorParallel;
        if (lower == "tensor-parallel-decode-mirrored-embedding")
            return DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding;
        if (lower == "prefill-tensor-parallel-decode-replicated")
            return DenseParallelPolicy::PrefillTensorParallelDecodeReplicated;
        return std::nullopt;
    }

    inline DenseParallelPolicy denseParallelPolicyFromFlags(
        bool dense_tp_enabled,
        bool dense_decode_replicated,
        bool dense_decode_mirrored_embedding = false)
    {
        if (!dense_tp_enabled)
            return DenseParallelPolicy::Replicated;
        if (dense_decode_replicated)
            return DenseParallelPolicy::PrefillTensorParallelDecodeReplicated;
        if (dense_decode_mirrored_embedding)
            return DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding;
        return DenseParallelPolicy::TensorParallel;
    }

    inline bool denseParallelPolicyEnablesTP(DenseParallelPolicy policy)
    {
        return policy == DenseParallelPolicy::TensorParallel ||
               policy == DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding ||
               policy == DenseParallelPolicy::PrefillTensorParallelDecodeReplicated;
    }

    inline bool denseParallelPolicyReplicatesDecode(DenseParallelPolicy policy)
    {
        return policy == DenseParallelPolicy::PrefillTensorParallelDecodeReplicated;
    }

    inline bool denseParallelPolicyMirrorsDecodeEmbedding(DenseParallelPolicy policy)
    {
        return policy == DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding ||
               policy == DenseParallelPolicy::PrefillTensorParallelDecodeReplicated;
    }

    /**
     * @brief Build an explicit MoE execution policy without deriving aliases.
     * @param dense_policy Dense/shared-model distribution policy.
     * @param routed_compute Routed-expert weight and GEMM distribution policy.
     * @param routed_decode_assignment Decode/grouped-verifier row scheduling.
     * @param routed_prefill_assignment Ordinary-prefill row scheduling.
     * @param routed_phase Phase-specific execution over resident weights.
     * @return Value object containing every argument unchanged.
     */
    inline MoEExecutionPolicy makeMoEExecutionPolicy(
        DenseParallelPolicy dense_policy,
        RoutedExpertComputePolicy routed_compute,
        RoutedExpertAssignmentPolicy routed_decode_assignment =
            RoutedExpertAssignmentPolicy::StaticOwner,
        RoutedExpertAssignmentPolicy routed_prefill_assignment =
            RoutedExpertAssignmentPolicy::StaticOwner,
        RoutedExpertPhasePolicy routed_phase =
            RoutedExpertPhasePolicy::Uniform)
    {
        return MoEExecutionPolicy{
            .dense = dense_policy,
            .routed_compute = routed_compute,
            .routed_phase = routed_phase,
            .routed_decode_assignment = routed_decode_assignment,
            .routed_prefill_assignment = routed_prefill_assignment,
        };
    }

    /**
     * @brief Render every MoE execution axis for logs and diagnostics.
     * @param policy Explicit policy value object to describe.
     * @return Comma-separated canonical key/value pairs for all five axes.
     */
    inline std::string describeMoEExecutionPolicy(const MoEExecutionPolicy &policy)
    {
        std::ostringstream out;
        out << "dense=" << denseParallelPolicyToString(policy.dense)
            << ",routed_compute="
            << routedExpertComputePolicyToString(policy.routed_compute)
            << ",routed_phase="
            << routedExpertPhasePolicyToString(policy.routed_phase)
            << ",routed_decode_assignment="
            << routedExpertAssignmentPolicyToString(
                   policy.routed_decode_assignment)
            << ",routed_prefill_assignment="
            << routedExpertAssignmentPolicyToString(
                   policy.routed_prefill_assignment);
        return out.str();
    }

    inline const char *expertReplicaPolicyToString(ExpertReplicaPolicy policy)
    {
        switch (policy)
        {
        case ExpertReplicaPolicy::None:
            return "none";
        case ExpertReplicaPolicy::HotExpertReplicaCache:
            return "hot-expert-replica-cache";
        default:
            return "unknown";
        }
    }

    inline std::optional<ExpertReplicaPolicy> parseExpertReplicaPolicy(const std::string &value)
    {
        const std::string lower = normalizeParallelPolicyToken(value);
        if (lower == "none" || lower == "off" || lower == "disabled")
            return ExpertReplicaPolicy::None;
        if (lower == "hot-expert-replica-cache" ||
            lower == "hot-expert-cache" ||
            lower == "hot-replica-cache" ||
            lower == "hotexpertreplicacache")
            return ExpertReplicaPolicy::HotExpertReplicaCache;
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

    inline ExpertReplicaPolicy expertReplicaPolicyFromHotExpertCache(
        const MoEHotExpertCacheConfig &config)
    {
        return config.enabled()
                   ? ExpertReplicaPolicy::HotExpertReplicaCache
                   : ExpertReplicaPolicy::None;
    }

    /**
     * @brief Runtime policy for ordinary routed-expert prefill assignment.
     *
     * This policy is deliberately independent of durable expert-residency
     * maintenance. Paper-style LLEP assigns the current batch; Dynamic
     * maintenance alters future residency. Neither policy may implicitly
     * enable, disable, or configure the other.
     */
    struct RoutedExpertPrefillRuntimeConfig
    {
        /**
         * @brief Stable token window used by current-batch assignment.
         *
         * Zero keeps one ordinary prefill transaction unless prefix cache
         * supplies a block boundary. Positive values split prefill into fixed
         * graph-stable windows of this many real tokens.
         */
        int assignment_window_tokens = 0;

        /**
         * @brief Maximum live token rows in one ExpertOverlay prefill segment.
         *
         * Heterogeneous overlay endpoints retain captured compact-route tensor
         * families.  Bounding a segment prevents a short or moderately sized
         * prompt from selecting the full-context family merely because the KV
         * cache admits a long context.  The continuation authority publishes
         * the resolved value to every rank before graph construction; longer
         * prompts run as ordered captured segments.  This transport/capture
         * boundary is independent of current-batch LLEP assignment windows.
         * The default is derived from the canonical bucket inventory and
         * retained-topology budget so setup leaves capacity for live request
         * and prefix-runtime graph identities.
         */
        int overlay_segment_rows =
            kDefaultExpertOverlayPrefillSegmentRows;

        /**
         * @brief Minimum routed rows required for least-loaded prefill.
         *
         * Ordinary prefill contributes `M * top_k` routed rows. Below this
         * explicit economy boundary graph lowering selects canonical
         * StaticOwner expert parallelism; at or above it a requested
         * LeastLoadedResident policy is always full, graph-captured,
         * transfer-backed current-batch LLEP. Zero forces LLEP for every
         * ordinary prefill shape and is useful for focused integration tests.
         */
        uint64_t least_loaded_min_routed_rows = 8192;

        /**
         * @brief Numerator of the current-batch LLEP capacity multiplier.
         *
         * Together with `llep_alpha_denominator`, this bounds the routed rows
         * assigned to one participant relative to the balanced batch load.
         * Both terms must be positive so the planner has one defined capacity.
         */
        uint32_t llep_alpha_numerator = 1;
        /**
         * @brief Denominator of the current-batch LLEP capacity multiplier.
         *
         * The ratio is represented as integers to keep CUDA and ROCm planning
         * decisions exact and independent of floating-point contraction.
         */
        uint32_t llep_alpha_denominator = 1;
        /**
         * @brief Numerator of the balanced-static-owner skip threshold.
         *
         * This ratio defines when the busiest static-owner participant is
         * already close enough to the mean participant load that transport
         * cannot repay its cost for the current batch. It deliberately does
         * not compare individual expert popularity: LLEP changes participant
         * execution load, not the router's expert-frequency distribution.
         */
        uint32_t llep_lambda_numerator = 13;
        /**
         * @brief Denominator of the balanced-static-owner skip threshold.
         *
         * Both lambda terms must be positive. Integer comparison preserves the
         * same branch decision on CPU, CUDA, and ROCm.
         */
        uint32_t llep_lambda_denominator = 10;
        /**
         * @brief Permit standard static-owner EP when current-batch loads are balanced.
         *
         * Focused movement tests may disable this to require a non-owner span,
         * but production keeps the economical no-movement decision available.
         */
        bool llep_enable_balanced_skip = true;
    };

    /**
     * @brief Durable routed-expert residency maintenance mode.
     *
     * Current-batch least-loaded prefill (LLEP) is deliberately absent: it is
     * selected by `RoutedExpertAssignmentPolicy` on the routed domain and does
     * not imply persistent ownership mutation.
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

    /**
     * @brief Durable ownership observation and migration configuration.
     *
     * These settings control cross-window expert residency maintenance. They
     * never choose decode/grouped row assignment or current-batch prefill LLEP;
     * those are independent typed domain policies.
     */
    struct MoERebalanceRuntimeConfig
    {
        MoERebalanceRuntimeMode mode = MoERebalanceRuntimeMode::Dynamic;
        int window_size = 256;
        int max_window_size = 4096;
        float window_growth_factor = 1.5f;
        /**
         * @brief Expected routed-token lifetime available to repay one move.
         *
         * This is deliberately independent of histogram cadence. A deployment
         * with stable traffic may amortize a measured transfer over many short
         * observation windows, while a rapidly changing workload can choose a
         * smaller lifetime without changing when maintenance is polled.
         */
        uint64_t migration_payoff_horizon_tokens =
            moe_rebalance_policy::kDefaultMigrationPayoffHorizonTokens;
        /**
         * @brief Maximum closed residency cycles staged in one async wave.
         *
         * One is the conservative production default. Larger values allow the
         * planner to fill independent endpoint/layer shadow capacity in one
         * publication epoch; the physical capacity and staging BOM remain
         * hard upper bounds and may admit fewer cycles.
         */
        uint32_t migration_max_cycles_per_wave = 1;
        uint32_t dynamic_imbalance_threshold_per_mille =
            moe_rebalance_policy::kDefaultDynamicImbalanceThresholdPerMille;
        uint32_t dynamic_min_improvement_per_mille =
            moe_rebalance_policy::kDefaultDynamicMinImprovementPerMille;
        uint32_t dynamic_max_swaps_per_layer =
            moe_rebalance_policy::kDefaultDynamicMaxSwapsPerLayer;
        uint32_t dynamic_max_plan_entries_per_wave =
            moe_rebalance_policy::kDefaultDynamicMaxPlanEntriesPerWave;
        uint64_t dynamic_min_window_activations =
            moe_rebalance_policy::kDefaultDynamicMinWindowActivations;
        uint32_t device_min_load_spread_improvement = 0;
        uint32_t device_min_load_spread_improvement_divisor =
            moe_rebalance_policy::kDefaultDeviceMinLoadSpreadImprovementDivisor;
        uint32_t device_min_wave_spread_improvement_per_payload_slot = 256;
        uint32_t device_min_foreign_rows_per_critical_path_payload_slot = 0;
        uint32_t device_min_router_spread_improvement_per_payload_slot = 128;
        uint32_t device_max_post_wave_load_spread_per_mille = 100;
        /**
         * @brief Tokens added to a full routing window before maintenance is due.
         *
         * `-1` delegates to the process-wide diagnostic default.  Any
         * non-negative CLI/YAML value is graph policy and therefore takes
         * precedence over environment diagnostics.
         */
        int device_maintenance_slack_tokens = -1;
        /**
         * @brief Lower bound for recurring device-owned maintenance cadence.
         *
         * Zero disables the additional floor, leaving `window_size + slack`
         * as the recurring period. `-1` delegates to the diagnostic default.
         */
        int device_min_maintenance_period_tokens = -1;
        /**
         * @brief Token period for the first device-owned maintenance decision.
         *
         * Zero selects the recurring period. `-1` delegates to the diagnostic
         * default. A benchmark that claims to exercise Dynamic residency
         * maintenance must set this explicitly so a short decode cannot
         * silently miss maintenance.
         */
        int device_initial_maintenance_period_tokens = -1;
        bool release_raw_expert_weights = false;
    };

    /**
     * @brief Immutable prefill schedule shared by an ExpertOverlay world.
     *
     * A heterogeneous routed-expert request crosses every participant in one
     * ordered sparse-collective protocol.  Captured prefill chunks therefore
     * need one globally agreed physical-bucket ladder and one maximum logical
     * row count.  The continuation root publishes the bucket ladder during
     * initialization and, when distributed, all ranks contribute their
     * planner-admitted local capacity; @ref graph_row_capacity is their
     * minimum.  This makes it
     * impossible for the root to launch a segment a remote expert endpoint
     * cannot hold in its immutable compact-route arena.
     *
     * An empty contract denotes non-overlay execution. It is immutable after
     * runner construction and is request control-plane state, never a hot-path
     * allocation policy.
     */
    struct OverlayPrefillScheduleContract
    {
        /** @brief Common live-row segment capacity across all participants. */
        int graph_row_capacity = 0;
        /** @brief Root-authoritative fixed capture buckets, sorted and unique. */
        std::vector<int> bucket_rows;

        /** @brief Return true only for a complete executable overlay contract. */
        bool enabled() const noexcept
        {
            return graph_row_capacity > 0 && !bucket_rows.empty();
        }
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

        /**
         * @brief Maximum rows owned by one resident forward graph family.
         *
         * Zero means memory planning has not selected the value yet. The
         * selected positive value sizes activation buffers, serially shared
         * kernel workspace, and captured prefill buckets. It never reduces KV
         * context capacity; longer prompts execute as ordered graph chunks.
         */
        int resident_graph_rows = 0;

        /**
         * @brief ExpertOverlay prefill authority, when one is required.
         *
         * Local graph sizing remains in @ref resident_graph_rows.  This
         * contract constrains request segmentation to a shape every sparse
         * endpoint can execute, whether participants are process-local or
         * distributed.
         */
        OverlayPrefillScheduleContract overlay_prefill_schedule;

        /// Maximum active request batch size for runner-owned state.
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

        /// Optional explicit transport precision for TP allreduces.
        std::string tp_allreduce_precision_override;

        /// Routed MoE expert execution mode.
        RoutedExpertComputePolicy routed_expert_compute_policy = RoutedExpertComputePolicy::Apportioned;

        /// Static whole-expert ownership ordering for apportioned execution.
        RoutedExpertOwnerOrder routed_expert_owner_order =
            RoutedExpertOwnerOrder::Ordinal;

        /// Bounded remote-expert cache for dynamic routed-row assignment.
        MoEHotExpertCacheConfig moe_hot_expert_cache;

        /// Ordinary prefill assignment economy and graph-window policy.
        RoutedExpertPrefillRuntimeConfig moe_routed_prefill;

        /// MoE rebalance runtime configuration.
        MoERebalanceRuntimeConfig moe_rebalance;

        /// Cross-request prefix-state cache configuration.
        PrefixCacheRuntimeConfig prefix_cache;

        /// Multi-token prediction speculative decode configuration.
        MTPRuntimeConfig mtp;

        RuntimeConfig() = default;

        explicit RuntimeConfig(int max_seq_len_) : max_seq_len(max_seq_len_) {}

        /**
         * @brief Create RuntimeConfig by parsing raw strings from OrchestrationConfig
         */
        static RuntimeConfig fromOrchestrationConfig(
            int max_seq_len,
            int batch_size,
            const std::string &activation_precision_str,
            const std::string &kv_cache_precision_str,
            FusedAttentionBackend fused_backend = FusedAttentionBackend::JIT,
            RoutedExpertComputePolicy routed_expert_compute_policy = RoutedExpertComputePolicy::Apportioned,
            MoEHotExpertCacheConfig moe_hot_expert_cache = {},
            RoutedExpertPrefillRuntimeConfig moe_routed_prefill = {},
            MoERebalanceRuntimeConfig moe_rebalance = {},
            PrefixCacheRuntimeConfig prefix_cache = {},
            MTPRuntimeConfig mtp = {},
            std::string tp_allreduce_precision_override = {})
        {
            RuntimeConfig rc;
            rc.max_seq_len = max_seq_len;
            rc.batch_size = resolveRuntimeBatchSizeForMTP(batch_size, mtp);
            rc.activation_precision = parseActivationPrecision(activation_precision_str);
            rc.kv_cache_precision = parseKVCachePrecision(kv_cache_precision_str);
            rc.tp_allreduce_precision_override = std::move(tp_allreduce_precision_override);
            rc.fused_attention_backend = fused_backend;
            rc.routed_expert_compute_policy = routed_expert_compute_policy;
            rc.moe_hot_expert_cache = moe_hot_expert_cache;
            rc.moe_routed_prefill = moe_routed_prefill;
            rc.moe_rebalance = moe_rebalance;
            rc.prefix_cache = prefix_cache;
            rc.mtp = mtp;
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
