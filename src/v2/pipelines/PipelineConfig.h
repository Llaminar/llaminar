/**
 * @file PipelineConfig.h
 * @brief Runtime configuration for pipeline initialization
 * @author David Sanftenberg
 * @date 2025-10-25
 *
 * Encapsulates runtime parameters that affect pipeline behavior but are not
 * part of the model architecture (which comes from GGUF metadata).
 */

#pragma once

#include "backends/ComputeBackend.h"
#include "utils/CPUFeatures.h"
#include "utils/Logger.h"

namespace llaminar2
{

    /**
     * @brief Compute precision mode for weights and activations
     *
     * Determines how weights are loaded and what precision is used for computation.
     *
     * MIXED (default): Keep weights in original quantized format, compute in FP32
     *   - Memory efficient (weights stay compressed)
     *   - Dequantization happens on-the-fly in GEMM kernels
     *   - Best balance of speed and accuracy for most use cases
     *
     * FP32/BF16/FP16/INT8: Dequantize ALL weights to target format at load time
     *   - Higher memory usage (weights decompressed)
     *   - Potential compute benefits from specialized kernels
     *   - Use when you have specific hardware acceleration (e.g., BF16 on Sapphire Rapids)
     */
    enum class ComputePrecision
    {
        MIXED, ///< Keep weights quantized, compute in FP32 (default, memory efficient)
        FP32,  ///< Dequantize all weights to FP32 at load (highest accuracy, high memory)
        BF16,  ///< Dequantize all weights to BF16 at load (Intel Sapphire Rapids+)
        FP16,  ///< Dequantize all weights to FP16 at load (ARM/mobile optimization)
        INT8,  ///< Dequantize all weights to INT8 at load (AVX512-VNNI, CUDA Tensor Cores)
        AUTO   ///< Automatic selection based on hardware capabilities
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
     * - PipelineConfig: Runtime behavior settings (this struct)
     *
     * This separation follows V2's philosophy of clear ownership and composability.
     */
    struct PipelineConfig
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
         * @brief Compute precision for activations and weight handling
         *
         * Determines precision mode for model execution:
         * - MIXED: Keep weights in original format, compute in FP32 (default, memory-efficient)
         * - FP32: Dequantize all weights to FP32 at load time (highest accuracy, high memory)
         * - BF16: Reduced memory bandwidth, 1.5-2× faster on Ice Lake+ CPUs (not yet implemented)
         * - FP16: Reduced memory bandwidth, faster on ARM/mobile hardware (not yet implemented)
         * - INT8: Dequantize to INT8 for AVX512-VNNI/CUDA acceleration (not yet implemented)
         * - AUTO: Select based on hardware (MIXED for CPU, optimized for GPU)
         *
         * MIXED mode (default):
         * - Weights stay in original quantized format (IQ4_NL, Q6_K, etc.)
         * - Dequantization happens on-the-fly in GEMM kernels
         * - Compute happens in FP32 for accuracy
         * - Lowest memory footprint, good performance
         *
         * INT8 mode (future):
         * - Dequantizes all weights to INT8 at model load time
         * - Enables AVX512-VNNI (CPU) and CUTLASS INT8 GEMM (CUDA)
         * - Requires INT8 GEMM kernel implementation
         *
         * Note: Lower precision modes reduce memory bandwidth but may introduce error.
         * Attention is typically robust to BF16/FP16, but operations like softmax/RMSNorm
         * may require FP32 for numerical stability.
         */
        ComputePrecision precision = ComputePrecision::MIXED;

        /**
         * @brief Default constructor with standard settings
         */
        PipelineConfig() = default;

        /**
         * @brief Construct with explicit max_seq_len (common case)
         */
        explicit PipelineConfig(int max_seq_len_) : max_seq_len(max_seq_len_) {}
    };

    /**
     * @brief Auto-select optimal compute precision for a device
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
     * @return Recommended precision mode
     */
    inline ComputePrecision selectOptimalPrecision(const ComputeDevice &device)
    {
        switch (device.type)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
        {
            // Priority 1: AMX-BF16 (Intel Sapphire Rapids+, 4th gen Xeon)
            if (cpu_supports_amx_bf16())
            {
                LOG_INFO("AUTO precision: Detected AMX-BF16 → selecting BF16");
                LOG_INFO("  Expected: 50% memory bandwidth, 1.5-2× throughput vs FP32");
                return ComputePrecision::BF16;
            }

            // Priority 2: AVX512-BF16 (Intel Cooper Lake+, 3rd gen Xeon)
            if (cpu_supports_avx512_bf16())
            {
                LOG_INFO("AUTO precision: Detected AVX512-BF16 → selecting BF16");
                LOG_INFO("  Expected: 50% memory bandwidth, 1.3-1.8× throughput vs FP32");
                return ComputePrecision::BF16;
            }

            // Priority 3: AVX512-FP16 (Intel Sapphire Rapids+, but no BF16?)
            if (cpu_supports_avx512_fp16())
            {
                LOG_INFO("AUTO precision: Detected AVX512-FP16 → selecting FP16");
                LOG_INFO("  Expected: 50% memory bandwidth, 1.2-1.6× throughput vs FP32");
                return ComputePrecision::FP16;
            }

            // Fallback: FP32 (universal)
            LOG_INFO("AUTO precision: No FP16/BF16 acceleration detected → selecting FP32");
            LOG_INFO("  CPU: " << cpu_vendor());
            LOG_INFO("  AVX512: " << (cpu_supports_avx512() ? "yes" : "no"));
            LOG_INFO("  AVX2: " << (cpu_supports_avx2() ? "yes" : "no"));
            return ComputePrecision::FP32;
        }

        case ComputeBackendType::GPU_CUDA:
        {
            // CUDA devices: Check hardware capabilities
            if (device.supports_bf16)
            {
                LOG_INFO("AUTO precision: CUDA device supports BF16 → selecting BF16");
                LOG_INFO("  Device: " << device.name);
                LOG_INFO("  Expected: 50% memory bandwidth, tensor core acceleration");
                return ComputePrecision::BF16;
            }
            else if (device.supports_fp16)
            {
                LOG_INFO("AUTO precision: CUDA device supports FP16 → selecting FP16");
                LOG_INFO("  Device: " << device.name);
                LOG_INFO("  Expected: 50% memory bandwidth, tensor core acceleration");
                return ComputePrecision::FP16;
            }
            else
            {
                LOG_INFO("AUTO precision: CUDA device, no FP16/BF16 → selecting FP32");
                LOG_INFO("  Device: " << device.name);
                return ComputePrecision::FP32;
            }
        }

        case ComputeBackendType::GPU_ROCM:
        {
            // ROCm devices: Prefer FP16 (better support than BF16 on AMD)
            if (device.supports_fp16)
            {
                LOG_INFO("AUTO precision: ROCm device supports FP16 → selecting FP16");
                LOG_INFO("  Device: " << device.name);
                LOG_INFO("  Expected: 50% memory bandwidth, matrix core acceleration");
                return ComputePrecision::FP16;
            }
            else if (device.supports_bf16)
            {
                LOG_INFO("AUTO precision: ROCm device supports BF16 → selecting BF16");
                LOG_INFO("  Device: " << device.name);
                return ComputePrecision::BF16;
            }
            else
            {
                LOG_INFO("AUTO precision: ROCm device, no FP16/BF16 → selecting FP32");
                LOG_INFO("  Device: " << device.name);
                return ComputePrecision::FP32;
            }
        }

        case ComputeBackendType::GPU_VULKAN:
        {
            // Vulkan: Conservative FP32 for now (extension-dependent)
            LOG_INFO("AUTO precision: Vulkan device → selecting FP32 (conservative)");
            LOG_INFO("  Device: " << device.name);
            LOG_INFO("  Note: FP16/BF16 support depends on extensions (not yet detected)");
            return ComputePrecision::FP32;
        }

        default:
            LOG_WARN("AUTO precision: Unknown device type → defaulting to FP32");
            return ComputePrecision::FP32;
        }
    }

} // namespace llaminar2
