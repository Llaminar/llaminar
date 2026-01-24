/**
 * @file CUDAKernelProfiler.h
 * @brief CUDA event-based kernel profiling for accurate GPU timing
 * @author David Sanftenberg
 *
 * Provides accurate GPU kernel timing using CUDA events. Unlike CPU-based
 * timing which only measures kernel launch time, CUDA events measure
 * actual kernel execution time on the GPU.
 *
 * Enable via LLAMINAR_PROFILE_KERNELS=1 environment variable (shared with
 * KernelProfiler.h for unified control).
 *
 * Usage:
 *   // Option 1: Scoped timing (RAII)
 *   {
 *       CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::FLASH_ATTENTION);
 *       // ... launch kernel ...
 *   }
 *
 *   // Option 2: Manual timing
 *   CUDA_KERNEL_PROFILE_BEGIN(timer);
 *   // ... launch kernel ...
 *   CUDA_KERNEL_PROFILE_END(timer, CUDAKernelType::GEMM_CUTLASS);
 *
 * At end of inference, call CUDAKernelProfiler::printSummary() to see results.
 */
#pragma once

#include <atomic>
#include <array>
#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>
#include <mutex>

#include "DebugEnv.h"

// Forward declare CUDA types to avoid including cuda_runtime.h in header
// (allows .cpp files to include this without nvcc)
struct CUevent_st;
typedef CUevent_st *cudaEvent_t;
typedef struct CUstream_st *cudaStream_t;

namespace llaminar2
{

    /**
     * @brief CUDA-specific kernel type categories for GPU profiling
     *
     * These are separate from KernelType to provide finer-grained GPU timing.
     */
    enum class CUDAKernelType : uint8_t
    {
        // GEMM variants
        GEMM_CUTLASS = 0,    ///< CUTLASS INT8 quantized GEMM
        GEMM_CUBLAS,         ///< cuBLAS FP32/FP16/BF16 GEMM
        GEMM_WEIGHT_CONVERT, ///< Weight quantization/conversion
        GEMM_SCALE_OUTPUT,   ///< Output rescaling after INT8 GEMM

        // Flash Attention
        FLASH_ATTN_PREFILL, ///< FA3-style prefill kernel
        FLASH_ATTN_DECODE,  ///< Flash Decoding kernel
        FLASH_ATTN_REDUCE,  ///< Partial sum reduction

        // Ops kernels
        RMS_NORM,     ///< RMS normalization
        SWIGLU,       ///< SwiGLU activation
        ROPE,         ///< Rotary position embedding
        RESIDUAL_ADD, ///< Residual connection addition
        BIAS_ADD,     ///< Bias addition
        VECTOR_ADD,   ///< General vector addition

        // Embedding
        EMBEDDING_LOOKUP, ///< Token embedding lookup

        // KV Cache
        KVCACHE_APPEND, ///< Append to KV cache
        KVCACHE_GATHER, ///< Gather from KV cache

        // Memory operations
        H2D_TRANSFER, ///< Host to Device transfer
        D2H_TRANSFER, ///< Device to Host transfer
        D2D_TRANSFER, ///< Device to Device transfer

        // Quantization
        QUANTIZE_ACTIVATIONS, ///< FP32 -> INT8 activation quantization

        COUNT ///< Sentinel for array sizing
    };

    /**
     * @brief Get human-readable name for a CUDA kernel type
     */
    inline const char *cudaKernelTypeName(CUDAKernelType type)
    {
        switch (type)
        {
        case CUDAKernelType::GEMM_CUTLASS:
            return "GEMM_CUTLASS";
        case CUDAKernelType::GEMM_CUBLAS:
            return "GEMM_CUBLAS";
        case CUDAKernelType::GEMM_WEIGHT_CONVERT:
            return "GEMM_WEIGHT_CONVERT";
        case CUDAKernelType::GEMM_SCALE_OUTPUT:
            return "GEMM_SCALE_OUTPUT";
        case CUDAKernelType::FLASH_ATTN_PREFILL:
            return "FLASH_ATTN_PREFILL";
        case CUDAKernelType::FLASH_ATTN_DECODE:
            return "FLASH_ATTN_DECODE";
        case CUDAKernelType::FLASH_ATTN_REDUCE:
            return "FLASH_ATTN_REDUCE";
        case CUDAKernelType::RMS_NORM:
            return "RMS_NORM";
        case CUDAKernelType::SWIGLU:
            return "SWIGLU";
        case CUDAKernelType::ROPE:
            return "ROPE";
        case CUDAKernelType::RESIDUAL_ADD:
            return "RESIDUAL_ADD";
        case CUDAKernelType::BIAS_ADD:
            return "BIAS_ADD";
        case CUDAKernelType::VECTOR_ADD:
            return "VECTOR_ADD";
        case CUDAKernelType::EMBEDDING_LOOKUP:
            return "EMBEDDING_LOOKUP";
        case CUDAKernelType::KVCACHE_APPEND:
            return "KVCACHE_APPEND";
        case CUDAKernelType::KVCACHE_GATHER:
            return "KVCACHE_GATHER";
        case CUDAKernelType::H2D_TRANSFER:
            return "H2D_TRANSFER";
        case CUDAKernelType::D2H_TRANSFER:
            return "D2H_TRANSFER";
        case CUDAKernelType::D2D_TRANSFER:
            return "D2D_TRANSFER";
        case CUDAKernelType::QUANTIZE_ACTIVATIONS:
            return "QUANTIZE_ACTIVATIONS";
        default:
            return "UNKNOWN";
        }
    }

    /**
     * @brief Thread-safe CUDA kernel profiling accumulator
     *
     * Uses mutex for thread-safe accumulation (CUDA event timing requires
     * synchronization anyway, so mutex overhead is acceptable).
     */
    class CUDAKernelProfiler
    {
    public:
        /**
         * @brief Per-kernel-type timing statistics (microseconds)
         */
        struct KernelStats
        {
            double total_us = 0.0;   ///< Total time in microseconds
            uint64_t call_count = 0; ///< Number of calls
            double max_us = 0.0;     ///< Maximum single call time
            double min_us = 1e12;    ///< Minimum single call time
        };

        /**
         * @brief Check if CUDA profiling is enabled
         */
        static bool isEnabled()
        {
            return debugEnv().profile.enabled;
        }

        /**
         * @brief Record a kernel timing (in microseconds)
         */
        static void record(CUDAKernelType type, double elapsed_us)
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);

            auto idx = static_cast<size_t>(type);
            auto &stats = inst.stats_[idx];

            stats.total_us += elapsed_us;
            stats.call_count++;
            stats.max_us = std::max(stats.max_us, elapsed_us);
            stats.min_us = std::min(stats.min_us, elapsed_us);
        }

        /**
         * @brief Reset all statistics
         */
        static void reset()
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);

            for (auto &stats : inst.stats_)
            {
                stats = KernelStats{};
            }
        }

        /**
         * @brief Get summary string
         */
        static std::string getSummary(uint64_t total_tokens = 0)
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);

            std::ostringstream oss;
            oss << "\n";
            oss << "╔══════════════════════════════════════════════════════════════════════════════════════╗\n";
            oss << "║                       CUDA KERNEL PROFILING SUMMARY                                  ║\n";
            oss << "╠══════════════════════════════════════════════════════════════════════════════════════╣\n";
            oss << "║  KERNEL TYPE           │   CALLS   │   TOTAL (ms)  │   AVG (μs)  │   MAX (μs)  │  %  ║\n";
            oss << "╠══════════════════════════════════════════════════════════════════════════════════════╣\n";

            // Calculate total time for percentage
            double total_time_us = 0.0;
            for (const auto &stats : inst.stats_)
            {
                total_time_us += stats.total_us;
            }

            // Sort by total time (descending) - create index array
            std::array<size_t, static_cast<size_t>(CUDAKernelType::COUNT)> indices;
            for (size_t i = 0; i < indices.size(); ++i)
                indices[i] = i;
            std::sort(indices.begin(), indices.end(), [&inst](size_t a, size_t b)
                      { return inst.stats_[a].total_us > inst.stats_[b].total_us; });

            for (size_t idx : indices)
            {
                const auto &stats = inst.stats_[idx];
                if (stats.call_count == 0)
                    continue;

                double avg_us = stats.total_us / static_cast<double>(stats.call_count);
                double pct = (total_time_us > 0) ? (stats.total_us / total_time_us * 100.0) : 0.0;
                double total_ms = stats.total_us / 1000.0;

                oss << "║  " << std::left << std::setw(20) << cudaKernelTypeName(static_cast<CUDAKernelType>(idx))
                    << " │ " << std::right << std::setw(9) << stats.call_count
                    << " │ " << std::right << std::setw(13) << std::fixed << std::setprecision(2) << total_ms
                    << " │ " << std::right << std::setw(11) << std::fixed << std::setprecision(1) << avg_us
                    << " │ " << std::right << std::setw(11) << std::fixed << std::setprecision(1) << stats.max_us
                    << " │ " << std::right << std::setw(3) << static_cast<int>(pct) << "% ║\n";
            }

            oss << "╠══════════════════════════════════════════════════════════════════════════════════════╣\n";
            oss << "║  TOTAL GPU TIME: " << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                << (total_time_us / 1000.0) << " ms";

            if (total_tokens > 0)
            {
                double ms_per_token = (total_time_us / 1000.0) / static_cast<double>(total_tokens);
                oss << "  │  " << std::setw(6) << std::fixed << std::setprecision(2) << ms_per_token << " ms/token";
            }
            oss << std::setw(30) << " " << "║\n";

            oss << "╚══════════════════════════════════════════════════════════════════════════════════════╝\n";

            return oss.str();
        }

        /**
         * @brief Print summary to stderr
         */
        static void printSummary(uint64_t total_tokens = 0)
        {
            fprintf(stderr, "%s", getSummary(total_tokens).c_str());
        }

    private:
        CUDAKernelProfiler() = default;

        static CUDAKernelProfiler &getInstance()
        {
            static CUDAKernelProfiler instance;
            return instance;
        }

        std::mutex mutex_;
        std::array<KernelStats, static_cast<size_t>(CUDAKernelType::COUNT)> stats_;
    };

    // ========================================================================
    // CUDA Event-based timer (implementation in .cu file)
    // ========================================================================

    /**
     * @brief Scoped CUDA kernel timer using CUDA events
     *
     * Records start event at construction, stop event and elapsed time at destruction.
     * The timing is synchronous - destructor blocks until GPU kernel completes.
     */
    class ScopedCUDAKernelTimer
    {
    public:
        /**
         * @brief Construct timer and record start event
         * @param type Kernel type for profiling categorization
         * @param stream CUDA stream (nullptr = default stream)
         */
        ScopedCUDAKernelTimer(CUDAKernelType type, cudaStream_t stream = nullptr);

        /**
         * @brief Record stop event, synchronize, and record elapsed time
         */
        ~ScopedCUDAKernelTimer();

        // Non-copyable, non-movable
        ScopedCUDAKernelTimer(const ScopedCUDAKernelTimer &) = delete;
        ScopedCUDAKernelTimer &operator=(const ScopedCUDAKernelTimer &) = delete;

    private:
        CUDAKernelType type_;
        bool enabled_;
        cudaEvent_t start_event_;
        cudaEvent_t stop_event_;
        cudaStream_t stream_;
    };

    /**
     * @brief Manual CUDA kernel timer for non-RAII usage patterns
     */
    class ManualCUDAKernelTimer
    {
    public:
        ManualCUDAKernelTimer();
        ~ManualCUDAKernelTimer();

        /**
         * @brief Record start event
         * @param stream CUDA stream (nullptr = default stream)
         */
        void begin(cudaStream_t stream = nullptr);

        /**
         * @brief Record stop event, synchronize, and record elapsed time
         * @param type Kernel type for profiling categorization
         */
        void end(CUDAKernelType type);

        // Non-copyable, non-movable
        ManualCUDAKernelTimer(const ManualCUDAKernelTimer &) = delete;
        ManualCUDAKernelTimer &operator=(const ManualCUDAKernelTimer &) = delete;

    private:
        bool enabled_;
        bool started_;
        cudaEvent_t start_event_;
        cudaEvent_t stop_event_;
        cudaStream_t stream_;
    };

} // namespace llaminar2

// ============================================================================
// Convenience macros for CUDA kernel profiling
// ============================================================================

/**
 * @brief Scoped CUDA kernel profiling (RAII-based, synchronous)
 *
 * Usage:
 *   {
 *       CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::FLASH_ATTN_DECODE);
 *       cudaKernel<<<grid, block>>>(args...);
 *   } // Timer synchronizes and records here
 */
#define CUDA_KERNEL_PROFILE_SCOPE(kernel_type) \
    ::llaminar2::ScopedCUDAKernelTimer _cuda_timer_##__LINE__(kernel_type)

/**
 * @brief Scoped CUDA kernel profiling with stream
 */
#define CUDA_KERNEL_PROFILE_SCOPE_STREAM(kernel_type, stream) \
    ::llaminar2::ScopedCUDAKernelTimer _cuda_timer_##__LINE__(kernel_type, stream)

/**
 * @brief Manual CUDA kernel profiling begin
 */
#define CUDA_KERNEL_PROFILE_BEGIN(timer_name)      \
    ::llaminar2::ManualCUDAKernelTimer timer_name; \
    timer_name.begin()

#define CUDA_KERNEL_PROFILE_BEGIN_STREAM(timer_name, stream) \
    ::llaminar2::ManualCUDAKernelTimer timer_name;           \
    timer_name.begin(stream)

#define CUDA_KERNEL_PROFILE_END(timer_name, kernel_type) \
    timer_name.end(kernel_type)
