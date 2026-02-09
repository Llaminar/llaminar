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
#include <unordered_map>
#include <vector>
#include <algorithm>

#include "DebugEnv.h"
#include "fort.hpp"

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
        FLASH_ATTN_PREFILL, ///< FA2-style pipelined prefill kernel (Ampere+)
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
     * @brief Thread-safe CUDA kernel profiling accumulator with per-device tracking
     *
     * Uses mutex for thread-safe accumulation (CUDA event timing requires
     * synchronization anyway, so mutex overhead is acceptable).
     * Supports per-device breakdown for multi-GPU tensor parallelism.
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

            void reset()
            {
                total_us = 0.0;
                call_count = 0;
                max_us = 0.0;
                min_us = 1e12;
            }

            void add(double elapsed_us)
            {
                total_us += elapsed_us;
                call_count++;
                max_us = std::max(max_us, elapsed_us);
                min_us = std::min(min_us, elapsed_us);
            }
        };

        /**
         * @brief Per-device kernel statistics array
         */
        struct DeviceStats
        {
            std::array<KernelStats, static_cast<size_t>(CUDAKernelType::COUNT)> stats;

            void reset()
            {
                for (auto &s : stats)
                    s.reset();
            }
        };

        /**
         * @brief Inference phase for profiling separation
         */
        enum class Phase : int
        {
            COMBINED = 0,
            PREFILL = 1,
            DECODE = 2
        };

        /**
         * @brief Set the current inference phase for this thread
         */
        static void setCurrentPhase(Phase phase)
        {
            current_phase() = phase;
        }

        /**
         * @brief Check if CUDA profiling is enabled
         */
        static bool isEnabled()
        {
            return debugEnv().profile.enabled;
        }

        /**
         * @brief Set the current CUDA device for this thread
         * @param device_id CUDA device ordinal (0, 1, 2, ...)
         */
        static void setCurrentDevice(int device_id)
        {
            current_device_id() = device_id;
        }

        /**
         * @brief Clear the current device context for this thread
         */
        static void clearCurrentDevice()
        {
            current_device_id() = -1;
        }

        /**
         * @brief Record a kernel timing (in microseconds, auto-attributes to current device)
         */
        static void record(CUDAKernelType type, double elapsed_us)
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);

            auto idx = static_cast<size_t>(type);

            // Record to global stats
            inst.stats_[idx].add(elapsed_us);

            // Record to phase-specific stats
            Phase phase = current_phase();
            if (phase == Phase::PREFILL)
                inst.prefill_stats_[idx].add(elapsed_us);
            else if (phase == Phase::DECODE)
                inst.decode_stats_[idx].add(elapsed_us);

            // Record to per-device stats if device context is set
            int device = current_device_id();
            if (device >= 0)
            {
                inst.device_stats_[device].stats[idx].add(elapsed_us);
            }
        }

        /**
         * @brief Record a kernel timing with explicit device
         */
        static void record(CUDAKernelType type, double elapsed_us, int device_id)
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);

            auto idx = static_cast<size_t>(type);

            // Record to global stats
            inst.stats_[idx].add(elapsed_us);

            // Record to phase-specific stats
            Phase phase = current_phase();
            if (phase == Phase::PREFILL)
                inst.prefill_stats_[idx].add(elapsed_us);
            else if (phase == Phase::DECODE)
                inst.decode_stats_[idx].add(elapsed_us);

            // Record to per-device stats
            if (device_id >= 0)
            {
                inst.device_stats_[device_id].stats[idx].add(elapsed_us);
            }
        }

        /**
         * @brief Get list of devices that have recorded stats
         */
        static std::vector<int> getDevices()
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);
            std::vector<int> devices;
            devices.reserve(inst.device_stats_.size());
            for (const auto &[id, _] : inst.device_stats_)
            {
                devices.push_back(id);
            }
            std::sort(devices.begin(), devices.end());
            return devices;
        }

        /**
         * @brief Get number of devices with recorded stats
         */
        static size_t getDeviceCount()
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);
            return inst.device_stats_.size();
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
                stats.reset();
            }
            for (auto &stats : inst.prefill_stats_)
                stats.reset();
            for (auto &stats : inst.decode_stats_)
                stats.reset();
            inst.device_stats_.clear();
        }

        /**
         * @brief Get summary string
         */
        static std::string getSummary(uint64_t total_tokens = 0)
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);

            // Check if any stats were recorded
            bool has_stats = false;
            for (const auto &stats : inst.stats_)
            {
                if (stats.call_count > 0)
                {
                    has_stats = true;
                    break;
                }
            }
            if (!has_stats)
            {
                return ""; // No CUDA kernels were profiled
            }

            size_t device_count = inst.device_stats_.size();
            std::vector<int> devices;
            for (const auto &[id, _] : inst.device_stats_)
            {
                devices.push_back(id);
            }
            std::sort(devices.begin(), devices.end());

            // Calculate total time for percentage
            double total_time_us = 0.0;
            for (const auto &stats : inst.stats_)
            {
                total_time_us += stats.total_us;
            }

            std::ostringstream oss;

            // Title table
            {
                fort::utf8_table title;
                title.set_border_style(FT_DOUBLE2_STYLE);
                if (device_count > 1)
                {
                    std::ostringstream title_ss;
                    title_ss << "CUDA KERNEL PROFILING SUMMARY (" << device_count << " GPUs)";
                    title << title_ss.str() << fort::endr;
                }
                else
                {
                    title << "CUDA KERNEL PROFILING SUMMARY" << fort::endr;
                }
                title[0][0].set_cell_text_align(fort::text_align::center);
                title.row(0).set_cell_row_type(fort::row_type::header);
                oss << "\n"
                    << title.to_string();
            }

            // Sort by total time (descending)
            std::array<size_t, static_cast<size_t>(CUDAKernelType::COUNT)> indices;
            for (size_t i = 0; i < indices.size(); ++i)
                indices[i] = i;
            std::sort(indices.begin(), indices.end(), [&inst](size_t a, size_t b)
                      { return inst.stats_[a].total_us > inst.stats_[b].total_us; });

            // Main data table
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            if (device_count > 1)
            {
                // Multi-device header
                table << fort::header << "Kernel Type" << "Total (ms)";
                for (int dev : devices)
                {
                    std::ostringstream dev_ss;
                    dev_ss << "cuda:" << dev;
                    table << dev_ss.str();
                }
                table << "Balance" << fort::endr;

                table.column(0).set_cell_text_align(fort::text_align::left);
                table.column(1).set_cell_text_align(fort::text_align::right);
                for (size_t i = 0; i < devices.size(); ++i)
                {
                    table.column(2 + i).set_cell_text_align(fort::text_align::right);
                }
                table.column(2 + devices.size()).set_cell_text_align(fort::text_align::right);

                for (size_t idx : indices)
                {
                    const auto &stats = inst.stats_[idx];
                    if (stats.call_count == 0)
                        continue;

                    double total_ms = stats.total_us / 1000.0;

                    // Collect per-device times
                    std::vector<double> dev_times;
                    double max_time = 0.0, min_time = 1e12;
                    for (int dev : devices)
                    {
                        auto it = inst.device_stats_.find(dev);
                        if (it != inst.device_stats_.end())
                        {
                            double dev_ms = it->second.stats[idx].total_us / 1000.0;
                            dev_times.push_back(dev_ms);
                            if (dev_ms > 0)
                            {
                                max_time = std::max(max_time, dev_ms);
                                min_time = std::min(min_time, dev_ms);
                            }
                        }
                        else
                        {
                            dev_times.push_back(0.0);
                        }
                    }

                    double balance = (max_time > 0) ? (min_time / max_time * 100.0) : 100.0;

                    std::ostringstream total_ss, balance_ss;
                    total_ss << std::fixed << std::setprecision(2) << total_ms;
                    balance_ss << static_cast<int>(balance) << "%";

                    table << cudaKernelTypeName(static_cast<CUDAKernelType>(idx)) << total_ss.str();
                    for (double t : dev_times)
                    {
                        std::ostringstream dev_time_ss;
                        dev_time_ss << std::fixed << std::setprecision(2) << t;
                        table << dev_time_ss.str();
                    }
                    table << balance_ss.str() << fort::endr;
                }

                // Separator and total row
                table << fort::separator;

                // Show effective wall-clock (max across devices)
                double max_device_time = 0.0;
                for (const auto &[_, dev_stats] : inst.device_stats_)
                {
                    double dev_total = 0.0;
                    for (const auto &s : dev_stats.stats)
                    {
                        dev_total += s.total_us;
                    }
                    max_device_time = std::max(max_device_time, dev_total);
                }

                std::ostringstream total_ss, wallclock_ss;
                total_ss << std::fixed << std::setprecision(2) << (total_time_us / 1000.0) << " ms";
                wallclock_ss << std::fixed << std::setprecision(2) << (max_device_time / 1000.0) << " ms wall";

                table << "TOTAL GPU TIME" << total_ss.str();
                for (size_t i = 0; i < devices.size(); ++i)
                {
                    table << "";
                }
                table << wallclock_ss.str() << fort::endr;
            }
            else
            {
                // Single device format
                table << fort::header << "Kernel Type" << "Calls" << "Total (ms)" << "Avg (µs)" << "Max (µs)" << "%" << fort::endr;
                table.column(0).set_cell_text_align(fort::text_align::left);
                table.column(1).set_cell_text_align(fort::text_align::right);
                table.column(2).set_cell_text_align(fort::text_align::right);
                table.column(3).set_cell_text_align(fort::text_align::right);
                table.column(4).set_cell_text_align(fort::text_align::right);
                table.column(5).set_cell_text_align(fort::text_align::right);

                for (size_t idx : indices)
                {
                    const auto &stats = inst.stats_[idx];
                    if (stats.call_count == 0)
                        continue;

                    double avg_us = stats.total_us / static_cast<double>(stats.call_count);
                    double pct = (total_time_us > 0) ? (stats.total_us / total_time_us * 100.0) : 0.0;
                    double total_ms = stats.total_us / 1000.0;

                    std::ostringstream total_ss, avg_ss, max_ss, pct_ss;
                    total_ss << std::fixed << std::setprecision(2) << total_ms;
                    avg_ss << std::fixed << std::setprecision(1) << avg_us;
                    max_ss << std::fixed << std::setprecision(1) << stats.max_us;
                    pct_ss << static_cast<int>(pct) << "%";

                    table << cudaKernelTypeName(static_cast<CUDAKernelType>(idx))
                          << stats.call_count
                          << total_ss.str()
                          << avg_ss.str()
                          << max_ss.str()
                          << pct_ss.str()
                          << fort::endr;
                }

                // Separator and total row
                table << fort::separator;

                std::ostringstream total_ss;
                total_ss << std::fixed << std::setprecision(2) << (total_time_us / 1000.0) << " ms";

                if (total_tokens > 0)
                {
                    double ms_per_token = (total_time_us / 1000.0) / static_cast<double>(total_tokens);
                    std::ostringstream throughput_ss;
                    throughput_ss << std::fixed << std::setprecision(2) << ms_per_token << " ms/tok";
                    table << "TOTAL GPU TIME" << "" << total_ss.str() << "" << "" << throughput_ss.str() << fort::endr;
                }
                else
                {
                    table << "TOTAL GPU TIME" << "" << total_ss.str() << "" << "" << "" << fort::endr;
                }
            }

            oss << table.to_string();

            // Phase breakdown (if phase data was collected)
            bool has_prefill = false, has_decode = false;
            for (size_t i = 0; i < static_cast<size_t>(CUDAKernelType::COUNT); ++i)
            {
                if (inst.prefill_stats_[i].call_count > 0)
                    has_prefill = true;
                if (inst.decode_stats_[i].call_count > 0)
                    has_decode = true;
            }

            if (has_prefill || has_decode)
            {
                fort::utf8_table phase_table;
                phase_table.set_border_style(FT_DOUBLE2_STYLE);

                phase_table << fort::header << "Kernel Type" << "Prefill (ms)" << "Decode (ms)" << "Decode %" << fort::endr;
                phase_table.column(0).set_cell_text_align(fort::text_align::left);
                phase_table.column(1).set_cell_text_align(fort::text_align::right);
                phase_table.column(2).set_cell_text_align(fort::text_align::right);
                phase_table.column(3).set_cell_text_align(fort::text_align::right);

                double decode_total_us = 0.0;
                for (size_t idx : indices)
                {
                    const auto &pf = inst.prefill_stats_[idx];
                    const auto &dc = inst.decode_stats_[idx];
                    if (pf.call_count == 0 && dc.call_count == 0)
                        continue;

                    decode_total_us += dc.total_us;

                    std::ostringstream pf_ss, dc_ss, pct_ss;
                    pf_ss << std::fixed << std::setprecision(2) << (pf.total_us / 1000.0);
                    dc_ss << std::fixed << std::setprecision(2) << (dc.total_us / 1000.0);
                    double dc_pct = total_time_us > 0 ? (dc.total_us / total_time_us) * 100.0 : 0;
                    pct_ss << static_cast<int>(dc_pct) << "%";

                    phase_table << cudaKernelTypeName(static_cast<CUDAKernelType>(idx))
                                << pf_ss.str() << dc_ss.str() << pct_ss.str() << fort::endr;
                }

                phase_table << fort::separator;
                {
                    double prefill_total_us = total_time_us - decode_total_us;
                    std::ostringstream pf_ss, dc_ss;
                    pf_ss << std::fixed << std::setprecision(2) << (prefill_total_us / 1000.0);
                    dc_ss << std::fixed << std::setprecision(2) << (decode_total_us / 1000.0);
                    phase_table << "PHASE TOTAL" << pf_ss.str() << dc_ss.str() << "" << fort::endr;
                }

                oss << "\nCUDA KERNEL PHASE BREAKDOWN (Prefill vs Decode):\n";
                oss << phase_table.to_string();
            }

            return oss.str();
        }

        /**
         * @brief Print summary to stderr
         */
        static void printSummary(uint64_t total_tokens = 0)
        {
            std::string summary = getSummary(total_tokens);
            if (!summary.empty())
            {
                fprintf(stderr, "%s", summary.c_str());
            }
        }

    private:
        CUDAKernelProfiler() = default;

        static CUDAKernelProfiler &getInstance()
        {
            static CUDAKernelProfiler instance;
            return instance;
        }

        static int &current_device_id()
        {
            thread_local int device_id = -1;
            return device_id;
        }

        static Phase &current_phase()
        {
            static thread_local Phase phase = Phase::COMBINED;
            return phase;
        }

        std::mutex mutex_;
        std::array<KernelStats, static_cast<size_t>(CUDAKernelType::COUNT)> stats_;
        std::array<KernelStats, static_cast<size_t>(CUDAKernelType::COUNT)> prefill_stats_;
        std::array<KernelStats, static_cast<size_t>(CUDAKernelType::COUNT)> decode_stats_;
        std::unordered_map<int, DeviceStats> device_stats_;
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
