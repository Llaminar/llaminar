/**
 * @file ROCmKernelProfiler.h
 * @brief ROCm/HIP event-based kernel profiling for accurate GPU timing
 * @author David Sanftenberg
 *
 * Provides accurate GPU kernel timing using HIP events. Unlike CPU-based
 * timing which only measures kernel launch time, HIP events measure
 * actual kernel execution time on the GPU.
 *
 * Enable via LLAMINAR_PROFILING=1 environment variable (shared with
 * KernelProfiler.h and CUDAKernelProfiler.h for unified control).
 *
 * Usage:
 *   // Option 1: Scoped timing (RAII)
 *   {
 *       ROCM_KERNEL_PROFILE_SCOPE(ROCmKernelType::FLASH_ATTENTION);
 *       // ... launch kernel ...
 *   }
 *
 *   // Option 2: Manual timing
 *   ROCM_KERNEL_PROFILE_BEGIN(timer);
 *   // ... launch kernel ...
 *   ROCM_KERNEL_PROFILE_END(timer, ROCmKernelType::GEMM_CK);
 *
 * At end of inference, call ROCmKernelProfiler::printSummary() to see results.
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

// Forward declare HIP types to avoid including hip_runtime.h in header
// (allows .cpp files to include this without hipcc)
struct ihipEvent_t;
typedef ihipEvent_t *hipEvent_t;
typedef struct ihipStream_t *hipStream_t;

namespace llaminar2
{

    /**
     * @brief ROCm-specific kernel type categories for GPU profiling
     *
     * These mirror CUDAKernelType but are specific to ROCm/HIP kernels.
     */
    enum class ROCmKernelType : uint8_t
    {
        // GEMM variants
        GEMM_CK = 0,         ///< Composable Kernel INT8/FP8 GEMM
        GEMM_ROCBLAS,        ///< rocBLAS FP32/FP16/BF16 GEMM
        GEMM_WEIGHT_CONVERT, ///< Weight quantization/conversion
        GEMM_SCALE_OUTPUT,   ///< Output rescaling after INT8 GEMM

        // Flash Attention
        FLASH_ATTN_PREFILL, ///< Flash Attention 2 prefill kernel
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
     * @brief Get human-readable name for a ROCm kernel type
     */
    inline const char *rocmKernelTypeName(ROCmKernelType type)
    {
        switch (type)
        {
        case ROCmKernelType::GEMM_CK:
            return "GEMM_CK";
        case ROCmKernelType::GEMM_ROCBLAS:
            return "GEMM_ROCBLAS";
        case ROCmKernelType::GEMM_WEIGHT_CONVERT:
            return "GEMM_WEIGHT_CONVERT";
        case ROCmKernelType::GEMM_SCALE_OUTPUT:
            return "GEMM_SCALE_OUTPUT";
        case ROCmKernelType::FLASH_ATTN_PREFILL:
            return "FLASH_ATTN_PREFILL";
        case ROCmKernelType::FLASH_ATTN_DECODE:
            return "FLASH_ATTN_DECODE";
        case ROCmKernelType::FLASH_ATTN_REDUCE:
            return "FLASH_ATTN_REDUCE";
        case ROCmKernelType::RMS_NORM:
            return "RMS_NORM";
        case ROCmKernelType::SWIGLU:
            return "SWIGLU";
        case ROCmKernelType::ROPE:
            return "ROPE";
        case ROCmKernelType::RESIDUAL_ADD:
            return "RESIDUAL_ADD";
        case ROCmKernelType::BIAS_ADD:
            return "BIAS_ADD";
        case ROCmKernelType::VECTOR_ADD:
            return "VECTOR_ADD";
        case ROCmKernelType::EMBEDDING_LOOKUP:
            return "EMBEDDING_LOOKUP";
        case ROCmKernelType::KVCACHE_APPEND:
            return "KVCACHE_APPEND";
        case ROCmKernelType::KVCACHE_GATHER:
            return "KVCACHE_GATHER";
        case ROCmKernelType::H2D_TRANSFER:
            return "H2D_TRANSFER";
        case ROCmKernelType::D2H_TRANSFER:
            return "D2H_TRANSFER";
        case ROCmKernelType::D2D_TRANSFER:
            return "D2D_TRANSFER";
        case ROCmKernelType::QUANTIZE_ACTIVATIONS:
            return "QUANTIZE_ACTIVATIONS";
        default:
            return "UNKNOWN";
        }
    }

    /**
     * @brief Thread-safe ROCm kernel profiling accumulator
     *
     * Uses mutex for thread-safe accumulation (HIP event timing requires
     * synchronization anyway, so mutex overhead is acceptable).
     *
     * Supports per-device statistics for multi-GPU tensor parallelism.
     */
    class ROCmKernelProfiler
    {
    public:
        static constexpr size_t COUNT = static_cast<size_t>(ROCmKernelType::COUNT);

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
         * @brief Per-device statistics container
         */
        struct DeviceStats
        {
            std::array<KernelStats, COUNT> stats;
        };

        /**
         * @brief Check if ROCm profiling is enabled
         */
        static bool isEnabled()
        {
            return debugEnv().profile.enabled;
        }

        /**
         * @brief Set the current device context for profiling
         * @param device_id HIP device ID (0, 1, 2, ...)
         */
        static void setCurrentDevice(int device_id)
        {
            current_device_id() = device_id;
        }

        /**
         * @brief Clear the current device context
         */
        static void clearCurrentDevice()
        {
            current_device_id() = -1;
        }

        /**
         * @brief Record a kernel timing (in microseconds)
         */
        static void record(ROCmKernelType type, double elapsed_us)
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);

            auto idx = static_cast<size_t>(type);

            // Update global stats
            auto &stats = inst.stats_[idx];
            stats.total_us += elapsed_us;
            stats.call_count++;
            stats.max_us = std::max(stats.max_us, elapsed_us);
            stats.min_us = std::min(stats.min_us, elapsed_us);

            // Update per-device stats if device context is set
            int dev_id = current_device_id();
            if (dev_id >= 0)
            {
                auto &dev_stats = inst.device_stats_[dev_id].stats[idx];
                dev_stats.total_us += elapsed_us;
                dev_stats.call_count++;
                dev_stats.max_us = std::max(dev_stats.max_us, elapsed_us);
                dev_stats.min_us = std::min(dev_stats.min_us, elapsed_us);
            }
        }

        /**
         * @brief Record a kernel timing with explicit device ID
         */
        static void record(ROCmKernelType type, double elapsed_us, int device_id)
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);

            auto idx = static_cast<size_t>(type);

            // Update global stats
            auto &stats = inst.stats_[idx];
            stats.total_us += elapsed_us;
            stats.call_count++;
            stats.max_us = std::max(stats.max_us, elapsed_us);
            stats.min_us = std::min(stats.min_us, elapsed_us);

            // Update per-device stats
            if (device_id >= 0)
            {
                auto &dev_stats = inst.device_stats_[device_id].stats[idx];
                dev_stats.total_us += elapsed_us;
                dev_stats.call_count++;
                dev_stats.max_us = std::max(dev_stats.max_us, elapsed_us);
                dev_stats.min_us = std::min(dev_stats.min_us, elapsed_us);
            }
        }

        /**
         * @brief Get list of devices with recorded stats
         */
        static std::vector<int> getDevices()
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);

            std::vector<int> devices;
            for (const auto &[id, _] : inst.device_stats_)
            {
                devices.push_back(id);
            }
            std::sort(devices.begin(), devices.end());
            return devices;
        }

        /**
         * @brief Get the number of unique devices
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
                stats = KernelStats{};
            }
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
                return ""; // No ROCm kernels were profiled
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
                    title_ss << "ROCm KERNEL PROFILING SUMMARY (" << device_count << " GPUs)";
                    title << title_ss.str() << fort::endr;
                }
                else
                {
                    title << "ROCm KERNEL PROFILING SUMMARY" << fort::endr;
                }
                title[0][0].set_cell_text_align(fort::text_align::center);
                title.row(0).set_cell_row_type(fort::row_type::header);
                oss << "\n"
                    << title.to_string();
            }

            // Sort by total time (descending)
            std::array<size_t, COUNT> indices;
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
                    dev_ss << "rocm:" << dev;
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

                    table << rocmKernelTypeName(static_cast<ROCmKernelType>(idx)) << total_ss.str();
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

                    table << rocmKernelTypeName(static_cast<ROCmKernelType>(idx))
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
        ROCmKernelProfiler() = default;

        static ROCmKernelProfiler &getInstance()
        {
            static ROCmKernelProfiler instance;
            return instance;
        }

        static int &current_device_id()
        {
            thread_local int device_id = -1;
            return device_id;
        }

        std::mutex mutex_;
        std::array<KernelStats, COUNT> stats_;
        std::unordered_map<int, DeviceStats> device_stats_;
    };

    // ========================================================================
    // HIP Event-based timer (implementation in .hip file)
    // ========================================================================

    /**
     * @brief Scoped ROCm kernel timer using HIP events
     *
     * Records start event at construction, stop event and elapsed time at destruction.
     * The timing is synchronous - destructor blocks until GPU kernel completes.
     */
    class ScopedROCmKernelTimer
    {
    public:
        /**
         * @brief Construct timer and record start event
         * @param type Kernel type for profiling categorization
         * @param stream HIP stream (nullptr = default stream)
         */
        ScopedROCmKernelTimer(ROCmKernelType type, hipStream_t stream = nullptr);

        /**
         * @brief Record stop event, synchronize, and record elapsed time
         */
        ~ScopedROCmKernelTimer();

        // Non-copyable, non-movable
        ScopedROCmKernelTimer(const ScopedROCmKernelTimer &) = delete;
        ScopedROCmKernelTimer &operator=(const ScopedROCmKernelTimer &) = delete;

    private:
        ROCmKernelType type_;
        bool enabled_;
        hipEvent_t start_event_;
        hipEvent_t stop_event_;
        hipStream_t stream_;
    };

    /**
     * @brief Manual ROCm kernel timer for non-RAII usage patterns
     */
    class ManualROCmKernelTimer
    {
    public:
        ManualROCmKernelTimer();
        ~ManualROCmKernelTimer();

        /**
         * @brief Record start event
         * @param stream HIP stream (nullptr = default stream)
         */
        void begin(hipStream_t stream = nullptr);

        /**
         * @brief Record stop event, synchronize, and record elapsed time
         * @param type Kernel type for profiling categorization
         */
        void end(ROCmKernelType type);

        // Non-copyable, non-movable
        ManualROCmKernelTimer(const ManualROCmKernelTimer &) = delete;
        ManualROCmKernelTimer &operator=(const ManualROCmKernelTimer &) = delete;

    private:
        bool enabled_;
        bool started_;
        hipEvent_t start_event_;
        hipEvent_t stop_event_;
        hipStream_t stream_;
    };

} // namespace llaminar2

// ============================================================================
// Convenience macros for ROCm kernel profiling
// ============================================================================

/**
 * @brief Scoped ROCm kernel profiling (RAII-based, synchronous)
 *
 * Usage:
 *   {
 *       ROCM_KERNEL_PROFILE_SCOPE(ROCmKernelType::FLASH_ATTN_DECODE);
 *       hipLaunchKernelGGL(...);
 *   } // Timer synchronizes and records here
 */
#define ROCM_KERNEL_PROFILE_SCOPE(kernel_type) \
    ::llaminar2::ScopedROCmKernelTimer _rocm_timer_##__LINE__(kernel_type)

/**
 * @brief Scoped ROCm kernel profiling with stream
 */
#define ROCM_KERNEL_PROFILE_SCOPE_STREAM(kernel_type, stream) \
    ::llaminar2::ScopedROCmKernelTimer _rocm_timer_##__LINE__(kernel_type, stream)

/**
 * @brief Manual ROCm kernel profiling begin
 */
#define ROCM_KERNEL_PROFILE_BEGIN(timer_name)      \
    ::llaminar2::ManualROCmKernelTimer timer_name; \
    timer_name.begin()

#define ROCM_KERNEL_PROFILE_BEGIN_STREAM(timer_name, stream) \
    ::llaminar2::ManualROCmKernelTimer timer_name;           \
    timer_name.begin(stream)

#define ROCM_KERNEL_PROFILE_END(timer_name, kernel_type) \
    timer_name.end(kernel_type)
