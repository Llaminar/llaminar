/**
 * @file KernelProfiler.h
 * @brief Per-kernel timing infrastructure for decode performance analysis
 * @author David Sanftenberg
 *
 * Provides thread-safe, low-overhead profiling for kernel operations.
 * Enable via LLAMINAR_PROFILE_KERNELS=1 environment variable.
 *
 * Usage:
 *   // At start of kernel operation
 *   KERNEL_PROFILE_BEGIN(KernelType::GEMM_Q8);
 *
 *   // ... kernel work ...
 *
 *   // At end of kernel operation
 *   KERNEL_PROFILE_END(KernelType::GEMM_Q8);
 *
 *   // Or use RAII scoped timing:
 *   {
 *       KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);
 *       // ... kernel work ...
 *   }
 *
 * At end of inference, call KernelProfiler::printSummary() to see results.
 */
#pragma once

#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <print>
#include <string>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include "DebugEnv.h"
#include "fort.hpp"

namespace llaminar2
{

    /**
     * @brief Kernel type categories for profiling
     */
    enum class KernelType : uint8_t
    {
        // GEMM variants
        GEMM_FP32 = 0, ///< FP32 GEMM (OpenBLAS fallback)
        GEMV_FP32,     ///< FP32 GEMV / M=1 matvec path
        GEMM_Q8,       ///< Q8_1 quantized GEMM (JIT microkernel)
        GEMV_Q8,       ///< Q8_1 quantized GEMV / M=1 decode matvec path
        GEMM_IQ4,      ///< IQ4_NL quantized GEMM

        // Attention
        ATTENTION,      ///< Full attention block (Q*K^T, softmax, *V)
        ATTENTION_QK,   ///< Q*K^T matmul only
        ATTENTION_SOFT, ///< Softmax only
        ATTENTION_V,    ///< Attention * V only

        // Normalization
        RMS_NORM, ///< RMS normalization

        // Activation / FFN
        SWIGLU,   ///< SwiGLU activation
        FFN_UP,   ///< FFN up projection
        FFN_DOWN, ///< FFN down projection
        FFN_GATE, ///< FFN gate projection

        // Positional encoding
        ROPE, ///< Rotary position embedding

        // Quantization
        QUANTIZE_Q8, ///< FP32 -> Q8_1 quantization

        // Misc
        EMBEDDING,    ///< Token embedding lookup
        LM_HEAD,      ///< Final LM head projection
        RESIDUAL_ADD, ///< Residual connection add
        SOFTMAX,      ///< Standalone softmax

        // Collective communication
        ALLREDUCE,  ///< AllReduce collective
        ALLGATHER,  ///< AllGather collective
        ALLGATHERV, ///< AllGatherV collective

        COUNT ///< Sentinel for array sizing
    };

    /**
     * @brief Get human-readable name for a kernel type
     */
    inline const char *kernelTypeName(KernelType type)
    {
        switch (type)
        {
        case KernelType::GEMM_FP32:
            return "GEMM_FP32";
        case KernelType::GEMV_FP32:
            return "GEMV_FP32";
        case KernelType::GEMM_Q8:
            return "GEMM_Q8";
        case KernelType::GEMV_Q8:
            return "GEMV_Q8";
        case KernelType::GEMM_IQ4:
            return "GEMM_IQ4";
        case KernelType::ATTENTION:
            return "ATTENTION";
        case KernelType::ATTENTION_QK:
            return "ATTENTION_QK";
        case KernelType::ATTENTION_SOFT:
            return "ATTENTION_SOFT";
        case KernelType::ATTENTION_V:
            return "ATTENTION_V";
        case KernelType::RMS_NORM:
            return "RMS_NORM";
        case KernelType::SWIGLU:
            return "SWIGLU";
        case KernelType::FFN_UP:
            return "FFN_UP";
        case KernelType::FFN_DOWN:
            return "FFN_DOWN";
        case KernelType::FFN_GATE:
            return "FFN_GATE";
        case KernelType::ROPE:
            return "ROPE";
        case KernelType::QUANTIZE_Q8:
            return "QUANTIZE_Q8";
        case KernelType::EMBEDDING:
            return "EMBEDDING";
        case KernelType::LM_HEAD:
            return "LM_HEAD";
        case KernelType::RESIDUAL_ADD:
            return "RESIDUAL_ADD";
        case KernelType::SOFTMAX:
            return "SOFTMAX";
        case KernelType::ALLREDUCE:
            return "ALLREDUCE";
        case KernelType::ALLGATHER:
            return "ALLGATHER";
        case KernelType::ALLGATHERV:
            return "ALLGATHERV";
        default:
            return "UNKNOWN";
        }
    }

    /**
     * @brief Thread-safe transfer profiling accumulator for coherence system
     *
     * Tracks bytes uploaded (H2D) and downloaded (D2H) when LLAMINAR_PROFILING is enabled.
     * Uses atomic operations for thread-safe accumulation without locks.
     * Supports per-stage breakdown via setCurrentStage()/clearCurrentStage().
     */
    class TransferProfiler
    {
    public:
        /**
         * @brief Per-direction transfer statistics
         */
        struct alignas(64) TransferStats // Cache-line aligned to prevent false sharing
        {
            std::atomic<uint64_t> total_bytes{0};    ///< Total bytes transferred
            std::atomic<uint64_t> transfer_count{0}; ///< Number of transfers
            std::atomic<uint64_t> total_ns{0};       ///< Total time in nanoseconds

            void reset()
            {
                total_bytes.store(0, std::memory_order_relaxed);
                transfer_count.store(0, std::memory_order_relaxed);
                total_ns.store(0, std::memory_order_relaxed);
            }

            void add(uint64_t bytes, uint64_t ns)
            {
                total_bytes.fetch_add(bytes, std::memory_order_relaxed);
                transfer_count.fetch_add(1, std::memory_order_relaxed);
                if (ns > 0)
                {
                    total_ns.fetch_add(ns, std::memory_order_relaxed);
                }
            }
        };

        /**
         * @brief Per-stage transfer statistics (both H2D and D2H)
         */
        struct StageTransferStats
        {
            TransferStats h2d;
            TransferStats d2h;
        };

        /**
         * @brief Check if profiling is enabled (from DebugEnv)
         */
        static bool isEnabled()
        {
            return debugEnv().profile.enabled;
        }

        /**
         * @brief Set the current stage context for transfer tracking (thread-local)
         * Call this before executing a stage to attribute transfers to that stage.
         */
        static void setCurrentStage(const std::string &stage_name)
        {
            if (!isEnabled())
                return;
            current_stage_name() = stage_name;
        }

        /**
         * @brief Clear the current stage context (thread-local)
         * Call this after stage execution completes.
         */
        static void clearCurrentStage()
        {
            current_stage_name().clear();
        }

        /**
         * @brief Record a host-to-device (upload) transfer
         * @param bytes Number of bytes transferred
         * @param duration_ns Optional duration in nanoseconds (0 if not measured)
         */
        static void recordH2D(uint64_t bytes, uint64_t duration_ns = 0)
        {
            if (!isEnabled())
                return;

            auto &inst = getInstance();

            // Record to global stats
            inst.h2d_stats_.add(bytes, duration_ns);

            // Record to per-stage stats if context is set
            const std::string &stage = current_stage_name();
            if (!stage.empty())
            {
                std::lock_guard<std::mutex> lock(inst.stage_mutex_);
                inst.stage_stats_[stage].h2d.add(bytes, duration_ns);
            }
        }

        /**
         * @brief Record a device-to-host (download) transfer
         * @param bytes Number of bytes transferred
         * @param duration_ns Optional duration in nanoseconds (0 if not measured)
         */
        static void recordD2H(uint64_t bytes, uint64_t duration_ns = 0)
        {
            if (!isEnabled())
                return;

            auto &inst = getInstance();

            // Record to global stats
            inst.d2h_stats_.add(bytes, duration_ns);

            // Record to per-stage stats if context is set
            const std::string &stage = current_stage_name();
            if (!stage.empty())
            {
                std::lock_guard<std::mutex> lock(inst.stage_mutex_);
                inst.stage_stats_[stage].d2h.add(bytes, duration_ns);
            }
        }

        /**
         * @brief Get H2D stats
         */
        static const TransferStats &getH2DStats()
        {
            return getInstance().h2d_stats_;
        }

        /**
         * @brief Get D2H stats
         */
        static const TransferStats &getD2HStats()
        {
            return getInstance().d2h_stats_;
        }

        /**
         * @brief Reset all statistics
         */
        static void reset()
        {
            auto &inst = getInstance();
            inst.h2d_stats_.reset();
            inst.d2h_stats_.reset();
            {
                std::lock_guard<std::mutex> lock(inst.stage_mutex_);
                inst.stage_stats_.clear();
            }
        }

        /**
         * @brief Get formatted summary of transfer statistics
         * @return Formatted string with transfer breakdown
         */
        static std::string getSummary()
        {
            if (!isEnabled())
            {
                return "";
            }

            auto &inst = getInstance();
            const auto &h2d = inst.h2d_stats_;
            const auto &d2h = inst.d2h_stats_;

            uint64_t h2d_bytes = h2d.total_bytes.load(std::memory_order_relaxed);
            uint64_t h2d_count = h2d.transfer_count.load(std::memory_order_relaxed);
            uint64_t h2d_ns = h2d.total_ns.load(std::memory_order_relaxed);

            uint64_t d2h_bytes = d2h.total_bytes.load(std::memory_order_relaxed);
            uint64_t d2h_count = d2h.transfer_count.load(std::memory_order_relaxed);
            uint64_t d2h_ns = d2h.total_ns.load(std::memory_order_relaxed);

            // Skip if no transfers
            if (h2d_count == 0 && d2h_count == 0)
            {
                return "";
            }

            auto formatBytes = [](uint64_t bytes) -> std::string
            {
                std::ostringstream s;
                s << std::fixed << std::setprecision(2);
                if (bytes >= 1024 * 1024 * 1024)
                    s << (bytes / (1024.0 * 1024.0 * 1024.0)) << " GB";
                else if (bytes >= 1024 * 1024)
                    s << (bytes / (1024.0 * 1024.0)) << " MB";
                else if (bytes >= 1024)
                    s << (bytes / 1024.0) << " KB";
                else
                    s << bytes << " B";
                return s.str();
            };

            std::ostringstream oss;

            // Title table
            {
                fort::utf8_table title;
                title.set_border_style(FT_DOUBLE2_STYLE);
                title << "TENSOR TRANSFER SUMMARY" << fort::endr;
                title[0][0].set_cell_text_align(fort::text_align::center);
                title.row(0).set_cell_row_type(fort::row_type::header);
                oss << "\n"
                    << title.to_string();
            }

            // Per-stage breakdown
            {
                std::lock_guard<std::mutex> lock(inst.stage_mutex_);
                if (!inst.stage_stats_.empty())
                {
                    fort::utf8_table stage_table;
                    stage_table.set_border_style(FT_DOUBLE2_STYLE);

                    stage_table << fort::header
                                << "Stage Name" << "H2D Count" << "H2D Bytes" << "D2H Count" << "D2H Bytes"
                                << fort::endr;

                    stage_table.column(0).set_cell_text_align(fort::text_align::left);
                    stage_table.column(1).set_cell_text_align(fort::text_align::right);
                    stage_table.column(2).set_cell_text_align(fort::text_align::right);
                    stage_table.column(3).set_cell_text_align(fort::text_align::right);
                    stage_table.column(4).set_cell_text_align(fort::text_align::right);

                    // Sort stages by total bytes (descending)
                    std::vector<std::pair<std::string, const StageTransferStats *>> sorted_stages;
                    for (const auto &kv : inst.stage_stats_)
                    {
                        sorted_stages.emplace_back(kv.first, &kv.second);
                    }
                    std::sort(sorted_stages.begin(), sorted_stages.end(),
                              [](const auto &a, const auto &b)
                              {
                                  uint64_t a_total = a.second->h2d.total_bytes.load() + a.second->d2h.total_bytes.load();
                                  uint64_t b_total = b.second->h2d.total_bytes.load() + b.second->d2h.total_bytes.load();
                                  return a_total > b_total;
                              });

                    for (const auto &[name, stats] : sorted_stages)
                    {
                        uint64_t sh2d_count = stats->h2d.transfer_count.load();
                        uint64_t sh2d_bytes = stats->h2d.total_bytes.load();
                        uint64_t sd2h_count = stats->d2h.transfer_count.load();
                        uint64_t sd2h_bytes = stats->d2h.total_bytes.load();

                        // Skip stages with no transfers
                        if (sh2d_count == 0 && sd2h_count == 0)
                            continue;

                        // Truncate stage name if too long
                        std::string display_name = name;
                        if (display_name.length() > 34)
                        {
                            display_name = display_name.substr(0, 31) + "...";
                        }

                        stage_table << display_name
                                    << sh2d_count
                                    << formatBytes(sh2d_bytes)
                                    << sd2h_count
                                    << formatBytes(sd2h_bytes)
                                    << fort::endr;
                    }

                    oss << stage_table.to_string();
                }
            }

            // Global totals table
            {
                fort::utf8_table totals;
                totals.set_border_style(FT_DOUBLE2_STYLE);

                totals << fort::header
                       << "Direction" << "Transfers" << "Total Bytes" << "Avg Size" << "Bandwidth"
                       << fort::endr;

                totals.column(0).set_cell_text_align(fort::text_align::left);
                totals.column(1).set_cell_text_align(fort::text_align::right);
                totals.column(2).set_cell_text_align(fort::text_align::right);
                totals.column(3).set_cell_text_align(fort::text_align::right);
                totals.column(4).set_cell_text_align(fort::text_align::right);

                // H2D row
                if (h2d_count > 0)
                {
                    double avg_bytes = static_cast<double>(h2d_bytes) / h2d_count;
                    double bandwidth_gbps = (h2d_ns > 0) ? (h2d_bytes / 1e9) / (h2d_ns / 1e9) : 0.0;
                    std::ostringstream bw_ss;
                    bw_ss << std::fixed << std::setprecision(2) << bandwidth_gbps << " GB/s";

                    totals << "H2D (Upload)"
                           << h2d_count
                           << formatBytes(h2d_bytes)
                           << formatBytes(static_cast<uint64_t>(avg_bytes))
                           << bw_ss.str()
                           << fort::endr;
                }

                // D2H row
                if (d2h_count > 0)
                {
                    double avg_bytes = static_cast<double>(d2h_bytes) / d2h_count;
                    double bandwidth_gbps = (d2h_ns > 0) ? (d2h_bytes / 1e9) / (d2h_ns / 1e9) : 0.0;
                    std::ostringstream bw_ss;
                    bw_ss << std::fixed << std::setprecision(2) << bandwidth_gbps << " GB/s";

                    totals << "D2H (Download)"
                           << d2h_count
                           << formatBytes(d2h_bytes)
                           << formatBytes(static_cast<uint64_t>(avg_bytes))
                           << bw_ss.str()
                           << fort::endr;
                }

                oss << totals.to_string();
            }

            return oss.str();
        }

        /**
         * @brief Print summary to stdout
         */
        static void printSummary()
        {
            std::string summary = getSummary();
            if (!summary.empty())
            {
                std::print("{}", summary);
            }
        }

        /**
         * @brief RAII helper to set/clear stage context
         */
        class StageScope
        {
        public:
            explicit StageScope(const std::string &stage_name)
            {
                TransferProfiler::setCurrentStage(stage_name);
            }
            ~StageScope()
            {
                TransferProfiler::clearCurrentStage();
            }
            StageScope(const StageScope &) = delete;
            StageScope &operator=(const StageScope &) = delete;
        };

    private:
        TransferProfiler() = default;

        static TransferProfiler &getInstance()
        {
            static TransferProfiler instance;
            return instance;
        }

        static std::string &current_stage_name()
        {
            thread_local std::string name;
            return name;
        }

        TransferStats h2d_stats_;
        TransferStats d2h_stats_;
        std::mutex stage_mutex_;
        std::unordered_map<std::string, StageTransferStats> stage_stats_;
    };

    /**
     * @brief Thread-safe kernel profiling accumulator with per-device tracking
     *
     * Uses atomic operations for thread-safe accumulation without locks.
     * Each kernel type has its own accumulator to avoid false sharing.
     * Supports per-device breakdown for multi-GPU tensor parallelism.
     */
    class KernelProfiler
    {
    public:
        using Clock = std::chrono::high_resolution_clock;
        using TimePoint = Clock::time_point;
        using Duration = std::chrono::nanoseconds;

        /**
         * @brief Per-kernel-type timing statistics
         */
        struct alignas(64) KernelStats // Cache-line aligned to prevent false sharing
        {
            std::atomic<uint64_t> total_ns{0};        ///< Total time in nanoseconds
            std::atomic<uint64_t> call_count{0};      ///< Number of calls
            std::atomic<uint64_t> max_ns{0};          ///< Maximum single call time
            std::atomic<uint64_t> min_ns{UINT64_MAX}; ///< Minimum single call time

            void reset()
            {
                total_ns.store(0, std::memory_order_relaxed);
                call_count.store(0, std::memory_order_relaxed);
                max_ns.store(0, std::memory_order_relaxed);
                min_ns.store(UINT64_MAX, std::memory_order_relaxed);
            }

            void add(uint64_t duration_ns)
            {
                total_ns.fetch_add(duration_ns, std::memory_order_relaxed);
                call_count.fetch_add(1, std::memory_order_relaxed);

                // Update max (lock-free CAS)
                uint64_t current_max = max_ns.load(std::memory_order_relaxed);
                while (duration_ns > current_max &&
                       !max_ns.compare_exchange_weak(current_max, duration_ns,
                                                     std::memory_order_relaxed,
                                                     std::memory_order_relaxed))
                {
                }

                // Update min (lock-free CAS)
                uint64_t current_min = min_ns.load(std::memory_order_relaxed);
                while (duration_ns < current_min &&
                       !min_ns.compare_exchange_weak(current_min, duration_ns,
                                                     std::memory_order_relaxed,
                                                     std::memory_order_relaxed))
                {
                }
            }
        };

        /**
         * @brief Per-device kernel statistics array
         */
        struct DeviceStats
        {
            std::array<KernelStats, static_cast<size_t>(KernelType::COUNT)> stats;

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
         * @brief Set the current inference phase for phase-separated profiling
         */
        static void setCurrentPhase(Phase phase)
        {
            current_phase() = phase;
        }

        /**
         * @brief Check if profiling is enabled (from DebugEnv)
         */
        static bool isEnabled()
        {
            // Use DebugEnv for centralized configuration
            return debugEnv().profile.enabled;
        }

        /**
         * @brief Set the current device context for this thread
         * @param device_key Device identifier (e.g., "cuda:0", "rocm:1", "cpu")
         */
        static void setCurrentDevice(const std::string &device_key)
        {
            current_device_key() = device_key;
        }

        /**
         * @brief Clear the current device context for this thread
         */
        static void clearCurrentDevice()
        {
            current_device_key().clear();
        }

        /**
         * @brief Record a kernel execution time (auto-attributes to current device)
         * @param type Kernel type
         * @param duration_ns Duration in nanoseconds
         */
        static void record(KernelType type, uint64_t duration_ns)
        {
            if (!isEnabled())
                return;

            auto &inst = getInstance();

            // Record to global stats (backward compatible)
            inst.stats_[static_cast<size_t>(type)].add(duration_ns);

            // Record to phase-specific stats
            Phase phase = current_phase();
            if (phase == Phase::PREFILL)
                inst.prefill_stats_[static_cast<size_t>(type)].add(duration_ns);
            else if (phase == Phase::DECODE)
                inst.decode_stats_[static_cast<size_t>(type)].add(duration_ns);

            // Record to per-device stats if device context is set
            const std::string &device = current_device_key();
            if (!device.empty())
            {
                std::lock_guard<std::mutex> lock(inst.device_mutex_);
                inst.device_stats_[device].stats[static_cast<size_t>(type)].add(duration_ns);
            }
        }

        /**
         * @brief Record a kernel execution time with explicit device
         * @param type Kernel type
         * @param duration_ns Duration in nanoseconds
         * @param device_key Device identifier
         */
        static void record(KernelType type, uint64_t duration_ns, const std::string &device_key)
        {
            if (!isEnabled())
                return;

            auto &inst = getInstance();

            // Record to global stats
            inst.stats_[static_cast<size_t>(type)].add(duration_ns);

            // Record to phase-specific stats
            Phase phase = current_phase();
            if (phase == Phase::PREFILL)
                inst.prefill_stats_[static_cast<size_t>(type)].add(duration_ns);
            else if (phase == Phase::DECODE)
                inst.decode_stats_[static_cast<size_t>(type)].add(duration_ns);

            // Record to per-device stats
            if (!device_key.empty())
            {
                std::lock_guard<std::mutex> lock(inst.device_mutex_);
                inst.device_stats_[device_key].stats[static_cast<size_t>(type)].add(duration_ns);
            }
        }

        /**
         * @brief Get stats for a kernel type (global/combined)
         */
        static KernelStats &getStats(KernelType type)
        {
            return getInstance().stats_[static_cast<size_t>(type)];
        }

        /**
         * @brief Get list of devices that have recorded stats
         */
        static std::vector<std::string> getDevices()
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.device_mutex_);
            std::vector<std::string> devices;
            devices.reserve(inst.device_stats_.size());
            for (const auto &[key, _] : inst.device_stats_)
            {
                devices.push_back(key);
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
            std::lock_guard<std::mutex> lock(inst.device_mutex_);
            return inst.device_stats_.size();
        }

        /**
         * @brief Reset all statistics
         */
        static void reset()
        {
            auto &inst = getInstance();
            for (auto &stats : inst.stats_)
                stats.reset();
            for (auto &stats : inst.prefill_stats_)
                stats.reset();
            for (auto &stats : inst.decode_stats_)
                stats.reset();
            current_phase() = Phase::COMBINED;
            // Reset per-device stats
            {
                std::lock_guard<std::mutex> lock(inst.device_mutex_);
                inst.device_stats_.clear();
            }
        }

        /**
         * @brief Print formatted summary of all kernel timings
         * @param total_tokens Total tokens processed (for tok/s calculation)
         * @param wall_clock_prefill_ms Wall-clock time for prefill phase
         * @param wall_clock_decode_ms Wall-clock time for decode phase
         * @param prefill_tokens Number of tokens in prefill phase
         * @param decode_tokens Number of tokens in decode phase
         * @return Formatted string with timing breakdown
         */
        static std::string getSummary(uint64_t total_tokens = 0,
                                      double wall_clock_prefill_ms = 0,
                                      double wall_clock_decode_ms = 0,
                                      uint64_t prefill_tokens = 0,
                                      uint64_t decode_tokens = 0)
        {
            if (!isEnabled())
            {
                return "[Kernel profiling disabled. Set LLAMINAR_PROFILING=1 to enable]\n";
            }

            auto &inst = getInstance();
            size_t device_count = getDeviceCount();
            std::vector<std::string> devices = getDevices();

            // Check if phase data was collected
            bool has_prefill = false, has_decode = false;
            for (size_t i = 0; i < static_cast<size_t>(KernelType::COUNT); ++i)
            {
                if (inst.prefill_stats_[i].call_count.load(std::memory_order_relaxed) > 0)
                    has_prefill = true;
                if (inst.decode_stats_[i].call_count.load(std::memory_order_relaxed) > 0)
                    has_decode = true;
            }

            std::ostringstream oss;

            // Helper lambda to render a single-device stats table
            auto renderPhaseTable = [&](const std::string &phase_label,
                                        const std::array<KernelStats, static_cast<size_t>(KernelType::COUNT)> &phase_stats,
                                        uint64_t phase_tokens,
                                        double wall_clock_ms)
            {
                // Calculate total for this phase
                uint64_t phase_total_ns = 0;
                for (size_t i = 0; i < static_cast<size_t>(KernelType::COUNT); ++i)
                {
                    phase_total_ns += phase_stats[i].total_ns.load(std::memory_order_relaxed);
                }

                if (phase_total_ns == 0)
                    return;

                // Title
                {
                    fort::utf8_table title;
                    title.set_border_style(FT_DOUBLE2_STYLE);
                    std::ostringstream title_ss;
                    title_ss << "CPU KERNEL PROFILING — " << phase_label;
                    if (phase_tokens > 0)
                        title_ss << " (" << phase_tokens << " tokens)";
                    title << title_ss.str() << fort::endr;
                    title[0][0].set_cell_text_align(fort::text_align::center);
                    title.row(0).set_cell_row_type(fort::row_type::header);
                    oss << "\n"
                        << title.to_string();
                }

                // Sort by total time (descending)
                std::array<size_t, static_cast<size_t>(KernelType::COUNT)> indices;
                for (size_t i = 0; i < indices.size(); ++i)
                    indices[i] = i;
                std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b)
                          { return phase_stats[a].total_ns.load(std::memory_order_relaxed) > phase_stats[b].total_ns.load(std::memory_order_relaxed); });

                double pct_base_ns = (wall_clock_ms > 0) ? (wall_clock_ms * 1e6) : static_cast<double>(phase_total_ns);

                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);
                table << fort::header << "Kernel Type" << "Calls" << "Total (ms)" << "Avg (µs)" << "Min/Max (µs)" << "%" << fort::endr;
                table.column(0).set_cell_text_align(fort::text_align::left);
                table.column(1).set_cell_text_align(fort::text_align::right);
                table.column(2).set_cell_text_align(fort::text_align::right);
                table.column(3).set_cell_text_align(fort::text_align::right);
                table.column(4).set_cell_text_align(fort::text_align::right);
                table.column(5).set_cell_text_align(fort::text_align::right);

                for (size_t idx : indices)
                {
                    uint64_t count = phase_stats[idx].call_count.load(std::memory_order_relaxed);
                    if (count == 0)
                        continue;

                    uint64_t total_ns_val = phase_stats[idx].total_ns.load(std::memory_order_relaxed);
                    uint64_t min_ns_val = phase_stats[idx].min_ns.load(std::memory_order_relaxed);
                    uint64_t max_ns_val = phase_stats[idx].max_ns.load(std::memory_order_relaxed);
                    double total_ms = total_ns_val / 1e6;
                    double avg_us = (total_ns_val / static_cast<double>(count)) / 1e3;
                    double min_us = min_ns_val / 1e3;
                    double max_us = max_ns_val / 1e3;
                    double pct = (pct_base_ns > 0) ? (static_cast<double>(total_ns_val) / pct_base_ns * 100.0) : 0.0;

                    std::ostringstream total_ss, avg_ss, minmax_ss, pct_ss;
                    total_ss << std::fixed << std::setprecision(2) << total_ms;
                    avg_ss << std::fixed << std::setprecision(1) << avg_us;
                    minmax_ss << std::fixed << std::setprecision(1) << min_us << "/" << max_us;
                    pct_ss << std::fixed << std::setprecision(1) << pct << "%";

                    table << kernelTypeName(static_cast<KernelType>(idx))
                          << count << total_ss.str() << avg_ss.str() << minmax_ss.str() << pct_ss.str()
                          << fort::endr;
                }

                // Separator and total
                table << fort::separator;

                double phase_total_ms = phase_total_ns / 1e6;
                double display_total_ms = (wall_clock_ms > 0) ? wall_clock_ms : phase_total_ms;
                std::ostringstream grand_ss;
                grand_ss << std::fixed << std::setprecision(2) << display_total_ms << " ms";

                if (phase_tokens > 0)
                {
                    double ms_per_tok = display_total_ms / static_cast<double>(phase_tokens);
                    double toks_per_sec = (phase_tokens * 1000.0) / display_total_ms;
                    std::ostringstream throughput_ss;
                    throughput_ss << std::fixed << std::setprecision(2) << toks_per_sec << " tok/s";
                    table << "TOTAL" << "" << grand_ss.str() << "" << "" << throughput_ss.str() << fort::endr;
                }
                else
                {
                    table << "TOTAL" << "" << grand_ss.str() << "" << "" << "" << fort::endr;
                }

                oss << table.to_string();
            };

            // Render phase-separated or combined tables
            if ((has_prefill || has_decode) && device_count <= 1)
            {
                // Phase-separated output for single device
                if (has_prefill)
                    renderPhaseTable("PREFILL", inst.prefill_stats_, prefill_tokens, wall_clock_prefill_ms);
                if (has_decode)
                    renderPhaseTable("DECODE", inst.decode_stats_, decode_tokens, wall_clock_decode_ms);
            }
            else
            {
                // Combined output (no phase data or multi-device)
                // Title table
                {
                    fort::utf8_table title;
                    title.set_border_style(FT_DOUBLE2_STYLE);
                    if (device_count > 1)
                    {
                        std::ostringstream title_ss;
                        title_ss << "CPU KERNEL PROFILING SUMMARY (" << device_count << " devices)";
                        title << title_ss.str() << fort::endr;
                    }
                    else
                    {
                        title << "CPU KERNEL PROFILING SUMMARY" << fort::endr;
                    }
                    title[0][0].set_cell_text_align(fort::text_align::center);
                    title.row(0).set_cell_row_type(fort::row_type::header);
                    oss << "\n"
                        << title.to_string();
                }

                // Main data table
                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);

                if (device_count > 1)
                {
                    // Multi-device header
                    table << fort::header << "Kernel Type" << "Total (ms)";
                    for (const auto &dev : devices)
                    {
                        std::string dev_short = dev.length() > 10 ? dev.substr(0, 10) : dev;
                        table << dev_short;
                    }
                    table << "Balance" << fort::endr;

                    table.column(0).set_cell_text_align(fort::text_align::left);
                    table.column(1).set_cell_text_align(fort::text_align::right);
                    for (size_t i = 0; i < devices.size(); ++i)
                    {
                        table.column(2 + i).set_cell_text_align(fort::text_align::right);
                    }
                    table.column(2 + devices.size()).set_cell_text_align(fort::text_align::right);
                }
                else
                {
                    // Single device header
                    table << fort::header << "Kernel Type" << "Calls" << "Total (ms)" << "Avg (µs)" << "Min/Max (µs)" << fort::endr;
                    table.column(0).set_cell_text_align(fort::text_align::left);
                    table.column(1).set_cell_text_align(fort::text_align::right);
                    table.column(2).set_cell_text_align(fort::text_align::right);
                    table.column(3).set_cell_text_align(fort::text_align::right);
                    table.column(4).set_cell_text_align(fort::text_align::right);
                }

                uint64_t grand_total_ns = 0;

                for (size_t i = 0; i < static_cast<size_t>(KernelType::COUNT); ++i)
                {
                    const auto &stats = inst.stats_[i];
                    uint64_t count = stats.call_count.load(std::memory_order_relaxed);
                    if (count == 0)
                        continue;

                    uint64_t total_ns = stats.total_ns.load(std::memory_order_relaxed);
                    double total_ms = total_ns / 1e6;
                    grand_total_ns += total_ns;

                    std::ostringstream total_ss;
                    total_ss << std::fixed << std::setprecision(2) << total_ms;

                    if (device_count > 1)
                    {
                        // Collect per-device times for this kernel
                        std::vector<double> device_times;
                        double max_time = 0.0, min_time = 1e12;

                        std::lock_guard<std::mutex> lock(inst.device_mutex_);
                        for (const auto &dev : devices)
                        {
                            auto it = inst.device_stats_.find(dev);
                            if (it != inst.device_stats_.end())
                            {
                                uint64_t dev_ns = it->second.stats[i].total_ns.load(std::memory_order_relaxed);
                                double dev_ms = dev_ns / 1e6;
                                device_times.push_back(dev_ms);
                                max_time = std::max(max_time, dev_ms);
                                min_time = std::min(min_time, dev_ms);
                            }
                            else
                            {
                                device_times.push_back(0.0);
                            }
                        }

                        // Calculate load balance (0-100%, higher is better)
                        double balance = (max_time > 0) ? (min_time / max_time * 100.0) : 100.0;

                        std::ostringstream balance_ss;
                        balance_ss << static_cast<int>(balance) << "%";

                        table << kernelTypeName(static_cast<KernelType>(i)) << total_ss.str();
                        for (double t : device_times)
                        {
                            std::ostringstream dev_ss;
                            dev_ss << std::fixed << std::setprecision(2) << t;
                            table << dev_ss.str();
                        }
                        table << balance_ss.str() << fort::endr;
                    }
                    else
                    {
                        // Single device format
                        uint64_t min_ns = stats.min_ns.load(std::memory_order_relaxed);
                        uint64_t max_ns = stats.max_ns.load(std::memory_order_relaxed);
                        double avg_us = (total_ns / static_cast<double>(count)) / 1e3;
                        double min_us = min_ns / 1e3;
                        double max_us = max_ns / 1e3;

                        std::ostringstream avg_ss, minmax_ss;
                        avg_ss << std::fixed << std::setprecision(1) << avg_us;
                        minmax_ss << std::fixed << std::setprecision(1) << min_us << "/" << max_us;

                        table << kernelTypeName(static_cast<KernelType>(i))
                              << count
                              << total_ss.str()
                              << avg_ss.str()
                              << minmax_ss.str()
                              << fort::endr;
                    }
                }

                // Separator and total row
                table << fort::separator;

                double grand_total_ms = grand_total_ns / 1e6;
                std::ostringstream grand_ss;
                grand_ss << std::fixed << std::setprecision(2) << grand_total_ms << " ms";

                if (total_tokens > 0)
                {
                    double toks_per_sec = (total_tokens * 1e9) / static_cast<double>(grand_total_ns);
                    std::ostringstream throughput_ss;
                    throughput_ss << std::fixed << std::setprecision(2) << toks_per_sec << " kernel tok/s";

                    if (device_count > 1)
                    {
                        table << "TOTAL KERNEL TIME" << grand_ss.str();
                        for (size_t i = 0; i < devices.size(); ++i)
                        {
                            table << "";
                        }
                        table << throughput_ss.str() << fort::endr;
                    }
                    else
                    {
                        table << "TOTAL" << "" << grand_ss.str() << "" << throughput_ss.str() << fort::endr;
                    }
                }
                else
                {
                    if (device_count > 1)
                    {
                        table << "TOTAL KERNEL TIME" << grand_ss.str();
                        for (size_t i = 0; i < devices.size() + 1; ++i)
                        {
                            table << "";
                        }
                        table << fort::endr;
                    }
                    else
                    {
                        table << "TOTAL" << "" << grand_ss.str() << "" << "" << fort::endr;
                    }
                }

                oss << table.to_string();
            }

            // Append transfer stats if any transfers occurred
            oss << TransferProfiler::getSummary();

            return oss.str();
        }

        /**
         * @brief Print summary to stdout
         */
        static void printSummary(uint64_t total_tokens = 0,
                                 double wall_clock_prefill_ms = 0,
                                 double wall_clock_decode_ms = 0,
                                 uint64_t prefill_tokens = 0,
                                 uint64_t decode_tokens = 0)
        {
            std::print("{}", getSummary(total_tokens, wall_clock_prefill_ms, wall_clock_decode_ms, prefill_tokens, decode_tokens));
        }

        /**
         * @brief Reset all statistics including transfer stats
         */
        static void resetAll()
        {
            reset();
            TransferProfiler::reset();
        }

    private:
        KernelProfiler() = default;

        static KernelProfiler &getInstance()
        {
            static KernelProfiler instance;
            return instance;
        }

        static std::string &current_device_key()
        {
            thread_local std::string device_key;
            return device_key;
        }

        static Phase &current_phase()
        {
            static thread_local Phase phase = Phase::COMBINED;
            return phase;
        }

        std::array<KernelStats, static_cast<size_t>(KernelType::COUNT)> stats_;
        std::array<KernelStats, static_cast<size_t>(KernelType::COUNT)> prefill_stats_;
        std::array<KernelStats, static_cast<size_t>(KernelType::COUNT)> decode_stats_;
        std::mutex device_mutex_;
        std::unordered_map<std::string, DeviceStats> device_stats_;
    };

    /**
     * @brief RAII scoped timer for kernel profiling
     */
    class ScopedKernelTimer
    {
    public:
        explicit ScopedKernelTimer(KernelType type)
            : type_(type), enabled_(KernelProfiler::isEnabled())
        {
            if (enabled_)
            {
                start_ = KernelProfiler::Clock::now();
            }
        }

        ~ScopedKernelTimer()
        {
            if (enabled_)
            {
                auto end = KernelProfiler::Clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_);
                KernelProfiler::record(type_, static_cast<uint64_t>(duration.count()));
            }
        }

        // Non-copyable, non-movable
        ScopedKernelTimer(const ScopedKernelTimer &) = delete;
        ScopedKernelTimer &operator=(const ScopedKernelTimer &) = delete;

    private:
        KernelType type_;
        bool enabled_;
        KernelProfiler::TimePoint start_;
    };

    /**
     * @brief Manual begin/end timing (for cases where RAII doesn't fit)
     */
    class ManualKernelTimer
    {
    public:
        void begin()
        {
            if (KernelProfiler::isEnabled())
            {
                start_ = KernelProfiler::Clock::now();
            }
        }

        void end(KernelType type)
        {
            if (KernelProfiler::isEnabled())
            {
                auto end = KernelProfiler::Clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_);
                KernelProfiler::record(type, static_cast<uint64_t>(duration.count()));
            }
        }

    private:
        KernelProfiler::TimePoint start_;
    };

} // namespace llaminar2

// ============================================================================
// Convenience macros for kernel profiling
// ============================================================================

/**
 * @brief Scoped kernel profiling (RAII-based)
 *
 * Usage:
 *   void myKernel() {
 *       KERNEL_PROFILE_SCOPE(KernelType::GEMM_Q8);
 *       // ... kernel work, automatically timed until scope exit ...
 *   }
 */
#define KERNEL_PROFILE_SCOPE(kernel_type) \
    ::llaminar2::ScopedKernelTimer _kernel_timer_##__LINE__(kernel_type)

/**
 * @brief Manual kernel profiling begin
 *
 * Usage:
 *   KERNEL_PROFILE_BEGIN(timer_name);
 *   // ... kernel work ...
 *   KERNEL_PROFILE_END(timer_name, KernelType::GEMM_Q8);
 */
#define KERNEL_PROFILE_BEGIN(timer_name)       \
    ::llaminar2::ManualKernelTimer timer_name; \
    timer_name.begin()

#define KERNEL_PROFILE_END(timer_name, kernel_type) \
    timer_name.end(kernel_type)
