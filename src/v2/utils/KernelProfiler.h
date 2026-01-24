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
#include <string>
#include <sstream>
#include <iomanip>

#include "DebugEnv.h"

namespace llaminar2
{

    /**
     * @brief Kernel type categories for profiling
     */
    enum class KernelType : uint8_t
    {
        // GEMM variants
        GEMM_FP32 = 0, ///< FP32 GEMM (OpenBLAS fallback)
        GEMM_Q8,       ///< Q8_1 quantized GEMM (JIT microkernel)
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
        case KernelType::GEMM_Q8:
            return "GEMM_Q8";
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

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2);

            oss << "\n";
            oss << "╔══════════════════════════════════════════════════════════════════════════════════════════════╗\n";
            oss << "║                              TENSOR TRANSFER SUMMARY                                         ║\n";
            oss << "╠══════════════════════════════════════════════════════════════════════════════════════════════╣\n";

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

            // Per-stage breakdown
            {
                std::lock_guard<std::mutex> lock(inst.stage_mutex_);
                if (!inst.stage_stats_.empty())
                {
                    oss << "║  STAGE BREAKDOWN (H2D Upload / D2H Download):                                                ║\n";
                    oss << "╠══════════════════════════════════════════════════════════════════════════════════════════════╣\n";
                    oss << "║  Stage Name                        │  H2D Count │   H2D Bytes   │  D2H Count │   D2H Bytes  ║\n";
                    oss << "╠══════════════════════════════════════════════════════════════════════════════════════════════╣\n";

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

                        oss << "║  " << std::left << std::setw(34) << display_name
                            << "│" << std::right << std::setw(10) << sh2d_count << "  "
                            << "│" << std::setw(13) << formatBytes(sh2d_bytes) << "  "
                            << "│" << std::setw(10) << sd2h_count << "  "
                            << "│" << std::setw(12) << formatBytes(sd2h_bytes) << " ║\n";
                    }

                    oss << "╠══════════════════════════════════════════════════════════════════════════════════════════════╣\n";
                }
            }

            // Global totals
            oss << "║  TOTALS:                                                                                     ║\n";
            oss << "╠══════════════════════════════════════════════════════════════════════════════════════════════╣\n";
            oss << "║  Direction        │  Transfers │  Total Bytes   │  Avg Size      │  Bandwidth               ║\n";
            oss << "╠══════════════════════════════════════════════════════════════════════════════════════════════╣\n";

            // H2D row
            if (h2d_count > 0)
            {
                double avg_bytes = static_cast<double>(h2d_bytes) / h2d_count;
                double bandwidth_gbps = (h2d_ns > 0) ? (h2d_bytes / 1e9) / (h2d_ns / 1e9) : 0.0;

                oss << "║  " << std::left << std::setw(16) << "H2D (Upload)"
                    << "│" << std::right << std::setw(10) << h2d_count << "  "
                    << "│" << std::setw(14) << formatBytes(h2d_bytes) << "  "
                    << "│" << std::setw(14) << formatBytes(static_cast<uint64_t>(avg_bytes)) << "  "
                    << "│" << std::setw(8) << (h2d_ns > 0 ? bandwidth_gbps : 0.0) << " GB/s           ║\n";
            }

            // D2H row
            if (d2h_count > 0)
            {
                double avg_bytes = static_cast<double>(d2h_bytes) / d2h_count;
                double bandwidth_gbps = (d2h_ns > 0) ? (d2h_bytes / 1e9) / (d2h_ns / 1e9) : 0.0;

                oss << "║  " << std::left << std::setw(16) << "D2H (Download)"
                    << "│" << std::right << std::setw(10) << d2h_count << "  "
                    << "│" << std::setw(14) << formatBytes(d2h_bytes) << "  "
                    << "│" << std::setw(14) << formatBytes(static_cast<uint64_t>(avg_bytes)) << "  "
                    << "│" << std::setw(8) << (d2h_ns > 0 ? bandwidth_gbps : 0.0) << " GB/s           ║\n";
            }

            oss << "╚══════════════════════════════════════════════════════════════════════════════════════════════╝\n";

            return oss.str();
        }

        /**
         * @brief Print summary to stderr
         */
        static void printSummary()
        {
            std::string summary = getSummary();
            if (!summary.empty())
            {
                fprintf(stderr, "%s", summary.c_str());
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
     * @brief Thread-safe kernel profiling accumulator
     *
     * Uses atomic operations for thread-safe accumulation without locks.
     * Each kernel type has its own accumulator to avoid false sharing.
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
        };

        /**
         * @brief Check if profiling is enabled (from DebugEnv)
         */
        static bool isEnabled()
        {
            // Use DebugEnv for centralized configuration
            return debugEnv().profile.enabled;
        }

        /**
         * @brief Record a kernel execution time
         * @param type Kernel type
         * @param duration_ns Duration in nanoseconds
         */
        static void record(KernelType type, uint64_t duration_ns)
        {
            if (!isEnabled())
                return;

            auto &stats = getStats(type);
            stats.total_ns.fetch_add(duration_ns, std::memory_order_relaxed);
            stats.call_count.fetch_add(1, std::memory_order_relaxed);

            // Update max (lock-free)
            uint64_t current_max = stats.max_ns.load(std::memory_order_relaxed);
            while (duration_ns > current_max &&
                   !stats.max_ns.compare_exchange_weak(current_max, duration_ns,
                                                       std::memory_order_relaxed,
                                                       std::memory_order_relaxed))
            {
            }

            // Update min (lock-free)
            uint64_t current_min = stats.min_ns.load(std::memory_order_relaxed);
            while (duration_ns < current_min &&
                   !stats.min_ns.compare_exchange_weak(current_min, duration_ns,
                                                       std::memory_order_relaxed,
                                                       std::memory_order_relaxed))
            {
            }
        }

        /**
         * @brief Get stats for a kernel type (for external access)
         */
        static KernelStats &getStats(KernelType type)
        {
            return getInstance().stats_[static_cast<size_t>(type)];
        }

        /**
         * @brief Reset all statistics
         */
        static void reset()
        {
            auto &inst = getInstance();
            for (auto &stats : inst.stats_)
            {
                stats.total_ns.store(0, std::memory_order_relaxed);
                stats.call_count.store(0, std::memory_order_relaxed);
                stats.max_ns.store(0, std::memory_order_relaxed);
                stats.min_ns.store(UINT64_MAX, std::memory_order_relaxed);
            }
        }

        /**
         * @brief Print formatted summary of all kernel timings
         * @param total_tokens Total tokens processed (for tok/s calculation)
         * @return Formatted string with timing breakdown
         */
        static std::string getSummary(uint64_t total_tokens = 0)
        {
            if (!isEnabled())
            {
                return "[Kernel profiling disabled. Set LLAMINAR_PROFILE_KERNELS=1 to enable]\n";
            }

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2);

            oss << "\n";
            oss << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
            oss << "║                         KERNEL PROFILING SUMMARY                             ║\n";
            oss << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
            oss << "║  Kernel Type      │  Calls   │  Total (ms)  │  Avg (µs)  │  Min/Max (µs)     ║\n";
            oss << "╠══════════════════════════════════════════════════════════════════════════════╣\n";

            uint64_t grand_total_ns = 0;

            for (size_t i = 0; i < static_cast<size_t>(KernelType::COUNT); ++i)
            {
                const auto &stats = getStats(static_cast<KernelType>(i));
                uint64_t count = stats.call_count.load(std::memory_order_relaxed);
                if (count == 0)
                    continue;

                uint64_t total_ns = stats.total_ns.load(std::memory_order_relaxed);
                uint64_t min_ns = stats.min_ns.load(std::memory_order_relaxed);
                uint64_t max_ns = stats.max_ns.load(std::memory_order_relaxed);

                double total_ms = total_ns / 1e6;
                double avg_us = (total_ns / static_cast<double>(count)) / 1e3;
                double min_us = min_ns / 1e3;
                double max_us = max_ns / 1e3;

                grand_total_ns += total_ns;

                // Format: kernel name (16 chars), calls (8), total ms (12), avg µs (10), min/max
                oss << "║  " << std::left << std::setw(16) << kernelTypeName(static_cast<KernelType>(i))
                    << "│" << std::right << std::setw(8) << count << "  "
                    << "│" << std::setw(12) << total_ms << "  "
                    << "│" << std::setw(10) << avg_us << "  "
                    << "│" << std::setw(8) << min_us << "/" << std::setw(8) << max_us << " ║\n";
            }

            oss << "╠══════════════════════════════════════════════════════════════════════════════╣\n";

            double grand_total_ms = grand_total_ns / 1e6;
            oss << "║  TOTAL KERNEL TIME: " << std::setw(10) << grand_total_ms << " ms";

            if (total_tokens > 0)
            {
                double toks_per_sec = (total_tokens * 1e9) / static_cast<double>(grand_total_ns);
                oss << "   (" << std::setw(6) << toks_per_sec << " kernel tok/s)";
            }
            oss << std::setw(20) << " " << "║\n";

            oss << "╚══════════════════════════════════════════════════════════════════════════════╝\n";

            // Append transfer stats if any transfers occurred
            oss << TransferProfiler::getSummary();

            return oss.str();
        }

        /**
         * @brief Print summary to stderr
         */
        static void printSummary(uint64_t total_tokens = 0)
        {
            fprintf(stderr, "%s", getSummary(total_tokens).c_str());
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

        std::array<KernelStats, static_cast<size_t>(KernelType::COUNT)> stats_;
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
