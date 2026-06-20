#pragma once

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "IProgressPublisher.h"

namespace llaminar2
{

    /**
     * @brief Renders ASCII progress bars for weight loading across devices.
     *
     * Thread-safe: multiple device loading threads can update concurrently.
     * Renders to stderr using ANSI cursor control to overwrite previous output.
     *
     * Multi-rank behaviour:
     *   - Single-rank (world_size=1): Renders directly (no MPI overhead).
     *   - Multi-rank: Only rank 0 renders. Remote ranks' progress is pulled via
     *     a WeightLoadProgressAggregator (MPI RMA window) and displayed alongside
     *     rank 0's local devices.
     *
     * Device labels include rank for disambiguation in multi-rank setups:
     *   Loading model weights...
     *   0:rocm:0   [################                        ]  42%   3.2 GB/s
     *   1:rocm:1   [########                                ]  21%   4.8 GB/s
     *   1:cpu:0    [########################################] 100%   5.1 GB/s  done
     */
    class WeightLoadProgress
    {
    public:
        static constexpr int BAR_WIDTH = 40;

        /// Callback fired per-weight as bytes are loaded.
        /// Args: (bytes_loaded_so_far, total_bytes_for_this_device)
        using ProgressCallback = std::function<void(size_t bytes_loaded, size_t total_bytes)>;

        struct DeviceState
        {
            std::string label;
            size_t bytes_loaded = 0;
            size_t total_bytes = 0;
            std::chrono::steady_clock::time_point start_time;
            bool finished = false;
            double final_elapsed_s = 0.0;
            int aggregator_idx = -1; // Index in aggregator's window slot (-1 if no aggregator)
        };

        /// @param rank         MPI rank of this process (0 = primary renderer)
        /// @param world_size   Total MPI ranks (1 = single process)
        explicit WeightLoadProgress(int rank = 0, int world_size = 1)
            : rank_(rank), world_size_(world_size), enabled_(rank == 0)
        {
        }

        /// Whether this instance will actually render (only rank 0).
        bool isEnabled() const { return enabled_; }

        /// Build a rank-qualified device label: "0:rocm:0", "1:cpu:0", etc.
        /// For single-rank (world_size=1), omits the rank prefix for brevity.
        std::string makeDeviceLabel(const std::string &device_str) const
        {
            if (world_size_ <= 1)
                return device_str;
            return std::to_string(rank_) + ":" + device_str;
        }

        /// Register a device before loading starts. Returns device index.
        /// Safe to call on non-rendering ranks (returns valid index, no I/O).
        /// If an aggregator is attached, publishes to MPI window for remote visibility.
        int registerDevice(const std::string &label, size_t total_bytes)
        {
            int agg_idx = -1;
            if (aggregator_)
                agg_idx = aggregator_->publishDevice(label, total_bytes);

            std::lock_guard<std::mutex> lock(mu_);
            int idx = static_cast<int>(devices_.size());
            DeviceState state;
            state.label = label;
            state.total_bytes = total_bytes;
            state.start_time = std::chrono::steady_clock::now();
            state.aggregator_idx = agg_idx;
            devices_.push_back(state);
            if (enabled_)
            {
                if (!header_printed_)
                {
                    fprintf(stderr, "\nLoading model weights...\n");
                    header_printed_ = true;
                }
                render_locked();
            }
            return idx;
        }

        /// Update bytes loaded for a device. Thread-safe. Rate-limited to ~20 FPS.
        /// Non-rendering ranks publish to aggregator window (zero MPI call overhead).
        void update(int device_idx, size_t bytes_loaded)
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (device_idx < 0 || device_idx >= static_cast<int>(devices_.size()))
                return;

            // Publish to aggregator window (all ranks, including non-rendering)
            if (aggregator_ && devices_[device_idx].aggregator_idx >= 0)
            {
                aggregator_->publishProgress(devices_[device_idx].aggregator_idx, bytes_loaded);
            }

            if (!enabled_)
                return;

            devices_[device_idx].bytes_loaded = bytes_loaded;

            // Throttle rendering to at most every 50ms
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double, std::milli>(now - last_render_time_).count() < 50.0)
                return;
            last_render_time_ = now;
            render_locked();
        }

        /// Mark device as finished.
        void finish(int device_idx)
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (device_idx < 0 || device_idx >= static_cast<int>(devices_.size()))
                return;

            // Publish to aggregator window
            if (aggregator_ && devices_[device_idx].aggregator_idx >= 0)
            {
                aggregator_->publishFinished(devices_[device_idx].aggregator_idx);
            }

            auto &dev = devices_[device_idx];
            dev.bytes_loaded = dev.total_bytes;
            dev.finished = true;
            dev.final_elapsed_s = std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - dev.start_time)
                                      .count();
            if (enabled_)
                render_locked();
        }

        /// Print final summary after all devices done. No-op on non-rendering ranks.
        void finalize()
        {
            if (!enabled_)
                return;
            std::lock_guard<std::mutex> lock(mu_);
            // Move cursor below the progress area
            fprintf(stderr, "\n");
            for (const auto &dev : devices_)
            {
                double elapsed = dev.final_elapsed_s > 0 ? dev.final_elapsed_s
                                                         : std::chrono::duration<double>(
                                                               std::chrono::steady_clock::now() - dev.start_time)
                                                               .count();
                double gb = static_cast<double>(dev.total_bytes) / (1024.0 * 1024.0 * 1024.0);
                double gbps = elapsed > 0 ? gb / elapsed : 0;
                fprintf(stderr, "  %-14s loaded %.1f GB in %.1fs (%.1f GB/s)\n",
                        dev.label.c_str(), gb, elapsed, gbps);
            }
            fprintf(stderr, "\n");
            fflush(stderr);
        }

        /// Create a ProgressCallback bound to a registered device index.
        /// This is the primary interface for callers to get a callback for their pipeline.
        /// Returns a valid callback if either rendering is enabled OR an aggregator is attached.
        ProgressCallback makeCallback(int device_idx)
        {
            if (device_idx < 0)
                return nullptr;
            // Need a callback if we're rendering OR publishing to aggregator
            if (!enabled_ && !aggregator_)
                return nullptr;
            return [this, device_idx](size_t bytes_loaded, size_t /*total_bytes*/)
            {
                update(device_idx, bytes_loaded);
            };
        }

        /// Attach an aggregator for cross-rank progress collection.
        /// When attached, registerDevice/update/finish also publish to the MPI window.
        void setAggregator(std::shared_ptr<IProgressPublisher> agg)
        {
            aggregator_ = std::move(agg);
        }

        std::shared_ptr<IProgressPublisher> aggregator() const
        {
            return aggregator_;
        }

    private:
        void render_locked()
        {
            // Move cursor up to overwrite previous render
            if (lines_rendered_ > 0)
            {
                fprintf(stderr, "\033[%dA", lines_rendered_);
            }

            lines_rendered_ = 0;
            for (const auto &dev : devices_)
            {
                double pct = dev.total_bytes > 0
                                 ? static_cast<double>(dev.bytes_loaded) / static_cast<double>(dev.total_bytes)
                                 : 0.0;
                if (pct > 1.0)
                    pct = 1.0;

                int filled = static_cast<int>(pct * BAR_WIDTH);
                double elapsed = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - dev.start_time)
                                     .count();
                double gb_loaded = static_cast<double>(dev.bytes_loaded) / (1024.0 * 1024.0 * 1024.0);
                double gbps = elapsed > 0.01 ? gb_loaded / elapsed : 0;

                // Build bar
                char bar[BAR_WIDTH + 1];
                for (int i = 0; i < BAR_WIDTH; ++i)
                    bar[i] = (i < filled) ? '#' : ' ';
                bar[BAR_WIDTH] = '\0';

                if (dev.finished)
                {
                    fprintf(stderr, "  %-14s [%s] %3d%%  %4.1f GB/s  done\033[K\n",
                            dev.label.c_str(), bar, static_cast<int>(pct * 100), gbps);
                }
                else
                {
                    fprintf(stderr, "  %-14s [%s] %3d%%  %4.1f GB/s\033[K\n",
                            dev.label.c_str(), bar, static_cast<int>(pct * 100), gbps);
                }
                ++lines_rendered_;
            }
            fflush(stderr);
        }

        int rank_ = 0;
        int world_size_ = 1;
        bool enabled_ = true;
        std::mutex mu_;
        std::vector<DeviceState> devices_;
        int lines_rendered_ = 0;
        bool header_printed_ = false;
        std::chrono::steady_clock::time_point last_render_time_;
        std::shared_ptr<IProgressPublisher> aggregator_;
    };

} // namespace llaminar2
