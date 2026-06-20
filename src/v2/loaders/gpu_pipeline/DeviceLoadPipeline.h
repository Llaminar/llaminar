#pragma once

#include "loaders/gpu_pipeline/RepackFormat.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace llaminar2
{

    class IBackend;
    class WeightVRAMPool;
    class PinnedRingBuffer;

    /// Describes a single weight to be uploaded and repacked on GPU.
    struct WeightJob
    {
        std::string name;          ///< Weight name (must match a planned weight in the pool)
        const void *host_raw_data; ///< mmap'd raw GGUF block data on host
        size_t raw_bytes;          ///< Raw GGUF byte count
        RepackFormat format;       ///< GPU repack kernel format dispatch
        int N;                     ///< Number of output features (rows in weight matrix)
        int K;                     ///< Number of input features (columns)
        bool is_asymmetric;        ///< True if format has mins (Q4_K etc.)
    };

    /// Per-device pipelined H2D transfer + GPU repack engine.
    ///
    /// Uses N H2D streams for overlapped transfers and 1 repack stream for
    /// GPU-side VNNI repacking. Each stream has a paired pinned host slot
    /// and device staging slot.
    ///
    /// Backend-agnostic: works with both CUDA and ROCm via IBackend.
    class DeviceLoadPipeline
    {
    public:
        /// @param backend         GPU backend (CUDA or ROCm)
        /// @param device_id       Device ordinal
        /// @param pool            Pre-allocated VRAM pool (must have staging slots)
        /// @param pinned           Pinned host ring buffer (num_slots >= num_h2d_streams)
        /// @param kernels          Backend-specific repack kernel function pointers
        /// @param num_h2d_streams  Number of concurrent H2D transfer streams (default 3)
        DeviceLoadPipeline(IBackend &backend,
                           int device_id,
                           WeightVRAMPool &pool,
                           PinnedRingBuffer &pinned,
                           const RepackKernels &kernels,
                           int num_h2d_streams = 3);
        ~DeviceLoadPipeline();

        DeviceLoadPipeline(const DeviceLoadPipeline &) = delete;
        DeviceLoadPipeline &operator=(const DeviceLoadPipeline &) = delete;

        /// Initialize streams and events via IBackend. Returns false on error.
        bool initialize();

        /// Callback fired after each weight is staged (memcpy to pinned buffer complete).
        /// Args: (bytes_loaded_so_far, total_bytes_for_all_jobs)
        using ProgressCallback = std::function<void(size_t bytes_loaded, size_t total_bytes)>;

        /// Process all weight jobs through the pipeline.
        /// Blocks until all H2D transfers and repack kernels complete.
        /// @param progress_cb  Optional per-job progress callback (fired after host memcpy)
        /// @return true if all weights were successfully processed
        bool processJobs(const std::vector<WeightJob> &jobs,
                         ProgressCallback progress_cb = nullptr);

        /// Release backend resources (streams, events).
        void release();

        /// Number of weights successfully processed in last processJobs() call.
        size_t numProcessed() const { return num_processed_; }

    private:
        IBackend &backend_;
        int device_id_;
        WeightVRAMPool &pool_;
        PinnedRingBuffer &pinned_;
        RepackKernels kernels_;
        int num_streams_;
        bool initialized_ = false;
        size_t num_processed_ = 0;

        // Backend resources (void* for header compatibility)
        std::vector<void *> h2d_streams_;        // Stream per H2D channel
        void *repack_stream_ = nullptr;          // Stream for repack kernels
        std::vector<void *> h2d_done_events_;    // Event per stream
        std::vector<void *> repack_done_events_; // Event per stream (staging slot reuse)
    };

} // namespace llaminar2
