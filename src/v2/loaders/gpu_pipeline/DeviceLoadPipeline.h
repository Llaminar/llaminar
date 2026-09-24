/**
 * @file DeviceLoadPipeline.h
 * @brief Contracts for bounded host-to-device weight staging and GPU repacking.
 *
 * This file also owns the source-range coalescing contract used to turn
 * individually addressed MoE expert views into large, contiguous preparation
 * jobs without transferring ownership away from the model tensor graph.
 */

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

        /**
         * Chunk coordinates for bounded staging.
         *
         * Every chunk contains complete, contiguous source rows. Quantized
         * repack kernels use full_N and row_offset when writing the final
         * block-major destination, so bounded loading never gathers K slices
         * from hundreds of thousands of source rows on the host.
         */
        int row_offset = 0;
        int full_N = 0;
        int full_K = 0;

        /**
         * Rows in one independently addressable packed matrix.
         *
         * Zero selects the ordinary block-major `[K-block][full_N row]`
         * layout. A positive value selects grouped block-major storage:
         * `[group][K-block][group row]`. Coalesced MoE parent runs set this to
         * one expert's row count, preserving every expert's standalone byte
         * layout while allowing one source upload and one high-occupancy repack
         * launch. Row chunks may cross group boundaries because kernels derive
         * group identity from the absolute row offset.
         */
        int packed_group_rows = 0;

        /**
         * Payload capacity reserved for each logical 32-value block.
         *
         * Zero asks the pipeline to infer the compact width from the allocated
         * slot. ExpertOverlay jobs set this explicitly to the union of their
         * initial and CPU-promotion layouts. Within each `packed_group_rows`
         * matrix the live payload remains tightly packed; only group starts
         * advance by this capacity.
         */
        int packed_payload_capacity_bytes_per_block = 0;

    };

    /**
     * @brief Maps one logical source matrix into a coalesced storage run.
     *
     * MoE graphs address experts individually, but GGUF stores every expert of
     * one layer/role parent contiguously. The loader therefore records the
     * original logical job index and the first packed row assigned to it inside
     * a larger run. Registration uses this row offset to construct an
     * individual GEMM engine without restoring one upload/repack transaction per
     * expert.
     */
    struct CoalescedWeightJobMember
    {
        size_t source_job_index = 0; ///< Index in the caller's logical job vector.
        int row_offset = 0;          ///< First row in the coalesced packed matrix.
    };

    /**
     * @brief One contiguous source range that can be staged and repacked as a unit.
     */
    struct CoalescedWeightJobRun
    {
        WeightJob job;                                 ///< Aggregate matrix submitted to the pipeline.
        std::vector<CoalescedWeightJobMember> members; ///< Logical matrices represented by the run.
    };

    /**
     * @brief Coalesce adjacent logical matrices from one immutable parent.
     *
     * Two jobs are combined only when they have the same non-null source identity,
     * format, N/K geometry, metadata layout, bytes-per-row, and adjacent source
     * ranges. A source gap, a different parent, or any format difference starts
     * a new run. This makes arbitrary expert subsets safe while reducing a fully
     * resident MoE layer from hundreds of tiny transactions to one bounded,
     * row-chunked transaction per contiguous parent range.
     *
     * Input jobs may be in graph discovery order. The helper orders them by
     * identity and source address while retaining original indices in the member
     * map. Pre-chunked jobs are rejected because coalescing is a planning
     * operation that must happen before LoadOrchestrator applies its staging
     * budget.
     *
     * @param jobs Logical whole-matrix source jobs.
     * @param source_identities Immutable, non-owning parent identity for each job.
     *        The pointed-to object is never dereferenced and need not own the
     *        source bytes; callers retain the actual source lifetime separately.
     * @return Coalesced runs and exact logical-to-packed row mappings.
     * @throws std::invalid_argument for malformed or mismatched input vectors.
     * @throws std::overflow_error when aggregate rows or bytes exceed their types.
     */
    std::vector<CoalescedWeightJobRun> coalesceContiguousWeightJobs(
        const std::vector<WeightJob> &jobs,
        const std::vector<const void *> &source_identities);

    /**
     * @brief Order upload jobs by source address to preserve GGUF read locality.
     *
     * Model weights are commonly discovered through unordered caches. The tensor
     * data still points into one monotonically laid-out GGUF mmap, so discovery
     * order otherwise turns a cold load into backward and forward page faults even
     * though the mapping carries a sequential-access hint. This helper restores
     * file order without changing destination slots or packed-layout coordinates.
     *
     * A stable sort is intentional. Aliases such as tied embeddings can share the
     * same source address, and bounded row chunks occupy adjacent ranges; retaining
     * their original relative order keeps publication deterministic.
     *
     * @param jobs Mutable upload job list to order in place.
     * @return Number of backward source-address transitions before sorting.
     */
    size_t orderWeightJobsForSequentialHostAccess(std::vector<WeightJob> &jobs);

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
