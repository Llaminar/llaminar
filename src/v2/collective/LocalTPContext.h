/**
 * @file LocalTPContext.h
 * @brief Implementation of LOCAL tensor parallelism context
 *
 * Provides concrete implementation of ILocalTPContext for managing
 * tensor parallelism across multiple devices within a single MPI rank.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "ILocalTPContext.h"
#include "DeviceGroup.h"
#include "ICollectiveBackend.h"
#include "../planning/PhysicalMemoryAuthority.h"
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace llaminar2
{

    /**
     * @brief Concrete implementation of LOCAL tensor parallelism context
     *
     * Manages device list, weight distribution, and collective operations
     * for LOCAL TP (multiple devices within a single MPI rank).
     *
     * Thread safety: All public methods are thread-safe.
     */
    class LocalTPContext : public ILocalTPContext
    {
    public:
        /**
         * @brief Construct a LocalTPContext
         *
         * @param devices Devices participating in LOCAL TP (must be non-empty)
         * @param weights Work distribution weights (empty for equal distribution)
         * @param backend Backend type for collectives (AUTO to detect from devices)
         * @throws std::invalid_argument if devices is empty or weights mismatch
         */
        LocalTPContext(
            std::vector<GlobalDeviceAddress> devices,
            std::vector<float> weights,
            CollectiveBackendType backend);

        ~LocalTPContext() override;

        // Disable copy (has mutex)
        LocalTPContext(const LocalTPContext &) = delete;
        LocalTPContext &operator=(const LocalTPContext &) = delete;

        // Enable move
        LocalTPContext(LocalTPContext &&) = default;
        LocalTPContext &operator=(LocalTPContext &&) = default;

        // =====================================================================
        // Configuration (ILocalTPContext)
        // =====================================================================

        const std::vector<GlobalDeviceAddress> &devices() const override;
        const std::vector<float> &weights() const override;
        CollectiveBackendType backend() const override;
        int degree() const override;

        /**
         * @brief Get the current device index for orchestrator-driven LOCAL TP
         *
         * In LOCAL TP, a single orchestrator thread calls collective operations on behalf
         * of multiple devices. This method returns the device index that was set via
         * setCurrentDeviceIndex().
         *
         * For stages that need to know "which device am I?", call setCurrentDeviceIndex()
         * before calling methods that use myIndex() for sharding calculations.
         *
         * @return Current device index (0 to degree-1)
         * @throws std::runtime_error if setCurrentDeviceIndex() was never called
         */
        int myIndex() const override;

        /**
         * @brief Set the current device index for orchestrator-driven LOCAL TP
         *
         * Called by the orchestrator before invoking operations that need to know
         * which device they're operating on behalf of.
         *
         * @param index Device index (0 to degree-1)
         * @throws std::out_of_range if index >= degree()
         */
        void setCurrentDeviceIndex(int index);

        // =====================================================================
        // Collective Operations (ILocalTPContext)
        // =====================================================================

        bool allreduce(TensorBase *tensor) override;
        bool allreduce(TensorBase *tensor, const std::string &stage_name, size_t count = 0) override;
        bool allreduceOnStream(TensorBase *tensor, const std::string &stage_name,
                               size_t count, void *stream,
                               const std::string &precision = "") override;
        bool allreduce(const TensorBase *input, TensorBase *output) override;
        bool allgather(const TensorBase *local_shard, TensorBase *global_tensor) override;
        bool allgatherRawOnStream(
            const void *local_send,
            void *full_recv,
            size_t send_count,
            CollectiveDataType dtype,
            int device_index,
            void *producer_stream,
            const std::string &stage_name) override;
        bool reduceRawOnStream(
            const void *local_send,
            void *root_recv,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op,
            int root_device_index,
            int device_index,
            void *producer_stream,
            const std::string &stage_name) override;
        bool broadcastRawOnStream(
            const void *root_send,
            void *local_recv,
            size_t count,
            CollectiveDataType dtype,
            int root_device_index,
            int device_index,
            void *producer_stream,
            const std::string &stage_name) override;
        bool groupedP2PRawOnStream(
            const std::vector<CollectiveP2POp> &ops,
            int device_index,
            void *producer_stream,
            const std::string &stage_name) override;
        bool supportsRawAllgatherOnStreamGraphCapture() const override;

        /**
         * @brief Coordinate every LocalTP participant at a named GPU graph-capture boundary.
         *
         * See ILocalTPContext::graphCaptureBoundaryRendezvous() for the lifecycle
         * contract. The concrete implementation is cyclic and may be reused for
         * successive prefill chunks and capture phases.
         */
        bool graphCaptureBoundaryRendezvous(
            const std::string &boundary_name,
            int device_index,
            int timeout_ms) override;

        /**
         * @brief Establish the backend-specific stream fence for a capture boundary.
         *
         * Homogeneous domains use a one-word native collective. Mixed CUDA/ROCm
         * domains use HeterogeneousBackend's persistent event-ticket protocol.
         * Both paths are generation checked and bounded by @p timeout_ms.
         *
         * @param boundary_name Stable lifecycle generation identity.
         * @param device_index Calling participant slot.
         * @param stream Exact stream entering or leaving native capture.
         * @param timeout_ms Maximum peer/ticket observation interval.
         * @return true only after every participant reaches the same safe point.
         */
        bool graphCaptureBoundaryOnStream(
            const std::string &boundary_name,
            int device_index,
            void *stream,
            int timeout_ms) override;
        bool collectiveSidebandOnStream(
            const std::vector<LocalTPCollectiveSidebandBuffer> &sidebands,
            int device_index,
            void *producer_stream,
            const std::string &anchor_stage_name) override;
        bool collectiveSidebandSpanOnStream(
            std::span<const LocalTPCollectiveSidebandBuffer> sidebands,
            int device_index,
            void *producer_stream,
            const std::string &anchor_stage_name) override;
        bool collectiveSidebandsMultiOnStreams(
            const std::vector<std::vector<LocalTPCollectiveSidebandBuffer>>
                &participant_sidebands,
            const std::vector<void *> &producer_streams,
            const std::string &publication_name) override;
        bool allreduceWithSidebandsOnStream(
            TensorBase *tensor,
            const std::string &stage_name,
            size_t count,
            void *producer_stream,
            const std::string &precision,
            const std::vector<LocalTPCollectiveSidebandBuffer> &sidebands,
            int device_index) override;
        bool supportsCollectiveSidebandOnStreamGraphCapture() const override;

        void setBackendForTesting(
            std::unique_ptr<ICollectiveBackend> backend,
            CollectiveBackendType backend_type,
            bool initialized);
        bool allreduceGroupedOnExplicitStreamsForTesting(
            void *buffer,
            size_t effective_count,
            CollectiveDataType dtype,
            int device_index,
            void *stream,
            const std::string &stage_name,
            const std::string &precision)
        {
            return allreduceGroupedOnExplicitStreams(
                buffer, effective_count, dtype, device_index, stream, stage_name, precision);
        }
        bool gatherFromDevices(
            const std::vector<const TensorBase *> &shards,
            TensorBase *output) override;
        bool reduceScatter(const TensorBase *input, TensorBase *output_shard) override;
        bool broadcast(TensorBase *tensor, int source_device_index = 0) override;

        // =====================================================================
        // Synchronization (ILocalTPContext)
        // =====================================================================

        void synchronize() override;

        // =====================================================================
        // Stream Configuration
        // =====================================================================

        void setComputeStreams(const std::vector<void *> &compute_streams) override;

        /**
         * @brief Return true when the current GPU graph configuration supports
         *        LocalTP GPU-native collectives for a backend.
         *
         * @param backend Backend whose GPU graph policy should be checked.
         * @param reason_out Optional pointer receiving human-readable reason.
         * @return true when policy permits LocalTP collective execution.
         */
        static bool isLocalTPGpuGraphPolicySupported(
            CollectiveBackendType backend,
            std::string *reason_out = nullptr);

        // =====================================================================
        // Output Tensor Registry (ILocalTPContext interface + concrete impl)
        // =====================================================================

        /**
         * @brief Register an output tensor for a stage (interface impl)
         *
         * Called during graph construction for row-parallel stages (FFN_DOWN, Wo)
         * to register output tensors for collective operations.
         *
         * @param stage_name Stage identifier (e.g., "layer0_ffn_down_allreduce")
         * @param device Device that owns this tensor (must be in devices())
         * @param tensor Tensor to register (must be FP32)
         */
        void registerBARBackedOutput(
            const std::string &stage_name,
            const GlobalDeviceAddress &device,
            TensorBase *tensor) override;

        /**
         * @brief Check if a stage has any registered outputs
         *
         * @param stage_name Stage identifier
         * @return true if at least one device has a tensor registered
         */
        bool hasBARBackedOutputs(const std::string &stage_name) const override;

        /**
         * @brief Clear all registered output tensor registrations
         *
         * Called when resetting the context or changing buffer sizes.
         */
        void clearBARBackedOutputs() override;

        /**
         * @brief Reserve backend workspace and per-device FP16 transport scratch.
         *
         * This method is initialization-only. Collective execution never grows
         * the reservation and fails hard when graph planning underestimates it.
         *
         * @param backend_payload_capacity_bytes Maximum logical backend payload.
         * @param fp16_scratch_elements Maximum FP16 transport element count.
         * @param memory_authority Sole rank-local allocation ledger.
         * @return true only when all backend and participant allocations succeed.
         */
        bool reserveCollectiveResources(
            size_t backend_payload_capacity_bytes,
            size_t fp16_scratch_elements,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory_authority) override;

        /**
         * @brief Get all registered tensors for a stage (concrete implementation)
         *
         * Returns tensors in device order (index i = tensor for devices()[i]).
         * May contain nullptr entries for devices without registered outputs.
         *
         * @param stage_name Stage identifier
         * @return Vector of FP32Tensor pointers (size = degree()), nullptr for missing entries
         */
        std::vector<FP32Tensor *> getBARBackedOutputs(const std::string &stage_name) const;

        // =====================================================================
        // Device Management (ILocalTPContext)
        // =====================================================================

        int indexForDevice(const GlobalDeviceAddress &device) const override;
        const GlobalDeviceAddress &deviceAt(int index) const override;
        float weightForDevice(const GlobalDeviceAddress &device) const override;

        // =====================================================================
        // Weight Sharding Utilities (ILocalTPContext)
        // =====================================================================

        int headsForDevice(const GlobalDeviceAddress &device, int total_heads) const override;
        std::pair<int, int> rowRangeForDevice(
            const GlobalDeviceAddress &device, int total_rows) const override;
        std::pair<int, int> colRangeForDevice(
            const GlobalDeviceAddress &device, int total_cols) const override;

        // =====================================================================
        // Abort (for one-sided failure recovery)
        // =====================================================================

        /**
         * @brief Publish fatal cancellation without destroying graph-owned communicators.
         *
         * The first caller closes collective admission and wakes every LocalTP
         * rendezvous. NCCL/RCCL destruction is deliberately deferred to this
         * context's destruction boundary because native graph executables may
         * retain communicator references until their owning device runners are
         * destroyed.
         */
        void requestAbort();

        /**
         * @brief Check if abort has been requested by any device thread.
         * @return true if requestAbort() was called
         */
        bool isAbortRequested() const { return abort_requested_.load(std::memory_order_acquire); }

    private:
        /**
         * @brief Abort the collective backend after all external graph owners are gone.
         *
         * `RankOrchestrator` declares its LocalTP context before its worker pool
         * and device runners. C++ reverse member destruction therefore retires
         * workers and native graph executables before this context is destroyed.
         * Keeping the backend abort private makes the unsafe inverse ordering
         * impossible through the public LocalTP interface.
         */
        void abortBackendAfterGraphOwnersReleased() noexcept;

        struct OnStreamCollectiveContract
        {
            bool initialized = false;
            std::string stage_name;
            size_t count = 0;
            int dtype = -1;
            std::string precision;
            uint64_t seen_slots = 0;
            int arrivals = 0;
            int departures = 0;
            bool ok = true;
            std::string error;
        };

        static std::atomic<uint64_t> next_context_id_;
        uint64_t context_id_ = 0;

        std::vector<GlobalDeviceAddress> devices_;
        std::vector<float> weights_; ///< Normalized weights (sum to 1.0)
        CollectiveBackendType backend_;
        std::unordered_map<GlobalDeviceAddress, int> device_to_index_;

        /// Current device index for orchestrator-driven operations (-1 = not set)
        int current_device_index_ = -1;

        mutable std::mutex mutex_; ///< Protects collective operations

        /// Backend implementation for collective operations
        std::unique_ptr<ICollectiveBackend> backend_impl_;

        /// Device group for backend initialization
        DeviceGroup device_group_;

        /// Track if backend was successfully initialized
        bool backend_initialized_ = false;

        // =====================================================================
        // Barrier Synchronization State
        // =====================================================================
        // For multi-GPU backends with heterogeneous GPUs, threads from
        // different devices call allreduce() concurrently. We need a rendezvous
        // barrier so all devices have contributed their data before the
        // collective transfer happens.

        /// Mutex for barrier synchronization (separate from mutex_ to avoid deadlock)
        mutable std::mutex barrier_mutex_;

        /// Condition variable for barrier wait/notify
        std::condition_variable barrier_cv_;

        /// Number of threads that have arrived at the barrier
        std::atomic<int> barrier_count_{0};

        /// Generation counter to prevent spurious wakeups and ensure barrier reusability
        std::atomic<uint64_t> barrier_generation_{0};

        /// Tensors being reduced from each device (one per participant)
        /// Key: arrival order (0, 1, ...), Value: tensor pointer
        std::vector<TensorBase *> barrier_tensors_;

        /// Exact producer stream paired with each barrier tensor.
        ///
        /// CPU participants publish nullptr. GPU participants are accepted only
        /// by the deliberately heterogeneous HOST collective and must publish
        /// the non-null stream that produced their tensor.
        std::vector<void *> barrier_producer_streams_;

        /// Tensor being reduced (set by first arrival, used by executor) [DEPRECATED: use barrier_tensors_]
        TensorBase *barrier_tensor_{nullptr};

        /// Stage name for current barrier operation (for registered tensor lookup)
        std::string barrier_stage_name_;

        /// Element count for current barrier operation (0 = use tensor->numel())
        /// CRITICAL: For decode with dynamic seq_len, this must be actual count, not buffer size
        size_t barrier_element_count_{0};

        /// Result of allreduce (set by executor, read by all waiters)
        bool barrier_result_{false};

        /// Optional watched-pointer checksum captured at barrier arrival (per slot)
        std::vector<uint64_t> barrier_watch_checksums_;
        std::vector<size_t> barrier_watch_sample_bytes_;
        std::vector<size_t> barrier_watch_sample_offsets_;
        std::vector<bool> barrier_watch_checksum_valid_;

        // =====================================================================
        // NCCL Telemetry
        // =====================================================================
        std::atomic<uint64_t> nccl_allreduce_attempts_{0};
        std::atomic<uint64_t> nccl_allreduce_success_{0};
        std::atomic<uint64_t> nccl_allreduce_failures_{0};
        std::atomic<bool> logged_real_path_marker_{false};
        std::atomic<bool> logged_graph_policy_reject_marker_{false};
        std::atomic<bool> logged_graph_policy_allow_marker_{false};
        std::atomic<bool> abort_requested_{false};

        // =====================================================================
        // On-stream collective rendezvous and contract state. Device worker
        // threads must enqueue the same collective generation together; otherwise
        // NCCL/RCCL can observe asymmetric launch order during first-use setup.
        // =====================================================================
        mutable std::mutex contract_trace_mutex_;
        std::condition_variable contract_trace_cv_;
        std::vector<uint64_t> onstream_sequence_by_slot_;
        std::unordered_map<uint64_t, OnStreamCollectiveContract> onstream_contracts_;

        // Eager homogeneous GPU allreduce rendezvous. Unlike the graph-capture
        // path, eager execution launches one grouped NCCL/RCCL collective over
        // every participant's explicit producer stream from the last arrival.
        mutable std::mutex grouped_onstream_allreduce_mutex_;
        std::condition_variable grouped_onstream_allreduce_cv_;
        uint64_t grouped_onstream_allreduce_generation_{0};
        int grouped_onstream_allreduce_arrivals_{0};
        int grouped_onstream_allreduce_departures_{0};
        bool grouped_onstream_allreduce_ready_{false};
        bool grouped_onstream_allreduce_result_{false};
        bool grouped_onstream_allreduce_ok_{true};
        size_t grouped_onstream_allreduce_count_{0};
        int grouped_onstream_allreduce_dtype_{-1};
        std::string grouped_onstream_allreduce_stage_;
        std::string grouped_onstream_allreduce_precision_;
        std::string grouped_onstream_allreduce_error_;
        std::vector<void *> grouped_onstream_allreduce_buffers_;
        std::vector<void *> grouped_onstream_allreduce_streams_;
        std::vector<bool> grouped_onstream_allreduce_seen_;
        bool grouped_onstream_allreduce_graph_capture_active_{false};
        size_t grouped_onstream_allreduce_sideband_count_{0};
        std::vector<LocalTPCollectiveSidebandBuffer> grouped_onstream_allreduce_reference_sidebands_;
        std::vector<std::vector<LocalTPCollectiveSidebandBuffer>> grouped_onstream_allreduce_sidebands_;

        // =====================================================================
        // GPU graph-capture lifecycle rendezvous.
        // =====================================================================
        mutable std::mutex graph_capture_boundary_mutex_;
        std::condition_variable graph_capture_boundary_cv_;
        uint64_t graph_capture_boundary_generation_{0};
        int graph_capture_boundary_arrivals_{0};
        int graph_capture_boundary_departures_{0};
        bool graph_capture_boundary_ready_{false};
        bool graph_capture_boundary_result_{false};
        std::string graph_capture_boundary_name_;
        std::string graph_capture_boundary_error_;
        std::vector<bool> graph_capture_boundary_seen_;
        /**
         * @brief Persistent device words used only for graph lifecycle fences.
         *
         * One INT32 word is allocated per homogeneous GPU participant by the
         * authoritative setup reservation. Values are immaterial; an in-place
         * allreduce exists solely to establish cross-device stream ordering.
         */
        std::vector<void *> graph_capture_boundary_device_words_;
        /// Physical-ledger claims paired with the graph-boundary words.
        std::vector<std::optional<PhysicalMemoryAllocationLease>>
            graph_capture_boundary_memory_leases_;

        // =====================================================================
        // FP16 Mixed-Precision Allreduce Scratch Buffers
        // =====================================================================
        // When allreduce precision is "fp16" (set per-layer via schema,
        // GraphConfig override, or LLAMINAR_ALLREDUCE_PRECISION), FP32 allreduces cast to FP16 first.
        // cast to FP16 first to halve PCIe transfer bandwidth. These device-local
        // scratch buffers hold the FP16 temporary. They are reserved once during
        // graph setup and are immutable throughout eager execution and capture.

        /// FP16 scratch buffer per device (void* to backend-owned device memory).
        std::vector<void *> fp16_scratch_buffers_;
        /// Setup-time element capacity per device.
        std::vector<size_t> fp16_scratch_counts_;
        /// Physical-ledger claims paired with the FP16 scratch buffers.
        std::vector<std::optional<PhysicalMemoryAllocationLease>>
            fp16_scratch_memory_leases_;
        /// Compute streams registered with the collective backend, retained so
        /// non-explicit-stream collectives can publish completion on the exact
        /// stream that receives the backend's completion wait.
        std::vector<void *> compute_streams_;

        // =====================================================================
        // BAR-Backed Tensor Registry
        // =====================================================================
        // For collective allreduce, we track which stage outputs are
        // allocated for collective operations.

        /// Map: stage_name -> (device_index -> registered FP32 tensor)
        /// The tensor at index i belongs to devices_[i]
        std::unordered_map<std::string, std::vector<FP32Tensor *>> bar_output_tensors_;

        /**
         * @brief Initialize the collective backend
         *
         * Creates the appropriate backend based on backend_ type and initializes it.
         * Called at the end of constructor after devices_ and backend_ are set.
         *
         * @return true if backend was successfully initialized
         */
        bool initializeBackend();
        bool initializeGraphCaptureBoundaryDeviceWords(
            const std::shared_ptr<PhysicalMemoryAuthority> &memory_authority);
        void releaseGraphCaptureBoundaryDeviceWords() noexcept;
        bool reserveFp16ScratchElements(
            size_t element_count,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory_authority);
        void releaseFp16ScratchBuffers() noexcept;
        void *requireReservedFp16Scratch(
            int device_index,
            size_t element_count,
            const std::string &stage_name,
            const char *caller);

        /**
         * @brief Get device pointers for all devices participating in collective
         *
         * For multi-GPU collectives, we need a buffer pointer for each device.
         * This helper extracts device pointers from a tensor that may have
         * multiple device buffers (one per device in the TP group).
         *
         * @param tensor Tensor with data on all devices
         * @return Vector of device pointers (one per device in devices_)
         */
        /**
         * @brief Convert our data type to CollectiveDataType
         * @param tensor Tensor to get dtype from
         * @return CollectiveDataType for the tensor
         */
        CollectiveDataType tensorDTypeToCollective(const TensorBase *tensor) const;

        /**
         * @brief Internal allreduce implementation (assumes lock is already held)
         *
         * Used by out-of-place allreduce after copying input to output.
         *
         * @param tensor Tensor to allreduce in-place
         * @return true on success
         */
        bool allreduceImpl(TensorBase *tensor);

        bool rendezvousOnStreamCollective(int device_index,
                                          TensorBase *tensor,
                                          const std::string &stage_name,
                                          size_t effective_count,
                                          CollectiveDataType dtype,
                                          void *stream,
                                          const std::string &precision);

        bool allreduceGroupedOnExplicitStreams(void *buffer,
                                               size_t effective_count,
                                               CollectiveDataType dtype,
                                               int device_index,
                                               void *stream,
                                               const std::string &stage_name,
                                               const std::string &precision,
                                               const std::vector<LocalTPCollectiveSidebandBuffer> *sidebands = nullptr);

        /**
         * @brief Enqueue the required per-device asynchronous GPU allreduce.
         *
         * Every homogeneous LocalTP GPU backend must implement this path. A
         * rejected launch is a hard operation failure; production never changes
         * to the host-barrier implementation because doing so would alter both
         * ordering and graph-capture behavior.
         *
         * @param tensor This device's tensor
         * @param stage_name Stage identifier for logging
         * @param count Number of elements (0 = use numel)
         * @return true on success
         */
        bool allreducePerDeviceRequired(TensorBase *tensor, const std::string &stage_name = "", size_t count = 0);

        /**
         * @brief Barrier-synchronized allreduce for CPU-only TP
         *
         * For LOCAL TP where all devices are CPU (e.g., multi-socket NUMA),
         * each worker thread has its tensor in host memory. This method uses
         * barrier synchronization to collect host pointers, performs element-wise
         * reduction on the last-arriving thread, and broadcasts the result to all.
         *
         * @param tensor This thread's tensor (in host memory)
         * @param stage_name Stage identifier for logging (optional)
         * @param count Number of elements to reduce (0 = use tensor->numel())
         * @return true on success (same result for all participants)
         */
        bool allreduceCpuBarrier(
            TensorBase *tensor,
            const std::string &stage_name = "",
            size_t count = 0,
            void *producer_stream = nullptr);

        /**
         * @brief Barrier-synchronized allgather for CPU-only TP
         *
         * For LOCAL TP where all devices are CPU, each worker thread contributes
         * its local shard. The last-arriving thread concatenates all shards into
         * each thread's global output tensor.
         *
         * @param local_shard This thread's local shard (in host memory)
         * @param global_tensor Output tensor to receive all gathered shards
         * @return true on success (same result for all participants)
         */
        bool allgatherCpuBarrier(const TensorBase *local_shard, TensorBase *global_tensor);

        /**
         * @brief Validate barrier-collected tensors before multi-GPU allreduce launch.
         *
         * This method enforces correctness invariants that protect against subtle
         * LOCAL TP bugs (wrong device mapping, dtype mismatch, invalid element count).
         *
         * @param effective_count Number of elements that will be reduced.
         * @param expected_dtype DType inferred from slot-0 tensor.
         * @return true when all invariants are satisfied and launch is safe.
         */
        bool validateBarrierTensorSetForMultiGpuAllreduce(
            size_t effective_count,
            CollectiveDataType expected_dtype) const;

        /**
         * @brief Normalize weights to sum to 1.0
         * @param weights Input weights (may not sum to 1.0)
         * @return Normalized weights
         */
        static std::vector<float> normalizeWeights(const std::vector<float> &weights);

        /**
         * @brief Compute cumulative counts for range calculations
         *
         * Given total count and weights, computes cumulative counts for
         * proportional distribution. Used by rowRangeForDevice/colRangeForDevice.
         *
         * @param total Total count to distribute
         * @param norm_weights Normalized weights (must sum to 1.0)
         * @return Cumulative counts (length = weights.size() + 1, starts at 0, ends at total)
         */
        static std::vector<int> computeCumulativeCounts(int total, const std::vector<float> &norm_weights);

        /**
         * @brief Auto-detect backend from device types
         *
         * - All CUDA devices → NCCL
         * - All ROCm devices → RCCL
         * - Mixed GPU types → HOST
         * - CPU involved → HOST
         *
         * @param devices Device list to analyze
         * @return Detected backend type
         */
        static CollectiveBackendType autoDetectBackend(const std::vector<GlobalDeviceAddress> &devices);

        /**
         * @brief Build device-to-index lookup map
         */
        void buildDeviceIndex();

        /**
         * @brief Return true when the current GPU graph configuration supports
         *        this context's LocalTP GPU-native collectives.
         *
         * @param reason_out Optional pointer receiving human-readable reason.
         * @return true when policy permits LocalTP collective execution.
         */
        bool isLocalTPGpuGraphPolicySupported(std::string *reason_out = nullptr) const;
    };

} // namespace llaminar2
