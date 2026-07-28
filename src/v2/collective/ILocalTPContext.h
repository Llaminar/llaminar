/**
 * @file ILocalTPContext.h
 * @brief Interface for LOCAL tensor parallelism operations
 *
 * LOCAL TP = multiple devices within a single MPI rank, decoupled from MPI world_size.
 * This enables tensor parallelism across GPUs owned by one rank, using high-bandwidth
 * backends like NCCL, RCCL, or HOST instead of cross-node MPI.
 *
 * Key concepts:
 * - LOCAL TP degree can be different from MPI world_size
 * - Supports proportional work distribution via weights (e.g., NVIDIA 73%, AMD 27%)
 * - Backend selection based on device types (NCCL for CUDA-only, HOST for mixed)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../backends/GlobalDeviceAddress.h"
#include "../config/OrchestrationConfig.h"
#include "../tensors/ITensor.h"
#include "ICollectiveBackend.h"
#include "ITPContext.h"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;

    /**
     * @brief Operation semantics for compact LocalTP control sidebands.
     *
     * These are intentionally explicit instead of reusing an activation
     * allreduce's type by convention. Rebalance histograms are additive and may
     * use AllreduceSum. Command metadata is not additive and must use Allgather
     * or Broadcast.
     */
    enum class LocalTPCollectiveSidebandKind
    {
        AllreduceSum,
        Allgather,
        Broadcast
    };

    inline const char *toString(LocalTPCollectiveSidebandKind kind)
    {
        switch (kind)
        {
        case LocalTPCollectiveSidebandKind::AllreduceSum:
            return "AllreduceSum";
        case LocalTPCollectiveSidebandKind::Allgather:
            return "Allgather";
        case LocalTPCollectiveSidebandKind::Broadcast:
            return "Broadcast";
        }
        return "Unknown";
    }

    /**
     * @brief One compact control sideband attached to a LocalTP collective stage.
     *
     * The buffers live on the participant identified by device_index in
     * collectiveSidebandOnStream(). For AllreduceSum, recv_buffer is the mutable
     * reduction buffer; send_buffer may be null for in-place reduction or point
     * at a distinct device buffer for out-of-place implementations. For
     * Allgather, send_buffer is the participant contribution and recv_buffer
     * receives degree() contiguous slices. For Broadcast, root_device_index owns
     * the source bytes and recv_buffer is overwritten on non-root participants.
     */
    struct LocalTPCollectiveSidebandBuffer
    {
        LocalTPCollectiveSidebandKind kind = LocalTPCollectiveSidebandKind::AllreduceSum;
        const void *send_buffer = nullptr;
        void *recv_buffer = nullptr;
        size_t element_count = 0;
        CollectiveDataType dtype = CollectiveDataType::INT32;
        int root_device_index = 0;
        std::string name;
    };

    /**
     * @brief Interface for LOCAL tensor parallelism operations
     *
     * LOCAL TP = multiple devices within a single MPI rank.
     * This decouples TP degree from MPI world_size, enabling:
     * - Single-rank multi-GPU execution
     * - Heterogeneous GPU TP (CUDA + ROCm on same rank)
     * - Proportional work distribution for mixed-capability GPUs
     *
     * Thread safety: All methods are thread-safe. Collective operations
     * must be called from all participating threads/devices.
     */
    class ILocalTPContext : public ITPContext
    {
    public:
        ~ILocalTPContext() override = default;

        /**
         * @brief LOCAL TP is always intra-rank
         * @return Always TPScope::LOCAL for LOCAL TP contexts
         */
        TPScope scope() const override { return TPScope::LOCAL; }

        // =====================================================================
        // Configuration
        // =====================================================================

        /**
         * @brief Get devices participating in LOCAL TP
         * @return Vector of GlobalDeviceAddress for all devices in this context
         */
        virtual const std::vector<GlobalDeviceAddress> &devices() const = 0;

        /**
         * @brief Get weights for proportional TP
         *
         * Weights determine work distribution. A device with weight 0.73 gets
         * 73% of the heads/columns in column-parallel operations.
         *
         * @return Vector of weights (sum to 1.0), same length as devices()
         */
        virtual const std::vector<float> &weights() const = 0;

        /**
         * @brief Get backend type for collective operations
         *
         * AUTO resolution:
         * - All CUDA devices → NCCL
         * - All ROCm devices → RCCL
         * - Mixed or CPU involved → HOST
         *
         * @return CollectiveBackendType
         */
        virtual CollectiveBackendType backend() const = 0;

        /**
         * @brief Get total TP degree (number of devices)
         * @return Number of devices participating in LOCAL TP
         */
        int degree() const override = 0;

        /**
         * @brief Get the current device's index within the LOCAL TP domain
         *
         * For orchestrator-driven LOCAL TP, this returns the device index that
         * the orchestrator is currently operating on behalf of. Must be set via
         * setCurrentDeviceIndex() before calling sharding methods.
         *
         * @return Index in range [0, degree()) for the current device
         */
        int myIndex() const override = 0;

        // =====================================================================
        // Collective Operations
        // =====================================================================

        /**
         * @brief All-reduce across LOCAL devices (in-place)
         *
         * Performs sum reduction across all devices, leaving result on all devices.
         * This is the core operation after row-parallel GEMM (e.g., Wo projection).
         *
         * @param tensor Tensor to all-reduce (modified in-place)
         * @return true on success, false on error
         */
        bool allreduce(TensorBase *tensor) override = 0;

        /**
         * @brief All-reduce across LOCAL devices with stage name (in-place)
         *
         * Like allreduce(), but with stage name for tensor lookup.
         * If output tensors are registered for this stage,
         * they are used for the reduction.
         *
         * @param tensor Tensor to all-reduce (modified in-place)
         * @param stage_name Stage identifier (e.g., "layer0_wo_allreduce")
         * @param count Number of elements to reduce (0 = use tensor->numel())
         *              IMPORTANT: For dynamic sequence lengths (decode), this must be
         *              set to actual_seq_len * hidden_dim, not the full buffer size.
         * @return true on success, false on error
         */
        virtual bool allreduce(TensorBase *tensor, const std::string &stage_name, size_t count = 0) = 0;

        /**
         * @brief All-reduce across LOCAL devices (out-of-place)
         *
         * @param input Source tensor (read-only)
         * @param output Destination tensor (must be pre-allocated)
         * @return true on success, false on error
         */
        virtual bool allreduce(const TensorBase *input, TensorBase *output) = 0;

        /**
         * @brief All-gather: gather shards from all devices
         *
         * Each device contributes its local shard, result is full tensor on all devices.
         * Used after column-parallel operations to reconstruct full activations.
         *
         * @param local_shard This device's shard (may differ in size due to proportional TP)
         * @param global_tensor Pre-allocated output for full tensor
         * @return true on success, false on error
         */
        bool allgather(const TensorBase *local_shard, TensorBase *global_tensor) override = 0;

        /**
         * @brief All-gather raw device buffers across LOCAL devices.
         *
         * This is for graph-visible state handoffs that operate on backend-owned
         * device allocations rather than TensorBase instances. It requires a
         * non-null producer stream so callers cannot accidentally order GPU
         * state transfers through the legacy default stream. Unsupported
         * contexts must fail loudly by returning false.
         *
         * @param local_send Device buffer contributed by this participant.
         * @param full_recv Device buffer receiving all participants' slices.
         * @param send_count Elements contributed by each participant.
         * @param dtype Element type for the collective.
         * @param device_index Participant index in devices().
         * @param producer_stream Explicit stream that produced local_send.
         * @param stage_name Stage identifier for diagnostics.
         * @return true on success, false when unsupported or failed.
         */
        virtual bool allgatherRawOnStream(
            const void *local_send,
            void *full_recv,
            size_t send_count,
            CollectiveDataType dtype,
            int device_index,
            void *producer_stream,
            const std::string &stage_name)
        {
            (void)local_send;
            (void)full_recv;
            (void)send_count;
            (void)dtype;
            (void)device_index;
            (void)producer_stream;
            (void)stage_name;
            return false;
        }

        /**
         * @brief Execute graph-visible grouped send/recv operations on one participant.
         *
         * This is the directed counterpart to allgatherRawOnStream() for
         * payloads where only a subset of participant edges carries useful
         * bytes. The caller supplies only operations involving device_index.
         * Each operation must use a non-null explicit producer stream.
         */
        virtual bool groupedP2PRawOnStream(
            const std::vector<CollectiveP2POp> &ops,
            int device_index,
            void *producer_stream,
            const std::string &stage_name)
        {
            (void)ops;
            (void)device_index;
            (void)producer_stream;
            (void)stage_name;
            return false;
        }

        /**
         * @brief Execute compact graph-visible control sidebands on an explicit stream.
         *
         * This lower-level ABI enqueues sidebands adjacent to an existing graph
         * collective stage. Production MoE rebalance traffic should prefer
         * allreduceWithSidebandsOnStream() so the anchor and sidebands enter one
         * backend group. This method remains for diagnostics and unsupported
         * fallback probes only. It must not use the legacy default stream or a
         * host-synchronized fallback.
         *
         * @param sidebands Compact sideband collectives for this participant.
         * @param device_index Participant index in devices().
         * @param producer_stream Explicit stream that produced the sideband buffers.
         * @param anchor_stage_name Existing collective stage these sidebands ride with.
         * @return true on success, false when unsupported or failed.
         */
        virtual bool collectiveSidebandOnStream(
            const std::vector<LocalTPCollectiveSidebandBuffer> &sidebands,
            int device_index,
            void *producer_stream,
            const std::string &anchor_stage_name)
        {
            (void)sidebands;
            (void)device_index;
            (void)anchor_stage_name;
            if (!producer_stream)
                throw std::invalid_argument("ILocalTPContext::collectiveSidebandOnStream requires a non-null GPU stream");
            return sidebands.empty();
        }

        /**
         * @brief Publish one complete sideband bundle across every LocalTP stream.
         *
         * This rank-level form makes participant ownership explicit: index @c i
         * in @p participant_sidebands and @p producer_streams always belongs to
         * devices()[i]. Implementations lower the participant-local descriptors
         * into one NCCL/RCCL group launch. The call is enqueue-only; completion
         * remains ordered by each supplied stream and is never observed through a
         * host rendezvous.
         *
         * This is the required production contract for compact MTP outcome
         * publication when there is no real activation allreduce to serve as an
         * anchor. Supplying a dummy allreduce, invoking one host worker per
         * participant, or falling back to host copies would all change the
         * operation's economics and ordering and are therefore forbidden.
         *
         * @param participant_sidebands Sideband descriptors grouped by LocalTP
         *        participant in devices() order.
         * @param producer_streams Exact non-null stream for every participant.
         * @param publication_name Stable diagnostic/PerfStats operation name.
         * @return true only when the complete multi-device group was enqueued.
         */
        virtual bool collectiveSidebandsMultiOnStreams(
            const std::vector<std::vector<LocalTPCollectiveSidebandBuffer>>
                &participant_sidebands,
            const std::vector<void *> &producer_streams,
            const std::string &publication_name)
        {
            (void)participant_sidebands;
            (void)publication_name;
            for (void *stream : producer_streams)
            {
                if (!stream)
                {
                    throw std::invalid_argument(
                        "ILocalTPContext::collectiveSidebandsMultiOnStreams requires non-null GPU streams");
                }
            }
            return false;
        }

        /**
         * @brief Execute an anchor allreduce and compact control sidebands as
         *        one grouped backend launch on an explicit stream.
         *
         * Production MoE rebalance traffic uses this path for homogeneous
         * NCCL/RCCL domains. The allreduce and sidebands are lowered together,
         * allowing the backend to enqueue them inside the same group region
         * instead of issuing separate rebalance-specific collectives.
         */
        virtual bool allreduceWithSidebandsOnStream(
            TensorBase *tensor,
            const std::string &stage_name,
            size_t count,
            void *producer_stream,
            const std::string &precision,
            const std::vector<LocalTPCollectiveSidebandBuffer> &sidebands,
            int device_index)
        {
            (void)tensor;
            (void)stage_name;
            (void)count;
            (void)precision;
            (void)sidebands;
            (void)device_index;
            if (!producer_stream)
                throw std::invalid_argument("ILocalTPContext::allreduceWithSidebandsOnStream requires a non-null GPU stream");
            return false;
        }

        /**
         * @brief True when collectiveSidebandOnStream can be captured into a GPU graph.
         */
        virtual bool supportsCollectiveSidebandOnStreamGraphCapture() const { return false; }

        /**
         * @brief True when raw all-gather handoffs can be captured into a GPU graph.
         *
         * This requires a homogeneous GPU LocalTP backend that implements
         * all-gather directly on the caller's explicit stream, without host
         * synchronization or a coordinator-thread wait.
         */
        virtual bool supportsRawAllgatherOnStreamGraphCapture() const { return false; }

        /**
         * @brief Rendezvous all LocalTP participants at a GPU graph-capture boundary.
         *
         * LocalTP GPU graph capture is a domain-level lifecycle, not an
         * independent per-device detail. A participant must not start recording a
         * graph while a sibling is still draining the previous eager prefill
         * chunk, and no participant should launch an immediately captured graph
         * before every sibling has exited capture. Implementations that own a
         * multi-device LocalTP domain should use this hook as a reusable cyclic
         * barrier keyed by @p boundary_name.
         *
         * @param boundary_name Human-readable boundary identifier used for
         *        contract checks and diagnostics.
         * @param device_index Participant index in devices().
         * @param timeout_ms Maximum wait time; non-positive means wait without a
         *        timeout.
         * @return true when all participants reached the same boundary, false on
         *         timeout, mismatch, duplicate arrival, or backend abort.
         */
        virtual bool graphCaptureBoundaryRendezvous(
            const std::string &boundary_name,
            int device_index,
            int timeout_ms)
        {
            (void)boundary_name;
            (void)device_index;
            (void)timeout_ms;
            return true;
        }

        /**
         * @brief Establish a device-domain fence before a graph lifecycle transition.
         *
         * A host rendezvous alone cannot order GPU work already queued by sibling
         * participants. Homogeneous LocalTP implementations must enqueue a tiny
         * NCCL/RCCL collective on each participant's exact graph stream, then
         * verify through the named rendezvous that every peer enqueued the same
         * fence. Returning true guarantees that subsequent work on each stream is
         * ordered after all device work preceding the fence across the domain.
         *
         * @param boundary_name Stable lifecycle boundary identifier.
         * @param device_index Participant index in devices().
         * @param stream Exact stream that will begin capture or launch a graph.
         * @param timeout_ms Host contract timeout for matching peer arrivals.
         * @return true only when the device fence and both contract rendezvous complete.
         */
        virtual bool graphCaptureBoundaryOnStream(
            const std::string &boundary_name,
            int device_index,
            void *stream,
            int timeout_ms)
        {
            (void)boundary_name;
            (void)device_index;
            (void)timeout_ms;
            if (!stream)
                throw std::invalid_argument(
                    "ILocalTPContext::graphCaptureBoundaryOnStream requires a non-null GPU stream");
            return false;
        }

        /**
         * @brief Gather shards from multiple devices into a single output tensor
         *
         * This is the orchestrator-friendly variant of allgather. Instead of requiring
         * each device to call allgather independently (SPMD pattern), this method
         * accepts all shards at once from a single control thread.
         *
         * Used by RankOrchestrator to gather partial logits from column-parallel
         * LM head execution across LOCAL TP devices.
         *
         * @param shards Vector of shard tensors (one per device, in device order)
         * @param output Pre-allocated output tensor for concatenated result
         * @return true on success, false on error
         *
         * @note The shards are concatenated in device order (device 0's shard first,
         *       then device 1's shard, etc.). For column-parallel LM head, this means
         *       output[0:vocab_local_0] = shard[0], output[vocab_local_0:vocab_local_0+vocab_local_1] = shard[1], etc.
         */
        virtual bool gatherFromDevices(
            const std::vector<const TensorBase *> &shards,
            TensorBase *output) = 0;

        /**
         * @brief Reduce-scatter: reduce then scatter result slices
         *
         * Reduces input across all devices, then scatters result so each device
         * gets a different portion. Useful for fused reduce-scatter operations.
         *
         * @param input Full tensor to reduce
         * @param output_shard This device's portion of the result
         * @return true on success, false on error
         */
        virtual bool reduceScatter(const TensorBase *input, TensorBase *output_shard) = 0;

        /**
         * @brief Broadcast tensor from one device to all others in the TP domain
         *
         * Replicates data from a source device to all other devices in the TP group.
         * Used when receiving PP activations that need to be available on all TP devices.
         *
         * For homogeneous backends (NCCL/RCCL), this uses the native broadcast.
         * For heterogeneous backends (HOST), this may use staged transfers.
         *
         * @param tensor Tensor to broadcast (must be valid on source device)
         * @param source_device_index Index of source device in devices() (0-based)
         * @return true on success, false on error
         */
        bool broadcast(TensorBase *tensor, int source_device_index = 0) override = 0;

        // =====================================================================
        // Synchronization
        // =====================================================================

        /**
         * @brief Synchronize all LOCAL devices
         *
         * Blocks until all devices have completed their pending operations.
         * Call after async collective operations to ensure completion.
         */
        virtual void synchronize() = 0;

        // =====================================================================
        // Stream Configuration
        // =====================================================================

        /**
         * @brief Register compute streams for event-based collective pre-synchronization
         *
         * When set, the collective backend can use lightweight event-based synchronization
         * (hipEventRecord + hipStreamWaitEvent) instead of hipDeviceSynchronize before
         * collective operations. One stream per device, in device order.
         *
         * @param compute_streams Opaque stream handles (hipStream_t* / cudaStream_t*), one per device
         */
        virtual void setComputeStreams(const std::vector<void *> &compute_streams) { (void)compute_streams; }

        // =====================================================================
        // Abort (for one-sided failure recovery)
        // =====================================================================

        /**
         * @brief Request abort of all pending collective operations.
         *
         * Called when one device thread fails and others may be stuck in
         * collective calls waiting for matching operations. Forcefully
         * tears down communicators to unblock pending operations.
         *
         * After calling this, the context is NOT usable for further collectives.
         */
        virtual void requestAbort() = 0;

        /**
         * @brief Check if abort has been requested by any device thread.
         */
        virtual bool isAbortRequested() const = 0;

        // =====================================================================
        // Device Management
        // =====================================================================

        /**
         * @brief Get index for a device (0-based)
         * @param device Device to look up
         * @return Index in devices() vector, or -1 if not found
         */
        virtual int indexForDevice(const GlobalDeviceAddress &device) const = 0;

        /**
         * @brief Get device by index
         * @param index 0-based index
         * @return GlobalDeviceAddress at that index
         * @throws std::out_of_range if index is invalid
         */
        virtual const GlobalDeviceAddress &deviceAt(int index) const = 0;

        /**
         * @brief Get work fraction for a device
         * @param device Device to look up
         * @return Weight for this device (0.0-1.0)
         */
        virtual float weightForDevice(const GlobalDeviceAddress &device) const = 0;

        // =====================================================================
        // Weight Sharding Utilities
        // =====================================================================

        /**
         * @brief Get head count for a device (for attention sharding)
         *
         * Distributes heads proportionally according to weights.
         * Example: 28 heads with weights [0.73, 0.27] → [20, 8] heads
         *
         * @param device Device to get head count for
         * @param total_heads Total attention heads in the model
         * @return Number of heads this device should process
         */
        virtual int headsForDevice(const GlobalDeviceAddress &device, int total_heads) const = 0;

        /**
         * @brief Get row range for a device (for row-parallel matrix sharding)
         *
         * Used for row-parallel operations (e.g., down projection).
         * Returns [start, end) range - end is exclusive.
         *
         * @param device Device to get range for
         * @param total_rows Total rows in the matrix
         * @return Pair of (start_row, end_row) - end is exclusive
         */
        virtual std::pair<int, int> rowRangeForDevice(
            const GlobalDeviceAddress &device, int total_rows) const = 0;

        /**
         * @brief Get column range for a device (for column-parallel matrix sharding)
         *
         * Used for column-parallel operations (e.g., Q/K/V projections, FFN up/gate).
         * Returns [start, end) range - end is exclusive.
         *
         * @param device Device to get range for
         * @param total_cols Total columns in the matrix
         * @return Pair of (start_col, end_col) - end is exclusive
         */
        virtual std::pair<int, int> colRangeForDevice(
            const GlobalDeviceAddress &device, int total_cols) const = 0;

        // =====================================================================
        // Tensor Registry (Collective Backend)
        // =====================================================================

        /**
         * @brief Register a tensor for a stage's output
         *
         * Called during graph construction for row-parallel stages (FFN_DOWN, Wo).
         * The registered tensors are used by collective allreduce operations.
         *
         * @param stage_name Stage identifier (e.g., "layer0_wo_allreduce")
         * @param device Device that owns this tensor (must be in devices())
         * @param tensor Tensor to register
         */
        virtual void registerBARBackedOutput(
            const std::string &stage_name,
            const GlobalDeviceAddress &device,
            TensorBase *tensor) = 0;

        /**
         * @brief Check if a stage has any outputs registered
         *
         * @param stage_name Stage identifier
         * @return true if at least one device has a tensor registered
         */
        virtual bool hasBARBackedOutputs(const std::string &stage_name) const = 0;

        /**
         * @brief Clear all tensor registrations
         *
         * Called when resetting the context or changing buffer sizes.
         */
        virtual void clearBARBackedOutputs() = 0;

        /**
         * @brief Reserve every persistent buffer required by collective execution.
         *
         * This setup-only call makes the collective memory contract explicit:
         * the backend receives its transport workspace byte capacity, while the
         * LocalTP context receives the maximum logical element count needed by
         * FP16 transport. Keeping these dimensions separate prevents quantized
         * activation byte counts from accidentally under-sizing FP16 scratch.
         *
         * No collective execution method may grow either reservation. A request
         * that exceeds this setup contract is a fatal planning error, not an
         * invitation to allocate while a graph is executing or being captured.
         *
         * Call this during initialization after model dimensions are known:
         * @code
         * const size_t max_elements = max_seq_len * hidden_size;
         * const size_t transport_bytes =
         *     activationPrecisionBufferBytes(max_elements, precision);
         * tp_ctx->reserveCollectiveResources(
         *     transport_bytes_with_margin,
         *     max_elements_with_margin);
         * @endcode
         *
         * @param backend_temp_bytes Minimum backend transport workspace capacity.
         * @param fp16_scratch_elements Maximum logical FP16 transport element count.
         * @return true only when every participant is fully reserved.
         */
        virtual bool reserveCollectiveResources(
            size_t backend_temp_bytes,
            size_t fp16_scratch_elements) = 0;
    };

    /**
     * @brief Factory function to create a LocalTPContext
     *
     * @param devices Devices participating in LOCAL TP
     * @param weights Work distribution weights (empty for equal distribution)
     * @param backend Backend type (AUTO to select based on device types)
     * @return Unique pointer to ILocalTPContext implementation
     */
    std::unique_ptr<ILocalTPContext> createLocalTPContext(
        std::vector<GlobalDeviceAddress> devices,
        std::vector<float> weights,
        CollectiveBackendType backend);

} // namespace llaminar2
