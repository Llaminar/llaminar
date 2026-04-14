/**
 * @file HeterogeneousBackend.h
 * @brief Heterogeneous multi-GPU collective backend orchestrating NCCL, RCCL, and HOST
 *
 * The HeterogeneousBackend enables collective operations across mixed NVIDIA (CUDA)
 * and AMD (ROCm) GPU configurations. It orchestrates sub-backends:
 * - NCCL for intra-NVIDIA communication (when >1 CUDA GPU)
 * - RCCL for intra-AMD communication (when >1 ROCm GPU)
 * - HostBackend for cross-vendor bridge transfers (host-staged)
 *
 * Example configuration: RTX 3090 (cuda:0) + 2x MI50 (rocm:0, rocm:1)
 *
 * AllReduce algorithm (3 GPUs: 1 CUDA + 2 ROCm):
 * 1. RCCL: AllReduce within ROCm domain → rocm:0 has ROCm partial
 * 2. HOST: Transfer cuda:0 ↔ rocm:0 partials via host-staged allreduce
 * 3. RCCL: Broadcast from rocm:0 to rocm:1
 *
 * Requirements:
 * - HAVE_CUDA and HAVE_ROCM both defined
 * - At least 1 CUDA device and 1 ROCm device in the group
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../ICollectiveBackend.h"
#include "../DeviceGroup.h"
#include <memory>
#include <vector>

namespace llaminar2
{

    // Forward declarations for sub-backends
    class NCCLBackend;
    class RCCLBackend;
    class HostBackend;

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

    /**
     * @brief Heterogeneous multi-GPU collective backend
     *
     * Orchestrates NCCL, RCCL, and HOST backends for mixed NVIDIA+AMD
     * GPU configurations. Uses a hierarchical reduction pattern:
     * 1. Reduce within each vendor domain (NCCL/RCCL if >1 GPU)
     * 2. Cross-vendor reduction via host-staged bridge
     * 3. Broadcast back within each domain
     *
     * Thread Safety: Not thread-safe. Use one instance per device/stream.
     */
    class HeterogeneousBackend : public ICollectiveBackend
    {
    public:
        /**
         * @brief Construct HeterogeneousBackend
         *
         * Sub-backends are created lazily during initialize().
         */
        HeterogeneousBackend();

        ~HeterogeneousBackend() override;

        // =====================================================================
        // Identity
        // =====================================================================

        CollectiveBackendType type() const override { return CollectiveBackendType::HETEROGENEOUS; }
        std::string name() const override { return "Heterogeneous"; }

        // =====================================================================
        // Capability Queries
        // =====================================================================

        bool supportsDevice(DeviceType type) const override
        {
            return type == DeviceType::CUDA || type == DeviceType::ROCm;
        }

        bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override;

        bool isAvailable() const override;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        /**
         * @brief Initialize backend for a heterogeneous device group
         *
         * Validates the group has both CUDA and ROCm devices, then:
         * 1. Separates devices into CUDA and ROCm vectors
         * 2. Selects bridge devices (cuda:0 and rocm:0)
         * 3. Creates sub-backends for each domain (if >1 device)
         * 4. Creates HOST backend for cross-domain bridge
         *
         * @param group Device group with mixed CUDA and ROCm devices
         * @return true on success, false if validation fails
         */
        bool initialize(const DeviceGroup &group) override;

        bool isInitialized() const override { return initialized_; }
        void shutdown() override;

        bool reserveTempBufferBytes(size_t bytes) override;

        // =====================================================================
        // Collective Operations (Stubs - Phase 1)
        // =====================================================================

        /**
         * @brief Indicates this backend manages multiple GPUs in a single process
         * @return true (HeterogeneousBackend manages multiple GPUs via sub-backends)
         */
        bool isMultiGpuSingleProcess() const override { return true; }

        bool allreduce(
            void *buffer,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op) override;

        /**
         * @brief Multi-GPU AllReduce across heterogeneous devices
         *
         * Performs 3-phase allreduce:
         * 1. Intra-domain reduce (NCCL for CUDA, RCCL for ROCm)
         * 2. Cross-domain bridge exchange (host-staged between cuda:0 ↔ rocm:0)
         * 3. Intra-domain broadcast (NCCL for CUDA, RCCL for ROCm)
         *
         * IMPORTANT: Buffer order must match device order in the DeviceGroup.
         * buffers[i] corresponds to device_group_.devices[i].
         *
         * @param buffers Array of device buffers (one per GPU in device order)
         * @param count Number of elements per buffer
         * @param dtype Data type of elements
         * @param op Reduction operation (typically SUM)
         * @return true on success
         */
        bool allreduceMulti(
            const std::vector<void *> &buffers,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op) override;

        bool allgather(
            const void *send_buf,
            void *recv_buf,
            size_t send_count,
            CollectiveDataType dtype) override;

        bool allgatherv(
            const void *send_buf,
            size_t send_count,
            void *recv_buf,
            const std::vector<int> &recv_counts,
            const std::vector<int> &displacements,
            CollectiveDataType dtype) override;

        bool reduceScatter(
            const void *send_buf,
            void *recv_buf,
            size_t recv_count,
            CollectiveDataType dtype,
            CollectiveOp op) override;

        bool broadcast(
            void *buffer,
            size_t count,
            CollectiveDataType dtype,
            int root_rank) override;

        // =====================================================================
        // Synchronization
        // =====================================================================

        bool synchronize() override;

        // =====================================================================
        // Diagnostics
        // =====================================================================

        std::string lastError() const override { return last_error_; }

        // =====================================================================
        // Test Accessors (for unit testing internal state)
        // =====================================================================

        /// Get list of CUDA devices in the group
        const std::vector<DeviceId> &cudaDevices() const { return cuda_devices_; }

        /// Get list of ROCm devices in the group
        const std::vector<DeviceId> &rocmDevices() const { return rocm_devices_; }

        /// Get the CUDA bridge device (used for cross-vendor transfers)
        DeviceId cudaBridge() const { return cuda_bridge_; }

        /// Get the ROCm bridge device (used for cross-vendor transfers)
        DeviceId rocmBridge() const { return rocm_bridge_; }

        /// Check if NCCL sub-backend is active (>1 CUDA device)
        bool hasNCCLBackend() const { return nccl_backend_ != nullptr; }

        /// Check if RCCL sub-backend is active (>1 ROCm device)
        bool hasRCCLBackend() const { return rccl_backend_ != nullptr; }

        /// Check if HOST bridge backend is active
        bool hasBridgeBackend() const { return bridge_backend_ != nullptr; }

        // =====================================================================
        // Reduce-Scatter Pattern (for large tensors)
        // =====================================================================

        /// Threshold in bytes above which reduce-scatter pattern is used
        static constexpr size_t REDUCE_SCATTER_THRESHOLD = 4 * 1024 * 1024; // 4 MB

        /**
         * @brief Check if reduce-scatter pattern should be used for this tensor size
         *
         * The reduce-scatter pattern reduces cross-vendor HOST bridge traffic for
         * large tensors. Instead of sending 100% of the tensor across the bridge,
         * it sends only 1/max(N,M) of the tensor, where N and M are the device
         * counts in each domain.
         *
         * @param tensor_bytes Size of tensor in bytes
         * @return true if reduce-scatter pattern should be used
         */
        bool shouldUseReduceScatterPattern(size_t tensor_bytes) const;

        /**
         * @brief Check if adaptive asymmetric reduce-scatter should be used
         *
         * Returns true when:
         * - Tensor is large enough (>= REDUCE_SCATTER_THRESHOLD)
         * - Device counts are asymmetric (CUDA != ROCm)
         * - At least one domain has >1 device (for reduce-scatter benefit)
         *
         * The adaptive pattern chunks based on the larger domain and handles
         * the asymmetry by having the smaller domain exchange multiple chunks.
         *
         * @param tensor_bytes Size of tensor in bytes
         * @return true if adaptive asymmetric pattern should be used
         */
        bool shouldUseAdaptiveAsymmetricPattern(size_t tensor_bytes) const;

        /**
         * @brief Plan information for reduce-scatter pattern
         *
         * Used for unit testing to verify chunk calculations without
         * actually invoking GPU operations.
         */
        struct ReduceScatterPlan
        {
            bool use_reduce_scatter_pattern = false; ///< True if pattern should be used
            size_t cuda_chunk_count = 0;             ///< Elements per CUDA device after reduce-scatter
            size_t rocm_chunk_count = 0;             ///< Elements per ROCm device after reduce-scatter
            size_t bridge_exchange_count = 0;        ///< Elements exchanged across bridge (min chunk)
            size_t cuda_device_count = 0;            ///< Number of CUDA devices
            size_t rocm_device_count = 0;            ///< Number of ROCm devices
        };

        /**
         * @brief Plan reduce-scatter operation
         *
         * Computes chunk sizes and determines if pattern should be used.
         * Used for unit testing chunk calculations.
         *
         * @param count Total number of elements in the tensor
         * @param element_size Size of each element in bytes
         * @return ReduceScatterPlan with chunk calculations
         */
        ReduceScatterPlan planReduceScatter(size_t count, size_t element_size) const;

        // =====================================================================
        // Adaptive Asymmetric Reduce-Scatter Pattern
        // =====================================================================

        /**
         * @brief Plan information for adaptive asymmetric reduce-scatter
         *
         * For asymmetric configs (e.g., 1 CUDA + 2 ROCm), this pattern:
         * - Chunks based on the LARGER domain's device count
         * - Larger domain does normal reduce-scatter
         * - Smaller domain handles multiple chunks via bridge
         */
        struct AsymmetricReduceScatterPlan
        {
            bool use_adaptive_pattern = false; ///< True if adaptive pattern should be used
            size_t num_chunks = 0;             ///< Number of chunks (= max(cuda_count, rocm_count))
            size_t chunk_elements = 0;         ///< Elements per chunk
            size_t last_chunk_elements = 0;    ///< Elements in last chunk (may differ due to rounding)
            size_t cuda_device_count = 0;      ///< Number of CUDA devices
            size_t rocm_device_count = 0;      ///< Number of ROCm devices
            bool cuda_is_larger = false;       ///< True if CUDA domain has more devices
            size_t larger_domain_count = 0;    ///< Device count in larger domain
            size_t smaller_domain_count = 0;   ///< Device count in smaller domain
        };

        // =====================================================================
        // Topology Analysis - Optimal Pattern Selection
        // =====================================================================

        /**
         * @brief Allreduce execution pattern based on topology analysis
         */
        enum class AllreducePattern
        {
            /// Standard 3-phase: reduce → bridge → broadcast
            /// Best for: small tensors, 1+1 config, or when other patterns aren't applicable
            STANDARD_3PHASE,

            /// Symmetric reduce-scatter: RS → bridge → AG
            /// Best for: N CUDA + N ROCm with large tensors (N-way parallel bridge)
            SYMMETRIC_REDUCE_SCATTER,

            /// Partial RS in larger domain + standard bridge + AG
            /// Best for: 1 CUDA + M ROCm or N CUDA + 1 ROCm with large tensors
            /// Reduces intra-domain traffic before/after bottleneck bridge
            PARTIAL_REDUCE_SCATTER,

            /// GCD-based chunking for general asymmetric configs
            /// Best for: N CUDA + M ROCm where GCD(N,M) > 1
            /// Not yet implemented
            GCD_REDUCE_SCATTER,

            /// GCD-way parallel bridge for asymmetric configs
            /// Best for: N CUDA + M ROCm where GCD(N,M) > 1
            /// Uses GCD parallel host-staged bridges for increased bandwidth
            GCD_MULTI_BRIDGE
        };

        /**
         * @brief Topology analysis result for optimal pattern selection
         */
        struct TopologyAnalysis
        {
            // Device counts
            size_t cuda_count = 0;
            size_t rocm_count = 0;

            // Topology classification
            bool is_symmetric = false;      ///< cuda_count == rocm_count
            bool is_minimal = false;        ///< 1 + 1 config (no intra-domain parallelism)
            bool is_cuda_singleton = false; ///< 1 CUDA + M ROCm (M > 1)
            bool is_rocm_singleton = false; ///< N CUDA + 1 ROCm (N > 1)
            size_t gcd = 1;                 ///< GCD(cuda_count, rocm_count)

            // Selected pattern
            AllreducePattern pattern = AllreducePattern::STANDARD_3PHASE;

            // Pattern-specific parameters
            size_t num_chunks = 1;              ///< Chunks for RS patterns
            size_t bridge_traffic_fraction = 1; ///< 1/N of tensor crosses bridge (denominator)

            // Theoretical bandwidth analysis
            double intra_domain_parallelism = 1.0; ///< Parallelism factor within each domain
            double bridge_parallelism = 1.0;       ///< Bridge transfer parallelism (for symmetric RS)

            // Pipelining eligibility
            bool pipelining_eligible = false; ///< True if tensor is large enough for pipelining
            size_t pipeline_chunks = 0;       ///< Number of pipeline chunks if eligible

            // Explanation for debugging/logging
            std::string reason;
        };

        /**
         * @brief Analyze topology and select optimal allreduce pattern
         *
         * Considers:
         * - Device counts in each domain
         * - Tensor size (large tensors benefit from RS patterns)
         * - Symmetry (enables parallel bridge transfers)
         * - Implementation status (some patterns not yet available)
         *
         * @param tensor_bytes Size of tensor in bytes
         * @return TopologyAnalysis with selected pattern and parameters
         */
        TopologyAnalysis analyzeTopology(size_t tensor_bytes) const;

        /**
         * @brief Plan adaptive asymmetric reduce-scatter operation
         *
         * @param count Total number of elements in the tensor
         * @param element_size Size of each element in bytes
         * @return AsymmetricReduceScatterPlan with chunk calculations
         */
        AsymmetricReduceScatterPlan planAsymmetricReduceScatter(size_t count, size_t element_size) const;

        /**
         * @brief Execute GCD-way multi-bridge allreduce pattern
         *
         * For asymmetric configs where GCD(cuda_count, rocm_count) > 1,
         * this pattern uses GCD parallel host-staged bridges:
         *
         * Phase 1: Parallel intra-domain allreduce
         *   - NCCL allreduce (all CUDA devices)
         *   - RCCL allreduce (all ROCm devices)
         *   - Both run in parallel
         *
         * Phase 2: Multi-pair host-staged allreduce
         *   - GCD pairs exchange in parallel via host-staged bridge
         *   - Each pair: CUDA[i] ↔ ROCm[i * (M/G)]
         *
         * Phase 3: Intra-domain broadcast from bridge devices
         *   - NCCL: No broadcast needed (all CUDA already have result from Phase 1)
         *   - RCCL: Broadcast from each bridge ROCm device to its group
         *
         * @param cuda_buffers Device buffers for CUDA GPUs
         * @param rocm_buffers Device buffers for ROCm GPUs
         * @param count Number of elements per buffer
         * @param dtype Data type
         * @param op Reduction operation
         * @return true on success
         */
        bool executeGcdMultiBridge(
            const std::vector<void *> &cuda_buffers,
            const std::vector<void *> &rocm_buffers,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op);

        /**
         * @brief Execute adaptive asymmetric reduce-scatter pattern
         *
         * For asymmetric device counts (e.g., 1 CUDA + 2 ROCm):
         *
         * Phase 1: Reduce-scatter in the larger domain
         *   - If ROCm is larger: RCCL reduce-scatter → each ROCm device has 1 chunk
         *   - CUDA (single device): keeps full tensor locally reduced
         *
         * Phase 2: Chunked bridge exchange
         *   - For each chunk i:
         *     - CUDA sends its chunk[i] to ROCm[i]
         *     - ROCm[i] sends its reduced chunk to CUDA
         *     - Both sides accumulate (SUM)
         *   - Pipelined: multiple exchanges can overlap
         *
         * Phase 3: AllGather in larger domain
         *   - ROCm devices exchange to get full tensor
         *
         * Benefits over standard 3-phase:
         *   - Smaller per-transfer size enables pipelining
         *   - Multiple bridge paths can overlap (CUDA↔ROCm[0], CUDA↔ROCm[1])
         *
         * @param cuda_buffers Device buffers for CUDA GPUs
         * @param rocm_buffers Device buffers for ROCm GPUs
         * @param count Number of elements per buffer
         * @param dtype Data type
         * @param op Reduction operation
         * @return true on success
         */
        bool executeAdaptiveAsymmetricReduceScatter(
            const std::vector<void *> &cuda_buffers,
            const std::vector<void *> &rocm_buffers,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op);

        // =====================================================================
        // Phase 1 Plan (for testing)
        // =====================================================================

        /**
         * @brief Information about which backends Phase 1 will invoke
         *
         * Used for unit testing to verify the planning logic without
         * actually invoking GPU operations.
         */
        struct Phase1Plan
        {
            bool will_call_nccl_reduce = false; ///< True if NCCL reduce will be called
            bool will_call_rccl_reduce = false; ///< True if RCCL reduce will be called
            int nccl_reduce_root = -1;          ///< Root index for NCCL reduce (-1 if not called)
            int rccl_reduce_root = -1;          ///< Root index for RCCL reduce (-1 if not called)
            size_t nccl_device_count = 0;       ///< Number of CUDA devices
            size_t rccl_device_count = 0;       ///< Number of ROCm devices
        };

        /**
         * @brief Plan Phase 1 without executing
         *
         * Returns information about which sub-backends would be invoked
         * for intra-domain reduction. Used for unit testing.
         *
         * @return Phase1Plan with execution details
         */
        Phase1Plan planPhase1() const;

        // =====================================================================
        // Phase 2 Plan (for testing)
        // =====================================================================

        /**
         * @brief Information about Phase 2 bridge exchange
         *
         * Used for unit testing to verify the planning logic without
         * actually invoking GPU operations.
         */
        struct Phase2Plan
        {
            bool will_call_bridge_allreduce = false; ///< True if host-staged bridge allreduce will be called
            DeviceId cuda_bridge_device;             ///< CUDA bridge device (cuda:0)
            DeviceId rocm_bridge_device;             ///< ROCm bridge device (rocm:0)
        };

        /**
         * @brief Plan Phase 2 without executing
         *
         * Returns information about cross-domain bridge exchange.
         * Phase 2 uses host-staged allreduce between cuda:0 and rocm:0.
         *
         * @return Phase2Plan with execution details
         */
        Phase2Plan planPhase2() const;

        // =====================================================================
        // Phase 3 Plan (for testing)
        // =====================================================================

        /**
         * @brief Information about Phase 3 intra-domain broadcast
         *
         * Used for unit testing to verify the planning logic without
         * actually invoking GPU operations.
         */
        struct Phase3Plan
        {
            bool will_call_nccl_broadcast = false; ///< True if NCCL broadcast will be called
            bool will_call_rccl_broadcast = false; ///< True if RCCL broadcast will be called
            int nccl_broadcast_root = -1;          ///< Root index for NCCL broadcast (-1 if not called)
            int rccl_broadcast_root = -1;          ///< Root index for RCCL broadcast (-1 if not called)
            size_t nccl_device_count = 0;          ///< Number of CUDA devices
            size_t rccl_device_count = 0;          ///< Number of ROCm devices
        };

        /**
         * @brief Plan Phase 3 without executing
         *
         * Returns information about which sub-backends would be invoked
         * for intra-domain broadcast. Used for unit testing.
         *
         * @return Phase3Plan with execution details
         */
        Phase3Plan planPhase3() const;

        // =====================================================================
        // Phase Execution (for integration testing)
        // =====================================================================

        /**
         * @brief Execute Phase 1: Parallel intra-domain reduction
         *
         * Reduces all CUDA buffers to cuda_bridge_ using NCCL (if >1 CUDA device)
         * and all ROCm buffers to rocm_bridge_ using RCCL (if >1 ROCm device).
         * Both reductions can run in parallel since they operate on independent
         * hardware domains.
         *
         * After this phase:
         * - cuda_buffers[0] (bridge) contains sum of all CUDA partials
         * - rocm_buffers[0] (bridge) contains sum of all ROCm partials
         *
         * @param cuda_buffers Device buffers for CUDA GPUs (one per device)
         * @param rocm_buffers Device buffers for ROCm GPUs (one per device)
         * @param count Number of elements per buffer
         * @param dtype Data type
         * @param op Reduction operation
         * @return true on success
         */
        bool executePhase1_IntraDomainReduce(
            const std::vector<void *> &cuda_buffers,
            const std::vector<void *> &rocm_buffers,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op);

        /**
         * @brief Execute Phase 2: Cross-domain bridge exchange
         *
         * Performs allreduce between the CUDA and ROCm bridge devices via
         * host-staged bridge. After Phase 1, each bridge has its domain's partial sum.
         * Phase 2 combines these via host-staged allreduce to produce the global sum.
         *
         * After this phase:
         * - cuda_bridge_buffer contains global sum
         * - rocm_bridge_buffer contains global sum
         *
         * @param cuda_bridge_buffer Buffer on cuda:0 (bridge device)
         * @param rocm_bridge_buffer Buffer on rocm:0 (bridge device)
         * @param count Number of elements per buffer
         * @param dtype Data type
         * @param op Reduction operation
         * @return true on success
         */
        bool executePhase2_BridgeExchange(
            void *cuda_bridge_buffer,
            void *rocm_bridge_buffer,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op);

        /**
         * @brief Execute Phase 3: Parallel intra-domain broadcast
         *
         * Broadcasts the global sum from each bridge device to all other
         * devices in its domain. NCCL broadcasts from cuda:0 to all CUDA GPUs
         * (if >1 CUDA device), and RCCL broadcasts from rocm:0 to all ROCm GPUs
         * (if >1 ROCm device). Both broadcasts run in parallel since they
         * operate on independent hardware domains.
         *
         * After this phase:
         * - ALL CUDA devices have the global sum
         * - ALL ROCm devices have the global sum
         *
         * @param cuda_buffers Device buffers for CUDA GPUs (one per device)
         * @param rocm_buffers Device buffers for ROCm GPUs (one per device)
         * @param count Number of elements per buffer
         * @param dtype Data type
         * @return true on success
         */
        bool executePhase3_IntraDomainBroadcast(
            const std::vector<void *> &cuda_buffers,
            const std::vector<void *> &rocm_buffers,
            size_t count,
            CollectiveDataType dtype);

        /**
         * @brief Execute standard 3-phase allreduce (non-pipelined)
         *
         * Convenience helper that runs the standard 3-phase pattern:
         * 1. Phase 1: Intra-domain reduce (NCCL/RCCL)
         * 2. Phase 2: Bridge exchange (host-staged)
         * 3. Phase 3: Intra-domain broadcast (NCCL/RCCL)
         *
         * Used as fallback when optimized patterns (reduce-scatter, pipelining)
         * are not applicable (e.g., asymmetric device counts).
         *
         * @param cuda_buffers Device buffers for CUDA GPUs
         * @param rocm_buffers Device buffers for ROCm GPUs
         * @param count Number of elements per buffer
         * @param dtype Data type
         * @param op Reduction operation
         * @return true on success
         */
        bool executeStandard3PhaseAllreduce(
            const std::vector<void *> &cuda_buffers,
            const std::vector<void *> &rocm_buffers,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op);

        /**
         * @brief Execute reduce-scatter pattern for large tensors
         *
         * Optimized allreduce using reduce-scatter + allgather pattern:
         * 1. Reduce-scatter in each domain (each device gets 1/N of reduced data)
         * 2. Bridge exchange (only exchange bridge device chunks, much smaller)
         * 3. AllGather in each domain (reassemble full tensor)
         *
         * This reduces cross-vendor HOST bridge traffic from 100% to 1/max(N,M).
         *
         * @param cuda_buffers Device buffers for CUDA GPUs
         * @param rocm_buffers Device buffers for ROCm GPUs
         * @param count Number of elements per buffer
         * @param dtype Data type
         * @param op Reduction operation
         * @return true on success
         */
        bool executeReduceScatterPattern(
            const std::vector<void *> &cuda_buffers,
            const std::vector<void *> &rocm_buffers,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op);

        // =====================================================================
        // Partial Reduce-Scatter Pattern (for singleton configurations)
        // =====================================================================

        /**
         * @brief Execute partial reduce-scatter pattern for singleton configs
         *
         * For singleton configurations (1 CUDA + N ROCm or N CUDA + 1 ROCm),
         * this pattern reduces memory pressure in the larger domain:
         *
         * 1. Reduce-scatter in larger domain (each device gets 1/N of data)
         * 2. Chunked bridge exchange with staging through bridge device
         * 3. Allgather in larger domain to reconstruct full tensor
         *
         * Example for 1 CUDA + 4 ROCm:
         *   Phase 1: RCCL reduce-scatter → ROCm[i] has chunk[i] (1/4 memory each)
         *   Phase 2: For each chunk i:
         *            - Stage chunk from ROCm[i] to ROCm[0] (if i != 0)
         *            - Bridge exchange: ROCm[0] ↔ CUDA[0]
         *            - Stage result back to ROCm[i] (if i != 0)
         *   Phase 3: RCCL allgather to reconstruct
         *
         * @param buffers All device buffers in device order
         * @param count Number of elements per buffer
         * @param dtype Data type
         * @return true on success
         */
        bool executePartialReduceScatter(
            std::vector<void *> &buffers,
            size_t count,
            CollectiveDataType dtype);

        /**
         * @brief Stage chunks through the bridge device for singleton configs
         *
         * For singleton configurations, the bridge is a serial bottleneck.
         * This method stages chunks through the bridge device to enable
         * the partial reduce-scatter pattern.
         *
         * @param singleton_buffer Buffer on the singleton device (full tensor)
         * @param bridge_buffer Buffer on the bridge device of the larger domain
         * @param chunk_buffers Pointers to chunks on each device in larger domain
         *                      (after reduce-scatter, each device has one chunk)
         * @param chunk_size Number of elements per chunk
         * @param dtype Data type
         * @param singleton_is_cuda true if singleton device is CUDA, false if ROCm
         * @param op Reduction operation for accumulation
         * @return true on success
         */
        bool stageChunksThroughBridge(
            void *singleton_buffer,
            void *bridge_buffer,
            const std::vector<void *> &chunk_buffers,
            size_t chunk_size,
            CollectiveDataType dtype,
            bool singleton_is_cuda,
            CollectiveOp op);

        /**
         * @brief Pipelined chunk staging through bridge with double buffering
         *
         * Optimized version of stageChunksThroughBridge that uses double buffering
         * and async operations to overlap:
         * - Stage chunk N from device[i] → bridge buffer[A]
         * - Bridge exchange for chunk N-1 using bridge buffer[B]
         * - Stage result N-2 back to device[i-2]
         *
         * This enables 3-stage pipelining:
         *   [Stage In N] → [Bridge Exchange N-1] → [Stage Out N-2]
         *
         * @param singleton_buffer Buffer on singleton device (full tensor)
         * @param bridge_buffer_0 First staging buffer on bridge device
         * @param bridge_buffer_1 Second staging buffer for double buffering
         * @param chunk_buffers Pointers to chunks on each device in larger domain
         * @param chunk_size Number of elements per chunk
         * @param dtype Data type
         * @param singleton_is_cuda true if singleton device is CUDA
         * @param op Reduction operation
         * @param pipeline_depth Number of chunks in flight (default 2)
         * @return true on success
         */
        bool stageChunksThroughBridgePipelined(
            void *singleton_buffer,
            void *bridge_buffer_0,
            void *bridge_buffer_1,
            const std::vector<void *> &chunk_buffers,
            size_t chunk_size,
            CollectiveDataType dtype,
            bool singleton_is_cuda,
            CollectiveOp op,
            size_t pipeline_depth = 2);

        // =====================================================================
        // Phase 2→3 Chunk-Based Pipelining
        // =====================================================================

        /// Chunk size for pipelined Phase 2→3 overlap (1 MB)
        static constexpr size_t PIPELINE_CHUNK_SIZE = 1 * 1024 * 1024;

        /// Minimum tensor size to enable pipelining (4 MB)
        static constexpr size_t PIPELINE_MIN_TENSOR_SIZE = 4 * 1024 * 1024;

        /// Default pipeline depth (2 = double buffering)
        static constexpr size_t DEFAULT_PIPELINE_DEPTH = 2;

        /**
         * @brief Pipeline plan information for testing
         *
         * Exposes pipelining decisions for unit testing without GPU invocation.
         */
        struct PipelinePlan
        {
            bool will_use_pipelining = false; ///< True if pipelining will be used
            size_t num_chunks = 0;            ///< Number of chunks for pipelined transfer
            size_t chunk_elements = 0;        ///< Elements per chunk (except possibly last)
            size_t last_chunk_elements = 0;   ///< Elements in last chunk (may differ due to rounding)
            size_t total_elements = 0;        ///< Total elements being transferred
        };

        /**
         * @brief Plan pipelining for a given tensor size
         *
         * Determines whether pipelining should be used and computes chunk parameters.
         * Used for unit testing the planning logic without GPU invocation.
         *
         * @param count Number of elements
         * @param element_size Size of each element in bytes
         * @return PipelinePlan with chunk calculations
         */
        PipelinePlan planPipelining(size_t count, size_t element_size) const;

        /**
         * @brief Execute reduce-scatter pattern with Phase 2→3 pipelining
         *
         * Optimized version of executeReduceScatterPattern that overlaps
         * Phase 2 (HOST bridge transfer) with Phase 3 (intra-domain allgather)
         * using chunk-based pipelining.
         *
         * Algorithm:
         * - Phase 1: Standard reduce-scatter (unchanged)
         * - Phase 2+3 pipelined: For each chunk:
         *   - Complete bridge exchange for chunk N
         *   - Kick off allgather for chunk N (can overlap with bridge for N+1)
         *
         * V1 Implementation Note: Currently uses serial execution per chunk
         * (bridge then allgather). True async overlap is documented as future work.
         *
         * @param cuda_buffers Device buffers for CUDA GPUs
         * @param rocm_buffers Device buffers for ROCm GPUs
         * @param count Number of elements per buffer
         * @param dtype Data type
         * @param op Reduction operation
         * @return true on success
         */
        bool executeReduceScatterPatternPipelined(
            const std::vector<void *> &cuda_buffers,
            const std::vector<void *> &rocm_buffers,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op);

    private:
        // =====================================================================
        // Device Grouping
        // =====================================================================

        /// Separate input devices into CUDA and ROCm groups
        bool partitionDevices(const DeviceGroup &group);

        /// Select bridge devices (lowest ordinal from each vendor)
        void selectBridgeDevices();

        /// Validate the device group for heterogeneous operation
        bool validateGroup(const DeviceGroup &group);

        // =====================================================================
        // Sub-Backend Management
        // =====================================================================

        /// Create NCCL backend for CUDA devices (if needed)
        bool createNCCLBackend();

        /// Create RCCL backend for ROCm devices (if needed)
        bool createRCCLBackend();

        /// Create HOST backend for cross-vendor bridge
        bool createBridgeBackend();

        // =====================================================================
        // Member Variables
        // =====================================================================

        /// Initialization state
        bool initialized_ = false;

        /// Device groupings
        std::vector<DeviceId> cuda_devices_;
        std::vector<DeviceId> rocm_devices_;

        /// Bridge devices for cross-vendor transfers
        DeviceId cuda_bridge_;
        DeviceId rocm_bridge_;

        /// The full device group (stored for reference)
        DeviceGroup device_group_;

        /// Sub-backends
        std::unique_ptr<NCCLBackend> nccl_backend_;   ///< For CUDA domain (only if >1 CUDA)
        std::unique_ptr<RCCLBackend> rccl_backend_;   ///< For ROCm domain (only if >1 ROCm)
        std::unique_ptr<HostBackend> bridge_backend_; ///< For cross-domain (host-staged)

        /// Last error message
        std::string last_error_;
    };

#else

    // Stub implementation when not compiled with both CUDA and ROCm
    class HeterogeneousBackend : public ICollectiveBackend
    {
    public:
        CollectiveBackendType type() const override { return CollectiveBackendType::HETEROGENEOUS; }
        std::string name() const override { return "Heterogeneous (unavailable)"; }

        bool supportsDevice(DeviceType) const override { return false; }
        bool supportsDirectTransfer(DeviceId, DeviceId) const override { return false; }
        bool isAvailable() const override { return false; }

        bool initialize(const DeviceGroup &) override { return false; }
        bool isInitialized() const override { return false; }
        void shutdown() override {}

        bool isMultiGpuSingleProcess() const override { return false; }
        bool allreduce(void *, size_t, CollectiveDataType, CollectiveOp) override { return false; }
        bool allreduceMulti(const std::vector<void *> &, size_t, CollectiveDataType, CollectiveOp) override
        {
            return false;
        }
        bool allgather(const void *, void *, size_t, CollectiveDataType) override { return false; }
        bool allgatherv(const void *, size_t, void *, const std::vector<int> &,
                        const std::vector<int> &, CollectiveDataType) override
        {
            return false;
        }
        bool reduceScatter(const void *, void *, size_t, CollectiveDataType, CollectiveOp) override
        {
            return false;
        }
        bool broadcast(void *, size_t, CollectiveDataType, int) override { return false; }
        bool synchronize() override { return false; }

        // Test accessors (return empty/invalid for stub)
        const std::vector<DeviceId> &cudaDevices() const
        {
            static std::vector<DeviceId> empty;
            return empty;
        }
        const std::vector<DeviceId> &rocmDevices() const
        {
            static std::vector<DeviceId> empty;
            return empty;
        }
        DeviceId cudaBridge() const { return DeviceId::cpu(); }
        DeviceId rocmBridge() const { return DeviceId::cpu(); }
        bool hasNCCLBackend() const { return false; }
        bool hasRCCLBackend() const { return false; }
        bool hasBridgeBackend() const { return false; }

        // Allreduce pattern enum (stub version)
        enum class AllreducePattern
        {
            STANDARD_3PHASE,
            SYMMETRIC_REDUCE_SCATTER,
            PARTIAL_REDUCE_SCATTER,
            GCD_REDUCE_SCATTER,
            GCD_MULTI_BRIDGE
        };

        // Topology analysis struct (stub version)
        struct TopologyAnalysis
        {
            size_t cuda_count = 0;
            size_t rocm_count = 0;
            bool is_symmetric = false;
            bool is_minimal = false;
            bool is_cuda_singleton = false;
            bool is_rocm_singleton = false;
            size_t gcd = 1;
            AllreducePattern pattern = AllreducePattern::STANDARD_3PHASE;
            size_t num_chunks = 1;
            size_t bridge_traffic_fraction = 1;
            double intra_domain_parallelism = 1.0;
            double bridge_parallelism = 1.0;
            bool pipelining_eligible = false;
            size_t pipeline_chunks = 0;
            std::string reason;
        };

        TopologyAnalysis analyzeTopology(size_t /*tensor_bytes*/) const
        {
            TopologyAnalysis analysis;
            analysis.is_minimal = true;
            return analysis;
        }
    };

#endif // defined(HAVE_CUDA) && defined(HAVE_ROCM)

} // namespace llaminar2
