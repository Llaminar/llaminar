/**
 * @file ITPContext.h
 * @brief Base interface for tensor parallelism contexts
 *
 * ITPContext provides a common interface for all TP scopes:
 *
 * - **LOCAL**:      Intra-rank, multi-device. All participants are devices owned
 *                   by a single MPI rank. Uses NVLink, HOST, or intra-process
 *                   NCCL/RCCL. Lowest latency, highest bandwidth.
 *
 * - **NODE_LOCAL**: Cross-rank, same physical machine. Participants are MPI ranks
 *                   on the same node, communicating via UPI, shared memory, or
 *                   cross-process NCCL. Medium latency, high bandwidth.
 *
 * - **GLOBAL**:     Cross-node, cross-rank. Participants span physical machines,
 *                   communicating via MPI over InfiniBand or Ethernet.
 *                   Highest latency, lowest bandwidth.
 *
 * A nested PP+TP topology might combine all three:
 *
 *   NodeLocalPipelineParallel(
 *       LocalTP(0:cuda:0, 0:cuda:1, 0:cuda:2, 0:cuda:3),
 *       LocalTP(1:cuda:0, 1:cuda:1, 1:cuda:2, 1:cuda:3),
 *       NodeLocalTP(0:cpu, 1:cpu)
 *   )
 *
 * This interface enables stages like TPAllreduceStage to work with any scope
 * without code changes — the scope only matters for backend selection and
 * scope-specific setup (e.g., buffer registration for LOCAL TP).
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include <string>
#include "config/OrchestrationConfig.h" // For CollectiveBackendType

namespace llaminar2
{

    // Forward declarations
    class TensorBase;

    // TPScope is defined in config/OrchestrationConfig.h (included above):
    //   LOCAL, NODE_LOCAL, GLOBAL, AUTO, HYBRID

    /**
     * @brief Base interface for tensor parallelism contexts
     *
     * Provides the minimal common interface shared by all TP context scopes:
     * - ILocalTPContext (LOCAL scope)
     * - INodeLocalTPContext (NODE_LOCAL scope)
     * - IGlobalTPContext (GLOBAL scope)
     *
     * Thread safety: All implementations must be thread-safe for collective operations.
     */
    class ITPContext
    {
    public:
        virtual ~ITPContext() = default;

        // =========================================================================
        // Identity & Configuration
        // =========================================================================

        /**
         * @brief Get the scope of this TP context
         *
         * @return TPScope indicating LOCAL, NODE_LOCAL, or GLOBAL
         */
        virtual TPScope scope() const = 0;

        /**
         * @brief Check if this is a LOCAL TP context (intra-rank)
         *
         * Convenience method. Equivalent to scope() == TPScope::LOCAL.
         * Code needing LOCAL TP-specific features (BAR registration, device lists)
         * should check this before static_cast<ILocalTPContext*>.
         */
        bool isLocal() const { return scope() == TPScope::LOCAL; }

        /**
         * @brief Check if this is a NODE_LOCAL TP context (cross-rank, same node)
         *
         * Convenience method. Equivalent to scope() == TPScope::NODE_LOCAL.
         */
        bool isNodeLocal() const { return scope() == TPScope::NODE_LOCAL; }

        /**
         * @brief Check if this is a GLOBAL TP context (cross-node)
         *
         * Convenience method. Equivalent to scope() == TPScope::GLOBAL.
         */
        bool isGlobal() const { return scope() == TPScope::GLOBAL; }

        /**
         * @brief Get number of participants in this TP domain
         *
         * For LOCAL TP: number of devices within the rank
         * For NODE_LOCAL TP: number of MPI ranks on the same node
         * For GLOBAL TP: number of MPI ranks in the domain
         *
         * @return TP degree (>= 1)
         */
        virtual int degree() const = 0;

        /**
         * @brief Get this participant's index within the domain
         *
         * For LOCAL TP: device index (0 to degree-1)
         * For NODE_LOCAL TP: rank-in-domain on the node (0 to degree-1)
         * For GLOBAL TP: rank-in-domain (0 to degree-1)
         *
         * Used for sharding calculations and collective operations.
         *
         * @return Index in range [0, degree())
         */
        virtual int myIndex() const = 0;

        /**
         * @brief Get the collective backend type used by this context
         *
         * For LOCAL TP: Returns NCCL, RCCL, HOST, etc.
         * For NODE_LOCAL TP: Returns NCCL, MPI, HOST, etc.
         * For GLOBAL TP: Returns MPI, UPI, etc.
         *
         * @return CollectiveBackendType enum value
         */
        virtual CollectiveBackendType backend() const = 0;

        // =========================================================================
        // Core Collective Operations
        // =========================================================================

        /**
         * @brief All-reduce sum across all participants (in-place)
         *
         * Performs element-wise sum reduction, leaving the same result on all participants.
         * This is the core operation after row-parallel GEMM (e.g., Wo projection, FFN down).
         *
         * @param tensor Tensor to all-reduce (modified in-place, must be same size on all participants)
         * @return true on success, false on error
         */
        virtual bool allreduce(TensorBase *tensor) = 0;

        /**
         * @brief All-reduce sum with stage name and count (in-place)
         *
         * Extended version with stage name and partial reduction support.
         * The count allows reducing a subset of elements (useful for decode phase).
         *
         * Default implementation delegates to allreduce(tensor) ignoring extra params.
         *
         * @param tensor Tensor to all-reduce (modified in-place)
         * @param stage_name Stage identifier for tensor lookup
         * @param count Elements to reduce (0 = use tensor->numel())
         * @return true on success, false on error
         */
        virtual bool allreduce(TensorBase *tensor, const std::string &stage_name, size_t count = 0)
        {
            (void)stage_name;
            (void)count;
            // Default: delegate to simple allreduce, ignore stage_name/count
            return allreduce(tensor);
        }

        /**
         * @brief All-reduce sum on a specific GPU stream (graph-capturable)
         *
         * Like allreduce() but issues the collective directly on the provided
         * GPU stream. This makes the operation compatible with GPU graph capture.
         * When stream is nullptr, falls back to the normal allreduce() path.
         *
         * @param tensor Tensor to all-reduce (modified in-place)
         * @param stage_name Stage identifier
         * @param count Elements to reduce (0 = use tensor->numel())
         * @param stream GPU stream (hipStream_t/cudaStream_t cast to void*), or nullptr
         * @param precision Override precision ("fp32", "fp16", "bf16"). Empty string = use global default.
         * @return true on success, false on error
         */
        virtual bool allreduceOnStream(TensorBase *tensor, const std::string &stage_name,
                                       size_t count, void *stream,
                                       const std::string &precision = "")
        {
            (void)stream;
            (void)precision;
            // Default: delegate to normal allreduce, ignoring stream and precision
            return allreduce(tensor, stage_name, count);
        }

        /**
         * @brief Broadcast tensor from source to all participants
         *
         * Copies tensor from source_index participant to all others.
         *
         * @param tensor Tensor to broadcast (overwritten on non-source participants)
         * @param source_index Index of the source participant (default 0)
         * @return true on success, false on error
         */
        virtual bool broadcast(TensorBase *tensor, int source_index = 0) = 0;

        /**
         * @brief All-gather: collect shards from all participants into full tensor
         *
         * Each participant contributes its local_shard, concatenated in index order
         * to produce global_tensor on all participants.
         *
         * @param local_shard This participant's shard (may differ in size for proportional TP)
         * @param global_tensor Pre-allocated output for concatenated result
         * @return true on success, false on error
         */
        virtual bool allgather(const TensorBase *local_shard, TensorBase *global_tensor) = 0;
    };

} // namespace llaminar2
