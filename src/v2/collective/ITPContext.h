/**
 * @file ITPContext.h
 * @brief Base interface for tensor parallelism contexts (local and global)
 *
 * ITPContext provides a common interface for both LOCAL TP (multiple devices within
 * a single MPI rank, using NCCL/RCCL/PCIeBAR) and GLOBAL TP (cross-MPI-rank, using
 * UPI/MPI collectives).
 *
 * The key difference:
 * - LOCAL TP: isLocal() returns true, uses high-bandwidth intra-node backends
 * - GLOBAL TP: isLocal() returns false, uses MPI-based cross-rank communication
 *
 * This interface enables stages like TPAllreduceStage to work with either TP type
 * without code changes.
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

    /**
     * @brief Base interface for tensor parallelism contexts
     *
     * Provides the minimal common interface shared by both LOCAL and GLOBAL TP contexts.
     * Local TP (ILocalTPContext) and Global TP (IGlobalTPContext) extend this interface
     * with their specific capabilities.
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
         * @brief Get number of participants in this TP domain
         *
         * For LOCAL TP: number of devices within the rank
         * For GLOBAL TP: number of MPI ranks in the domain
         *
         * @return TP degree (>= 1)
         */
        virtual int degree() const = 0;

        /**
         * @brief Get this participant's index within the domain
         *
         * For LOCAL TP: device index (0 to degree-1)
         * For GLOBAL TP: rank-in-domain (0 to degree-1)
         *
         * Used for sharding calculations and collective operations.
         *
         * @return Index in range [0, degree())
         */
        virtual int myIndex() const = 0;

        /**
         * @brief Check if this is a LOCAL TP context
         *
         * @return true if all participants are within the same MPI rank (intra-rank)
         * @return false if participants span multiple MPI ranks (cross-rank, global)
         */
        virtual bool isLocal() const = 0;

        /**
         * @brief Check if this is a GLOBAL TP context
         *
         * Convenience method, equivalent to !isLocal().
         *
         * @return true if participants span multiple MPI ranks
         * @return false if all participants are within the same MPI rank
         */
        bool isGlobal() const { return !isLocal(); }

        /**
         * @brief Get the collective backend type used by this context
         *
         * For LOCAL TP: Returns NCCL, RCCL, PCIE_BAR, etc.
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
         * Extended version for BAR-backed tensor lookup and partial reductions.
         * The stage_name allows PCIeBAR backends to locate pre-registered tensors.
         * The count allows reducing a subset of elements (useful for decode phase).
         *
         * Default implementation delegates to allreduce(tensor) ignoring extra params.
         * LocalTPContext provides specialized implementation with BAR support.
         *
         * @param tensor Tensor to all-reduce (modified in-place)
         * @param stage_name Stage identifier for BAR-backed tensor lookup
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
