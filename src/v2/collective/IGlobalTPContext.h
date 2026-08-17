/**
 * @file IGlobalTPContext.h
 * @brief Interface for GLOBAL tensor parallelism (cross-MPI-rank)
 *
 * IGlobalTPContext extends ITPContext for tensor parallelism that spans multiple
 * MPI ranks. This is used for CPU-only cross-socket/cross-node TP via UPI/MPI.
 *
 * Key differences from LOCAL TP:
 * - Participants are MPI ranks (not local devices)
 * - Communication uses MPI collectives (not NCCL/RCCL)
 * - Has an MPI communicator for the domain
 * - Supports point-to-point operations (send/recv)
 *
 * Design constraints (from project plan):
 * - Global TP is CPU-only (GPUs use LOCAL TP for performance)
 * - Same physical machine only (UPI interconnect ~50 GB/s)
 * - Equal 1/n weight distribution (no proportional weights)
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "ITPContext.h"
#include "../backends/GlobalDeviceAddress.h"
#include <mpi.h>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Interface for GLOBAL tensor parallelism across MPI ranks
     *
     * Extends ITPContext for cross-rank communication via MPI. Participants are
     * identified by their rank within the domain communicator (0 to degree-1).
     *
     * Global TP is designed for CPU-only tensor parallelism where:
     * - Multiple sockets participate in sharded computation
     * - Communication happens over UPI interconnect (~50 GB/s)
     * - MPI provides the collective communication primitives
     *
     * Thread safety: Single thread should call methods on a given instance.
     * Multiple instances (different domains) can be used concurrently.
     */
    class IGlobalTPContext : public ITPContext
    {
    public:
        ~IGlobalTPContext() override = default;

        // =========================================================================
        // ITPContext Implementation
        // =========================================================================

        /**
         * @brief GLOBAL TP is always cross-node
         * @return Always TPScope::GLOBAL for GLOBAL TP contexts
         */
        TPScope scope() const override { return TPScope::GLOBAL; }

        // =========================================================================
        // Global-Specific Configuration
        // =========================================================================

        /**
         * @brief Get the MPI communicator for this TP domain
         *
         * This is NOT MPI_COMM_WORLD - it's a domain-specific communicator
         * created via MPI_Comm_split that includes only the ranks participating
         * in this global TP domain.
         *
         * @return MPI communicator for collective operations
         */
        virtual MPI_Comm communicator() const = 0;

        /**
         * @brief Get the global TP domain identifier
         *
         * Domain IDs are used to distinguish different global TP groups
         * when multiple are active (e.g., different PP stages using different
         * global TP configurations).
         *
         * @return Domain ID (from RankExecutionPlan or configuration)
         */
        virtual int domainId() const = 0;

        /**
         * @brief Get all MPI world ranks participating in this domain
         *
         * These are ranks in MPI_COMM_WORLD, not ranks in the domain communicator.
         * Useful for logging and debugging.
         *
         * @return Vector of world ranks (size == degree())
         */
        virtual const std::vector<int> &worldRanks() const = 0;

        // =========================================================================
        // Global-Specific Collective Operations
        // =========================================================================

        /**
         * @brief Get the local rank's device address for this global TP domain
         *
         * Returns a GlobalDeviceAddress representing this MPI rank's CPU device
         * in the global TP domain. This is used by HierarchicalPP to identify
         * the "representative device" for PP transfers.
         *
         * The returned address uses the world rank as the rank component,
         * NUMA node 0, and CPU device type.
         *
         * @return GlobalDeviceAddress for local CPU (e.g., "rank0:0:cpu:0")
         */
        virtual GlobalDeviceAddress localDevice() const = 0;

        /**
         * @brief Barrier synchronization across all ranks in domain
         *
         * Blocks until all ranks in the domain have reached this point.
         * Uses MPI_Barrier on the domain communicator.
         */
        virtual void barrier() const = 0;

        /**
         * @brief All-gather fixed-size scalar/control payloads across the TP domain.
         *
         * Use this for small coordination records, not tensor payloads. Tensor
         * data should continue to move through allgather(), broadcast(),
         * TransferEngine, or graph-stage buffer contracts.
         *
         * @param send_data Local byte payload
         * @param recv_data Output buffer sized degree() * byte_count
         * @param byte_count Bytes contributed by each participant
         * @return true on success, false when unsupported or invalid
         */
        virtual bool allgatherBytes(const void *send_data, void *recv_data, size_t byte_count) const
        {
            (void)send_data;
            (void)recv_data;
            (void)byte_count;
            return false;
        }

        /**
         * @brief Gather variable-count fixed-width FP32 records to one root.
         *
         * Every participant contributes `local_record_count` contiguous records
         * of `record_width_elements` floats. The root-local block appears first;
         * remote participant blocks may follow in arrival order. Callers must
         * therefore carry an explicit record identity and must not infer semantic
         * order from packed position. Non-root receive storage is ignored.
         * Implementations must use preallocated protocol metadata and must not
         * emulate this transaction with a dense allreduce or a preliminary count
         * collective.
         *
         * This operation is intentionally rooted: canonical CPU MoE publication
         * performs its ordered arithmetic once, then broadcasts a compact final
         * row. Replicating sparse records to every participant would restore the
         * avoidable `top_k`-scaled communication cost.
         *
         * @param local_records Contiguous local record payload.
         * @param local_record_count Number of records contributed locally.
         * @param root_records Root receive storage; may alias `local_records`.
         * @param root_record_capacity Capacity of root storage in records.
         * @param record_width_elements Width of one record in FP32 elements.
         * @param root_index Fixed root participant index.
         * @param gathered_record_count Total records received on root; zero on
         *        non-root participants.
         * @param stage_name Stable graph-stage identity for diagnostics.
         * @return true only after the complete rooted gather finishes.
         */
        virtual bool gatherVariableFloatRecordsToRoot(
            const float *local_records,
            size_t local_record_count,
            float *root_records,
            size_t root_record_capacity,
            size_t record_width_elements,
            int root_index,
            size_t &gathered_record_count,
            const std::string &stage_name) = 0;

        /**
         * @brief Broadcast an exact FP32 tensor prefix from one root.
         *
         * Unlike `broadcast(TensorBase*)`, this method carries the active
         * element count explicitly. Bucket-capacity padding and unrelated arena
         * tail storage must not become collective payload by accident.
         *
         * @param tensor Preallocated host tensor on every participant.
         * @param element_count Number of leading FP32 elements to broadcast.
         * @param root_index Fixed source participant index.
         * @param stage_name Stable graph-stage identity for diagnostics.
         * @return true only when the backend completes the broadcast.
         */
        virtual bool broadcastFloatElements(
            TensorBase *tensor,
            size_t element_count,
            int root_index,
            const std::string &stage_name) = 0;

        /**
         * @brief Point-to-point send to another rank in domain
         *
         * Blocking send of tensor data to another participant in this domain.
         * dest_index is the domain-local index (0 to degree-1), NOT world rank.
         *
         * @param tensor Tensor to send (must be on CPU memory)
         * @param dest_index Destination index within domain (0 to degree-1)
         * @return true on success, false on error
         */
        virtual bool send(const TensorBase *tensor, int dest_index) = 0;

        /**
         * @brief Point-to-point receive from another rank in domain
         *
         * Blocking receive of tensor data from another participant.
         * source_index is the domain-local index (0 to degree-1), NOT world rank.
         *
         * @param tensor Tensor to receive into (must be pre-allocated on CPU memory)
         * @param source_index Source index within domain (0 to degree-1)
         * @return true on success, false on error
         */
        virtual bool recv(TensorBase *tensor, int source_index) = 0;
    };

} // namespace llaminar2
