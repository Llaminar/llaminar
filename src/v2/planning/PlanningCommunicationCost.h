/**
 * @file PlanningCommunicationCost.h
 * @brief Pure, topology-bound communication predictions from bounded startup evidence.
 *
 * Native allreduce, host request/reply and GPU/host byte primitives remain
 * different pricing operations. GPU DMA and mapped-kernel access have their
 * own measured directions and first-touch scopes, not an inferred MPI rate. The model
 * retains measured receipts unchanged and labels subset/reordered-group proxies
 * explicitly; it never presents an unobserved candidate as a benchmark result.
 */
#pragma once
#include "PlanningCommunicationService.h"
#include "OrchestrationPerformanceEvidence.h"

namespace llaminar2
{
    /** @brief What relationship actually connects a query to its observation basis. */
    enum class PlanningCommunicationBasis
    {
        NoCommunication, ///< Exactly one native participant; no collective exists.
        SameProtocolPayloadCurve, ///< Same physical group/order or directed MPI pair; payload may be predicted.
        ReorderedNativeGroup, ///< Same GPU set, different order; full observed group cost, no speedup assumed.
        ContainingNativeGroup, ///< Unmeasured subset uses its containing group's complete cost, not a divided rate.
    };

    /** @brief Estimated seconds and explicit provenance; never measured whole-model throughput. */
    struct PlanningCommunicationPrediction
    {
        double seconds;
        PlanningCommunicationBasis basis;
        std::string evidence;
    };

    /**
     * @brief Immutable cost queries with no device work, execution allocations, probes or transport selection.
     *
     * Native subgroups use the narrowest observed containing physical group
     * without a guessed degree discount. This is an explicitly qualified proxy,
     * not a guaranteed upper bound: native libraries can change algorithms with
     * group size. MPI separates outbound and return incremental service while
     * retaining one complete measured control RTT. Locality is inventory-owned.
     */
    class PlanningCommunicationCost final
    {
    public:
        /**
         * @brief Bind completed observations to the same discovery inventory as candidate compilation.
         * @param inventory Canonical physical membership, rank-local ordinals and UUID aliases.
         * @param native Complete native receipts, including exact endpoint order and precision.
         * @param mpi Complete control/outbound-heavy/return-heavy basis for every included directed pair.
         * @param host_device Complete per-GPU/rank-first-touch DMA and mapped-kernel byte primitives.
         * @throws std::invalid_argument for incomplete, duplicate or physically substituted evidence.
         *
         * Empty families are valid when the requested search has no such
         * transport. A query needing absent evidence fails, rather than using
         * another backend, peer, precision or protocol. Copies retain metadata
         * only and never become another live inventory or allocation authority.
         */
        PlanningCommunicationCost(const ClusterInventory &inventory,
            std::span<const PlanningNativeCollectiveService> native,
            std::span<const PlanningMPITransferObservation> mpi,
            std::span<const PlanningHostDeviceObservations> host_device = {});

        /**
         * @brief Price an in-place FP32 activation allreduce over a homogeneous rank-visible GPU group.
         * @param discovery_rank Process namespace in the canonical inventory, not execution rank.
         * @param devices Ordered local ordinals visible to that process.
         * @param hidden_width Logical FP32 columns; transport precision changes wire bytes only.
         * @param rows Logical activation rows, not the allocated buffer capacity.
         * @param precision Actual native collective wire precision.
         * @return Zero only for one validated participant; otherwise a qualified positive curve prediction.
         *
         * Physical aliases are resolved by node/backend/UUID. An identical
         * ordinal or UUID on another machine does not supply this group's cost.
         */
        PlanningCommunicationPrediction nativeAllreduce(int discovery_rank,
            std::span<const DeviceId> devices, int hidden_width, int rows,
            PlanningAllreducePrecision precision) const;

        /**
         * @brief Price one complete ordered host-MPI request/reply, including one control RTT.
         * @param request Positive exact outbound/return bytes and discovery endpoint order.
         * @return Positive prediction with physical locality and protocol limitations in its evidence.
         *
         * Each direction's payload contribution uses its own completed sample.
         * There is no division by two, reverse-pair substitution, inferred node
         * membership, GPU staging charge or claim of measured asymmetric latency.
         */
        PlanningCommunicationPrediction mpiExchange(const PlanningMPITransferRequest &request) const;

        /**
         * @brief Price one GPU/host byte primitive with an explicit mechanism, direction and page owner.
         * @param gpu_rank Discovery rank whose visible GPU will issue the operation.
         * @param device GPU ordinal in gpu_rank's namespace.
         * @param host_rank Same-physical-node rank whose setup thread first-touched the pages.
         * @param mechanism DMA or mapped-kernel access; neither substitutes for the other.
         * @param direction Actual byte movement direction.
         * @param payload_bytes Positive live traffic, excluding unused staging capacity.
         * @return Qualified primitive estimate, excluding packet routing, waits and remote MPI service.
         *
         * GPU aliases may differ between ranks. Host scope may not: an absent
         * first-touch observation is an error, not a guessed GPU-local sample.
         */
        PlanningCommunicationPrediction hostTransfer(int gpu_rank, DeviceId device, int host_rank,
            PlanningHostTransferMechanism mechanism, MappedTransferDirection direction, size_t payload_bytes) const;

    private:
        /** @return Checked rank record; getRank's empty sentinel is not admissible cost evidence. */
        const RankInventory &rank(int discovery_rank) const;
        /** @return Exact observed GPU identity; duplicate or missing aliases fail before pricing. */
        const DeviceInfo &gpu(int discovery_rank, DeviceId device) const;
        ClusterInventory inventory_;
        std::vector<PlanningNativeCollectiveService> native_;
        std::vector<PlanningMPITransferObservation> mpi_;
        std::vector<PlanningHostDeviceObservations> host_device_;
    };
}
