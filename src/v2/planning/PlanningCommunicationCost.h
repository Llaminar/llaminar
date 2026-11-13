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
#include "PlanningInterconnectCostMatrix.h"
#include "PlanningPCIeCommunicationCost.h"
#include <optional>

namespace llaminar2
{
    /** @brief What relationship actually connects a query to its observation basis. */
    enum class PlanningCommunicationBasis
    {
        NoCommunication, ///< Exactly one native participant; no collective exists.
        SameProtocolPayloadCurve, ///< Same physical group/order or directed MPI pair; payload may be predicted.
        ReorderedNativeGroup, ///< Same GPU set, different order; full observed group cost, no speedup assumed.
        ContainingNativeGroup, ///< Unmeasured subset uses its containing group's complete cost, not a divided rate.
        PhysicalDirectedPeerPlacementProxy, ///< Physical destination-owned peer-copy curve; selection proxy, never collective latency.
        PhysicalHostStagedPlacementProxy, ///< Explicit measured source D2H plus destination H2D service.
        PhysicalGroupWorstPeerPlacementProxy, ///< Same directed-pair basis for every group, including exact measured groups.
        CalibratedNativeAllreduceProxy, ///< Measured native floor plus topology-scaled payload excess; not a subgroup benchmark.
        CalibratedNativeAllgatherProxy, ///< Native equal gather floor and measured bulk service, adjusted for physical placement.
        CalibratedNativeCountedAllgatherProxy, ///< Actual counted root sequence, including empty-owner control and completion.
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
         * @brief Retain measured native latency while adjusting payload service for the candidate's fabric.
         * @param discovery_rank Exact original observer owning both native and pair measurements.
         * @param devices Ordered homogeneous physical participants.
         * @param hidden_width Live model-facing FP32 columns.
         * @param rows Live rows, never workspace capacity.
         * @param precision Measured native wire representation.
         * @return Native small-payload floor plus topology-scaled excess, bounded by known PCIe cuts.
         *
         * One containing native group's smallest measured payload supplies the
         * positive-payload latency floor. Only cost above that floor is scaled
         * by candidate/reference pair service and shared-cut pressure. A faster
         * copy never scales away native control latency. The measured degree
         * is retained without an invented ring/tree or degree discount.
         * Unknown PCIe cuts remain explicit; measured physical services are
         * mandatory. Contradictory measured native service below a known wire
         * floor fails instead of silently fabricating a corrected observation.
         */
        PlanningCommunicationPrediction calibratedNativeAllreduce(int discovery_rank,
            std::span<const DeviceId> devices, int hidden_width, int rows,
            PlanningAllreducePrecision precision) const;

        /**
         * @brief Price the native equal-byte allgather used for completed output columns.
         * @param discovery_rank Original observer owning every endpoint and native receipt.
         * @param devices Ordered homogeneous participants.
         * @param bytes_per_owner Positive live source bytes, never receive-bank capacity.
         * @return Whole native gather floor plus topology-adjusted payload service.
         *
         * The observed native allgather algorithm supplies the timing basis;
         * it is not replaced with a loop of measured peer copies. Containing
         * groups retain the full control floor without a degree discount.
         * Equal device-row collectives share this payload-service estimate,
         * but their additional device-count binding is not timed by this probe.
         */
        PlanningCommunicationPrediction calibratedNativeAllgather(int discovery_rank,
            std::span<const DeviceId> devices, size_t bytes_per_owner) const;

        /**
         * @brief Price the production counted gather, including extent, completion and empty-owner progress.
         * @param discovery_rank Original observer owning every endpoint and native receipt.
         * @param devices Ordered homogeneous participants, including empty owners.
         * @param owner_live_bytes Exact live source fragments; all-zero data still requires control.
         * @return Sum of measured per-root service estimates, bounded by live PCIe cut demand.
         *
         * The complete zero-payload and bulk counted invocations include frame
         * preparation, ascending root broadcasts and validation. Their mean
         * per original root prices unequal owners without discarding empty
         * participants. A changed degree is explicitly an estimate: the one
         * frame cost is amortized with the measured root sequence. Callers must
         * not add another extent gather. Only payload excess is adjusted using
         * bulk physical service; tiny copy latency never replaces native control.
         */
        PlanningCommunicationPrediction calibratedNativeCountedAllgather(int discovery_rank,
            std::span<const DeviceId> devices, std::span<const size_t> owner_live_bytes) const;

        /**
         * @brief Price directed live bytes on the exact observer's measured physical peer-copy basis.
         * @param discovery_rank Original observer; another rank's physical aliases cannot substitute.
         * @param source Original rank-visible GPU sender.
         * @param destination Distinct same-backend rank-visible GPU receiver.
         * @param payload_bytes Positive live bytes, excluding capacity padding.
         * @return Explicit placement proxy with original direction and packetization provenance.
         *
         * An enabled direct-copy edge uses its destination-owned measured clock curve.
         * An observed absent peer edge uses measured source-to-host DMA followed
         * by measured same-rank-host-to-destination DMA, never an RCCL sample.
         * Below the smallest direct-copy anchor, charge that anchor's service.
         * Above the largest anchor, charge whole maximum-sized chunks plus the
         * exact remainder, using bounded interpolation inside the measured range.
         * This is a sequential packetization surrogate, not an observed graph
         * schedule or an extrapolated bandwidth. Missing evidence fails hard.
         */
        PlanningCommunicationPrediction physicalPeerTransfer(int discovery_rank,
            DeviceId source, DeviceId destination, size_t payload_bytes) const;

        /**
         * @brief Price a group's worst directed live-payload service on one common pair basis.
         * @param discovery_rank Original observer owning every endpoint and its matrix.
         * @param devices Distinct homogeneous physical GPUs, any positive group size.
         * @param payload_bytes Positive operation payload; caller owns model traffic geometry.
         * @return Zero for a validated singleton, otherwise the worst directed pair proxy.
         *
         * Every group uses direct-copy or explicit two-leg host evidence, including groups with exact collective
         * observations. This avoids comparing an exact small collective against
         * a larger containing-group proxy. It is neither RCCL/NCCL latency nor
         * an upper bound, ring/tree schedule, aggregate wire traffic or measured
         * end-to-end throughput; nativeAllreduce retains protocol observations.
         */
        PlanningCommunicationPrediction physicalGroupPlacement(int discovery_rank,
            std::span<const DeviceId> devices, size_t payload_bytes) const;

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
         * @param gpu_rank Original NUMA-local rank authorized to execute this GPU.
         * @param device GPU ordinal in gpu_rank's namespace.
         * @param host_rank Same-physical-node rank whose setup thread first-touched the pages.
         * @param mechanism DMA or mapped-kernel access; neither substitutes for the other.
         * @param direction Actual byte movement direction.
         * @param payload_bytes Positive live traffic, excluding unused staging capacity.
         * @return Qualified primitive estimate, excluding packet routing, waits and remote MPI service.
         *
         * GPU ordinals may alias between discovery namespaces. The exact executor
         * and first-touch pair must still have its own completed observation;
         * another rank's GPU-local curve cannot supply missing cross-role evidence.
         */
        PlanningCommunicationPrediction hostTransfer(int gpu_rank, DeviceId device, int host_rank,
            PlanningHostTransferMechanism mechanism, MappedTransferDirection direction, size_t payload_bytes) const;

    private:
        /** @brief Original immutable observation and its relationship to a requested native group. */
        struct NativeReference
        {
            const PlanningNativeCollectiveService *service;
            PlanningCommunicationBasis relationship;
        };
        /**
         * @brief Resolve native membership once for reductions and gathers.
         * @param discovery_rank Canonical physical observer.
         * @param devices Ordered native query endpoints, validated even for a singleton.
         * @param precision Required reduction wire type; byte gathers have no floating precision.
         * @param exact_observer Whether another rank's physical alias may supply the observation.
         * @return Null only for a validated singleton; otherwise exact, reordered or containing evidence.
         */
        NativeReference nativeReference(int discovery_rank, std::span<const DeviceId> devices,
            std::optional<PlanningAllreducePrecision> precision, bool exact_observer) const;
        /**
         * @brief Compose one actual gather family from its immutable native anchors and physical placement.
         * @param discovery_rank Exact original observer.
         * @param devices Native ordered endpoints.
         * @param owner_live_bytes Live source fragments; fixed gathers require equal positive sizes.
         * @param protocol Actual production operation, never inferred from allocation capacity.
         * @return Qualified native service with required control charged once.
         */
        PlanningCommunicationPrediction calibratedGather(int discovery_rank, std::span<const DeviceId> devices,
            std::span<const size_t> owner_live_bytes, PlanningNativeGatherProtocol protocol) const;
        /** @brief Selected immutable native receipt and its payload-curve prediction. */
        struct NativeAllreducePrice
        {
            PlanningCommunicationPrediction prediction;
            const PlanningNativeCollectiveService *reference;
            size_t payload;
        };
        /** @return Checked curve/receipt; calibrated queries require the exact original observer. */
        NativeAllreducePrice priceNativeAllreduce(int discovery_rank, std::span<const DeviceId> devices,
            int hidden_width, int rows, PlanningAllreducePrecision precision, bool exact_observer) const;
        /** @return Checked rank record; getRank's empty sentinel is not admissible cost evidence. */
        const RankInventory &rank(int discovery_rank) const;
        /** @return Exact observed GPU identity; duplicate or missing aliases fail before pricing. */
        const DeviceInfo &gpu(int discovery_rank, DeviceId device) const;
        ClusterInventory inventory_;
        std::vector<PlanningNativeCollectiveService> native_;
        std::vector<PlanningMPITransferObservation> mpi_;
        std::vector<PlanningHostDeviceObservations> host_device_;
        /** @brief One immutable precision-independent matrix per exact original observer/backend. */
        struct PhysicalPeerMatrix
        {
            PlanningInterconnectCostMatrix matrix;
            size_t minimum_payload, maximum_payload;
        };
        /** @return Numeric directed proxy using already validated physical UUIDs and one immutable matrix. */
        static double directPeerSeconds(const PhysicalPeerMatrix &basis,
            const std::string &source_uuid, const std::string &destination_uuid, size_t payload_bytes);
        /** @return One final provenance string; numeric group-edge evaluation never formats evidence. */
        static std::string directPeerEvidence(const PhysicalPeerMatrix &basis,
            const std::string &source_uuid, const std::string &destination_uuid, size_t payload_bytes);
        /** @return Direct peer measurement or explicit same-rank-host DMA composition, without evidence formatting. */
        double physicalEdgeSeconds(const PhysicalPeerMatrix &basis, DeviceId source, DeviceId destination,
            const std::string &source_uuid, const std::string &destination_uuid, size_t payload_bytes) const;
        /** @brief Numeric host primitive and original page-owner affinity, resolved once per query. */
        struct HostTransferPrice { double seconds; int numa; };
        /** @return One existing measured host primitive without allocating provenance strings. */
        HostTransferPrice hostTransferPrice(int gpu_rank, DeviceId device, int host_rank,
            PlanningHostTransferMechanism mechanism, MappedTransferDirection direction, size_t payload_bytes) const;
        std::map<std::pair<int, DeviceType>, PhysicalPeerMatrix> physical_peers_;
        std::map<int, PlanningPCIeCommunicationCost> pcie_;
    };
}
