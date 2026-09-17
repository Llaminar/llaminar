/**
 * @file PlanningKernelServiceCatalog.h
 * @brief One topology-authenticated catalog for ordinary projection and expert service.
 *
 * Discovery owns physical identity. MPI owns the identity of the reporting rank.
 * A sampler supplies only completed work and execution geometry. The catalog
 * joins these three facts without discovering hardware, inferring locality from
 * speed, or maintaining another allocation ledger. Ordinary projections and
 * complete experts share this collection lifecycle, but retain distinct typed
 * work identities. Streaming residual-add has its own byte-service identity;
 * neither it nor a source kernel is collective or full-model timing.
 */
#pragma once
#include "PlanningCPUExpertMeasurement.h"
#include "PlanningGPUExpertMeasurement.h"
#include "PlanningProjectionMeasurement.h"
#include "PlanningMemoryBandwidthMeasurement.h"
#include "PlanningPublication.h"
#include "config/OrchestrationPlanningPolicy.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include <variant>
#include <map>
#include <tuple>

namespace llaminar2
{
    /** @brief Ordinary projection source and immutable positive prefill sample geometry. */
    class PlanningProjectionServicePlan final
    {
    public:
        /** @brief Validate row capacity before publication or any rank enters admission. */
        PlanningProjectionServicePlan(PlanningMatrixSamplePlan matrix, int prefill_rows);
        /** @return Sealed whole matrix or actual source-axis shard. */
        const PlanningMatrixSamplePlan &matrix() const noexcept { return matrix_; }
        /** @return Physical M measured in addition to ordinary single-row decode. */
        int prefillRows() const noexcept { return prefill_rows_; }
    private:
        PlanningMatrixSamplePlan matrix_;
        int prefill_rows_;
    };

    /** @brief Source-free streaming request; exact geometry belongs to each inventory endpoint. */
    struct PlanningStreamingServicePlan final {};
    /** @brief Mutually exclusive expert, source projection, FP32 arithmetic proxy or streaming service. */
    using PlanningKernelSamplePlan = std::variant<PlanningExpertSamplePlan, PlanningProjectionServicePlan,
        PlanningFP32ArithmeticPlan, PlanningStreamingServicePlan>;
    /** @brief Native source owner retained through the matching service transaction. */
    using PlanningLoadedKernelSample = std::variant<PlanningLoadedExpertSample, PlanningLoadedMatrixSample, std::monostate>;
    /** @brief Different operation/regime observations can never silently substitute for one another. */
    using PlanningKernelObservation =
        std::variant<PlanningCPUExpertObservations, PlanningGPUExpertObservations, PlanningProjectionObservations,
            PlanningFP32ArithmeticObservations, PlanningMemoryBandwidthObservation>;

    /** @brief Immutable binding to an observed physical endpoint and its designated reporter. */
    class PlanningServiceObserver final
    {
    public:
        /** @return Discovery rank whose ordinary worker measures this endpoint. */
        int discoveryRank() const noexcept { return rank_; }
        /** @return Physical machine from inventory, never a timing classification. */
        int physicalNode() const noexcept { return node_; }
        /** @return Rank-local allocator/ordinal; CPU NUMA scope is reported separately. */
        DeviceId device() const noexcept { return device_; }
        /** @return Observed CPU workshare/GPU affinity; -1 explicitly means unspecified. */
        int numaNode() const noexcept { return numa_; }
        /** @return Inventory UUID for a GPU, empty for a rank's CPU workshare. */
        const std::string &uuid() const noexcept { return uuid_; }
        /** @brief Exact identity comparison includes the designated reporting rank. */
        bool operator==(const PlanningServiceObserver &) const = default;
    private:
        friend class PlanningKernelServiceCatalog;
        /** @brief Only the inventory resolver may create a reporter binding. */
        PlanningServiceObserver(int rank, int node, DeviceId device, int numa, std::string uuid)
            : rank_(rank), node_(node), device_(device), numa_(numa), uuid_(std::move(uuid)) {}
        int rank_, node_;
        DeviceId device_;
        int numa_;
        std::string uuid_;
    };

    /** @brief Completed service with its independently authenticated physical identity. */
    struct PlanningKernelServiceRecord final
    {
        PlanningServiceObserver observer;
        PlanningKernelObservation observation;
    };

    /** @brief Sealed complete evidence batch, returned only to the selecting root. */
    class PlanningKernelServiceCatalog final
    {
    public:
        /**
         * @brief Choose one observer per physical GPU and each permitted rank CPU workshare.
         * @param inventory Canonical discovery snapshot, including visibility aliases.
         * @param request Hard backend constraints; preferences never change measurements.
         * @return Deterministic rank/device order. GPU ownership prefers observed NUMA
         *         affinity, then discovery rank, without assigning a vendor to a socket.
         *
         * CPU measurements remain rank-qualified: a whole-host team must never be
         * relabeled as a socket or borrowed by another rank with a different team.
         */
        static std::vector<PlanningServiceObserver> observers(const ClusterInventory &inventory,
            const AutomaticOrchestrationRequest &request);

        /** @return Versioned completed observations, with sealed source identity but no claimed rank/node. */
        static std::vector<uint8_t> encode(const PlanningKernelSamplePlan &plan,
            std::span<const PlanningKernelObservation> observations);

        /**
         * @brief Authenticate complete rank receipts against source and inventory.
         * @param receipts Transport-owned rank IDs, never IDs decoded from payload.
         * @throws std::exception on missing/duplicate observations, foreign
         *         devices, wrong source/format/phase/geometry or invalid execution evidence.
         * @return Immutable catalog; partial results cannot be priced.
         */
        static PlanningKernelServiceCatalog accept(const ClusterInventory &inventory,
            const AutomaticOrchestrationRequest &request, const PlanningKernelSamplePlan &plan,
            std::span<const RankPlanningSample> receipts);

        /**
         * @brief Admit, distribute and measure one exact kernel source on the discovery communicator.
         * @param mpi Exact discovery context; null explicitly denotes one local rank.
         * @param inventory Same canonical inventory used by both plan and serve.
         * @param request Hard participant constraints used for the observation batch.
         * @param plan Previously published source plan; every envelope authenticates it.
         * @param root_load Root-only reader receiving the admitted rank authority;
         *        required for source kernels and forbidden for source-free arithmetic/streaming.
         * @return Complete catalog at root; followers return absence after consensus.
         *
         * Call on all ranks before root-only candidate selection. Allocation
         * admission, payload publication and completed observations have separate
         * collective failure boundaries. Samplers run serially within a rank to
         * avoid measuring self-contention and share the PMA's per-owner envelope;
         * GPU work runs on its normal owning worker, never in a native MPI progress callback.
         */
        static std::optional<PlanningKernelServiceCatalog> collect(
            const std::shared_ptr<IMPIContext> &mpi, const ClusterInventory &inventory,
            const AutomaticOrchestrationRequest &request, const PlanningKernelSamplePlan &plan,
            const std::function<PlanningLoadedKernelSample(const std::shared_ptr<PhysicalMemoryAuthority> &)> &root_load = {});

        /** @return Source geometry/format identity applicable to every record. */
        const PlanningKernelSamplePlan &source() const noexcept { return source_; }
        /** @return Complete immutable evidence, not a mutable performance cache. */
        const std::vector<PlanningKernelServiceRecord> &records() const noexcept { return records_; }
        /**
         * @brief Resolve a candidate endpoint against the inventory that authenticated this catalog.
         * @param discovery_rank Rank before execution membership selection, not an execution rank ID.
         * @param device Ordinal visible to that discovery rank.
         * @return The designated physical GPU reporter or that exact rank's CPU workshare.
         * @throws std::out_of_range for excluded, missing or foreign endpoints.
         *
         * Visibility aliases share GPU compute evidence only when their physical
         * node, backend and UUID agree. CPU workshares are never aliased across
         * ranks. The retained index contains no new hardware facts or measurements.
         */
        const PlanningKernelServiceRecord &serviceFor(int discovery_rank, DeviceId device) const;
    private:
        using EndpointKey = std::tuple<int, DeviceType, int>;
        /** @brief Only complete, validated receipt acceptance creates a catalog. */
        PlanningKernelServiceCatalog(PlanningKernelSamplePlan source,
            std::vector<PlanningKernelServiceRecord> records, std::map<EndpointKey, size_t> endpoints)
            : source_(std::move(source)), records_(std::move(records)), endpoints_(std::move(endpoints)) {}
        PlanningKernelSamplePlan source_;
        std::vector<PlanningKernelServiceRecord> records_;
        std::map<EndpointKey, size_t> endpoints_; ///< Derived references into records_, not a second inventory.
    };
}
