/**
 * @file PlanningLocalTPMeasurement.h
 * @brief Bounded native rank-local collective observations for topology costing.
 *
 * A request names one concrete homogeneous local GPU group. LocalTPContext
 * selects and executes its production NCCL/RCCL protocol, including precision
 * conversion. The receipt describes collective latency at exact payload sizes,
 * not a host-MPI rate, a per-link bandwidth, or an inference benchmark.
 * Physical-only mode creates no communicator and publishes no collective
 * observations; it retains only independently captured direct-copy evidence.
 * Machine/rank identity remains the enclosing inventory collector's authority.
 */
#pragma once
#include "planning/PhysicalMemoryAuthority.h"
#include "planning/CollectiveMemoryEstimator.h"
#include <span>
#include <vector>

namespace llaminar2
{
    class DeviceCountedAllGatherTopology;
    /** @brief Wire representation only; the model-facing activation remains FP32. */
    enum class PlanningAllreducePrecision { FP32, FP16 };

    /** @brief One disjoint live payload and aligned untouched-tail guard in the admitted sample allocation. */
    struct PlanningLocalTPPayloadRegion final
    {
        size_t byte_offset, live_bytes, guard_bytes;
    };

    /** @brief Validated same-process native GPU group and two positive payload geometries. */
    class PlanningLocalTPRequest final
    {
    public:
        /**
         * @brief Reject duplicate, mixed-backend, CPU and single-device groups before allocation.
         * @param devices Rank-local ordinals in production collective order; never remote ordinals.
         * @param hidden_width Logical FP32 activation columns.
         * @param prefill_rows Positive physical prefill rows; decode always uses one row.
         * @param precision Explicit production collective transport precision.
         */
        PlanningLocalTPRequest(std::vector<DeviceId> devices, int hidden_width,
            int prefill_rows, PlanningAllreducePrecision precision);
        /** @return Immutable native collective order. */
        const std::vector<DeviceId> &devices() const noexcept { return devices_; }
        /** @return Logical activation width, independent of allocation padding. */
        int hiddenWidth() const noexcept { return hidden_width_; }
        /** @return Exact requested prefill sample geometry. */
        int prefillRows() const noexcept { return prefill_rows_; }
        /** @return Explicit wire representation; no runtime environment override. */
        PlanningAllreducePrecision precision() const noexcept { return precision_; }
        /** @return NCCL for CUDA or RCCL for ROCm; never a substitute host transport. */
        CollectiveBackendType backend() const noexcept;
        /** @return Exact in-place FP32 allocation extent at the maximum rows. */
        size_t activationBytes() const noexcept;
        /** @return Exact logical reduction payload; not aggregate wire traffic. */
        size_t payloadBytes(int rows) const;
        /** @return Deduplicated raw-byte anchors: metadata, one FP32 row and admitted maximum. */
        std::vector<size_t> pairPayloadAnchors() const;
        /** @return Disjoint FP32-aligned live payloads followed by16-byte checked guards. */
        std::vector<PlanningLocalTPPayloadRegion> pairPayloadRegions() const;
        /** @return Sole payload allocation extent for ordinary reduction plus all guarded pair windows. */
        size_t storageBytes() const;
        /** @return Full fixed-gather receive bank plus the existing sixteen-byte oracle guard. */
        size_t gatherStorageBytes() const;
    private:
        std::vector<DeviceId> devices_;
        int hidden_width_, prefill_rows_;
        PlanningAllreducePrecision precision_;
    };

    /** @brief One endpoint's completed event timing of the same coupled collective. */
    struct PlanningLocalTPEndpointObservation final
    {
        DeviceId device;
        size_t graph_nodes;
        double seconds_per_collective;
    };

    /** @brief Complete collective phase, retaining all endpoints instead of an optimistic mean. */
    struct PlanningLocalTPPhaseObservation final
    {
        int rows;
        size_t payload_bytes;
        std::vector<PlanningLocalTPEndpointObservation> endpoints;
        /** @return Slowest endpoint's completed latency; endpoints are not independent transfers. */
        double secondsPerCollective() const;
    };

    /** @brief Distinct production collective protocols; neither substitutes for the other. */
    enum class PlanningNativeGatherProtocol { FixedAllgather, CountedAllgather };
    /** @brief One complete native gather invocation; payload is each owner's live bytes. */
    struct PlanningLocalTPGatherPhaseObservation final
    {
        PlanningNativeGatherProtocol protocol;
        size_t payload_bytes;
        size_t participant_count;
        std::vector<PlanningLocalTPEndpointObservation> endpoints;
        /** @return Slowest original endpoint's whole-invocation service, not a per-root mean. */
        double secondsPerCollective() const;
    };

    /** @brief Whether this enclosing group already supplied its precision-independent pair matrix. */
    enum class PlanningPairSampling { Disabled, Enabled };

    /** @brief Explicit functional-test observation; production setup never downloads sampled payloads. */
    enum class PlanningMeasurementValidation { None, ObservePayloads };

    /** @brief One physical peer-copy edge; indices retain original discovery order. */
    struct PlanningLocalTPPairObservation final
    {
        size_t source_index, destination_index, payload_bytes;
        std::vector<double> seconds_per_transfer; ///< Three raw destination-owned fixed-clock samples.
        size_t graph_nodes = 0;
    };

    /** @brief Complete directed protocol matrix; never an exact allreduce prediction. */
    struct PlanningLocalTPPairSampling final
    {
        std::vector<size_t> payload_anchors;
        std::vector<std::uint8_t> direct_peer_access; ///< Row-major data direction; zero edges require measured host-leg composition.
        std::vector<PlanningLocalTPPairObservation> directed_pairs;
        double setup_wall_seconds = 0, measurement_wall_seconds = 0;
        size_t capture_count = 0, measured_launches = 0;
        static constexpr size_t kBatchTransfers = 2; ///< Short retained graph, bounded independently of device count.
        static constexpr size_t kSamples = 3; ///< Raw repeated device-clock intervals, no startup bootstrap.
        /** @brief Reject missing/duplicate directions, anchors and invalid device-clock observations. */
        void validate(const PlanningLocalTPRequest &request) const;
    };

    /** @brief Separate physical copy evidence from native collective protocol work. */
    enum class PlanningLocalTPMeasurementMode { NativeCollective, PhysicalPeersOnly };

    /** @brief Typed evidence: two allreduce and four native-gather phases, or physical-only pairs, never a fabricated native timing. */
    struct PlanningLocalTPObservations final
    {
        PlanningLocalTPRequest request;
        std::vector<PlanningLocalTPPhaseObservation> phases;
        PlanningLocalTPPairSampling pair_sampling;
        PlanningLocalTPMeasurementMode mode = PlanningLocalTPMeasurementMode::NativeCollective;
        std::vector<PlanningLocalTPGatherPhaseObservation> gather_phases;

    };

    /** @brief Admit once, capture once per payload, observe briefly, retire before returning. */
    class PlanningLocalTPMeasurement final
    {
    public:
        /**
         * @brief Compose ordinary LocalTP scratch, FP32 payloads and retained graph-family BOM.
         * @param request Validated local group; no backend initialization occurs here.
         * @param host Same-rank CPU staging allocator.
         * @param devices Exact resources in the request's collective order.
         * @param builder Sole canonical aggregate builder; this method does not admit capacity.
         * @param mode Physical-only excludes collective scratch and reserves one retained pair executable.
         */
        static void contributeMemory(const PlanningLocalTPRequest &request,
            PhysicalMemoryResource host, std::span<const PhysicalMemoryResource> devices,
            PhysicalMemoryPlanBuilder &builder,
            PlanningLocalTPMeasurementMode mode = PlanningLocalTPMeasurementMode::NativeCollective);
        /**
         * @brief Measure actual captured LocalTP on the normal owning GPU workers.
         * @param request Same immutable group/payload policy used for admission.
         * @param memory Rank-bound authority containing the complete contributed BOM.
         * @param pairs Disable only when this exact ordered group already supplied its raw-byte matrix.
         * @param validation Native functional tests may inspect exact source/tail bytes outside timing.
         * @param direct_peer_access Original directed copy eligibility; mandatory when pair sampling is enabled.
         * @param mode Physical-only requires enabled pairs and never creates or times a native communicator.
         * @param counted_topology Genuine original inventory-derived topology, required for native mode.
         * @return Both complete phases after native graphs, payloads and collective borrows retire.
         * @throws std::exception on admission, preparation, capture or asynchronous failure.
         *
         * Invoke from the rank setup owner, not inside a GPU worker. All workers
         * are submitted before joining any one of them. Native collective phases
         * retain four unique graphs and one two-slot clock bank: one original
         * warmup precedes three timed allreduces in a single ordered parent.
         * Storage, capture and instantiation have all-participant cohort fences;
         * each phase retires before physical pair sampling. Warmup is outside
         * the captured device-clock interval; no external event timestamps,
         * eager collective, model warmup or alternative transport is introduced.
         */
        static PlanningLocalTPObservations measure(const PlanningLocalTPRequest &request,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory,
            PlanningPairSampling pairs = PlanningPairSampling::Disabled,
            PlanningMeasurementValidation validation = PlanningMeasurementValidation::None,
            std::span<const std::uint8_t> direct_peer_access = {},
            PlanningLocalTPMeasurementMode mode = PlanningLocalTPMeasurementMode::NativeCollective,
            const DeviceCountedAllGatherTopology *counted_topology = nullptr);
    };
}
