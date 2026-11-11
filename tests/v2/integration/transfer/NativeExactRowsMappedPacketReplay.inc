/**
 * @file NativeExactRowsMappedPacketReplay.inc
 * @brief Original-rank, two-physical-GPU exact packet parents and changed-publication replay.
 *
 * Three retained native parent pairs execute exact4/3/1 rows against4-row tensor
 * storage. Original MPI inventory and one admitted PMA own both endpoint grants,
 * tensors, batch resources and executable definitions. CPU admission operates
 * only the existing immutable packet protocol; model work and live extents stay
 * in the original production packet stages. No mock ranks or remote ordinals
 * participate in this native proof.
 */
#include "execution/compute_stages/stages/MoEOverlayActivationPacketStages.h"
#include "execution/compute_stages/stages/ResidualAddStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/moe/MoEOverlayActivationChannelPlan.h"
#include "execution/moe/MoEOverlayActivationEpochProtocol.h"
#include "execution/moe/MoERuntimeTable.h"
#include "execution/moe/MoEOverlayActivationBatchResources.h"
#include <atomic>
#include <bit>
#include <tuple>
#include <sstream>
#include <functional>
#include <map>
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
namespace {
    /**
     * @brief Enable packet and graph fork/join evidence for this process.
     *
     * CTest requests the same domain, but the binary is also run directly by
     * profiler and repetition workflows. Reloading the parsed environment here
     * keeps those invocations self-certifying without enabling unrelated hot-
     * path instrumentation.
     */
    class ScopedActivationEpochPerfStats final
    {
    public:
        ScopedActivationEpochPerfStats()
        {
            preserve("LLAMINAR_PERF_STATS_SUMMARY", summary_);
            preserve("LLAMINAR_PERF_STATS_FILTER", filter_);
            ::setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
            ::setenv(
                "LLAMINAR_PERF_STATS_FILTER",
                "moe_overlay_activation_epoch,forward_graph",
                1);
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ~ScopedActivationEpochPerfStats()
        {
            restore("LLAMINAR_PERF_STATS_SUMMARY", summary_);
            restore("LLAMINAR_PERF_STATS_FILTER", filter_);
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ScopedActivationEpochPerfStats(
            const ScopedActivationEpochPerfStats &) = delete;
        ScopedActivationEpochPerfStats &operator=(
            const ScopedActivationEpochPerfStats &) = delete;

    private:
        struct SavedValue
        {
            bool present = false;
            std::string value;
        };

        /** @brief Snapshot one variable before the focused test override. */
        static void preserve(const char *name, SavedValue &saved)
        {
            if (const char *const value = ::getenv(name))
            {
                saved.present = true;
                saved.value = value;
            }
        }

        /** @brief Restore one variable exactly as the enclosing process supplied it. */
        static void restore(const char *name, const SavedValue &saved)
        {
            if (saved.present)
                ::setenv(name, saved.value.c_str(), 1);
            else
                ::unsetenv(name);
        }

        SavedValue summary_;
        SavedValue filter_;
    };

/**
 * @brief Authenticate every actually retained row family and the original endpoint totals.
 * @param row_families Complete captured row roster, including repeated widths if retained separately.
 * @param layer_count Actual layer manifest size embedded by each family.
 * @param distributed Whether the two actual endpoint owners require original MPI aggregation.
 * @param source_rank Authenticated original source rank from rankReplayEndpoints, never inferred from backend.
 * Row-tagged source publications/forks/joins are checked individually. Untagged
 * timeline totals sum the exact scalar or cooperative policy for each family.
 */
void assertMappedCaptureEvidence(std::span<const int> row_families, std::size_t layer_count,
    bool distributed,int source_rank) {
        require(!row_families.empty(),"Original packet capture evidence has no retained family");
        const auto mpi=MPIContextFactory::global();
        require(!distributed || (mpi && source_rank>=0 && source_rank<mpi->world_size()),
            "Original packet capture evidence lost its observed source owner");
        const bool owns_source=!distributed || mpi->rank()==source_rank;
        std::map<std::string,double> expected_tagged;
        std::map<std::string,std::array<double,3>> observed_tagged;
        std::array<double,11> expected{};
        for(const int rows:row_families) {
            require(rows>0,"Original packet capture evidence has invalid retained rows");
            const bool scalar=rows==1;
            const auto key=std::to_string(rows);
            expected_tagged[key]+=owns_source && !scalar?double(layer_count):0.;
            observed_tagged.try_emplace(key,std::array<double,3>{});
            expected[0]+=scalar?2.:2.*(layer_count+1);
            expected[1]+=scalar?0.:2.*layer_count;
            expected[2]+=scalar?2.*layer_count:0.;expected[3]+=scalar?2.*layer_count:0.;
            for(const auto metric:{8u,9u,10u})expected[metric]+=scalar?0.:double(layer_count);
        }
        const auto observe_row=[&](const auto& record,std::size_t metric) {
            const auto found=observed_tagged.find(record.tags.at("rows"));
            require(found!=observed_tagged.end(),"Original packet capture observation has an undeclared retained row family");
            found->second[metric]+=record.value;
        };
        double waits_enqueued = 0.0;
        double publications_enqueued = 0.0;
        double kernel_wait_bindings = 0.0;
        double kernel_publication_bindings = 0.0;
        double dispatch_d2h_bytes = 0.0;
        double dispatch_h2d_bytes = 0.0;
        double return_d2h_bytes = 0.0;
        double return_h2d_bytes = 0.0;
        double shared_dispatch_payload_publications = 0.0;
        double asynchronous_lane_forks = 0.0;
        double asynchronous_lane_joins = 0.0;
        const auto inspect = [&] {
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"moe_overlay_activation_epoch", "forward_graph"}))
        {
            if (record.domain == "forward_graph" &&
                record.name == "moe_overlay_async_lane_forks")
            {
                asynchronous_lane_forks += record.value;
                require((record.tags.at("lanes")) == ("1"), "Original packet capture instrumentation changed its exact policy tags");
                observe_row(record,0);
            }
            else if (record.domain == "forward_graph" &&
                     record.name == "moe_overlay_async_lane_joins")
            {
                asynchronous_lane_joins += record.value;
                require((record.tags.at("lanes")) == ("1"), "Original packet capture instrumentation changed its exact policy tags");
                observe_row(record,1);
                require((record.tags.at("canonical_materialization")) == ("true"), "Original packet capture instrumentation changed its exact policy tags");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name == "device_timeline_waits_enqueued")
            {
                waits_enqueued += record.value;
                require((record.tags.at("host_blocking")) == ("false"), "Original packet capture instrumentation changed its exact policy tags");
                require((record.tags.at("ordering")) == ("exact_stream_64bit_geq"), "Original packet capture instrumentation changed its exact policy tags");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                     "device_timeline_publications_enqueued")
            {
                publications_enqueued += record.value;
                require((record.tags.at("host_blocking")) == ("false"), "Original packet capture instrumentation changed its exact policy tags");
                require((record.tags.at("ordering")) == ("exact_stream_64bit_fenced"), "Original packet capture instrumentation changed its exact policy tags");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                     "device_timeline_kernel_wait_bindings")
            {
                kernel_wait_bindings += record.value;
                require((record.tags.at("host_blocking")) == ("false"), "Original packet capture instrumentation changed its exact policy tags");
                require((record.tags.at("ordering")) == ("fused_packet_system_acquire"), "Original packet capture instrumentation changed its exact policy tags");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                     "device_timeline_kernel_publication_bindings")
            {
                kernel_publication_bindings += record.value;
                require((record.tags.at("host_blocking")) == ("false"), "Original packet capture instrumentation changed its exact policy tags");
                require((record.tags.at("ordering")) == ("fused_packet_system_release"), "Original packet capture instrumentation changed its exact policy tags");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                         "shared_physical_dispatch_d2h_bytes")
            {
                dispatch_d2h_bytes += record.value;
                require((record.tags.at("host_blocking")) == ("false"), "Original packet capture instrumentation changed its exact policy tags");
                require((record.tags.at("payload_path")) == ("shared_physical_mapped"), "Original packet capture instrumentation changed its exact policy tags");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                         "shared_physical_dispatch_h2d_bytes")
            {
                dispatch_h2d_bytes += record.value;
                require((record.tags.at("host_blocking")) == ("false"), "Original packet capture instrumentation changed its exact policy tags");
                require((record.tags.at("ordering")) == ("mapped_timeline_then_bulk_dma"), "Original packet capture instrumentation changed its exact policy tags");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                         "shared_physical_return_d2h_bytes")
            {
                return_d2h_bytes += record.value;
                require((record.tags.at("host_blocking")) == ("false"), "Original packet capture instrumentation changed its exact policy tags");
                require((record.tags.at("ordering")) == ("device_payload_then_bulk_dma"), "Original packet capture instrumentation changed its exact policy tags");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                         "shared_physical_return_h2d_bytes")
            {
                return_h2d_bytes += record.value;
                require((record.tags.at("host_blocking")) == ("false"), "Original packet capture instrumentation changed its exact policy tags");
                require((record.tags.at("ordering")) == ("mapped_timeline_then_bulk_dma"), "Original packet capture instrumentation changed its exact policy tags");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                         "shared_dispatch_payload_publications")
            {
                shared_dispatch_payload_publications += record.value;
                require((record.tags.at("lanes")) == ("1"), "Original packet capture instrumentation changed its exact policy tags");
                observe_row(record,2);
            }
        }
        for(const auto& [rows,observed]:observed_tagged) for(const auto count:observed)
            require(count==expected_tagged.at(rows),
                ("Original packet capture row-family count changed rows="+rows+
                    " observed="+std::to_string(count)+" expected="+std::to_string(expected_tagged.at(rows))).c_str());
        };
        if (distributed) acceptPlanningCostPreparation(MPIContextFactory::global(), inspect); else inspect();
        // Shared row stores are device-counted at terminal; setup has no fixed-capacity DMA.
        if (distributed) {
            std::array<double, 11> local{waits_enqueued, publications_enqueued, kernel_wait_bindings, kernel_publication_bindings,
                dispatch_d2h_bytes, dispatch_h2d_bytes, return_d2h_bytes, return_h2d_bytes,
                shared_dispatch_payload_publications, asynchronous_lane_forks, asynchronous_lane_joins}, total{};
            require(MPI_Allreduce(local.data(), total.data(), int(total.size()), MPI_DOUBLE, MPI_SUM,
                MPIContextFactory::global()->communicator()) == MPI_SUCCESS,
                "Original packet capture evidence aggregation failed");
            std::tie(waits_enqueued, publications_enqueued, kernel_wait_bindings, kernel_publication_bindings,
                dispatch_d2h_bytes, dispatch_h2d_bytes, return_d2h_bytes, return_h2d_bytes,
                shared_dispatch_payload_publications, asynchronous_lane_forks, asynchronous_lane_joins) =
                std::tie(total[0], total[1], total[2], total[3], total[4], total[5], total[6], total[7], total[8], total[9], total[10]);
        }
        EXPECT_EQ(waits_enqueued,expected[0]);
        EXPECT_EQ(publications_enqueued,expected[1]);
        EXPECT_EQ(kernel_wait_bindings,expected[2]);
        EXPECT_EQ(kernel_publication_bindings,expected[3]);
        EXPECT_EQ(dispatch_d2h_bytes,expected[4]);EXPECT_EQ(dispatch_h2d_bytes,expected[5]);
        EXPECT_EQ(return_d2h_bytes,expected[6]);EXPECT_EQ(return_h2d_bytes,expected[7]);
        EXPECT_EQ(shared_dispatch_payload_publications,expected[8]);
        EXPECT_EQ(asynchronous_lane_forks,expected[9]);
        EXPECT_EQ(asynchronous_lane_joins,expected[10]);

 }

/** Exact original inputs; changed-publication also queues before the previous admission retires. */
enum class NativeExactPacketReplay { ExactRows, ChangedPublication, Ordinary, StaleAdmission, AdmissionAbort, FollowerPrequeued, FollowerPrequeuedAbort };
/** One retained physical family program, independently authored before admission. */
struct NativeMappedPacketGeometry {
    int backing_rows = 4;
    int width = 8;
    std::vector<int> shapes{4, 3, 1};
    std::uint64_t generations = 20;
};
/** Stable minimal placement view consumed by the actual packet kernels. */
struct NativeExactPacketPlacement {
    std::int32_t routes[2][18]{};
    std::uint32_t epochs[2]{};
    DeviceMoEOverlayEpochTicket ticket;
    DeviceMoEOverlayEpochStatus status;
    std::int32_t active_rows = 0;
};
static_assert(sizeof(NativeExactPacketPlacement) % sizeof(std::int32_t) == 0);
/** One original declared tensor; native capacity equals its exact retained backing. */
struct NativeExactPacketTensor {
    CPUTensorStorageDeclaration declaration;
    std::shared_ptr<TensorBase> tensor;
    NativeExactPacketTensor(std::size_t rows, std::size_t width, BufferTensorType type = BufferTensorType::FP32)
        : declaration(rows, width, type, PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu()) {}
    /** Add exact original CPU and native backing to the sole source admission. */
    void declare(PhysicalMemoryPlanBuilder &plan, const PhysicalMemoryResource &host, const PhysicalMemoryResource &gpu) const {
        plan.add(host, declaration.owner(), declaration.physicalBytes());
        plan.add(gpu, declaration.owner(), declaration.rows() * declaration.columns() * sizeof(std::int32_t));
    }
    /** Materialize the same declared original backing, once before capture. */
    void prepare(const std::shared_ptr<PhysicalMemoryAuthority> &memory, DeviceId device, void *stream) {
        tensor = declaration.materialize(memory);
        TransferEngine::prepareDeviceInput(tensor.get(), device, stream);
    }
    /** Original mutable recorded input storage, never a download or live device decision. */
    template<class T> T *host() { return static_cast<T *>(tensor->raw_mutable_data()); }
};
/** @brief Same exact terminal snapshot geometry used by the original PMA quote and view. */
struct NativeExactPacketSnapshotLayout {
    /** @return Existing numerical snapshot plus the original active-row scalar. */
    static std::size_t payloadBytes(std::size_t rows,std::size_t width) {
        return (5u*rows*width+4u*rows)*sizeof(float)+sizeof(std::int32_t);
    }
    /** @return Aligned bounded failure-only copies of the two actual private grant rows. */
    static std::size_t grantsOffset(std::size_t rows,std::size_t width) {
        constexpr auto alignment=alignof(MoEOverlayActivationDeviceEpochGrant);
        return (payloadBytes(rows,width)+alignment-1)&~(alignment-1);
    }
    /** @return Complete original terminal backing, never a second ledger. */
    static std::size_t bytes(std::size_t rows,std::size_t width) {
        return grantsOffset(rows,width)+2*sizeof(MoEOverlayActivationDeviceEpochGrant);
    }
};
/** All native resources remain live until every original retained definition retires. */
class NativeExactPacketFixture final {
    static constexpr int topk = 2, experts = 18;
    static constexpr float poison = -9876.5f;
    static constexpr std::array<int, 3> layers{5, 9, 14};
    static constexpr std::array<bool, 8> foreign{true, false, false, false, true, true, false, true};
    static constexpr std::array<float, 8> weights{.75f, .25f, .6f, .4f, .55f, .45f, .9f, .1f};
    using Graph = test::RetainedParticipantGraphExecutableFixtureAccess;
    struct Family {
        int rows = 0;
        std::array<MoEOverlayMappedActivationDeviceLane, 2> lane;
        std::array<std::vector<std::shared_ptr<IComputeStage>>, 2> stages;
        std::array<std::vector<GraphCaptureDependencyLedger::StagePlan>, 2> dependencies;
        std::array<std::vector<std::uint64_t>, 2> signatures;
        std::array<std::shared_ptr<RetainedParticipantGraphExecutable>, 2> graph;
        std::array<std::uint64_t, 2> submissions{};
        std::array<bool, 2> pending{};
        std::unique_ptr<MoEOverlayActivationEpochProtocol> protocol;
        MoEOverlayActivationEpochConfig config;
        std::optional<MoEOverlayActivationEpochIdentity> identity;
    };
public:
    /**
     * @brief Retain the cohort already authenticated by originalRankLocalReplay.
     * @param devices Actual source/target GPUs in this original owner's namespace.
     * @param replay Exact-row or changed-publication recorded-input profile.
     */
    NativeExactPacketFixture(std::vector<DeviceId> devices, NativeExactPacketReplay replay,
        NativeMappedPacketGeometry geometry = {}, std::optional<std::array<int, 2>> original_ranks = std::nullopt)
        : devices_(std::move(devices)), ranks_(original_ranks.value_or(std::array{MPIContextFactory::global()->rank(), MPIContextFactory::global()->rank()})),
          geometry_(std::move(geometry)), capacity(geometry_.backing_rows), width(geometry_.width), replay_(replay), family_(geometry_.shapes.size()) {
        require(devices_.size() == 2 && devices_[0].is_gpu() && devices_[1].is_gpu() &&
            (ranks_[0] != ranks_[1] || devices_[0] != devices_[1]) && capacity > 0 && width > 0 && !family_.empty(),
            "Mapped packet requires complete original endpoint coordinates and positive geometry");
        for (std::size_t family = 0; family < geometry_.shapes.size(); ++family) {
            const int rows = geometry_.shapes[family];
            require(rows > 0 && rows <= capacity, "Mapped packet retained shape exceeds original backing");
            family_[family].rows = rows;
        }
        std::vector<DeviceId> local;
        for (std::size_t endpoint = 0; endpoint < 2; ++endpoint) if (owns(endpoint)) local.push_back(devices_[endpoint]);
        if (!local.empty()) original_ = test::PreparedNativeLookupFixture::prepare(local, MPIContextFactory::global()->rank());
    }
    /** Failure cancellation drains original queues before native or source owners leave scope. */
    ~NativeExactPacketFixture() noexcept { try { dispose(); } catch (...) { std::terminate(); } }
    /** Admit the exact original channel/typed-bank/graph/batch declarations, then materialize. */
    void prepare();
    /** Twenty generations retain every original graph pointer and consume compact bytes only. */
    void run();
private:
    /** Original rank identity, independent of native ordinal coincidence across ranks. */
    bool owns(std::size_t endpoint) const { return ranks_[endpoint] == MPIContextFactory::global()->rank(); }
    /** Join explicit fixture submission checkpoints and terminal evidence on the original communicator. */
    void joinTerminal() const { originalOutcome([] {}); }
    /**
     * @brief Join one actual original-rank phase, including its failure outcome.
     * @param body Local original-owner work; it contains no rank collective.
     * A throwing owner and a nonfatal GTest assertion must reach the same
     * canonical consensus before either process can enter another phase.
     */
    void originalOutcome(const std::function<void()>& body) const {
        const auto observe=[&] {
            body();
            require(!::testing::Test::HasFailure(), "Original mapped packet owner reported a failed native oracle");
        };
        if (ranks_[0] != ranks_[1]) acceptPlanningCostPreparation(MPIContextFactory::global(), observe);
        else observe();
    }
    /** Exact bounded terminal snapshot payload, shared by declaration and materialization. */
    std::size_t snapshotBytes() const { return NativeExactPacketSnapshotLayout::bytes(capacity,width); }
    /** Resolve one real RankLocalGPU channel against the complete original inventory. */
    void declareChannel();
    /** Create the one actual mapping with every original endpoint grant. */
    void materializeChannel();
    /** Build the production batch/scalar stage sequence and its explicit tensor DAG. */
    void buildFamily(std::size_t family);
    /** Prepare and record one original complete parent on its retained worker/stream. */
    void capture(std::size_t family, std::size_t endpoint);
    /**
     * @brief Upload recorded rows only after previous original terminals complete.
     * @param rows Exact retained graph geometry, bounded by four-row backing.
     * @param live Original device scalar's next value, including zero work.
     * @param bias New recorded hidden-row value, never a host execution decision.
     */
    void inputs(int rows, int live, float bias);
    /** Publish the existing authenticated admission for this family. */
    void arm(Family &family, std::uint64_t generation);
    /** Submit a complete original endpoint and publish its exact original output event. */
    void submit(Family &family, std::size_t endpoint);
    /** Poll original terminal streams; no blocking/default-stream backend synchronization. */
    void await();
    /** Authenticate completed parent receipts and copy bounded diagnostic outputs. */
    void validate(Family &family, int live, float bias);
    /** @return Bounded original header/metadata/grant evidence after both native terminals retired. */
    std::string terminalFailure(const Family&,std::size_t endpoint,int live) const;
    /** Existing normal retirement clears data without changing graph topology. */
    void reset(Family &family);
    /** Absorbing failure cancellation plus original off-worker graph/source retirement. */
    void dispose();
    std::vector<DeviceId> devices_;
    std::array<int, 2> ranks_;
    NativeMappedPacketGeometry geometry_;
    const int capacity, width;
    NativeExactPacketReplay replay_;
    std::unique_ptr<test::PreparedNativeLookupFixture> original_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    std::array<std::shared_ptr<IWorkerGPUContext>, 2> workers_;
    std::array<void *, 2> streams_{};
    std::array<std::unique_ptr<IDeviceContext>, 2> contexts_;
    NativeExactPacketTensor hidden_{std::size_t(capacity),std::size_t(width)}, indices_{std::size_t(capacity),2}, weights_{std::size_t(capacity),2}, canonical_{std::size_t(2*capacity),std::size_t(width)}, zero_{std::size_t(2*capacity),std::size_t(width)};
    NativeExactPacketTensor received_{std::size_t(capacity),std::size_t(width)}, received_indices_{std::size_t(capacity),2}, received_weights_{std::size_t(capacity),2}, local_{std::size_t(2*capacity),std::size_t(width)}, route_input_{std::size_t(2*capacity),std::size_t(width)};
    std::array<NativeExactPacketTensor, 2> placement_{
        NativeExactPacketTensor(1,sizeof(NativeExactPacketPlacement)/4,BufferTensorType::INT32),
        NativeExactPacketTensor(1,sizeof(NativeExactPacketPlacement)/4,BufferTensorType::INT32)};
    std::vector<Family> family_;
    std::uint64_t source_generation_ = 0;
    std::uint32_t source_process_ = 0;
    MoEOverlayActivationChannelPlan channel_plan_;
    MoEOverlayInferenceTopologyIdentity topology_;
    std::shared_ptr<IMoEOverlayRankBatchTransport> transport_;
    std::shared_ptr<IMoEOverlayMappedActivationTransport> mapped_;
    std::shared_ptr<MappedHostTransferRegion> snapshot_;
    std::optional<PhysicalMemoryAllocationLease> snapshot_claim_;
    int target_ = -1;
    bool disposed_ = false;
    bool admitted_ = false;
};

void NativeExactPacketFixture::declareChannel() {
    const auto mpi = MPIContextFactory::global();
    MoERoutedExpertPlacementPlan placement;
    placement.enabled = true; placement.topology = RoutedExpertPlacementTopology::TieredOverlay;
    placement.continuation_domain = placement.base_model_domain = placement.shared_expert_domain = "source";
    placement.continuation_domain_spec.domain = "source"; placement.continuation_domain_spec.logical_root_participant = 0;
    for (std::size_t p = 0; p < devices_.size(); ++p) {
        RoutedExpertDomain domain; domain.name = p ? "follower" : "source";
        domain.scope = ExecutionDomainScope::RANK_LOCAL; domain.owner_rank = ranks_[p]; domain.world_ranks = {ranks_[p]};
        domain.participants = {GlobalDeviceAddress::fromLocalDeviceId(devices_[p])};
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        placement.domains.push_back(domain);
        placement.routed_tiers.push_back({.name = domain.name, .domain = domain.name, .priority = p ? 30 : 80});
    }
    placement.authority_execution = resolveMoEOverlayAuthorityExecutionKind(placement);
    channel_plan_ = MoEOverlayActivationChannelPlanner::plan({.placement_plan = &placement,
        .cluster_inventory = mpi->clusterInventory().get(), .row_capacity = std::size_t(capacity), .d_model = width, .top_k = topk, .graph_family_count = family_.size()});
    require(channel_plan_.channels.size() == 1, "Exact packet requires its one original rank-local channel");
    const auto &channel = channel_plan_.channels.front();
    require(channel.source_world_rank == ranks_[0] && channel.target_world_rank == ranks_[1] &&
        channel.source_device == devices_[0] && channel.localLanes(mpi->rank()).size() == (ranks_[0] == ranks_[1] ? 2u : 1u),
        "Exact packet channel changed original rank or endpoint ownership");
    const auto targets = channel.targetParticipantIds();
    require(targets.size() == 1, "Exact packet has no unique actual target"); target_ = targets.front();
    topology_ = {.workspace_generation = source_generation_,
        .topology_fingerprint_low = 0x726f7773343331ULL, .topology_fingerprint_high = 0x6f726967696e616cULL,
        .source_world_rank = ranks_[0], .target_world_rank = ranks_[1]};
}

void NativeExactPacketFixture::materializeChannel() {
    const auto mpi = MPIContextFactory::global(); const auto &channel = channel_plan_.channels.front();
    MoEOverlayRankBatchTransportConfig config;
    config.mpi_ctx = mpi;
    if (ranks_[0] == ranks_[1]) {
        config.endpoint_scope = MoEOverlayActivationEndpointScope::RankLocalGPU;
        config.rank_local_source_device = devices_[0];
    }
    config.source_world_rank = ranks_[0]; config.target_world_rank = ranks_[1];
    config.workspace = std::make_shared<MoEOverlayRankBatchWireWorkspace>(MoEOverlayRankBatchWireWorkspace::Config{
        .participant_ids = {target_}, .max_total_rows = std::size_t(capacity), .max_total_entries = std::size_t(capacity) * topk, .d_model = width, .top_k = topk});
    config.max_rows_per_participant = capacity; config.max_entries_per_participant = capacity * topk;
    config.d_model = width; config.top_k = topk; config.tier_index = channel.tier_index; config.domain_ordinal = channel.domain_ordinal;
    config.channel_identity = "original_packet_" + std::to_string(source_process_); config.transaction_slot_count = 64;
    config.transaction_topology = topology_;
    config.source_endpoint = {.world_rank = ranks_[0], .participant_id = 0, .tier_priority = 80, .domain_ordinal = 0};
    config.physical_memory_authority = memory_;
    config.target_tier_priority = 30; config.activation_layout = channel.mapping_layout; config.local_lanes = channel.localLanes(mpi->rank());
    for (std::size_t f = 0; f < family_.size(); ++f)
        config.activation_graph_families.push_back({.graph_role_mask = 1u << static_cast<unsigned>(MoEOverlayInferenceGraphRole::MainPrefill),
            .model_layer_indices = {layers.begin(), layers.end()}});
    transport_ = ranks_[0] == ranks_[1] ? createMoEOverlayRankLocalGPUActivationTransport(std::move(config))
        : std::make_shared<MoEOverlayNodeLocalRankBatchTransport>(std::move(config));
    mapped_ = std::dynamic_pointer_cast<IMoEOverlayMappedActivationTransport>(transport_);
    require(bool(mapped_), "Exact packet lost its real mapped transport");
    for (std::size_t f = 0; f < family_.size(); ++f) {
        auto &family = family_[f]; family.rows = geometry_.shapes[f];
        for (std::size_t p = 0; p < 2; ++p) if (owns(p)) family.lane[p] = mapped_->activationDeviceLane(target_, f, devices_[p]);
        const auto &header = mapped_->activationEpochControl(target_, f).channel;
        family.config = {.channel_nonce = header.channel_nonce, .topology_fingerprint_low = header.topology_fingerprint_low,
            .topology_fingerprint_high = header.topology_fingerprint_high, .workspace_generation = header.workspace_generation,
            .source = {.world_rank = header.source_world_rank, .participant_id = header.source_participant_id,
                .tier_priority = header.source_tier_priority, .domain_ordinal = header.source_domain_ordinal},
            .target = {.world_rank = header.target_world_rank, .participant_id = header.target_participant_id,
                .tier_priority = header.target_tier_priority, .domain_ordinal = header.target_domain_ordinal},
            .lane_ordinal = header.lane_ordinal, .graph_role_mask = header.graph_role_mask,
            .model_layer_indices = {layers.begin(), layers.end()},
            .return_role = static_cast<MoEOverlayActivationReturnRole>(header.return_role), .cohort_target = header.cohort_target};
        family.protocol = std::make_unique<MoEOverlayActivationEpochProtocol>(mapped_->activationEpochControl(target_, f), family.config);
    }
}

/** Views into one declared mapped terminal snapshot; they own no duplicate payload. */
struct NativeExactPacketSnapshot {
    std::span<float> canonical, hidden, indices, weights, local;
    std::int32_t *active_rows;
    MoEOverlayActivationDeviceEpochGrant* grants;
    NativeExactPacketSnapshot(void *data, std::size_t rows, std::size_t width) {
        auto *next = static_cast<float *>(data);
        const auto take = [&](std::size_t count) { auto out = std::span(next, count); next += count; return out; };
        canonical = take(rows * 2 * width); hidden = take(rows * width);
        indices = take(rows * 2); weights = take(rows * 2); local = take(rows * 2 * width);
        active_rows = reinterpret_cast<std::int32_t *>(next);
        grants=reinterpret_cast<MoEOverlayActivationDeviceEpochGrant*>(
            static_cast<std::byte*>(data)+NativeExactPacketSnapshotLayout::grantsOffset(rows,width));
    }
};

void NativeExactPacketFixture::prepare() {
    if (owns(0)) { source_generation_ = original_->permit(devices_[0]).generation(); source_process_ = ::getpid(); }
    if (ranks_[0] != ranks_[1]) {
        std::array<std::int32_t,3> identity{std::bit_cast<std::int32_t>(std::uint32_t(source_generation_)),
            std::bit_cast<std::int32_t>(std::uint32_t(source_generation_ >> 32)), std::bit_cast<std::int32_t>(source_process_)};
        MPIContextFactory::global()->broadcast_int32(identity.data(), identity.size(), ranks_[0]);
        source_generation_ = std::uint64_t(std::bit_cast<std::uint32_t>(identity[0])) |
            (std::uint64_t(std::bit_cast<std::uint32_t>(identity[1])) << 32);
        source_process_ = std::bit_cast<std::uint32_t>(identity[2]);
    }
    if (!original_) return;
    declareChannel(); const auto &channel = channel_plan_.channels.front();
    const auto host = original_->resource(DeviceId::cpu());
    PhysicalMemoryPlanBuilder plan;
    const std::array<std::vector<NativeExactPacketTensor *>, 2> tensors{{
        {&hidden_, &indices_, &weights_, &canonical_, &zero_, &placement_[0]},
        {&received_, &received_indices_, &received_weights_, &local_, &route_input_, &placement_[1]}}};
    for (std::size_t p = 0; p < 2; ++p) if (owns(p)) {
        for (const auto *tensor : tensors[p]) tensor->declare(plan, host, original_->resource(devices_[p]));
        const auto graphs = RetainedParticipantGraphExecutable::generalBOM(host, original_->resource(devices_[p]), family_.size());
        for (const auto &line : graphs.resources()) plan.add(line);
        plan.add(original_->resource(devices_[p]), PhysicalMemoryOwner::ActivationTransportStaging,
            channel_plan_.stagingBytesFor(MPIContextFactory::global()->rank(), devices_[p]));
    }
    plan.add(host, PhysicalMemoryOwner::ActivationTransportStaging,
        channel_plan_.stagingBytesFor(MPIContextFactory::global()->rank(), DeviceId::cpu()) +
        TransferEngine::mappedHostRegionAllocationBytes(snapshotBytes()));
    const auto add = [&](const PhysicalMemoryPlan &resource) { for (const auto &line : resource.resources()) plan.add(line); };
    if (owns(0)) {
        add(MoEOverlayActivationBatchResources::sharedLaneStreamsBOM(channel, host, original_->resource(devices_[0])));
        for (const int rows : geometry_.shapes) if (rows > 1) for (std::uint32_t stage = 0; stage < layers.size(); ++stage)
            add(MoEOverlayActivationBatchResources::bom(channel, rows, stage, host, original_->resource(devices_[0])));
    }
    if (owns(1)) add(MoEOverlayActivationBatchResources::coldPublicationBOM(host, original_->resource(devices_[1])));
    original_->admit(plan.build()); admitted_ = true; memory_ = original_->memory();
    for (std::size_t p = 0; p < 2; ++p) if (owns(p)) {
        workers_[p] = original_->permit(devices_[p]).worker(); streams_[p] = workers_[p]->defaultStream();
        contexts_[p] = IDeviceContext::create(devices_[p]);
        workers_[p]->submitAndWait([&, p] { for (auto *tensor : tensors[p]) tensor->prepare(memory_, devices_[p], streams_[p]); });
    }
    snapshot_claim_.emplace(memory_->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
        TransferEngine::mappedHostRegionAllocationBytes(snapshotBytes())));
    std::vector<DeviceId> local_devices;
    for (std::size_t endpoint = 0; endpoint < 2; ++endpoint) if (owns(endpoint)) local_devices.push_back(devices_[endpoint]);
    snapshot_ = TransferEngine::instance().allocateMappedHostRegion(snapshotBytes(), local_devices);
    materializeChannel(); inputs(capacity, capacity, 0); await();
    for (std::size_t f = 0; f < family_.size(); ++f) {
        buildFamily(f);
        for (std::size_t p = 0; p < 2; ++p) if (owns(p)) family_[f].graph[p] = Graph::original(original_->permit(devices_[p]), streams_[p]);
        for (std::size_t p = 0; p < 2; ++p) if (owns(p)) capture(f, p);
    }
    if (owns(0) && std::any_of(geometry_.shapes.begin(), geometry_.shapes.end(), [](int rows) { return rows > 1; }))
        workers_[0]->submitAndWait([&] {
            EXPECT_THROW((void)workers_[0]->getOrCreateAuxiliaryStream("moe_overlay_activation_lane:0"), std::logic_error);
        });
}

void NativeExactPacketFixture::buildFamily(std::size_t index) {
    auto &family = family_[index];
    const auto add = [&](std::size_t p, std::shared_ptr<IComputeStage> stage,
        std::vector<const TensorBase *> external, std::vector<GraphCaptureDependencyLedger::InternalInput> internal,
        std::vector<const TensorBase *> outputs) {
        const auto slot = family.stages[p].size();
        family.dependencies[p].push_back({.stage_identity = stage.get(), .stage_name = stage->name(),
            .external_inputs = std::move(external), .internal_inputs = std::move(internal), .outputs = std::move(outputs), .recording_stream = streams_[p]});
        family.stages[p].push_back(std::move(stage)); return slot;
    };
    const auto binding = [&](std::size_t p) {
        auto *meta = static_cast<NativeExactPacketPlacement *>(placement_[p].tensor->gpu_data_ptr());
        return MoEOverlayRoutePlacementDeviceBinding{.banks = {{meta->routes[0], &meta->epochs[0]}, {meta->routes[1], &meta->epochs[1]}},
            .ticket = &meta->ticket, .status = &meta->status, .expert_count = experts};
    };
    std::size_t previous = 0;
    if (owns(0)) {
    ResidualAddStage::Params clear; clear.device_id = devices_[0]; clear.input = zero_.tensor.get(); clear.output = canonical_.tensor.get();
    clear.num_elements = family.rows * topk * width;
    previous = add(0, std::make_shared<ResidualAddStage>(clear), {zero_.tensor.get()}, {}, {canonical_.tensor.get()});
    }
    for (std::uint32_t stage = 0; stage < layers.size(); ++stage) {
        if (owns(0)) {
        if (family.rows > 1) {
            auto batch = std::make_shared<MoEOverlayActivationLaneBatchState>(devices_[0],
                std::vector{family.lane[0]}, family.rows, stage, layers[stage], memory_);
            MoEOverlayActivationDispatchPackBatchStage::Params dispatch;
            dispatch.device_id = devices_[0]; dispatch.transaction = batch; dispatch.hidden = hidden_.tensor.get();
            dispatch.routing_indices = indices_.tensor.get(); dispatch.routing_weights = weights_.tensor.get();
            dispatch.placement = binding(0); dispatch.active_row_count_device =
                &static_cast<NativeExactPacketPlacement *>(placement_[0].tensor->gpu_data_ptr())->active_rows;
            add(0, std::make_shared<MoEOverlayActivationDispatchPackBatchStage>(dispatch),
                {hidden_.tensor.get(), indices_.tensor.get(), weights_.tensor.get()}, {}, {});
            MoEOverlayActivationReturnConsumeBatchStage::Params returned;
            returned.device_id = devices_[0]; returned.transaction = std::move(batch); returned.canonical_route_contributions = canonical_.tensor.get();
            previous = add(0, std::make_shared<MoEOverlayActivationReturnConsumeBatchStage>(returned), {},
                {{canonical_.tensor.get(), previous}}, {canonical_.tensor.get()});
        } else {
            MoEOverlayActivationDispatchPackStage::Params dispatch;
            dispatch.device_id = devices_[0]; dispatch.lane = family.lane[0]; dispatch.hidden = hidden_.tensor.get();
            dispatch.routing_indices = indices_.tensor.get(); dispatch.routing_weights = weights_.tensor.get(); dispatch.placement = binding(0);
            dispatch.active_row_count_device = &static_cast<NativeExactPacketPlacement *>(placement_[0].tensor->gpu_data_ptr())->active_rows;
            dispatch.physical_rows = family.rows; dispatch.stage_ordinal = stage; dispatch.model_layer_index = layers[stage];
            add(0, std::make_shared<MoEOverlayActivationDispatchPackStage>(dispatch),
                {hidden_.tensor.get(), indices_.tensor.get(), weights_.tensor.get()}, {}, {});
            MoEOverlayActivationReturnConsumeStage::Params returned;
            returned.device_id = devices_[0]; returned.lane = family.lane[0]; returned.canonical_route_contributions = canonical_.tensor.get();
            returned.physical_rows = family.rows; returned.stage_ordinal = stage; returned.model_layer_index = layers[stage];
            previous = add(0, std::make_shared<MoEOverlayActivationReturnConsumeStage>(returned), {},
                {{canonical_.tensor.get(), previous}}, {canonical_.tensor.get()});
        }
        }
        if (owns(1)) {
        MoEOverlayActivationDispatchConsumeStage::Params consume;
        consume.device_id = devices_[1]; consume.lane = family.lane[1]; consume.placement = binding(1);
        consume.hidden = received_.tensor.get(); consume.routing_indices = received_indices_.tensor.get(); consume.routing_weights = received_weights_.tensor.get();
        consume.active_row_count_device = &static_cast<NativeExactPacketPlacement *>(placement_[1].tensor->gpu_data_ptr())->active_rows;
        consume.physical_rows = family.rows; consume.stage_ordinal = stage; consume.model_layer_index = layers[stage];
        add(1, std::make_shared<MoEOverlayActivationDispatchConsumeStage>(consume), {}, {},
            {received_.tensor.get(), received_indices_.tensor.get(), received_weights_.tensor.get()});
        ResidualAddStage::Params work; work.device_id = devices_[1]; work.input = route_input_.tensor.get(); work.output = local_.tensor.get();
        work.num_elements = family.rows * topk * width;
        const auto produced = add(1, std::make_shared<ResidualAddStage>(work), {route_input_.tensor.get()}, {}, {local_.tensor.get()});
        MoEOverlayActivationReturnPackStage::Params returned;
        returned.device_id = devices_[1]; returned.lane = family.lane[1]; returned.local_canonical_route_contributions = local_.tensor.get();
        returned.physical_rows = family.rows; returned.stage_ordinal = stage; returned.model_layer_index = layers[stage];
        add(1, std::make_shared<MoEOverlayActivationReturnPackStage>(returned), {}, {{local_.tensor.get(), produced}}, {});
        }
    }
}

void NativeExactPacketFixture::capture(std::size_t f, std::size_t p) {
    auto &family = family_[f];
    GraphCaptureDependencyLedger ledger(devices_[p], streams_[p], family.dependencies[p], "original exact packet complete parent");
    workers_[p]->submitAndWait([&, p] {
        for (auto &stage : family.stages[p]) require(stage->prepareGraphLaunch(contexts_[p].get(), streams_[p]), "Exact packet cold stage preparation failed");
        // Every original producer event is joined before capture, including the
        // first retained parent. Neither capture nor a different stream invents it.
        const auto tensors = p == 0 ? std::vector{&hidden_, &indices_, &weights_, &canonical_, &zero_, &placement_[0]}
                                    : std::vector{&received_, &received_indices_, &received_weights_, &local_, &route_input_, &placement_[1]};
        for (auto *tensor : tensors) TransferEngine::requireDeviceInput(tensor->tensor.get(), devices_[p], streams_[p]);
        auto &definition = Graph::capture(family.graph[p]);
        ScopedBackendGraphCapture recording(*workers_[p], definition, "original exact packet parent", &ledger);
        require(recording.begin(), "Exact packet original capture failed");
        for (auto &stage : family.stages[p]) {
            ScopedGraphCaptureStage captured(stage.get(), streams_[p]);
            require(stage->execute(contexts_[p].get()), "Exact packet original stage recording failed"); captured.complete();
        }
        recording.finish();
        const std::size_t expected_nodes = p == 0 ? (family.rows == 1 ? 2u + 2u * layers.size() : 2u + 6u * layers.size())
                                                  : (family.rows == 1 ? 1u + 3u * layers.size() : 1u + 7u * layers.size());
        if (devices_[p].is_rocm()) require(definition.nodeCount() == expected_nodes,
            "Exact packet HIP parent changed its complete native node topology");
        require(definition.nodeCount() > 0 && definition.prepareRuntimeContextStorage(memory_) && definition.instantiate(), "Exact packet original parent instantiation failed");
        std::vector<GPUGraphKernelNodeInfo> kernels; std::string detail;
        require(definition.inspectKernelNodes(kernels, &detail) && !kernels.empty(), "Exact packet native kernel DAG is missing");
        if (devices_[p].is_rocm()) {
            std::size_t waits = 0;
            for (const auto &node : kernels) {
                require(node.nesting_depth == 0, "Exact packet HIP parent retained a nested kernel");
                waits += node.name.find("waitSystemValue64") != std::string::npos;
            }
            require(waits == (family.rows == 1 ? 1u : layers.size() + 1),
                "Original packet HIP parent changed its exact system-wait node population");
        }
        for (const auto &stage : family.stages[p]) family.signatures[p].push_back(stage->graphCaptureVariantSignature());
    });
}

void NativeExactPacketFixture::inputs(int rows, int live, float bias) {
    require(rows > 0 && rows <= capacity && live >= 0 && live <= rows, "Exact packet input exceeds its retained row geometry");
    const auto value = [bias](int row, int column) { return float(row * 100 + column) + .25f + bias; };
    for (std::size_t p = 0; p < 2; ++p) if (owns(p)) workers_[p]->submitAndWait([&, p] {
        auto *meta = placement_[p].host<NativeExactPacketPlacement>(); *meta = {};
        for (int bank = 0; bank < 2; ++bank) { meta->epochs[bank] = 41;
            for (int expert = 0; expert < experts; ++expert) meta->routes[bank][expert] = 0; }
        for (int slot = 0; slot < capacity * topk; ++slot) meta->routes[1][10 + slot % 8] = foreign[slot % foreign.size()] ? target_ : 0;
        meta->ticket = {.epoch = 41, .selector = deviceMoEOverlayEpochSelector(3, 1)};
        meta->status = {.epoch = 41, .selector = meta->ticket.selector,
            .operation = static_cast<std::uint32_t>(DeviceMoEOverlayEpochOperation::Acquire),
            .code = static_cast<std::uint32_t>(DeviceMoEOverlayEpochStatusCode::Success), .bank = 1,
            .observed_state = static_cast<std::uint32_t>(DeviceMoEOverlayEpochBankState::Published)};
        meta->active_rows = p ? 0 : live;
        if (p == 0) {
            for (int row = 0; row < capacity; ++row) for (int column = 0; column < width; ++column)
                hidden_.host<float>()[row * width + column] = value(row, column);
            for (int slot = 0; slot < capacity * topk; ++slot) {
                indices_.host<float>()[slot] = float(10 + slot % 8); weights_.host<float>()[slot] = weights[slot % weights.size()];
            }
            std::fill_n(zero_.host<float>(), capacity * topk * width, 0.f);
            std::fill_n(canonical_.host<float>(), capacity * topk * width, poison);
        } else {
            for (auto *tensor : {&received_, &received_indices_, &received_weights_, &local_})
                std::fill_n(tensor->host<float>(), tensor->declaration.rows() * tensor->declaration.columns(), poison);
            std::fill_n(route_input_.host<float>(), capacity * topk * width, 0.f);
            int compact = 0;
            for (int row = 0; row < live; ++row) {
                int route = 0;
                for (int slot = 0; slot < topk; ++slot) if (foreign[(row * topk + slot) % foreign.size()]) {
                    for (int column = 0; column < width; ++column)
                        route_input_.host<float>()[((compact * topk + route) * width) + column] = weights[(row * topk + slot) % weights.size()] * value(row, column);
                    ++route;
                }
                compact += route != 0;
            }
        }
        const auto tensors = p == 0 ? std::vector{&hidden_, &indices_, &weights_, &canonical_, &zero_, &placement_[0]}
                                    : std::vector{&received_, &received_indices_, &received_weights_, &local_, &route_input_, &placement_[1]};
        for (auto *tensor : tensors) {
            TransferEngine::publishHostWrite(tensor->tensor.get());
            TransferEngine::prepareDeviceInput(tensor->tensor.get(), devices_[p], streams_[p]);
        }
    });
}

void NativeExactPacketFixture::arm(Family &family, std::uint64_t generation) {
    if (!owns(0)) return;
    require(!family.identity && generation > 0, "Exact packet must retire its prior immutable admission");
    const MoEOverlayInferenceCommandIdentity command{.request_generation = generation, .command_id = generation, .initial_placement_epoch = 40};
    const auto ticket = makeMoEOverlayInferenceExecutionTicket(topology_, command, generation, generation, 40,
        MoEOverlayInferenceGraphRole::MainPrefill, 1, family.rows, family.rows);
    std::string error;
    family.identity = family.protocol->arm(ticket, generation, 10'000'000'000ULL, &error);
    require(bool(family.identity), error.c_str());
}

void NativeExactPacketFixture::submit(Family &family, std::size_t p) {
    if (!owns(p)) return;
    workers_[p]->submitAndWait([&, p] {
        family.pending[p] = true; // A partially accepted native submit must also drain before teardown.
        require(Graph::submit(family.graph[p], ++family.submissions[p]), "Exact packet original complete parent submit failed");
        const auto outputs = p == 0 ? std::vector{&canonical_}
            : std::vector{&received_, &received_indices_, &received_weights_, &local_, &placement_[1]};
        for (auto *tensor : outputs) TransferEngine::publishDeviceWrite(tensor->tensor.get(), devices_[p], streams_[p]);
    });
}

void NativeExactPacketFixture::await() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (std::size_t p = 0; p < 2; ++p) if (workers_[p]) for (;;) {
        GPUStreamExecutionState state = GPUStreamExecutionState::Pending;
        workers_[p]->submitAndWait([&] { state = workers_[p]->queryStreamExecutionState(streams_[p], "original exact packet terminal"); });
        if (state == GPUStreamExecutionState::Complete) break;
        require(std::chrono::steady_clock::now() < deadline, "Original exact packet terminal timed out");
        std::this_thread::yield();
    }
}

void NativeExactPacketFixture::validate(Family &family, int live, float bias) {
    originalOutcome([&] { await(); });
    originalOutcome([&] {
    if (!original_) return;
    for (std::size_t p = 0; p < 2; ++p) if (owns(p)) {
        require(Graph::observeTerminal(family.graph[p]), "Exact packet original parent lacks terminal proof");
        family.pending[p] = false;
    }
    for (std::size_t p = 0; p < 2; ++p) if (owns(p)) workers_[p]->submitAndWait([&, p] {
        NativeExactPacketSnapshot view(snapshot_->mutableHostData(), capacity, width); auto *out = &view; auto *backend = getBackendFor(devices_[p]);
        const auto& control=*family.lane[p].control_host;
        if (control.continuation_status.typedCode()!=MoEOverlayActivationStatusCode::Success ||
            control.follower_status.typedCode()!=MoEOverlayActivationStatusCode::Success) {
            // Existing bounded infrastructure copy, only after both original
            // native terminal proofs. No passing replay acquires extra D2H.
            TransferEngine::instance().enqueuePersistentDeviceRegionToMappedHost(
                family.lane[p].grant_device,sizeof(MoEOverlayActivationDeviceEpochGrant),0,*snapshot_,
                NativeExactPacketSnapshotLayout::grantsOffset(capacity,width)+p*sizeof(MoEOverlayActivationDeviceEpochGrant),
                sizeof(MoEOverlayActivationDeviceEpochGrant),devices_[p],streams_[p]);
        }
        const auto copy = [&](void *destination, NativeExactPacketTensor &tensor, std::size_t bytes) {
            TransferEngine::requireDeviceInput(tensor.tensor.get(), devices_[p], streams_[p]);
            require(backend->deviceToHostOnStream(destination, tensor.tensor->gpu_data_ptr(), bytes, devices_[p].ordinal, streams_[p]), "Exact packet terminal copy failed");
        };
        if (!p) copy(out->canonical.data(), canonical_, out->canonical.size_bytes());
        else {
            copy(out->hidden.data(), received_, out->hidden.size_bytes()); copy(out->indices.data(), received_indices_, out->indices.size_bytes());
            copy(out->weights.data(), received_weights_, out->weights.size_bytes()); copy(out->local.data(), local_, out->local.size_bytes());
            TransferEngine::requireDeviceInput(placement_[1].tensor.get(), devices_[1], streams_[1]);
            require(backend->deviceToHostOnStream(out->active_rows,
                &static_cast<NativeExactPacketPlacement *>(placement_[1].tensor->gpu_data_ptr())->active_rows,
                sizeof(*out->active_rows), devices_[1].ordinal, streams_[1]), "Exact packet live-row terminal copy failed");
        }
    }); await();
    });
    originalOutcome([&] {
    if (!original_) return;
    for(std::size_t endpoint=0;endpoint<2;++endpoint) if(owns(endpoint)) {
        const auto& control=*family.lane[endpoint].control_host;
        if(control.continuation_status.typedCode()!=MoEOverlayActivationStatusCode::Success ||
            control.follower_status.typedCode()!=MoEOverlayActivationStatusCode::Success)
            throw std::runtime_error(terminalFailure(family,endpoint,live));
    }
    NativeExactPacketSnapshot out(snapshot_->mutableHostData(), capacity, width);
    const auto value = [bias](int row, int column) { return float(row * 100 + column) + .25f + bias; };
    int compact = 0, entries = 0;
    for (int row = 0; row < capacity; ++row) {
        int route = 0;
        for (int slot = 0; slot < topk; ++slot) {
            const int original = row * topk + slot;
            const bool useful = row < live && foreign[original % foreign.size()];
            if (owns(0)) for (int column = 0; column < width; ++column) {
                const float expected = row >= family.rows ? poison : useful ? weights[original % weights.size()] * value(row, column) : 0.f;
                ASSERT_EQ(std::bit_cast<std::uint32_t>(out.canonical[original * width + column]), std::bit_cast<std::uint32_t>(expected));
            }
            if (useful) {
                if (owns(1)) ASSERT_EQ(out.indices[compact * topk + route], float(10 + original % 8));
                if (owns(1)) ASSERT_EQ(out.weights[compact * topk + route], weights[original % weights.size()]); ++route; ++entries;
            }
        }
        if (route) { if (owns(1)) for (int column = 0; column < width; ++column) ASSERT_EQ(out.hidden[compact * width + column], value(row, column)); ++compact; }
    }
    if (owns(1)) ASSERT_EQ(*out.active_rows, compact);
    if (owns(1)) for (int row = family.rows; row < capacity; ++row) {
        for (int column = 0; column < width; ++column) ASSERT_EQ(out.hidden[row * width + column], poison);
        for (int slot = 0; slot < topk; ++slot) {
            ASSERT_EQ(out.indices[row * topk + slot], poison); ASSERT_EQ(out.weights[row * topk + slot], poison);
            for (int column = 0; column < width; ++column) ASSERT_EQ(out.local[(row * topk + slot) * width + column], poison);
        }
    }
    if (owns(0)) {
    std::string error; const auto traffic = family.protocol->completedTraffic(*family.identity, &error);
    require(bool(traffic), error.c_str());
    ASSERT_EQ(traffic->dispatch_payload_bytes, layers.size() * moeOverlayDispatchPayloadBytes(compact, entries, width));
    ASSERT_EQ(traffic->dispatch_shared_hidden_bytes, family.rows == 1 ? 0u : layers.size() * compact * width * sizeof(float));
    ASSERT_EQ(traffic->return_payload_bytes, layers.size() * moeOverlayReturnPayloadBytes(entries, width));
    ASSERT_EQ(traffic->dispatch_live_rows, layers.size() * compact); ASSERT_EQ(traffic->return_live_rows, layers.size() * compact);
    ASSERT_EQ(traffic->dispatch_live_entries, layers.size() * entries); ASSERT_EQ(traffic->return_live_entries, layers.size() * entries);
    ASSERT_EQ(traffic->dispatch_stage_count, layers.size()); ASSERT_EQ(traffic->return_stage_count, layers.size());
    }
    for (std::size_t p = 0; p < 2; ++p) if (owns(p)) for (std::size_t stage = 0; stage < family.stages[p].size(); ++stage)
        ASSERT_EQ(family.stages[p][stage]->graphCaptureVariantSignature(), family.signatures[p][stage]);
    });
}


std::string NativeExactPacketFixture::terminalFailure(const Family& family,std::size_t endpoint,int live) const {
    const auto& lane=family.lane[endpoint];const auto& control=*lane.control_host;
    const NativeExactPacketSnapshot snapshot(snapshot_->mutableHostData(),capacity,width);
    const auto& grant=snapshot.grants[endpoint];
    std::ostringstream out;
    out << "Original mapped packet rejected terminal metadata rank=" << MPIContextFactory::global()->rank()
        << " endpoint=" << endpoint << " device=" << devices_[endpoint].toString()
        << " generation=" << control.identity.epoch_generation << " physical_rows=" << family.rows
        << " live_rows=" << live << " grant={generation=" << grant.generation << ",placement=" << grant.placement_epoch
        << ",state=" << grant.state << ",code=" << grant.code << ",rows=" << grant.physical_rows
        << ",digest_matches_identity=" << (grant.digest==control.identity.digest) << ",published=" << grant.published_stage_count
        << ",last_published=" << grant.last_published_stage << ",last_consumed=" << grant.last_consumed_stage << "}";
    for(const auto* status:{&control.continuation_status,&control.follower_status})
        out << " status={operation=" << status->operation << ",state=" << status->state << ",code=" << status->code
            << ",generation=" << status->generation << ",layer=" << status->last_model_layer
            << ",timeline=" << status->observed_timeline << ",diagnostic=" << status->failure_diagnostic
            << ",auxiliary=" << status->failure_auxiliary << "}";
    for(std::size_t bank=0;bank<kMoEOverlayActivationBufferCount;++bank) {
        const auto& descriptor=control.buffers[bank].dispatch_descriptor;
        out << " dispatch[" << bank << "]={signal=" << control.buffers[bank].dispatch_signal.value
            << ",timeline=" << descriptor.timeline << ",stage=" << descriptor.stage_ordinal
            << ",layer=" << descriptor.model_layer_index << ",placement=" << descriptor.placement_epoch
            << ",digest_matches_grant=" << (descriptor.digest==grant.digest)
            << ",rows=" << descriptor.live_rows << ",entries=" << descriptor.live_entries
            << ",bytes=" << descriptor.payload_bytes << "}";
    }
    // These original mapped bytes are inspected only after every native stream
    // retired. Translate through this exact registered alias; no foreign pointer
    // or reconstructed allocation is accepted as a host view.
    const auto host=[&]<class T>(const T* pointer,std::size_t count)->const T* {
        const auto base=reinterpret_cast<std::uintptr_t>(lane.mapped_region->deviceAlias(devices_[endpoint]));
        const auto address=reinterpret_cast<std::uintptr_t>(pointer);
        require(address>=base && count<=std::numeric_limits<std::size_t>::max()/sizeof(T) &&
            lane.mapped_region->contains(address-base,count*sizeof(T)), "Original packet failure metadata escaped its registered region");
        return static_cast<const T*>(lane.mapped_region->mutableHostData(address-base));
    };
    const auto* row_ids=host(lane.dispatch.row_ids,capacity);
    const auto* offsets=host(lane.dispatch.entry_offsets,capacity+1);
    const auto* ids=host(lane.dispatch.expert_ids,capacity*topk);
    const auto* route_weights=host(lane.dispatch.route_weights,capacity*topk);
    const auto* original_slots=host(lane.dispatch.original_route_slots,capacity*topk);
    const auto* compact_slots=host(lane.dispatch.compact_route_slots,capacity*topk);
    int compact=0,entries=0,first_bad_row=-1,first_bad_entry=-1;
    std::string row_evidence,entry_evidence;
    for(int row=0;row<live;++row) {
        const int begin=entries;int matches=0;
        for(int slot=0;slot<topk;++slot) if(foreign[(row*topk+slot)%foreign.size()]) {
            const int original=row*topk+slot;
            const auto expected_weight=weights[original%weights.size()];
            if(first_bad_entry<0 && (ids[entries]!=10+original%8 || original_slots[entries]!=original ||
                compact_slots[entries]!=compact*topk+matches ||
                std::bit_cast<std::uint32_t>(route_weights[entries])!=std::bit_cast<std::uint32_t>(expected_weight))) {
                first_bad_entry=entries;std::ostringstream evidence;
                evidence << " index=" << entries << " expert=" << ids[entries] << "/" << 10+original%8
                    << " original=" << original_slots[entries] << "/" << original << " compact=" << compact_slots[entries]
                    << "/" << compact*topk+matches << " weight=" << route_weights[entries] << "/" << expected_weight;
                entry_evidence=evidence.str();
            }
            ++entries;++matches;
        }
        if(matches) {
            if(first_bad_row<0 && (row_ids[compact]!=row || offsets[compact]!=begin || offsets[compact+1]!=entries)) {
                first_bad_row=compact;std::ostringstream evidence;
                evidence << " index=" << compact << " row=" << row_ids[compact] << "/" << row
                    << " begin=" << offsets[compact] << "/" << begin << " end=" << offsets[compact+1] << "/" << entries;
                row_evidence=evidence.str();
            }
            ++compact;
        }
    }
    out << " original_wire={expected_rows=" << compact << ",expected_entries=" << entries
        << ",first_bad_row=" << first_bad_row << row_evidence << ",first_bad_entry=" << first_bad_entry << entry_evidence << "}";
    return out.str();
}

void NativeExactPacketFixture::reset(Family &family) {
    if (!owns(0)) return;
    std::string error; require(family.identity && family.protocol->reset(*family.identity, &error), error.c_str());
    const auto &control = mapped_->activationEpochControl(target_, &family - family_.data());
    for (const auto &bank : control.buffers) { EXPECT_EQ(bank.dispatch_signal.value, 0); EXPECT_EQ(bank.return_signal.value, 0); }
    family.identity.reset();
}

void NativeExactPacketFixture::run() {
    for (std::uint64_t generation = 1; generation <= geometry_.generations; ++generation) for (auto &family : family_) {
        const bool follower_prequeued = replay_ == NativeExactPacketReplay::FollowerPrequeued ||
            replay_ == NativeExactPacketReplay::FollowerPrequeuedAbort;
        const bool changed = replay_ == NativeExactPacketReplay::ChangedPublication || follower_prequeued;
        const int live = !changed || generation == 1 ? family.rows : std::array{family.rows, family.rows - 1, 0, 1}[(generation - 2) % 4];
        const float bias = changed ? float(generation * 16) : 0.f;
        SCOPED_TRACE(::testing::Message() << "original_rank=" << MPIContextFactory::global()->rank()
            << " source=" << devices_[0].toString() << " target=" << devices_[1].toString()
            << " exact_rows=" << family.rows << " live_rows=" << live << " generation=" << generation);
        const bool queued = (changed || replay_ == NativeExactPacketReplay::StaleAdmission || replay_ == NativeExactPacketReplay::AdmissionAbort) && generation % 2 == 0;
        if (follower_prequeued && generation>1) {
            // The actual follower is queued while the previous successful
            // admission and dispatch timelines remain published. Its native
            // scheduling waits may pass; the private grant must await rearm.
            originalOutcome([&] {
                if(original_)inputs(family.rows,live,bias);
                submit(family,1);
                if(owns(1)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(25));
                    GPUStreamExecutionState state=GPUStreamExecutionState::Complete;
                    workers_[1]->submitAndWait([&] { state=workers_[1]->queryStreamExecutionState(streams_[1],"prequeued follower awaits new grant"); });
                    EXPECT_EQ(state,GPUStreamExecutionState::Pending);
                }
            });
            originalOutcome([&] { reset(family);arm(family,generation); });
            // New identity is now admitted, but no source graph for it exists
            // yet. The old single-load validator completes here with code8.
            originalOutcome([&] {
                if(owns(1)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(25));
                    GPUStreamExecutionState state=GPUStreamExecutionState::Complete;
                    workers_[1]->submitAndWait([&] { state=workers_[1]->queryStreamExecutionState(streams_[1],"new grant awaits its own delayed dispatch"); });
                    EXPECT_EQ(state,GPUStreamExecutionState::Pending);
                }
            });
            if(replay_==NativeExactPacketReplay::FollowerPrequeuedAbort) {
                originalOutcome([&] {
                    if(owns(0)) { std::string error;
                        require(family.protocol->cancelOriginalInvocation(*family.identity,MoEOverlayActivationStatusCode::ExplicitAbort,&error),error.c_str()); }
                });
                originalOutcome([&] { await(); });
                originalOutcome([&] {
                    if(owns(1)) {
                        require(Graph::observeTerminal(family.graph[1]),"Cancelled follower lost its original native terminal");
                        family.pending[1]=false;
                        EXPECT_EQ(family.lane[1].control_host->follower_status.typedState(),MoEOverlayActivationEndpointState::Aborted);
                        // Actual acquisition cutpoint is observed only at the
                        // terminal boundary; it never drives native work.
                        workers_[1]->submitAndWait([&] {
                            TransferEngine::instance().enqueuePersistentDeviceRegionToMappedHost(family.lane[1].grant_device,
                                sizeof(MoEOverlayActivationDeviceEpochGrant),0,*snapshot_,
                                NativeExactPacketSnapshotLayout::grantsOffset(capacity,width)+sizeof(MoEOverlayActivationDeviceEpochGrant),
                                sizeof(MoEOverlayActivationDeviceEpochGrant),devices_[1],streams_[1]);
                        });
                        await();const NativeExactPacketSnapshot snapshot(snapshot_->mutableHostData(),capacity,width);
                        const auto& grant=snapshot.grants[1];
                        EXPECT_EQ(grant.state,static_cast<std::uint32_t>(MoEOverlayActivationEndpointState::Aborted));
                        if(grant.generation==generation) {
                            EXPECT_EQ(grant.code,static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::PeerAborted));
                            EXPECT_EQ(grant.published_stage_count,0u);
                            EXPECT_EQ(grant.live_rows,0u);EXPECT_EQ(grant.live_entries,0u);
                        }
                        else { EXPECT_EQ(grant.generation,generation-1);
                            EXPECT_EQ(grant.code,static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::InvalidControl)); }
                    }
                    if(owns(0)) { EXPECT_EQ(family.protocol->admissionState(),MoEOverlayActivationAdmissionState::Failed);
                        EXPECT_EQ(family.lane[0].control_host->admission.code,static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::ExplicitAbort));
                        std::string error;EXPECT_FALSE(family.protocol->reset(*family.identity,&error)); }
                });
                continue; // Cancel each independent retained 4/3/1 family, including fused scalar.
            }
            originalOutcome([&] { submit(family,0); });
        } else if (queued) {
            originalOutcome([&] {
            if (original_) inputs(family.rows, live, bias);
            if (owns(0)) {
            require(bool(family.identity), "Queued exact packet lost the prior admission");
            submit(family, 0);
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            GPUStreamExecutionState state = GPUStreamExecutionState::Complete;
            workers_[0]->submitAndWait([&] { state = workers_[0]->queryStreamExecutionState(streams_[0], "exact packet old admission wait"); });
            EXPECT_EQ(state, GPUStreamExecutionState::Pending);
            EXPECT_EQ(family.lane[0].control_host->continuation_status.typedState(), MoEOverlayActivationEndpointState::Complete);
            EXPECT_EQ(family.lane[0].control_host->continuation_status.typedCode(), MoEOverlayActivationStatusCode::Success);
            }
            });
            if (replay_ == NativeExactPacketReplay::AdmissionAbort) {
                originalOutcome([&] {
                if (owns(0)) {
                    std::string error;
                    require(family.protocol->cancelOriginalInvocation(*family.identity, MoEOverlayActivationStatusCode::ExplicitAbort, &error), error.c_str());
                }
                await();
                });
                originalOutcome([&] {
                if (owns(0)) {
                    require(Graph::observeTerminal(family.graph[0]), "Cancelled original packet lost its actual native terminal");
                    family.pending[0] = false;
                    const auto &control = *family.lane[0].control_host;
                    EXPECT_EQ(control.continuation_status.typedState(), MoEOverlayActivationEndpointState::Aborted);
                    EXPECT_EQ(family.protocol->admissionState(), MoEOverlayActivationAdmissionState::Failed);
                    std::string error;
                    EXPECT_FALSE(family.protocol->reset(*family.identity, &error));
                }
                });
                return;
            }
            originalOutcome([&] { reset(family); arm(family, generation); submit(family, 1); });
        } else {
            originalOutcome([&] {
            if (original_) inputs(family.rows, live, bias);
            if (family.identity) reset(family); arm(family, generation); submit(family, 1);
            if (owns(1)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            GPUStreamExecutionState state = GPUStreamExecutionState::Complete;
            workers_[1]->submitAndWait([&] { state = workers_[1]->queryStreamExecutionState(streams_[1], "exact packet unpublished dispatch wait"); });
            EXPECT_EQ(state, GPUStreamExecutionState::Pending); }
            });
            originalOutcome([&] { submit(family, 0); });
        }
        validate(family, live, bias);
    }
    if(replay_==NativeExactPacketReplay::FollowerPrequeuedAbort)return;
    originalOutcome([&] { for (auto &family : family_) reset(family); });
}

void NativeExactPacketFixture::dispose() {
    if (disposed_) return;
    if (!original_) { disposed_ = true; return; }
    const bool abort_pending = std::any_of(family_.begin(), family_.end(), [](const Family &family) { return family.pending[0] || family.pending[1]; });
    for (std::size_t index = 0; abort_pending && mapped_ && index < family_.size(); ++index) {
        auto &control = mapped_->activationEpochControl(target_, index);
        for (auto &bank : control.buffers) {
            std::atomic_ref<std::uint64_t>(bank.dispatch_signal.value).store(kMoEOverlayActivationAbortTimeline, std::memory_order_release);
            std::atomic_ref<std::uint64_t>(bank.return_signal.value).store(kMoEOverlayActivationAbortTimeline, std::memory_order_release);
        }
        std::atomic_ref<std::uint64_t>(control.admission.ready_signal).store(kMoEOverlayActivationAbortTimeline, std::memory_order_release);
    }
    await();
    for (auto &family : family_) for (std::size_t p = 0; p < 2; ++p)
        if (family.graph[p] && family.submissions[p]) require(Graph::observeTerminal(family.graph[p]), "Exact packet failure retirement lost native terminal proof");
    for (auto &family : family_) { family.graph = {}; family.stages = {}; family.dependencies = {}; }
    for (auto &family : family_) { family.lane = {}; family.protocol.reset(); family.identity.reset(); }
    mapped_.reset(); transport_.reset();
    snapshot_.reset(); snapshot_claim_.reset(); contexts_ = {};
    for (auto *tensor : {&hidden_, &indices_, &weights_, &canonical_, &zero_, &received_, &received_indices_, &received_weights_, &local_, &route_input_, &placement_[0], &placement_[1]}) tensor->tensor.reset();
    workers_ = {};
    if (admitted_) original_->requirePayloadRetired();
    original_->retire(); original_->requireExactRetired(); disposed_ = true;
}

/** Retain the original cohort scope through all payload, graph and MPI retirement. */
void nativeExactMappedPacket(DeviceType backend, NativeExactPacketReplay replay) {
    originalRankLocalReplay(backend, 2, [&](const auto &cohort) {
        auto devices = cohort;
        // Changed-publication reverses actual roles so source ordinal1 and
        // target ordinal0 use the same original ownership/PMA path.
        if (replay == NativeExactPacketReplay::ChangedPublication) std::reverse(devices.begin(), devices.end());
        NativeExactPacketFixture fixture(std::move(devices), replay); fixture.prepare(); fixture.run();
    });
}
/**
 * @brief Retain the original distributed endpoint owner through both captured parents.
 * @param source_family Actual continuation family selected from immutable original inventory.
 * @param target_family Actual follower family on another original rank of the same host.
 * @param geometry Exact retained row/width program; allocation capacity never becomes transfer extent.
 * @param replay Ordinary numerical/grid replay, stale-admission or absorbing cancellation proof.
 * @details No raw pointer or process-local graph handle crosses MPI. Both native
 * endpoints submit through their own original prepared workers; only cold setup
 * identity and immutable test-phase completion use the original communicator.
 */
void nativeMappedOriginalRanks(DeviceType source_family, DeviceType target_family,
    NativeMappedPacketGeometry geometry, NativeExactPacketReplay replay = NativeExactPacketReplay::Ordinary) {
    ScopedActivationEpochPerfStats perf_stats;
    const auto mpi = MPIContextFactory::global();
    require(mpi && mpi->clusterInventory(), "Mapped packet requires its original discovered MPI inventory");
    const auto endpoints = rankReplayEndpoints(*mpi->clusterInventory(), source_family, target_family);
    NativeExactPacketFixture fixture({endpoints[0].device, endpoints[1].device}, replay,
        geometry, std::array{endpoints[0].rank, endpoints[1].rank});
    fixture.prepare();
    assertMappedCaptureEvidence(geometry.shapes,3,true,endpoints[0].rank);
    fixture.run();
}
}
#ifdef HAVE_CUDA
TEST(NativeExactMappedPacket, CUDA2ExactRows) { nativeExactMappedPacket(DeviceType::CUDA, NativeExactPacketReplay::ExactRows); }
TEST(NativeExactMappedPacket, CUDA2ChangedPublication) { nativeExactMappedPacket(DeviceType::CUDA, NativeExactPacketReplay::ChangedPublication); }
#endif
#ifdef HAVE_ROCM
TEST(NativeExactMappedPacket, ROCm2ExactRows) { nativeExactMappedPacket(DeviceType::ROCm, NativeExactPacketReplay::ExactRows); }
TEST(NativeExactMappedPacket, ROCm2ChangedPublication) { nativeExactMappedPacket(DeviceType::ROCm, NativeExactPacketReplay::ChangedPublication); }
#endif

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
/** @test Fresh follower grants cannot consume the previous generation's dispatch wait. */
TEST(Test__MappedActivationPacketCUDAAndROCm, CUDAContinuationDelayedDispatchAfterFreshFollowerGrant) {
    nativeMappedOriginalRanks(DeviceType::CUDA,DeviceType::ROCm,
        {.backing_rows=4,.width=67,.shapes={4,3,1},.generations=20},NativeExactPacketReplay::FollowerPrequeued);
}
/** @test The reverse native backend retains the identical grant/publication order. */
TEST(Test__MappedActivationPacketCUDAAndROCm, ROCmContinuationDelayedDispatchAfterFreshFollowerGrant) {
    nativeMappedOriginalRanks(DeviceType::ROCm,DeviceType::CUDA,
        {.backing_rows=4,.width=67,.shapes={4,3,1},.generations=20},NativeExactPacketReplay::FollowerPrequeued);
}
/** @test Absorbing original cancellation retires the waiting ROCm follower without payload. */
TEST(Test__MappedActivationPacketCUDAAndROCm, CUDAContinuationCancelsFreshFollowerBeforeDispatch) {
    nativeMappedOriginalRanks(DeviceType::CUDA,DeviceType::ROCm,
        {.backing_rows=4,.width=67,.shapes={4,3,1},.generations=2},NativeExactPacketReplay::FollowerPrequeuedAbort);
}
/** @test Absorbing original cancellation also retires the waiting CUDA follower. */
TEST(Test__MappedActivationPacketCUDAAndROCm, ROCmContinuationCancelsFreshFollowerBeforeDispatch) {
    nativeMappedOriginalRanks(DeviceType::ROCm,DeviceType::CUDA,
        {.backing_rows=4,.width=67,.shapes={4,3,1},.generations=2},NativeExactPacketReplay::FollowerPrequeuedAbort);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, PlannerSelectedCUDAContinuationROCmFollowerIsExact) {
    nativeMappedOriginalRanks(DeviceType::CUDA, DeviceType::ROCm,
        {.backing_rows=4, .width=8, .shapes={4}, .generations=2}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, PlannerSelectedROCmContinuationCUDAFollowerIsExact) {
    nativeMappedOriginalRanks(DeviceType::ROCm, DeviceType::CUDA,
        {.backing_rows=4, .width=8, .shapes={4}, .generations=2}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, CooperativeReturnDescriptorCUDAReplaysTwentyGenerations) {
    nativeMappedOriginalRanks(DeviceType::CUDA, DeviceType::ROCm,
        {.backing_rows=17, .width=8, .shapes={17}, .generations=20}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, CooperativeReturnDescriptorROCmReplaysTwentyGenerations) {
    nativeMappedOriginalRanks(DeviceType::ROCm, DeviceType::CUDA,
        {.backing_rows=17, .width=8, .shapes={17}, .generations=20}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, CUDAContinuationReturnRowsCoverGridStrideAndUnalignedWidth) {
    nativeMappedOriginalRanks(DeviceType::CUDA, DeviceType::ROCm,
        {.backing_rows=257, .width=67, .shapes={257}, .generations=20}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, ROCmContinuationReturnRowsCoverGridStrideAndUnalignedWidth) {
    nativeMappedOriginalRanks(DeviceType::ROCm, DeviceType::CUDA,
        {.backing_rows=257, .width=67, .shapes={257}, .generations=20}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, CUDAFollowerDispatchRowsPreserveWidePayloadsAcrossTwentyGenerations) {
    nativeMappedOriginalRanks(DeviceType::ROCm, DeviceType::CUDA,
        {.backing_rows=33, .width=3073, .shapes={33}, .generations=20}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, ROCmFollowerDispatchRowsPreserveWidePayloadsAcrossTwentyGenerations) {
    nativeMappedOriginalRanks(DeviceType::CUDA, DeviceType::ROCm,
        {.backing_rows=33, .width=3073, .shapes={33}, .generations=20}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, SingleRowCUDAContinuationROCmFollowerFusesTimelineAndIsExact) {
    nativeMappedOriginalRanks(DeviceType::CUDA, DeviceType::ROCm,
        {.backing_rows=1, .width=8, .shapes={1}, .generations=2}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, SingleRowWideReturnRetainsBytesAcrossTwentyGenerations_ROCMToCUDA_Width3073) {
    nativeMappedOriginalRanks(DeviceType::ROCm, DeviceType::CUDA,
        {.backing_rows=1, .width=3073, .shapes={1}, .generations=20}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, SingleRowWideReturnRetainsBytesAcrossTwentyGenerations_ROCMToCUDA_Width3072) {
    nativeMappedOriginalRanks(DeviceType::ROCm, DeviceType::CUDA,
        {.backing_rows=1, .width=3072, .shapes={1}, .generations=20}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, SingleRowWideReturnRetainsBytesAcrossTwentyGenerations_CUDAToROCM_Width3073) {
    nativeMappedOriginalRanks(DeviceType::CUDA, DeviceType::ROCm,
        {.backing_rows=1, .width=3073, .shapes={1}, .generations=20}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, SingleRowWideReturnRetainsBytesAcrossTwentyGenerations_CUDAToROCM_Width3072) {
    nativeMappedOriginalRanks(DeviceType::CUDA, DeviceType::ROCm,
        {.backing_rows=1, .width=3072, .shapes={1}, .generations=20}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, CUDAContinuationLargeSparseReturnTraversesEveryGridStride) {
    nativeMappedOriginalRanks(DeviceType::CUDA, DeviceType::ROCm,
        {.backing_rows=16385, .width=8, .shapes={16385}, .generations=2}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, ROCmContinuationLargeSparseReturnTraversesEveryGridStride) {
    nativeMappedOriginalRanks(DeviceType::ROCm, DeviceType::CUDA,
        {.backing_rows=16385, .width=8, .shapes={16385}, .generations=2}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, SingleRowROCmContinuationCUDAFollowerFusesTimelineAndIsExact) {
    nativeMappedOriginalRanks(DeviceType::ROCm, DeviceType::CUDA,
        {.backing_rows=1, .width=8, .shapes={1}, .generations=2}, NativeExactPacketReplay::Ordinary);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, CUDAContinuationWaitsForNewGenerationAfterStaleAdmission) {
    nativeMappedOriginalRanks(DeviceType::CUDA, DeviceType::ROCm,
        {.backing_rows=1, .width=8, .shapes={1}, .generations=2}, NativeExactPacketReplay::StaleAdmission);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, ROCmContinuationWaitsForNewGenerationAfterStaleAdmission) {
    nativeMappedOriginalRanks(DeviceType::ROCm, DeviceType::CUDA,
        {.backing_rows=1, .width=8, .shapes={1}, .generations=2}, NativeExactPacketReplay::StaleAdmission);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, CUDAForkedPrefillWaitsForNewGenerationAfterStaleAdmission) {
    nativeMappedOriginalRanks(DeviceType::CUDA, DeviceType::ROCm,
        {.backing_rows=4, .width=8, .shapes={4}, .generations=2}, NativeExactPacketReplay::StaleAdmission);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, ROCmForkedPrefillWaitsForNewGenerationAfterStaleAdmission) {
    nativeMappedOriginalRanks(DeviceType::ROCm, DeviceType::CUDA,
        {.backing_rows=4, .width=8, .shapes={4}, .generations=2}, NativeExactPacketReplay::StaleAdmission);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, CUDAForkedPrefillAbortDrainsPendingAdmission) {
    nativeMappedOriginalRanks(DeviceType::CUDA, DeviceType::ROCm,
        {.backing_rows=4, .width=8, .shapes={4}, .generations=2}, NativeExactPacketReplay::AdmissionAbort);
}
/** Original numerical/grid/lifecycle oracle on authenticated distinct-rank endpoint GPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, ROCmForkedPrefillAbortDrainsPendingAdmission) {
    nativeMappedOriginalRanks(DeviceType::ROCm, DeviceType::CUDA,
        {.backing_rows=4, .width=8, .shapes={4}, .generations=2}, NativeExactPacketReplay::AdmissionAbort);
}
#endif
