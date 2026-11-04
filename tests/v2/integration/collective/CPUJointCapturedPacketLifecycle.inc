/**
 * @file CPUJointCapturedPacketLifecycle.inc
 * @brief Genuine two-socket prepared fragments behind retained GPU dispatch and
 * return.
 *
 * Rank0 is both the GPU continuation and the original CPU root. Rank1 is the
 * other original CPU member. The source-local edge uses its real mapped ticket;
 * only the remote member has a mapped activation channel. No extra rank, local
 * CPU packet channel, synthetic completion or terminal cancellation substitutes
 * for an ordinary successful invocation/reset.
 */
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
#include "../../utils/CPUJointSourceTestPlan.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/ServingGGUFFixturePayload.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/compute_stages/ComputeStageFactory.h"
#include "execution/compute_stages/stages/MoECPUExpertShardRouteStage.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoEOverlayTicketConsumeStage.h"
#include "execution/compute_stages/stages/MoEOverlayTicketPublishStage.h"
#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/moe/CanonicalMoERouteRecord.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEGraphOwnerProjection.h"
#include "execution/moe/MoEOverlayActivationEpochProtocol.h"
#include "execution/moe/MoEOverlayCapacityResolver.h"
#include "execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#ifdef HAVE_CUDA
#include "kernels/cuda/moe/CUDAMoEKernel.h"
#endif
#ifdef HAVE_ROCM
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#endif
#include "loaders/WeightManager.h"
#include "memory/BufferArena.h"
#include "transfer/TransferEngine.h"
#include "utils/PerfStatsCollector.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <thread>

namespace rooted_packet_lifecycle
{
using namespace llaminar2;
constexpr int rows = 64, hidden = 256, intermediate = 16, experts = 8, top_k = 2;
constexpr size_t slots = size_t(rows) * top_k;
constexpr uint64_t workspace_generation = 17;
constexpr uint64_t invocation_count = 20;
/** @brief Recorded valid request rows and independently expected CPU cohort work. */
struct RecordedLayerWork
{
    int32_t logical_rows;
    int32_t cohort_rows;
};
constexpr std::array<RecordedLayerWork, 4> live_row_trace{{{9, 9}, {9, 0}, {1, 1}, {9, 9}}};
/** @brief Build immutable recorded routes to the first two experts of one tier.
 * @param first_expert First actual expert in the original placement.
 * @return Exact FP32 expert IDs, retained for every asynchronous replay upload.
 */
constexpr std::array<float, rows * top_k> recordedRoutes(int first_expert)
{
    std::array<float, rows * top_k> result{};
    for (int row = 0; row < rows; ++row)
        for (int slot = 0; slot < top_k; ++slot)
            result[row * top_k + slot] = float(first_expert + slot);
    return result;
}
constexpr auto cohort_routes = recordedRoutes(2);
constexpr auto continuation_routes = recordedRoutes(0);
/** @brief Abort the actual MPI job before admitted owners unwind.
 * @param message Original failure diagnostic. */
[[noreturn]] inline void abortUnexpected(const char *message) noexcept;
/** @brief Collect only the actual source-owned canonical publication phase in
 * this scoped test. */
class PublicationEvidence final
{
  public:
    /** @brief Preserve caller collection policy before selecting the owning
     * domain. */
    PublicationEvidence()
    {
        if (const auto *value = std::getenv("LLAMINAR_PERF_STATS_SUMMARY"))
            summary_ = value;
        if (const auto *value = std::getenv("LLAMINAR_PERF_STATS_FILTER"))
            filter_ = value;
        setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
        setenv("LLAMINAR_PERF_STATS_FILTER", "moe_canonical_publication", 1);
        PerfStatsCollector::reloadConfigurationFromEnvironment();
    }
    /** @brief Restore the original process collection policy after all
     * observations. */
    ~PublicationEvidence()
    {
        if (summary_)
            setenv("LLAMINAR_PERF_STATS_SUMMARY", summary_->c_str(), 1);
        else
            unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
        if (filter_)
            setenv("LLAMINAR_PERF_STATS_FILTER", filter_->c_str(), 1);
        else
            unsetenv("LLAMINAR_PERF_STATS_FILTER");
        PerfStatsCollector::reloadConfigurationFromEnvironment();
    }
    /** @param name Exact production counter. @return Sum only for this actual
     * prefill/rooted stage. */
    double value(const char *name) const
    {
        double total = 0;
        for (const auto &record : PerfStatsCollector::snapshot({"moe_canonical_publication"}))
        {
            const auto stage = record.tags.find("stage");
            if (stage == record.tags.end() || stage->second != "native.rooted_packet_repeat" ||
                record.name != name)
                continue;
            EXPECT_EQ(record.phase, "prefill");
            EXPECT_EQ(record.tags.at("expert_compute"), "tensor_sharded");
            total += record.value;
        }
        return total;
    }

  private:
    std::optional<std::string> summary_, filter_;
};

/** @brief Fatal test setup cannot leave a peer inside an unmatched native
 * operation.
 * @param accepted Actual original API outcome.
 * @param operation Exact failed operation, retained before native abort.
 */
inline void requireOriginal(bool accepted, const char *operation)
{
    if (accepted)
        return;
    ADD_FAILURE() << operation;
    MPI_Abort(MPI_COMM_WORLD, 1);
    throw std::runtime_error(operation);
}

/** @brief Own the actual captured source endpoint; CPU roots remain in the
 * original actor.
 *
 * Small raw device allocations are recorded input/ABI storage at the existing
 * native-kernel test boundary. Model CPU prepared roots, arenas and native
 * domain storage are separately admitted through the original PMA below.
 */
class Source final
{
  public:
    /** @brief Capture the original source endpoint without dispatching it.
     * @param device Actual continuation backend.
     * @param transport Original remote contributor channel owned on rank0.
     * @param owners Same original source map used by both CPU members.
     * @param canonical Actual CPU root publication; no test payload publisher is
     * provided.
     */
    Source(DeviceId device, MoEOverlayNodeLocalRankBatchTransport &transport,
           const MoEExpertOwnerMap &owners,
           std::shared_ptr<MoEOverlayCanonicalRouteReturnTicketStorage> canonical)
        : device_(device), transport_(transport), canonical_(std::move(canonical)),
          hidden_({rows, hidden}), indices_({rows, top_k}), weights_({rows, top_k}),
          output_({slots, size_t(hidden)}), output_poison_(slots * hidden, -991.f)
    {
        backend_ = getBackendFor(device_);
        requireOriginal(backend_ != nullptr, "original source backend");
        stream_ = backend_->createStream(device_.ordinal);
        terminal_ = backend_->createEvent(device_.ordinal);
        requireOriginal(stream_ && terminal_, "original source stream/event");
        if (device_.is_cuda())
        {
#ifdef HAVE_CUDA
            kernel_ = std::make_unique<CUDAMoEKernel>(device_.ordinal);
#else
            throw std::invalid_argument("CUDA rooted packet case requires its compiled backend");
#endif
        }
        else if (device_.is_rocm())
        {
#ifdef HAVE_ROCM
            kernel_ = std::make_unique<ROCmMoEKernel>(device_.ordinal);
#else
            throw std::invalid_argument("ROCm rooted packet case requires its compiled backend");
#endif
        }
        else
            throw std::invalid_argument("Rooted packet case requires its explicit GPU backend");
        banks_ =
            static_cast<DeviceMoEPlacementBank *>(allocate(2 * sizeof(DeviceMoEPlacementBank)));
        epoch_control_ = static_cast<DeviceMoEOverlayEpochControl *>(
            allocate(sizeof(DeviceMoEOverlayEpochControl)));
        epoch_ticket_ = static_cast<DeviceMoEOverlayEpochTicket *>(
            allocate(sizeof(DeviceMoEOverlayEpochTicket)));
        epoch_status_ = static_cast<DeviceMoEOverlayEpochStatus *>(
            allocate(sizeof(DeviceMoEOverlayEpochStatus)));
        active_ = static_cast<int32_t *>(allocate(sizeof(int32_t)));
        initial_banks_ = std::make_unique<DeviceMoEPlacementBank[]>(2);
        auto &initial = initial_banks_;
        const auto targets = ownerParticipantsFromMap(owners, 0, experts);
        for (unsigned bank = 0; bank < 2; ++bank)
        {
            initial[bank].epoch = 1;
            std::copy(targets.begin(), targets.end(), initial[bank].overlay_route_participant);
        }
        auto &control = initial_control_;
        control.bank_epochs[0] = 1;
        control.bank_states[0] = static_cast<uint32_t>(DeviceMoEOverlayEpochBankState::Published);
        control.published_selector = deviceMoEOverlayEpochSelector(1, 0);
        requireOriginal(backend_->hostToDevice(banks_, initial.get(),
                                               2 * sizeof(DeviceMoEPlacementBank), device_.ordinal,
                                               stream_) &&
                            backend_->hostToDevice(epoch_control_, &control, sizeof(control),
                                                   device_.ordinal, stream_),
                        "original source bank upload");
        requireOriginal(
            backend_->memset(epoch_ticket_, 0, sizeof(*epoch_ticket_), device_.ordinal, stream_) &&
                backend_->memset(epoch_status_, 0, sizeof(*epoch_status_), device_.ordinal,
                                 stream_),
            "pristine native acquisition ticket/status");
        requireOriginal(kernel_->acquireMoEOverlayEpoch({.stream = stream_}, epoch_control_,
                                                        epoch_ticket_, epoch_status_),
                        "actual source epoch acquire");
        observeEpoch(DeviceMoEOverlayEpochOperation::Acquire);
        auto metadata = TransferEngine::instance().createMappedHostArena(std::array{device_});
        dispatch_ticket_ = std::make_shared<MoEOverlayDispatchTicketStorage>();
        dispatch_ticket_->bindFixedCapacity(0, rows, top_k, hidden, device_, workspace_generation,
                                            metadata, epoch_ticket_);
        initializeInputs();
        arena_ = std::make_unique<BufferArena>();
        requireOriginal(
            arena_->registerExternalBuffer(BufferId::NORMALIZED, &hidden_) &&
                arena_->registerExternalBuffer(BufferId::MOE_EXPERT_INDICES, &indices_) &&
                arena_->registerExternalBuffer(BufferId::MOE_EXPERT_WEIGHTS, &weights_) &&
                arena_->registerExternalBuffer(BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS,
                                               &output_),
            "source arena bindings");
        context_ = IDeviceContext::create(device_);
        worker_ = &GPUDeviceContextPool::instance().getContext(device_);
        GraphExecutorConfig config;
        config.default_device = device_;
        config.worker_gpu_context_resolver = [this](DeviceId requested)
        { return requested == device_ ? worker_ : nullptr; };
        config.worker_gpu_context_uses_process_pool = false;
        executor_ = std::make_unique<DeviceGraphExecutor>(std::move(config));
        executor_->setArena(arena_.get());
        graph_ = std::make_unique<ComputeGraph>();
        graph_->setNativeCaptureEnvelope(
            GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
        const auto remote =
            transport.activationEpochConfig(remoteParticipant(owners), 0).target.participant_id;
        const auto lane = transport.activationDeviceLane(remote, 0, device_);
        auto transaction = std::make_shared<MoEOverlayActivationLaneBatchState>(
            device_, std::vector{lane}, rows, 0, 0);
        MoEOverlayTicketPublishStage::Params publication;
        publication.device_id = device_;
        publication.hidden = &hidden_;
        publication.routing_indices = &indices_;
        publication.routing_weights = &weights_;
        publication.hidden_buffer_id = BufferId::NORMALIZED;
        publication.routing_indices_buffer_id = BufferId::MOE_EXPERT_INDICES;
        publication.routing_weights_buffer_id = BufferId::MOE_EXPERT_WEIGHTS;
        publication.active_row_count_device = active_;
        publication.layer_idx = 0;
        publication.bucket_rows = rows;
        publication.top_k = top_k;
        publication.d_model = hidden;
        publication.ticket_storage = dispatch_ticket_;
        graph_->addNode("local_dispatch",
                        std::make_unique<MoEOverlayTicketPublishStage>(publication), device_);
        MoEOverlayActivationDispatchPackBatchStage::Params dispatch;
        dispatch.device_id = device_;
        dispatch.transaction = transaction;
        dispatch.hidden = &hidden_;
        dispatch.routing_indices = &indices_;
        dispatch.routing_weights = &weights_;
        dispatch.hidden_buffer_id = BufferId::NORMALIZED;
        dispatch.routing_indices_buffer_id = BufferId::MOE_EXPERT_INDICES;
        dispatch.routing_weights_buffer_id = BufferId::MOE_EXPERT_WEIGHTS;
        dispatch.active_row_count_device = active_;
        dispatch.placement = {.banks = {bankView(0), bankView(1)},
                              .ticket = epoch_ticket_,
                              .status = epoch_status_,
                              .expert_count = experts};
        dispatch.joint_route_targets = {jointRouteTargetFromMap(owners, 0, experts, remote)};
        graph_->addNode("remote_dispatch",
                        ComputeStageFactory::createMoEOverlayActivationDispatchPackBatch(dispatch),
                        device_);
        graph_->addDependency("remote_dispatch", "local_dispatch");
        MoEOverlayActivationReturnConsumeBatchStage::Params returned;
        returned.device_id = device_;
        returned.transaction = transaction;
        returned.canonical_route_contributions = &output_;
        returned.canonical_route_contributions_buffer_id =
            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
        graph_->addNode("contributor_return",
                        ComputeStageFactory::createMoEOverlayActivationReturnConsumeBatch(returned),
                        device_);
        graph_->addDependency("contributor_return", "remote_dispatch");
        MoEOverlayTicketConsumeStage::Params root;
        root.device_id = device_;
        root.output = &output_;
        root.output_buffer_id = BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
        root.layer_idx = 0;
        root.bucket_rows = rows;
        root.top_k = top_k;
        root.d_model = hidden;
        root.canonical_route_ticket_storage = canonical_;
        graph_->addNode("root_return", std::make_unique<MoEOverlayTicketConsumeStage>(root),
                        device_);
        graph_->addDependency("root_return", "contributor_return");
        cache_.steady_replay_host_policy =
            DeviceGraphExecutor::GraphSegmentCache::SteadyReplayHostPolicy::RetainedFullGraph;
        requireOriginal(cache_.bindBorrowedCaptureStream(worker_, stream_, device_, false),
                        "source capture stream");
        graph_->reset();
        requireOriginal(
            submit(DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch),
            "source retained capture");
        requireOriginal(cache_.retained_full_graph_replay.valid() && cache_.segments.size() == 1 &&
                            cache_.segments.front().capture &&
                            cache_.segments.front().capture->hasExecutable() &&
                            cache_.executable_submission_state ==
                                DeviceGraphExecutor::GraphSegmentCache::ExecutableSubmissionState::
                                    MaterializedUnlaunched,
                        "one original materialized-unlaunched source executable");
        captured_ = cache_.segments.front().capture.get();
        captured_node_count_ = captured_->nodeCount();
    }
    /** @brief Abort only failure cleanup, drain accepted work before releasing
     * any mapped/root owner. */
    ~Source()
    {
        if (pending_)
        {
            (void)canonical_->publishAbort();
            auto &control = transport_.activationEpochControl(remote_, 0);
            MoEOverlayActivationEpochProtocol protocol(
                control, transport_.activationEpochConfig(remote_, 0));
            (void)protocol.cancelOriginalInvocation(protocol.activeIdentity(),
                                                    MoEOverlayActivationStatusCode::ExplicitAbort);
        }
        // Reset uploads can exist before any graph is submitted. Retain their
        // immutable staging owner until this exact stream is drained as well.
        if (backend_ && stream_)
            backend_->synchronizeStream(stream_, device_.ordinal);
        cache_.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Destroy);
        graph_.reset();
        executor_.reset();
        context_.reset();
        arena_.reset();
        for (void *address : allocations_)
            backend_->free(address, device_.ordinal);
        if (terminal_)
            backend_->destroyEvent(terminal_, device_.ordinal);
        if (stream_)
            backend_->destroyStream(stream_, device_.ordinal);
    }
    /** @return Actual source-local publication consumed by the production host
     * dispatcher. */
    const auto &dispatchTicket() const noexcept { return dispatch_ticket_; }
    /** @brief Replay the same executable after original protocol admission. */
    void launch()
    {
        requireOriginal(!pending_, "one original source submission at a time");
        requireOriginal(dispatch_ticket_->armCapturedPublication(),
                        "original source-local publication reset");
        // Own failure cleanup before submission: an exception may follow a
        // native enqueue, which still needs its original abort/drain boundary.
        pending_ = true;
        requireOriginal(
            submit(DeviceGraphExecutor::GraphInitialSubmissionPolicy::CaptureInstantiateAndLaunch),
            "source retained launch");
        requireOriginal(addresses() == bound_addresses_, "launch preserves original GPU operand bindings");
        requireOriginal(cache_.segments.size() == 1 &&
                            cache_.segments.front().capture.get() == captured_ &&
                            captured_->hasExecutable() && captured_->nodeCount() == captured_node_count_,
                        "same captured source executable");
        requireOriginal(backend_->recordEvent(terminal_, device_.ordinal, stream_),
                        "source terminal event");
    }
    /** @brief Observe only terminal completion; never use a host live-count
     * shadow. */
    void await()
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        bool ready = false;
        while (!ready && std::chrono::steady_clock::now() < deadline)
        {
            requireOriginal(backend_->queryEvent(terminal_, device_.ordinal, &ready),
                            "source terminal query");
            std::this_thread::yield();
        }
        requireOriginal(ready, "source actual round trip deadline");
        pending_ = false;
    }
    /**
     * @brief Publish recorded routes and reset GPU output before the next admission.
     * @param live Expected cohort row count from the immutable recorded trace.
     *
     * Captured output storage is GPU-owned even before its first launch. A full
     * overwrite therefore uses the original GPU address and publishes the actual
     * producer event; it must not reacquire stale host tensor contents.
     */
    void prepareInputs(int live)
    {
        requireOriginal(!pending_ && captured_,
                        "replay reset requires a quiescent retained capture");
        requireOriginal(addresses() == bound_addresses_, "original GPU replay buffer addresses");
        const auto recorded = std::find_if(live_row_trace.begin(), live_row_trace.end(),
            [live](const auto& work) { return work.cohort_rows == live; });
        requireOriginal(recorded != live_row_trace.end(),
                        "cohort work missing from recorded trace");
        // Empty participants still belong to a nonempty model request. Route
        // that recorded request to the original GPU-owned experts instead of
        // inventing a zero-row dispatch ticket.
        const auto& routes = live == 0 ? continuation_routes : cohort_routes;
        requireOriginal(backend_->hostToDevice(indices_.gpu_data_ptr(), routes.data(),
                                               routes.size() * sizeof(float),
                                               device_.ordinal, stream_),
                        "ordered recorded expert-route publication");
        TransferEngine::publishDeviceWrite(&indices_, device_, stream_);
        requireOriginal(backend_->hostToDevice(output_.gpu_data_ptr(), output_poison_.data(),
                                               output_poison_.size() * sizeof(float),
                                               device_.ordinal, stream_),
                        "ordered overwrite of original GPU output storage");
        TransferEngine::publishDeviceWrite(&output_, device_, stream_);
        requireOriginal(
            backend_->hostToDevice(active_, &recorded->logical_rows, sizeof(int32_t),
                                   device_.ordinal, stream_),
            "ordered recorded active-count publication");
        requireOriginal(addresses() == bound_addresses_,
                        "replay reset must not rebind GPU storage");
    }
    /** @brief Check forwarding, original-slot placement and untouched tails against actual root
     * rows. This is not an independent FFN arithmetic oracle; that remains in its owning
     * full/fragment test.
     * @param live Recorded original logical prefix; no device state is inferred
     * from it.
     */
    void verifyOutput(int live)
    {
        std::vector<float> actual(slots * hidden), expected(slots * hidden, -991.f);
        requireOriginal(backend_->deviceToHost(actual.data(), output_.gpu_data_ptr(),
                                               actual.size() * sizeof(float), device_.ordinal,
                                               stream_),
                        "terminal route observation");
        requireOriginal(backend_->recordEvent(terminal_, device_.ordinal, stream_),
                        "terminal observation event");
        await();
        for (size_t entry = 0; entry < size_t(live * top_k); ++entry)
        {
            const auto original = canonical_->originalRouteSlotsHost()[entry];
            const auto compact = canonical_->compactRouteSlotsHost()[entry];
            EXPECT_GE(original, 0);
            EXPECT_LT(size_t(original), slots);
            EXPECT_GE(compact, 0);
            EXPECT_LT(size_t(compact), canonical_->routeCapacity());
            if (original < 0 || size_t(original) >= slots || compact < 0 ||
                size_t(compact) >= canonical_->routeCapacity())
                continue;
            std::copy_n(canonical_->contributionRowsHost() + size_t(compact) * hidden, hidden,
                        expected.data() + size_t(original) * hidden);
        }
        if (live > 0)
            EXPECT_TRUE(std::any_of(actual.begin(), actual.begin() + live * top_k * hidden,
                                    [](float value)
                                    { return std::isfinite(value) && value > 0.f; }));
        EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0)
            << "Actual captured root return changed live rows, original slots or "
               "untouched tails";
        if (live == live_row_trace.front().cohort_rows)
        {
            if (!first_nonempty_output_)
                first_nonempty_output_ = actual;
            else
                EXPECT_EQ(std::memcmp(actual.data(), first_nonempty_output_->data(),
                                      actual.size() * sizeof(float)), 0)
                    << "Same recorded nonempty work changed after a genuine success/reset";
        }
    }
    /** @brief Release the actual source bank reader only after every captured
     * transaction is terminal. */
    void finish()
    {
        requireOriginal(!pending_, "no live source graph at epoch retirement");
        requireOriginal(kernel_->releaseMoEOverlayEpoch({.stream = stream_}, epoch_control_,
                                                        epoch_ticket_, epoch_status_),
                        "actual source epoch release");
        observeEpoch(DeviceMoEOverlayEpochOperation::Release);
        DeviceMoEOverlayEpochControl retired{};
        requireOriginal(backend_->deviceToHost(&retired, epoch_control_, sizeof(retired),
                                               device_.ordinal, stream_) &&
                            backend_->recordEvent(terminal_, device_.ordinal, stream_),
                        "retired source reader observation");
        await();
        EXPECT_EQ(retired.bank_readers[0], 0u);
        EXPECT_EQ(retired.bank_readers[1], 0u);
    }

  private:
    /** @brief Initialize read-only operands once, before capture owns any GPU output. */
    void initializeInputs()
    {
        requireOriginal(!pending_ && !captured_, "cold operand initialization precedes capture");
        for (int row = 0; row < rows; ++row)
        {
            for (int column = 0; column < hidden; ++column)
                hidden_.mutable_data()[row * hidden + column] = 0.001f * float(1 + row + column);
            for (int slot = 0; slot < top_k; ++slot)
            {
                indices_.mutable_data()[row * top_k + slot] = float(2 + slot);
                weights_.mutable_data()[row * top_k + slot] = slot ? 0.75f : 0.25f;
            }
        }
        std::fill_n(output_.mutable_data(), output_.numel(), -991.f);
        requireOriginal(hidden_.ensureOnDevice(device_, stream_) &&
                            indices_.ensureOnDevice(device_, stream_) &&
                            weights_.ensureOnDevice(device_, stream_) &&
                            output_.ensureOnDevice(device_, stream_) &&
                            backend_->hostToDevice(active_, &live_row_trace.front().logical_rows, sizeof(int32_t),
                                                   device_.ordinal, stream_),
                        "recorded source inputs");
        bound_addresses_ = addresses();
        requireOriginal(std::all_of(bound_addresses_.begin(), bound_addresses_.end(),
                                    [](const void *address) { return address != nullptr; }),
                        "complete original GPU operand bindings");
    }
    /** @return Original storage identities only, never a shadow of live tensor values. */
    std::array<const void *, 5> addresses() const noexcept
    {
        return {hidden_.gpu_data_ptr(), indices_.gpu_data_ptr(), weights_.gpu_data_ptr(),
                output_.gpu_data_ptr(), active_};
    }
    /** @brief Validate a setup/retirement receipt, not a host substitute for an
     * acquired ticket.
     * @param operation Actual native operation submitted on this source stream.
     */
    void observeEpoch(DeviceMoEOverlayEpochOperation operation)
    {
        DeviceMoEOverlayEpochStatus observed{};
        requireOriginal(backend_->deviceToHost(&observed, epoch_status_, sizeof(observed),
                                               device_.ordinal, stream_) &&
                            backend_->recordEvent(terminal_, device_.ordinal, stream_),
                        "native source epoch receipt");
        await();
        requireOriginal(observed.succeeded() && observed.operation == uint32_t(operation) &&
                            observed.epoch == 1 && observed.bank == 0,
                        "native source epoch semantic receipt");
    }
    /** @brief Resolve only the actual remote original member.
     * @param owners Same authoritative original source map.
     * @return Actual rank1 CPU member, never a fabricated source-local channel. */
    int remoteParticipant(const MoEExpertOwnerMap &owners)
    {
        for (const auto &p : owners.participants())
            if (p.device.is_cpu() && p.world_rank == 1)
                return remote_ = p.participant_id;
        throw std::logic_error("Original source map has no remote CPU member");
    }
    /** @return Original encoded word/epoch addresses in one real physical bank.
     */
    MoEOverlayRoutePlacementBankDeviceView bankView(unsigned bank) const
    {
        auto *base = reinterpret_cast<std::byte *>(banks_) + bank * sizeof(DeviceMoEPlacementBank);
        return {.route_participants = reinterpret_cast<const int32_t *>(
                    base + offsetof(DeviceMoEPlacementBank, overlay_route_participant)),
                .epoch = reinterpret_cast<const uint32_t *>(
                    base + offsetof(DeviceMoEPlacementBank, epoch))};
    }
    /** @brief Own one native replay operand at the existing low-level test boundary.
     * @param bytes Exact operand byte extent.
     * @return Backend-owned address retained until actual stream retirement. */
    void *allocate(size_t bytes)
    {
        auto *address = backend_->allocate(bytes, device_.ordinal);
        requireOriginal(address != nullptr, "source input allocation");
        allocations_.push_back(address);
        return address;
    }
    /** @brief Submit the retained original source graph through its actual executor.
     * @param initial Original materialize/launch selection.
     * @return Actual executor outcome with no force-recapture. */
    bool submit(DeviceGraphExecutor::GraphInitialSubmissionPolicy initial)
    {
        return executor_->executeWithCachedGraphReplay(
            *graph_, context_.get(), cache_, stream_, worker_, nullptr, false, false, true, {},
            DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph, {}, {}, {}, initial);
    }
    DeviceId device_;
    MoEOverlayNodeLocalRankBatchTransport &transport_;
    std::shared_ptr<MoEOverlayCanonicalRouteReturnTicketStorage> canonical_;
    std::shared_ptr<MoEOverlayDispatchTicketStorage> dispatch_ticket_;
    IBackend *backend_ = nullptr;
    IWorkerGPUContext *worker_ = nullptr;
    std::unique_ptr<IMoEKernel> kernel_;
    void *stream_ = nullptr;
    void *terminal_ = nullptr;
    std::vector<void *> allocations_;
    DeviceMoEPlacementBank *banks_ = nullptr;
    std::unique_ptr<DeviceMoEPlacementBank[]> initial_banks_;
    DeviceMoEOverlayEpochControl initial_control_{};
    DeviceMoEOverlayEpochControl *epoch_control_ = nullptr;
    DeviceMoEOverlayEpochTicket *epoch_ticket_ = nullptr;
    DeviceMoEOverlayEpochStatus *epoch_status_ = nullptr;
    int32_t *active_ = nullptr;
    int remote_ = -1;
    bool pending_ = false;
    FP32Tensor hidden_, indices_, weights_, output_;
    const std::vector<float> output_poison_; ///< Cold immutable reset input, retained through every upload.
    std::array<const void*, 5> bound_addresses_{}; ///< Immutable capture identities, not data mirrors.
    std::unique_ptr<BufferArena> arena_;
    std::unique_ptr<IDeviceContext> context_;
    std::unique_ptr<ComputeGraph> graph_;
    std::unique_ptr<DeviceGraphExecutor> executor_;
    DeviceGraphExecutor::GraphSegmentCache cache_;
    const IGPUGraphCapture *captured_ = nullptr;
    size_t captured_node_count_ = 0;
    std::optional<std::vector<float>> first_nonempty_output_; ///< Terminal replay oracle only; never feeds execution.
};

/** @brief Replay prepared two-socket work through one retained executable and original resets.
 * @param device Original rank0 continuation; the other rank never requests a
 * GPU backend.
 */
inline void runOriginal(DeviceId device)
{
    PublicationEvidence evidence;
    int rank = -1, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    requireOriginal(size == 2, "exactly one original rank per socket");
    const auto world = MPIContextFactory::global();
    const auto inventory = world->clusterInventory();
    requireOriginal(inventory && inventory->ranks.size() == 2, "original two-socket inventory");
    const auto affinity = NUMATopology::detectLocalNUMANode();
    requireOriginal(affinity.detection_succeeded && affinity.local_numa_node >= 0,
                    "original source NUMA detection");
    const auto topology = NodeTopology::detect();
    const auto node =
        std::find_if(topology.numaNodes().begin(), topology.numaNodes().end(),
                     [&](const auto &item) { return item.node_id == affinity.local_numa_node; });
    requireOriginal(node != topology.numaNodes().end() && node->socket_id >= 0,
                    "original source socket");
    std::array<int, 2> nodes{}, sockets{};
    requireOriginal(MPI_Allgather(&affinity.local_numa_node, 1, MPI_INT, nodes.data(), 1, MPI_INT,
                                  MPI_COMM_WORLD) == MPI_SUCCESS &&
                        MPI_Allgather(&node->socket_id, 1, MPI_INT, sockets.data(), 1, MPI_INT,
                                      MPI_COMM_WORLD) == MPI_SUCCESS,
                    "original socket membership exchange");
    std::vector<NativeCPUCollectiveProgram::Participant> members;
    std::vector<llaminar2::test::CPUExpertSocketEndpoint> endpoints;
    for (int member = 0; member < 2; ++member)
    {
        endpoints.push_back({member, nodes[member], sockets[member]});
        members.push_back(
            {member, GlobalDeviceAddress::cpu(nodes[member], inventory->ranks[member].hostname)});
    }
    requireOriginal(llaminar2::test::validCPUExpertSocketMembership(endpoints, 2),
                    "original distinct physical socket endpoints");
    for (int member = 0; member < 2; ++member)
    {
        requireOriginal(inventory->ranks[member].rank == member &&
                            inventory->ranks[member].hostname == inventory->ranks[0].hostname,
                        "same original node/rank inventory");
        requireOriginal(inventory->ranks[member].cpu.numa_node < 0 ||
                            inventory->ranks[member].cpu.numa_node == nodes[member],
                        "original inventory/observed NUMA identity");
    }
    initCPUBackend(nodes[rank]);
    llaminar2::test::PlanningGGUFFixture file(
        true, false, GGUFTensorType::F32, intermediate, std::nullopt, experts,
        llaminar2::test::PlanningGGUFFixture::Geometry::Directory, 1);
    llaminar2::test::populateServingGGUFFixture(file);
    TensorFactory factory(*world);
    ModelLoader loader(&factory);
    requireOriginal(loader.loadModel(file.path()), "actual native source directory");
    auto plan = llaminar2::test::cpuJointSourceTestPlan();
    plan.first_model_layer = 0;
    plan.domains.front().participants = {device.is_cuda()
                                             ? GlobalDeviceAddress::cuda(device.ordinal)
                                             : GlobalDeviceAddress::rocm(device.ordinal)};
    plan.domains.front().backend =
        device.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
    plan.domains.back().participants = {members[0].address, members[1].address};
    plan.domains.back().world_ranks = {0, 1};
    plan.routed_tiers[0].max_experts_per_layer = 2;
    plan.routed_tiers[1].max_experts_per_layer = 6;
    plan.placements = {{.layer = 0, .routed_expert_tier = {0, 0, 1, 1, 1, 1, 1, 1}}};
    plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
    plan.authority_execution = resolveMoEOverlayAuthorityExecutionKind(plan);
    const auto runtime = resolveMoEExpertOverlayRuntimePlan(
        std::make_shared<const MoERoutedExpertPlacementPlan>(plan),
        {.current_world_rank = rank, .validate_mvp_root_reachability = false});
    requireOriginal(bool(runtime), "original resolved CPU/GPU source plan");
    const auto preparation = std::make_shared<const MoEExpertOverlayPreparationPlan>(
        MoEExpertOverlayPreparationPlan::build(*runtime, loader));
    const auto local = preparation->filteredForDevice(DeviceId::cpu());
    requireOriginal(!local.requests().empty(), "original source member requests");
    const auto selected = local.requests().front().cpu_joint_source;
    requireOriginal(selected && selected->participant().world_rank == rank &&
                        selected->participant().address.numa_node == nodes[rank],
                    "same original source member placement");
    requireOriginal(local.requests().size() == selected->expertIds().size() * 3,
                    "complete original local expert triplet inventory");
    for (const auto &request : local.requests())
        requireOriginal(request.cpu_joint_source == selected &&
                            request.participant_world_rank == rank,
                        "same original local source selection for every prepared role");
    const auto owners = preparation->cpuJointOwnerMap();
    requireOriginal(bool(owners), "original fixed cohort map");
    const auto local_id = selected->participant().participant_id;
    int remote_id = -1;
    for (const auto &p : owners->participants())
        if (p.world_rank == 1 && p.device.is_cpu())
            remote_id = p.participant_id;
    requireOriginal(remote_id >= 0, "actual remote cohort participant");
    const auto manifest = selected->cohort()->localManifest(selected->participantIndex());
    const size_t prepared_bytes =
        6 * MoEOverlayCapacityResolver::preparedFootprints({manifest}).front().cpu_live_bytes;
    CPUJointExpertRouteStorageLayout joint(selected->cohort(), selected->participantIndex(), rows,
                                           top_k);
    const bool is_root = joint.isRoot();
    requireOriginal(is_root == (rank == 0), "original CPU root/continuation topology");
    const auto consumer = is_root ? std::optional<DeviceId>{device} : std::nullopt;
    const auto layout = std::make_shared<const CPUSerialExpertStorageLayout>(
        DeviceId::cpu(), rows, hidden, top_k, std::vector<size_t>{rows}, is_root, intermediate / 2,
        experts, consumer, joint);
    // Serial grouped roots and named projection/SwiGLU scratch coexist.
    // Price the latter using the real stage's source-only metadata declaration,
    // which creates no executor, prepared engine or physical grouped owner.
    const auto workspace_source =
        MoELocalExpertStage::CPUExpertWorkspaceSource{.gate = TensorType::FP32,
                                                      .up = TensorType::FP32,
                                                      .down = TensorType::FP32,
                                                      .workers = 1,
                                                      .execution = CPUExecutionGeometry::local()};
    MoELocalExpertStage::Params source_declaration;
    source_declaration.device_id = DeviceId::cpu();
    source_declaration.num_experts = experts;
    source_declaration.top_k = top_k;
    source_declaration.d_model = hidden;
    source_declaration.expert_intermediate = joint.intermediate();
    source_declaration.graph_row_capacity = rows;
    source_declaration.cpu_workspace_source = workspace_source;
    source_declaration.expert_weight_resolution_policy =
        MoELocalExpertStage::ExpertWeightResolutionPolicy::RegistryOnly;
    MoELocalExpertStage declared_stage(source_declaration);
    const auto named_requirements = declared_stage.getWorkspaceRequirements(rows);
    requireOriginal(!named_requirements.cpu_grouped_moe_storage &&
                        !named_requirements.cpu_router_q8_storage,
                    "source-only declaration must not duplicate retained grouped roots");
    const auto named_workspace_bytes = named_requirements.total_bytes_with_alignment();
    const auto program = NativeCPUCollectiveProgram::expertShardRoutePartials(
        1, CollectiveBackendType::UPI, members);
    const auto capacity = observeSystemMemoryCapacity();
    const PhysicalMemoryResource host{rank, DeviceId::cpu(), capacity.total_bytes,
                                      capacity.admission_available_bytes};
    PhysicalMemoryPlanBuilder bill;
    program.contributeDomainBOM(bill, host, rank);
    NativeCPUCollectiveProgram::contributeProcessPoolBOM(bill, host);
    if (is_root)
        bill.add(host, PhysicalMemoryOwner::ActivationTransportStaging,
                 ShmemSpinArena::allocationGeometry(2).mapped_bytes);
    bill.add(host, PhysicalMemoryOwner::RoutedExpertWeights, prepared_bytes)
        .add(host, PhysicalMemoryOwner::ActivationArena, layout->physicalBytes())
        .add(host, PhysicalMemoryOwner::ExecutionWorkspace, layout->workspaceBytes())
        .add(host, PhysicalMemoryOwner::ExecutionWorkspace, named_workspace_bytes);
    const auto memory = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(bill.build()), rank);
    WeightManager manager(loader, world);
    manager.installPhysicalMemoryAuthority(memory);
    const ModelContextId model{47367};
    const auto prepared = manager.materialize(local.sourceWeightPlan(loader, model));
    requireOriginal(!prepared.bindings().empty(), "actual original prepared fragment weights");
    auto residency = std::make_shared<MoEOverlayParticipantResidencyRegistry>(
        MoEOverlayParticipantResidencyRegistry::Config{.local_participant_ids = {local_id},
                                                       .num_layers = 1,
                                                       .num_experts = experts,
                                                       .initial_epoch = 1,
                                                       .projection_preparation = preparation,
                                                       .first_model_layer = 0,
                                                       .original_model = model});
    std::string error;
    requireOriginal(residency->registerInitialCPUJointBank(manager.expertGemmRegistry(), model,
                                                           local_id, &error),
                    error.c_str());
    const auto layer = residency->initialCPUJointLayer(local_id, 0);
    requireOriginal(bool(layer), "actual cold fragment bank");
    auto arena = std::make_shared<
        MoELocalExpertSerialBufferArena>(MoELocalExpertSerialBufferArena::Config{
        .device_id = DeviceId::cpu(),
        .row_capacity = rows,
        .row_capacity_buckets = {rows},
        .d_model = hidden,
        .routing_top_k = top_k,
        .cpu_canonical_route_storage =
            is_root ? MoELocalExpertSerialBufferArena::CPUCanonicalRouteStoragePolicy::
                          RetainSerialMaximum
                    : MoELocalExpertSerialBufferArena::CPUCanonicalRouteStoragePolicy::Disabled,
        .cpu_canonical_route_gpu_consumer = consumer,
        .cpu_grouped_scratch_storage =
            MoELocalExpertSerialBufferArena::CPUGroupedScratchStoragePolicy::RetainSerialMaximum,
        .num_experts = experts,
        .expert_intermediate = intermediate / 2,
        .logical_participant_id = local_id,
        .debug_name = "native.rooted_packet_repeat",
        .cpu_storage_layout = layout,
        .physical_memory_authority = memory});
    requireOriginal(bool(arena->cpuCanonicalRoutes()) == is_root,
                    "canonical storage belongs only to the original root");
    requireOriginal(is_root || (arena->cpuJointSourceRanges().empty() &&
                                arena->cpuJointRouteMaps().empty() &&
                                arena->cpuJointRequiredRoutes().empty()),
                    "contributors retain no original root reduction scratch");
    std::variant<NativeCPUCollectiveProgram::DomainUse, NativeCPUCollectiveProgram::ProcessUse>
        use = decltype(use)(std::in_place_type<NativeCPUCollectiveProgram::DomainUse>,
                            program.admit(memory, rank));
    auto domain =
        GlobalTPContext::createAdmittedCPUExpertDomain(MPI_COMM_WORLD, program, std::move(use));
    requireOriginal(domain && domain->authenticatesOriginalCPUExpertBorrow(domain),
                    "genuine admitted CPU domain");
    // Catch while the original admitted domain is still alive. Its destructor
    // deliberately terminates an unretired owner, which would hide the initial
    // setup exception if the only catch were outside this function.
    try
    {
    std::shared_ptr<MoEOverlayCanonicalRouteReturnTicketStorage> canonical;
    if (is_root)
    {
        auto metadata = TransferEngine::instance().createMappedHostArena(std::array{device});
        canonical = std::make_shared<MoEOverlayCanonicalRouteReturnTicketStorage>();
        canonical->bindFixedCapacity(0, slots, hidden, device, workspace_generation,
                                     arena->mappedCPUCanonicalRoutes(device), metadata);
    }
    int channel_pid = rank == 0 ? int(::getpid()) : 0;
    requireOriginal(MPI_Bcast(&channel_pid, 1, MPI_INT, 0, MPI_COMM_WORLD) == MPI_SUCCESS,
                    "original channel identity publication");
    auto wire = std::make_shared<MoEOverlayRankBatchWireWorkspace>(
        MoEOverlayRankBatchWireWorkspace::Config{.participant_ids = {remote_id},
                                                 .max_total_rows = rows,
                                                 .max_total_entries = slots,
                                                 .d_model = hidden,
                                                 .top_k = top_k});
    MoEOverlayRankBatchTransportConfig config{
        .mpi_ctx = world,
        .source_world_rank = 0,
        .target_world_rank = 1,
        .workspace = wire,
        .max_rows_per_participant = rows,
        .max_entries_per_participant = slots,
        .d_model = hidden,
        .top_k = top_k,
        .tier_index = 1,
        .domain_ordinal = 1,
        .channel_identity = "cpu_joint_packet_" + std::to_string(channel_pid),
        .transaction_slot_count = 8,
        .transaction_topology = {.workspace_generation = workspace_generation,
                                 .topology_fingerprint_low = 367,
                                 .topology_fingerprint_high = 736,
                                 .source_world_rank = 0,
                                 .target_world_rank = 1},
        .source_endpoint = {.world_rank = 0,
                            .participant_id = 0,
                            .tier_priority = 0,
                            .domain_ordinal = 0},
        .target_tier_priority = 1,
        .activation_graph_families = {{.graph_role_mask =
                                           uint32_t{1}
                                           << uint32_t(MoEOverlayInferenceGraphRole::MainPrefill),
                                       .model_layer_indices = {0}}},
        .activation_layout =
            planMoEOverlayNodeLocalActivationLayout({.participant_count = 1,
                                                     .max_rows_per_participant = rows,
                                                     .max_entries_per_participant = slots,
                                                     .d_model = hidden,
                                                     .activation_graph_family_count = 1}),
        .local_lanes = {{.participant_id = remote_id,
                         .device = rank == 0 ? device : DeviceId::cpu()}},
        .original_cpu_joint_owners = owners};
    MoEOverlayNodeLocalRankBatchTransport transport(config);
    auto &control = transport.activationEpochControl(remote_id, 0);
    const auto protocol_config = transport.activationEpochConfig(remote_id, 0);
    MoEOverlayActivationEpochProtocol protocol(control, protocol_config);
    std::unique_ptr<Source> source;
    if (rank == 0)
        source = std::make_unique<Source>(device, transport, *owners, canonical);
    MoEOverlayCollectiveWorkspace packets(
        {.max_rows = rows,
         .max_entries = slots,
         .d_model = hidden,
         .top_k = top_k,
         .device = DeviceId::cpu(),
         .return_layout = MoEOverlayReturnLayout::CanonicalExpertRoutes});
    auto input = rank == 0 ? packets.localExpertInput(0, 1)
                           : transport.activationDeviceLane(remote_id, 0, DeviceId::cpu())
                                 .hostDispatchPayload(rows);
    auto output =
        rank == 0 ? packets.localExpertOutput(0, 1) : transport.sharedReturnRows(remote_id);
    auto local_context = IDeviceContext::create(DeviceId::cpu());
    auto local_collective = std::make_shared<MoEOverlayRankLocalSparseCollectiveContext>(
        MoEOverlayRankLocalSparseCollectiveContext::Config{8});
    auto authority =
        std::make_shared<MoEOverlayResidencyAuthority>(MoEOverlayResidencyAuthority::Config{
            .initial_plan = plan,
            .model_metadata = {.num_layers = 1,
                               .num_experts = experts,
                               .d_model = hidden,
                               .routed_intermediate_size = intermediate,
                               .main_inference_layer_count = 1,
                               .first_model_layer = 0},
            .projection_preparation = preparation,
            .maintenance_mode = MoERebalanceRuntimeMode::Off,
            .participant_rebalance_policy = {.enabled = false}});
    auto dispatched = std::make_shared<MoEExpertDispatchOutput>();
    std::unique_ptr<MoEExpertDispatchStage> host_dispatch;
    std::unique_ptr<MoESparseDispatchStage> local_dispatch;
    if (rank == 0)
    {
        MoEExpertDispatchStage::Params p;
        p.device_id = DeviceId::cpu();
        p.ticket_storage = source->dispatchTicket();
        p.seq_len = rows;
        p.top_k = top_k;
        p.d_model = hidden;
        p.continuation_domain = plan.continuation_domain;
        p.placement = plan.placements.front();
        p.routed_tiers = plan.routed_tiers;
        p.residency_authority = authority;
        p.output = dispatched.get();
        host_dispatch = std::make_unique<MoEExpertDispatchStage>(p);
        MoESparseDispatchStage::Params sparse;
        sparse.device_id = DeviceId::cpu();
        sparse.collective_context_lifetime = local_collective;
        sparse.workspace = &packets;
        sparse.key = {.generation_id = 1,
                      .step_id = 1,
                      .layer_idx = 0,
                      .tier_idx = 1,
                      .domain_id = 1,
                      .participant_id = local_id,
                      .direction = MoEOverlayCollectiveDirection::Dispatch};
        sparse.source_participant = 0;
        sparse.target_participant = local_id;
        sparse.ticket_storage = source->dispatchTicket();
        sparse.seq_len = rows;
        sparse.top_k = top_k;
        sparse.d_model = hidden;
        sparse.dispatch_output = dispatched.get();
        sparse.tier_index = 1;
        sparse.inbound_rows = &input;
        sparse.require_explicit_transaction_identity = true;
        sparse.require_explicit_execution_semantics = true;
        local_dispatch = std::make_unique<MoESparseDispatchStage>(sparse);
    }
    MoELocalExpertStage::Params local_params;
    local_params.device_id = DeviceId::cpu();
    local_params.input_rows = &input;
    local_params.output_rows = &output;
    local_params.serial_compact_buffer_arena = arena;
    local_params.graph_row_capacity = rows;
    local_params.num_experts = experts;
    local_params.top_k = top_k;
    local_params.d_model = hidden;
    local_params.expert_intermediate = intermediate / 2;
    local_params.layer_idx = 0;
    local_params.runtime_participant_index = local_id;
    local_params.cpu_joint_layer = layer;
    local_params.expert_registry = &manager.expertGemmRegistry();
    local_params.physical_memory_authority = memory;
    local_params.return_layout = MoEOverlayReturnLayout::ParticipantTokenPartials;
    local_params.expert_weight_resolution_policy =
        MoELocalExpertStage::ExpertWeightResolutionPolicy::RegistryOnly;
    local_params.cpu_workspace_source = workspace_source;
    local_params.cpu_joint_route_producer = MoELocalExpertStage::CPUJointRouteProducerBinding{
        .packed_rows = arena->cpuJointPackedRecords(),
        .original_route_slots = input.original_route_slots_host,
        .compact_route_slots = input.compact_route_slots_host,
        .route_slot_capacity = slots,
        .deferred_canonical_ticket = canonical};
    local_params.expert_mask.assign(experts, false);
    local_params.prepared_gate_gemm.assign(experts, nullptr);
    local_params.prepared_up_gemm.assign(experts, nullptr);
    local_params.prepared_down_gemm.assign(experts, nullptr);
    {
        const auto initial = layer->acquireInitialConstruction();
        requireOriginal(bool(initial), "actual initial bank borrow");
        for (int expert : selected->expertIds())
        {
            const auto &e = initial->layerForModelLayer(0).cpu_fragments[expert];
            local_params.expert_mask[expert] = true;
            local_params.prepared_gate_gemm[expert] = e.gate().get();
            local_params.prepared_up_gemm[expert] = e.up().get();
            local_params.prepared_down_gemm[expert] = e.down().get();
        }
    } // Cold construction borrow ends before any invocation.
    MoELocalExpertStage compute(local_params);
    const auto actual_requirements = compute.getWorkspaceRequirements(rows);
    requireOriginal(!actual_requirements.cpu_grouped_moe_storage &&
                        !actual_requirements.cpu_router_q8_storage &&
                        actual_requirements.total_bytes_with_alignment() == named_workspace_bytes,
                    "actual retained stage must match its source-only scratch quote");
    DeviceWorkspaceManager workspace(DeviceId::cpu(), named_workspace_bytes, memory);
    requireOriginal(workspace.allocate(actual_requirements),
                    "same-PMA declared CPU execution workspace");
    compute.bindWorkspace(&workspace);
    MoECPUExpertShardRouteStage::Params reduction{
        .original_source = selected,
        .tp = domain.get(),
        .packed_records = arena->cpuJointPackedRecords().get(),
        .combined_routes = is_root ? arena->cpuCanonicalRoutes().get() : nullptr,
        .source_ranges =
            is_root ? arena->cpuJointSourceRanges() : std::span<RootedFloatRecordRange>{},
        .participant_route_maps = is_root ? arena->cpuJointRouteMaps() : std::span<int32_t>{},
        .required_route_weights = is_root ? arena->cpuJointRequiredRoutes() : std::span<float>{},
        .rows = rows,
        .top_k = top_k,
        .columns = hidden,
        .root_participant = static_cast<int>(joint.rootMember()),
        .stage_name = "native.rooted_packet_repeat",
        .original_return_rows = &output,
        .original_return_route_capacity = slots,
        .original_sparse_routes = &input,
        .original_prepared_layer = layer,
        .original_tp_owner = domain,
        .original_canonical_ticket = canonical,
        .original_continuation_device = is_root ? device : DeviceId::invalid()};
    auto reduce = std::make_unique<MoECPUExpertShardRouteStage>(reduction);
    std::unique_ptr<MoESparseReturnReduceStage> local_return;
    auto inbound_return = packets.returnReceive(0, 1);
    if (rank == 0)
    {
        MoESparseReturnReduceStage::Params returned;
        returned.device_id = DeviceId::cpu();
        returned.collective_context_lifetime = local_collective;
        returned.key = {.generation_id = 1,
                        .step_id = 1,
                        .layer_idx = 0,
                        .tier_idx = 1,
                        .domain_id = 1,
                        .participant_id = local_id,
                        .direction = MoEOverlayCollectiveDirection::ReturnReduce};
        returned.source_participant = local_id;
        returned.target_participant = 0;
        returned.outbound_rows = &output;
        returned.inbound_rows = &inbound_return;
        // Match the continuation's certificate-only host edge: the actual
        // canonical ticket owns GPU ingress, so this edge must not gather it again.
        returned.return_layout = MoEOverlayReturnLayout::ParticipantTokenPartials;
        returned.inbound_consumer_role =
            MoESparseReturnReduceStage::InboundConsumerRole::CanonicalRouteTicketCompletion;
        returned.canonical_route_ticket_storage = canonical;
        returned.seq_len = rows;
        returned.d_model = hidden;
        returned.require_explicit_transaction_identity = true;
        returned.require_explicit_execution_semantics = true;
        returned.dispatch_output_lifetime = dispatched;
        returned.residency_lease_terminal = MoEOverlayHostDispatchLeaseTerminal::Release;
        local_return = std::make_unique<MoESparseReturnReduceStage>(returned);
    }
    for (uint64_t generation = 1; generation <= invocation_count; ++generation)
    {
        const auto& recorded = live_row_trace[(generation - 1) % live_row_trace.size()];
        const int live = recorded.cohort_rows;
        SCOPED_TRACE("generation=" + std::to_string(generation) + " live=" + std::to_string(live));
        if (rank == 0)
        {
            // Replay recorded cohort work inside the same valid request/capture
            // envelope. Empty CPU participants retain positive logical rows.
            source->prepareInputs(live);
            const auto ticket = makeMoEOverlayInferenceExecutionTicket(
                config.transaction_topology,
                {.request_generation = generation,
                 .command_id = generation,
                 .initial_placement_epoch = 1},
                generation, generation, 1, MoEOverlayInferenceGraphRole::MainPrefill, 1, rows,
                rows);
            requireOriginal(bool(protocol.arm(ticket, generation, 30'000'000'000ull, &error)),
                            error.c_str());
            source->launch();
        }
        requireOriginal(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS, "original invocation boundary");
        const auto identity = protocol.activeIdentity();
        if (rank == 0)
        {
            requireOriginal(host_dispatch->execute(local_context.get()),
                            "actual captured source-local dispatch ticket");
            local_dispatch->updateMoEOverlayCollectiveRuntimeParams(
                {.generation_id = generation,
                 .step_id = generation,
                 .execution_semantics = IComputeStage::MoEOverlayCollectiveRuntimeParams::
                     ExecutionSemantics::Prefill});
            requireOriginal(local_dispatch->execute(local_context.get()),
                            "actual source-local sparse dispatch");
        }
        else
        {
            // Match the original CPU service: activate once, wait for the
            // producer timeline, then consume exactly once. Repeated semantic
            // consumption would hide a rejected endpoint behind a timeout.
            requireOriginal(protocol.activate(MoEOverlayActivationEndpoint::Follower, identity,
                                               &error), error.c_str());
            const auto bank = moeOverlayActivationBufferIndex(0);
            const auto expected = moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(0));
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            uint64_t observed = protocol.dispatchTimeline(bank);
            while (observed < expected && std::chrono::steady_clock::now() < deadline)
            {
                std::this_thread::yield();
                observed = protocol.dispatchTimeline(bank);
            }
            const std::string timeline_error = "actual captured remote dispatch timeline: observed=" +
                std::to_string(observed) + " expected=" + std::to_string(expected);
            requireOriginal(observed == expected, timeline_error.c_str());
            const auto descriptor = protocol.consumeDispatch(identity, 0, &error);
            requireOriginal(bool(descriptor), error.c_str());
            input.live_row_count = descriptor->live_rows;
            input.live_entry_count = descriptor->live_entries;
            input.residency_epoch = descriptor->placement_epoch;
            input.source_participant = 0;
            input.target_participant = local_id;
            input.key = {.generation_id = generation,
                         .step_id = generation,
                         .histogram_source = ExpertHistogramSource::PrefillChunk,
                         .layer_idx = 0,
                         .tier_idx = 1,
                         .domain_id = 1,
                         .participant_id = local_id,
                         .direction = MoEOverlayCollectiveDirection::Dispatch};
        }
        EXPECT_EQ(input.live_row_count, size_t(live));
        EXPECT_EQ(input.live_entry_count, size_t(live * top_k));
        auto& packed = *arena->cpuJointPackedRecords();
        const float unused_record = -1000.0f - static_cast<float>(generation);
        std::fill_n(packed.mutable_data(), packed.numel(), unused_record);
        requireOriginal(compute.execute(local_context.get()), "actual prepared CPU shard compute");
        EXPECT_TRUE(std::all_of(packed.data() + size_t(live * top_k) * (hidden + 1),
                               packed.data() + packed.numel() - 1,
                               [unused_record](float value) { return value == unused_record; }))
            << "CPU producer touched inactive packed capacity";
        EXPECT_EQ(
            canonical_moe_route_record::readRecordCount(arena->cpuJointPackedRecords()->data(),
                                                        arena->cpuJointPackedRecords()->numel()),
            size_t(live * top_k));
        const double gathered_bytes_before = evidence.value("cpu_packed_route_gather_local_bytes");
        const double gathered_calls_before = evidence.value("cpu_packed_route_gather_calls");
        if (rank == 0)
            requireOriginal(reduce->execute(local_context.get()),
                            "actual source-local rooted reduction/publication");
        else
        {
            auto completion =
                reduce->executeOriginalJointPublication(local_context.get(), protocol, identity, 0);
            requireOriginal(bool(completion), "actual remote rooted completion");
            requireOriginal(protocol.publishJointReturn(identity, 0, 0, *completion, &error),
                            error.c_str());
            requireOriginal(
                protocol.complete(MoEOverlayActivationEndpoint::Follower, identity, &error),
                error.c_str());
        }
        const size_t gathered_prefix = size_t(live * top_k) * (is_root ? members.size() : 1);
        EXPECT_TRUE(std::all_of(packed.data() + gathered_prefix * (hidden + 1),
                               packed.data() + packed.numel() - 1,
                               [unused_record](float value) { return value == unused_record; }))
            << "Native gather touched inactive packed capacity";
        EXPECT_EQ(evidence.value("cpu_packed_route_gather_calls") - gathered_calls_before, 1.0);
        EXPECT_EQ(evidence.value("cpu_packed_route_gather_local_bytes") - gathered_bytes_before,
                  double(size_t(live * top_k) * (hidden + 1) * sizeof(float)));
        if (rank == 0)
        {
            local_return->updateMoEOverlayCollectiveRuntimeParams(
                {.generation_id = generation,
                 .step_id = generation,
                 .execution_semantics = IComputeStage::MoEOverlayCollectiveRuntimeParams::
                     ExecutionSemantics::Prefill});
            requireOriginal(local_return->execute(local_context.get()),
                            "actual original final source-local return edge");
            EXPECT_FALSE(bool(dispatched->residency_lease));
            const auto ranges = arena->cpuJointSourceRanges();
            requireOriginal(validRootedFloatRecordRanges(ranges, size_t(2 * live * top_k)),
                            "actual native root source range extents");
            EXPECT_EQ(ranges[0].record_count, size_t(live * top_k));
            EXPECT_EQ(ranges[1].record_count, size_t(live * top_k));
            EXPECT_EQ(ranges[0].first_record, 0u);
            EXPECT_EQ(ranges[1].first_record, size_t(live * top_k));
            EXPECT_EQ(ranges[0].record_count + ranges[1].record_count, size_t(2 * live * top_k));
            source->await();
            source->verifyOutput(live);
            const auto traffic = protocol.completedTraffic(identity, &error);
            requireOriginal(bool(traffic), error.c_str());
            EXPECT_EQ(traffic->dispatch_live_rows, uint64_t(live));
            EXPECT_EQ(traffic->dispatch_live_entries, uint64_t(live * top_k));
            EXPECT_EQ(traffic->dispatch_payload_bytes,
                      moeOverlayDispatchPayloadBytes(live, live * top_k, hidden));
            EXPECT_EQ(traffic->dispatch_shared_hidden_bytes,
                      uint64_t(live) * hidden * sizeof(float));
            EXPECT_EQ(traffic->return_payload_bytes, 0u);
            EXPECT_EQ(traffic->return_live_rows, 0u);
            EXPECT_EQ(traffic->return_live_entries, 0u);
            EXPECT_EQ(traffic->dispatch_stage_count, 1u);
            EXPECT_EQ(traffic->return_stage_count, 1u);
        }
        requireOriginal(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS, "all actual returns complete");
        if (rank == 0)
            requireOriginal(protocol.reset(identity, &error), error.c_str());
        requireOriginal(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS,
                        "original success reset visible");
        for (unsigned bank = 0; bank < kMoEOverlayActivationBufferCount; ++bank)
        {
            EXPECT_EQ(protocol.dispatchTimeline(bank), 0u);
            EXPECT_EQ(protocol.returnTimeline(bank), 0u);
        }
    }
    compute.unbindWorkspace();
    if (source)
        source->finish();
    source.reset();
    // Destroy every graph/stage borrower before the original native domain
    // retires.
    reduce.reset();
    reduction.original_tp_owner.reset();
    requireOriginal(domain->retireOriginalCPUExpertDomain(),
                    "actual native domain retirement after all stage borrowers");
    EXPECT_EQ(
        memory->reservedBytes(DeviceId::cpu(), PhysicalMemoryOwner::NativeCPUCollectiveDomain), 0u);
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
                                   PhysicalMemoryMaterializationKind::NewAllocation),
              0u);
    ::testing::Test::RecordProperty("actual_distinct_physical_sockets", 2);
    ::testing::Test::RecordProperty("actual_complete_packet_invocations", invocation_count);
    ::testing::Test::RecordProperty("actual_cohort_live_row_trace", "9,0,1,9 repeated five times");
    ::testing::Test::RecordProperty("actual_logical_row_trace", "9,9,1,9 repeated five times");
    ::testing::Test::RecordProperty("actual_admitted_request_rows", rows);
    }
    catch (const std::exception &failure)
    {
        abortUnexpected(failure.what());
    }
    catch (...)
    {
        abortUnexpected("unknown failure with an admitted native domain");
    }
}
/**
 * @brief Fail the original two-rank job before GTest can swallow one actor's exception.
 * @param message Original diagnostic from the failed actor.
 * @note No success, return timeline or readiness is fabricated to wake a peer.
 */
[[noreturn]] inline void abortUnexpected(const char *message) noexcept
{
    std::fprintf(stderr, "CPU joint captured packet fatal exception: %s\n", message);
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

/**
 * @brief Preserve the original fatal MPI boundary for all unexpected native-fixture failures.
 * @param device Exact continuation device supplied by the owning native case.
 *
 * Setup exceptions must not escape to GTest while the other original member is
 * entering a native operation. An admitted domain deliberately terminates if it
 * unwinds without explicit retirement, so the scoped terminate boundary also
 * aborts that same original communicator before any peer can remain stranded.
 * The previous process handler is restored after normal test completion.
 */
inline void run(DeviceId device)
{
    struct RestoreTerminate
    {
        std::terminate_handler original;
        /** @brief Restore the exact original process handler after normal completion. */
        ~RestoreTerminate() { std::set_terminate(original); }
    } restore{std::set_terminate(
        []
        {
            if (const auto error = std::current_exception())
            {
                try
                {
                    std::rethrow_exception(error);
                }
                catch (const std::exception &failure)
                {
                    abortUnexpected(failure.what());
                }
                catch (...)
                {
                    abortUnexpected("unknown exception during native owner retirement");
                }
            }
            abortUnexpected("native owner terminated without an active exception");
        })};
    try
    {
        runOriginal(device);
    }
    catch (const std::exception &failure)
    {
        abortUnexpected(failure.what());
    }
    catch (...)
    {
        abortUnexpected("unknown native-fixture exception");
    }
}
} // namespace rooted_packet_lifecycle

#endif // HAVE_CUDA || HAVE_ROCM
