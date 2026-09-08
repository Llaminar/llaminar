/**
 * @file Test__MappedActivationAdmissionMPI.cpp
 * @brief Two-process CUDA/ROCm proof for mapped graph admission publication.
 *
 * The production heterogeneous ExpertOverlay path maps the same POSIX shared
 * object independently in two MPI processes, then registers each process-local
 * virtual address with the GPU family owned by that rank.  Same-process tests
 * cannot prove that a scheduler publication made through one mapping is visible
 * to a retained graph launched from another process.  This fixture exercises
 * exactly that boundary in both topology directions: the follower rank arms the
 * production activation protocol, and both endpoint GPUs wait on the mapped
 * admission word before publishing a distinct mapped timeline word. A second
 * proof retains the production-shaped alternating-bank ping-pong across 48
 * ordered MoE stages, which admission-only and same-process tests cannot prove.
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/moe/MoEOverlayActivationEpochProtocol.h"
#include "execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "transfer/TransferEngine.h"
#include "utils/MPIContext.h"

#include <atomic>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        constexpr std::int32_t kSourceParticipant = 2;
        constexpr std::int32_t kTargetParticipant = 7;
        constexpr std::int32_t kModelLayer = 5;
        constexpr std::int32_t kDModel = 8;
        constexpr std::int32_t kTopK = 2;
        constexpr std::uint64_t kPlacementEpoch = 41u;
        constexpr std::uint64_t kPublishedTimeline = 9u;
        constexpr std::uint32_t kRetainedMoEStageCount = 48u;

        /** Complete retained transaction shape built on each MPI endpoint. */
        enum class RetainedMappedGraphRecipe : std::uint8_t
        {
            AdmissionPublication, ///< Wait for admission, then publish once.
            AlternatingBankRoundTrip, ///< Exchange every ordered MoE stage.
        };

        /** @return Main-prefill role bit accepted by the one-layer fixture. */
        constexpr std::uint32_t prefillRoleBit() noexcept
        {
            return std::uint32_t{1}
                   << static_cast<std::uint32_t>(
                          MoEOverlayInferenceGraphRole::MainPrefill);
        }

        /** @return Backend authority for one exact planner-selected GPU. */
        IBackend *backendFor(DeviceId device) noexcept
        {
            if (device.is_cuda())
                return getCUDABackend();
            if (device.is_rocm())
                return getROCmBackend();
            return nullptr;
        }

        /**
         * @brief Turn one rank-local predicate into a shared test checkpoint.
         *
         * Every rank reaches the same all-reduce even when its local operation
         * failed. This prevents an assertion on one process from abandoning its
         * peer inside a later MPI or shared-memory rendezvous.
         */
        bool allRanksSucceeded(bool local_success) noexcept
        {
            int local = local_success ? 1 : 0;
            int global = 0;
            return MPI_Allreduce(
                       &local,
                       &global,
                       1,
                       MPI_INT,
                       MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS &&
                   global != 0;
        }

        /** @brief Poll one exact terminal event without synchronizing a device. */
        bool awaitEvent(
            IBackend *backend,
            int ordinal,
            void *event,
            std::chrono::steady_clock::duration timeout) noexcept
        {
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            while (backend && event &&
                   std::chrono::steady_clock::now() < deadline)
            {
                bool ready = false;
                if (!backend->queryEvent(event, ordinal, &ready))
                    return false;
                if (ready)
                    return true;
                std::this_thread::yield();
            }
            return false;
        }

        /** @return Fixed rank-batch storage used only to derive mapping layout. */
        std::shared_ptr<MoEOverlayRankBatchWireWorkspace> makeWorkspace()
        {
            return std::make_shared<MoEOverlayRankBatchWireWorkspace>(
                MoEOverlayRankBatchWireWorkspace::Config{
                    .participant_ids = {kTargetParticipant},
                    .max_total_rows = 1u,
                    .max_total_entries = static_cast<std::size_t>(kTopK),
                    .d_model = kDModel,
                    .top_k = kTopK,
                });
        }

        /** @brief Build the identical topology recipe on both MPI endpoints. */
        MoEOverlayRankBatchTransportConfig makeTransportConfig(
            std::shared_ptr<MPIContext> context,
            int source_rank,
            int target_rank,
            DeviceId local_device,
            std::string channel_identity)
        {
            return {
                .mpi_ctx = std::move(context),
                .source_world_rank = source_rank,
                .target_world_rank = target_rank,
                .workspace = makeWorkspace(),
                .max_rows_per_participant = 1u,
                .max_entries_per_participant =
                    static_cast<std::size_t>(kTopK),
                .d_model = kDModel,
                .top_k = kTopK,
                .tier_index = 1,
                .domain_ordinal = 9,
                .channel_identity = std::move(channel_identity),
                .transaction_slot_count = 8u,
                .transaction_topology = {
                    .workspace_generation = 17u,
                    .topology_fingerprint_low = 0x1234567812345678ull,
                    .topology_fingerprint_high = 0x8765432187654321ull,
                    .source_world_rank = source_rank,
                    .target_world_rank = target_rank,
                },
                .source_endpoint = {
                    .world_rank = source_rank,
                    .participant_id = kSourceParticipant,
                    .tier_priority = 0,
                    .domain_ordinal = 4,
                },
                .target_tier_priority = 1,
                .activation_graph_families = {{
                    .graph_role_mask = prefillRoleBit(),
                    .model_layer_indices = {kModelLayer},
                }},
                .local_lanes = {{
                    .participant_id = kTargetParticipant,
                    .device = local_device,
                }},
            };
        }

        /**
         * @brief Own one process-local endpoint of the two-process transaction.
         */
        class AdmissionEndpoint final
        {
        public:
            /** @brief Bind immutable endpoint topology; resources arrive later. */
            AdmissionEndpoint(
                int world_rank,
                int source_rank,
                int target_rank,
                DeviceId device)
                : world_rank_(world_rank),
                  source_rank_(source_rank),
                  target_rank_(target_rank),
                  device_(device),
                  backend_(backendFor(device))
            {
            }

            /**
             * @brief Release blocked waits, then destroy graph/stream/mapping.
             */
            ~AdmissionEndpoint()
            {
                publishAbortFromHost();
                if (backend_ && stream_ && terminal_ &&
                    backend_->recordEvent(
                        terminal_, device_.gpu_ordinal(), stream_))
                {
                    (void)awaitEvent(
                        backend_,
                        device_.gpu_ordinal(),
                        terminal_,
                        std::chrono::seconds(2));
                }
                graph_.reset();
                if (backend_ && terminal_)
                    backend_->destroyEvent(
                        terminal_, device_.gpu_ordinal());
                if (backend_ && stream_)
                    backend_->destroyStream(
                        stream_, device_.gpu_ordinal());
                transport_.reset();
            }

            AdmissionEndpoint(const AdmissionEndpoint &) = delete;
            AdmissionEndpoint &operator=(const AdmissionEndpoint &) = delete;

            /** @brief Construct the real POSIX transport on this MPI process. */
            bool initializeTransport(
                std::shared_ptr<MPIContext> context,
                const std::string &channel_identity)
            {
                if (!backend_ ||
                    backend_->deviceCount() <= device_.gpu_ordinal())
                {
                    error_ = "required local GPU is unavailable";
                    return false;
                }
                try
                {
                    transport_ = std::make_unique<
                        MoEOverlayNodeLocalRankBatchTransport>(
                        makeTransportConfig(
                            std::move(context),
                            source_rank_,
                            target_rank_,
                            device_,
                            channel_identity));
                    lane_ = transport_->activationDeviceLane(
                        kTargetParticipant,
                        /*graph_family_ordinal=*/0u,
                        device_);
                }
                catch (const std::exception &exception)
                {
                    error_ = exception.what();
                    return false;
                }
                if (!lane_.valid())
                {
                    error_ = "production transport returned an invalid lane";
                    return false;
                }
                return true;
            }

            /**
             * @brief Capture one complete endpoint-owned mapped transaction.
             *
             * The round-trip recipe matches production ordering: the source
             * publishes dispatch then acquires return, while the follower
             * acquires dispatch then publishes return. Both reuse two physical
             * banks across all retained stages with no intervening host work.
             *
             * @param recipe Typed retained transaction shape.
             * @param stage_count Ordered stages for the round-trip recipe.
             * @return True only when the exact-stream graph is executable.
             */
            bool materializeGraph(
                RetainedMappedGraphRecipe recipe,
                std::uint32_t stage_count = 0u)
            {
                if ((recipe ==
                         RetainedMappedGraphRecipe::AdmissionPublication &&
                     stage_count != 0u) ||
                    (recipe ==
                         RetainedMappedGraphRecipe::AlternatingBankRoundTrip &&
                     stage_count == 0u))
                {
                    error_ =
                        "retained mapped graph recipe has invalid stage geometry";
                    return false;
                }
                try
                {
                    auto &context =
                        GPUDeviceContextPool::instance().getContext(device_);
                    stream_ = backend_->createStream(device_.gpu_ordinal());
                    terminal_ = backend_->createEvent(device_.gpu_ordinal());
                    graph_ = context.createGraphCapture(stream_);
                    if (!stream_ || !terminal_ || !graph_ ||
                        !graph_->beginCapture())
                    {
                        error_ = "could not begin exact-stream admission graph";
                        return false;
                    }
                    engine_.enqueueMappedTimelineWait64(
                        *lane_.mapped_region,
                        lane_.admission_signal_offset,
                        kMoEOverlayActivationAdmissionTimeline,
                        device_,
                        stream_);
                    if (recipe ==
                        RetainedMappedGraphRecipe::AdmissionPublication)
                    {
                        const std::size_t output_offset =
                            world_rank_ == source_rank_
                                ? lane_.dispatch_signal_offsets[0]
                                : lane_.return_signal_offsets[0];
                        engine_.enqueueMappedTimelinePublish64(
                            *lane_.mapped_region,
                            output_offset,
                            kPublishedTimeline,
                            device_,
                            stream_);
                    }
                    else
                    {
                        const bool source = world_rank_ == source_rank_;
                        for (std::uint32_t stage = 0u;
                             stage < stage_count;
                             ++stage)
                        {
                            const std::uint32_t bank =
                                moeOverlayActivationBufferIndex(stage);
                            const std::uint64_t timeline =
                                moeOverlayActivationLeasedTimelineValue(
                                    moeOverlayActivationBufferVisit(stage));
                            if (source)
                            {
                                engine_.enqueueMappedTimelinePublish64(
                                    *lane_.mapped_region,
                                    lane_.dispatch_signal_offsets[bank],
                                    timeline,
                                    device_,
                                    stream_);
                                engine_.enqueueMappedTimelineWait64(
                                    *lane_.mapped_region,
                                    lane_.return_signal_offsets[bank],
                                    timeline,
                                    device_,
                                    stream_);
                            }
                            else
                            {
                                engine_.enqueueMappedTimelineWait64(
                                    *lane_.mapped_region,
                                    lane_.dispatch_signal_offsets[bank],
                                    timeline,
                                    device_,
                                    stream_);
                                engine_.enqueueMappedTimelinePublish64(
                                    *lane_.mapped_region,
                                    lane_.return_signal_offsets[bank],
                                    timeline,
                                    device_,
                                    stream_);
                            }
                        }
                    }
                    const std::size_t expected_nodes =
                        recipe ==
                                RetainedMappedGraphRecipe::AdmissionPublication
                            ? 2u
                            : 1u + 2u * stage_count;
                    if (!graph_->endCapture() ||
                        graph_->nodeCount() != expected_nodes ||
                        !graph_->instantiate())
                    {
                        error_ =
                            "mapped capture did not produce the exact retained executable";
                        return false;
                    }
                }
                catch (const std::exception &exception)
                {
                    error_ = exception.what();
                    return false;
                }
                return true;
            }

            /** @brief Arm through the same CPU protocol used by the follower. */
            bool armIfFollower()
            {
                if (world_rank_ != target_rank_)
                    return true;
                try
                {
                    const auto config = transport_->activationEpochConfig(
                        kTargetParticipant, 0u);
                    protocol_ = std::make_unique<
                        MoEOverlayActivationEpochProtocol>(
                        transport_->activationEpochControl(
                            kTargetParticipant, 0u),
                        config);
                    const MoEOverlayInferenceTopologyIdentity topology{
                        .workspace_generation = config.workspace_generation,
                        .topology_fingerprint_low =
                            config.topology_fingerprint_low,
                        .topology_fingerprint_high =
                            config.topology_fingerprint_high,
                        .source_world_rank = source_rank_,
                        .target_world_rank = target_rank_,
                    };
                    const MoEOverlayInferenceCommandIdentity command{
                        .request_generation = 1u,
                        .command_id = 1u,
                        .initial_placement_epoch = kPlacementEpoch,
                    };
                    const auto ticket =
                        makeMoEOverlayInferenceExecutionTicket(
                            topology,
                            command,
                            /*transaction_ordinal=*/1u,
                            /*logical_step_id=*/1u,
                            kPlacementEpoch,
                            MoEOverlayInferenceGraphRole::MainPrefill,
                            /*request_count=*/1,
                            /*logical_rows_per_request=*/1,
                            /*physical_rows_per_request=*/1);
                    const auto deadline =
                        std::chrono::steady_clock::now() +
                        std::chrono::seconds(10);
                    const auto deadline_ns = static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            deadline.time_since_epoch())
                            .count());
                    identity_ = protocol_->arm(
                        ticket,
                        /*epoch_generation=*/1u,
                        deadline_ns,
                        &error_);
                }
                catch (const std::exception &exception)
                {
                    error_ = exception.what();
                    return false;
                }
                return identity_.has_value();
            }

            /** @brief Submit the retained graph and its terminal event. */
            bool launch()
            {
                return graph_ && graph_->launch() &&
                       backend_->recordEvent(
                           terminal_, device_.gpu_ordinal(), stream_);
            }

            /** @brief Observe only the exact endpoint terminal event. */
            bool awaitTerminal()
            {
                const bool ready = awaitEvent(
                    backend_,
                    device_.gpu_ordinal(),
                    terminal_,
                    std::chrono::seconds(5));
                if (!ready)
                    error_ = "retained admission graph terminal timed out";
                return ready;
            }

            /** @return Host-acquired source publication from either mapping. */
            std::uint64_t dispatchTimeline(
                std::uint32_t bank = 0u) const noexcept
            {
                return lane_.control_host &&
                               bank < kMoEOverlayActivationBufferCount
                           ? std::atomic_ref<std::uint64_t>(
                                 lane_.control_host->buffers[bank]
                                     .dispatch_signal.value)
                                 .load(std::memory_order_acquire)
                           : 0u;
            }

            /** @return Host-acquired follower publication from either mapping. */
            std::uint64_t returnTimeline(
                std::uint32_t bank = 0u) const noexcept
            {
                return lane_.control_host &&
                               bank < kMoEOverlayActivationBufferCount
                           ? std::atomic_ref<std::uint64_t>(
                                 lane_.control_host->buffers[bank]
                                     .return_signal.value)
                                 .load(std::memory_order_acquire)
                           : 0u;
            }

            /** @return Last rank-local diagnostic. */
            const std::string &error() const noexcept { return error_; }

        private:
            /** @brief Release any outstanding mapped GEQ wait during teardown. */
            void publishAbortFromHost() noexcept
            {
                if (!lane_.control_host)
                    return;
                std::atomic_ref<std::uint64_t>(
                    lane_.control_host->admission.ready_signal)
                    .store(
                        kMoEOverlayActivationAbortTimeline,
                        std::memory_order_release);
                for (std::uint32_t bank = 0u;
                     bank < kMoEOverlayActivationBufferCount;
                     ++bank)
                {
                    std::atomic_ref<std::uint64_t>(
                        lane_.control_host->buffers[bank]
                            .dispatch_signal.value)
                        .store(
                            kMoEOverlayActivationAbortTimeline,
                            std::memory_order_release);
                    std::atomic_ref<std::uint64_t>(
                        lane_.control_host->buffers[bank]
                            .return_signal.value)
                        .store(
                            kMoEOverlayActivationAbortTimeline,
                            std::memory_order_release);
                }
            }

            int world_rank_ = -1;
            int source_rank_ = -1;
            int target_rank_ = -1;
            DeviceId device_ = DeviceId::invalid();
            IBackend *backend_ = nullptr;
            TransferEngine engine_;
            std::unique_ptr<MoEOverlayNodeLocalRankBatchTransport> transport_;
            MoEOverlayMappedActivationDeviceLane lane_;
            void *stream_ = nullptr;
            void *terminal_ = nullptr;
            std::unique_ptr<IGPUGraphCapture> graph_;
            std::unique_ptr<MoEOverlayActivationEpochProtocol> protocol_;
            std::optional<MoEOverlayActivationEpochIdentity> identity_;
            std::string error_;
        };

        /** @brief Execute one topology direction with matched MPI checkpoints. */
        bool runDirection(
            int source_rank,
            int target_rank,
            DeviceId source_device,
            DeviceId target_device,
            const std::string &identity,
            RetainedMappedGraphRecipe recipe =
                RetainedMappedGraphRecipe::AdmissionPublication,
            std::uint32_t stage_count = 0u)
        {
            int world_rank = -1;
            int world_size = 0;
            if (MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) != MPI_SUCCESS ||
                MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS ||
                world_size != 2)
            {
                return false;
            }
            const DeviceId local_device =
                world_rank == source_rank ? source_device : target_device;
            auto context = std::make_shared<MPIContext>(
                world_rank, world_size, MPI_COMM_WORLD);
            AdmissionEndpoint endpoint(
                world_rank, source_rank, target_rank, local_device);

            const bool transport_ok = endpoint.initializeTransport(
                context, identity);
            if (!allRanksSucceeded(transport_ok))
            {
                ADD_FAILURE() << "transport setup failed on rank "
                              << world_rank << ": " << endpoint.error();
                return false;
            }

            const bool capture_ok = endpoint.materializeGraph(
                recipe, stage_count);
            if (!allRanksSucceeded(capture_ok))
            {
                ADD_FAILURE() << "graph materialization failed on rank "
                              << world_rank << ": " << endpoint.error();
                return false;
            }

            const bool arm_ok = endpoint.armIfFollower();
            if (!allRanksSucceeded(arm_ok))
            {
                ADD_FAILURE() << "admission arm failed on rank "
                              << world_rank << ": " << endpoint.error();
                return false;
            }

            const bool launch_ok = endpoint.launch();
            if (!allRanksSucceeded(launch_ok))
            {
                ADD_FAILURE() << "graph launch failed on rank "
                              << world_rank << ": " << endpoint.error();
                return false;
            }

            const bool terminal_ok = endpoint.awaitTerminal();
            if (!allRanksSucceeded(terminal_ok))
            {
                ADD_FAILURE() << "admission graph failed on rank "
                              << world_rank << ": " << endpoint.error();
                return false;
            }

            bool visible = true;
            if (recipe ==
                RetainedMappedGraphRecipe::AdmissionPublication)
            {
                visible =
                    endpoint.dispatchTimeline() == kPublishedTimeline &&
                    endpoint.returnTimeline() == kPublishedTimeline;
            }
            else
            {
                std::array<std::uint64_t,
                           kMoEOverlayActivationBufferCount>
                    expected{};
                for (std::uint32_t stage = 0u;
                     stage < stage_count;
                     ++stage)
                {
                    expected[moeOverlayActivationBufferIndex(stage)] =
                        moeOverlayActivationLeasedTimelineValue(
                            moeOverlayActivationBufferVisit(stage));
                }
                for (std::uint32_t bank = 0u;
                     bank < kMoEOverlayActivationBufferCount;
                     ++bank)
                {
                    visible = visible &&
                              endpoint.dispatchTimeline(bank) ==
                                  expected[bank] &&
                              endpoint.returnTimeline(bank) ==
                                  expected[bank];
                }
            }
            if (!allRanksSucceeded(visible))
            {
                ADD_FAILURE() << "cross-process mapped publications were not "
                                 "visible on rank "
                              << world_rank << " dispatch="
                              << endpoint.dispatchTimeline() << " return="
                              << endpoint.returnTimeline();
                return false;
            }
            return true;
        }
    } // namespace

    TEST(Test__MappedActivationAdmissionMPI,
         FollowerHostArmReleasesCUDAAndROCmGraphsInBothRankDirections)
    {
        int world_rank = -1;
        int world_size = 0;
        ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank), MPI_SUCCESS);
        ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &world_size), MPI_SUCCESS);
        if (world_size != 2)
            GTEST_SKIP() << "requires exactly two MPI ranks";

        const bool forward_hardware =
            world_rank == 0
                ? backendFor(DeviceId::cuda(0)) &&
                      backendFor(DeviceId::cuda(0))->deviceCount() >= 1
                : backendFor(DeviceId::rocm(0)) &&
                      backendFor(DeviceId::rocm(0))->deviceCount() >= 1;
        if (!allRanksSucceeded(forward_hardware))
            GTEST_SKIP() << "requires CUDA on rank 0 and ROCm on rank 1";

        ASSERT_TRUE(runDirection(
            /*source_rank=*/0,
            /*target_rank=*/1,
            DeviceId::cuda(0),
            DeviceId::rocm(0),
            "mapped_admission_mpi_cuda_source"));
        ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
        ASSERT_TRUE(runDirection(
            /*source_rank=*/1,
            /*target_rank=*/0,
            DeviceId::rocm(0),
            DeviceId::cuda(0),
            "mapped_admission_mpi_rocm_source"));
    }

    /**
     * @test Two process-local mappings sustain a full retained MoE timeline.
     */
    TEST(Test__MappedActivationAdmissionMPI,
         FortyEightStageRetainedRoundTripCompletesInBothRankDirections)
    {
        int world_rank = -1;
        int world_size = 0;
        ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank), MPI_SUCCESS);
        ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &world_size), MPI_SUCCESS);
        if (world_size != 2)
            GTEST_SKIP() << "requires exactly two MPI ranks";

        const bool forward_hardware =
            world_rank == 0
                ? backendFor(DeviceId::cuda(0)) &&
                      backendFor(DeviceId::cuda(0))->deviceCount() >= 1
                : backendFor(DeviceId::rocm(0)) &&
                      backendFor(DeviceId::rocm(0))->deviceCount() >= 1;
        if (!allRanksSucceeded(forward_hardware))
            GTEST_SKIP() << "requires CUDA on rank 0 and ROCm on rank 1";

        ASSERT_TRUE(runDirection(
            /*source_rank=*/0,
            /*target_rank=*/1,
            DeviceId::cuda(0),
            DeviceId::rocm(0),
            "mapped_round_trip_mpi_cuda_source",
            RetainedMappedGraphRecipe::AlternatingBankRoundTrip,
            kRetainedMoEStageCount));
        ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
        ASSERT_TRUE(runDirection(
            /*source_rank=*/1,
            /*target_rank=*/0,
            DeviceId::rocm(0),
            DeviceId::cuda(0),
            "mapped_round_trip_mpi_rocm_source",
            RetainedMappedGraphRecipe::AlternatingBankRoundTrip,
            kRetainedMoEStageCount));
    }
} // namespace llaminar2::test
