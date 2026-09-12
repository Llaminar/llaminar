/**
 * @file Test__CUDAMoEOverlayEpoch.cpp
 * @brief Real-CUDA integration proof for captured ExpertOverlay epoch RCU.
 *
 * The tests exercise the public MoE kernel interface on two non-default
 * streams.  They prove that a maintenance publication can overlap a held
 * inference ticket, that the old bank cannot be reused early, and that one
 * captured acquire/consume/release graph observes a newer epoch on replay
 * without recapture or host-side ticket reset.
 */

#include "execution/moe/MoEOverlayDeviceEpochProtocol.h"
#include "execution/moe/MoEOverlayEpochLeaseLifecycle.h"
#include "execution/moe/DeviceMoERebalanceController.h"
#include "execution/moe/MoERuntimeTable.h"
#include "execution/moe/NativeMoEMovementArchive.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>

namespace llaminar2::test
{
    namespace
    {
        constexpr std::size_t kStatusCount = 9u;

        /** @return Whether at least one CUDA device is usable by this process. */
        bool hasCUDADevice()
        {
            int count = 0;
            return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
        }
    } // namespace

    /**
     * @brief Own exact CUDA streams, events, and persistent epoch graph state.
     *
     * Setup/teardown synchronization is outside inference.  Individual tests
     * deliberately enqueue their complete cross-stream DAG before performing
     * one terminal observation, matching production's event-driven ownership.
     */
    class CUDAMoEOverlayEpochTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if (!hasCUDADevice())
                GTEST_SKIP() << "No CUDA device available";
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            ASSERT_EQ(
                cudaStreamCreateWithFlags(
                    &inference_stream_, cudaStreamNonBlocking),
                cudaSuccess);
            ASSERT_EQ(
                cudaStreamCreateWithFlags(
                    &maintenance_stream_, cudaStreamNonBlocking),
                cudaSuccess);
            ASSERT_EQ(
                cudaEventCreateWithFlags(
                    &inference_event_, cudaEventDisableTiming),
                cudaSuccess);
            ASSERT_EQ(
                cudaEventCreateWithFlags(
                    &maintenance_event_, cudaEventDisableTiming),
                cudaSuccess);

            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_control_),
                          sizeof(*device_control_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_tickets_),
                          2u * sizeof(*device_tickets_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_statuses_),
                          kStatusCount * sizeof(*device_statuses_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_epochs_),
                          2u * sizeof(*device_epochs_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_evidence_),
                          sizeof(*device_evidence_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_runtime_),
                          sizeof(*device_runtime_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_runtime_family_),
                          2u * sizeof(*device_runtime_family_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_apply_status_),
                          sizeof(*device_apply_status_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_controller_),
                          sizeof(*device_controller_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_plan_entry_),
                          2u * sizeof(*device_plan_entry_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_command_header_),
                          sizeof(*device_command_header_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_transfer_slot_),
                          sizeof(*device_transfer_slot_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_placement_banks_),
                          2u * sizeof(*device_placement_banks_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_route_expert_),
                          sizeof(*device_route_expert_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_route_participant_),
                          sizeof(*device_route_participant_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_position_),
                          sizeof(*device_position_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_active_rows_),
                          sizeof(*device_active_rows_)),
                      cudaSuccess);
            ASSERT_EQ(cudaMalloc(
                          reinterpret_cast<void **>(&device_route_evidence_),
                          sizeof(*device_route_evidence_)),
                      cudaSuccess);

            DeviceMoEOverlayEpochControl initial_control{};
            ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&device_movement_waves_),
                                  20u * sizeof(*device_movement_waves_)), cudaSuccess);
            ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&device_movement_edges_),
                                  40u * sizeof(*device_movement_edges_)), cudaSuccess);
            MoEOverlayDeviceEpochProtocol::initialize(
                initial_control, /*initial_epoch=*/1u);
            const std::array<std::uint64_t, 2> epochs{2u, 1u};
            ASSERT_EQ(cudaMemcpyAsync(
                          device_control_,
                          &initial_control,
                          sizeof(initial_control),
                          cudaMemcpyHostToDevice,
                          inference_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaMemsetAsync(
                          device_tickets_,
                          0,
                          2u * sizeof(*device_tickets_),
                          inference_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaMemsetAsync(
                          device_statuses_,
                          0,
                          kStatusCount * sizeof(*device_statuses_),
                          inference_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(
                          device_epochs_,
                          epochs.data(),
                          sizeof(epochs),
                          cudaMemcpyHostToDevice,
                          inference_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaMemsetAsync(
                          device_evidence_,
                          0,
                          sizeof(*device_evidence_),
                          inference_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaMemsetAsync(
                          device_transfer_slot_, 0,
                          sizeof(*device_transfer_slot_), inference_stream_),
                      cudaSuccess);

            /* Canonical main-model bank zero owns expert zero on participant
             * zero. Bank one is already prepared for epoch two and owns it on
             * participant one. The sidecar's embedded banks remain empty and
             * its mutable active_bank points at bank one: only the request
             * ticket plus canonical placement pointer can select correctly. */
            DeviceMoELayerRuntime runtime{};
            runtime.active_bank = 1u;
            runtime.active_epoch = 2u;
            runtime.expert_count = 1u;
            runtime.top_k = 1u;
            runtime.participant_id = 0u;
            runtime.participant_count = 2u;
            runtime.prefill_route_capacity = 1u;
            runtime.route_expert_ids = device_route_expert_;
            runtime.route_participant_ids = device_route_participant_;
            runtime.overlay_epoch_ticket = &device_tickets_[0];
            runtime.overlay_placement_banks = device_placement_banks_;
            std::array<DeviceMoEPlacementBank, 2> placement_banks{};
            placement_banks[0].epoch = 1u;
            placement_banks[0].expert_count = 1u;
            placement_banks[0].experts[0].owner_participant = 0;
            placement_banks[0].resident_participant_mask[0] = 0b01u;
            placement_banks[1].epoch = 2u;
            placement_banks[1].expert_count = 1u;
            placement_banks[1].experts[0].owner_participant = 1;
            placement_banks[1].resident_participant_mask[0] = 0b10u;
            const int32_t route_expert = 0;
            const int32_t route_participant = -1;
            const int32_t position = 7;
            const int32_t active_rows = 1;
            ASSERT_EQ(cudaMemcpyAsync(
                          device_runtime_, &runtime, sizeof(runtime),
                          cudaMemcpyHostToDevice, inference_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(
                          device_placement_banks_, placement_banks.data(),
                          sizeof(placement_banks), cudaMemcpyHostToDevice,
                          inference_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(
                          device_route_expert_, &route_expert,
                          sizeof(route_expert), cudaMemcpyHostToDevice,
                          inference_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(
                          device_route_participant_, &route_participant,
                          sizeof(route_participant), cudaMemcpyHostToDevice,
                          inference_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(
                          device_position_, &position, sizeof(position),
                          cudaMemcpyHostToDevice, inference_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(
                          device_active_rows_, &active_rows,
                          sizeof(active_rows), cudaMemcpyHostToDevice,
                          inference_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(inference_stream_), cudaSuccess);

            kernel_ = llaminar::v2::kernels::KernelFactory::createMoEKernel(
                DeviceId::cuda(0));
            ASSERT_NE(kernel_, nullptr);
        }

        void TearDown() override
        {
            if (!hasCUDADevice())
                return;
            (void)cudaSetDevice(0);
            if (inference_stream_)
                (void)cudaStreamSynchronize(inference_stream_);
            if (maintenance_stream_)
                (void)cudaStreamSynchronize(maintenance_stream_);
            kernel_.reset();
            if (graph_exec_)
                (void)cudaGraphExecDestroy(graph_exec_);
            if (graph_)
                (void)cudaGraphDestroy(graph_);
            if (device_evidence_)
                (void)cudaFree(device_evidence_);
            if (device_route_evidence_)
                (void)cudaFree(device_route_evidence_);
            if (device_active_rows_)
                (void)cudaFree(device_active_rows_);
            if (device_position_)
                (void)cudaFree(device_position_);
            if (device_route_participant_)
                (void)cudaFree(device_route_participant_);
            if (device_route_expert_)
                (void)cudaFree(device_route_expert_);
            if (device_runtime_)
                (void)cudaFree(device_runtime_);
            if (device_apply_status_)
                (void)cudaFree(device_apply_status_);
            if (device_controller_)
                (void)cudaFree(device_controller_);
            if (device_command_header_)
                (void)cudaFree(device_command_header_);
            if (device_plan_entry_)
                (void)cudaFree(device_plan_entry_);
            if (device_movement_waves_)
                (void)cudaFree(device_movement_waves_);
            if (device_movement_edges_)
                (void)cudaFree(device_movement_edges_);
            if (device_transfer_slot_)
                (void)cudaFree(device_transfer_slot_);
            if (device_runtime_family_)
                (void)cudaFree(device_runtime_family_);
            if (device_placement_banks_)
                (void)cudaFree(device_placement_banks_);
            if (device_epochs_)
                (void)cudaFree(device_epochs_);
            if (device_statuses_)
                (void)cudaFree(device_statuses_);
            if (device_tickets_)
                (void)cudaFree(device_tickets_);
            if (device_control_)
                (void)cudaFree(device_control_);
            if (maintenance_event_)
                (void)cudaEventDestroy(maintenance_event_);
            if (inference_event_)
                (void)cudaEventDestroy(inference_event_);
            if (maintenance_stream_)
                (void)cudaStreamDestroy(maintenance_stream_);
            if (inference_stream_)
                (void)cudaStreamDestroy(inference_stream_);
        }

        /** @return Public launch contract for the inference stream. */
        MoEKernelLaunchContext inferenceLaunch() const noexcept
        {
            return {.stream = inference_stream_, .workspace = nullptr};
        }

        /** @return Public launch contract for the migration stream. */
        MoEKernelLaunchContext maintenanceLaunch() const noexcept
        {
            return {.stream = maintenance_stream_, .workspace = nullptr};
        }

        /** @return Persistent append storage owned until captured graphs retire. */
        DeviceMoERebalanceMovementJournalView journalView() const noexcept
        {
            return {&device_controller_->movement_journal, device_movement_waves_,
                    device_movement_edges_, 20u, 40u};
        }

        cudaStream_t inference_stream_ = nullptr;
        cudaStream_t maintenance_stream_ = nullptr;
        cudaEvent_t inference_event_ = nullptr;
        cudaEvent_t maintenance_event_ = nullptr;
        cudaGraph_t graph_ = nullptr;
        cudaGraphExec_t graph_exec_ = nullptr;
        DeviceMoEOverlayEpochControl *device_control_ = nullptr;
        DeviceMoEOverlayEpochTicket *device_tickets_ = nullptr;
        DeviceMoEOverlayEpochStatus *device_statuses_ = nullptr;
        std::uint64_t *device_epochs_ = nullptr;
        DeviceMoEOverlayEpochTicket *device_evidence_ = nullptr;
        DeviceMoELayerRuntime *device_runtime_ = nullptr;
        DeviceMoELayerRuntime *device_runtime_family_ = nullptr;
        DeviceMoERebalanceApplyStatus *device_apply_status_ = nullptr;
        DeviceMoERebalanceGraphControllerState *device_controller_ = nullptr;
        DeviceMoERebalancePlanEntry *device_plan_entry_ = nullptr;
        DeviceMoERebalanceMovementWave *device_movement_waves_ = nullptr;
        DeviceMoERebalanceMovementEdge *device_movement_edges_ = nullptr;
        DeviceMoERebalanceCommandBufferHeader *device_command_header_ = nullptr;
        DeviceMoEExpertDirectoryEntry *device_transfer_slot_ = nullptr;
        DeviceMoEPlacementBank *device_placement_banks_ = nullptr;
        int32_t *device_route_expert_ = nullptr;
        int32_t *device_route_participant_ = nullptr;
        int32_t *device_position_ = nullptr;
        int32_t *device_active_rows_ = nullptr;
        int32_t *device_route_evidence_ = nullptr;
        std::unique_ptr<IMoEKernel> kernel_;
    };

    /**
     * @brief Prove a reused leased signal cannot admit an empty old descriptor.
     *
     * The peer timeline begins at the captured value while its descriptor is
     * deliberately empty. The acquire must remain resident until a newer
     * generation's matching descriptor is release-published from mapped host
     * memory, then select that descriptor's exact placement epoch.
     */
    TEST_F(CUDAMoEOverlayEpochTest,
           PeerEpochAcquireAuthenticatesGenerationAndPinsPreparedCandidate)
    {
        MoEOverlayActivationEpochControl *host_control = nullptr;
        ASSERT_EQ(
            cudaHostAlloc(
                reinterpret_cast<void **>(&host_control),
                sizeof(*host_control),
                cudaHostAllocMapped),
            cudaSuccess);
        auto host_owner = std::unique_ptr<
            MoEOverlayActivationEpochControl,
            void (*)(MoEOverlayActivationEpochControl *)>(
            host_control,
            [](MoEOverlayActivationEpochControl *pointer)
            {
                if (pointer)
                    (void)cudaFreeHost(pointer);
            });
        void *device_control_alias_raw = nullptr;
        ASSERT_EQ(
            cudaHostGetDevicePointer(
                &device_control_alias_raw,
                host_control,
                0u),
            cudaSuccess);
        auto *const device_activation_control =
            static_cast<MoEOverlayActivationEpochControl *>(
                device_control_alias_raw);

        MoEOverlayActivationDeviceEpochGrant *device_grant = nullptr;
        ASSERT_EQ(
            cudaMalloc(
                reinterpret_cast<void **>(&device_grant),
                sizeof(*device_grant)),
            cudaSuccess);
        auto grant_owner = std::unique_ptr<
            MoEOverlayActivationDeviceEpochGrant,
            void (*)(MoEOverlayActivationDeviceEpochGrant *)>(
            device_grant,
            [](MoEOverlayActivationDeviceEpochGrant *pointer)
            {
                if (pointer)
                    (void)cudaFree(pointer);
            });

        *host_control = {};
        host_control->channel.stage_count = 1u;
        host_control->admission.ready_signal =
            kMoEOverlayActivationAdmissionTimeline;
        host_control->identity.epoch_generation = 8u;
        host_control->identity.digest = {0x1234u, 0x5678u};
        host_control->buffers[0].dispatch_signal.value =
            moeOverlayActivationLeasedTimelineValue(1u);

        MoEOverlayActivationDeviceEpochGrant grant{};
        grant.generation = 7u;
        ASSERT_EQ(
            cudaMemcpyAsync(
                device_grant,
                &grant,
                sizeof(grant),
                cudaMemcpyHostToDevice,
                inference_stream_),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemsetAsync(
                &device_tickets_[1],
                0,
                sizeof(device_tickets_[1]),
                inference_stream_),
            cudaSuccess);

        /* Complete global preparation without flipping this participant's
         * selector. A peer whose selector flips first may now authenticate E2;
         * this participant must pin its Ready bank during fan-out. */
        ASSERT_TRUE(kernel_->reserveMoEOverlayEpochCandidate(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[0],
            &device_statuses_[0]));
        ASSERT_TRUE(kernel_->markMoEOverlayEpochCandidateReady(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[0],
            &device_statuses_[1]));
        ASSERT_EQ(
            cudaEventRecord(maintenance_event_, maintenance_stream_),
            cudaSuccess);
        ASSERT_EQ(
            cudaStreamWaitEvent(inference_stream_, maintenance_event_, 0),
            cudaSuccess);

        ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[1],
            &device_statuses_[8],
            nullptr,
            {},
            {
                .control = device_activation_control,
                .grant = device_grant,
                .stage_ordinal = 0u,
            }));

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        EXPECT_EQ(cudaStreamQuery(inference_stream_), cudaErrorNotReady)
            << "a reused timeline admitted the empty prior descriptor";
        (void)cudaGetLastError();

        auto &descriptor =
            host_control->buffers[0].dispatch_descriptor;
        descriptor.digest = host_control->identity.digest;
        descriptor.timeline = moeOverlayActivationLeasedTimelineValue(1u);
        descriptor.placement_epoch = 2u;
        descriptor.stage_ordinal = 0u;
        std::atomic_thread_fence(std::memory_order_release);
        std::atomic_ref<std::uint64_t>(
            host_control->buffers[0].dispatch_signal.value)
            .store(descriptor.timeline, std::memory_order_release);

        ASSERT_EQ(cudaStreamSynchronize(inference_stream_), cudaSuccess);
        DeviceMoEOverlayEpochTicket ticket{};
        DeviceMoEOverlayEpochStatus status{};
        ASSERT_EQ(
            cudaMemcpy(
                &ticket,
                &device_tickets_[1],
                sizeof(ticket),
                cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                &status,
                &device_statuses_[8],
                sizeof(status),
                cudaMemcpyDeviceToHost),
            cudaSuccess);
        EXPECT_EQ(ticket.epoch, 2u);
        EXPECT_EQ(ticket.bank(), 1u);
        EXPECT_EQ(ticket.generation(), 2u);
        EXPECT_EQ(
            status.code,
            static_cast<std::uint32_t>(
                DeviceMoEOverlayEpochStatusCode::Success));
        EXPECT_EQ(
            status.observed_state,
            static_cast<std::uint32_t>(
                DeviceMoEOverlayEpochBankState::Ready));

        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[1],
            &device_statuses_[8]));
        ASSERT_EQ(cudaStreamSynchronize(inference_stream_), cudaSuccess);
    }

    TEST_F(CUDAMoEOverlayEpochTest,
           IndependentStreamsPublishWithoutReusingHeldInferenceBank)
    {
        ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[0],
            &device_statuses_[0]));
        ASSERT_EQ(cudaEventRecord(inference_event_, inference_stream_), cudaSuccess);

        ASSERT_EQ(
            cudaStreamWaitEvent(maintenance_stream_, inference_event_, 0),
            cudaSuccess);
        ASSERT_TRUE(kernel_->reserveMoEOverlayEpochCandidate(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[0],
            &device_statuses_[1]));
        ASSERT_TRUE(kernel_->markMoEOverlayEpochCandidateReady(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[0],
            &device_statuses_[2]));
        ASSERT_TRUE(kernel_->publishMoEOverlayEpochCandidate(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[0],
            &device_statuses_[3]));
        ASSERT_EQ(
            cudaEventRecord(maintenance_event_, maintenance_stream_),
            cudaSuccess);

        ASSERT_EQ(
            cudaStreamWaitEvent(inference_stream_, maintenance_event_, 0),
            cudaSuccess);
        ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[1],
            &device_statuses_[4]));
        ASSERT_EQ(cudaEventRecord(inference_event_, inference_stream_), cudaSuccess);

        ASSERT_EQ(
            cudaStreamWaitEvent(maintenance_stream_, inference_event_, 0),
            cudaSuccess);
        ASSERT_TRUE(kernel_->retireMoEOverlayEpoch(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[1],
            &device_statuses_[5]));
        ASSERT_EQ(
            cudaEventRecord(maintenance_event_, maintenance_stream_),
            cudaSuccess);

        ASSERT_EQ(
            cudaStreamWaitEvent(inference_stream_, maintenance_event_, 0),
            cudaSuccess);
        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[0],
            &device_statuses_[6]));
        ASSERT_EQ(cudaEventRecord(inference_event_, inference_stream_), cudaSuccess);

        ASSERT_EQ(
            cudaStreamWaitEvent(maintenance_stream_, inference_event_, 0),
            cudaSuccess);
        ASSERT_TRUE(kernel_->retireMoEOverlayEpoch(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[1],
            &device_statuses_[7]));
        ASSERT_EQ(
            cudaEventRecord(maintenance_event_, maintenance_stream_),
            cudaSuccess);

        ASSERT_EQ(
            cudaStreamWaitEvent(inference_stream_, maintenance_event_, 0),
            cudaSuccess);
        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[1],
            &device_statuses_[8]));

        DeviceMoEOverlayEpochControl control{};
        std::array<DeviceMoEOverlayEpochTicket, 2> tickets{};
        std::array<DeviceMoEOverlayEpochStatus, kStatusCount> statuses{};
        ASSERT_EQ(cudaMemcpyAsync(
                      &control,
                      device_control_,
                      sizeof(control),
                      cudaMemcpyDeviceToHost,
                      inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      tickets.data(),
                      device_tickets_,
                      sizeof(tickets),
                      cudaMemcpyDeviceToHost,
                      inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      statuses.data(),
                      device_statuses_,
                      sizeof(statuses),
                      cudaMemcpyDeviceToHost,
                      inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(inference_stream_), cudaSuccess);

        EXPECT_TRUE(statuses[0].succeeded());
        EXPECT_EQ(statuses[0].epoch, 1u);
        EXPECT_TRUE(statuses[1].succeeded());
        EXPECT_TRUE(statuses[2].succeeded());
        EXPECT_TRUE(statuses[3].succeeded());
        EXPECT_EQ(statuses[3].epoch, 2u);
        EXPECT_TRUE(statuses[4].succeeded());
        EXPECT_EQ(statuses[4].epoch, 2u);
        EXPECT_EQ(
            statuses[5].typedCode(),
            DeviceMoEOverlayEpochStatusCode::Busy);
        EXPECT_TRUE(statuses[6].succeeded());
        EXPECT_TRUE(statuses[7].succeeded());
        EXPECT_TRUE(statuses[8].succeeded());
        EXPECT_EQ(control.bank_epochs[0], 0u);
        EXPECT_EQ(control.bank_readers[0], 0u);
        EXPECT_EQ(control.bank_epochs[1], 2u);
        EXPECT_EQ(control.bank_readers[1], 0u);
        EXPECT_EQ(control.published_selector,
                  deviceMoEOverlayEpochSelector(2u, 1u));
        EXPECT_FALSE(tickets[0].valid());
        EXPECT_FALSE(tickets[1].valid());
    }

    TEST_F(CUDAMoEOverlayEpochTest,
           GlobalAdmissionSelectsRetiringBankDuringPublicationFanout)
    {
        ASSERT_TRUE(kernel_->reserveMoEOverlayEpochCandidate(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[0],
            &device_statuses_[0]));
        ASSERT_TRUE(kernel_->markMoEOverlayEpochCandidateReady(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[0],
            &device_statuses_[1]));
        ASSERT_TRUE(kernel_->publishMoEOverlayEpochCandidate(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[0],
            &device_statuses_[2]));
        ASSERT_EQ(
            cudaEventRecord(maintenance_event_, maintenance_stream_),
            cudaSuccess);
        ASSERT_EQ(
            cudaStreamWaitEvent(inference_stream_, maintenance_event_, 0),
            cudaSuccess);

        // device_epochs_[1] remains one: the local selector is already two,
        // but topology-wide admission still requires the retiring bank.
        ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[0],
            &device_statuses_[3],
            &device_epochs_[1]));
        DeviceMoEOverlayEpochTicket old_ticket{};
        ASSERT_EQ(
            cudaMemcpyAsync(
                &old_ticket,
                &device_tickets_[0],
                sizeof(old_ticket),
                cudaMemcpyDeviceToHost,
                inference_stream_),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(inference_stream_), cudaSuccess);
        EXPECT_EQ(old_ticket.epoch, 1u);
        EXPECT_EQ(old_ticket.bank(), 0u);
        EXPECT_EQ(old_ticket.generation(), 1u);

        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[0],
            &device_statuses_[4]));
        const std::uint64_t admitted_epoch = 2u;
        ASSERT_EQ(
            cudaMemcpyAsync(
                &device_epochs_[1],
                &admitted_epoch,
                sizeof(admitted_epoch),
                cudaMemcpyHostToDevice,
                inference_stream_),
            cudaSuccess);
        ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[1],
            &device_statuses_[5],
            &device_epochs_[1]));
        DeviceMoEOverlayEpochTicket new_ticket{};
        ASSERT_EQ(
            cudaMemcpyAsync(
                &new_ticket,
                &device_tickets_[1],
                sizeof(new_ticket),
                cudaMemcpyDeviceToHost,
                inference_stream_),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(inference_stream_), cudaSuccess);
        EXPECT_EQ(new_ticket.epoch, 2u);
        EXPECT_EQ(new_ticket.bank(), 1u);
        EXPECT_EQ(new_ticket.generation(), 2u);
        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[1],
            &device_statuses_[6]));
    }

    /**
     * @brief KV-only completion cannot donate an obsolete reader to main decode.
     *
     * Record a KV-only stand-in (a persistent device mailbox write), publish a
     * newer placement on the independent maintenance stream, then launch the
     * retained main acquire/consume/release graph. The whole DAG is event-ordered;
     * the host observes only the final ticket. Production uses the same typed
     * sidecar policy, so changing KV-only to acquire recreates the stale reader.
     */
    TEST_F(CUDAMoEOverlayEpochTest,
           KVOnlySidecarDoesNotPinPlacementBeforeMainAdmission)
    {
        ASSERT_EQ(cudaStreamBeginCapture(
                      inference_stream_, cudaStreamCaptureModeGlobal),
                  cudaSuccess);
        const auto owner = moeOverlaySidecarEpochOwnership(
            MTPSidecarCaptureRole::KVOnly, true);
        if (owner == MoEOverlaySidecarEpochOwnership::ExternalReader)
        {
            ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
                inferenceLaunch(), device_control_, &device_tickets_[0],
                &device_statuses_[0]));
        }
        ASSERT_EQ(cudaMemsetAsync(device_evidence_, 0,
                                     sizeof(*device_evidence_), inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamEndCapture(inference_stream_, &graph_),
                  cudaSuccess);
        ASSERT_EQ(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0),
                  cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(graph_exec_, inference_stream_), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(inference_event_, inference_stream_), cudaSuccess);
        ASSERT_EQ(cudaStreamWaitEvent(maintenance_stream_, inference_event_, 0),
                  cudaSuccess);

        ASSERT_TRUE(kernel_->reserveMoEOverlayEpochCandidate(
            maintenanceLaunch(), device_control_, &device_epochs_[0],
            &device_statuses_[2]));
        ASSERT_TRUE(kernel_->markMoEOverlayEpochCandidateReady(
            maintenanceLaunch(), device_control_, &device_epochs_[0],
            &device_statuses_[3]));
        ASSERT_TRUE(kernel_->publishMoEOverlayEpochCandidate(
            maintenanceLaunch(), device_control_, &device_epochs_[0],
            &device_statuses_[4]));
        ASSERT_EQ(cudaEventRecord(maintenance_event_, maintenance_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamWaitEvent(inference_stream_, maintenance_event_, 0),
                  cudaSuccess);

        // The next graph owns a fresh acquire, not the sidecar's old ticket.
        ASSERT_EQ(cudaStreamBeginCapture(
                      inference_stream_, cudaStreamCaptureModeGlobal),
                  cudaSuccess);
        ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
            inferenceLaunch(), device_control_, &device_tickets_[0],
            &device_statuses_[5]));
        ASSERT_EQ(cudaMemcpyAsync(device_evidence_, &device_tickets_[0],
                                     sizeof(*device_evidence_),
                                     cudaMemcpyDeviceToDevice, inference_stream_),
                  cudaSuccess);
        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(), device_control_, &device_tickets_[0],
            &device_statuses_[6]));
        cudaGraph_t main_graph = nullptr;
        ASSERT_EQ(cudaStreamEndCapture(inference_stream_, &main_graph),
                  cudaSuccess);
        cudaGraphExec_t main_exec = nullptr;
        ASSERT_EQ(cudaGraphInstantiate(&main_exec, main_graph, nullptr, nullptr, 0),
                  cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(main_exec, inference_stream_), cudaSuccess);
        DeviceMoEOverlayEpochTicket evidence{};
        ASSERT_EQ(cudaMemcpyAsync(&evidence, device_evidence_, sizeof(evidence),
                                     cudaMemcpyDeviceToHost, inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(inference_stream_), cudaSuccess);
        EXPECT_EQ(owner, MoEOverlaySidecarEpochOwnership::NoExpertAccess);
        EXPECT_TRUE(evidence.valid());
        EXPECT_EQ(evidence.epoch, 2u);
        EXPECT_EQ(cudaGraphExecDestroy(main_exec), cudaSuccess);
        EXPECT_EQ(cudaGraphDestroy(main_graph), cudaSuccess);
    }

    TEST_F(CUDAMoEOverlayEpochTest,
           CapturedAdmissionReplayObservesNewPublicationWithoutRecapture)
    {
        ASSERT_EQ(
            cudaStreamBeginCapture(
                inference_stream_, cudaStreamCaptureModeGlobal),
            cudaSuccess);
        ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[0],
            &device_statuses_[0]));
        ASSERT_EQ(cudaMemcpyAsync(
                      device_evidence_,
                      &device_tickets_[0],
                      sizeof(*device_evidence_),
                      cudaMemcpyDeviceToDevice,
                      inference_stream_),
                  cudaSuccess);
        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[0],
            &device_statuses_[1]));
        ASSERT_EQ(cudaStreamEndCapture(inference_stream_, &graph_), cudaSuccess);
        ASSERT_NE(graph_, nullptr);
        ASSERT_EQ(
            cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0),
            cudaSuccess);

        DeviceMoEOverlayEpochTicket evidence{};
        ASSERT_EQ(cudaGraphLaunch(graph_exec_, inference_stream_), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      &evidence,
                      device_evidence_,
                      sizeof(evidence),
                      cudaMemcpyDeviceToHost,
                      inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(inference_stream_), cudaSuccess);
        ASSERT_TRUE(evidence.valid());
        EXPECT_EQ(evidence.epoch, 1u);
        EXPECT_EQ(evidence.bank(), 0u);

        ASSERT_TRUE(kernel_->reserveMoEOverlayEpochCandidate(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[0],
            &device_statuses_[2]));
        ASSERT_TRUE(kernel_->markMoEOverlayEpochCandidateReady(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[0],
            &device_statuses_[3]));
        ASSERT_TRUE(kernel_->publishMoEOverlayEpochCandidate(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[0],
            &device_statuses_[4]));
        ASSERT_EQ(cudaStreamSynchronize(maintenance_stream_), cudaSuccess);

        evidence = {};
        ASSERT_EQ(cudaGraphLaunch(graph_exec_, inference_stream_), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      &evidence,
                      device_evidence_,
                      sizeof(evidence),
                      cudaMemcpyDeviceToHost,
                      inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(inference_stream_), cudaSuccess);
        ASSERT_TRUE(evidence.valid());
        EXPECT_EQ(evidence.epoch, 2u);
        EXPECT_EQ(evidence.bank(), 1u);
        EXPECT_EQ(evidence.generation(), 2u);

        ASSERT_TRUE(kernel_->retireMoEOverlayEpoch(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[1],
            &device_statuses_[5]));
        DeviceMoEOverlayEpochStatus retire_status{};
        ASSERT_EQ(cudaMemcpyAsync(
                      &retire_status,
                      &device_statuses_[5],
                      sizeof(retire_status),
                      cudaMemcpyDeviceToHost,
                      maintenance_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(maintenance_stream_), cudaSuccess);
        EXPECT_TRUE(retire_status.succeeded());
    }

    TEST_F(CUDAMoEOverlayEpochTest,
           CapturedSidecarRoutingSelectsCanonicalTicketBankAcrossPublication)
    {
        ASSERT_EQ(
            cudaStreamBeginCapture(
                inference_stream_, cudaStreamCaptureModeGlobal),
            cudaSuccess);
        ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[0],
            &device_statuses_[0]));
        ASSERT_TRUE(kernel_->assignPrefillRoutesLeastLoadedResident(
            inferenceLaunch(),
            device_runtime_,
            /*current_tokens=*/1,
            /*max_tokens=*/1,
            /*num_experts=*/1,
            /*top_k=*/1,
            device_position_,
            device_active_rows_));
        ASSERT_EQ(cudaMemcpyAsync(
                      device_route_evidence_,
                      device_route_participant_,
                      sizeof(*device_route_evidence_),
                      cudaMemcpyDeviceToDevice,
                      inference_stream_),
                  cudaSuccess);
        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[0],
            &device_statuses_[1]));
        ASSERT_EQ(cudaStreamEndCapture(inference_stream_, &graph_), cudaSuccess);
        ASSERT_NE(graph_, nullptr);
        ASSERT_EQ(
            cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0),
            cudaSuccess);

        int32_t routed_participant = -1;
        ASSERT_EQ(cudaGraphLaunch(graph_exec_, inference_stream_), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      &routed_participant,
                      device_route_evidence_,
                      sizeof(routed_participant),
                      cudaMemcpyDeviceToHost,
                      inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(inference_stream_), cudaSuccess);
        EXPECT_EQ(routed_participant, 0)
            << "epoch one must ignore mutable active_bank=1 and empty sidecar banks";

        ASSERT_TRUE(kernel_->reserveMoEOverlayEpochCandidate(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[0],
            &device_statuses_[2]));
        ASSERT_TRUE(kernel_->markMoEOverlayEpochCandidateReady(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[0],
            &device_statuses_[3]));
        ASSERT_TRUE(kernel_->publishMoEOverlayEpochCandidate(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[0],
            &device_statuses_[4]));
        ASSERT_EQ(cudaStreamSynchronize(maintenance_stream_), cudaSuccess);

        routed_participant = -1;
        ASSERT_EQ(cudaGraphLaunch(graph_exec_, inference_stream_), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      &routed_participant,
                      device_route_evidence_,
                      sizeof(routed_participant),
                      cudaMemcpyDeviceToHost,
                      inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(inference_stream_), cudaSuccess);
        EXPECT_EQ(routed_participant, 1)
            << "the same graph must acquire epoch two and select bank one";
    }

    /**
     * @brief Prove request-local LLEP moves placement in a private child bank.
     *
     * The captured graph acquires canonical epoch one, applies a source-side
     * ownership move to the LLEP child, consumes that child placement in the
     * ordinary resident router, and releases the same ticket. The parent bank
     * must remain immutable throughout: durable publication is a separate
     * maintenance transaction owned by the ExpertOverlay controller.
     */
    TEST_F(CUDAMoEOverlayEpochTest,
           CapturedCurrentBatchLLEPPublishesPrivateChildWithoutMutatingParent)
    {
        DeviceMoERebalancePlanEntry plan{};
        plan.op = static_cast<std::uint32_t>(
            DeviceMoERebalancePlanOp::OwnershipTransfer);
        plan.layer = 0u;
        plan.expert = 0u;
        plan.source_participant = 0u;
        plan.destination_participant = 1u;
        plan.destination_overlay_participant = 1;
        plan.source_resident_mask = 0b01u;
        plan.flags = moe_rebalance_abi::kPlanFlagCurrentBatchLLEP;

        DeviceMoERebalanceCommandBufferHeader command_header{};
        command_header.epoch = 1u;
        command_header.command_count = 1u;
        command_header.command_capacity = 1u;
        command_header.participant_id = 0u;
        command_header.participant_count = 2u;

        DeviceMoERebalanceConfig config{};
        config.num_layers = 1u;
        config.num_experts = 1u;
        config.top_k = 1u;
        config.participant_id = 0u;
        config.participant_count = 2u;
        config.window_size_tokens = 1u;

        ASSERT_EQ(cudaMemcpyAsync(
                      device_plan_entry_, &plan, sizeof(plan),
                      cudaMemcpyHostToDevice, inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      device_command_header_, &command_header,
                      sizeof(command_header), cudaMemcpyHostToDevice,
                      inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(inference_stream_), cudaSuccess);

        ASSERT_EQ(cudaStreamBeginCapture(
                      inference_stream_, cudaStreamCaptureModeGlobal),
                  cudaSuccess);
        ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
            inferenceLaunch(), device_control_, &device_tickets_[0],
            &device_statuses_[0]));
        ASSERT_TRUE(kernel_->applyDeviceRebalanceArrivals(
            inferenceLaunch(), device_runtime_, device_plan_entry_,
            /*plan_count=*/nullptr, /*plan_capacity=*/1u,
            device_transfer_slot_, /*local_transfer_slot_count=*/1u,
            config, device_apply_status_,
            device_command_header_));
        ASSERT_TRUE(kernel_->assignPrefillRoutesLeastLoadedResident(
            inferenceLaunch(), device_runtime_, /*current_tokens=*/1,
            /*max_tokens=*/1, /*num_experts=*/1, /*top_k=*/1,
            device_position_, device_active_rows_));
        ASSERT_EQ(cudaMemcpyAsync(
                      device_route_evidence_, device_route_participant_,
                      sizeof(*device_route_evidence_), cudaMemcpyDeviceToDevice,
                      inference_stream_),
                  cudaSuccess);
        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(), device_control_, &device_tickets_[0],
            &device_statuses_[1]));
        ASSERT_EQ(cudaStreamEndCapture(inference_stream_, &graph_), cudaSuccess);
        ASSERT_NE(graph_, nullptr);
        ASSERT_EQ(cudaGraphInstantiate(
                      &graph_exec_, graph_, nullptr, nullptr, 0),
                  cudaSuccess);

        ASSERT_EQ(cudaGraphLaunch(graph_exec_, inference_stream_), cudaSuccess);
        DeviceMoELayerRuntime runtime{};
        DeviceMoERebalanceApplyStatus apply_status{};
        std::array<DeviceMoEPlacementBank, 2> parent_banks{};
        int32_t routed_participant = -1;
        ASSERT_EQ(cudaMemcpyAsync(
                      &runtime, device_runtime_, sizeof(runtime),
                      cudaMemcpyDeviceToHost, inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      &apply_status, device_apply_status_, sizeof(apply_status),
                      cudaMemcpyDeviceToHost, inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      parent_banks.data(), device_placement_banks_,
                      sizeof(parent_banks), cudaMemcpyDeviceToHost,
                      inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      &routed_participant, device_route_evidence_,
                      sizeof(routed_participant), cudaMemcpyDeviceToHost,
                      inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(inference_stream_), cudaSuccess);

        ASSERT_EQ(apply_status.status_code, static_cast<std::uint32_t>(
            DeviceMoERebalanceApplyStatusCode::Ok));
        EXPECT_EQ(apply_status.plan_entries_seen, 1u);
        EXPECT_EQ(apply_status.invalid_plan_entries, 0u);
        EXPECT_EQ(apply_status.changed_layers, 1u);
        EXPECT_EQ(apply_status.applied_arrivals, 0u)
            << "the source retires ownership; only the destination copies bytes";
        ASSERT_EQ(runtime.current_batch_llep_transient_bank_active, 1u);
        ASSERT_EQ(runtime.current_batch_llep_movement_observed, 1u);
        ASSERT_LE(runtime.active_bank, 1u);
        EXPECT_EQ(runtime.active_epoch, 1u);
        const auto &child_bank = runtime.banks[runtime.active_bank];
        EXPECT_EQ(child_bank.epoch, 1u);
        EXPECT_EQ(child_bank.experts[0].owner_participant, 1);
        EXPECT_EQ(child_bank.resident_participant_mask[0], 0b10u);
        EXPECT_EQ(child_bank.local_compute_mask[0], 0u);
        EXPECT_EQ(routed_participant, 1)
            << "the post-apply router must consume the private child bank";

        EXPECT_EQ(parent_banks[0].epoch, 1u);
        EXPECT_EQ(parent_banks[0].experts[0].owner_participant, 0);
        EXPECT_EQ(parent_banks[0].resident_participant_mask[0], 0b01u)
            << "CurrentBatch LLEP must not mutate canonical durable placement";
        EXPECT_EQ(parent_banks[1].epoch, 2u);
        EXPECT_EQ(parent_banks[1].experts[0].owner_participant, 1);
        EXPECT_EQ(parent_banks[1].resident_participant_mask[0], 0b10u);
    }

    /**
     * @brief Apply durable maintenance after inference releases its epoch ticket.
     *
     * Production maintenance reserves the unpublished bank only after the
     * request reader has released its ticket. The reservation status is then
     * the authority for the published source bank and candidate destination;
     * consulting the deliberately invalid old ticket would reject every plan
     * as InvalidRuntime. This is the exact ordering used by the retained
     * Dynamic maintenance graph before an MTP parent takes over execution.
     */
    TEST_F(CUDAMoEOverlayEpochTest,
           DurableMaintenanceUsesReservationAfterInferenceTicketRelease)
    {
        DeviceMoELayerRuntime runtime{};
        runtime.active_bank = 0u;
        runtime.active_epoch = 1u;
        runtime.expert_count = 1u;
        runtime.top_k = 1u;
        runtime.participant_id = 0u;
        runtime.participant_count = 2u;
        runtime.overlay_epoch_ticket = &device_tickets_[0];
        runtime.banks[0].epoch = 1u;
        runtime.banks[0].expert_count = 1u;
        runtime.banks[0].experts[0].owner_participant = 0;
        runtime.banks[0].resident_participant_mask[0] = 0b01u;
        runtime.banks[0].local_compute_mask[0] = 1u;

        DeviceMoERebalancePlanEntry plan{};
        plan.op = static_cast<std::uint32_t>(
            DeviceMoERebalancePlanOp::OwnershipTransfer);
        plan.layer = 0u;
        plan.expert = 0u;
        plan.source_participant = 0u;
        plan.destination_participant = 1u;
        plan.destination_overlay_participant = 1;
        plan.source_resident_mask = 0b01u;

        DeviceMoERebalanceCommandBufferHeader command_header{};
        command_header.epoch = 1u;
        command_header.command_count = 1u;
        command_header.command_capacity = 1u;
        command_header.participant_id = 0u;
        command_header.participant_count = 2u;

        DeviceMoERebalanceGraphControllerState controller{};
        controller.participant_count = 2u;
        controller.waves[0].epoch = 1u;
        controller.waves[0].state = static_cast<std::uint32_t>(
            DeviceMoERebalanceWaveLifecycle::ReadyToApply);
        controller.waves[0].planned_start_layer = 0u;
        controller.waves[0].planned_layer_count = 1u;
        controller.waves[0].command_count = 1u;

        DeviceMoERebalanceConfig config{};
        config.num_layers = 1u;
        config.num_experts = 1u;
        config.top_k = 1u;
        config.participant_id = 0u;
        config.participant_count = 2u;
        config.window_size_tokens = 1u;

        ASSERT_EQ(cudaMemcpyAsync(
                      device_runtime_, &runtime, sizeof(runtime),
                      cudaMemcpyHostToDevice, maintenance_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      device_plan_entry_, &plan, sizeof(plan),
                      cudaMemcpyHostToDevice, maintenance_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      device_command_header_, &command_header,
                      sizeof(command_header), cudaMemcpyHostToDevice,
                      maintenance_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      device_controller_, &controller, sizeof(controller),
                      cudaMemcpyHostToDevice, maintenance_stream_),
                  cudaSuccess);

        ASSERT_TRUE(kernel_->reserveMoEOverlayEpochCandidate(
            maintenanceLaunch(), device_control_, &device_epochs_[0],
            &device_statuses_[2]));
        ASSERT_TRUE(kernel_->applyReadyDeviceRebalanceWave(
            maintenanceLaunch(), device_runtime_, device_plan_entry_,
            /*plan_count=*/nullptr, /*plan_capacity=*/1u,
            device_transfer_slot_, /*local_transfer_slot_count=*/1u,
            config, device_apply_status_, device_controller_,
            device_command_header_, /*target_layer=*/-1,
            /*command_buffer_count=*/1u, &device_statuses_[2]));

        DeviceMoEOverlayEpochStatus reservation_status{};
        DeviceMoEOverlayEpochTicket released_ticket{};
        DeviceMoERebalanceApplyStatus apply_status{};
        ASSERT_EQ(cudaMemcpyAsync(
                      &reservation_status, &device_statuses_[2],
                      sizeof(reservation_status), cudaMemcpyDeviceToHost,
                      maintenance_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      &released_ticket, &device_tickets_[0],
                      sizeof(released_ticket), cudaMemcpyDeviceToHost,
                      maintenance_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      &apply_status, device_apply_status_, sizeof(apply_status),
                      cudaMemcpyDeviceToHost, maintenance_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      &runtime, device_runtime_, sizeof(runtime),
                      cudaMemcpyDeviceToHost, maintenance_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      &controller, device_controller_, sizeof(controller),
                      cudaMemcpyDeviceToHost, maintenance_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(maintenance_stream_), cudaSuccess);

        ASSERT_TRUE(reservation_status.succeeded());
        EXPECT_EQ(reservation_status.typedOperation(),
                  DeviceMoEOverlayEpochOperation::ReserveCandidate);
        ASSERT_EQ(apply_status.status_code, static_cast<std::uint32_t>(
            DeviceMoERebalanceApplyStatusCode::Ok));
        EXPECT_EQ(apply_status.invalid_plan_entries, 0u);
        EXPECT_EQ(apply_status.plan_entries_seen, 1u);
        EXPECT_EQ(apply_status.changed_layers, 1u);
        EXPECT_EQ(controller.last_error_code, 0u);
        EXPECT_EQ(controller.decode_apply_hits, 1u);
        // Preparing the peer bank has not committed placement. The exact
        // commands must survive until the finalizer publishes the selector.
        EXPECT_EQ(controller.waves[0].state, static_cast<std::uint32_t>(
            DeviceMoERebalanceWaveLifecycle::PreparedForPublication));
        ASSERT_EQ(cudaMemcpyAsync(
                      &command_header, device_command_header_,
                      sizeof(command_header), cudaMemcpyDeviceToHost,
                      maintenance_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(maintenance_stream_), cudaSuccess);
        EXPECT_EQ(command_header.epoch, 1u);
        EXPECT_EQ(command_header.command_count, 1u);

        /* Apply prepares only the reserved bank. Publication is a separate
         * all-layer transaction, so active placement must remain at epoch one. */
        EXPECT_FALSE(released_ticket.valid());
        EXPECT_EQ(runtime.active_bank, 0u);
        EXPECT_EQ(runtime.active_epoch, 1u);
        EXPECT_EQ(runtime.banks[1].epoch, 2u);
        EXPECT_EQ(runtime.banks[1].experts[0].owner_participant, 1);
        EXPECT_EQ(runtime.banks[1].resident_participant_mask[0], 0b10u);
        ASSERT_TRUE(moe_rebalance_publication::preparedWaveMatches(
            &controller, &command_header, 1u, &apply_status));
        ASSERT_TRUE(kernel_->finalizeMoEOverlayRebalancePublication(
            maintenanceLaunch(), device_runtime_, 1u, 1u, device_control_,
            &device_epochs_[0], &device_statuses_[2], device_apply_status_,
            device_controller_, device_command_header_, 1u,
            device_plan_entry_, 2u, 4096u, journalView()));
        ASSERT_EQ(cudaMemcpyAsync(&controller, device_controller_, sizeof(controller),
                                   cudaMemcpyDeviceToHost, maintenance_stream_), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(&command_header, device_command_header_, sizeof(command_header),
                                   cudaMemcpyDeviceToHost, maintenance_stream_), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(&reservation_status, &device_statuses_[2], sizeof(reservation_status),
                                   cudaMemcpyDeviceToHost, maintenance_stream_), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(maintenance_stream_), cudaSuccess);
        ASSERT_TRUE(reservation_status.succeeded());
        EXPECT_EQ(reservation_status.typedOperation(), DeviceMoEOverlayEpochOperation::PublishCandidate);
        EXPECT_EQ(controller.waves[0].state,
                  static_cast<std::uint32_t>(DeviceMoERebalanceWaveLifecycle::Applied));
        EXPECT_EQ(command_header.epoch, 0u);
        EXPECT_EQ(command_header.command_count, 0u);
    }

    TEST_F(CUDAMoEOverlayEpochTest,
           CapturedAllLayerFinalizeAbortsNoWorkThenPublishesTwentyEpochs)
    {
        DeviceMoERebalanceApplyStatus apply_status{};
        auto upload_family = [&](std::uint64_t candidate_epoch,
                                 std::uint32_t published_bank,
                                 std::int32_t published_owner,
                                 bool prepare_changed_layer)
        {
            // This fixture supplies the same immutable apply identity as the
            // real apply kernel. Finalization must reject unrelated headers.
            DeviceMoERebalanceGraphControllerState controller{};
            controller.participant_count = 2u;
            auto &wave = controller.waves[0];
            wave.epoch = static_cast<std::uint32_t>(candidate_epoch);
            wave.state = static_cast<std::uint32_t>(
                DeviceMoERebalanceWaveLifecycle::PreparedForPublication);
            wave.command_count = 1u;
            wave.planned_layer_count = 2u;
            wave.applied_layer_count = 2u;
            DeviceMoERebalanceCommandBufferHeader header{};
            header.participant_count = 2u;
            header.epoch = wave.epoch;
            header.command_count = header.command_capacity = 1u;
            apply_status.transaction_wave_index = 0u;
            apply_status.transaction_epoch = wave.epoch;
            apply_status.transaction_command_count = 1u;
            ASSERT_EQ(cudaMemcpyAsync(device_controller_, &controller, sizeof(controller),
                                       cudaMemcpyHostToDevice, maintenance_stream_), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(device_command_header_, &header, sizeof(header),
                                       cudaMemcpyHostToDevice, maintenance_stream_), cudaSuccess);
            const std::uint32_t candidate_bank = 1u - published_bank;
            std::array<DeviceMoELayerRuntime, 2> family{};
            for (std::size_t layer = 0; layer < family.size(); ++layer)
            {
                auto &runtime = family[layer];
                runtime.active_bank = published_bank;
                runtime.active_epoch =
                    static_cast<std::uint32_t>(candidate_epoch - 1u);
                runtime.expert_count = 1u;
                runtime.top_k = 1u;
                runtime.participant_id = 0u;
                runtime.participant_count = 2u;
                auto &published = runtime.banks[published_bank];
                published.epoch =
                    static_cast<std::uint32_t>(candidate_epoch - 1u);
                published.expert_count = 1u;
                published.experts[0].owner_participant =
                    layer == 0u ? published_owner : 0;
                published.resident_participant_mask[0] =
                    layer == 0u && published_owner == 1 ? 0b10u : 0b01u;
                runtime.banks[candidate_bank].epoch =
                    candidate_epoch > 2u
                        ? static_cast<std::uint32_t>(candidate_epoch - 2u)
                        : 0u;
            }
            if (prepare_changed_layer)
            {
                auto &candidate = family[0].banks[candidate_bank];
                candidate = family[0].banks[published_bank];
                candidate.epoch = static_cast<std::uint32_t>(candidate_epoch);
                candidate.experts[0].owner_participant =
                    static_cast<std::int32_t>(candidate_epoch & 1u);
                candidate.resident_participant_mask[0] =
                    (candidate_epoch & 1u) != 0u ? 0b10u : 0b01u;
            }
            ASSERT_EQ(cudaMemcpyAsync(
                          device_runtime_family_, family.data(),
                          sizeof(family), cudaMemcpyHostToDevice,
                          maintenance_stream_),
                      cudaSuccess);
        };

        upload_family(/*candidate_epoch=*/2u, /*published_bank=*/0u,
                      /*published_owner=*/0, /*prepare_changed_layer=*/false);
        apply_status.changed_layers = 0u;
        ASSERT_EQ(cudaMemcpyAsync(
                      device_apply_status_, &apply_status,
                      sizeof(apply_status), cudaMemcpyHostToDevice,
                      maintenance_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(maintenance_stream_), cudaSuccess);

        ASSERT_EQ(cudaStreamBeginCapture(
                      maintenance_stream_, cudaStreamCaptureModeGlobal),
                  cudaSuccess);
        ASSERT_TRUE(kernel_->reserveMoEOverlayEpochCandidate(
            maintenanceLaunch(), device_control_, &device_epochs_[0],
            &device_statuses_[2]));
        ASSERT_TRUE(kernel_->finalizeMoEOverlayRebalancePublication(
            maintenanceLaunch(), device_runtime_family_,
            /*layer_count=*/2u, /*expert_count=*/1u,
            device_control_, &device_epochs_[0], &device_statuses_[2],
            device_apply_status_, device_controller_, device_command_header_, 1u,
            device_plan_entry_, 2u, 4096u, journalView()));
        ASSERT_EQ(cudaStreamEndCapture(maintenance_stream_, &graph_), cudaSuccess);
        ASSERT_NE(graph_, nullptr);
        ASSERT_EQ(cudaGraphInstantiate(
                      &graph_exec_, graph_, nullptr, nullptr, 0),
                  cudaSuccess);

        ASSERT_EQ(cudaGraphLaunch(graph_exec_, maintenance_stream_), cudaSuccess);
        DeviceMoEOverlayEpochControl control{};
        DeviceMoEOverlayEpochStatus publication_status{};
        std::uint64_t next_epoch = 0u;
        ASSERT_EQ(cudaMemcpyAsync(
                      &control, device_control_, sizeof(control),
                      cudaMemcpyDeviceToHost, maintenance_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      &publication_status, &device_statuses_[2],
                      sizeof(publication_status), cudaMemcpyDeviceToHost,
                      maintenance_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      &next_epoch, &device_epochs_[0], sizeof(next_epoch),
                      cudaMemcpyDeviceToHost, maintenance_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(maintenance_stream_), cudaSuccess);
        EXPECT_TRUE(publication_status.succeeded());
        EXPECT_EQ(publication_status.typedOperation(),
                  DeviceMoEOverlayEpochOperation::AbortCandidate);
        EXPECT_EQ(control.published_selector,
                  deviceMoEOverlayEpochSelector(1u, 0u));
        EXPECT_EQ(control.bank_states[1], static_cast<std::uint32_t>(
                                              DeviceMoEOverlayEpochBankState::Empty));
        EXPECT_EQ(next_epoch, 2u);

        std::uint32_t published_bank = 0u;
        std::int32_t published_owner = 0;
        for (std::uint64_t epoch = 2u; epoch < 22u; ++epoch)
        {
            upload_family(epoch, published_bank, published_owner,
                          /*prepare_changed_layer=*/true);
            apply_status.changed_layers = 1u;
            ASSERT_EQ(cudaMemcpyAsync(
                          device_apply_status_, &apply_status,
                          sizeof(apply_status), cudaMemcpyHostToDevice,
                          maintenance_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaGraphLaunch(graph_exec_, maintenance_stream_), cudaSuccess);

            std::array<DeviceMoELayerRuntime, 2> observed_family{};
            ASSERT_EQ(cudaMemcpyAsync(
                          observed_family.data(), device_runtime_family_,
                          sizeof(observed_family), cudaMemcpyDeviceToHost,
                          maintenance_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(
                          &control, device_control_, sizeof(control),
                          cudaMemcpyDeviceToHost, maintenance_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(
                          &publication_status, &device_statuses_[2],
                          sizeof(publication_status), cudaMemcpyDeviceToHost,
                          maintenance_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(
                          &next_epoch, &device_epochs_[0], sizeof(next_epoch),
                          cudaMemcpyDeviceToHost, maintenance_stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(maintenance_stream_), cudaSuccess);

            const std::uint32_t candidate_bank = 1u - published_bank;
            ASSERT_TRUE(publication_status.succeeded()) << "epoch=" << epoch;
            EXPECT_EQ(publication_status.typedOperation(),
                      DeviceMoEOverlayEpochOperation::PublishCandidate);
            EXPECT_EQ(publication_status.epoch, epoch);
            EXPECT_EQ(control.published_selector,
                      deviceMoEOverlayEpochSelector(epoch, candidate_bank));
            EXPECT_EQ(control.bank_states[candidate_bank],
                      static_cast<std::uint32_t>(
                          DeviceMoEOverlayEpochBankState::Published));
            EXPECT_EQ(control.bank_states[published_bank],
                      static_cast<std::uint32_t>(
                          DeviceMoEOverlayEpochBankState::Retiring));
            for (const auto &runtime : observed_family)
            {
                EXPECT_EQ(runtime.active_bank, candidate_bank);
                EXPECT_EQ(runtime.active_epoch, epoch);
                EXPECT_EQ(runtime.banks[candidate_bank].epoch, epoch);
            }
            EXPECT_EQ(observed_family[0].banks[candidate_bank]
                          .experts[0].owner_participant,
                      static_cast<std::int32_t>(epoch & 1u));
            EXPECT_EQ(observed_family[1].banks[candidate_bank]
                          .experts[0].owner_participant,
                      0) << "untouched layer must be cloned from published bank";
            EXPECT_EQ(next_epoch, epoch + 1u);
            published_bank = candidate_bank;
            published_owner = static_cast<std::int32_t>(epoch & 1u);
        }

        // A stale apply receipt must not publish or recycle the prepared wave.
        // This uses the same retained graph, so the identity is device data,
        // not a host choice of another execution path.
        upload_family(22u, published_bank, published_owner, true);
        ++apply_status.transaction_epoch;
        ASSERT_EQ(cudaMemcpyAsync(device_apply_status_, &apply_status, sizeof(apply_status),
                                   cudaMemcpyHostToDevice, maintenance_stream_), cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(graph_exec_, maintenance_stream_), cudaSuccess);
        DeviceMoERebalanceGraphControllerState rejected_controller{};
        DeviceMoERebalanceCommandBufferHeader retained_header{};
        ASSERT_EQ(cudaMemcpyAsync(&rejected_controller, device_controller_, sizeof(rejected_controller),
                                   cudaMemcpyDeviceToHost, maintenance_stream_), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(&retained_header, device_command_header_, sizeof(retained_header),
                                   cudaMemcpyDeviceToHost, maintenance_stream_), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(&control, device_control_, sizeof(control),
                                   cudaMemcpyDeviceToHost, maintenance_stream_), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(&publication_status, &device_statuses_[2], sizeof(publication_status),
                                   cudaMemcpyDeviceToHost, maintenance_stream_), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(maintenance_stream_), cudaSuccess);
        EXPECT_EQ(publication_status.code,
                  static_cast<std::uint32_t>(DeviceMoEOverlayEpochStatusCode::InvalidControl));
        EXPECT_EQ(control.published_selector, deviceMoEOverlayEpochSelector(21u, published_bank));
        EXPECT_EQ(retained_header.epoch, 22u);
        EXPECT_EQ(retained_header.command_count, 1u);
        EXPECT_NE(rejected_controller.last_error_code, 0u);
        EXPECT_EQ(rejected_controller.waves[0].state, static_cast<std::uint32_t>(
            DeviceMoERebalanceWaveLifecycle::Error));

        DeviceMoELayerRuntime sidecar{};
        sidecar.active_bank = 1u - published_bank;
        sidecar.active_epoch = 1u;
        sidecar.expert_count = 1u;
        sidecar.top_k = 1u;
        sidecar.participant_id = 0u;
        sidecar.participant_count = 2u;
        sidecar.prefill_route_capacity = 1u;
        sidecar.route_expert_ids = device_route_expert_;
        sidecar.route_participant_ids = device_route_participant_;
        sidecar.overlay_epoch_ticket = &device_tickets_[0];
        sidecar.overlay_placement_banks =
            &device_runtime_family_[0].banks[0];
        const std::int32_t unset_participant = -1;
        ASSERT_EQ(cudaMemcpyAsync(
                      device_runtime_, &sidecar, sizeof(sidecar),
                      cudaMemcpyHostToDevice, inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      device_route_participant_, &unset_participant,
                      sizeof(unset_participant), cudaMemcpyHostToDevice,
                      inference_stream_),
                  cudaSuccess);
        ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
            inferenceLaunch(), device_control_, &device_tickets_[0],
            &device_statuses_[0]));
        ASSERT_TRUE(kernel_->assignPrefillRoutesLeastLoadedResident(
            inferenceLaunch(), device_runtime_, 1, 1, 1, 1,
            device_position_, device_active_rows_));
        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(), device_control_, &device_tickets_[0],
            &device_statuses_[1]));
        std::int32_t routed_participant = -1;
        ASSERT_EQ(cudaMemcpyAsync(
                      &routed_participant, device_route_participant_,
                      sizeof(routed_participant), cudaMemcpyDeviceToHost,
                      inference_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(inference_stream_), cudaSuccess);
        EXPECT_EQ(routed_participant, published_owner);
    }
/**
     * @brief Captured native commits retain exact paired receipts across reuse.
     *
     * This is a publication-protocol test: apply and copy records are explicit
     * inputs. The transfer-parity suites independently prove the actual payload
     * copy. No PerfStats data participates in the publication or assertions.
     */
    TEST_F(CUDAMoEOverlayEpochTest, NativeMovementJournalCapturedPublicationAndReset)
    {
        NativeMoEMovementArchive archive({.workspace_generation = 1, .layers = 2, .experts = 2,
            .wave_capacity = 20, .edge_capacity = 40,
            .participants = {{DeviceId::cuda(0), 0, 0}, {DeviceId::cuda(1), 0, 0}}});
        DeviceMoERebalanceGraphControllerState controller{};
        controller.participant_count = 2u;
        // Build one retained graph before supplying any wave. Its pointers and
        // capacity never change through successful, exhausted and invalid cases.
        ASSERT_EQ(cudaMemcpyAsync(device_controller_, &controller, sizeof(controller),
                                 cudaMemcpyHostToDevice, maintenance_stream_), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(maintenance_stream_), cudaSuccess);
        ASSERT_EQ(cudaStreamBeginCapture(maintenance_stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
        ASSERT_TRUE(kernel_->reserveMoEOverlayEpochCandidate(
            maintenanceLaunch(), device_control_, &device_epochs_[0], &device_statuses_[2]));
        ASSERT_TRUE(kernel_->finalizeMoEOverlayRebalancePublication(
            maintenanceLaunch(), device_runtime_family_, 2u, 2u, device_control_,
            &device_epochs_[0], &device_statuses_[2], device_apply_status_,
            device_controller_, device_command_header_, 1u,
            device_plan_entry_, 2u, 4096u, journalView()));
        ASSERT_EQ(cudaStreamEndCapture(maintenance_stream_, &graph_), cudaSuccess);
        ASSERT_EQ(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0), cudaSuccess);

        std::array<DeviceMoERebalanceMovementWave, 20> waves{};
        std::array<DeviceMoERebalanceMovementEdge, 40> edges{};
        DeviceMoEOverlayEpochControl control{};
        for (std::uint32_t replay = 0u; replay < 27u; ++replay)
        {
            SCOPED_TRACE(replay);
            const bool invalid = replay >= 22u;
            const std::uint32_t epoch = invalid ? 24u : replay + 2u;
            const auto bank = static_cast<std::uint32_t>((epoch - 2u) & 1u);
            if (replay == 21u)
            {
                // A new request clears only its small journal prefix. Retained
                // payload storage and model-lifetime selector epochs survive.
                controller.movement_journal = {};
            }
            controller.last_error_code = 0u;
            auto &wave = controller.waves[0];
            wave = {};
            wave.epoch = epoch;
            wave.state = static_cast<std::uint32_t>(
                DeviceMoERebalanceWaveLifecycle::PreparedForPublication);
            wave.planned_layer_count = wave.applied_layer_count = 2u;
            wave.command_count = wave.copied_arrivals = 2u;
            wave.copied_payload_bytes = 6144u;
            DeviceMoERebalanceCommandBufferHeader header{};
            header.participant_count = 2u;
            header.epoch = epoch;
            header.command_count = header.command_capacity = 2u;
            header.movement_decision.kind =
                DeviceMoERebalanceMovementDecisionKind::NativeOwnershipSpread;
            header.movement_decision.load = {
                .accepted_spread_improvement = 40, .pre_wave_spread = 100,
                .post_wave_spread = 60, .pre_wave_total = 200, .post_wave_total = 200,
                .pre_participant_spread = 0, .post_participant_spread = 20,
                .pre_participant_total = 200, .post_participant_total = 200,
                .requested_payload_slots = 1, .minimum_improvement_per_slot = 40,
                .maximum_post_spread_per_mille = 300, .ownership_swap_accepts = 1};
            std::array<DeviceMoERebalancePlanEntry, 2> plans{};
            for (std::uint32_t i = 0; i < 2u; ++i)
            {
                plans[i].op = static_cast<std::uint32_t>(DeviceMoERebalancePlanOp::OwnershipTransfer);
                plans[i].expert = i;
                plans[i].source_participant = i;
                plans[i].destination_participant = 1u - i;
                plans[i].activation_count = i == 0u ? 100u : 20u;
            }
            if (replay == 22u) wave.copied_payload_bytes = 0u;
            if (replay == 23u) ++header.movement_decision.load.post_wave_total;
            if (replay == 24u) plans[1].destination_participant = 1u;
            if (replay == 25u) header.movement_decision.kind =
                static_cast<DeviceMoERebalanceMovementDecisionKind>(UINT32_MAX);
            if (replay == 26u) plans[0].source_participant = 2u;

            DeviceMoERebalanceApplyStatus applied{};
            applied.changed_layers = 1u;
            applied.transaction_wave_index = 0u;
            applied.transaction_epoch = epoch;
            applied.transaction_command_count = 2u;
            std::array<DeviceMoELayerRuntime, 2> family{};
            for (auto &runtime : family)
            {
                runtime.active_bank = bank;
                runtime.active_epoch = epoch - 1u;
                runtime.expert_count = 2u;
                runtime.participant_count = 2u;
                auto &source = runtime.banks[bank];
                source.epoch = epoch - 1u;
                source.expert_count = 2u;
                source.experts[0].owner_participant = 0;
                source.experts[1].owner_participant = 1;
            }
            // Only one layer changed. The finalizer must clone the other layer
            // before publishing this same complete model-family generation.
            auto &candidate = family[0].banks[1u - bank];
            candidate = family[0].banks[bank];
            candidate.epoch = epoch;
            candidate.experts[0].owner_participant = 1;
            candidate.experts[1].owner_participant = 0;

            ASSERT_EQ(cudaMemcpyAsync(device_runtime_family_, family.data(), sizeof(family),
                                     cudaMemcpyHostToDevice, maintenance_stream_), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(device_controller_, &controller, sizeof(controller),
                                     cudaMemcpyHostToDevice, maintenance_stream_), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(device_command_header_, &header, sizeof(header),
                                     cudaMemcpyHostToDevice, maintenance_stream_), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(device_apply_status_, &applied, sizeof(applied),
                                     cudaMemcpyHostToDevice, maintenance_stream_), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(device_plan_entry_, plans.data(), sizeof(plans),
                                     cudaMemcpyHostToDevice, maintenance_stream_), cudaSuccess);
            ASSERT_EQ(cudaGraphLaunch(graph_exec_, maintenance_stream_), cudaSuccess);
            DeviceMoEOverlayEpochStatus publication{};
            ASSERT_EQ(cudaMemcpyAsync(&controller, device_controller_, sizeof(controller),
                                     cudaMemcpyDeviceToHost, maintenance_stream_), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(&header, device_command_header_, sizeof(header),
                                     cudaMemcpyDeviceToHost, maintenance_stream_), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(&control, device_control_, sizeof(control),
                                     cudaMemcpyDeviceToHost, maintenance_stream_), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(&publication, &device_statuses_[2], sizeof(publication),
                                     cudaMemcpyDeviceToHost, maintenance_stream_), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(waves.data(), device_movement_waves_, sizeof(waves),
                                     cudaMemcpyDeviceToHost, maintenance_stream_), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(edges.data(), device_movement_edges_, sizeof(edges),
                                     cudaMemcpyDeviceToHost, maintenance_stream_), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(maintenance_stream_), cudaSuccess);
            const auto &history = controller.movement_journal;
            // Feed actual captured publisher bytes through the production
            // terminal archive, with profiling neither consulted nor required.
            ASSERT_NO_THROW(archive.observe(replay >= 21u ? 2u : 1u, 1u, history,
                std::span(waves).first(history.committed_waves),
                std::span(edges).first(history.committed_edges)));
            EXPECT_EQ(archive.ledger().edges.size(), 2u * std::min(replay + 1u, 20u));
            EXPECT_EQ(archive.totals().physical_bytes,
                6144u * (replay >= 21u ? 21u : std::min(replay + 1u, 20u)));
            EXPECT_EQ(archive.ledger().complete(), replay < 20u);
            if (invalid)
            {
                EXPECT_FALSE(publication.succeeded());
                EXPECT_NE(controller.last_error_code, 0u);
                EXPECT_EQ(control.published_selector, deviceMoEOverlayEpochSelector(23u, 0u));
                EXPECT_EQ(history.committed_waves, 1u);
                EXPECT_EQ(history.last_candidate_epoch, 23u);
                EXPECT_EQ(header.command_count, 2u); // Failed commands are retained.
                continue;
            }
            ASSERT_TRUE(publication.succeeded());
            EXPECT_EQ(control.published_selector, deviceMoEOverlayEpochSelector(epoch, 1u - bank));
            EXPECT_EQ(header.command_count, 0u);
            EXPECT_EQ(header.movement_decision.kind, DeviceMoERebalanceMovementDecisionKind::None);
            EXPECT_EQ(history.last_candidate_epoch, epoch);
            EXPECT_EQ(history.committed_waves, replay == 21u ? 1u : std::min(replay + 1u, 20u));
            EXPECT_EQ(history.committed_edges, history.committed_waves * 2u);
            EXPECT_EQ(history.discarded_waves, replay == 20u ? 1u : 0u);
            EXPECT_EQ(history.discarded_edges, replay == 20u ? 2u : 0u);
            for (std::uint32_t i = 0; i < history.committed_waves; ++i)
            {
                EXPECT_EQ(waves[i].candidate_epoch, replay == 21u ? 23u : i + 2u);
                EXPECT_EQ(waves[i].first_edge, 2u * i);
                EXPECT_EQ(waves[i].edge_count, 2u);
                EXPECT_EQ(waves[i].physical_payload_bytes, 6144u);
                EXPECT_TRUE(waves[i].proof.valid());
                EXPECT_EQ(edges[2u * i].source_participant, 0u);
                EXPECT_EQ(edges[2u * i + 1u].source_participant, 1u);
                EXPECT_EQ(edges[2u * i].destination_participant, 1u);
                EXPECT_EQ(edges[2u * i + 1u].destination_participant, 0u);
                EXPECT_EQ(edges[2u * i].activation_count, 100u);
                EXPECT_EQ(edges[2u * i + 1u].activation_count, 20u);
                EXPECT_EQ(edges[2u * i].estimated_weight_bytes, 4096u);
            }
        }
    }

} // namespace llaminar2::test
