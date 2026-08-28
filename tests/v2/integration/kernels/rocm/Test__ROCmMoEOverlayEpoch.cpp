/**
 * @file Test__ROCmMoEOverlayEpoch.cpp
 * @brief Real-ROCm integration proof for captured ExpertOverlay epoch RCU.
 *
 * The suite mirrors CUDA exactly: a non-default inference stream holds the old
 * bank while a non-default maintenance stream publishes the next bank, and a
 * captured acquire/consume/release graph observes that publication on replay.
 */

#include "execution/moe/MoEOverlayDeviceEpochProtocol.h"
#include "execution/moe/DeviceMoERebalanceController.h"
#include "execution/moe/MoERuntimeTable.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"

#include <hip/hip_runtime.h>
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

        /** @return Whether at least one ROCm device is usable by this process. */
        bool hasROCmDevice()
        {
            int count = 0;
            return hipGetDeviceCount(&count) == hipSuccess && count > 0;
        }
    } // namespace

    /**
     * @brief Own exact HIP streams, events, and persistent epoch graph state.
     *
     * Tests enqueue the entire inter-stream dependency DAG before one terminal
     * observation.  No inference-side stream or device synchronization is used.
     */
    class ROCmMoEOverlayEpochTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if (!hasROCmDevice())
                GTEST_SKIP() << "No ROCm device available";
            ASSERT_EQ(hipSetDevice(0), hipSuccess);
            ASSERT_EQ(
                hipStreamCreateWithFlags(
                    &inference_stream_, hipStreamNonBlocking),
                hipSuccess);
            ASSERT_EQ(
                hipStreamCreateWithFlags(
                    &maintenance_stream_, hipStreamNonBlocking),
                hipSuccess);
            ASSERT_EQ(
                hipEventCreateWithFlags(
                    &inference_event_, hipEventDisableTiming),
                hipSuccess);
            ASSERT_EQ(
                hipEventCreateWithFlags(
                    &maintenance_event_, hipEventDisableTiming),
                hipSuccess);

            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_control_),
                         sizeof(*device_control_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_tickets_),
                         2u * sizeof(*device_tickets_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_statuses_),
                         kStatusCount * sizeof(*device_statuses_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_epochs_),
                         2u * sizeof(*device_epochs_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_evidence_),
                         sizeof(*device_evidence_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_runtime_),
                         sizeof(*device_runtime_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_runtime_family_),
                         2u * sizeof(*device_runtime_family_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_apply_status_),
                         sizeof(*device_apply_status_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_controller_),
                         sizeof(*device_controller_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_plan_entry_),
                         sizeof(*device_plan_entry_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_command_header_),
                         sizeof(*device_command_header_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_transfer_slot_),
                         sizeof(*device_transfer_slot_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_placement_banks_),
                         2u * sizeof(*device_placement_banks_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_route_expert_),
                         sizeof(*device_route_expert_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_route_participant_),
                         sizeof(*device_route_participant_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_position_),
                         sizeof(*device_position_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_active_rows_),
                         sizeof(*device_active_rows_)),
                      hipSuccess);
            ASSERT_EQ(hipMalloc(
                         reinterpret_cast<void **>(&device_route_evidence_),
                         sizeof(*device_route_evidence_)),
                      hipSuccess);

            DeviceMoEOverlayEpochControl initial_control{};
            MoEOverlayDeviceEpochProtocol::initialize(
                initial_control, /*initial_epoch=*/1u);
            const std::array<std::uint64_t, 2> epochs{2u, 1u};
            ASSERT_EQ(hipMemcpyAsync(
                         device_control_,
                         &initial_control,
                         sizeof(initial_control),
                         hipMemcpyHostToDevice,
                         inference_stream_),
                      hipSuccess);
            ASSERT_EQ(hipMemsetAsync(
                         device_tickets_,
                         0,
                         2u * sizeof(*device_tickets_),
                         inference_stream_),
                      hipSuccess);
            ASSERT_EQ(hipMemsetAsync(
                         device_statuses_,
                         0,
                         kStatusCount * sizeof(*device_statuses_),
                         inference_stream_),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(
                         device_epochs_,
                         epochs.data(),
                         sizeof(epochs),
                         hipMemcpyHostToDevice,
                         inference_stream_),
                      hipSuccess);
            ASSERT_EQ(hipMemsetAsync(
                         device_evidence_,
                         0,
                         sizeof(*device_evidence_),
                         inference_stream_),
                      hipSuccess);
            ASSERT_EQ(hipMemsetAsync(
                         device_transfer_slot_, 0,
                         sizeof(*device_transfer_slot_), inference_stream_),
                      hipSuccess);

            /* Canonical main-model bank zero owns epoch one and bank one owns
             * epoch two. The sidecar's embedded banks remain empty and its
             * mutable selector names bank one, so only the ticket plus stable
             * canonical placement pointer can select correctly. */
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
            ASSERT_EQ(hipMemcpyAsync(
                         device_runtime_, &runtime, sizeof(runtime),
                         hipMemcpyHostToDevice, inference_stream_),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(
                         device_placement_banks_, placement_banks.data(),
                         sizeof(placement_banks), hipMemcpyHostToDevice,
                         inference_stream_),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(
                         device_route_expert_, &route_expert,
                         sizeof(route_expert), hipMemcpyHostToDevice,
                         inference_stream_),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(
                         device_route_participant_, &route_participant,
                         sizeof(route_participant), hipMemcpyHostToDevice,
                         inference_stream_),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(
                         device_position_, &position, sizeof(position),
                         hipMemcpyHostToDevice, inference_stream_),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(
                         device_active_rows_, &active_rows,
                         sizeof(active_rows), hipMemcpyHostToDevice,
                         inference_stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(inference_stream_), hipSuccess);

            kernel_ = llaminar::v2::kernels::KernelFactory::createMoEKernel(
                DeviceId::rocm(0));
            ASSERT_NE(kernel_, nullptr);
        }

        void TearDown() override
        {
            if (!hasROCmDevice())
                return;
            (void)hipSetDevice(0);
            if (inference_stream_)
                (void)hipStreamSynchronize(inference_stream_);
            if (maintenance_stream_)
                (void)hipStreamSynchronize(maintenance_stream_);
            kernel_.reset();
            if (graph_exec_)
                (void)hipGraphExecDestroy(graph_exec_);
            if (graph_)
                (void)hipGraphDestroy(graph_);
            if (device_evidence_)
                (void)hipFree(device_evidence_);
            if (device_route_evidence_)
                (void)hipFree(device_route_evidence_);
            if (device_active_rows_)
                (void)hipFree(device_active_rows_);
            if (device_position_)
                (void)hipFree(device_position_);
            if (device_route_participant_)
                (void)hipFree(device_route_participant_);
            if (device_route_expert_)
                (void)hipFree(device_route_expert_);
            if (device_runtime_)
                (void)hipFree(device_runtime_);
            if (device_apply_status_)
                (void)hipFree(device_apply_status_);
            if (device_controller_)
                (void)hipFree(device_controller_);
            if (device_command_header_)
                (void)hipFree(device_command_header_);
            if (device_plan_entry_)
                (void)hipFree(device_plan_entry_);
            if (device_transfer_slot_)
                (void)hipFree(device_transfer_slot_);
            if (device_runtime_family_)
                (void)hipFree(device_runtime_family_);
            if (device_placement_banks_)
                (void)hipFree(device_placement_banks_);
            if (device_epochs_)
                (void)hipFree(device_epochs_);
            if (device_statuses_)
                (void)hipFree(device_statuses_);
            if (device_tickets_)
                (void)hipFree(device_tickets_);
            if (device_control_)
                (void)hipFree(device_control_);
            if (maintenance_event_)
                (void)hipEventDestroy(maintenance_event_);
            if (inference_event_)
                (void)hipEventDestroy(inference_event_);
            if (maintenance_stream_)
                (void)hipStreamDestroy(maintenance_stream_);
            if (inference_stream_)
                (void)hipStreamDestroy(inference_stream_);
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

        hipStream_t inference_stream_ = nullptr;
        hipStream_t maintenance_stream_ = nullptr;
        hipEvent_t inference_event_ = nullptr;
        hipEvent_t maintenance_event_ = nullptr;
        hipGraph_t graph_ = nullptr;
        hipGraphExec_t graph_exec_ = nullptr;
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
     * The HIP acquire must keep waiting after the captured signal value is
     * already visible, then accept only a descriptor whose digest belongs to a
     * generation newer than the endpoint's retained grant.
     */
    TEST_F(ROCmMoEOverlayEpochTest,
           PeerEpochAcquireAuthenticatesGenerationAndPinsPreparedCandidate)
    {
        MoEOverlayActivationEpochControl *host_control = nullptr;
        ASSERT_EQ(
            hipHostMalloc(
                reinterpret_cast<void **>(&host_control),
                sizeof(*host_control),
                hipHostMallocMapped),
            hipSuccess);
        auto host_owner = std::unique_ptr<
            MoEOverlayActivationEpochControl,
            void (*)(MoEOverlayActivationEpochControl *)>(
            host_control,
            [](MoEOverlayActivationEpochControl *pointer)
            {
                if (pointer)
                    (void)hipHostFree(pointer);
            });
        void *device_control_alias_raw = nullptr;
        ASSERT_EQ(
            hipHostGetDevicePointer(
                &device_control_alias_raw,
                host_control,
                0u),
            hipSuccess);
        auto *const device_activation_control =
            static_cast<MoEOverlayActivationEpochControl *>(
                device_control_alias_raw);

        MoEOverlayActivationDeviceEpochGrant *device_grant = nullptr;
        ASSERT_EQ(
            hipMalloc(
                reinterpret_cast<void **>(&device_grant),
                sizeof(*device_grant)),
            hipSuccess);
        auto grant_owner = std::unique_ptr<
            MoEOverlayActivationDeviceEpochGrant,
            void (*)(MoEOverlayActivationDeviceEpochGrant *)>(
            device_grant,
            [](MoEOverlayActivationDeviceEpochGrant *pointer)
            {
                if (pointer)
                    (void)hipFree(pointer);
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
            hipMemcpyAsync(
                device_grant,
                &grant,
                sizeof(grant),
                hipMemcpyHostToDevice,
                inference_stream_),
            hipSuccess);
        ASSERT_EQ(
            hipMemsetAsync(
                &device_tickets_[1],
                0,
                sizeof(device_tickets_[1]),
                inference_stream_),
            hipSuccess);

        /* Match CUDA's exact fan-out interleaving: E2 is globally prepared on
         * this participant while its local selector remains on E1. */
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
            hipEventRecord(maintenance_event_, maintenance_stream_),
            hipSuccess);
        ASSERT_EQ(
            hipStreamWaitEvent(inference_stream_, maintenance_event_, 0),
            hipSuccess);

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
        EXPECT_EQ(hipStreamQuery(inference_stream_), hipErrorNotReady)
            << "a reused timeline admitted the empty prior descriptor";
        (void)hipGetLastError();

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

        ASSERT_EQ(hipStreamSynchronize(inference_stream_), hipSuccess);
        DeviceMoEOverlayEpochTicket ticket{};
        DeviceMoEOverlayEpochStatus status{};
        ASSERT_EQ(
            hipMemcpy(
                &ticket,
                &device_tickets_[1],
                sizeof(ticket),
                hipMemcpyDeviceToHost),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpy(
                &status,
                &device_statuses_[8],
                sizeof(status),
                hipMemcpyDeviceToHost),
            hipSuccess);
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
        ASSERT_EQ(hipStreamSynchronize(inference_stream_), hipSuccess);
    }

    TEST_F(ROCmMoEOverlayEpochTest,
           IndependentStreamsPublishWithoutReusingHeldInferenceBank)
    {
        ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[0],
            &device_statuses_[0]));
        ASSERT_EQ(hipEventRecord(inference_event_, inference_stream_), hipSuccess);

        ASSERT_EQ(
            hipStreamWaitEvent(maintenance_stream_, inference_event_, 0),
            hipSuccess);
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
            hipEventRecord(maintenance_event_, maintenance_stream_),
            hipSuccess);

        ASSERT_EQ(
            hipStreamWaitEvent(inference_stream_, maintenance_event_, 0),
            hipSuccess);
        ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[1],
            &device_statuses_[4]));
        ASSERT_EQ(hipEventRecord(inference_event_, inference_stream_), hipSuccess);

        ASSERT_EQ(
            hipStreamWaitEvent(maintenance_stream_, inference_event_, 0),
            hipSuccess);
        ASSERT_TRUE(kernel_->retireMoEOverlayEpoch(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[1],
            &device_statuses_[5]));
        ASSERT_EQ(
            hipEventRecord(maintenance_event_, maintenance_stream_),
            hipSuccess);

        ASSERT_EQ(
            hipStreamWaitEvent(inference_stream_, maintenance_event_, 0),
            hipSuccess);
        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[0],
            &device_statuses_[6]));
        ASSERT_EQ(hipEventRecord(inference_event_, inference_stream_), hipSuccess);

        ASSERT_EQ(
            hipStreamWaitEvent(maintenance_stream_, inference_event_, 0),
            hipSuccess);
        ASSERT_TRUE(kernel_->retireMoEOverlayEpoch(
            maintenanceLaunch(),
            device_control_,
            &device_epochs_[1],
            &device_statuses_[7]));
        ASSERT_EQ(
            hipEventRecord(maintenance_event_, maintenance_stream_),
            hipSuccess);

        ASSERT_EQ(
            hipStreamWaitEvent(inference_stream_, maintenance_event_, 0),
            hipSuccess);
        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[1],
            &device_statuses_[8]));

        DeviceMoEOverlayEpochControl control{};
        std::array<DeviceMoEOverlayEpochTicket, 2> tickets{};
        std::array<DeviceMoEOverlayEpochStatus, kStatusCount> statuses{};
        ASSERT_EQ(hipMemcpyAsync(
                     &control,
                     device_control_,
                     sizeof(control),
                     hipMemcpyDeviceToHost,
                     inference_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     tickets.data(),
                     device_tickets_,
                     sizeof(tickets),
                     hipMemcpyDeviceToHost,
                     inference_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     statuses.data(),
                     device_statuses_,
                     sizeof(statuses),
                     hipMemcpyDeviceToHost,
                     inference_stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(inference_stream_), hipSuccess);

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

    TEST_F(ROCmMoEOverlayEpochTest,
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
            hipEventRecord(maintenance_event_, maintenance_stream_),
            hipSuccess);
        ASSERT_EQ(
            hipStreamWaitEvent(inference_stream_, maintenance_event_, 0),
            hipSuccess);

        // device_epochs_[1] remains one while the participant has locally
        // published two, so acquire must select the retiring old bank.
        ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[0],
            &device_statuses_[3],
            &device_epochs_[1]));
        DeviceMoEOverlayEpochTicket old_ticket{};
        ASSERT_EQ(
            hipMemcpyAsync(
                &old_ticket,
                &device_tickets_[0],
                sizeof(old_ticket),
                hipMemcpyDeviceToHost,
                inference_stream_),
            hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(inference_stream_), hipSuccess);
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
            hipMemcpyAsync(
                &device_epochs_[1],
                &admitted_epoch,
                sizeof(admitted_epoch),
                hipMemcpyHostToDevice,
                inference_stream_),
            hipSuccess);
        ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[1],
            &device_statuses_[5],
            &device_epochs_[1]));
        DeviceMoEOverlayEpochTicket new_ticket{};
        ASSERT_EQ(
            hipMemcpyAsync(
                &new_ticket,
                &device_tickets_[1],
                sizeof(new_ticket),
                hipMemcpyDeviceToHost,
                inference_stream_),
            hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(inference_stream_), hipSuccess);
        EXPECT_EQ(new_ticket.epoch, 2u);
        EXPECT_EQ(new_ticket.bank(), 1u);
        EXPECT_EQ(new_ticket.generation(), 2u);
        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[1],
            &device_statuses_[6]));
    }

    TEST_F(ROCmMoEOverlayEpochTest,
           CapturedAdmissionReplayObservesNewPublicationWithoutRecapture)
    {
        ASSERT_EQ(
            hipStreamBeginCapture(
                inference_stream_, hipStreamCaptureModeGlobal),
            hipSuccess);
        ASSERT_TRUE(kernel_->acquireMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[0],
            &device_statuses_[0]));
        ASSERT_EQ(hipMemcpyAsync(
                     device_evidence_,
                     &device_tickets_[0],
                     sizeof(*device_evidence_),
                     hipMemcpyDeviceToDevice,
                     inference_stream_),
                  hipSuccess);
        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[0],
            &device_statuses_[1]));
        ASSERT_EQ(hipStreamEndCapture(inference_stream_, &graph_), hipSuccess);
        ASSERT_NE(graph_, nullptr);
        ASSERT_EQ(
            hipGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0),
            hipSuccess);

        DeviceMoEOverlayEpochTicket evidence{};
        ASSERT_EQ(hipGraphLaunch(graph_exec_, inference_stream_), hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     &evidence,
                     device_evidence_,
                     sizeof(evidence),
                     hipMemcpyDeviceToHost,
                     inference_stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(inference_stream_), hipSuccess);
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
        ASSERT_EQ(hipStreamSynchronize(maintenance_stream_), hipSuccess);

        evidence = {};
        ASSERT_EQ(hipGraphLaunch(graph_exec_, inference_stream_), hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     &evidence,
                     device_evidence_,
                     sizeof(evidence),
                     hipMemcpyDeviceToHost,
                     inference_stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(inference_stream_), hipSuccess);
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
        ASSERT_EQ(hipMemcpyAsync(
                     &retire_status,
                     &device_statuses_[5],
                     sizeof(retire_status),
                     hipMemcpyDeviceToHost,
                     maintenance_stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(maintenance_stream_), hipSuccess);
        EXPECT_TRUE(retire_status.succeeded());
    }

    TEST_F(ROCmMoEOverlayEpochTest,
           CapturedSidecarRoutingSelectsCanonicalTicketBankAcrossPublication)
    {
        ASSERT_EQ(
            hipStreamBeginCapture(
                inference_stream_, hipStreamCaptureModeGlobal),
            hipSuccess);
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
        ASSERT_EQ(hipMemcpyAsync(
                     device_route_evidence_,
                     device_route_participant_,
                     sizeof(*device_route_evidence_),
                     hipMemcpyDeviceToDevice,
                     inference_stream_),
                  hipSuccess);
        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(),
            device_control_,
            &device_tickets_[0],
            &device_statuses_[1]));
        ASSERT_EQ(hipStreamEndCapture(inference_stream_, &graph_), hipSuccess);
        ASSERT_NE(graph_, nullptr);
        ASSERT_EQ(hipGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0),
                  hipSuccess);

        int32_t routed_participant = -1;
        ASSERT_EQ(hipGraphLaunch(graph_exec_, inference_stream_), hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     &routed_participant,
                     device_route_evidence_,
                     sizeof(routed_participant),
                     hipMemcpyDeviceToHost,
                     inference_stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(inference_stream_), hipSuccess);
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
        ASSERT_EQ(hipStreamSynchronize(maintenance_stream_), hipSuccess);

        routed_participant = -1;
        ASSERT_EQ(hipGraphLaunch(graph_exec_, inference_stream_), hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     &routed_participant,
                     device_route_evidence_,
                     sizeof(routed_participant),
                     hipMemcpyDeviceToHost,
                     inference_stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(inference_stream_), hipSuccess);
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
    TEST_F(ROCmMoEOverlayEpochTest,
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

        ASSERT_EQ(hipMemcpyAsync(
                     device_plan_entry_, &plan, sizeof(plan),
                     hipMemcpyHostToDevice, inference_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     device_command_header_, &command_header,
                     sizeof(command_header), hipMemcpyHostToDevice,
                     inference_stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(inference_stream_), hipSuccess);

        ASSERT_EQ(hipStreamBeginCapture(
                      inference_stream_, hipStreamCaptureModeGlobal),
                  hipSuccess);
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
        ASSERT_EQ(hipMemcpyAsync(
                     device_route_evidence_, device_route_participant_,
                     sizeof(*device_route_evidence_), hipMemcpyDeviceToDevice,
                     inference_stream_),
                  hipSuccess);
        ASSERT_TRUE(kernel_->releaseMoEOverlayEpoch(
            inferenceLaunch(), device_control_, &device_tickets_[0],
            &device_statuses_[1]));
        ASSERT_EQ(hipStreamEndCapture(inference_stream_, &graph_), hipSuccess);
        ASSERT_NE(graph_, nullptr);
        ASSERT_EQ(hipGraphInstantiate(
                      &graph_exec_, graph_, nullptr, nullptr, 0),
                  hipSuccess);

        ASSERT_EQ(hipGraphLaunch(graph_exec_, inference_stream_), hipSuccess);
        DeviceMoELayerRuntime runtime{};
        DeviceMoERebalanceApplyStatus apply_status{};
        std::array<DeviceMoEPlacementBank, 2> parent_banks{};
        int32_t routed_participant = -1;
        ASSERT_EQ(hipMemcpyAsync(
                     &runtime, device_runtime_, sizeof(runtime),
                     hipMemcpyDeviceToHost, inference_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     &apply_status, device_apply_status_, sizeof(apply_status),
                     hipMemcpyDeviceToHost, inference_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     parent_banks.data(), device_placement_banks_,
                     sizeof(parent_banks), hipMemcpyDeviceToHost,
                     inference_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     &routed_participant, device_route_evidence_,
                     sizeof(routed_participant), hipMemcpyDeviceToHost,
                     inference_stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(inference_stream_), hipSuccess);

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
     * The durable reservation identifies both the published source bank and
     * the unpublished destination bank. A released request ticket is invalid by
     * design at this boundary and must not be consulted by the apply kernel.
     * This mirrors CUDA and the production Dynamic maintenance transaction.
     */
    TEST_F(ROCmMoEOverlayEpochTest,
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

        ASSERT_EQ(hipMemcpyAsync(
                     device_runtime_, &runtime, sizeof(runtime),
                     hipMemcpyHostToDevice, maintenance_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     device_plan_entry_, &plan, sizeof(plan),
                     hipMemcpyHostToDevice, maintenance_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     device_command_header_, &command_header,
                     sizeof(command_header), hipMemcpyHostToDevice,
                     maintenance_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     device_controller_, &controller, sizeof(controller),
                     hipMemcpyHostToDevice, maintenance_stream_),
                  hipSuccess);

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
        ASSERT_EQ(hipMemcpyAsync(
                     &reservation_status, &device_statuses_[2],
                     sizeof(reservation_status), hipMemcpyDeviceToHost,
                     maintenance_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     &released_ticket, &device_tickets_[0],
                     sizeof(released_ticket), hipMemcpyDeviceToHost,
                     maintenance_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     &apply_status, device_apply_status_, sizeof(apply_status),
                     hipMemcpyDeviceToHost, maintenance_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     &runtime, device_runtime_, sizeof(runtime),
                     hipMemcpyDeviceToHost, maintenance_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     &controller, device_controller_, sizeof(controller),
                     hipMemcpyDeviceToHost, maintenance_stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(maintenance_stream_), hipSuccess);

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
        EXPECT_EQ(controller.waves[0].state, static_cast<std::uint32_t>(
            DeviceMoERebalanceWaveLifecycle::Applied));

        EXPECT_FALSE(released_ticket.valid());
        EXPECT_EQ(runtime.active_bank, 0u);
        EXPECT_EQ(runtime.active_epoch, 1u);
        EXPECT_EQ(runtime.banks[1].epoch, 2u);
        EXPECT_EQ(runtime.banks[1].experts[0].owner_participant, 1);
        EXPECT_EQ(runtime.banks[1].resident_participant_mask[0], 0b10u);
    }

    TEST_F(ROCmMoEOverlayEpochTest,
           CapturedAllLayerFinalizeAbortsNoWorkThenPublishesTwentyEpochs)
    {
        DeviceMoERebalanceApplyStatus apply_status{};
        auto upload_family = [&](std::uint64_t candidate_epoch,
                                 std::uint32_t published_bank,
                                 std::int32_t published_owner,
                                 bool prepare_changed_layer)
        {
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
            ASSERT_EQ(hipMemcpyAsync(
                         device_runtime_family_, family.data(),
                         sizeof(family), hipMemcpyHostToDevice,
                         maintenance_stream_),
                      hipSuccess);
        };

        upload_family(/*candidate_epoch=*/2u, /*published_bank=*/0u,
                      /*published_owner=*/0, /*prepare_changed_layer=*/false);
        apply_status.changed_layers = 0u;
        ASSERT_EQ(hipMemcpyAsync(
                     device_apply_status_, &apply_status,
                     sizeof(apply_status), hipMemcpyHostToDevice,
                     maintenance_stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(maintenance_stream_), hipSuccess);

        ASSERT_EQ(hipStreamBeginCapture(
                      maintenance_stream_, hipStreamCaptureModeGlobal),
                  hipSuccess);
        ASSERT_TRUE(kernel_->reserveMoEOverlayEpochCandidate(
            maintenanceLaunch(), device_control_, &device_epochs_[0],
            &device_statuses_[2]));
        ASSERT_TRUE(kernel_->finalizeMoEOverlayRebalancePublication(
            maintenanceLaunch(), device_runtime_family_,
            /*layer_count=*/2u, /*expert_count=*/1u,
            device_control_, &device_epochs_[0], &device_statuses_[2],
            device_apply_status_));
        ASSERT_EQ(hipStreamEndCapture(maintenance_stream_, &graph_), hipSuccess);
        ASSERT_NE(graph_, nullptr);
        ASSERT_EQ(hipGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0),
                  hipSuccess);

        ASSERT_EQ(hipGraphLaunch(graph_exec_, maintenance_stream_), hipSuccess);
        DeviceMoEOverlayEpochControl control{};
        DeviceMoEOverlayEpochStatus publication_status{};
        std::uint64_t next_epoch = 0u;
        ASSERT_EQ(hipMemcpyAsync(
                     &control, device_control_, sizeof(control),
                     hipMemcpyDeviceToHost, maintenance_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     &publication_status, &device_statuses_[2],
                     sizeof(publication_status), hipMemcpyDeviceToHost,
                     maintenance_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     &next_epoch, &device_epochs_[0], sizeof(next_epoch),
                     hipMemcpyDeviceToHost, maintenance_stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(maintenance_stream_), hipSuccess);
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
            ASSERT_EQ(hipMemcpyAsync(
                         device_apply_status_, &apply_status,
                         sizeof(apply_status), hipMemcpyHostToDevice,
                         maintenance_stream_),
                      hipSuccess);
            ASSERT_EQ(hipGraphLaunch(graph_exec_, maintenance_stream_), hipSuccess);

            std::array<DeviceMoELayerRuntime, 2> observed_family{};
            ASSERT_EQ(hipMemcpyAsync(
                         observed_family.data(), device_runtime_family_,
                         sizeof(observed_family), hipMemcpyDeviceToHost,
                         maintenance_stream_),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(
                         &control, device_control_, sizeof(control),
                         hipMemcpyDeviceToHost, maintenance_stream_),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(
                         &publication_status, &device_statuses_[2],
                         sizeof(publication_status), hipMemcpyDeviceToHost,
                         maintenance_stream_),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(
                         &next_epoch, &device_epochs_[0], sizeof(next_epoch),
                         hipMemcpyDeviceToHost, maintenance_stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(maintenance_stream_), hipSuccess);

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
        ASSERT_EQ(hipMemcpyAsync(
                     device_runtime_, &sidecar, sizeof(sidecar),
                     hipMemcpyHostToDevice, inference_stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                     device_route_participant_, &unset_participant,
                     sizeof(unset_participant), hipMemcpyHostToDevice,
                     inference_stream_),
                  hipSuccess);
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
        ASSERT_EQ(hipMemcpyAsync(
                     &routed_participant, device_route_participant_,
                     sizeof(routed_participant), hipMemcpyDeviceToHost,
                     inference_stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(inference_stream_), hipSuccess);
        EXPECT_EQ(routed_participant, published_owner);
    }
} // namespace llaminar2::test
