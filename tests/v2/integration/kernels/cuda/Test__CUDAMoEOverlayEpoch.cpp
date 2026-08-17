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
#include "execution/moe/DeviceMoERebalanceController.h"
#include "execution/moe/MoERuntimeTable.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>

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
                          sizeof(*device_plan_entry_)),
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
        EXPECT_EQ(controller.waves[0].state, static_cast<std::uint32_t>(
            DeviceMoERebalanceWaveLifecycle::Applied));

        /* Apply prepares only the reserved bank. Publication is a separate
         * all-layer transaction, so active placement must remain at epoch one. */
        EXPECT_FALSE(released_ticket.valid());
        EXPECT_EQ(runtime.active_bank, 0u);
        EXPECT_EQ(runtime.active_epoch, 1u);
        EXPECT_EQ(runtime.banks[1].epoch, 2u);
        EXPECT_EQ(runtime.banks[1].experts[0].owner_participant, 1);
        EXPECT_EQ(runtime.banks[1].resident_participant_mask[0], 0b10u);
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
            device_apply_status_));
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
} // namespace llaminar2::test
