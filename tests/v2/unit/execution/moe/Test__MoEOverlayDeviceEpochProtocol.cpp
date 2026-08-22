/**
 * @file Test__MoEOverlayDeviceEpochProtocol.cpp
 * @brief Adversarial CPU tests for captured ExpertOverlay device epoch RCU.
 */

#include "execution/moe/MoEOverlayDeviceEpochProtocol.h"
#include "execution/moe/DeviceMoEOverlayEpochArena.h"
#include "execution/moe/MoEOverlayDeviceControllerABI.h"
#include "execution/moe/MoEOverlayDeviceControllerRuntimeBinding.h"
#include "kernels/cpu/moe/CPUMoEKernel.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace llaminar2::test
{
    TEST(Test__MoEOverlayDeviceEpochProtocol,
         FollowerBoundaryReceiptNeverRetargetsACompletedBoundaryToNewWork)
    {
        MoEOverlayInferenceBoundaryReceipt receipt;

        const auto pristine = receipt.consumeLatestCompleted();
        EXPECT_FALSE(pristine.hasCompletedBoundary());
        EXPECT_FALSE(pristine.newerSubmissionInFlight());

        /* Admission precedes backend launch. A rejected launch removes only
         * its newest receipt and restores the exact prior serial state. */
        const std::uint64_t rejected = receipt.beginSubmission();
        ASSERT_EQ(rejected, 1u);
        EXPECT_TRUE(receipt.consumeLatestCompleted().newerSubmissionInFlight());
        EXPECT_TRUE(receipt.cancelUnsubmitted(rejected));
        EXPECT_EQ(receipt.submittedGeneration(), 0u);
        EXPECT_EQ(receipt.completedGeneration(), 0u);
        EXPECT_FALSE(receipt.cancelUnsubmitted(rejected));

        const std::uint64_t first = receipt.beginSubmission();
        ASSERT_EQ(first, 1u);
        const auto first_in_flight = receipt.consumeLatestCompleted();
        EXPECT_FALSE(first_in_flight.hasCompletedBoundary());
        EXPECT_TRUE(first_in_flight.newerSubmissionInFlight());
        EXPECT_FALSE(first_in_flight.fresh_completion);

        ASSERT_TRUE(receipt.completeSubmission(first));
        EXPECT_FALSE(receipt.cancelUnsubmitted(first));
        const auto first_committed = receipt.consumeLatestCompleted();
        EXPECT_TRUE(first_committed.hasCompletedBoundary());
        EXPECT_EQ(first_committed.completed_generation, first);
        EXPECT_TRUE(first_committed.fresh_completion);
        EXPECT_FALSE(first_committed.newerSubmissionInFlight());

        /*
         * This is the deadlock-producing interleaving from production: the
         * next sparse graph is submitted before background maintenance consumes
         * the prior command boundary.  The receipt remains pinned to generation
         * one and reports generation two as live.  A topology-wide maintenance
         * submitter must defer this snapshot: beginning from the older terminal
         * can deadlock against the newer transaction on another participant.
         */
        const std::uint64_t second = receipt.beginSubmission();
        ASSERT_EQ(second, 2u);
        const auto overlap = receipt.consumeLatestCompleted();
        EXPECT_EQ(overlap.submitted_generation, second);
        EXPECT_EQ(overlap.completed_generation, first);
        EXPECT_FALSE(overlap.fresh_completion);
        EXPECT_TRUE(overlap.newerSubmissionInFlight());

        EXPECT_FALSE(receipt.completeSubmission(first));
        EXPECT_FALSE(receipt.completeSubmission(second + 1u));
        ASSERT_TRUE(receipt.completeSubmission(second));
        const auto second_committed = receipt.consumeLatestCompleted();
        EXPECT_EQ(second_committed.completed_generation, second);
        EXPECT_TRUE(second_committed.fresh_completion);
        EXPECT_FALSE(second_committed.newerSubmissionInFlight());
        EXPECT_EQ(receipt.submittedGeneration(), second);
        EXPECT_EQ(receipt.completedGeneration(), second);
        EXPECT_EQ(receipt.consumedGeneration(), second);
    }

    TEST(Test__MoEOverlayDeviceEpochProtocol,
         ArenaOwnsStableSlotsAndHonorsPreparedInitialBank)
    {
        DeviceMoEOverlayEpochArena arena({
            .device_id = DeviceId::cpu(),
            .initial_epoch = 41u,
            .initial_bank = 1u,
            .request_slot_capacity = 3u,
        });

        ASSERT_NE(arena.control(), nullptr);
        EXPECT_EQ(arena.deviceId(), DeviceId::cpu());
        EXPECT_EQ(arena.requestSlotCapacity(), 3u);
        EXPECT_EQ(
            arena.allocationBytes(),
            sizeof(DeviceMoEOverlayEpochControl) +
                3u * sizeof(DeviceMoEOverlayEpochTicket) +
                3u * sizeof(DeviceMoEOverlayEpochStatus) +
                sizeof(std::uint64_t) +
                sizeof(DeviceMoEOverlayEpochStatus));

        MoEOverlayDeviceEpochProtocol protocol(*arena.control());
        EXPECT_EQ(protocol.publishedEpoch(), 41u);
        EXPECT_EQ(protocol.publishedBank(), 1u);
        EXPECT_EQ(protocol.publicationGeneration(), 1u);
        EXPECT_EQ(
            protocol.bankState(1u),
            DeviceMoEOverlayEpochBankState::Published);
        EXPECT_EQ(
            protocol.bankState(0u),
            DeviceMoEOverlayEpochBankState::Empty);

        EXPECT_NE(arena.requestTicket(0u), arena.requestTicket(1u));
        EXPECT_NE(arena.requestStatus(1u), arena.requestStatus(2u));
        EXPECT_EQ(arena.requestTicket(2u)->epoch, 0u);
        EXPECT_EQ(
            arena.requestStatus(2u)->bank,
            kDeviceMoEOverlayInvalidBank);
        EXPECT_EQ(*arena.maintenanceEpoch(), 42u);
        EXPECT_EQ(
            arena.maintenanceStatus()->bank,
            kDeviceMoEOverlayInvalidBank);
        EXPECT_THROW((void)arena.requestTicket(3u), std::out_of_range);
        EXPECT_THROW((void)arena.requestStatus(3u), std::out_of_range);

        CPUMoEKernel kernel;
        const MoEKernelLaunchContext launch{};
        ASSERT_TRUE(kernel.acquireMoEOverlayEpoch(
            launch,
            arena.control(),
            arena.requestTicket(1u),
            arena.requestStatus(1u)));
        EXPECT_TRUE(arena.requestStatus(1u)->succeeded());
        EXPECT_EQ(arena.requestTicket(1u)->epoch, 41u);
        EXPECT_EQ(arena.requestTicket(1u)->bank(), 1u);
        ASSERT_TRUE(kernel.releaseMoEOverlayEpoch(
            launch,
            arena.control(),
            arena.requestTicket(1u),
            arena.requestStatus(1u)));
        EXPECT_TRUE(arena.requestStatus(1u)->succeeded());
        EXPECT_EQ(arena.requestTicket(1u)->epoch, 0u);
    }

    TEST(Test__MoEOverlayDeviceEpochProtocol,
         PublicationOverlapsOldReaderAndReusesOnlyAfterRetirement)
    {
        DeviceMoEOverlayEpochControl control;
        MoEOverlayDeviceEpochProtocol::initialize(control, 1u);
        MoEOverlayDeviceEpochProtocol protocol(control);

        auto old_ticket = protocol.tryAcquirePublished();
        ASSERT_TRUE(old_ticket.has_value());
        EXPECT_EQ(old_ticket->epoch, 1u);
        EXPECT_EQ(old_ticket->bank(), 0u);
        EXPECT_EQ(old_ticket->generation(), 1u);
        EXPECT_EQ(protocol.bankReaderCount(0u), 1u);

        ASSERT_EQ(protocol.reserveCandidate(2u), 1u);
        EXPECT_EQ(
            protocol.bankState(1u),
            DeviceMoEOverlayEpochBankState::Candidate);
        protocol.markCandidateReady(2u);
        const auto publication = protocol.publishReadyCandidate(2u);
        ASSERT_TRUE(publication.valid());
        EXPECT_EQ(protocol.publishedEpoch(), 2u);
        EXPECT_EQ(protocol.publishedBank(), 1u);
        EXPECT_EQ(protocol.publicationGeneration(), 2u);
        EXPECT_EQ(
            protocol.bankState(0u),
            DeviceMoEOverlayEpochBankState::Retiring);
        EXPECT_FALSE(protocol.tryRetire(1u));

        auto new_ticket = protocol.tryAcquirePublished();
        ASSERT_TRUE(new_ticket.has_value());
        EXPECT_EQ(new_ticket->epoch, 2u);
        EXPECT_EQ(new_ticket->bank(), 1u);
        EXPECT_EQ(new_ticket->generation(), 2u);

        EXPECT_TRUE(protocol.release(*old_ticket));
        EXPECT_FALSE(protocol.release(*old_ticket))
            << "A ticket reader may be released exactly once";
        EXPECT_TRUE(protocol.tryRetire(1u));
        EXPECT_EQ(protocol.bankEpoch(0u), 0u);
        EXPECT_EQ(
            protocol.bankState(0u),
            DeviceMoEOverlayEpochBankState::Empty);

        ASSERT_EQ(protocol.reserveCandidate(3u), 0u)
            << "Only the completely retired physical bank may be reused";
        protocol.markCandidateReady(3u);
        const auto next = protocol.publishReadyCandidate(3u);
        EXPECT_EQ(next.previous_epoch, 2u);
        EXPECT_EQ(next.published_epoch, 3u);
        EXPECT_EQ(next.generation, 3u);

        EXPECT_TRUE(protocol.release(*new_ticket));
        EXPECT_TRUE(protocol.tryRetire(2u));
    }

    TEST(Test__MoEOverlayDeviceEpochProtocol,
         TopologyAdmissionPinsRetiringBankUntilGlobalEpochAdvances)
    {
        DeviceMoEOverlayEpochControl control;
        MoEOverlayDeviceEpochProtocol::initialize(control, 17u);
        MoEOverlayDeviceEpochProtocol protocol(control);

        ASSERT_EQ(protocol.reserveCandidate(18u), 1u);
        protocol.markCandidateReady(18u);
        ASSERT_TRUE(protocol.publishReadyCandidate(18u).valid());
        ASSERT_EQ(
            protocol.bankState(0u),
            DeviceMoEOverlayEpochBankState::Retiring);

        // The participant has locally flipped, but the topology leader has
        // not admitted the new epoch. A request must retain the old bank.
        auto old_admission = protocol.tryAcquireAdmitted(17u);
        ASSERT_TRUE(old_admission.has_value());
        EXPECT_EQ(old_admission->epoch, 17u);
        EXPECT_EQ(old_admission->bank(), 0u);
        EXPECT_EQ(old_admission->generation(), 1u);
        EXPECT_FALSE(protocol.tryRetire(17u));
        EXPECT_TRUE(protocol.release(*old_admission));

        // Once every participant has published, the same local control admits
        // the new bank through the topology-wide epoch word.
        auto new_admission = protocol.tryAcquireAdmitted(18u);
        ASSERT_TRUE(new_admission.has_value());
        EXPECT_EQ(new_admission->epoch, 18u);
        EXPECT_EQ(new_admission->bank(), 1u);
        EXPECT_EQ(new_admission->generation(), 2u);
        EXPECT_TRUE(protocol.tryRetire(17u));
        EXPECT_TRUE(protocol.release(*new_admission));
        EXPECT_FALSE(protocol.tryAcquireAdmitted(17u).has_value())
            << "A globally admitted epoch may not resolve after retirement";
    }

    TEST(Test__MoEOverlayDeviceEpochProtocol,
         PublicCPUKernelConsumesExternalAdmissionEpoch)
    {
        DeviceMoEOverlayEpochControl control;
        MoEOverlayDeviceEpochProtocol::initialize(control, 31u);
        MoEOverlayDeviceEpochProtocol protocol(control);
        ASSERT_EQ(protocol.reserveCandidate(32u), 1u);
        protocol.markCandidateReady(32u);
        ASSERT_TRUE(protocol.publishReadyCandidate(32u).valid());

        std::uint64_t admission_epoch = 31u;
        CPUMoEKernel kernel;
        DeviceMoEOverlayEpochTicket ticket{};
        DeviceMoEOverlayEpochStatus status{};
        const MoEKernelLaunchContext launch{};
        ASSERT_TRUE(kernel.acquireMoEOverlayEpoch(
            launch,
            &control,
            &ticket,
            &status,
            &admission_epoch));
        ASSERT_TRUE(status.succeeded());
        EXPECT_EQ(ticket.epoch, 31u);
        EXPECT_EQ(ticket.bank(), 0u);
        ASSERT_TRUE(kernel.releaseMoEOverlayEpoch(
            launch, &control, &ticket, &status));
        ASSERT_TRUE(status.succeeded());

        std::atomic_ref<std::uint64_t>(admission_epoch).store(
            32u, std::memory_order_release);
        ASSERT_TRUE(kernel.acquireMoEOverlayEpoch(
            launch,
            &control,
            &ticket,
            &status,
            &admission_epoch));
        ASSERT_TRUE(status.succeeded());
        EXPECT_EQ(ticket.epoch, 32u);
        EXPECT_EQ(ticket.bank(), 1u);
        ASSERT_TRUE(kernel.releaseMoEOverlayEpoch(
            launch, &control, &ticket, &status));
    }

    TEST(Test__MoEOverlayDeviceEpochProtocol,
         SymmetricContinuationFreezesOneEpochAfterEveryLocalGuardArrives)
    {
        constexpr std::uint32_t kSiblingParticipant = 1u;
        constexpr std::uint32_t kPublisherParticipant = 4u;
        constexpr std::uint32_t kMask =
            (1u << kSiblingParticipant) | (1u << kPublisherParticipant);

        std::array<DeviceMoEOverlayEpochControl, 2> controls{};
        for (auto &control : controls)
        {
            MoEOverlayDeviceEpochProtocol::initialize(control, 1u);
            MoEOverlayDeviceEpochProtocol protocol(control);
            ASSERT_EQ(protocol.reserveCandidate(2u), 1u);
            protocol.markCandidateReady(2u);
            ASSERT_TRUE(protocol.publishReadyCandidate(2u).valid());
        }

        MoEOverlayDeviceControllerInferenceEpochRecord barrier{};
        barrier.participant_mask = kMask;
        barrier.publisher_participant_id = kPublisherParticipant;
        barrier.topology_fingerprint = 0xabc123u;
        barrier.epoch = 1u;
        std::uint64_t live_admission_epoch = 1u;

        std::array<DeviceMoEOverlayEpochTicket, 2> tickets{};
        std::array<DeviceMoEOverlayEpochStatus, 2> statuses{};
        std::array<CPUMoEKernel, 2> kernels{};
        const MoEKernelLaunchContext launch{};
        std::array<bool, 2> submitted{};

        /* The publisher reaches admission first while epoch one is live. It
         * must not sample yet: the sibling has not raised its local retirement
         * guard. Flip admission in exactly that vulnerable interval. */
        std::thread publisher(
            [&]
            {
                submitted[0] = kernels[0].acquireMoEOverlayEpoch(
                    launch,
                    &controls[0],
                    &tickets[0],
                    &statuses[0],
                    &live_admission_epoch,
                    {
                        .record = &barrier,
                        .participant_id = kPublisherParticipant,
                    });
            });

        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(2);
        bool publisher_arrived = false;
        while (std::chrono::steady_clock::now() < deadline)
        {
            publisher_arrived =
                std::atomic_ref<std::uint64_t>(
                    barrier.arrival_sequence[kPublisherParticipant])
                    .load(std::memory_order_acquire) == 1u;
            if (publisher_arrived)
                break;
            std::this_thread::yield();
        }
        std::atomic_ref<std::uint64_t>(live_admission_epoch).store(
            2u, std::memory_order_release);

        std::thread sibling(
            [&]
            {
                submitted[1] = kernels[1].acquireMoEOverlayEpoch(
                    launch,
                    &controls[1],
                    &tickets[1],
                    &statuses[1],
                    &live_admission_epoch,
                    {
                        .record = &barrier,
                        .participant_id = kSiblingParticipant,
                    });
            });
        publisher.join();
        sibling.join();

        ASSERT_TRUE(publisher_arrived)
            << "publisher did not hold its local RCU guard before the adversarial flip";
        ASSERT_TRUE(submitted[0]);
        ASSERT_TRUE(submitted[1]);
        ASSERT_TRUE(statuses[0].succeeded());
        ASSERT_TRUE(statuses[1].succeeded());
        EXPECT_EQ(tickets[0].epoch, 2u);
        EXPECT_EQ(tickets[1].epoch, 2u);
        EXPECT_EQ(barrier.epoch, 2u);
        EXPECT_EQ(barrier.publication_sequence, 1u);
        EXPECT_EQ(
            barrier.arrival_sequence[kPublisherParticipant], 1u);
        EXPECT_EQ(barrier.arrival_sequence[kSiblingParticipant], 1u);
        EXPECT_EQ(controls[0].acquisitions_in_flight, 0u);
        EXPECT_EQ(controls[1].acquisitions_in_flight, 0u);

        ASSERT_TRUE(kernels[0].releaseMoEOverlayEpoch(
            launch, &controls[0], &tickets[0], &statuses[0]));
        ASSERT_TRUE(kernels[1].releaseMoEOverlayEpoch(
            launch, &controls[1], &tickets[1], &statuses[1]));
        ASSERT_TRUE(statuses[0].succeeded());
        ASSERT_TRUE(statuses[1].succeeded());

        /* Replay with the non-publisher arriving first proves the monotonic
         * record has no fixed device-order assumption and cannot reuse N. */
        std::atomic_ref<std::uint64_t>(live_admission_epoch).store(
            1u, std::memory_order_release);
        std::thread sibling_first(
            [&]
            {
                submitted[1] = kernels[1].acquireMoEOverlayEpoch(
                    launch,
                    &controls[1],
                    &tickets[1],
                    &statuses[1],
                    &live_admission_epoch,
                    {
                        .record = &barrier,
                        .participant_id = kSiblingParticipant,
                    });
            });
        std::thread publisher_second(
            [&]
            {
                submitted[0] = kernels[0].acquireMoEOverlayEpoch(
                    launch,
                    &controls[0],
                    &tickets[0],
                    &statuses[0],
                    &live_admission_epoch,
                    {
                        .record = &barrier,
                        .participant_id = kPublisherParticipant,
                    });
            });
        sibling_first.join();
        publisher_second.join();

        ASSERT_TRUE(statuses[0].succeeded());
        ASSERT_TRUE(statuses[1].succeeded());
        EXPECT_EQ(tickets[0].epoch, 1u);
        EXPECT_EQ(tickets[1].epoch, 1u);
        EXPECT_EQ(barrier.epoch, 1u);
        EXPECT_EQ(barrier.publication_sequence, 2u);
        EXPECT_EQ(
            barrier.arrival_sequence[kPublisherParticipant], 2u);
        EXPECT_EQ(barrier.arrival_sequence[kSiblingParticipant], 2u);
        ASSERT_TRUE(kernels[0].releaseMoEOverlayEpoch(
            launch, &controls[0], &tickets[0], &statuses[0]));
        ASSERT_TRUE(kernels[1].releaseMoEOverlayEpoch(
            launch, &controls[1], &tickets[1], &statuses[1]));
    }

    TEST(Test__MoEOverlayDeviceEpochProtocol,
         CandidateLifecycleRejectsEarlyPublishAndSupportsAbort)
    {
        DeviceMoEOverlayEpochControl control;
        MoEOverlayDeviceEpochProtocol::initialize(control, 7u);
        MoEOverlayDeviceEpochProtocol protocol(control);

        ASSERT_EQ(protocol.reserveCandidate(8u), 1u);
        EXPECT_THROW(
            (void)protocol.publishReadyCandidate(8u),
            std::logic_error);
        EXPECT_TRUE(protocol.abortCandidate(8u));
        EXPECT_EQ(protocol.bankEpoch(1u), 0u);
        EXPECT_EQ(protocol.publishedEpoch(), 7u);

        ASSERT_EQ(protocol.reserveCandidate(9u), 1u);
        protocol.markCandidateReady(9u);
        EXPECT_THROW(protocol.markCandidateReady(9u), std::logic_error);
        EXPECT_TRUE(protocol.abortCandidate(9u));
        EXPECT_FALSE(protocol.abortCandidate(9u));
        EXPECT_THROW(
            (void)protocol.reserveCandidate(7u),
            std::invalid_argument);
    }

    TEST(Test__MoEOverlayDeviceEpochProtocol,
         TwentyEpochsPreserveEveryHeldTicketAcrossPublication)
    {
        DeviceMoEOverlayEpochControl control;
        MoEOverlayDeviceEpochProtocol::initialize(control, 1u);
        MoEOverlayDeviceEpochProtocol protocol(control);

        for (std::uint64_t epoch = 1u; epoch <= 20u; ++epoch)
        {
            auto held = protocol.tryAcquirePublished();
            ASSERT_TRUE(held.has_value());
            ASSERT_EQ(held->epoch, epoch);

            if (epoch != 20u)
            {
                const std::uint64_t candidate_epoch = epoch + 1u;
                ASSERT_TRUE(
                    protocol.reserveCandidate(candidate_epoch).has_value());
                protocol.markCandidateReady(candidate_epoch);
                const auto publication =
                    protocol.publishReadyCandidate(candidate_epoch);
                ASSERT_EQ(publication.previous_epoch, epoch);
                ASSERT_EQ(publication.published_epoch, candidate_epoch);
                EXPECT_FALSE(protocol.tryRetire(epoch));
            }

            EXPECT_TRUE(protocol.release(*held));
            if (epoch != 20u)
                EXPECT_TRUE(protocol.tryRetire(epoch));
        }
    }

    TEST(Test__MoEOverlayDeviceEpochProtocol,
         ConcurrentAdmissionSeesOnlyCompleteOldOrNewPublication)
    {
        DeviceMoEOverlayEpochControl control;
        MoEOverlayDeviceEpochProtocol::initialize(control, 1u);
        MoEOverlayDeviceEpochProtocol protocol(control);

        constexpr int kWorkers = 4;
        constexpr int kAcquisitionsPerWorker = 5000;
        std::atomic<bool> start{false};
        std::atomic<std::uint64_t> invalid_tickets{0u};
        std::atomic<std::uint64_t> release_failures{0u};
        std::vector<std::thread> workers;
        workers.reserve(kWorkers);
        for (int worker = 0; worker < kWorkers; ++worker)
        {
            workers.emplace_back(
                [&]
                {
                    while (!start.load(std::memory_order_acquire))
                        std::this_thread::yield();
                    for (int iteration = 0;
                         iteration < kAcquisitionsPerWorker;
                         ++iteration)
                    {
                        auto ticket = protocol.tryAcquirePublished();
                        if (!ticket.has_value())
                        {
                            invalid_tickets.fetch_add(
                                1u, std::memory_order_relaxed);
                            continue;
                        }
                        const bool identity_is_valid =
                            (ticket->epoch == 1u || ticket->epoch == 2u) &&
                            (ticket->epoch != 1u || ticket->bank() == 0u) &&
                            (ticket->epoch != 2u || ticket->bank() == 1u);
                        if (!identity_is_valid)
                        {
                            invalid_tickets.fetch_add(
                                1u, std::memory_order_relaxed);
                        }
                        /*
                         * Even a malformed acquired identity owns a reader.
                         * Releasing it keeps the test diagnostic from turning
                         * the first defect into a second retirement leak.
                         */
                        if (!protocol.release(*ticket))
                        {
                            release_failures.fetch_add(
                                1u, std::memory_order_relaxed);
                        }
                    }
                });
        }

        ASSERT_EQ(protocol.reserveCandidate(2u), 1u);
        protocol.markCandidateReady(2u);
        start.store(true, std::memory_order_release);
        const auto publication = protocol.publishReadyCandidate(2u);
        ASSERT_TRUE(publication.valid());

        for (auto &worker : workers)
            worker.join();

        EXPECT_EQ(invalid_tickets.load(std::memory_order_relaxed), 0u);
        EXPECT_EQ(release_failures.load(std::memory_order_relaxed), 0u);
        EXPECT_EQ(protocol.acquisitionsInFlight(), 0u);
        EXPECT_EQ(protocol.bankReaderCount(0u), 0u);
        EXPECT_EQ(protocol.bankReaderCount(1u), 0u);
        EXPECT_TRUE(protocol.tryRetire(1u));
    }

    TEST(Test__MoEOverlayDeviceEpochProtocol,
         PublicCPUKernelContractReportsSemanticStatusAndClearsTickets)
    {
        DeviceMoEOverlayEpochControl control;
        MoEOverlayDeviceEpochProtocol::initialize(control, 11u);
        CPUMoEKernel kernel;
        const MoEKernelLaunchContext launch{};
        DeviceMoEOverlayEpochTicket old_ticket{};
        DeviceMoEOverlayEpochTicket new_ticket{};
        DeviceMoEOverlayEpochStatus status{};

        ASSERT_TRUE(kernel.acquireMoEOverlayEpoch(
            launch, &control, &old_ticket, &status));
        ASSERT_TRUE(status.succeeded());
        EXPECT_EQ(status.typedOperation(),
                  DeviceMoEOverlayEpochOperation::Acquire);
        EXPECT_EQ(old_ticket.epoch, 11u);
        EXPECT_EQ(old_ticket.bank(), 0u);

        const std::uint64_t candidate_epoch = 12u;
        ASSERT_TRUE(kernel.reserveMoEOverlayEpochCandidate(
            launch, &control, &candidate_epoch, &status));
        ASSERT_TRUE(status.succeeded());
        EXPECT_EQ(status.bank, 1u);
        ASSERT_TRUE(kernel.markMoEOverlayEpochCandidateReady(
            launch, &control, &candidate_epoch, &status));
        ASSERT_TRUE(status.succeeded());
        ASSERT_TRUE(kernel.publishMoEOverlayEpochCandidate(
            launch, &control, &candidate_epoch, &status));
        ASSERT_TRUE(status.succeeded());
        EXPECT_EQ(status.selector, deviceMoEOverlayEpochSelector(2u, 1u));

        ASSERT_TRUE(kernel.acquireMoEOverlayEpoch(
            launch, &control, &new_ticket, &status));
        ASSERT_TRUE(status.succeeded());
        EXPECT_EQ(new_ticket.epoch, 12u);
        EXPECT_EQ(new_ticket.bank(), 1u);

        const std::uint64_t retiring_epoch = 11u;
        ASSERT_TRUE(kernel.retireMoEOverlayEpoch(
            launch, &control, &retiring_epoch, &status));
        EXPECT_EQ(status.typedCode(),
                  DeviceMoEOverlayEpochStatusCode::Busy);

        ASSERT_TRUE(kernel.releaseMoEOverlayEpoch(
            launch, &control, &old_ticket, &status));
        ASSERT_TRUE(status.succeeded());
        EXPECT_FALSE(old_ticket.valid());
        EXPECT_EQ(old_ticket.epoch, 0u);
        EXPECT_EQ(old_ticket.selector, 0u);

        ASSERT_TRUE(kernel.retireMoEOverlayEpoch(
            launch, &control, &retiring_epoch, &status));
        ASSERT_TRUE(status.succeeded());
        EXPECT_EQ(control.bank_epochs[0], 0u);

        ASSERT_TRUE(kernel.releaseMoEOverlayEpoch(
            launch, &control, &new_ticket, &status));
        ASSERT_TRUE(status.succeeded());
        EXPECT_FALSE(new_ticket.valid());
        ASSERT_TRUE(kernel.releaseMoEOverlayEpoch(
            launch, &control, &new_ticket, &status));
        EXPECT_EQ(status.typedCode(),
                  DeviceMoEOverlayEpochStatusCode::InvalidTicket);
    }
} // namespace llaminar2::test
