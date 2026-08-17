/**
 * @file Test__MoEOverlayDeviceEpochProtocol.cpp
 * @brief Adversarial CPU tests for captured ExpertOverlay device epoch RCU.
 */

#include "execution/moe/MoEOverlayDeviceEpochProtocol.h"
#include "execution/moe/DeviceMoEOverlayEpochArena.h"
#include "kernels/cpu/moe/CPUMoEKernel.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace llaminar2::test
{
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
