/**
 * @file Test__MoEDeviceRebalanceHostRendezvous.cpp
 * @brief Unit tests for host-side MoE device maintenance graph rendezvous.
 */

#include "execution/moe/MoEDeviceRebalanceHostRendezvous.h"

#include <gtest/gtest.h>

#include <future>
#include <string>
#include <utility>

using namespace llaminar2;

namespace
{
    MoEDeviceRebalanceHostRendezvous::ProbeOutcome okNoWork()
    {
        MoEDeviceRebalanceHostRendezvous::ProbeOutcome outcome;
        outcome.valid = true;
        outcome.status_code = 0;
        return outcome;
    }

    MoEDeviceRebalanceHostRendezvous::ProbeOutcome okPayload(uint32_t slots, uint64_t edge_mask)
    {
        auto outcome = okNoWork();
        outcome.useful_work = true;
        outcome.payload_bucket_slots = slots;
        outcome.payload_edge_mask = edge_mask;
        return outcome;
    }
} // namespace

TEST(Test__MoEDeviceRebalanceHostRendezvous, RootOnlyPayloadDecisionIsBroadcastToAllParticipants)
{
    int domain_key;
    constexpr uint64_t generation = 101;
    constexpr int timeout_ms = 1000;

    auto participant0 = std::async(std::launch::async, [&]()
                                   {
                                       MoEDeviceRebalanceHostRendezvous::PayloadDecision decision;
                                       std::string error;
                                       const bool ok = MoEDeviceRebalanceHostRendezvous::rendezvousProbeOutcome(
                                           &domain_key,
                                           generation,
                                           2,
                                           "cuda:0",
                                           okPayload(/*slots=*/1, /*edge_mask=*/0x100ULL),
                                           timeout_ms,
                                           &decision,
                                           &error);
                                       return std::make_pair(ok, decision);
                                   });

    auto participant1 = std::async(std::launch::async, [&]()
                                   {
                                       MoEDeviceRebalanceHostRendezvous::PayloadDecision decision;
                                       std::string error;
                                       const bool ok = MoEDeviceRebalanceHostRendezvous::rendezvousProbeOutcome(
                                           &domain_key,
                                           generation,
                                           2,
                                           "cuda:1",
                                           okNoWork(),
                                           timeout_ms,
                                           &decision,
                                           &error);
                                       return std::make_pair(ok, decision);
                                   });

    const auto result0 = participant0.get();
    const auto result1 = participant1.get();

    ASSERT_TRUE(result0.first);
    ASSERT_TRUE(result1.first);
    EXPECT_TRUE(result0.second.launchPayloadGraph());
    EXPECT_TRUE(result1.second.launchPayloadGraph());
    EXPECT_EQ(result0.second.payload_bucket_slots, 1u);
    EXPECT_EQ(result1.second.payload_bucket_slots, 1u);
    EXPECT_EQ(result0.second.payload_edge_mask, 0x100ULL);
    EXPECT_EQ(result1.second.payload_edge_mask, 0x100ULL);
}

TEST(Test__MoEDeviceRebalanceHostRendezvous, NoWorkIsDomainWide)
{
    int domain_key;
    constexpr uint64_t generation = 202;
    constexpr int timeout_ms = 1000;

    auto participant0 = std::async(std::launch::async, [&]()
                                   {
                                       MoEDeviceRebalanceHostRendezvous::PayloadDecision decision;
                                       std::string error;
                                       const bool ok = MoEDeviceRebalanceHostRendezvous::rendezvousProbeOutcome(
                                           &domain_key,
                                           generation,
                                           2,
                                           "cuda:0",
                                           okNoWork(),
                                           timeout_ms,
                                           &decision,
                                           &error);
                                       return std::make_pair(ok, decision);
                                   });

    auto participant1 = std::async(std::launch::async, [&]()
                                   {
                                       MoEDeviceRebalanceHostRendezvous::PayloadDecision decision;
                                       std::string error;
                                       const bool ok = MoEDeviceRebalanceHostRendezvous::rendezvousProbeOutcome(
                                           &domain_key,
                                           generation,
                                           2,
                                           "cuda:1",
                                           okNoWork(),
                                           timeout_ms,
                                           &decision,
                                           &error);
                                       return std::make_pair(ok, decision);
                                   });

    const auto result0 = participant0.get();
    const auto result1 = participant1.get();

    ASSERT_TRUE(result0.first);
    ASSERT_TRUE(result1.first);
    EXPECT_FALSE(result0.second.launchPayloadGraph());
    EXPECT_FALSE(result1.second.launchPayloadGraph());
    EXPECT_FALSE(result0.second.useful_work);
    EXPECT_FALSE(result1.second.useful_work);
}

TEST(Test__MoEDeviceRebalanceHostRendezvous, MissingParticipantFailsFast)
{
    int domain_key;
    MoEDeviceRebalanceHostRendezvous::PayloadDecision decision;
    std::string error;

    const bool ok = MoEDeviceRebalanceHostRendezvous::rendezvousProbeOutcome(
        &domain_key,
        /*generation=*/303,
        /*expected_participants=*/2,
        "cuda:0",
        okNoWork(),
        /*timeout_ms=*/5,
        &decision,
        &error);

    EXPECT_FALSE(ok);
    EXPECT_NE(error.find("timed out waiting for 2 participants"), std::string::npos);
}

TEST(Test__MoEDeviceRebalanceHostRendezvous, CompletionReadinessReportsPartialDomainReady)
{
    int domain_key;
    constexpr uint64_t generation = 404;
    constexpr uint64_t operation = 7;
    constexpr int timeout_ms = 1000;

    auto participant0 = std::async(std::launch::async, [&]()
                                   {
                                       MoEDeviceRebalanceHostRendezvous::ReadinessDecision decision;
                                       std::string error;
                                       const bool ok = MoEDeviceRebalanceHostRendezvous::rendezvousCompletionReadiness(
                                           &domain_key,
                                           generation,
                                           operation,
                                           2,
                                           "rocm:0",
                                           true,
                                           timeout_ms,
                                           &decision,
                                           &error);
                                       return std::make_pair(ok, decision);
                                   });

    auto participant1 = std::async(std::launch::async, [&]()
                                   {
                                       MoEDeviceRebalanceHostRendezvous::ReadinessDecision decision;
                                       std::string error;
                                       const bool ok = MoEDeviceRebalanceHostRendezvous::rendezvousCompletionReadiness(
                                           &domain_key,
                                           generation,
                                           operation,
                                           2,
                                           "rocm:1",
                                           false,
                                           timeout_ms,
                                           &decision,
                                           &error);
                                       return std::make_pair(ok, decision);
                                   });

    const auto result0 = participant0.get();
    const auto result1 = participant1.get();

    ASSERT_TRUE(result0.first);
    ASSERT_TRUE(result1.first);
    EXPECT_TRUE(result0.second.valid);
    EXPECT_TRUE(result1.second.valid);
    EXPECT_FALSE(result0.second.all_ready);
    EXPECT_FALSE(result1.second.all_ready);
    EXPECT_EQ(result0.second.ready_count, 1u);
    EXPECT_EQ(result1.second.ready_count, 1u);
    EXPECT_EQ(result0.second.participant_count, 2u);
    EXPECT_EQ(result1.second.participant_count, 2u);
}

TEST(Test__MoEDeviceRebalanceHostRendezvous, CompletionReadinessCanRetrySameWaveAfterPartialReady)
{
    int domain_key;
    constexpr uint64_t generation = 505;
    constexpr uint64_t operation = 11;
    constexpr int timeout_ms = 1000;

    auto run_participants = [&](bool local0_ready, bool local1_ready)
    {
        auto participant0 = std::async(std::launch::async, [&]()
                                       {
                                           MoEDeviceRebalanceHostRendezvous::ReadinessDecision decision;
                                           std::string error;
                                           const bool ok = MoEDeviceRebalanceHostRendezvous::rendezvousCompletionReadiness(
                                               &domain_key,
                                               generation,
                                               operation,
                                               2,
                                               "cuda:0",
                                               local0_ready,
                                               timeout_ms,
                                               &decision,
                                               &error);
                                           return std::make_pair(ok, decision);
                                       });

        auto participant1 = std::async(std::launch::async, [&]()
                                       {
                                           MoEDeviceRebalanceHostRendezvous::ReadinessDecision decision;
                                           std::string error;
                                           const bool ok = MoEDeviceRebalanceHostRendezvous::rendezvousCompletionReadiness(
                                               &domain_key,
                                               generation,
                                               operation,
                                               2,
                                               "cuda:1",
                                               local1_ready,
                                               timeout_ms,
                                               &decision,
                                               &error);
                                           return std::make_pair(ok, decision);
                                       });

        return std::make_pair(participant0.get(), participant1.get());
    };

    const auto partial = run_participants(true, false);
    ASSERT_TRUE(partial.first.first);
    ASSERT_TRUE(partial.second.first);
    EXPECT_FALSE(partial.first.second.all_ready);
    EXPECT_FALSE(partial.second.second.all_ready);
    EXPECT_EQ(partial.first.second.ready_count, 1u);
    EXPECT_EQ(partial.second.second.ready_count, 1u);

    const auto complete = run_participants(true, true);
    ASSERT_TRUE(complete.first.first);
    ASSERT_TRUE(complete.second.first);
    EXPECT_TRUE(complete.first.second.all_ready);
    EXPECT_TRUE(complete.second.second.all_ready);
    EXPECT_EQ(complete.first.second.ready_count, 2u);
    EXPECT_EQ(complete.second.second.ready_count, 2u);
}

TEST(Test__MoEDeviceRebalanceHostRendezvous, CompletionReadinessMissingParticipantFailsFast)
{
    int domain_key;
    MoEDeviceRebalanceHostRendezvous::ReadinessDecision decision;
    std::string error;

    const bool ok = MoEDeviceRebalanceHostRendezvous::rendezvousCompletionReadiness(
        &domain_key,
        /*generation=*/606,
        /*operation_key=*/13,
        /*expected_participants=*/2,
        "rocm:0",
        true,
        /*timeout_ms=*/5,
        &decision,
        &error);

    EXPECT_FALSE(ok);
    EXPECT_NE(error.find("timed out waiting for 2 readiness participants"), std::string::npos);
}
