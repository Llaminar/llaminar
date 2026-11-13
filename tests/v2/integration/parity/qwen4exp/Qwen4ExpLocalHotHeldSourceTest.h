/**
 * @file Qwen4ExpLocalHotHeldSourceTest.h
 * @brief Actual Rank/cohort prefills proving unequal local progress under held original source.
 *
 * A fresh original mapped source request authenticates every current consumer
 * Probe before the existing mapped ReadyReadComplete word is inspected. This
 * avoids stale warm Ready2 evidence without a new GPU probe/event or pause hook.
 * Native execution, original source work and whole-parent retirement remain
 * with the public runner. This is not throughput or frozen-panel qualification.
 */
#pragma once
#include "Qwen4ExpLocalHotHeldSourceCases.h"
#include "Qwen4ExpParityTestBase.h"
#include "fixtures/qwen4exp/ImmutableLookupWorkerGateReplayAccess.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotReadyProtocol.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <thread>

namespace llaminar2::test::parity::qwen4exp {
/** @brief Real model fixture; no standalone source/cohort issuer or alternate launcher. */
class Qwen4ExpLocalHotHeldSourceTest
    : public Qwen4ExpConfigDrivenParityTest<Qwen4ExpLocalHotHeldSourceTest>, public ModelParityCaseParameter
{
    using Base = Qwen4ExpConfigDrivenParityTest<Qwen4ExpLocalHotHeldSourceTest>;
    using Access = ::llaminar2::test::ImmutableLookupWorkerGateReplayAccess;
protected:
    /** @return Existing real-source/native proof setup; no HF scores are claimed here. */
    MTPParityProofPurpose mtpParityProofPurpose() const noexcept override
    { return MTPParityProofPurpose::NativeExactReplay; }
    /** @brief Native causality cannot regenerate numerical reference rows. */
    bool regeneratePyTorchSnapshots() override
    { ADD_FAILURE() << "Held-source native proof cannot generate reference snapshots"; return false; }
    /**
     * @brief Warm actual public history, then repeat with the same original source worker held.
     * Captured LocalRead completes on the all-hit consumer before source release,
     * while the C0 peer and whole request remain pending. Reset and repeat twice.
     */
    void runHeldSourceProof()
    {
        ASSERT_EQ(modelParityCase().mtp, ModelParityMTP::Off);
        ASSERT_NO_FATAL_FAILURE(Base::beginProductionParitySnapshotAdmission());
        ASSERT_NO_FATAL_FAILURE(Base::setupProductionParityPipeline());
        auto& runner = *Base::orch_runner_;
        const auto& contract = Base::qfnObservationOwner();
        ASSERT_TRUE(contract.context);
        const auto placement = contract.context->resolvedImmutableRowTablePlacement();
        ASSERT_TRUE(placement && placement->isGPUAdaptive());
        const auto& request = placement->gpuAdaptiveRequest();
        ASSERT_TRUE(request.local_hot_cache);
        const auto consumers = placement->gpuAdaptiveLayout().consumers();
        ASSERT_EQ(consumers.size(), 2u);
        ASSERT_TRUE(request.placement.coordinator);
        const auto coordinator = *request.placement.coordinator;
        const auto boundary = contract.context->adaptiveImmutableLookupBoundary(coordinator.device());
        ASSERT_TRUE(boundary && boundary->requiresSourceGroupAdmission());
        const auto participants = contract.context->adaptiveImmutableLookupParticipants(coordinator.device());
        ASSERT_TRUE(participants);
        ASSERT_EQ(participants->boundary(), boundary);
        ASSERT_NO_THROW(participants->requireCurrent());
        const auto transport = participants->transport();
        ASSERT_TRUE(transport && transport->localHotGeometry() && transport->readerGeometry());
        ASSERT_EQ(transport->readerGeometry()->consumers().size(), consumers.size());
        ASSERT_EQ(transport->readers().size(), 1u);
        const auto native_storage = transport->readers().front()->nativeResult().nativeStorage();
        ASSERT_TRUE(native_storage);
        ASSERT_NO_THROW(native_storage->requireCurrent());
        ASSERT_EQ(transport->readers().front()->consumerCount(), consumers.size());
        // The model owns the original boundary/participants. Native transport
        // additionally admits storage-only devices without making them readers.
        for (const auto& consumer : consumers)
        {
            const auto coordinate = native_storage->coordinate(consumer.device);
            const auto original = native_storage->context();
            ASSERT_EQ(original.context().devices().at(coordinate).toLocalDeviceId(), consumer.device);
        }
        const auto original_service = Access::originalService(*boundary);
        ASSERT_TRUE(original_service);
        std::array<std::shared_ptr<::llaminar2::qwen4exp::PooledImmutableLocalHotPreparedStorage>, 2> locals;
        std::array<std::uint64_t, 2> identities;
        std::size_t positive = 2;
        for (std::size_t index = 0; index < consumers.size(); ++index)
        {
            locals[index] = contract.context->adaptiveImmutableLookupLocalHot(consumers[index].device);
            ASSERT_TRUE(locals[index]);
            ASSERT_NO_THROW(locals[index]->requireAuthority(contract.physical_memory_authority));
            ASSERT_EQ(locals[index]->originalSource(), boundary->pooledStorage());
            identities[index] = locals[index]->captureIdentity();
            const auto authored = request.local_hot_cache->ceilingsFor(
                {consumers[index].world_rank, consumers[index].device.type, consumers[index].device.ordinal});
            if (authored.cache_physical_byte_limit)
            {
                ASSERT_EQ(positive, 2u);
                positive = index;
                ASSERT_GT(locals[index]->geometry().slots(), 0u);
            }
            else ASSERT_EQ(locals[index]->geometry().slots(), 0u);
        }
        ASSERT_LT(positive, 2u);
        const auto miss = 1 - positive;
        const auto source_identity = boundary->captureIdentity();
        const std::vector<int32_t> prompt(Base::config_.token_ids.begin(), Base::config_.token_ids.end());
        ASSERT_FALSE(prompt.empty());
        ASSERT_TRUE(runner.prefill(prompt)) << runner.lastError();
        ASSERT_GT(runner.vocabSize(), 0);
        // The canonical diagnostic stage snapshot is already enabled by Base.
        // Rank logits remain device-owned; never request a serving host export.
        const auto original_snapshot = Base::activeTypedSnapshot("LM_HEAD");
        ASSERT_NE(original_snapshot.data, nullptr);
        ASSERT_EQ(original_snapshot.cols, static_cast<size_t>(runner.vocabSize()));
        ASSERT_GT(original_snapshot.rows, 0u);
        ASSERT_LE(original_snapshot.rows, prompt.size());
        ASSERT_EQ(original_snapshot.size, original_snapshot.rows * original_snapshot.cols);
        const std::vector<float> original_logits(
            original_snapshot.data, original_snapshot.data + original_snapshot.size);
        for (int replay = 0; replay < 2; ++replay)
        {
            runner.clearCache();
            // Clear only the public prefix archive so this explicit source
            // proof executes a genuine current query. Graphs and L1 stay live.
            ASSERT_TRUE(runner.purgePrefixCache()) << runner.lastError();
            Access::Gate gate(original_service);
            const auto previous = Access::retiredSequence(gate, coordinator.device());
            const auto before = original_service->stats();
            Access::installNextOriginalDispatch(gate);
            auto pending = std::async(std::launch::async, [&] { return runner.prefill(prompt); });
            struct ReleaseBeforeJoin
            {
                Access::Gate& gate;
                std::future<bool>& pending;
                /** @brief Assertion/exception cannot strand the original captured source request. */
                ~ReleaseBeforeJoin()
                {
                    gate.release();
                    if (pending.valid()) pending.wait();
                }
            } cleanup{gate, pending};
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            const auto require_pending = [&]
            {
                if (pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
                    throw std::runtime_error("Current source request completed before its held original worker");
                if (std::chrono::steady_clock::now() >= deadline)
                    throw std::runtime_error("Current unequal-consumer source proof exceeded its standard deadline");
                std::this_thread::yield();
            };
            while (!gate.entered()) require_pending();
            std::optional<::llaminar2::qwen4exp::ImmutableLookupRequest> current;
            while (!(current = Access::currentRequest(gate, coordinator.device(), previous))) require_pending();
            ASSERT_GT(current->idCount, 0u);
            ASSERT_TRUE(gate.held());
            // This original CPU publication follows every compact current
            // Request/Probe, hence current Reset. A retained warm2 cannot pass.
            const auto ready = [](const auto& local)
            {
                auto* word = static_cast<std::uint64_t*>(local->readyRegion()->mutableHostData());
                return std::atomic_ref(*word).load(std::memory_order_acquire);
            };
            while (ready(locals[positive]) != ::llaminar2::qwen4exp::local_hot::ReadyReadComplete)
                require_pending();
            EXPECT_EQ(ready(locals[miss]), ::llaminar2::qwen4exp::local_hot::ReadyEmpty);
            EXPECT_EQ(pending.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
            EXPECT_TRUE(gate.held());
            gate.release();
            ASSERT_TRUE(pending.get()) << runner.lastError();
            const auto replay_snapshot = Base::activeTypedSnapshot("LM_HEAD");
            ASSERT_NE(replay_snapshot.data, nullptr);
            ASSERT_EQ(replay_snapshot.rows, original_snapshot.rows);
            ASSERT_EQ(replay_snapshot.cols, original_snapshot.cols);
            ASSERT_EQ(replay_snapshot.size, original_logits.size());
            EXPECT_EQ(std::memcmp(replay_snapshot.data, original_logits.data(),
                                  original_logits.size() * sizeof(float)), 0);
            runner.clearCache();
            const auto after = original_service->stats();
            ASSERT_GE(after.ids, before.ids + current->idCount);
            EXPECT_EQ(after.nativeBytes - before.nativeBytes, (after.ids - before.ids) * current->rowBytes);
            EXPECT_EQ(after.idBytes - before.idBytes, (after.ids - before.ids) * sizeof(std::int64_t));
            EXPECT_FALSE(boundary->hasActiveInterval());
            EXPECT_FALSE(gate.timedOut());
            EXPECT_EQ(boundary->captureIdentity(), source_identity);
            for (std::size_t index = 0; index < locals.size(); ++index)
                EXPECT_EQ(locals[index]->captureIdentity(), identities[index]);
        }
        if (::testing::Test::HasFailure()) return;
        std::ofstream record(Base::ensureResultsDir() / "held_source_causality.txt");
        ASSERT_TRUE(record);
        record << "actual_rank_cohort=1\nsource_pool_c0=1\npublic_prefill_replays=2\n"
            << "all_hit_read_before_source_release=1\nfresh_mapped_source_frontier=1\n"
            << "original_logit_bytes_equal=1\nhf_scoring=0\nperformance_qualified=0\n";
        record.flush();
        ASSERT_TRUE(record.good());
    }
};
} // namespace llaminar2::test::parity::qwen4exp
