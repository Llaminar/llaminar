/**
 * @file Test__PooledImmutableRetirementBody.cpp
 * @brief Actual read/decoder→last-reader gate→physical release/retirement common replay.
 *
 * This isolated test TU reuses the original read replay helpers verbatim rather
 * than inventing a successful reader. Its decoded CPU roots are not native
 * storage/transport/capture evidence; all gate and retirement bodies are real.
 */
#include "../../unit/kernels/qwen4exp/Test__PooledImmutableRead.cpp"
#include "kernels/qwen4exp/PooledImmutableRetirementOps.h"
namespace llaminar2::test {
/** @brief Stop only the old fixture's cleanup after the new actual terminal has already completed it. */
struct OriginalFixtureTerminalReplaced {};
/** @brief Stop a deliberately invalid CPU trace; actual native gate dispatch traps before any release. */
struct InvalidReaderGateObserved {};
/** @brief Execute the actual source read requests/imports/seal and original-order decoder before retirement. */
void terminalReadOutput(ReadReplay& replay){
    ASSERT_TRUE(sr::planReads(replay.args));for(uint32_t owner=0;owner<replay.pool.owners.size();++owner)replay.import(owner);
    ASSERT_TRUE(sr::sealReadRows(replay.args));ASSERT_EQ(sr::stage(replay.args),sr::ResponseStage::RowsReady);
    originalOrder<TensorType::IQ4_NL>(replay);
}
TEST(PooledImmutableRetirementBody, ActualOutputGateAndCancellationKeepReadyRowsUntilEveryOriginalRelease){
    for(const auto capacities:{std::vector<uint32_t>{3},std::vector<uint32_t>{3,0,2},std::vector<uint32_t>{2,0,2,0,1,0,1,0}})
    for(unsigned cancellation=0;cancellation<3;++cancellation){
        Pool pool(capacities);pool.complete(std::array<uint64_t,4>{101,102,103,104});
        pool.begin(std::array<uint64_t,6>{104,101,104,105,102,101});pool.wave<false>();pool.wave<true>();
        const auto transaction=[&]{pool.finish([&](const n::SourceFrontier& actual){
            ReadReplay replay(pool,actual,GGUFTensorType::IQ4_NL);auto& args=replay.args;
            // Recorded native setup initializes these original reporting words;
            // the older read-only fixture deliberately leaves unrelated padding poisoned.
            *n::at<uint64_t>(args.native,args.plan.offsets.observation_ordinal)=0;
            *n::at<uint64_t>(args.native,args.plan.offsets.reported_ordinal)=0;
std::array<int32_t,sampling_math::kDeviceGenerationControlCount> controller{};args.control=controller.data();
            terminalReadOutput(replay);const auto responses=std::vector<std::byte>(args.frame+args.plan.offsets.responses,args.frame+args.plan.offsets.responses+pool.view().frame->unique_rows*args.plan.owner.row_bytes);
            if(cancellation==1){replay.error=static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure);replay.healthy=0;}ASSERT_TRUE(sr::retireReaders(args));ASSERT_TRUE(sr::planReleases(args));
            for(uint32_t owner=0;owner<pool.owners.size();++owner){auto& x=pool.owners[owner]->args;ASSERT_TRUE(sr::packReaderRetirement(args,owner));
                std::memcpy(x.native+x.plan.native.header,args.native+args.plan.offsets.seal_receipt,sizeof(n::ReadersRetiredGate));
                x.input={pool.invocation(),nullptr,nullptr,x.native+x.plan.native.header,nullptr,nullptr};
                ASSERT_TRUE((n::owner<n::OwnerAction::AcquireReaderRetirement,a::Slot64>(x)));
                const auto& acquired=*n::at<n::SourceFrontier>(x.native,x.plan.native.source_outcome);
                EXPECT_TRUE(n::same(acquired.interval,sr::frontier(args).interval));EXPECT_TRUE(n::same(acquired.source,sr::frontier(args).source));
                EXPECT_EQ(acquired.disposition,sr::frontier(args).disposition);EXPECT_EQ(acquired.version,n::Version);
                EXPECT_EQ(n::at<n::SourceFrontier>(x.native,x.plan.native.source_outcome)->status,sr::frontier(args).status);
            }
            if(cancellation==2){replay.error=static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure);replay.healthy=0;} // Cancellation after all original gate sends still reconciles actual released data truthfully.
            uint64_t total=0;
            for(uint32_t owner=0;owner<pool.owners.size();++owner){auto& x=pool.owners[owner]->args;x.input={pool.invocation(),nullptr,nullptr,x.native+x.plan.native.source_outcome,nullptr,nullptr};
                ASSERT_TRUE((n::owner<n::OwnerAction::DrainSourceFailure,a::Slot64>(x)));x.input={pool.invocation(),nullptr,nullptr,nullptr,nullptr,nullptr};
                ASSERT_TRUE((n::owner<n::OwnerAction::Release,a::Slot64>(x)));const auto bytes=n::progress(x).release_bytes;total+=bytes;
                if(bytes)std::memcpy(args.frame+args.plan.offsets.fills,x.frame+x.plan.frame.release_receipts,bytes);
                *n::at<uint64_t>(args.native,args.plan.offsets.record_bytes)=x.plan.native.reply_capacity?bytes:~uint64_t{0};
                ASSERT_TRUE(sr::importRemoteReleases(args,owner));
            }
            EXPECT_EQ(sr::releasedBytes(args),total);ASSERT_TRUE(sr::reconcileReleases(args));ASSERT_TRUE(sr::snapshotReleasedObservation(args));
            for(uint32_t owner=0;owner<pool.owners.size();++owner){auto& x=pool.owners[owner]->args;
                ASSERT_TRUE((n::owner<n::OwnerAction::Retire,a::Slot64>(x)));ASSERT_TRUE((n::owner<n::OwnerAction::RetiredWire,a::Slot64>(x)));
                std::memcpy(args.native+args.plan.offsets.seal_receipt,x.native+x.plan.native.header,60);
                ASSERT_TRUE(sr::importRemoteRetired(args,owner));EXPECT_EQ(n::ownerView<a::Slot64>(x).frame->held_pins,0u);EXPECT_EQ(n::ownerView<a::Slot64>(x).frame->reservations,0u);
            }
            ASSERT_TRUE(sr::finishReleased(args));ASSERT_TRUE(sr::observeReleased(args));EXPECT_EQ(pool.view().frame->phase,p::LookupPhase::Idle);
            EXPECT_EQ(std::memcmp(responses.data(),args.frame+args.plan.offsets.responses,responses.size()),0);
            q::QwenAdaptiveCounterTotals totals;ASSERT_TRUE(q::readQwenAdaptiveCounters(controller.data(),controller.size(),totals));EXPECT_EQ(totals.calls,1u);EXPECT_EQ(totals.failed_frames,cancellation?1u:0u);EXPECT_EQ(totals.completed_live_ids,cancellation?0u:6u);
            throw OriginalFixtureTerminalReplaced{};
        });};EXPECT_THROW(transaction(),OriginalFixtureTerminalReplaced);
        EXPECT_NO_THROW(pool.complete(std::array<uint64_t,1>{101},2));
    }
}
TEST(PooledImmutableRetirementBody, MissingWrongKindOldOrUnretiredGateNeverMutatesOutcomeOrReleasesPins){
    for(unsigned defect=0;defect<7;++defect){Pool pool(std::array{3u});pool.complete(std::array<uint64_t,2>{101,102});pool.begin(std::array<uint64_t,2>{102,103});pool.wave<false>();pool.wave<true>();
        const auto transaction=[&]{pool.finish([&](const n::SourceFrontier& actual){ReadReplay replay(pool,actual,GGUFTensorType::IQ4_NL);auto& args=replay.args;
            // Recorded native setup initializes these original reporting words;
            // the older read-only fixture deliberately leaves unrelated padding poisoned.
            *n::at<uint64_t>(args.native,args.plan.offsets.observation_ordinal)=0;
            *n::at<uint64_t>(args.native,args.plan.offsets.reported_ordinal)=0;
terminalReadOutput(replay);
            ASSERT_TRUE(sr::retireReaders(args));ASSERT_TRUE(sr::planReleases(args));ASSERT_TRUE(sr::packReaderRetirement(args,0));auto& x=pool.owners[0]->args;
            std::memcpy(x.native+x.plan.native.header,args.native+args.plan.offsets.seal_receipt,64);*n::at<n::SourceFrontier>(x.native,x.plan.native.source_outcome)=actual;
            const auto before=*n::at<n::SourceFrontier>(x.native,x.plan.native.source_outcome);const auto pins=n::ownerView<a::Slot64>(x).frame->held_pins;
            auto* gate=n::at<n::ReadersRetiredGate>(x.native,x.plan.native.header);
            if(defect==1)gate->kind=static_cast<uint32_t>(n::OwnerHeaderKind::HitRead);
            if(defect==2)--gate->version;
            if(defect==3)--gate->interval.sequence;
            if(defect==4)++gate->source.generation;
            if(defect==5)gate->disposition=n::SourceDisposition::Published;
            if(defect==6)gate->kind=static_cast<uint32_t>(n::OwnerHeaderKind::SourceSeal);
            x.input={pool.invocation(),nullptr,nullptr,defect==0?nullptr:gate,nullptr,nullptr};
            EXPECT_FALSE((n::owner<n::OwnerAction::AcquireReaderRetirement,a::Slot64>(x)));
            EXPECT_EQ(std::memcmp(&before,x.native+x.plan.native.source_outcome,sizeof(before)),0);EXPECT_EQ(n::ownerView<a::Slot64>(x).frame->held_pins,pins);EXPECT_EQ(n::progress(x).released,0u);
            throw InvalidReaderGateObserved{};
        });};EXPECT_THROW(transaction(),InvalidReaderGateObserved);
    }
}
TEST(PooledImmutableRetirementBody, ReaderPhaseCannotReplaceActualFullIssuedDrainAndNominalRetiredCannotBeProjectedEarly){
    for(unsigned defect=0;defect<3;++defect){Pool pool(std::array{2u,0u});pool.begin(std::array<uint64_t,1>{101});pool.wave<false>();pool.wave<true>();
        const auto transaction=[&]{pool.finish([&](const n::SourceFrontier& actual){ReadReplay replay(pool,actual,GGUFTensorType::IQ4_NL);auto& args=replay.args;
            // Recorded native setup initializes these original reporting words;
            // the older read-only fixture deliberately leaves unrelated padding poisoned.
            *n::at<uint64_t>(args.native,args.plan.offsets.observation_ordinal)=0;
            *n::at<uint64_t>(args.native,args.plan.offsets.reported_ordinal)=0;
terminalReadOutput(replay);
            if(defect==0)--sr::progress(args).read_owners;if(defect==1)sr::stage(args)=sr::ResponseStage::ReadPlanned;if(defect==2)sr::stage(args)=sr::ResponseStage::Consumed;
            EXPECT_FALSE(sr::retireReaders(args));EXPECT_NE(sr::stage(args),sr::ResponseStage::ReadersComplete);
            auto& x=pool.owners[0]->args;x.input={pool.invocation(),nullptr,nullptr,nullptr,nullptr,nullptr};
            EXPECT_FALSE((n::owner<n::OwnerAction::RetiredWire,a::Slot64>(x)));EXPECT_EQ(n::progress(x).header_bytes,0u);EXPECT_EQ(n::progress(x).header_kind,n::OwnerHeaderKind::None);
            throw InvalidReaderGateObserved{};
        });};EXPECT_THROW(transaction(),InvalidReaderGateObserved);
    }
}
}
