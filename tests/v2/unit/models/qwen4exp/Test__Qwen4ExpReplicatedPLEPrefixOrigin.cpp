/**
 * @file Test__Qwen4ExpReplicatedPLEPrefixOrigin.cpp
 * @brief Read-only complete rollback declaration checks before any restore.
 *
 * These are pure metadata tests. No backend, Rank capability, native checkpoint
 * producer or prefix payload is fabricated. Only actual Rank capture can issue
 * an origin; canonical full-model prefix lifecycle remains a separate gate.
 */
#include "models/qwen4exp/Qwen4ExpReplicatedPLEPrefixOrigin.h"
#include <gtest/gtest.h>
#include <array>
#include <stdexcept>
using namespace llaminar2;
namespace {
/** @return Two immutable declaration-only handles with distinct original owner/event identities. */
PrefixStateSnapshot declaration()
{
    PrefixStateSnapshot result;result.valid=true;result.logical_checkpoint=true;
    result.cached_tokens=7;result.provenance=PrefixStateProvenance::LogicalCheckpoint;
    for(const int ordinal:{0,1}) {
        PrefixStateSnapshot child;child.valid=true;child.logical_checkpoint=true;
        child.cached_tokens=7;child.provenance=result.provenance;
        DeviceKVSequenceStateCheckpoint checkpoint;checkpoint.cache_depth=-1;checkpoint.sequence_index=0;
        checkpoint.metadata_layer_count=48;checkpoint.bytes=16;checkpoint.device=DeviceId::cuda(ordinal);
        // These host allocations are opaque metadata identity sentinels only.
        // They are never read, submitted or represented as actual GPU owners.
        checkpoint.storage=std::make_shared<std::array<std::byte,16>>();
        checkpoint.physical_memory_lease=std::make_shared<int>(ordinal);
        checkpoint.ready_event=std::make_shared<int>(ordinal);
        child.device_sequence_state_checkpoints.push_back(std::move(checkpoint));
        child.ready_device=DeviceId::cuda(ordinal);
        result.participant_snapshots.push_back(std::move(child));
    }
    return result;
}
}
/** @test An unchanged copy shares its original immutable handles, without payload comparisons. */
TEST(Qwen4ExpReplicatedPLEPrefixOrigin, CompleteOriginalChildDeclarationsRemainReusable)
{
    const auto original=declaration();const auto copy=original;
    EXPECT_NO_THROW(Qwen4ExpReplicatedPLEPrefixOrigin::requireDeclaration(original,copy));
    EXPECT_EQ(original.participant_snapshots[0].device_sequence_state_checkpoints[0].storage,
        copy.participant_snapshots[0].device_sequence_state_checkpoints[0].storage);
    EXPECT_NO_THROW(Qwen4ExpReplicatedPLEPrefixOrigin::requireDeclaration(original,original));
}
/** @test A later invalid participant cannot be noticed only after an earlier child mutates. */
TEST(Qwen4ExpReplicatedPLEPrefixOrigin, EveryChildAndOuterDeclarationRejectsBeforeRestore)
{
    const auto original=declaration();
    for(const int change:{0,1,2,3,4,5,6,7,8,9}) {
        auto supplied=original;
        switch(change) {
        case 0:supplied.participant_snapshots.pop_back();break;
        case 1:std::swap(supplied.participant_snapshots[0],supplied.participant_snapshots[1]);break;
        case 2:supplied.participant_snapshots[1].cached_tokens++;break;
        case 3:supplied.participant_snapshots[1].device_sequence_state_checkpoints[0].device=DeviceId::cuda(9);break;
        case 4:supplied.participant_snapshots[1].device_sequence_state_checkpoints[0].storage=std::make_shared<int>(0);break;
        case 5:supplied.participant_snapshots[1].device_sequence_state_checkpoints[0].ready_event=std::make_shared<int>(0);break;
        case 6:supplied.participant_snapshots[1].device_sequence_state_checkpoints[0].bytes++;break;
        case 7:supplied.participant_snapshots[1].mtp_shifted_origin=MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero;break;
        case 8:supplied.cached_tokens++;break;
        case 9:supplied.participant_snapshots[1].ready_event_valid=true;break;
        }
        EXPECT_THROW(Qwen4ExpReplicatedPLEPrefixOrigin::requireDeclaration(original,supplied),std::invalid_argument);
        EXPECT_NO_THROW(Qwen4ExpReplicatedPLEPrefixOrigin::requireDeclaration(original,original));
    }
}
