/**
 * @file Test__MoEOverlayMeasurementCompletionAllocations.cpp
 * @brief Executable allocation witness for actual sealed evidence validation and journal handoff.
 *
 * Allocation replacements belong only to this dedicated device-free executable.
 * The counter is armed around production validation/record calls, excluding
 * fixture construction, GTest formatting and exchange/consume preparation.
 * These passive observations certify no device timing or native readiness.
 */
#include "execution/moe/MoEOverlayMigrationMeasurementExchange.h"
#include "execution/moe/MoEOverlayMigrationMeasurementLedger.h"
#include "utils/RecordedMigrationEvidence.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <new>
namespace {
thread_local bool counting=false;
thread_local std::size_t allocations=0;
/** @brief Observe only allocations on the exact completion thread under test. */
void observe() noexcept {if(counting)++allocations;}
}
/** @brief Count ordinary C++ allocation in this dedicated test executable. */
void *operator new(std::size_t size) {if(auto *p=std::malloc(std::max(size,std::size_t(1)))){observe();return p;}throw std::bad_alloc();}
/** @brief Count array allocation through the same witness. */
void *operator new[](std::size_t size){return ::operator new(size);}
/** @brief Preserve requested alignment while counting C++ storage. */
void *operator new(std::size_t size,std::align_val_t alignment){void *p=nullptr;if(posix_memalign(&p,std::size_t(alignment),std::max(size,std::size_t(1))))throw std::bad_alloc();observe();return p;}
/** @brief Count aligned arrays. */
void *operator new[](std::size_t size,std::align_val_t alignment){return ::operator new(size,alignment);}
/** @brief Release ordinary test-process storage. */
void operator delete(void *p)noexcept{std::free(p);}
/** @brief Release ordinary arrays. */
void operator delete[](void *p)noexcept{std::free(p);}
/** @brief Release sized ordinary storage. */
void operator delete(void *p,std::size_t)noexcept{std::free(p);}
/** @brief Release sized ordinary arrays. */
void operator delete[](void *p,std::size_t)noexcept{std::free(p);}
/** @brief Release aligned storage. */
void operator delete(void *p,std::align_val_t)noexcept{std::free(p);}
/** @brief Release aligned arrays. */
void operator delete[](void *p,std::align_val_t)noexcept{std::free(p);}
/** @brief Release sized aligned storage. */
void operator delete(void *p,std::size_t,std::align_val_t)noexcept{std::free(p);}
/** @brief Release sized aligned arrays. */
void operator delete[](void *p,std::size_t,std::align_val_t)noexcept{std::free(p);}
namespace llaminar2::test {
TEST(MeasurementCompletionAllocations, SealedValidationAndOwnedJournalAllocateNothing)
{
    MoEOverlayPhysicalProjectionTarget source{.tier_idx=0},destination{.tier_idx=1};
    MoEOverlayPhysicalProjectionMember whole;whole.participant_id=0;whole.device=DeviceId::cpu();
    whole.projections={{{MoEProjectionResidence::ExpertOwner,128,32,0,128,0,32},
        {MoEProjectionResidence::ExpertOwner,128,32,0,128,0,32},
        {MoEProjectionResidence::ExpertOwner,32,128,0,32,0,128}}};source.members.push_back(whole);
    for(int member=0;member<2;++member){auto part=whole;part.participant_id=member+1;
        part.projections={{{MoEProjectionResidence::ParticipantOutputSlice,128,32,member*64,64,0,32},
            {MoEProjectionResidence::ParticipantOutputSlice,128,32,member*64,64,0,32},
            {MoEProjectionResidence::ParticipantInputSlice,32,128,0,32,member*64,64}}};destination.members.push_back(part);}
    const auto format=ExpertWeightFormat::floating(TensorType::FP32);
    const auto geometry=MoEOverlayMigrationMeasurementGeometry::checked({source,destination},{format,format,format});
    MoEOverlayCompletedMigrationMeasurement row{.expected_epoch=1,.candidate_epoch=2,
        .source_participant=0,.destination_participant=-1,.layer=0,.expert=1,.wave_wall_nanoseconds=1,.geometry=geometry};
    row.piece_observations.resize(geometry->pieces().size());
    for(auto &slot:row.piece_observations) {
        slot.quoted_payload_bytes=8192;
        slot.observation=ExpertTierProjectionTransferMeasurement{.sequence=1,.bytes=8192,.wall_nanoseconds=1,.host_nanoseconds=1};
    }
    std::vector<MoEOverlayCompletedMigrationMeasurement> wave;wave.push_back(std::move(row));
    const auto *original=wave.data();const auto *samples=wave.front().piece_observations.data();
    MoEOverlayMigrationMeasurementJournal journal({.maximum_migrations_per_wave=1});
    allocations=0;counting=true;
    const bool valid=wave.front().valid();
    const bool recorded=journal.recordCompletedWave(std::move(wave),nullptr);
    counting=false;const auto count=allocations;
    EXPECT_TRUE(valid);EXPECT_TRUE(recorded);EXPECT_EQ(count,0u);
    std::vector<MoEOverlayCompletedMigrationMeasurement> received;
    ASSERT_TRUE(journal.consume(&received,nullptr));EXPECT_EQ(received.data(),original);
    EXPECT_EQ(received.front().piece_observations.data(),samples);EXPECT_EQ(received.front().geometry,geometry);
}
TEST(MeasurementCompletionAllocations, OriginalScalarConstCopyStillUsesReservedStorage)
{
    MoEOverlayCompletedMigrationMeasurement row{.expected_epoch=1,.candidate_epoch=2,.source_participant=0,
        .destination_participant=1,.layer=0,.expert=0,.wave_wall_nanoseconds=1};
    const std::vector<MoEOverlayCompletedMigrationMeasurement> wave{row};
    MoEOverlayMigrationMeasurementJournal journal({.maximum_migrations_per_wave=1});
    allocations=0;counting=true;const bool recorded=journal.recordCompletedWave(wave,nullptr);counting=false;
    const auto count=allocations;EXPECT_TRUE(recorded);EXPECT_EQ(count,0u);
}
}

namespace llaminar2::test {
TEST(MeasurementCompletionAllocations, DirectLedgerSingletonProjectionAllocatesNothingAndRejectsBeforeMutation) {
    MoEOverlayCompletedMigrationMeasurement scalar{.expected_epoch=1,.candidate_epoch=2,.source_participant=0,
        .destination_participant=1,.layer=0,.expert=0,.wave_wall_nanoseconds=1};
    for(auto &sample:scalar.projections)sample=ExpertTierProjectionTransferMeasurement{.sequence=1,.bytes=8192,
        .wall_nanoseconds=1,.host_nanoseconds=1};
    std::vector<MoEOverlayCompletedMigrationMeasurement> wave{scalar};
    const auto context=attachRecordedMigrationContext(wave);
    MoEOverlayMigrationMeasurementLedger ledger({.required_coordinates={{0,1,0}},.measurement_identity="recorded-allocation-witness"});
    allocations=0;counting=true;const bool accepted=ledger.recordCompletedWave(wave,nullptr);counting=false;
    const auto count=allocations;EXPECT_TRUE(accepted);EXPECT_EQ(count,0u);
    const auto before=ledger.stats();
    auto partial=wave;partial[0].piece_observations[0].observation.reset();
    allocations=0;counting=true;const bool rejected_partial=ledger.recordCompletedWave(partial,nullptr);counting=false;
    const auto partial_count=allocations;EXPECT_FALSE(rejected_partial);EXPECT_EQ(partial_count,0u);
    auto malformed=wave;malformed[0].piece_observations[0].observation->bytes++;
    allocations=0;counting=true;const bool rejected_malformed=ledger.recordCompletedWave(malformed,nullptr);counting=false;
    const auto malformed_count=allocations;EXPECT_FALSE(rejected_malformed);EXPECT_EQ(malformed_count,0u);
    auto cohort=wave;const auto format=ExpertWeightFormat::floating(TensorType::FP32);
    cohort[0].geometry=MoEOverlayMigrationMeasurementGeometry::checked(
        {recordedMigrationTarget(1,0,0,0),recordedMigrationTarget(2,1,1,1)},{format,format,format});
    cohort[0].destination_participant=-1;cohort[0].piece_observations.resize(6);
    for(auto &slot:cohort[0].piece_observations)slot={.quoted_payload_bytes=8192,
        .observation=ExpertTierProjectionTransferMeasurement{.sequence=1,.bytes=8192,.wall_nanoseconds=1,.host_nanoseconds=1}};
    allocations=0;counting=true;const bool rejected_cohort=ledger.recordCompletedWave(cohort,nullptr);counting=false;
    const auto cohort_count=allocations;EXPECT_FALSE(rejected_cohort);EXPECT_EQ(cohort_count,0u);
    const auto after=ledger.stats();EXPECT_EQ(after.completed_waves_observed,before.completed_waves_observed);
    EXPECT_EQ(after.migration_observations_seen,before.migration_observations_seen);
    EXPECT_EQ(after.stored_migration_observations,before.stored_migration_observations);
}
}
