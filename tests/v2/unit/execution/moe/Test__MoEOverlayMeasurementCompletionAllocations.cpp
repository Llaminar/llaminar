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
