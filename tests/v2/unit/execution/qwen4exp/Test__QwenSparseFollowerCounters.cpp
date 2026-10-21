/**
 * @file Test__QwenSparseFollowerCounters.cpp
 * @brief Exact follower admission reset and terminal-result byte extent regressions.
 *
 * Injectable submission callbacks exercise the production span/observation
 * helpers with host-owned test words. No GPU, model, allocator or controller
 * mirror is created; the native complete-frontier proof is registered separately.
 */
#include "execution/qwen4exp/QwenSparseFollowerCounters.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
using namespace llaminar2;using namespace llaminar2::qwen4exp;using namespace llaminar2::sampling_math;
/** @brief Two dirty requests reset only52..59, preserving all command/policy words and outside guards. */
TEST(QwenSparseFollowerCounters, AdmissionResetsExactReportingSpanAcrossDirtyRequests){
    alignas(8)std::array<std::int32_t,kDeviceGenerationControlCount+4>control;control.fill(0x12345678);
    for(int request=0;request<2;++request){
        const auto before=control;int calls=0;std::size_t bytes=0;
        ASSERT_TRUE(resetQwenSparseFollowerCounters({control.data(),kDeviceGenerationControlCount},
            [&](void* destination,std::size_t count){++calls;bytes+=count;
                EXPECT_EQ(destination,control.data()+52);std::memset(destination,0,count);return true;}));
        EXPECT_EQ(calls,1);EXPECT_EQ(bytes,32u);
        for(int word=0;word<kDeviceGenerationControlCount+4;++word)
            EXPECT_EQ(control[word],word>=52&&word<60 ? 0:before[word]);
        ASSERT_TRUE(addQwenSparseHostCounters(control.data(),kDeviceGenerationControlCount,
            {16,1536,0x100000007ull,1024}));
    }
}
/** @brief The existing terminal result owns one exact32B copy, including high-word carry and legitimate zero observations. */
TEST(QwenSparseFollowerCounters, TerminalObservationCopiesOnlyFourPairsAndNeverDuplicatesTail){
    alignas(8)std::array<std::int32_t,kDeviceGenerationControlCount>control{};
    for(auto values:{QwenSparseCounterDelta{16,1536,0x100000007ull,1024},QwenSparseCounterDelta{}}){
        control.fill(0);ASSERT_TRUE(addQwenSparseHostCounters(control.data(),kDeviceGenerationControlCount,values));
        QwenSparseFollowerCounterObservation observed;observed.words.fill(-91);int calls=0;std::size_t bytes=0;
        ASSERT_TRUE(observeQwenSparseFollowerCounters(QwenSparseTerminalOwner::PipelineFollower,{control.data(),kDeviceGenerationControlCount},observed,
            [&](void*destination,const void*source,std::size_t count){++calls;bytes+=count;
                EXPECT_EQ(source,control.data()+52);std::memcpy(destination,source,count);return true;}));
        EXPECT_EQ(calls,1);EXPECT_EQ(bytes,32u);const auto actual=observed.totals();
        EXPECT_EQ(actual.prior_header,values.prior_header);EXPECT_EQ(actual.prior_payload,values.prior_payload);
        EXPECT_EQ(actual.candidate_payload,values.candidate_payload);EXPECT_EQ(actual.accepted_payload,values.accepted_payload);
    }
    QwenSparseFollowerCounterObservation untouched;untouched.words.fill(-7);int tailCopies=0;
    EXPECT_TRUE(observeQwenSparseFollowerCounters(QwenSparseTerminalOwner::DeviceControllerTail,{},untouched,
        [&](void*,const void*,std::size_t){++tailCopies;return true;}));
    EXPECT_EQ(tailCopies,0);for(auto word:untouched.words)EXPECT_EQ(word,-7);
    static_assert(sizeof(QwenSparseFollowerCounterObservation)==32);
}
/** @brief Invalid bindings fail before native submission, and rejected submission cannot invent an observation. */
TEST(QwenSparseFollowerCounters, MalformedOrFailedSubmissionCannotPublishStaleBytes){
    alignas(8)std::array<std::int32_t,kDeviceGenerationControlCount+2>control{};QwenSparseFollowerCounterObservation observation;
    observation.words.fill(-17);const auto before=observation.words;int calls=0;
    const auto reset=[&](void*,std::size_t){++calls;return true;};
    const auto copy=[&](void*,const void*,std::size_t){++calls;return true;};
    EXPECT_FALSE(resetQwenSparseFollowerCounters({control.data()+1,kDeviceGenerationControlCount},reset));
    EXPECT_FALSE(observeQwenSparseFollowerCounters(QwenSparseTerminalOwner::PipelineFollower,{control.data(),52},observation,copy));EXPECT_EQ(calls,0);
    EXPECT_EQ(observation.words,before);
    EXPECT_FALSE(observeQwenSparseFollowerCounters(QwenSparseTerminalOwner::PipelineFollower,{control.data(),kDeviceGenerationControlCount},observation,
        [&](void*,const void*,std::size_t){++calls;return false;}));
    EXPECT_EQ(calls,1);EXPECT_EQ(observation.words,before);
}

/** @brief Appended native-send reporting words never enter the frozen QSA follower reset or terminal field. */
TEST(QwenSparseFollowerCounters, AdditionalReportingPairsNeverExpandFollowerResetOrCopy){
    static_assert(QwenSparseFollowerCounterObservation::first_word==52);
    static_assert(QwenSparseFollowerCounterObservation::last_word==59);
    static_assert(QwenSparseFollowerCounterObservation::word_count==8);
    static_assert(QwenSparseFollowerCounterObservation::byte_count==32);
    static_assert(kDeviceGenerationControlFixedHeadCallsLow==60);
    static_assert(kDeviceGenerationControlCount == 90);
    alignas(8)std::array<std::int32_t,kDeviceGenerationControlCount+2> control{};
    for(std::size_t word=0;word<control.size();++word)control[word]=static_cast<std::int32_t>(0x12340000+word);
    const auto before=control;
    ASSERT_TRUE(resetQwenSparseFollowerCounters({control.data(),kDeviceGenerationControlCount},
        [&](void *destination,std::size_t bytes){EXPECT_EQ(destination,control.data()+52);EXPECT_EQ(bytes,32u);
            std::memset(destination,0,bytes);return true;}));
    for(int word=kDeviceGenerationControlFixedHeadCallsLow;word<static_cast<int>(control.size());++word)
        EXPECT_EQ(control[word],before[word]);
    ASSERT_TRUE(addQwenSparseHostCounters(control.data(),kDeviceGenerationControlCount,{16,1536,0x100000007ull,1024}));
    QwenSparseFollowerCounterObservation result;
    ASSERT_TRUE(observeQwenSparseFollowerCounters(QwenSparseTerminalOwner::PipelineFollower,
        {control.data(),kDeviceGenerationControlCount},result,
        [&](void *destination,const void *source,std::size_t bytes){EXPECT_EQ(source,control.data()+52);EXPECT_EQ(bytes,32u);
            std::memcpy(destination,source,bytes);return true;}));
    const auto actual=result.totals();EXPECT_EQ(actual.prior_header,16u);EXPECT_EQ(actual.prior_payload,1536u);
    EXPECT_EQ(actual.candidate_payload,0x100000007ull);EXPECT_EQ(actual.accepted_payload,1024u);
    for(int word=kDeviceGenerationControlFixedHeadCallsLow;word<static_cast<int>(control.size());++word)
        EXPECT_EQ(control[word],before[word]);
}
