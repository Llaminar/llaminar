/**
 * @file QwenSparseFollowerCounterDeviceTest.h
 * @brief Native follower admission reset and exact once-only terminal observation on existing events.
 *
 * Included after the sealed terminal-counter native fixture, reusing its real
 * independent atomic writers and delayed branch. No GPU scratch or custom
 * statistics kernel is introduced by the production follower extension.
 */
#pragma once
#include "execution/qwen4exp/QwenSparseFollowerCounters.h"
namespace llaminar2::qwen4exp::QSC_SUITE {
/** @brief Two dirty requests and large/small/empty work retain exact32B terminal copies after complete writers. */
TEST(QSC_SUITE, FollowerResetAndTerminalCopyWaitForCompleteIndependentWriters){
    check(QSC_RT(SetDevice)(0));QSC_STREAM producer{},consumer{};
    check(QSC_RT(StreamCreateWithFlags)(&producer,QSC_RT(StreamNonBlocking)));
    check(QSC_RT(StreamCreateWithFlags)(&consumer,QSC_RT(StreamNonBlocking)));
    QSC_RT(Event_t) writerReady{},hostReady{};check(QSC_RT(EventCreateWithFlags)(&writerReady,QSC_RT(EventDisableTiming)));
    check(QSC_RT(EventCreateWithFlags)(&hostReady,QSC_RT(EventDisableTiming)));
    {
        Buffer<int>control(sampling_math::kDeviceGenerationControlCount),live(1),release(2);
        const QwenSparseWorkCounterRow row{control.data,sampling_math::kDeviceGenerationControlCount};
        auto body=std::make_unique<QSC_CAPTURE>(producer,0),open=std::make_unique<QSC_CAPTURE>(producer,0),
            worker=std::make_unique<QSC_CAPTURE>(producer,0),close=std::make_unique<QSC_CAPTURE>(producer,0),
            parent=std::make_unique<QSC_CAPTURE>(producer,0);
        ASSERT_TRUE(body->beginCapture());fastWriter<<<32,128,0,producer>>>(row,live.data,release.data+1);ASSERT_TRUE(body->endCapture());
        ASSERT_TRUE(open->beginCapture());marker<<<1,1,0,producer>>>();ASSERT_TRUE(open->endCapture());
        ASSERT_TRUE(worker->beginCapture());delayedWriter<<<1,32,0,producer>>>(row,live.data,release.data,false);ASSERT_TRUE(worker->endCapture());
        ASSERT_TRUE(close->beginCapture());marker<<<1,1,0,producer>>>();ASSERT_TRUE(close->endCapture());
        ASSERT_TRUE(body->appendParallelBranch({*open,*worker,*close}));
        const GPUOrderedTimelineStep steps[]{{.name="complete_follower_writers",.capture=body.get()}};
        ASSERT_TRUE(parent->buildOrderedTimelineTransaction(steps));ASSERT_TRUE(parent->instantiate());
        std::array<int,sampling_math::kDeviceGenerationControlCount>dirty;dirty.fill(0x12345678);control.upload(dirty.data(),producer);
        for(int active:{4096,1,0,4095,0}){
            int resets=0;ASSERT_TRUE(resetQwenSparseFollowerCounters(row,[&](void*source,std::size_t bytes){
                ++resets;EXPECT_EQ(source,control.data+52);EXPECT_EQ(bytes,32u);
                check(QSC_RT(MemsetAsync)(source,0,bytes,producer));return true;}));EXPECT_EQ(resets,1);
            const int zero[2]{};live.upload(&active,producer);release.upload(zero,producer);ASSERT_TRUE(parent->launch());
            // The original complete parent publishes one producer event. The
            // terminal stream joins it before the sole32B follower observation.
            check(QSC_RT(EventRecord)(writerReady,producer));check(QSC_RT(StreamWaitEvent)(consumer,writerReady,0));
            QwenSparseFollowerCounterObservation observed;observed.words.fill(-99);int copies=0;
            ASSERT_TRUE(observeQwenSparseFollowerCounters(QwenSparseTerminalOwner::PipelineFollower,row,observed,
                [&](void*destination,const void*source,std::size_t bytes){++copies;EXPECT_EQ(bytes,32u);
                    check(QSC_RT(MemcpyAsync)(destination,source,bytes,QSC_RT(MemcpyDeviceToHost),consumer));return true;}));
            check(QSC_RT(EventRecord)(hostReady,consumer));check(QSC_RT(EventSynchronize)(hostReady));
            EXPECT_EQ(copies,1);const auto totals=observed.totals();
            EXPECT_EQ(totals.candidate_payload,active ? 0x100000000ull+active :0u);
            EXPECT_EQ(totals.prior_header|totals.prior_payload|totals.accepted_payload,0u);
            const auto original=control.download(producer);
            for(int word=0;word<52;++word)EXPECT_EQ(original[word],dirty[word]);
            ASSERT_TRUE(observeQwenSparseFollowerCounters(QwenSparseTerminalOwner::DeviceControllerTail,row,observed,
                [&](void*,const void*,std::size_t){++copies;return true;}));EXPECT_EQ(copies,1);
        }
    }
    check(QSC_RT(EventDestroy)(hostReady));check(QSC_RT(EventDestroy)(writerReady));
    check(QSC_RT(StreamDestroy)(consumer));check(QSC_RT(StreamDestroy)(producer));
}
} // namespace llaminar2::qwen4exp::QSC_SUITE
