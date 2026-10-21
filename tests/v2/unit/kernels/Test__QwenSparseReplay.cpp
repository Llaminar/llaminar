/**
 * @file Test__QwenSparseReplay.cpp
 * @brief Device-free preappend admission and exact auxiliary copy-extent regressions.
 *
 * The original ring origin and absolute position are independent authorities.
 * Invalid admission must fail before publishing any replay/row state. Checkpoint
 * capacity retains inactive rows, but native transfers leave their poison bytes
 * untouched and preserve arbitrary integer/float bit patterns.
 */
#include "kernels/qwen4exp/QwenSparseReplay.h"
#include "kernels/qwen4exp/Qwen4ExpStateIO.h"
#include <gtest/gtest.h>
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "../../integration/kernels/qwen4exp/QwenSparseRotaryReference.h"
#include <array>
#include <cstring>
#include <vector>
namespace llaminar2::qwen4exp
{
TEST(QwenSparseReplay, PreAppendAuthorityAndAdmissionAreExact)
{
    constexpr int capacity=17,rows=5;
    int cursor=2,prefix=7,live=3;
    const int positions[rows]={107,108,109,-7,-7};
    QwenSparseReplay replay{-99,-99,-99};
    std::array<attention::AttentionDeviceParams,rows> metadata{};
    QwenSparsePrepareReplay p{{&cursor,&prefix,capacity,QwenSparseRingCursor::NextWriteRow},
        &live,positions,&replay,metadata.data(),rows,0};
    cpuQwenSparsePrepareReplay(p);
    EXPECT_EQ(replay.prefix_tokens,prefix);EXPECT_EQ(replay.live_rows,live);EXPECT_EQ(replay.position_origin,100);
    for(int row=0;row<rows;++row) {
        const auto &m=metadata[row];
        EXPECT_EQ(m.kv_len,row<live ? prefix+row+1 : 0);
        EXPECT_EQ(m.ring_row_origin,row<live ? 12 : 0);
        EXPECT_EQ(m.position_offset,row<live ? positions[row] : 0);
    }
    // An admitted full row bucket needs no fabricated constant-count tensor.
    p.live_rows=nullptr;p.position_ids=nullptr;p.first_position=107;
    cpuQwenSparsePrepareReplay(p);EXPECT_EQ(replay.live_rows,rows);
    p.sequence.encoding=QwenSparseRingCursor::OldestRow;cursor=12;
    cpuQwenSparsePrepareReplay(p);EXPECT_EQ(metadata[0].ring_row_origin,12);
    // Reject the same over-capacity transaction before a cache append can run.
    const auto prior=replay;const auto priorRows=metadata;prefix=capacity-1;
    EXPECT_THROW(cpuQwenSparsePrepareReplay(p),std::invalid_argument);
    EXPECT_EQ(std::memcmp(&prior,&replay,sizeof(prior)),0);
    EXPECT_EQ(std::memcmp(priorRows.data(),metadata.data(),sizeof(metadata)),0);
    prefix=7;p.live_rows=&live;p.position_ids=positions;
    int broken[rows]={107,109,110,-7,-7};p.position_ids=broken;
    EXPECT_THROW(cpuQwenSparsePrepareReplay(p),std::invalid_argument);
    EXPECT_EQ(std::memcmp(&prior,&replay,sizeof(prior)),0);
}
TEST(QwenSparseReplay, CheckpointCopiesOnlyCanonicalLiveTailAndRestoresNativeBits)
{
    constexpr std::size_t rowWords=128;
    std::array<std::uint32_t,3*rowWords> source,destination,restored;
    for(std::size_t i=0;i<source.size();++i)source[i]=0x7fc00000u+static_cast<unsigned>(i);
    for(int frontier=0;frontier<12;++frontier) {
        destination.fill(0xdeadbeefu);restored.fill(0xaabbccddu);int header=-1;
        cpuQwenStateTransfer({source.data(),destination.data(),rowWords*4,3,
            QwenStateRowSource::CanonicalModuloFour,&frontier,0,&header});
        ASSERT_EQ(header,frontier%4);
        for(std::size_t i=0;i<source.size();++i)EXPECT_EQ(destination[i],i<std::size_t(header)*rowWords ? source[i] : 0xdeadbeefu);
        cpuQwenStateTransfer({destination.data(),restored.data(),rowWords*4,3,
            QwenStateRowSource::CheckpointHeader,&header,0,nullptr});
        for(std::size_t i=0;i<source.size();++i)EXPECT_EQ(restored[i],i<std::size_t(header)*rowWords ? source[i] : 0xaabbccddu);
    }
    int invalid=4;
    EXPECT_THROW(cpuQwenStateTransfer((QwenStateTransfer{source.data(),destination.data(),rowWords*4,3,
        QwenStateRowSource::CheckpointHeader,&invalid,0,nullptr})),std::invalid_argument);
    EXPECT_THROW(validateQwenStateTransfer((QwenStateTransfer{source.data(),source.data()+1,rowWords*4,2,
        QwenStateRowSource::Fixed,nullptr,1,nullptr})),std::invalid_argument);
    cpuQwenStateFill({restored.data(),restored.size(),0x12345678u});
    for(auto bits:restored)EXPECT_EQ(bits,0x12345678u);
}
TEST(QwenSparseReplay, AnalyticRotaryMatchesIndependentPyTorchFP32Phases)
{
    for(const auto &reference:test::RotaryReferences) {
        SCOPED_TRACE(reference.position);
        SCOPED_TRACE(reference.pair);
        SCOPED_TRACE(reference.theta);
        float c,s;
        detail::rotaryFactors(QwenSparseRotarySource::AnalyticTheta,nullptr,nullptr,
            reference.position,reference.pair,64,reference.theta,c,s);
        EXPECT_NEAR(c,reference.cosine,1.2e-7f);EXPECT_NEAR(s,reference.sine,1.2e-7f);
    }
}

}
