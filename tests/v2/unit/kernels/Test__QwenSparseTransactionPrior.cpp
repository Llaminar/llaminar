/**
 * @file Test__QwenSparseTransactionPrior.cpp
 * @brief Immutable original-tail publication and exact prior traffic regressions.
 */
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
using namespace llaminar2;
using namespace llaminar2::qwen4exp;
/** @brief Every compression phase and empty/nonempty snapshot copies only live original rows. */
TEST(QwenSparseTransactionPrior, SnapshotTrafficAndUnusedBytesFollowOriginalLiveTail)
{
    for(int phase=0;phase<4;++phase)for(int live:{0,1,5}) {
        QwenSparseReplay replay{16+phase,live,123};
        std::array<float,4*128> accepted;for(int i=0;i<512;++i)accepted[i]=float(i+1);
        QwenSparseTransactionPrior prior;std::memset(&prior,0x6b,sizeof(prior));
        const auto before=prior;QwenSparsePriorTraffic traffic{};
        cpuQwenSparseSnapshotPrior({&replay,accepted.data(),&prior,&traffic,5,128});
        EXPECT_EQ(prior.replay.prefix_tokens,replay.prefix_tokens);EXPECT_EQ(prior.replay.live_rows,live);
        EXPECT_EQ(prior.replay.position_origin,123);EXPECT_EQ(prior.row_capacity,5);
        const auto bytes=live ? phase*128*sizeof(float) :0;
        EXPECT_EQ(traffic.metadata_bytes,16);EXPECT_EQ(traffic.payload_bytes,bytes);
        EXPECT_EQ(std::memcmp(prior.partial,accepted.data(),bytes),0);
        EXPECT_EQ(std::memcmp(reinterpret_cast<const char*>(prior.partial)+bytes,
            reinterpret_cast<const char*>(before.partial)+bytes,sizeof(prior.partial)-bytes),0);
    }
}
/** @brief A later smaller/zero decision after completing a pool recovers the original tail. */
TEST(QwenSparseTransactionPrior, RepeatedAcceptanceAcrossPoolCompletionUsesImmutableOriginalTail)
{
    constexpr int capacity=5,context=128;
    for(int phase=0;phase<4;++phase) {
        QwenSparseReplay replay{4+phase,capacity,79};
        std::array<float,512> accepted;std::array<float,capacity*128> candidates;
        for(int i=0;i<512;++i)accepted[i]=1000.0f+i;
        const auto original=accepted;
        for(int i=0;i<capacity*128;++i)candidates[i]=10000.0f+i;
        QwenSparseTransactionPrior prior{};QwenSparsePriorTraffic traffic{};
        cpuQwenSparseSnapshotPrior({&replay,accepted.data(),&prior,&traffic,capacity,context});
        const auto saved=prior;
        for(int decision:{4,0,2,1,5,0,3}) {
            auto acceptance=AcceptedRequestPrefixes::fixed(
                DeviceRequestRowRanges::deviceCounted(1,capacity,&prior.replay.live_rows),decision);
            cpuQwenSparseCommitPartial({&prior.replay,prior.partial,candidates.data(),acceptance,
                accepted.data(),&prior,context});
            const int total=replay.prefix_tokens+decision,begin=total/4*4;
            for(int row=0;row<total%4;++row) {
                const int token=begin+row;
                const auto* expected=token<replay.prefix_tokens ? original.data()+(token-replay.prefix_tokens/4*4)*128 :
                    candidates.data()+(token-replay.prefix_tokens)*128;
                EXPECT_EQ(std::memcmp(accepted.data()+row*128,expected,128*sizeof(float)),0)
                    << "phase="<<phase<<" accepted="<<decision;
            }
            EXPECT_EQ(std::memcmp(&prior,&saved,sizeof(prior)),0);
        }
    }
}
/** @brief Empty transaction commits preserve all accepted bytes; malformed later geometry mutates nothing. */
TEST(QwenSparseTransactionPrior, EmptyAndMalformedTransactionsPreserveAcceptedState)
{
    QwenSparseReplay replay{7,0,91};std::array<float,512> accepted;accepted.fill(-91.5f);
    std::array<float,5*128> raw;raw.fill(std::numeric_limits<float>::quiet_NaN());
    QwenSparseTransactionPrior prior{};QwenSparsePriorTraffic traffic{};
    cpuQwenSparseSnapshotPrior({&replay,accepted.data(),&prior,&traffic,5,128});
    const auto before=accepted;
    auto acceptance=AcceptedRequestPrefixes::fixed(DeviceRequestRowRanges::deviceCounted(1,5,&prior.replay.live_rows),0);
    cpuQwenSparseCommitPartial({&prior.replay,prior.partial,raw.data(),acceptance,accepted.data(),&prior,128});
    EXPECT_EQ(std::memcmp(accepted.data(),before.data(),sizeof(accepted)),0);
    prior.row_capacity=6;
    EXPECT_THROW(cpuQwenSparseCommitPartial((QwenSparsePartialCommit{&prior.replay,prior.partial,raw.data(),acceptance,accepted.data(),&prior,128})),std::invalid_argument);
    EXPECT_EQ(std::memcmp(accepted.data(),before.data(),sizeof(accepted)),0);
    auto malformed=replay;malformed.live_rows=6;const auto frozen=prior;
    EXPECT_THROW(cpuQwenSparseSnapshotPrior((QwenSparsePriorSnapshot{&malformed,accepted.data(),&prior,&traffic,5,128})),std::invalid_argument);
    EXPECT_EQ(std::memcmp(&prior,&frozen,sizeof(prior)),0);
}
