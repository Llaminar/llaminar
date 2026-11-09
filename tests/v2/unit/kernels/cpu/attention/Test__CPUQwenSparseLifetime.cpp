/**
 * @file Test__CPUQwenSparseLifetime.cpp
 * @brief Actual native CPU QSA admission, same-head alias and retained-team lifetime regressions.
 *
 * Every numeric receipt admits a genuine native program and calls the public
 * production entrypoint. Tests poison/authenticate complete results and guards,
 * retain original query/KV/selection roots, and observe delayed participants
 * immediately at function return, before the enclosing OpenMP join. No model,
 * accelerator, private helper substitute or scalar production replay is used.
 */
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "kernels/cpu/attention/CPUQwenSparseFP32Program.h"
#include "kernels/qwen4exp/QwenSparseTerminalCounters.h"
#include "kernels/common/SamplingMath.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <limits>
#include <thread>
#include <vector>
namespace llaminar2::qwen4exp::test
{
namespace
{
/** @brief Restore the actual caller's OpenMP budget and dynamic policy after every test. */
class TeamPolicy final
{
public:
    /** @brief Retain the original positive worker capacity, not a fabricated topology width. */
    TeamPolicy():threads_(omp_get_max_threads()),dynamic_(omp_get_dynamic()){omp_set_dynamic(0);}
    /** @brief Restore unchanged caller policy even after a failed assertion. */
    ~TeamPolicy(){omp_set_num_threads(threads_);omp_set_dynamic(dynamic_);}
    /** @return Actual original maximum team width for all-positive-worker coverage. */
    int threads()const noexcept{return threads_;}
private:int threads_,dynamic_;
};
/** @brief Own genuine original query/KV/selection/publication storage throughout each invocation. */
struct Fixture final
{
    static constexpr int prefix=2,guard=8;
    QwenSparseReplay replay;std::vector<float>keys,values,query,gate,output;
    std::vector<std::int32_t>selected,count;std::vector<QwenSparseAttentionObservation>observations;
    QwenSparseAttention p;
    /** @brief Prepare real one-head original banks and causal selected rows without running an oracle.
     * @param rows Original physical row capacity/live extent.
     */
    explicit Fixture(int rows):replay{prefix,rows,0},keys(std::size_t(prefix+rows)*256),values(keys.size()),
        query(std::size_t(rows)*256+guard+9,919.25f),gate(std::size_t(rows)*256),output(query.size(),-719.5f),
        selected(std::size_t(rows)*QwenSparseSelected,-987),count(rows),observations(rows)
    {
        for(std::size_t i=0;i<keys.size();++i){keys[i]=float(int(i%127)-63)*0.0078125f;values[i]=float(int(i%61)-30)*0.03125f;}
        for(int row=0;row<rows;++row)
        {
            count[row]=prefix+row+1;for(int i=0;i<count[row];++i)selected[std::size_t(row)*QwenSparseSelected+i]=i;
            for(int c=0;c<256;++c){query[guard+std::size_t(row)*256+c]=float((row*17+c)%101-50)*0.015625f;gate[std::size_t(row)*256+c]=float(c%17-8)*0.125f;}
        }
        const auto bank=[](const std::vector<float>&data){return QwenSparseNativeBank{.data=data.data(),.token_stride_bytes=256*sizeof(float),.head_stride_bytes=256*sizeof(float),.codec=TensorType::FP32};};
        p={.replay=&replay,.query=query.data()+guard,.kv={.keys=bank(keys),.values=bank(values)},.selected=selected.data(),
            .selected_count=count.data(),.output=output.data()+guard,.observations=observations.data(),.row_capacity=rows,.token_capacity=prefix+rows,.query_heads=1,.kv_heads=1};
    }
};
/** @brief Own only the original consumed rows/planes of a one-row actual indexer invocation. */
struct IndexerFixture final
{
    QwenSparseReplay replay;std::vector<float>query,raw,partial,pools,cosine,sine,prepared,scores;
    std::array<float,128>queryNorm,keyNorm;std::array<std::int32_t,QwenSparseSelected>selected{};std::int32_t count=0;
    QwenSparseIndexer p;
    /** @brief Build one real causal boundary without reading future input rows.
     * @param prefix Original prefix3 (short) or2051 (actual ranking boundary).
     */
    explicit IndexerFixture(int prefix):replay{prefix,1,0},query(512,1),raw(128,1),partial(512,1),
        pools(std::size_t((prefix+4)/4)*128,1),cosine(std::size_t(prefix+1)*32,1),sine(cosine.size(),0),prepared(512),scores((prefix+4)/4)
    {
        queryNorm.fill(1);keyNorm.fill(1);
        p={.replay=&replay,.query=query.data(),.raw_keys=raw.data(),.partial_keys=partial.data(),.pooled_keys=pools.data(),
            .query_norm=queryNorm.data(),.key_norm=keyNorm.data(),.cosine=cosine.data(),.sine=sine.data(),
            .prepared_query=prepared.data(),.scores=scores.data(),.selected=selected.data(),.selected_count=&count,
            .row_capacity=1,.token_capacity=prefix+1,.block_capacity=(prefix+4)/4,.position_capacity=prefix+1};
    }
};
/** @brief Admit genuine native arithmetic before any independently filtered receipt. */
void requireNative(){::llaminar2::cpu::qsa_native::requireNativeProgram(activeISALevel());}
/** @brief Require every original result word to be finite and byte-identical.
 * @param actual Complete actual guarded storage; @param expected Complete original reference storage.
 */
void authenticate(const std::vector<float>&actual,const std::vector<float>&expected)
{
    ASSERT_EQ(actual.size(),expected.size());for(std::size_t i=0;i<actual.size();++i)
    {ASSERT_TRUE(std::isfinite(actual[i]));EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),std::bit_cast<std::uint32_t>(expected[i]));}
}
/** @brief Delay one participant and observe completion immediately, not after the outer join.
 * @tparam Function Actual collective kernel call; @param call Original invocation.
 * @return Participants already entering the call when owner0 immediately returned.
 */
template<class Function>int immediatelyRetired(Function&&call)
{
    std::atomic<int>entered{0};int observed=-1,actual=0;
#pragma omp parallel num_threads(3)
    {
        const auto id=omp_get_thread_num();
#pragma omp single
        actual=omp_get_num_threads();
        if(id==2)std::this_thread::sleep_for(std::chrono::milliseconds(30));
        entered.fetch_add(1,std::memory_order_release);call();
        if(id==0)observed=entered.load(std::memory_order_acquire);
    }
    EXPECT_EQ(actual,3);return observed;
}
/** @brief Require the public root diagnostic, distinguishing a before-load rejection from late UB.
 * @tparam Function Actual public primitive call with one malformed replay root.
 * @param call Original callable; the subprocess must reject before any pointer load.
 */
template<class Function>void rejectsReplayBeforeLoad(Function&&call)
{
    EXPECT_EXIT({
        try{call();}
        catch(const std::invalid_argument&error)
        {
            if(std::string(error.what()).find("missing, wrapping or misaligned original storage")!=std::string::npos)std::_Exit(0);
            std::_Exit(18);
        }
        catch(...){std::_Exit(19);}
        std::_Exit(20);
    },::testing::ExitedWithCode(0),"");
}
} // namespace

/** @test Native public admission excludes partial query, original banks, gates and diagnostic writers before stores. */
TEST(CPUQwenSparseLifetime,CompleteOriginalExclusionsRejectBeforePublication)
{
    requireNative();TeamPolicy policy;omp_set_num_threads(1);Fixture f(3);
    const auto keys=f.keys,values=f.values,query=f.query,gate=f.gate,output=f.output;const auto selected=f.selected,count=f.count;
    const auto rejects=[&](QwenSparseAttention p){EXPECT_THROW(cpuQwenSparseAttention(p),std::invalid_argument);};
    auto p=f.p;p.output=f.keys.data();rejects(p);
    p=f.p;p.output=f.query.data()+Fixture::guard+1;rejects(p);
    p=f.p;p.output=reinterpret_cast<float*>(f.selected.data());rejects(p);
    p=f.p;p.gate=f.gate.data();p.output=f.gate.data();rejects(p);
    p=f.p;p.gate=p.output;rejects(p);
    p=f.p;p.observations=reinterpret_cast<QwenSparseAttentionObservation*>(p.output);rejects(p);
    p=f.p;p.observations=reinterpret_cast<QwenSparseAttentionObservation*>(f.values.data());rejects(p);
    EXPECT_EQ(f.keys,keys);EXPECT_EQ(f.values,values);EXPECT_EQ(f.query,query);EXPECT_EQ(f.gate,gate);EXPECT_EQ(f.output,output);
    EXPECT_EQ(f.selected,selected);EXPECT_EQ(f.count,count);
}
/** @test Actual public admission rejects replay alignment/wrapping before its first word is loaded. */
TEST(CPUQwenSparseLifetime,OriginalReplayIntervalRejectsBeforeAnyWordLoad)
{
    requireNative();TeamPolicy policy;omp_set_num_threads(1);GTEST_FLAG_SET(death_test_style,"threadsafe");Fixture f(1);
    const std::array<const QwenSparseReplay*,2>invalid={
        reinterpret_cast<const QwenSparseReplay*>(std::numeric_limits<std::uintptr_t>::max()-3),
        reinterpret_cast<const QwenSparseReplay*>(reinterpret_cast<const unsigned char*>(&f.replay)+1)};
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(invalid.front())%alignof(QwenSparseReplay),0u);
    IndexerFixture indexer(3);QwenSparseTransactionPrior snapshot{};QwenSparsePriorTraffic traffic{};
    std::array<float,512>destination{};const int accepted=0;
    for(const auto*root:invalid)
    {
        auto p=f.p;p.replay=root;
        // Separate subprocesses turn an old before-validation load/UB into a
        // focused failed exit assertion, never a crashing conformance runner.
        rejectsReplayBeforeLoad([&](){cpuQwenSparseAttention(p);});
        auto index=indexer.p;index.replay=root;
        rejectsReplayBeforeLoad([&](){cpuQwenSparseIndexer(index);});
        rejectsReplayBeforeLoad([&](){cpuQwenSparseSelectPools(index);});
        const QwenSparsePriorSnapshot prior{.replay=root,.accepted_partial=indexer.partial.data(),.output=&snapshot,
            .traffic=&traffic,.row_capacity=1,.token_capacity=4};
        rejectsReplayBeforeLoad([&](){cpuQwenSparseSnapshotPrior(prior);});
        const QwenSparsePartialCommit partial{.replay=root,.prior_partial=indexer.partial.data(),.raw_keys=indexer.raw.data(),
            .acceptance=AcceptedRequestPrefixes::counted(DeviceRequestRowRanges::deviceCounted(1,1,&f.replay.live_rows),&accepted),.partial_keys=destination.data()};
        rejectsReplayBeforeLoad([&](){cpuQwenSparseCommitPartial(partial);});
    }
}
/** @test A complete same-head query/output alias preserves original bytes, gates and inactive rows. */
TEST(CPUQwenSparseLifetime,FullSameHeadQueryAliasMatchesOriginalNativeProgram)
{
    requireNative();TeamPolicy policy;omp_set_num_threads(1);Fixture f(3);f.replay.live_rows=2;f.p.observations=nullptr;
    for(bool gated:{false,true})
    {
        f.p.gate=gated?f.gate.data():nullptr;cpuQwenSparseAttention(f.p);
        auto actual=f.query,expected=f.query;const auto keys=f.keys,values=f.values;
        std::copy_n(f.output.data()+Fixture::guard,2*256,expected.data()+Fixture::guard);
        auto p=f.p;p.query=actual.data()+Fixture::guard;p.output=actual.data()+Fixture::guard;
        cpuQwenSparseAttention(p);authenticate(actual,expected);EXPECT_EQ(f.keys,keys);EXPECT_EQ(f.values,values);
    }
}
/** @test Attention, indexer, selector and accepted-tail calls retire a delayed retained participant. */
TEST(CPUQwenSparseLifetime,EmptyAndNonemptyCallsCompleteBeforeOuterJoin)
{
    requireNative();TeamPolicy policy;Fixture f(1);f.p.observations=nullptr;
    for(int live:{1,0})
    {
        f.replay.live_rows=live;EXPECT_EQ(immediatelyRetired([&](){cpuQwenSparseAttention(f.p);}),3);
    }
    for(auto representation:{QwenSparseScalar::FP32,QwenSparseScalar::FP16,QwenSparseScalar::BF16})
        for(int prefix:{3,2051})
        {
            IndexerFixture indexer(prefix);indexer.p.representation=representation;
            for(int live:{1,0})
            {
                indexer.replay.live_rows=live;
                // The existing delayed-entry witness observes participant arrival
                // before the outer join, including empty native work frontiers.
                EXPECT_EQ(immediatelyRetired([&](){cpuQwenSparseIndexer(indexer.p);}),3);
                EXPECT_EQ(immediatelyRetired([&](){cpuQwenSparseSelectPools(indexer.p);}),3);
            }
        }
    std::array<float,4*128>prior{},destination{};std::array<float,128>raw{};const int accepted=0;
    for(int live:{1,0})
    {
        f.replay.live_rows=live;const QwenSparsePartialCommit p{.replay=&f.replay,.prior_partial=prior.data(),.raw_keys=raw.data(),
            .acceptance=AcceptedRequestPrefixes::counted(DeviceRequestRowRanges::deviceCounted(1,1,&f.replay.live_rows),&accepted),.partial_keys=destination.data()};
        EXPECT_EQ(immediatelyRetired([&](){cpuQwenSparseCommitPartial(p);}),3);
    }
}
/** @test Canonical terminal counter corruption is fatal at public CPU publication, never a collective exception. */
TEST(CPUQwenSparseLifetime,CanonicalCounterOverflowIsFatalAtPublicPublication)
{
    requireNative();TeamPolicy policy;omp_set_num_threads(1);GTEST_FLAG_SET(death_test_style,"threadsafe");
    using namespace sampling_math;
    alignas(8)std::array<std::int32_t,kDeviceGenerationControlCount>control{};
    const QwenSparseWorkCounterRow counters{control.data(),kDeviceGenerationControlCount};
    ASSERT_TRUE(initialize_device_generation_control(4,4,DeviceGenerationPolicy::fixed(1),control.data()));
    ASSERT_TRUE(addNativeCounter64(counters.metric(QwenSparseWorkCounter::PriorHeader),UINT64_MAX));
    QwenSparseReplay replay{2,1,0};std::array<float,512>prior{},destination{};std::array<float,128>raw{};
    QwenSparseTransactionPrior snapshot{};QwenSparsePriorTraffic traffic{};
    EXPECT_DEATH(cpuQwenSparseSnapshotPrior({.replay=&replay,.accepted_partial=prior.data(),.output=&snapshot,
        .traffic=&traffic,.row_capacity=1,.token_capacity=8,.work_counters=counters}),"canonical counter overflow in original-tail snapshot");
    ASSERT_TRUE(initialize_device_generation_control(4,4,DeviceGenerationPolicy::fixed(1),control.data()));
    ASSERT_TRUE(addNativeCounter64(counters.metric(QwenSparseWorkCounter::AcceptedPayload),UINT64_MAX));
    const int accepted=1;
    EXPECT_DEATH(cpuQwenSparseCommitPartial({.replay=&replay,.prior_partial=prior.data(),.raw_keys=raw.data(),
        .acceptance=AcceptedRequestPrefixes::counted(DeviceRequestRowRanges::deviceCounted(1,1,&replay.live_rows),&accepted),
        .partial_keys=destination.data(),.work_counters=counters}),"canonical counter overflow in accepted-tail commit");
}
/** @test Only consumed indexer values are checked, and nonfinite dots cannot be repaired by ReLU or selection. */
TEST(CPUQwenSparseLifetime,ConsumedIndexerNumericCorruptionIsFatalNotClamped)
{
    requireNative();TeamPolicy policy;omp_set_num_threads(1);GTEST_FLAG_SET(death_test_style,"threadsafe");
    IndexerFixture keys(3);keys.raw[0]=std::numeric_limits<float>::quiet_NaN();
    EXPECT_DEATH(cpuQwenSparseIndexer(keys.p),"invalid consumed numeric data in indexer RMS summary");
    IndexerFixture query(3);query.query[1]=std::numeric_limits<float>::quiet_NaN();
    EXPECT_DEATH(cpuQwenSparseIndexer(query.p),"invalid consumed numeric data in indexer RMS summary");
    IndexerFixture gamma(3);gamma.queryNorm[1]=std::numeric_limits<float>::infinity();
    EXPECT_DEATH(cpuQwenSparseIndexer(gamma.p),"invalid consumed numeric data in indexer normalized representation");
    IndexerFixture score(2051);
    // Finite opposite-sign native chains overflow to +/-Inf, then fold to NaN.
    // ReLU's max(0,NaN) would hide this unless the dot is checked first.
    for(unsigned i=0;i<score.keyNorm.size();++i)score.keyNorm[i]=(i%2?-1.0f:1.0f)*std::numeric_limits<float>::max();
    EXPECT_DEATH(cpuQwenSparseIndexer(score.p),"invalid consumed numeric data in indexer native score");
}
/** @test Actual grouped outputs retain serial request-row bytes for every positive original worker budget. */
TEST(CPUQwenSparseLifetime,AllPositiveWorkersRetainOriginalSerialRowsAndGuards)
{
    requireNative();TeamPolicy policy;
    for(int rows:{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31,33})
    {
        Fixture f(rows);omp_set_num_threads(1);auto expected=f.output;
        for(int row=0;row<rows;++row)
        {
            QwenSparseReplay replay{Fixture::prefix+row,1,0};auto one=f.p;
            one.replay=&replay;one.query=f.query.data()+Fixture::guard+std::size_t(row)*256;
            one.selected=f.selected.data()+std::size_t(row)*QwenSparseSelected;one.selected_count=f.count.data()+row;
            one.output=expected.data()+Fixture::guard+std::size_t(row)*256;one.row_capacity=1;one.observations=nullptr;
            cpuQwenSparseAttention(one);
        }
        const auto keys=f.keys,values=f.values,query=f.query;const auto selected=f.selected,count=f.count;
        for(int workers=1;workers<=policy.threads();++workers)
        {
            omp_set_num_threads(workers);auto actual=expected;
            for(std::size_t i=Fixture::guard;i<std::size_t(Fixture::guard)+std::size_t(rows)*256;++i)
                actual[i]=std::bit_cast<float>(std::bit_cast<std::uint32_t>(expected[i])^0x80000000u);
            auto p=f.p;p.output=actual.data()+Fixture::guard;p.observations=nullptr;cpuQwenSparseAttention(p);authenticate(actual,expected);
        }
        EXPECT_EQ(f.keys,keys);EXPECT_EQ(f.values,values);EXPECT_EQ(f.query,query);EXPECT_EQ(f.selected,selected);EXPECT_EQ(f.count,count);
    }
}
} // namespace llaminar2::qwen4exp::test
