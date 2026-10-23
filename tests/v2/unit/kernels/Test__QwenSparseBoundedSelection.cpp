/**
 * @file Test__QwenSparseBoundedSelection.cpp
 * @brief Device-free exact candidate tests using actual CPU entrypoints.
 *
 * Independent numeric sort covers ties/radix boundaries, causal tails and dirty
 * replay. The scalar bounded helper is the actual CPU implementation, never a
 * test replacement for native selection. Worker changes are restored by RAII.
 */
#include "kernels/qwen4exp/QwenSparseBoundedSelection.h"
#include "../../integration/kernels/qwen4exp/QwenSparseSelectionCases.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <cstring>
namespace llaminar2::qwen4exp
{
namespace
{
struct Workers final
{
    int prior = omp_get_max_threads();
    ~Workers() { omp_set_num_threads(prior); }
};
struct Selection final
{
    int rows, capacity, blocks;
    QwenSparseReplay replay;
    std::vector<float> scores;
    std::vector<std::int32_t> selected, counts;
    std::vector<QwenSparseSelectionObservation> observations;
    QwenSparseIndexer p;
    Selection(int m, int tokens, int prefix, selection_test::Distribution distribution)
        : rows(m), capacity(tokens), blocks((tokens + 3) / 4), replay{prefix, m, 0},
          scores(selection_test::scores(m, blocks, distribution)),
          selected(std::size_t(m) * QwenSparseSelected, -77), counts(m, -77), observations(m)
    {
        p.replay = &replay; p.scores = scores.data(); p.selected = selected.data();
        p.selected_count = counts.data(); p.row_capacity = m; p.token_capacity = tokens; p.block_capacity = blocks;
    }
    void check()
    {
        for (int row = 0; row < rows; ++row)
        {
            const auto expected = row < replay.live_rows ? selection_test::oracle(
                scores.data() + std::size_t(row) * blocks, replay.prefix_tokens + row + 1) : std::vector<std::int32_t>{};
            ASSERT_EQ(counts[row], expected.size());
            for (std::size_t i = 0; i < expected.size(); ++i)
                ASSERT_EQ(selected[std::size_t(row) * QwenSparseSelected + i], expected[i]);
            for (std::size_t i = expected.size(); i < QwenSparseSelected; ++i)
                ASSERT_EQ(selected[std::size_t(row) * QwenSparseSelected + i], -77);
        }
    }
};
/** @test Persistent high keys force exact aligned work and honest carry-tail digit counts. */
TEST(QwenSparseBoundedSelection, PersistentFourDigitFrontierWork)
{
    Workers workers;omp_set_num_threads(1);
    for(auto policy:selection_test::policies)
        for(int pools:{767,768,769,1535,1536,1537,16383,16384})
        {
            SCOPED_TRACE(int(policy));
            SCOPED_TRACE(pools);
            Selection s(1,pools*4+3,pools*4+2,selection_test::Distribution::PersistentFourDigitFrontier);
            s.p.pool_selection=policy;s.p.selection_observations=s.observations.data();
            cpuQwenSparseSelectPools(s.p);s.check();
            const auto expected=selection_test::persistentFrontierWork(pools,policy);
            const auto observed=s.observations[0];
            EXPECT_EQ(observed.score_elements,expected.score_elements);
            EXPECT_EQ(observed.compactions,expected.compactions);
            EXPECT_EQ(observed.compacted_keys,expected.compacted_keys);
            EXPECT_EQ(observed.radix_passes,expected.radix_passes);
            if(pools%256==0)EXPECT_EQ(observed.radix_passes,4*observed.compactions);
            if(pools%256==255)EXPECT_EQ(observed.radix_passes,4*observed.compactions-1);
        }
}

/** @test A single high key stresses common-prefix population with exact aligned work. */
TEST(QwenSparseBoundedSelection, HotHistogramFrontierWork)
{
    Workers workers;omp_set_num_threads(1);
    for(auto policy:selection_test::policies)
        for(int pools:{767,768,769,1535,1536,1537,16383,16384})
        {
            SCOPED_TRACE(int(policy));
            SCOPED_TRACE(pools);
            Selection s(1,pools*4+3,pools*4+2,selection_test::Distribution::HotHistogramFrontier);
            s.p.pool_selection=policy;s.p.selection_observations=s.observations.data();
            cpuQwenSparseSelectPools(s.p);s.check();
            // All arrivals are retained until compaction in both authored
            // frontiers, so N/C/W identities apply. Do not reuse ID7 tail digits.
            const auto expected=selection_test::persistentFrontierWork(pools,policy);
            const auto observed=s.observations[0];
            EXPECT_EQ(observed.score_elements,expected.score_elements);
            EXPECT_EQ(observed.compactions,expected.compactions);
            EXPECT_EQ(observed.compacted_keys,expected.compacted_keys);
            EXPECT_LE(observed.radix_passes,4*observed.compactions);
            if(pools%256==0)EXPECT_EQ(observed.radix_passes,4*observed.compactions);
        }
}

TEST(QwenSparseBoundedSelection, ExactTiesCommonPrefixesThresholdAndEveryTail)
{
    Workers workers; omp_set_num_threads(1);
    for (auto distribution : selection_test::distributions)
        for (int blocks : {511, 512, 513, 1025, 4097})
            for (int tail = 0; tail < 4; ++tail)
                for (auto policy : selection_test::policies)
                {
                    SCOPED_TRACE(int(distribution));
                    SCOPED_TRACE(blocks);
                    SCOPED_TRACE(tail);
                    SCOPED_TRACE(int(policy));
                    Selection s(1, blocks * 4 + tail, blocks * 4 + tail - 1, distribution);
                    const auto original = s.scores;
                    s.p.pool_selection = policy; s.p.selection_observations = s.observations.data();
                    cpuQwenSparseSelectPools(s.p); s.check();
                    EXPECT_EQ(std::memcmp(s.scores.data(), original.data(), original.size() * sizeof(float)), 0);
                    EXPECT_EQ(s.observations[0].score_elements, blocks > 512 ? std::uint64_t(blocks) : 0u);
                    if (blocks > 512) { EXPECT_GT(s.observations[0].compactions, 0u); }
                    else { EXPECT_EQ(s.observations[0].compactions, 0u); }
                }
}
TEST(QwenSparseBoundedSelection, GroupedSerialAndRetainedLargeSmallEmpty)
{
    Workers workers;
    for (int threads : {1, 3})
    {
        omp_set_num_threads(threads);
        for (auto policy : selection_test::policies)
        {
            Selection s(33, 8199, 8199 - 33, selection_test::Distribution::Hash);
            s.p.pool_selection = policy;
            for (int repeat = 0; repeat < 20; ++repeat)
            {
                s.replay.live_rows = repeat % 4 == 0 ? 0 : repeat % 4 == 1 ? 1 : repeat % 4 == 2 ? 9 : 33;
                s.replay.prefix_tokens = repeat % 3 == 0 ? 2048 : 8199 - 33;
                std::fill(s.selected.begin(), s.selected.end(), -77);
                cpuQwenSparseSelectPools(s.p); s.check();
                for (int row = 0; row < s.replay.live_rows; ++row)
                {
                    auto p = s.p; QwenSparseReplay r{s.replay.prefix_tokens + row, 1, 0};
                    std::array<std::int32_t, QwenSparseSelected> result; result.fill(-77);
                    int count = -1; p.replay = &r; p.row_capacity = 1;
                    p.scores += std::size_t(row) * s.blocks; p.selected = result.data(); p.selected_count = &count;
                    cpuQwenSparseSelectPools(p);
                    ASSERT_EQ(count, s.counts[row]);
                    EXPECT_EQ(std::memcmp(result.data(), s.selected.data() + std::size_t(row) * QwenSparseSelected,
                        sizeof(result)), 0);
                }
            }
        }
    }
}
TEST(QwenSparseBoundedSelection, StaticAndFiniteRejectionBeforeWritesUnusedScoresUntouched)
{
    Workers workers; omp_set_num_threads(1);
    Selection s(2, 4099, 4097, selection_test::Distribution::Equal);
    s.p.pool_selection = QwenSparsePoolSelection::Bounded256_512;
    auto p = s.p; p.pool_selection = static_cast<QwenSparsePoolSelection>(255);
    EXPECT_THROW(cpuQwenSparseSelectPools(p), std::invalid_argument);
    p = s.p; p.selected_count = p.selected; EXPECT_THROW(cpuQwenSparseSelectPools(p), std::invalid_argument);
    p = s.p; p.block_capacity = 1; EXPECT_THROW(cpuQwenSparseSelectPools(p), std::invalid_argument);
    for (float invalid : {-1.0f, INFINITY, NAN})
    {
        s.scores[0] = invalid;
        EXPECT_THROW(cpuQwenSparseSelectPools(s.p), std::invalid_argument);
        EXPECT_TRUE(std::all_of(s.selected.begin(), s.selected.end(), [](auto v) { return v == -77; }));
        EXPECT_EQ(s.counts[0], -77); EXPECT_EQ(s.counts[1], -77);
    }
    s.scores[0] = 0.0f;
    const int laterLivePools = (s.replay.prefix_tokens + 2) / QwenSparsePool;
    ASSERT_EQ(laterLivePools, 1024);
    const auto lastLive = std::size_t(s.blocks) + laterLivePools - 1;
    const auto firstUnused = std::size_t(s.blocks) + laterLivePools;
    s.scores[lastLive] = NAN; // The last consumed score of row1 must fail before row0 writes.
    EXPECT_THROW(cpuQwenSparseSelectPools(s.p), std::invalid_argument);
    EXPECT_TRUE(std::all_of(s.selected.begin(), s.selected.end(), [](auto v) { return v == -77; }));
    EXPECT_EQ(s.counts[0], -77); EXPECT_EQ(s.counts[1], -77);
    s.scores[lastLive] = 0.0f;
    ASSERT_LT(firstUnused, s.scores.size());
    s.scores[firstUnused] = NAN; // Allocated suffix is neither validated nor selected.
    EXPECT_NO_THROW(cpuQwenSparseSelectPools(s.p));
    s.check();
    EXPECT_TRUE(std::isnan(s.scores[firstUnused]));
    p = s.p; p.scores = reinterpret_cast<float *>(UINTPTR_MAX - 3u);
    EXPECT_THROW(validateQwenSparseSelection(p), std::invalid_argument);
    s.replay = {2048, 1, 0}; // 512 complete pools plus one tail: scores are NOT consumed.
    EXPECT_NO_THROW(cpuQwenSparseSelectPools(s.p));
    EXPECT_EQ(s.counts[0], 2049); EXPECT_EQ(s.counts[1], 0);
    s.replay.live_rows = 0; EXPECT_NO_THROW(cpuQwenSparseSelectPools(s.p));
}
/** @test Actual CPU selectors consume signed zeros and one-ULP cutoff ties through true64K. */
TEST(QwenSparseBoundedSelection, AdversarialScoresAt64K)
{
    Workers workers;omp_set_num_threads(1);
    for(auto policy:selection_test::policies)for(int rows:{1,16})
        for(auto distribution:{selection_test::Adversarial::SignedZero,selection_test::Adversarial::CutoffOneULP}){
            Selection s(rows,65536,65536-rows,selection_test::Distribution::Equal);
            s.scores=selection_test::adversarialScores(rows,s.blocks,distribution);s.p.scores=s.scores.data();s.p.pool_selection=policy;
            const auto original=s.scores;cpuQwenSparseSelectPools(s.p);s.check();
            EXPECT_EQ(std::memcmp(s.scores.data(),original.data(),original.size()*sizeof(float)),0);
        }
}
/** @test Every admitted grouped M1..16 equals original-row serial selection. */
TEST(QwenSparseBoundedSelection, EveryGroupedMThrough16KeepsSerialBytes)
{
    Workers workers;omp_set_num_threads(3);
    for(int rows=1;rows<=16;++rows)for(auto policy:selection_test::policies){
        Selection s(rows,4096+rows,4096,selection_test::Distribution::Hash);s.p.pool_selection=policy;
        cpuQwenSparseSelectPools(s.p);s.check();
        for(int row=0;row<rows;++row){
            auto p=s.p;QwenSparseReplay replay{s.replay.prefix_tokens+row,1,0};
            std::array<std::int32_t,QwenSparseSelected> selected;selected.fill(-77);int count=-1;
            p.replay=&replay;p.row_capacity=1;p.scores+=std::size_t(row)*s.blocks;p.selected=selected.data();p.selected_count=&count;
            cpuQwenSparseSelectPools(p);ASSERT_EQ(count,s.counts[row]);
            EXPECT_EQ(std::memcmp(selected.data(),s.selected.data()+std::size_t(row)*QwenSparseSelected,sizeof(selected)),0);
        }
    }
}
TEST(QwenSparseBoundedSelection, GeometrySharedBytesAndUniqueTotalKey)
{
    using namespace bounded_selection;
    EXPECT_EQ(sizeof(Storage<256, 512>), 21744u);
    EXPECT_EQ(sizeof(Storage<512, 512>), 25840u);
    EXPECT_EQ(sizeof(Storage<1024, 1024>), 42224u);
    EXPECT_LT(sizeof(Storage<1024, 1024>), 48u * 1024u);
    EXPECT_LT(sizeof(Storage<256, 512>), sizeof(Storage<512, 512>));
    EXPECT_GT(key(0x3f800000u, 0), key(0x3f800000u, 1));
    EXPECT_EQ(key(0x80000000u, 0), key(0u, 0));
    EXPECT_FALSE(validScoreBits(0x7f800000u)); EXPECT_FALSE(validScoreBits(0xbf800000u));
    EXPECT_LT(firstShift(key(0x3f800000u, 0), key(0x3f800000u, 1024)), 32);
}

/**
 * @brief Check one real scalar compaction at the largest valid signed-token pool IDs.
 * @tparam Block Existing scan geometry, retaining its actual lexical bank size.
 * @tparam Slack Existing compaction trigger slack; storage layout is unchanged.
 * @details Each case makes one compaction, so observed digit bounds apply to
 * that actual call rather than an average over several calls. Numeric FP32/ID
 * sorting is independent of the production radix/tie predicate.
 */
template<int Block,int Slack> void scoreOnlyCompactionCases()
{
    constexpr int size=640,maxId=std::numeric_limits<std::int32_t>::max()/4-1;
    constexpr std::array<std::uint32_t,8> highBits{0x3f800000u,0u,0x3f800001u,0x3f808000u,
        0x3f900000u,0x40800000u,0x7f7fffffu,0x00808080u};
    for(int pattern=0;pattern<=int(highBits.size())+2;++pattern) {
        SCOPED_TRACE(pattern);
        bounded_selection::Storage<Block,Slack> storage;
        bounded_selection::begin(storage);storage.size=size;
        std::vector<float> values(size);
        std::vector<int> expected(size);std::iota(expected.begin(),expected.end(),0);
        for(int i=0;i<size;++i) {
            // Pattern 0 is all equal. Other patterns have a 128-key strict
            // upper region and a long cutoff tie, crossing several warp tiles.
            std::uint32_t bits=pattern==10 ? (i<255 ? 0x40000000u : 0x3f800000u) :
                pattern==9 ? (i<128 ? 0x40000000u : i<512 ? 0x3f001000u+std::uint32_t(i-128)*0x1000u : 0x3e000000u) :
                pattern==8 ? (i<511 ? 0x7f7fffffu : i==511 ? 0x01010101u : 0x01010100u) :
                pattern==0 ? highBits[0] : i%5==0 ? highBits[pattern] : 0u;
            if(pattern==1)bits=i%2 ? 0x80000000u : 0u;
            std::memcpy(&values[i],&bits,sizeof(bits));
            storage.keys[0][i]=bounded_selection::key(bits,std::uint32_t(maxId-size+1+i));
        }
        std::sort(expected.begin(),expected.end(),[&](int a,int b){
            return values[a]>values[b] || (values[a]==values[b] && a<b);
        });
        expected.resize(512);std::sort(expected.begin(),expected.end());
        const auto original=std::vector<std::uint64_t>(storage.keys[0],storage.keys[0]+size);
        ASSERT_TRUE(bounded_selection::compactScalar(storage));
        EXPECT_EQ(storage.size,512);EXPECT_EQ(storage.observation.compactions,1u);
        EXPECT_EQ(storage.observation.compacted_keys,std::uint64_t(size));
        EXPECT_LE(storage.observation.radix_passes,4u);
        if(pattern<=1)EXPECT_EQ(storage.observation.radix_passes,0u);
        if(pattern==8)EXPECT_EQ(storage.observation.radix_passes,4u);
        if(pattern==9)EXPECT_EQ(storage.observation.radix_passes,1u); // Coarse bucket population exactly fills residual rank.
        if(pattern==10)EXPECT_EQ(storage.observation.radix_passes,4u); // Cutoff tie spans 32/256-lane boundaries.
        std::uint64_t minimum=UINT64_MAX;
        for(int i=0;i<512;++i) {
            const auto actual=storage.keys[storage.active][i];
            EXPECT_EQ(actual,original[expected[i]]);minimum=std::min(minimum,actual);
        }
        EXPECT_EQ(storage.threshold,minimum); // Full 64-bit true minimum, never a coarse score cutoff.
        if(pattern==9)EXPECT_GT(storage.threshold&0xffffffff00000000ULL,0x3f00000000000000ULL);
        EXPECT_EQ(std::memcmp(original.data(),storage.keys[0],size*sizeof(std::uint64_t)),0);
    }
}
/** @test Equal-score banks skip every ID digit and preserve exact high-ID tie order/threshold. */
TEST(QwenSparseBoundedSelection, ScoreOnlyRadixCanonicalTiesAndFourDigitBound)
{
    scoreOnlyCompactionCases<256,512>();
    scoreOnlyCompactionCases<512,512>();
    scoreOnlyCompactionCases<1024,1024>();
    Workers workers;omp_set_num_threads(1);
    for(auto policy:selection_test::policies) {
        Selection s(3,16384,16381,selection_test::Distribution::Equal);
        s.p.pool_selection=policy;s.p.selection_observations=s.observations.data();
        cpuQwenSparseSelectPools(s.p);s.check();
        for(const auto& observation:s.observations) {
            EXPECT_GT(observation.compactions,0u);EXPECT_EQ(observation.radix_passes,0u);
            EXPECT_GE(observation.compacted_keys,512u*observation.compactions);
        }
    }
}

/** @test Bound admission and constant evaluation never touch native execution or invalid output. */
TEST(QwenSparseBoundedSelection, WorkBoundAdmissionAndArithmetic)
{
    using bounded_selection::WorkBound;using bounded_selection::workBound;
    constexpr auto proof=[] {WorkBound b;return workBound(QwenSparsePoolSelection::Bounded1024_1024,2049,b) &&
        b.compactions_max==2 && b.compacted_keys_max==2561 && b.score_radix_passes_max==8 &&
        b.score_radix_key_visits_max==10244 && b.compact_output_rounds_max==11 && b.body_barriers_max==106;};
    static_assert(proof());
    constexpr std::int64_t limit=std::numeric_limits<int>::max()/QwenSparsePool;
    for(auto policy:selection_test::policies) {
        WorkBound b;
        for(std::int64_t n:{std::int64_t{0},std::int64_t{1},std::int64_t{511},std::int64_t{512}}) {
            ASSERT_TRUE(workBound(policy,n,b));EXPECT_EQ(b,WorkBound{});
        }
        ASSERT_TRUE(workBound(policy,limit,b));
        EXPECT_EQ(b.score_elements,std::uint64_t(limit));EXPECT_EQ(b.score_radix_key_visits_max,4*b.compacted_keys_max);
        EXPECT_GT(b.body_barriers_max,b.compactions_max);
        EXPECT_LT(b.body_barriers_max,std::numeric_limits<std::uint64_t>::max()/16);
        const auto saved=b;
        for(auto n:{std::int64_t{-1},limit+1,std::numeric_limits<std::int64_t>::max(),std::numeric_limits<std::int64_t>::min()}) {
            EXPECT_FALSE(workBound(policy,n,b));EXPECT_EQ(b,saved);
        }
        ASSERT_TRUE(workBound(policy,2048,b));
        if(policy==QwenSparsePoolSelection::Bounded1024_1024) {
            EXPECT_EQ(b.compactions_max,1u);EXPECT_EQ(b.compacted_keys_max,2048u);
            EXPECT_EQ(b.body_barriers_max,74u);
        } else {EXPECT_EQ(b.compactions_max,3u);EXPECT_EQ(b.compacted_keys_max,3072u);}
    }
    WorkBound b;ASSERT_TRUE(workBound(QwenSparsePoolSelection::Bounded1024_1024,2049,b));const auto saved=b;
    for(auto policy:{QwenSparsePoolSelection::Installed,static_cast<QwenSparsePoolSelection>(255)}) {
        EXPECT_FALSE(workBound(policy,2049,b));EXPECT_EQ(b,saved);
    }
}

/**
 * @test Independent ascending arrival accounting proves tight count/volume over every small pool frontier.
 * @details This accounting follows block arrivals/removals, not radix keys or the
 * closed formula. It has no selector invocation, allocation or hardware timing.
 */
TEST(QwenSparseBoundedSelection, WorkBoundTightAscendingFrontiers)
{
    for(auto policy:selection_test::policies) {
        const int block=policy==QwenSparsePoolSelection::Bounded256_512?256:
            policy==QwenSparsePoolSelection::Bounded512_512?512:1024;
        const int slack=block==1024?1024:512;
        for(int n=513;n<=4097;++n) {
            int bank=0;std::uint64_t calls=0,volume=0,tiles=0;
            for(int start=0;start<n;start+=block) {
                bank+=std::min(block,n-start);
                if(bank>=512+slack) {++calls;volume+=bank;tiles+=(bank+255)/256;bank=512;}
            }
            if(bank>512) {++calls;volume+=bank;tiles+=(bank+255)/256;}
            bounded_selection::WorkBound b;ASSERT_TRUE(bounded_selection::workBound(policy,n,b));
            ASSERT_EQ(b.compactions_max,calls)<<int(policy)<<':'<<n;
            ASSERT_EQ(b.compacted_keys_max,volume)<<int(policy)<<':'<<n;
            ASSERT_EQ(b.compact_output_rounds_max,tiles)<<int(policy)<<':'<<n;
        }
    }
}


/**
 * @test Block-aligned tile bounds include filtered arrivals, skipped final calls and partial tails.
 * @details The small accounting model changes accepted arrival counts only; it
 * neither copies selector/radix arithmetic nor adds production observations.
 * All arrivals pass before the first compact. Later deterministic patterns
 * include all-rejected, all-accepted and mixed source blocks, with dirty pending
 * banks compacted only at the existing1536/1024 triggers or at end.
 */
TEST(QwenSparseBoundedSelection, WorkBoundAlignedCompactionTiles)
{
    for(auto policy:selection_test::policies) {
        const int block=policy==QwenSparsePoolSelection::Bounded256_512?256:
            policy==QwenSparsePoolSelection::Bounded512_512?512:1024;
        const int slack=block==1024?1024:512;
        for(int n:{513,767,768,769,1023,1024,1025,1535,1536,1537,2048,2049,3072,3073,16383,16384})
            for(int pattern=0;pattern<16;++pattern) {
                std::uint64_t bank=0,calls=0,tiles=0;
                for(int start=0,step=0;start<n;start+=block,++step) {
                    const int count=std::min(block,n-start);
                    const int accepted=calls==0 || pattern==0 ? count : pattern==1 ? 0 :
                        int((std::uint64_t(pattern)*131+std::uint64_t(step)*257)%(count+1));
                    bank+=accepted;
                    if(bank>=512+std::uint64_t(slack)) {++calls;tiles+=(bank+255)/256;bank=512;}
                }
                if(bank>512) {++calls;tiles+=(bank+255)/256;}
                ASSERT_GT(calls,0u);
                const auto sourceTiles=std::uint64_t(n/256+(n%256!=0));
                EXPECT_LE(tiles,sourceTiles+2*(calls-1));
                bounded_selection::WorkBound b;ASSERT_TRUE(bounded_selection::workBound(policy,n,b));
                EXPECT_LE(calls,b.compactions_max);EXPECT_LE(tiles,b.compact_output_rounds_max);
                if(pattern==0)EXPECT_EQ(tiles,b.compact_output_rounds_max);
            }
    }
    bounded_selection::WorkBound b;
    ASSERT_TRUE(bounded_selection::workBound(QwenSparsePoolSelection::Bounded1024_1024,3072,b));
    EXPECT_EQ(b.compact_output_rounds_max,14u);EXPECT_EQ(b.body_barriers_max,127u);
    ASSERT_TRUE(bounded_selection::workBound(QwenSparsePoolSelection::Bounded1024_1024,16384,b));
    EXPECT_EQ(b.compact_output_rounds_max,92u);EXPECT_EQ(b.body_barriers_max,816u);
    ASSERT_TRUE(bounded_selection::workBound(QwenSparsePoolSelection::Bounded1024_1024,2049,b));
    EXPECT_EQ(b.compact_output_rounds_max,11u);EXPECT_EQ(b.body_barriers_max,106u);
}

/** @test Actual CPU diagnostics respect geometry bounds, with numeric-sort output authority and dirty empty replay. */
TEST(QwenSparseBoundedSelection, WorkBoundMatchesCPUObservationBoundaries)
{
    Workers workers;omp_set_num_threads(1);
    for(auto policy:selection_test::policies)
        for(int n:{511,512,513,767,768,769,1023,1024,1025,1535,1536,1537,2047,2048,2049,3071,3072,3073,4097,16384})
            for(int tail=0;tail<4;++tail) {
                if(n==16384 && tail!=0)continue; // This witness ends exactly at64K tokens.
                for(auto distribution:{selection_test::Distribution::Ascending,selection_test::Distribution::Equal,selection_test::Distribution::Hash}) {
                    SCOPED_TRACE(int(policy));
                    SCOPED_TRACE(n);
                    SCOPED_TRACE(tail);
                    SCOPED_TRACE(int(distribution));
                    const int tokens=n*4+tail;
                    Selection s(1,tokens,tokens-1,distribution);s.p.pool_selection=policy;s.p.selection_observations=s.observations.data();
                    const auto original=s.scores;cpuQwenSparseSelectPools(s.p);s.check();ASSERT_FALSE(::testing::Test::HasFailure());
                    EXPECT_EQ(std::memcmp(original.data(),s.scores.data(),original.size()*sizeof(float)),0);
                    bounded_selection::WorkBound b;ASSERT_TRUE(bounded_selection::workBound(policy,n,b));
                    const auto& o=s.observations[0];
                    EXPECT_EQ(o.score_elements,b.score_elements);EXPECT_LE(o.compactions,b.compactions_max);
                    EXPECT_LE(o.compacted_keys,b.compacted_keys_max);EXPECT_LE(o.radix_passes,4*o.compactions);
                    if(distribution==selection_test::Distribution::Ascending) {
                        EXPECT_EQ(o.compactions,b.compactions_max);EXPECT_EQ(o.compacted_keys,b.compacted_keys_max);
                    }
                    if(distribution==selection_test::Distribution::Equal)EXPECT_EQ(o.radix_passes,0u);
                    const auto prior=s.selected;s.replay={0,0,0};s.observations[0]={91,92,93,94};
                    cpuQwenSparseSelectPools(s.p);EXPECT_EQ(s.counts[0],0);EXPECT_EQ(s.selected,prior);
                    EXPECT_EQ(s.observations[0].score_elements,0u);EXPECT_EQ(s.observations[0].compactions,0u);
                    EXPECT_EQ(s.observations[0].compacted_keys,0u);EXPECT_EQ(s.observations[0].radix_passes,0u);
                }
            }
}

} // namespace
} // namespace llaminar2::qwen4exp

/** @test Numeric zero ties canonicalize without changing other score bit patterns. */
TEST(QwenSparseBoundedSelection, CanonicalScoreBitsPreserveNumericZero)
{
    EXPECT_EQ(llaminar2::qwen4exp::bounded_selection::canonicalScoreBits(0x80000000u),0u);
    EXPECT_EQ(llaminar2::qwen4exp::bounded_selection::canonicalScoreBits(0xbf800000u),0xbf800000u);
    EXPECT_EQ(llaminar2::qwen4exp::bounded_selection::canonicalScoreBits(0x7f800000u),0x7f800000u);
    EXPECT_EQ(llaminar2::qwen4exp::bounded_selection::canonicalScoreBits(0x7fc00000u),0x7fc00000u);
}
