/**
 * @file Test__MTPSurvivalScoreInterval.cpp
 * @brief Device-free exact-extrema and evidence-classification regressions.
 *
 * Exhaustive small histograms independently supply min/max truncation sums.
 * A host-only128-bit oracle verifies classifications at uint input extremes.
 * These authored samples establish arithmetic, not model provenance, live
 * controller selection, probability confidence or end-to-end throughput.
 */
#include "execution/mtp/MTPSurvivalScoreInterval.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

namespace
{
    using namespace llaminar2;
    using Scope = MTPPrefixEvidenceScope;
    using Evidence = MTPPrefixWindowEvidence;
    using Error = MTPSurvivalScoreError;
    using Class = MTPSurvivalScoreClass;
    using Direction = MTPPrefixCandidateDirection;

    /** @return Explicit supported evidence; no caller silently gains a proof. */
    Evidence supported(uint32_t d, uint32_t r, uint64_t a, uint32_t z, uint32_t f)
    {
        return {Scope::NonStoppedUnclippedVerifiedState,d,r,a,z,f};
    }

    /** @brief Check fail-closed score output, with no residual valid interval. */
    void expectError(const Evidence &e, uint32_t k, uint32_t c, uint32_t ck, Error error)
    {
        const auto result = evaluateMTPPrefixSurvivalScore(e,k,c,ck);
        EXPECT_FALSE(result.valid());
        EXPECT_EQ(result.error,error);
        EXPECT_EQ(result.classification,Class::Invalid);
        EXPECT_EQ(result.progress.incumbent,0u);
        EXPECT_EQ(result.progress.lower,0u);
        EXPECT_EQ(result.progress.upper,0u);
    }

    /** @return Classification from independent wide products, not core comparator. */
    Class wideOracle(const MTPSurvivalScoreResult &result, uint32_t d, uint32_t k,
        uint32_t c, uint32_t ck)
    {
        if (k == d) return Class::Incumbent;
        const unsigned __int128 current = static_cast<unsigned __int128>(result.progress.incumbent)*ck;
        if (static_cast<unsigned __int128>(result.progress.upper)*c <= current)
            return Class::Dominated;
        if (static_cast<unsigned __int128>(result.progress.lower)*c > current)
            return Class::Supported;
        return Class::Unresolved;
    }
}

TEST(MTPSurvivalScoreInterval, UnsupportedEmptyAndMalformedCountsNeverBecomeScores)
{
    expectError({},1,1,1,Error::UnsupportedEvidence);
    auto badScope = supported(1,1,1,0,1);
    badScope.scope = static_cast<Scope>(99);
    expectError(badScope,1,1,1,Error::UnsupportedEvidence);
    for (auto d : {0u,16u,UINT32_MAX})
        expectError(supported(d,1,0,1,0),1,1,1,Error::InvalidDepth);
    expectError(supported(1,0,0,0,0),1,1,1,Error::EmptyWindow);
    for (const auto e : {
        supported(1,2,1,0,1), // d1 cannot contain a partial prefix.
        supported(1,2,0,1,1), // Full count requires an accepted token.
        supported(2,2,0,3,0), // Z exceeds R; do not subtract unsigned first.
        supported(2,2,2,1,2), // F exceeds remaining runs.
        supported(2,2,5,0,2), // A exceeds dR.
        supported(3,2,5,0,2), // A below dF.
        supported(3,2,0,0,0), // Interior sum below n.
        supported(3,2,5,0,0), // Interior sum above(d-1)n.
        supported(15,UINT32_MAX,UINT64_MAX,0,0)})
        expectError(e,1,1,1,Error::InconsistentCounts);
    for (auto k : {0u,16u,UINT32_MAX})
        expectError(supported(2,2,2,0,0),k,1,1,Error::InvalidCandidateDepth);
    expectError(supported(2,2,2,0,0),1,0,1,Error::InvalidCost);
    expectError(supported(2,2,2,0,0),1,1,0,Error::InvalidCost);
    expectError(supported(2,2,2,0,0),2,1,2,Error::InvalidCost);
}

TEST(MTPSurvivalScoreInterval, DepthOneTwoTiesAndRightCensoringHaveExplicitMeanings)
{
    const auto one = supported(1,4,3,1,3);
    const auto incumbent = evaluateMTPPrefixSurvivalScore(one,1,100,100);
    ASSERT_TRUE(incumbent.valid());
    EXPECT_EQ(incumbent.classification,Class::Incumbent);
    EXPECT_EQ(incumbent.progress.incumbent,7u);
    EXPECT_EQ(incumbent.progress.lower,7u);
    EXPECT_EQ(incumbent.progress.upper,7u);
    const auto censored = evaluateMTPPrefixSurvivalScore(one,2,100,100);
    ASSERT_TRUE(censored.valid());
    EXPECT_EQ(censored.direction,Direction::Higher);
    EXPECT_EQ(censored.progress.lower,7u);
    EXPECT_EQ(censored.progress.upper,10u);
    EXPECT_EQ(censored.classification,Class::Unresolved);
    EXPECT_EQ(evaluateMTPPrefixSurvivalScore(one,2,100,200).classification,Class::Dominated);

    const auto two = supported(2,5,5,1,1); // Three partial1s, one full2, one zero.
    const auto lower = evaluateMTPPrefixSurvivalScore(two,1,100,80);
    ASSERT_TRUE(lower.valid());
    EXPECT_EQ(lower.direction,Direction::Lower);
    EXPECT_EQ(lower.progress.incumbent,10u);
    EXPECT_EQ(lower.progress.lower,9u);
    EXPECT_EQ(lower.progress.upper,9u);
    EXPECT_EQ(lower.classification,Class::Supported);
    // 9/9 ==10/10:ties never become supported improvements.
    EXPECT_EQ(evaluateMTPPrefixSurvivalScore(two,1,10,9).classification,Class::Dominated);
    // Positive costs need not be monotone. Bounds, not direction, own evidence.
    EXPECT_EQ(evaluateMTPPrefixSurvivalScore(one,2,100,50).classification,Class::Supported);
}

TEST(MTPSurvivalScoreInterval, InteriorAmbiguityAndAdjacentIdentityRemainDistinct)
{
    const auto e = supported(4,63,192,5,39);
    const auto two = evaluateMTPPrefixSurvivalScore(e,2,100,100);
    ASSERT_TRUE(two.valid());
    EXPECT_EQ(two.progress.lower,169u); //63+106; histogram[5,10,1,8,39].
    EXPECT_EQ(two.progress.upper,177u); //63+114; histogram[5,2,17,0,39].
    const auto adjacent = evaluateMTPPrefixSurvivalScore(e,3,100,100);
    EXPECT_EQ(adjacent.progress.lower,216u); // R+A-F.
    EXPECT_EQ(adjacent.progress.upper,216u);
    for (auto d = 3u; d <= 15; ++d)
    {
        // Every run is interior1 or interior d-1; include n0 and q=n/rem0.
        for (auto full : {0u,7u})
            for (auto partial : {0u,9u})
                for (auto upperInterior : {false,true})
                {
                    const uint32_t r = 1+full+partial;
                    const uint64_t a = static_cast<uint64_t>(d)*full+
                        static_cast<uint64_t>(partial)*(upperInterior ? d-1 : 1);
                    const auto result = evaluateMTPPrefixSurvivalScore(supported(d,r,a,1,full),d-1,1,1);
                    ASSERT_TRUE(result.valid());
                    EXPECT_EQ(result.progress.lower,static_cast<uint64_t>(r)+a-full);
                    EXPECT_EQ(result.progress.upper,result.progress.lower);
                }
    }
}

TEST(MTPSurvivalScoreInterval, RepeatedInterleavedCallsHaveNoOwnedStateOrInputMutation)
{
    const auto first = supported(4,63,192,5,39);
    const auto second = supported(1,4,3,1,3);
    for (int repeat = 0; repeat < 100; ++repeat)
    {
        const auto a = evaluateMTPPrefixSurvivalScore(first,2,100,100);
        const auto b = evaluateMTPPrefixSurvivalScore(second,2,100,100);
        ASSERT_TRUE(a.valid());
        ASSERT_TRUE(b.valid());
        EXPECT_EQ(a.progress.lower,169u);
        EXPECT_EQ(a.progress.upper,177u);
        EXPECT_EQ(a.classification,Class::Dominated);
        EXPECT_EQ(b.progress.lower,7u);
        EXPECT_EQ(b.progress.upper,10u);
        EXPECT_EQ(b.classification,Class::Unresolved);
    }
    EXPECT_EQ(first.depth,4u);
    EXPECT_EQ(first.runs,63u);
    EXPECT_EQ(first.accepted,192u);
    EXPECT_EQ(first.zero,5u);
    EXPECT_EQ(first.full,39u);
    EXPECT_EQ(first.scope,Scope::NonStoppedUnclippedVerifiedState);
}

TEST(MTPSurvivalScoreInterval, AllDepthsAndUintExtremesMatchWideClassificationOracle)
{
    constexpr auto max = UINT32_MAX;
    const std::array<uint32_t,5> costs{1,2,65535,max-1,max};
    for (uint32_t d = 1; d <= 15; ++d)
        for (uint32_t k = 1; k <= 15; ++k)
            for (auto c : costs)
                for (auto ck : costs)
                {
                    if (k == d && c != ck) continue;
                    for (const auto e : {
                        supported(d,max,static_cast<uint64_t>(d)*max,0,max),
                        supported(d,max,0,max,0),
                        supported(d,max,d, max-1,1)})
                    {
                        const auto result = evaluateMTPPrefixSurvivalScore(e,k,c,ck);
                        ASSERT_TRUE(result.valid());
                        EXPECT_EQ(result.classification,wideOracle(result,d,k,c,ck));
                        EXPECT_LE(result.progress.upper,uint64_t{16}*max);
                    }
                }
    const auto maximum = evaluateMTPPrefixSurvivalScore(
        supported(15,max,uint64_t{15}*max,0,max),14,max,max-1);
    EXPECT_EQ(maximum.progress.incumbent,uint64_t{16}*max);
    // Direct portable64-bit cross multiplication WOULD overflow here.
    EXPECT_GT(static_cast<unsigned __int128>(maximum.progress.incumbent)*(max-1),
        static_cast<unsigned __int128>(UINT64_MAX));
}

TEST(MTPSurvivalScoreInterval, ExhaustiveHistogramsSupplyIndependentExactIntegerExtrema)
{
    using Key = std::tuple<uint64_t,uint32_t,uint32_t>; //A,Z,F; d/R fixed per map.
    struct Extrema
    {
        std::array<uint64_t,16> lower{}, upper{};
    };
    uint64_t histograms = 0, classes = 0;
    for (uint32_t d = 1; d <= 15; ++d)
        for (uint32_t r = 1; r <= 6; ++r)
        {
            std::map<Key,Extrema> extrema;
            std::vector<uint32_t> values;
            std::function<void(uint32_t)> enumerate = [&](uint32_t first)
            {
                if (values.size() < r)
                {
                    for (uint32_t v = first; v <= d; ++v)
                    {
                        values.push_back(v);
                        enumerate(v);
                        values.pop_back();
                    }
                    return;
                }
                ++histograms;
                uint64_t a = 0;
                uint32_t z = 0, f = 0;
                std::array<uint64_t,16> truncated{};
                for (auto v : values)
                {
                    a += v; z += v == 0; f += v == d;
                    for (uint32_t k = 1; k <= d; ++k) truncated[k] += std::min(v,k);
                }
                auto [it,inserted] = extrema.try_emplace(Key{a,z,f},Extrema{truncated,truncated});
                if (!inserted)
                    for (uint32_t k = 1; k <= d; ++k)
                    {
                        it->second.lower[k] = std::min(it->second.lower[k],truncated[k]);
                        it->second.upper[k] = std::max(it->second.upper[k],truncated[k]);
                    }
            };
            enumerate(0);
            classes += extrema.size();
            for (const auto &[key,range] : extrema)
            {
                const auto [a,z,f] = key;
                const auto e = supported(d,r,a,z,f);
                for (uint32_t k = 1; k <= 15; ++k)
                {
                    const auto result = evaluateMTPPrefixSurvivalScore(e,k,1,1);
                    ASSERT_TRUE(result.valid());
                    EXPECT_EQ(result.progress.incumbent,r+a);
                    const auto lo = k <= d ? range.lower[k] : a;
                    const auto hi = k <= d ? range.upper[k] : a+static_cast<uint64_t>(k-d)*f;
                    ASSERT_EQ(result.progress.lower,r+lo);
                    ASSERT_EQ(result.progress.upper,r+hi);
                    EXPECT_EQ(result.classification,wideOracle(result,d,k,1,1));
                }
            }
        }
    EXPECT_EQ(histograms,245134u);
    EXPECT_EQ(classes,12655u);
}
