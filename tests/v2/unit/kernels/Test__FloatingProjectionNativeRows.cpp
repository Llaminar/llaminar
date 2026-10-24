/**
 * @file Test__FloatingProjectionNativeRows.cpp
 * @brief Device-free native argument selection and original-row admission regressions.
 *
 * These tests authenticate immutable host dispatch and compact argument layout.
 * They do not pretend to prove GPU register allocation: the existing native
 * tiny-projection resource/capture tests own that architecture-specific evidence.
 */
#include <gtest/gtest.h>
#include "kernels/common/FloatingProjectionRows.h"
#include <array>
#include <type_traits>
using namespace llaminar2;
namespace {
/** @brief Identify actual native argument alternatives without looking at live values. */
enum class Selection { Prefix, Requests, Work };
/** @brief Return the compile-time argument selected for one immutable launch. */
template<class Rows> constexpr Selection selectionOf() {
    if constexpr(std::is_same_v<Rows,FloatingProjectionPrefixRows>)return Selection::Prefix;
    else if constexpr(std::is_same_v<Rows,FloatingProjectionRequestRows>)return Selection::Requests;
    else {static_assert(std::is_same_v<Rows,FloatingProjectionRows>);return Selection::Work;}
}
}
TEST(FloatingProjectionNativeRows, LegacyPrefixArgumentRetainsOriginalCompactLayout) {
    static_assert(sizeof(FloatingProjectionPrefixRows)==sizeof(DeviceRowRange));
    static_assert(sizeof(FloatingProjectionRequestRows)==sizeof(DeviceRequestRowRanges));
    static_assert(!FloatingProjectionPrefixRows::masksTileArithmetic);
    static_assert(FloatingProjectionRequestRows::masksTileArithmetic);
    static_assert(FloatingProjectionRows::masksTileArithmetic);
    EXPECT_LT(sizeof(FloatingProjectionPrefixRows),sizeof(FloatingProjectionRows));
    const auto selected=FloatingProjectionRows::dispatchNative(8,nullptr,nullptr,[](auto rows){
        EXPECT_EQ(rows.physicalRows(),8);
        for(int row=-1;row<=8;++row)EXPECT_EQ(rows.rowAdmissionFor(row,8),int(row>=0&&row<8));
        return selectionOf<decltype(rows)>();
    });
    EXPECT_EQ(selected,Selection::Prefix);
}
TEST(FloatingProjectionNativeRows, LegacySlicedRepeatedPrefixMatchesCanonicalOriginalRows) {
    std::int32_t live=0;
    const auto prefix=DeviceRowRange::deviceCounted(8,&live).repeatEachRow(2).slice(3,10);
    const auto canonical=ProjectionRowWork::fromPrefix(prefix);
    for(live=0;live<=8;++live){
        const auto selected=FloatingProjectionRows::dispatchNative(10,&prefix,nullptr,[&](auto rows){
            const auto span=canonical.resolveHost();
            for(int row=0;row<10;++row)EXPECT_EQ(rows.rowAdmissionFor(row,live),int(span.contains(row)));
            EXPECT_EQ(rows.rowAdmissionFor(0,-1),-1);
            EXPECT_EQ(rows.rowAdmissionFor(0,9),-1);
            return selectionOf<decltype(rows)>();
        });
        EXPECT_EQ(selected,Selection::Prefix);
    }
}
TEST(FloatingProjectionNativeRows, IndependentRequestPrefixesRemainCompactAcrossDirtyReplay) {
    std::array<std::int32_t,3> lengths{8,8,8};
    const auto requests=DeviceRequestRowRanges::deviceCounted(3,8,lengths.data());
    const auto canonical=ProjectionRowWork::fromRequests(requests);
    for(const auto publication:{std::array<std::int32_t,3>{8,8,8},{0,1,5},{0,0,0},{8,8,8}}){
        lengths=publication;
        const auto selected=FloatingProjectionRows::dispatchNative(24,nullptr,&requests,[&](auto rows){
            for(int row=0;row<24;++row)
                EXPECT_EQ(rows.rowAdmissionFor(row,lengths[row/8]),int(canonical.resolveHost(row/8).contains(row)));
            return selectionOf<decltype(rows)>();
        });
        EXPECT_EQ(selected,Selection::Requests);
    }
}
TEST(FloatingProjectionNativeRows, ExplicitWorkRetainsMixedRequestThresholdSliceAndRepeatMapping) {
    std::int32_t publication[3][3]={{0,0,0},{2048,6,0},{4096,7,0}};
    const auto work=ProjectionRowWork::causalStrided(3,8,&publication[0][0],&publication[0][1],3,8192,2051)
        .repeatEachRow(2).slice(5,36);
    ProjectionRowWorkScope scope(work);
    const auto selected=FloatingProjectionRows::dispatchNative(36,nullptr,nullptr,[&](auto rows){
        for(int row=0;row<36;++row){
            const int request=work.requestForRow(row);
            EXPECT_EQ(rows.rowAdmissionFor(row,publication[request][1],publication[request][0]),
                int(work.resolveHost(request).contains(row)));
        }
        return selectionOf<decltype(rows)>();
    });
    EXPECT_EQ(selected,Selection::Work);
}
TEST(FloatingProjectionNativeRows, ExplicitWorkValidatesInheritedAuthorityInsteadOfReplacingIt) {
    std::int32_t publication[2][3]={{2048,6,0},{4096,7,0}};
    std::array<std::int32_t,2> inherited{6,7};
    const auto requests=DeviceRequestRowRanges::deviceCounted(2,8,inherited.data());
    const auto work=ProjectionRowWork::causalStrided(2,8,&publication[0][0],&publication[0][1],3,8192,2051);
    ProjectionRowWorkScope scope(work);
    auto validate=[&](auto rows){
        if constexpr(std::is_same_v<decltype(rows),FloatingProjectionRows>)rows.validateHostPublications();
        return selectionOf<decltype(rows)>();
    };
    EXPECT_EQ(FloatingProjectionRows::dispatchNative(16,nullptr,&requests,validate),Selection::Work);
    inherited[1]=6;
    EXPECT_THROW((void)FloatingProjectionRows::dispatchNative(16,nullptr,&requests,validate),std::invalid_argument);
}
TEST(FloatingProjectionNativeRows, GeometryAndConflictingAuthoritiesFailBeforeLaunchCallback) {
    int calls=0;auto launch=[&](auto){++calls;return 0;};
    const auto prefix=DeviceRowRange::fullyActive(8);
    const auto requests=DeviceRequestRowRanges::fullyActiveRequests(2,4);
    EXPECT_THROW((void)FloatingProjectionRows::dispatchNative(8,&prefix,&requests,launch),std::invalid_argument);
    EXPECT_THROW((void)FloatingProjectionRows::dispatchNative(7,&prefix,nullptr,launch),std::invalid_argument);
    EXPECT_THROW((void)FloatingProjectionRows::dispatchNative(0,nullptr,nullptr,launch),std::invalid_argument);
    ProjectionRowWorkScope scope(ProjectionRowWork::fromPrefix(prefix));
    EXPECT_THROW((void)FloatingProjectionRows::dispatchNative(7,nullptr,nullptr,launch),std::logic_error);
    EXPECT_EQ(calls,0);
}
