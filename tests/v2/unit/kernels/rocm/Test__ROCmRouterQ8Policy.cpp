/**
 * @file Test__ROCmRouterQ8Policy.cpp
 * @brief Device-free replay of router admission, ownership and resource boundaries.
 *
 * Recorded hardware/geometry inputs certify the installed finite policy without
 * opening a runtime. Native captured replay separately authenticates the emitted
 * tile and bytes. Invalid traces fail before an engine can be selected.
 */
#include "kernels/rocm/moe/ROCmRouterQ8Policy.h"
#include <gtest/gtest.h>
#include <array>
#include <utility>

namespace
{
using namespace llaminar2;
using namespace llaminar2::rocm;
/** @return Complete policy-relevant recorded gfx1100 hardware. */
ROCmDeviceHardware hardware()
{
    ROCmDeviceHardware result;
    result.ordinal = 0;
    result.isa_target = "gfx1100";
    result.maximum_threads_per_block = 1024;
    result.lds_per_block = 65536;
    return result;
}
/** @brief Keep ordinary-capacity fixtures explicit, including malformed shape records.
 * @param mode Requested engine. @param tile Diagnostic tile. @param device Recorded hardware.
 * @param shape Original geometry; a separate valid row descriptor lets admission reject malformed capacity.
 * @return Selection without any exact-hosted scheduling fact. */
RouterQ8PrefillSelection guardedSelection(RouterQ8PrefillMode mode,RouterQ8WaveTile tile,
    const ROCmDeviceHardware &device,RouterQ8PrefillShape shape)
{
    return RouterQ8PrefillSelection::resolve(mode,tile,device,shape,
        DeviceRowRange::fullyActive(shape.capacity>0?shape.capacity:1),{});
}
constexpr RouterQ8PrefillShape qualified{512,2560,512,RouterQ8Basis::Original,RouterQ8RowOwnership::WholeRows};
/** @test Both bases select sixteen waves independently of valid diagnostic tiles. */
TEST(ROCmRouterQ8Policy, AutomaticExactKeyReplay)
{
    for (auto basis : {RouterQ8Basis::Original,RouterQ8Basis::SignedHadamard32})
        for (auto requested : {RouterQ8WaveTile::Four,RouterQ8WaveTile::Eight,RouterQ8WaveTile::Sixteen})
            for (const auto *isa : {"gfx1100","gfx1100:xnack-"})
            {
                auto shape = qualified; shape.basis = basis;
                auto device = hardware(); device.isa_target = isa;
                const auto selection = guardedSelection(RouterQ8PrefillMode::Automatic,requested,device,shape);
                ASSERT_TRUE(selection.usesWmma());
                EXPECT_EQ(selection.nativeTile(),RouterQ8WaveTile::Sixteen);
                EXPECT_EQ(selection.device(),0);
                EXPECT_EQ(selection.shape().basis,basis);
                EXPECT_EQ(selection.shape().capacity,512);
            }
}
/** @test Unmeasured shapes and compact ownership retain their installed ordered engine. */
TEST(ROCmRouterQ8Policy, NeighborAndOwnedReplay)
{
    std::array shapes{qualified,qualified,qualified,qualified,qualified,qualified,qualified,qualified};
    shapes[0].capacity = 511; shapes[1].capacity = 513;
    shapes[2].width = 2528; shapes[3].width = 2592;
    shapes[4].experts = 511; shapes[5].experts = 513;
    shapes[6].ownership = RouterQ8RowOwnership::CompactOwned;
    shapes[7].ownership = RouterQ8RowOwnership::CompactOwned; shapes[7].capacity = 1024;
    for (const auto &shape : shapes)
    {
        const auto selection = guardedSelection(RouterQ8PrefillMode::Automatic,RouterQ8WaveTile::Four,hardware(),shape);
        EXPECT_FALSE(selection.usesWmma());
        EXPECT_THROW((void)selection.nativeTile(),std::logic_error);
    }
}
/** @test Architecture and exact resource boundaries prevent unsupported native selection. */
TEST(ROCmRouterQ8Policy, HardwareResourceReplay)
{
    for (const auto *isa : {"gfx906","gfx1101","gfx11000","gfx1100-invalid",""})
    {
        auto device = hardware(); device.isa_target = isa;
        EXPECT_FALSE(guardedSelection(RouterQ8PrefillMode::Automatic,RouterQ8WaveTile::Four,device,qualified).usesWmma());
        EXPECT_THROW((void)guardedSelection(RouterQ8PrefillMode::NativeWmma,RouterQ8WaveTile::Four,device,qualified),std::invalid_argument);
    }
    for (int resource : {0,1})
    {
        auto device = hardware();
        device.maximum_threads_per_block = resource == 0 ? 1023 : 1024;
        device.lds_per_block = resource == 1 ? 16383 : 16384;
        EXPECT_FALSE(guardedSelection(RouterQ8PrefillMode::Automatic,RouterQ8WaveTile::Four,device,qualified).usesWmma());
        EXPECT_THROW((void)guardedSelection(RouterQ8PrefillMode::NativeWmma,RouterQ8WaveTile::Sixteen,device,qualified),std::invalid_argument);
        EXPECT_TRUE(guardedSelection(RouterQ8PrefillMode::NativeWmma,RouterQ8WaveTile::Eight,device,qualified).usesWmma());
    }
    auto boundary = hardware(); boundary.lds_per_block = 16384;
    EXPECT_TRUE(guardedSelection(RouterQ8PrefillMode::Automatic,RouterQ8WaveTile::Four,boundary,qualified).usesWmma());
}
/** @test Explicit engines remain independent of Automatic's finite installation domain. */
TEST(ROCmRouterQ8Policy, ExplicitModesReplay)
{
    for (auto ownership : {RouterQ8RowOwnership::WholeRows,RouterQ8RowOwnership::CompactOwned})
        for (auto tile : {RouterQ8WaveTile::Four,RouterQ8WaveTile::Eight,RouterQ8WaveTile::Sixteen})
        {
            auto shape = qualified; shape.ownership = ownership; shape.capacity = 17; shape.width = 96; shape.experts = 17;
            const auto native = guardedSelection(RouterQ8PrefillMode::NativeWmma,tile,hardware(),shape);
            EXPECT_TRUE(native.usesWmma()); EXPECT_EQ(native.nativeTile(),tile);
            EXPECT_FALSE(guardedSelection(RouterQ8PrefillMode::OrderedDot4,tile,hardware(),qualified).usesWmma());
        }
}
/** @test Malformed records cannot hide behind Automatic no-match or forced Ordered. */
TEST(ROCmRouterQ8Policy, RejectsIncompleteReplay)
{
    std::array shapes{qualified,qualified,qualified,qualified,qualified,qualified};
    shapes[0].capacity = 0; shapes[1].width = 0; shapes[2].width = 2559; shapes[3].experts = 0;
    shapes[4].basis = static_cast<RouterQ8Basis>(255);
    shapes[5].ownership = static_cast<RouterQ8RowOwnership>(255);
    for (auto mode : {RouterQ8PrefillMode::Automatic,RouterQ8PrefillMode::OrderedDot4,RouterQ8PrefillMode::NativeWmma})
    {
        for (const auto &shape : shapes)
            EXPECT_THROW((void)guardedSelection(mode,RouterQ8WaveTile::Four,hardware(),shape),std::invalid_argument);
        EXPECT_THROW((void)guardedSelection(mode,static_cast<RouterQ8WaveTile>(255),hardware(),qualified),std::invalid_argument);
        auto invalid = hardware(); invalid.ordinal = -1;
        EXPECT_THROW((void)guardedSelection(mode,RouterQ8WaveTile::Four,invalid,qualified),std::invalid_argument);
    }
    EXPECT_THROW((void)guardedSelection(static_cast<RouterQ8PrefillMode>(255),RouterQ8WaveTile::Four,hardware(),qualified),std::invalid_argument);
}
/** @test Exact hosted selection is total over physical widths, both bases and measured geometry. */
TEST(ROCmRouterQ8Policy, ExactHostedWidthsReplay)
{
    int32_t count=0;
    for(int m=1;m<=33;++m)
      for(const auto [k,e]:std::array<std::pair<int,int>,6>{{{2048,256},{2560,512},{3072,256},{8192,512},{2048,257},{8160,512}}})
        for(auto basis:{RouterQ8Basis::Original,RouterQ8Basis::SignedHadamard32})
          for(auto requested:{RouterQ8WaveTile::Four,RouterQ8WaveTile::Eight,RouterQ8WaveTile::Sixteen}) {
            const auto rows=DeviceRowRange::deviceCounted(m,&count);
            const auto extent=m>=2 ? CapturedVerifierExtent::exactHostedRequest(DeviceId::rocm(0),rows) : CapturedVerifierExtent{};
            const RouterQ8PrefillShape shape{m,k,e,basis,RouterQ8RowOwnership::WholeRows};
            const int first=k==8192 && e==512 ? 9 :
                (k==3072 && e==256) || (k==2048 && e==256 && basis==RouterQ8Basis::Original) ? 6 : 5;
            const bool measured=m>=first && m<=16 && ((k==2048 && e==256)||(k==2560 && e==512)||(k==3072 && e==256)||(k==8192 && e==512));
            const auto selected=RouterQ8PrefillSelection::resolve(RouterQ8PrefillMode::Automatic,requested,hardware(),shape,rows,extent);
            EXPECT_EQ(selected.usesWmma(),measured);
            if(measured)EXPECT_EQ(selected.nativeTile(),RouterQ8WaveTile::Sixteen);
            EXPECT_TRUE(selected.matchesRows(rows));
            EXPECT_FALSE(RouterQ8PrefillSelection::resolve(RouterQ8PrefillMode::Automatic,requested,hardware(),shape,rows,{}).usesWmma());
        }
}

/** @test Short-width native admission and submission cannot borrow another count or transformed rows. */
TEST(ROCmRouterQ8Policy, ExactHostedAuthorityReplay)
{
    int32_t count=0,other=0;
    const auto rows=DeviceRowRange::deviceCounted(16,&count);
    const auto extent=CapturedVerifierExtent::exactHostedRequest(DeviceId::rocm(0),rows);
    const auto wrong_device=CapturedVerifierExtent::exactHostedRequest(DeviceId::rocm(1),rows);
    const auto other_rows=DeviceRowRange::deviceCounted(16,&other);
    const auto prefix=DeviceRowRange::deviceCounted(17,&count).slice(0,16);
    const auto shifted=DeviceRowRange::deviceCounted(17,&count).slice(1,16);
    const auto repeated=DeviceRowRange::deviceCounted(8,&count).repeatEachRow(2);
    for(const auto [k,e]:std::array<std::pair<int,int>,4>{{{2048,256},{2560,512},{3072,256},{8192,512}}})
      for(auto basis:{RouterQ8Basis::Original,RouterQ8Basis::SignedHadamard32}) {
        RouterQ8PrefillShape shape{16,k,e,basis,RouterQ8RowOwnership::WholeRows};
        const auto resolve=[&](const DeviceRowRange &original,CapturedVerifierExtent fact) {
            return RouterQ8PrefillSelection::resolve(RouterQ8PrefillMode::Automatic,RouterQ8WaveTile::Four,hardware(),shape,original,fact);
        };
        const auto selected=resolve(rows,extent);
        ASSERT_TRUE(selected.usesWmma());
        EXPECT_FALSE(selected.matchesRows(other_rows));
        EXPECT_FALSE(selected.matchesRows(DeviceRowRange::fullyActive(16)));
        EXPECT_FALSE(selected.matchesRows(prefix));
        EXPECT_FALSE(selected.matchesRows(shifted));
        EXPECT_FALSE(selected.matchesRows(repeated));
        EXPECT_FALSE(resolve(rows,{}).usesWmma());
        EXPECT_FALSE(resolve(rows,wrong_device).usesWmma());
        EXPECT_FALSE(resolve(other_rows,extent).usesWmma());
        EXPECT_FALSE(resolve(DeviceRowRange::fullyActive(16),extent).usesWmma());
        EXPECT_FALSE(resolve(prefix,extent).usesWmma());
        EXPECT_THROW((void)resolve(shifted,extent),std::invalid_argument);
        EXPECT_THROW((void)resolve(repeated,extent),std::invalid_argument);
        EXPECT_THROW((void)resolve(DeviceRowRange::deviceCounted(15,&count),extent),std::invalid_argument);
        for(int live:{0,1,16}) {count=live;EXPECT_TRUE(resolve(rows,extent).usesWmma());}
        shape.ownership=RouterQ8RowOwnership::CompactOwned;
        EXPECT_FALSE(resolve(rows,extent).usesWmma());
        shape.ownership=RouterQ8RowOwnership::WholeRows;
        for(int edge:{0,1}) {
            auto hw=hardware();if(edge==0)hw.maximum_threads_per_block=1023;else hw.lds_per_block=16383;
            EXPECT_FALSE(RouterQ8PrefillSelection::resolve(RouterQ8PrefillMode::Automatic,RouterQ8WaveTile::Four,hw,shape,rows,extent).usesWmma());
        }
        for(auto mode:{RouterQ8PrefillMode::OrderedDot4,RouterQ8PrefillMode::NativeWmma}) {
            const auto explicit_mode=RouterQ8PrefillSelection::resolve(mode,RouterQ8WaveTile::Eight,hardware(),shape,rows,extent);
            EXPECT_EQ(explicit_mode.usesWmma(),mode==RouterQ8PrefillMode::NativeWmma);
            if(explicit_mode.usesWmma())EXPECT_EQ(explicit_mode.nativeTile(),RouterQ8WaveTile::Eight);
            EXPECT_TRUE(explicit_mode.matchesRows(other_rows));
        }
    }
    const auto guarded_rows=DeviceRowRange::deviceCounted(512,&count);
    const auto legacy=RouterQ8PrefillSelection::resolve(RouterQ8PrefillMode::Automatic,RouterQ8WaveTile::Four,hardware(),qualified,guarded_rows,extent);
    ASSERT_TRUE(legacy.usesWmma());
    EXPECT_TRUE(legacy.matchesRows(DeviceRowRange::deviceCounted(512,&other)));
}

}
