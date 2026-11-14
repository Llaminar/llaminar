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
                const auto selection = RouterQ8PrefillSelection::resolve(RouterQ8PrefillMode::Automatic,requested,device,shape);
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
        const auto selection = RouterQ8PrefillSelection::resolve(RouterQ8PrefillMode::Automatic,RouterQ8WaveTile::Four,hardware(),shape);
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
        EXPECT_FALSE(RouterQ8PrefillSelection::resolve(RouterQ8PrefillMode::Automatic,RouterQ8WaveTile::Four,device,qualified).usesWmma());
        EXPECT_THROW((void)RouterQ8PrefillSelection::resolve(RouterQ8PrefillMode::NativeWmma,RouterQ8WaveTile::Four,device,qualified),std::invalid_argument);
    }
    for (int resource : {0,1})
    {
        auto device = hardware();
        device.maximum_threads_per_block = resource == 0 ? 1023 : 1024;
        device.lds_per_block = resource == 1 ? 16383 : 16384;
        EXPECT_FALSE(RouterQ8PrefillSelection::resolve(RouterQ8PrefillMode::Automatic,RouterQ8WaveTile::Four,device,qualified).usesWmma());
        EXPECT_THROW((void)RouterQ8PrefillSelection::resolve(RouterQ8PrefillMode::NativeWmma,RouterQ8WaveTile::Sixteen,device,qualified),std::invalid_argument);
        EXPECT_TRUE(RouterQ8PrefillSelection::resolve(RouterQ8PrefillMode::NativeWmma,RouterQ8WaveTile::Eight,device,qualified).usesWmma());
    }
    auto boundary = hardware(); boundary.lds_per_block = 16384;
    EXPECT_TRUE(RouterQ8PrefillSelection::resolve(RouterQ8PrefillMode::Automatic,RouterQ8WaveTile::Four,boundary,qualified).usesWmma());
}
/** @test Explicit engines remain independent of Automatic's finite installation domain. */
TEST(ROCmRouterQ8Policy, ExplicitModesReplay)
{
    for (auto ownership : {RouterQ8RowOwnership::WholeRows,RouterQ8RowOwnership::CompactOwned})
        for (auto tile : {RouterQ8WaveTile::Four,RouterQ8WaveTile::Eight,RouterQ8WaveTile::Sixteen})
        {
            auto shape = qualified; shape.ownership = ownership; shape.capacity = 17; shape.width = 96; shape.experts = 17;
            const auto native = RouterQ8PrefillSelection::resolve(RouterQ8PrefillMode::NativeWmma,tile,hardware(),shape);
            EXPECT_TRUE(native.usesWmma()); EXPECT_EQ(native.nativeTile(),tile);
            EXPECT_FALSE(RouterQ8PrefillSelection::resolve(RouterQ8PrefillMode::OrderedDot4,tile,hardware(),qualified).usesWmma());
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
            EXPECT_THROW((void)RouterQ8PrefillSelection::resolve(mode,RouterQ8WaveTile::Four,hardware(),shape),std::invalid_argument);
        EXPECT_THROW((void)RouterQ8PrefillSelection::resolve(mode,static_cast<RouterQ8WaveTile>(255),hardware(),qualified),std::invalid_argument);
        auto invalid = hardware(); invalid.ordinal = -1;
        EXPECT_THROW((void)RouterQ8PrefillSelection::resolve(mode,RouterQ8WaveTile::Four,invalid,qualified),std::invalid_argument);
    }
    EXPECT_THROW((void)RouterQ8PrefillSelection::resolve(static_cast<RouterQ8PrefillMode>(255),RouterQ8WaveTile::Four,hardware(),qualified),std::invalid_argument);
}
}
