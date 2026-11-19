/**
 * @file Test__CPUQ8AttentionValue.cpp
 * @brief Original-Q8 native value FMA and complete-fragment publication contracts.
 *
 * Independent memory-rounded weight/half-scale products feed scalar IEEE FMAs.
 * All signed codes, representative normal/subnormal/zero half scales and
 * non-power-of-two weights discriminate premature code scaling. No model,
 * device, decoded value bank or production fallback is involved.
 */
#include "kernels/cpu/attention/CPUQ8AttentionValue.h"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>

using namespace llaminar2;

/** @test Every native value word retains the separate combined-weight edge and intentional FMA. */
TEST(CPUQ8AttentionValue, NativeOriginalCodesMatchIndependentCombinedWeightFMA)
{
    ASSERT_NE(activeISALevel(),ISALevel::Scalar)<<"this explicit native conformance cell requires its native program";
    std::size_t distinguished=0;
    for(const std::uint16_t half:{0x0000u,0x0001u,0x03ffu,0x0400u,0x3555u,0x3c01u,0x7bffu})
        for(const float weight:{0.0f,0.1f,1e-10f,1.0f})
            for(int first=-128;first<128;first+=32)
            {
                Q8_1Block block{};block.d=half;
                std::array<float,34> actual{},expected{};
                actual.front()=actual.back()=-91.25f;expected=actual;
                volatile float scale=fp16_to_fp32(half);
                volatile float combined=weight*scale;
                for(int lane=0;lane<32;++lane)
                {
                    block.qs[lane]=std::int8_t(first+lane);
                    block.sum_qs+=block.qs[lane];
                    actual[lane+1]=expected[lane+1]=float(lane-13)*0.171f;
                    expected[lane+1]=std::fma(float(block.qs[lane]),float(combined),expected[lane+1]);
                    volatile float prematurely_scaled=float(block.qs[lane])*scale;
                    const float wrong=std::fma(float(prematurely_scaled),weight,actual[lane+1]);
                    distinguished+=std::bit_cast<std::uint32_t>(wrong)!=std::bit_cast<std::uint32_t>(expected[lane+1]);
                }
                const auto saved=block;
                cpu::q8_attention::accumulateValueBlock({actual.data()+1,32},block,weight);
                ASSERT_EQ(std::memcmp(actual.data(),expected.data(),sizeof(actual)),0)
                    <<"half="<<half<<" weight="<<weight<<" first="<<first;
                EXPECT_EQ(std::memcmp(&block,&saved,sizeof(block)),0);
            }
    EXPECT_GT(distinguished,0u)<<"fixture must distinguish premature code scaling from the declared combined-weight program";
}

/** @test Malformed fragments and source intersections reject before publication. */
TEST(CPUQ8AttentionValue, CompleteNumeratorAndSourceAliasesRejectBeforeStores)
{
    Q8_1Block block{};block.d=0x3c00;
    std::array<float,34> output;output.fill(-17.25f);const auto saved=output;
    EXPECT_THROW((cpu::q8_attention::accumulateValueBlock({},block,0.5f)),std::invalid_argument);
    EXPECT_THROW((cpu::q8_attention::accumulateValueBlock({output.data(),31},block,0.5f)),std::invalid_argument);
    EXPECT_EQ(std::memcmp(output.data(),saved.data(),sizeof(output)),0);
    // This intentionally invalid infrastructure alias is rejected by byte
    // geometry before reading the source or interpreting numerator elements.
    alignas(64) std::array<std::byte,256> storage{};
    auto *source=std::construct_at(reinterpret_cast<Q8_1Block*>(storage.data()));
    const auto before=storage;
    EXPECT_THROW((cpu::q8_attention::accumulateValueBlock(
        {reinterpret_cast<float*>(storage.data()),32},*source,0.5f)),std::invalid_argument);
    EXPECT_EQ(storage,before);
    std::destroy_at(source);
}
