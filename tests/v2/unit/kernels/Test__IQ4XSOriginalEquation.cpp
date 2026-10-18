/**
 * @file Test__IQ4XSOriginalEquation.cpp
 * @brief Actual CPU fixed-Q8 source-equation regression independent of packed layout.
 *
 * The ordinary literal detects today's derived-half loss. A second sweep covers
 * all signed source coefficients, finite half domains, view offsets and K tails;
 * IQ4_NL explicitly remains compatible. SIMD/optimized fused proofs are separate
 * integration cases, so this Unit fixture stays small and device-free.
 */
#include <gtest/gtest.h>
#include <omp.h>
#include <array>
#include <bit>
#include "../../utils/IQ4XSOriginalEquationSource.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemv.h"
namespace llaminar2::test
{
/** @brief Retain the caller's worker policy across bounded source preparation. */
class IQ4XSOriginalEquation : public ::testing::Test
{
protected:
    /** @brief Use two preparation workers for tiny independent chunks. */
    void SetUp() override { previous_=omp_get_max_threads();omp_set_num_threads(2); }
    /** @brief Restore the caller even after a fatal assertion. */
    void TearDown() override { omp_set_num_threads(previous_); }
private:
    int previous_=1;
};
/** @brief Original half×coefficient must survive unchanged fixed-Q8 scalar arithmetic. */
TEST_F(IQ4XSOriginalEquation, OrdinaryFiniteFactorWithFixedQ8)
{
    IQ4_XSBlock block{};block.d=0x3c01u;block.scales_l[0]=3u;block.scales_h=2u;
    for(auto &q:block.qs)q=0x88u;
    // Other subblocks encode coefficient0, not negative32.
    for(int sub=1;sub<8;++sub)block.scales_h|=uint16_t(2u<<(2*sub));
    std::vector<uint8_t> bytes(sizeof(block));std::memcpy(bytes.data(),&block,sizeof(block));
    IQ4_XSTensor source({1,32},bytes);
    cpu::native_vnni::CPUNativeVNNIPackedWeights packed;
    ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(&source,packed));
    Q8_1Block input{};input.d=0x3c00u;input.qs[0]=1;input.qs[31]=127;input.sum_qs=128;
    const auto unchanged=input;float output=-123.0f;
    ASSERT_EQ(float(double(fp16_to_fp32(block.d))*3.0*128.0),384.375f);
    cpu::native_vnni::gemv_native_vnni_scalar(packed,&input,&output,1,1);
    EXPECT_EQ(std::bit_cast<uint32_t>(output),std::bit_cast<uint32_t>(384.375f));
    EXPECT_EQ(std::memcmp(&input,&unchanged,sizeof(input)),0);
    EXPECT_EQ(std::memcmp(source.raw_data(),bytes.data(),bytes.size()),0);
}
/** @brief Cover source coefficients/domains/tails without choosing a future physical layout. */
TEST_F(IQ4XSOriginalEquation, AllCoefficientsOriginalFactorsAndIQ4NLCompatibility)
{
    using namespace iq4xs_original_equation;
    for(const auto format:{Format::IQ4XS,Format::IQ4NL})
    {
        auto original=source(format);cpu::native_vnni::CPUNativeVNNIPackedWeights packed;
        ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(original.view.get(),packed));
        std::array<Q8_1Block,k/32> input{};std::vector<float> actual(original.n);
        for(int active:{0,1,7,8})for(int sign:{1,-1})
        {
            input={};for(auto &q:input)q.d=0x3c00u;
            input[active].qs[0]=int8_t(sign);input[active].qs[31]=int8_t(sign*127);input[active].sum_qs=128*sign;
            const auto unchanged=input;
            cpu::native_vnni::gemv_native_vnni_scalar(packed,input.data(),actual.data(),original.n,k/32);
            for(int n=0;n<original.n;++n)
                EXPECT_EQ(std::bit_cast<uint32_t>(actual[n]),std::bit_cast<uint32_t>(expected(original,n,active,sign)))
                    <<"format="<<int(format)<<" column="<<n<<" block="<<active<<" sign="<<sign;
            EXPECT_EQ(std::memcmp(input.data(),unchanged.data(),sizeof(input)),0);
        }
        EXPECT_EQ(std::memcmp(original.parent->raw_data(),original.bytes.data(),original.bytes.size()),0);
    }
}
} // namespace llaminar2::test
