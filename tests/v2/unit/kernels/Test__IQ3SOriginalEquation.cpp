/**
 * @file Test__IQ3SOriginalEquation.cpp
 * @brief Original-source fixed-Q8 CPU regression independent of future metadata.
 *
 * This tests-only negative compiles before the source-factor representation is
 * installed. Expected values read original blocks, never prepared scale words;
 * the same source equation remains unchanged after the representation repair.
 */
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstring>
#include <omp.h>
#include "tensors/TensorClasses.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemv.h"
#include "kernels/cpu/gemm/CPUNativeAVX2Gemv.h"
#include "../../utils/IQ3SOriginalEquationSource.h"
namespace llaminar2::test {
/** @brief Original d times odd source coefficient survives unchanged Q8 arithmetic. */
TEST(IQ3SOriginalEquation, OriginalFiniteFactorWithFixedQ8)
{
    IQ3_SBlock block{};block.d=0x3c01u;block.scales[0]=1;
    std::vector<uint8_t> bytes(sizeof(block));std::memcpy(bytes.data(),&block,sizeof(block));
    IQ3_STensor source({1,32},bytes);
    cpu::native_vnni::CPUNativeVNNIPackedWeights packed;
    const int threads=omp_get_max_threads();omp_set_num_threads(1);
    const bool prepared=cpu::native_vnni::packWeightsCPUNativeVNNI(&source,packed);
    omp_set_num_threads(threads);ASSERT_TRUE(prepared);
    Q8_1Block input{};input.d=0x3c00u;input.qs[0]=1;input.qs[31]=127;input.sum_qs=128;
    const auto original=input;float out=-123.f;
    // Exact original equation: (1+2^-10)*3*(1+127), no half product.
    ASSERT_EQ(float((1.0+1.0/1024.0)*3.0*128.0),384.375f);
    cpu::native_vnni::gemv_native_vnni_scalar(packed,&input,&out,1,1);
    EXPECT_EQ(std::bit_cast<uint32_t>(out),std::bit_cast<uint32_t>(384.375f));
    EXPECT_EQ(std::memcmp(&input,&original,sizeof(input)),0);
    EXPECT_EQ(std::memcmp(source.raw_data(),bytes.data(),bytes.size()),0);
}
/** @brief All source factors/signs and K/N tails enter actual scalar/ISA consumers. */
TEST(IQ3SOriginalEquation, ScalarAndCompiledSIMDOriginalEquation)
{
    using namespace cpu::native_vnni;namespace eq=iq3s_original_equation;
    auto source=eq::source();CPUNativeVNNIPackedWeights packed;
    const int threads=omp_get_max_threads();omp_set_num_threads(1);
    const bool prepared=packWeightsCPUNativeVNNI(source.view.get(),packed);
    omp_set_num_threads(threads);ASSERT_TRUE(prepared);
    for(int active:{0,1,7,8})for(int sign:{1,-1}) {
        std::array<Q8_1Block,eq::k/32> input{};for(auto&q:input)q.d=0x3c00u;
        input[active].qs[0]=int8_t(sign);input[active].qs[31]=int8_t(sign*127);input[active].sum_qs=128*sign;
        const auto original=input;std::vector<float> scalar(source.n);
        gemv_native_vnni_scalar(packed,input.data(),scalar.data(),source.n,eq::k/32);
        for(int chunk=0;chunk<packed.N_padded/64;++chunk) {
            alignas(64) std::array<float,64> avx2{},avx512{};
            gemv_avx2_chunk_non_nibble(packed,input.data(),avx2.data(),chunk,0,eq::k/32);
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
            gemv_native_vnni_avx512_chunk_non_nibble(packed,input.data(),avx512.data(),chunk,0,eq::k/32);
#endif
            for(int n=0;n<64 && chunk*64+n<source.n;++n) {
                const int column=chunk*64+n;const auto expected=std::bit_cast<uint32_t>(eq::expected(source,column,active,sign));
                EXPECT_EQ(std::bit_cast<uint32_t>(scalar[column]),expected)<<column;
                EXPECT_EQ(std::bit_cast<uint32_t>(avx2[n]),expected)<<column;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                EXPECT_EQ(std::bit_cast<uint32_t>(avx512[n]),expected)<<column;
#endif
            }
        }
        EXPECT_EQ(std::memcmp(input.data(),original.data(),sizeof(input)),0);
    }
}
} // namespace llaminar2::test
