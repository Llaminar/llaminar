/**
 * @file Test__CPUQuantizedRMSNorm.cpp
 * @brief Actual production-TU quantized RMSNorm geometry, expansion and retained-owner regressions.
 *
 * Original Q16 blocks remain compressed. Tests exercise the public fused call,
 * not a substitute row implementation, and compare grouped results with actual
 * production M=1. Geometry and complete-bank aliases must reject before stores;
 * empty retained calls must complete before any caller retires its metadata.
 * These device-free checks do not certify factories, arena leases or economy.
 */
#include "kernels/cpu/primitives/RMSNormPrimitives.h"
#include "utils/CPUFeatures.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <limits>
#include <span>
#include <thread>
#include <vector>

namespace llaminar2::test
{
using namespace primitives;
namespace
{
/** @brief Fill truthful original Q16 storage with finite, distinct row/block programs.
 * @param blocks Complete owner-issued physical blocks; @param seed Original fixture identity.
 */
void fillQ16(std::vector<Q16_1Block> &blocks,std::uint32_t seed=19)
{
    for(auto &block:blocks)
    {
        seed=1664525u*seed+1013904223u;
        block.d=std::bit_cast<float>(0x39800000u+(seed&0x007fffffu));
        block.sum_qs=0; // Summary metadata is deliberately not the RMS square authority.
        for(auto &q:block.qs)
        {
            seed=1664525u*seed+1013904223u;
            q=std::int16_t(int(seed>>16)-32768);
            block.sum_qs+=q;
        }
    }
}

/** @brief Independent declared reduction identities, never selected from observed output. */
enum class Q16Reduction { ScalarSerial, NativeEight, AVX512Sixteen };
/** @brief The square's rounding boundary is separate from its FP64 reduction identity. */
enum class Q16Square { RoundedFP32, WideFP64 };

/** @brief Bind the reference to the actual declared ISA and explicit scalar intent.
 * @param options Original invocation policy; @return Independent reduction identity.
 * @throws std::logic_error For an unknown ISA rather than silently choosing a reference.
 */
Q16Reduction q16ReferenceProgram(const RMSNormExecOptions &options)
{
    if(options.force_scalar)return Q16Reduction::ScalarSerial;
    switch(activeISALevel())
    {
    case ISALevel::Scalar:return Q16Reduction::ScalarSerial;
    case ISALevel::AVX2:
    case ISALevel::NEON:
    case ISALevel::NEONDotProd:return Q16Reduction::NativeEight;
    case ISALevel::AVX512:return Q16Reduction::AVX512Sixteen;
    }
    throw std::logic_error("Q16 reference requires an explicit known ISA program");
}

/** @brief Independently decode and square one original code with retained scalar rounding.
 * @param block Original block; @param lane Original0..31 code.
 * @param square Explicit FP32-before-widen or deliberately different FP64 square.
 * @return Squared value for the independent reduction, without production helpers.
 */
double q16ReferenceSquare(const Q16_1Block &block,std::size_t lane,Q16Square square)
{
    volatile float decoded=block.d*float(block.qs[lane]);
    if(square==Q16Square::RoundedFP32)
    {
        volatile float value=decoded*decoded;
        return double(value);
    }
    volatile double value=double(decoded)*double(decoded);
    return value;
}

/** @brief Reproduce one declared ISA's original FP64 chains and fixed fold independently.
 * @param input Complete original physical rows; @param reduction Declared ISA/intent identity.
 * @param square Declared square rounding; @return Original FP64 sum for one row.
 * @details AVX512 retains sixteen chains, per-lane low/high merge, then its
 * established half/quarter fold. Scalar is serial; native NEON/AVX2 has eight
 * chains. No tree is inferred from closeness or an actual published result.
 */
double q16ReferenceSum(std::span<const Q16_1Block> input,Q16Reduction reduction,Q16Square square)
{
    if(reduction==Q16Reduction::ScalarSerial)
    {
        double sum=0.0;
        for(const auto &block:input)
            for(std::size_t lane=0;lane<32;++lane)sum+=q16ReferenceSquare(block,lane,square);
        return sum;
    }
    std::array<double,16> chains{};
    if(reduction==Q16Reduction::NativeEight)
    {
        for(const auto &block:input)
            for(std::size_t lane=0;lane<32;++lane)chains[lane%8]+=q16ReferenceSquare(block,lane,square);
        return ((chains[0]+chains[2])+(chains[4]+chains[6]))+
               ((chains[1]+chains[3])+(chains[5]+chains[7]));
    }
    if(reduction!=Q16Reduction::AVX512Sixteen)
        throw std::logic_error("Q16 reference requires a declared reduction identity");
    for(const auto &block:input)
        for(std::size_t lane=0;lane<8;++lane)
        {
            chains[lane]+=q16ReferenceSquare(block,lane,square);
            chains[lane]+=q16ReferenceSquare(block,16+lane,square);
            chains[8+lane]+=q16ReferenceSquare(block,8+lane,square);
            chains[8+lane]+=q16ReferenceSquare(block,24+lane,square);
        }
    std::array<double,8> merged;
    for(std::size_t lane=0;lane<8;++lane)merged[lane]=chains[lane]+chains[8+lane];
    return ((merged[0]+merged[4])+(merged[2]+merged[6]))+
           ((merged[1]+merged[5])+(merged[3]+merged[7]));
}
}

/** @test The raw public API must never silently floor an incomplete compressed row. */
TEST(CPUQuantizedRMSNorm, Q16CompleteGeometryRejectsBeforeStores)
{
    std::vector<Q16_1Block> input(8);fillQ16(input);
    std::vector<float> gamma(128,1.0f),output(256,-37.0f);
    for(const auto columns:{1u,31u,33u,63u,65u})
    {
        const auto before=output;
        EXPECT_THROW(rmsnorm_q16_1_fp32_fused(input.data(),gamma.data(),output.data(),2,columns,1e-5f),std::invalid_argument);
        EXPECT_EQ(output,before)<<"columns="<<columns;
    }
}

/** @test Expanding72-byte blocks to128-byte FP32 fragments cannot reuse input/gamma storage. */
TEST(CPUQuantizedRMSNorm, Q16ExpansionProtectsCompleteOriginalBanks)
{
    std::vector<Q16_1Block> input(8);fillQ16(input);
    std::vector<float> gamma(64,1.0f),output(128,-29.0f);
    const auto before=input;
    EXPECT_THROW(rmsnorm_q16_1_fp32_fused(input.data(),gamma.data(),reinterpret_cast<float*>(input.data()),2,32,1e-5f),std::invalid_argument);
    EXPECT_EQ(std::memcmp(input.data(),before.data(),input.size()*sizeof(Q16_1Block)),0);
    input=before; // Keep the independent gamma-alias witness on pristine source bytes.
    const auto originalGamma=gamma;
    EXPECT_THROW(rmsnorm_q16_1_fp32_fused(input.data(),gamma.data(),gamma.data(),2,32,1e-5f),std::invalid_argument);
    EXPECT_EQ(gamma,originalGamma);
    EXPECT_EQ(output,std::vector<float>(128,-29.0f));
}

/** @test Empty compressed work still joins a retained team before returning. */
TEST(CPUQuantizedRMSNorm, Q16EmptyRetainedInvocationCompletes)
{
    std::atomic<int> arrivals{0},observed{0};
    RMSNormExecOptions options{.allow_parallel=false};
#pragma omp parallel num_threads(3)
    {
        if(omp_get_thread_num())std::this_thread::sleep_for(std::chrono::milliseconds(2*omp_get_thread_num()));
        arrivals.fetch_add(1,std::memory_order_release);
        rmsnorm_q16_1_fp32_fused(nullptr,nullptr,nullptr,0,0,1e-5f,options);
        if(!omp_get_thread_num())observed.store(arrivals.load(std::memory_order_acquire));
    }
    EXPECT_EQ(observed.load(),3);
}

/** @test Native original-row tiles retain actual M=1 bytes for every positive worker budget. */
TEST(CPUQuantizedRMSNorm, Q16AllPositiveWorkersThroughTenRetainSerialBytes)
{
    for(const auto columns:{32u,64u,96u,1056u})
    {
        const auto blocks=columns/32;
        std::vector<Q16_1Block> input(33*blocks+2);fillQ16(input);
        std::vector<float> gamma(columns+2),reference(33*columns),actual(reference.size()+2,-17.0f);
        for(std::size_t column=0;column<gamma.size();++column)gamma[column]=float(int(column%19)-7)*0.0625f;
        const auto saved=input;
        RMSNormExecOptions serial{.allow_parallel=false};
        for(std::size_t row=0;row<33;++row)
            rmsnorm_q16_1_fp32_fused(input.data()+1+row*blocks,gamma.data()+1,reference.data()+row*columns,1,columns,1e-5f,serial);
        for(std::size_t rows=1;rows<=33;++rows)
        {
            if(rows>16&&rows!=31&&rows!=33)continue;
            for(int workers=1;workers<=10;++workers)
            {
                std::fill(actual.begin(),actual.end(),-17.0f);
#pragma omp parallel num_threads(workers)
                {
                    rmsnorm_q16_1_fp32_fused(input.data()+1,gamma.data()+1,actual.data()+1,rows,columns,1e-5f,serial);
                }
                ASSERT_EQ(std::memcmp(actual.data()+1,reference.data(),rows*columns*sizeof(float)),0)
                    <<"rows="<<rows<<" columns="<<columns<<" workers="<<workers;
                ASSERT_EQ(actual.front(),-17.0f);
                ASSERT_TRUE(std::all_of(actual.begin()+1+rows*columns,actual.end(),[](float value){return value==-17.0f;}));
            }
        }
        EXPECT_EQ(std::memcmp(input.data(),saved.data(),input.size()*sizeof(Q16_1Block)),0);
    }
}
/** @test FP32-square-before-widening is a distinct original-storage numerical program. */
TEST(CPUQuantizedRMSNorm, Q16RoundedSquaresAndOriginalMetadataMatchIndependentProgram)
{
    std::size_t distinguished=0;
    for(std::uint32_t seed=1;seed<=64;++seed)
        for(const std::size_t columns:{32u,64u,96u,1056u})
            for(const bool force_scalar:{false,true})
            {
                std::vector<Q16_1Block> input(columns/32);fillQ16(input,seed);
                std::vector<float> gamma(columns),expected(columns),actual(columns);
                const RMSNormExecOptions options{.allow_parallel=false,.force_scalar=force_scalar};
                const auto program=q16ReferenceProgram(options);
                const double sum=q16ReferenceSum(input,program,Q16Square::RoundedFP32);
                const double wrong=q16ReferenceSum(input,program,Q16Square::WideFP64);
                const float inverse=1.0f/std::sqrt(float(sum/double(columns))+1e-5f);
                const float wrong_inverse=1.0f/std::sqrt(float(wrong/double(columns))+1e-5f);
                distinguished+=std::bit_cast<std::uint32_t>(inverse)!=std::bit_cast<std::uint32_t>(wrong_inverse);
                for(std::size_t column=0;column<columns;++column)
                {
                    gamma[column]=float(int(column%17)-5)*0.125f;
                    const auto &block=input[column/32];
                    volatile float decoded=block.d*float(block.qs[column%32]);
                    volatile float normalized=decoded*inverse;
                    volatile float published=normalized*gamma[column];
                    expected[column]=published;
                }
                // The stored signed sum is not an energy summary. This raw math
                // borrow never substitutes it for the original codes/scales.
                for(auto &block:input)block.sum_qs=std::numeric_limits<std::int32_t>::min();
                rmsnorm_q16_1_fp32_fused(input.data(),gamma.data(),actual.data(),1,columns,1e-5f,options);
                ASSERT_EQ(std::memcmp(actual.data(),expected.data(),columns*sizeof(float)),0)
                    <<"seed="<<seed<<" columns="<<columns<<" force_scalar="<<force_scalar;
            }
    EXPECT_GT(distinguished,0u)<<"fixture must distinguish FP32 versus widened FP64 square programs";
}

/** @test The declared serial/eight/sixteen trees remain independently distinguishable at an RN tie. */
TEST(CPUQuantizedRMSNorm, Q16ReductionTreeIdentityRemainsISASpecific)
{
    constexpr std::size_t columns=128;
    std::vector<Q16_1Block> input(4);
    input[0].d=1.0f;input[1].d=std::ldexp(1.0f,-12);
    input[2].d=input[3].d=std::ldexp(1.0f,-27);
    input[0].qs[0]=input[1].qs[0]=1;
    input[2].qs[8]=input[2].qs[24]=input[3].qs[8]=1;
    input[0].sum_qs=input[1].sum_qs=input[3].sum_qs=1;input[2].sum_qs=2;
    const double eight=q16ReferenceSum(input,Q16Reduction::NativeEight,Q16Square::RoundedFP32);
    const double sixteen=q16ReferenceSum(input,Q16Reduction::AVX512Sixteen,Q16Square::RoundedFP32);
    const double serial=q16ReferenceSum(input,Q16Reduction::ScalarSerial,Q16Square::RoundedFP32);
    EXPECT_EQ(eight,1.0+std::ldexp(1.0,-24));
    EXPECT_EQ(serial,eight);
    EXPECT_EQ(sixteen,eight+std::ldexp(1.0,-52));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(float(eight/double(columns))),0x3c000000u);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(float(sixteen/double(columns))),0x3c000001u);
    const float inverse_eight=1.0f/std::sqrt(float(eight/double(columns)));
    const float inverse_sixteen=1.0f/std::sqrt(float(sixteen/double(columns)));
    ASSERT_NE(std::bit_cast<std::uint32_t>(inverse_eight),std::bit_cast<std::uint32_t>(inverse_sixteen));
    std::vector<float> gamma(columns,1.0f),actual(columns),expected(columns);
    for(const bool force_scalar:{false,true})
    {
        const RMSNormExecOptions options{.allow_parallel=false,.force_scalar=force_scalar};
        const auto program=q16ReferenceProgram(options);
        const double sum=q16ReferenceSum(input,program,Q16Square::RoundedFP32);
        const float inverse=1.0f/std::sqrt(float(sum/double(columns)));
        for(std::size_t column=0;column<columns;++column)
        {
            volatile float decoded=input[column/32].d*float(input[column/32].qs[column%32]);
            volatile float normalized=decoded*inverse;
            expected[column]=normalized;
        }
        rmsnorm_q16_1_fp32_fused(input.data(),gamma.data(),actual.data(),1,columns,0.0f,options);
        EXPECT_EQ(std::memcmp(actual.data(),expected.data(),columns*sizeof(float)),0)<<"force_scalar="<<force_scalar;
        EXPECT_EQ(actual[0],program==Q16Reduction::AVX512Sixteen?inverse_sixteen:inverse_eight);
    }
}

/** @test Invalid policy/undefined inverse never publishes a silently repaired result. */
TEST(CPUQuantizedRMSNorm, Q16PolicyAndNonfiniteSummaryFailHard)
{
    std::vector<Q16_1Block> input(1);fillQ16(input);
    std::array<float,32> gamma,output;gamma.fill(1.0f);output.fill(-41.0f);
    const auto before=output;
    for(const float epsilon:{-1.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})
        EXPECT_THROW(rmsnorm_q16_1_fp32_fused(input.data(),gamma.data(),output.data(),1,32,epsilon),std::invalid_argument);
    RMSNormExecOptions t5{.t5_compat_mode=true};
    EXPECT_THROW(rmsnorm_q16_1_fp32_fused(input.data(),gamma.data(),output.data(),1,32,1e-5f,t5),std::invalid_argument);
    // Numeric corruption is fatal inside a collective: an exception cannot
    // cross an OpenMP parallel boundary. The child distinguishes an escaped
    // exception (exit 97) and std::terminate (exit 98) from the real SIGABRT,
    // so only the typed sink can satisfy KilledBySignal(SIGABRT) here.
    const auto death_style=GTEST_FLAG_GET(death_test_style);
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    input[0].d=std::numeric_limits<float>::infinity();
    EXPECT_EXIT({
        std::set_terminate([](){std::_Exit(98);});
        try{rmsnorm_q16_1_fp32_fused(input.data(),gamma.data(),output.data(),1,32,1e-5f);}
        catch(...){std::_Exit(97);}
    },::testing::KilledBySignal(SIGABRT),"CPU RMS numeric contract violation \\[NonfiniteSummary\\]");
    input[0].d=0.0f;
    EXPECT_EXIT({
        std::set_terminate([](){std::_Exit(98);});
        try{rmsnorm_q16_1_fp32_fused(input.data(),gamma.data(),output.data(),1,32,0.0f);}
        catch(...){std::_Exit(97);}
    },::testing::KilledBySignal(SIGABRT),"CPU RMS numeric contract violation \\[NonfiniteInverse\\]");
    GTEST_FLAG_SET(death_test_style,death_style);
    EXPECT_EQ(output,before);
}
} // namespace llaminar2::test
