/**
 * @file Test__CPUInt32RMSNorm.cpp
 * @brief Actual production-TU full-width integer RMS summary, policy and team-ownership regressions.
 *
 * The exact low-bit witness cannot be explained by ISA reduction drift. Scalar
 * intent uses an independent increasing-column FP64 FMA program; grouped rows
 * additionally compare actual native M1 publications without claiming a common
 * cross-ISA tree. These tests do not qualify the separate INT8 quantizer/factory.
 */
#include "kernels/cpu/primitives/RMSNormPrimitives.h"
#include "utils/CPUFeatures.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>
#include <omp.h>

namespace llaminar2::test
{
using namespace primitives;
namespace
{
constexpr std::uint64_t poisonBits=0x7ff8deadcafebeefu;
/** @brief Exercise the named positive teams without leaking OpenMP runtime policy. */
struct SummaryTeamPolicy
{
    int dynamic=omp_get_dynamic();
    /** @brief Disable dynamic shrinking for this test's explicit team-size receipt. */
    SummaryTeamPolicy(){omp_set_dynamic(0);}
    /** @brief Restore the incoming process policy even after an assertion returns. */
    ~SummaryTeamPolicy(){omp_set_dynamic(dynamic);}
};
/** @brief Require a genuine admitted native profile before certifying an integer summary.
 * @throws std::runtime_error For scalar, foreign or uncompiled native execution intent.
 */
void requireNativeIntegerSummary()
{
    const auto active=activeISALevel();
    if(active==ISALevel::Scalar||!isNativeCPUExecutionISAPair(compiledCPUISALevel(),active))
        throw std::runtime_error("INT32 RMS summary conformance requires an admitted native ISA");
}
/** @brief Execute the independent declared scalar FP64 program on original integers.
 * @param row Complete original signed integers; @return Increasing-K FMA summary.
 */
double serialIntegerSquares(std::span<const std::int32_t>row)
{
    double sum=0.0;
    for(auto integer:row){const auto value=double(integer);sum=std::fma(value,value,sum);}
    return sum;
}
/** @brief Independently spell the declared native eight-chain FP64 program.
 * @param row Complete original integers; @return Fixed original native-tree summary.
 * @details This applies to NEON/AVX2 only, never by choosing the closest output.
 */
double eightIntegerSquares(std::span<const std::int32_t>row)
{
    std::array<double,8>accumulators{};
    for(std::size_t c=0;c<row.size();++c)
    {
        const auto value=double(row[c]);
        accumulators[c%8]=std::fma(value,value,accumulators[c%8]);
    }
    const auto &a=accumulators;
    return ((a[0]+a[2])+(a[4]+a[6]))+((a[1]+a[3])+(a[5]+a[7]));
}
/** @brief Generate original full-width signed words without quantizing through FP32.
 * @param count Exact live values; @param seed Deterministic fixture identity.
 * @return Original integers, including values well beyond FP32's exact domain.
 */
std::vector<std::int32_t> integers(std::size_t count,std::uint32_t seed)
{
    std::vector<std::int32_t>result(count);
    for(auto &value:result){seed=seed*1664525u+1013904223u;value=std::bit_cast<std::int32_t>(seed);}
    return result;
}
} // namespace

TEST(CPUInt32RMSNorm,FullWidthSummaryNeverRoundsOriginalIntegersThroughFP32)
{
    requireNativeIntegerSummary();
    constexpr double expected=2251800082120712.0;
    std::array<std::int32_t,24>source;source.fill(16777217);
    std::array<double,3>output;output.fill(std::bit_cast<double>(poisonBits));
    rmsnorm_compute_row_sumsq_int32_vectorized(source.data(),3,8,output.data(),{.allow_parallel=false});
    for(auto value:output)EXPECT_EQ(std::bit_cast<std::uint64_t>(value),std::bit_cast<std::uint64_t>(expected));
    EXPECT_NE(expected,2251799813685248.0);
}

TEST(CPUInt32RMSNorm,ExplicitScalarUsesOriginalIncreasingColumnFP64Program)
{
    requireNativeIntegerSummary();
    for(std::uint32_t seed=1;seed<=64;++seed)
    {
        const auto source=integers(127,seed);
        const auto expected=serialIntegerSquares(source);
        double actual=std::bit_cast<double>(~std::bit_cast<std::uint64_t>(expected));
        rmsnorm_compute_row_sumsq_int32_vectorized(source.data(),1,source.size(),&actual,
            {.allow_parallel=false,.force_scalar=true});
        ASSERT_EQ(std::bit_cast<std::uint64_t>(actual),std::bit_cast<std::uint64_t>(expected))<<"seed "<<seed;
    }
}

TEST(CPUInt32RMSNorm,OriginalBankIntersectionsAndUndeclaredPolicyRejectBeforeStores)
{
    alignas(double) std::array<std::int32_t,64>source;source.fill(11);const auto before=source;
    auto *intersecting=reinterpret_cast<double*>(source.data()+8);
    EXPECT_THROW(rmsnorm_compute_row_sumsq_int32_vectorized(source.data(),2,8,intersecting,
        {.allow_parallel=false}),std::invalid_argument);
    EXPECT_EQ(source,before);
    std::array<double,2>output;output.fill(std::bit_cast<double>(poisonBits));const auto original=output;
    EXPECT_THROW(rmsnorm_compute_row_sumsq_int32_vectorized(source.data(),2,8,output.data(),
        {.allow_parallel=false,.t5_compat_mode=true}),std::invalid_argument);
    for(std::size_t i=0;i<output.size();++i)EXPECT_EQ(std::bit_cast<std::uint64_t>(output[i]),std::bit_cast<std::uint64_t>(original[i]));
    EXPECT_THROW(rmsnorm_compute_row_sumsq_int32_vectorized(source.data(),std::numeric_limits<std::size_t>::max(),8,output.data()),std::invalid_argument);
    EXPECT_THROW(rmsnorm_compute_row_sumsq_int32_vectorized(reinterpret_cast<const std::int32_t*>(std::numeric_limits<std::uintptr_t>::max()-4),1,8,output.data()),std::invalid_argument);
    EXPECT_THROW(rmsnorm_compute_row_sumsq_int32_vectorized(source.data(),1,8,reinterpret_cast<double*>(std::numeric_limits<std::uintptr_t>::max()-4)),std::invalid_argument);
    RMSNormExecOptions options{.allow_parallel=false,.parallel_threshold_elems=31};
    auto *policyOutput=reinterpret_cast<double*>(&options.parallel_threshold_elems);
    EXPECT_THROW(rmsnorm_compute_row_sumsq_int32_vectorized(source.data(),1,8,policyOutput,options),std::invalid_argument);
    EXPECT_FALSE(options.allow_parallel);EXPECT_EQ(options.parallel_threshold_elems,31u);
    EXPECT_FALSE(options.t5_compat_mode);EXPECT_FALSE(options.force_scalar);
}

TEST(CPUInt32RMSNorm,EmptyRetainedSummaryCompletesBeforeRetirement)
{
    std::atomic<int>arrived{0},early{0};
#pragma omp parallel num_threads(3)
    {
        if(omp_get_thread_num()==2)std::this_thread::sleep_for(std::chrono::milliseconds(10));
        arrived.fetch_add(1,std::memory_order_release);
        rmsnorm_compute_row_sumsq_int32_vectorized(nullptr,0,0,nullptr,{.allow_parallel=false});
        if(arrived.load(std::memory_order_acquire)!=3)early.fetch_add(1,std::memory_order_relaxed);
    }
    EXPECT_EQ(arrived.load(),3);EXPECT_EQ(early.load(),0);
}

TEST(CPUInt32RMSNorm,SmallRetainedSummaryHasOnePublicationOwner)
{
    std::array<std::int32_t,8>source;source.fill(16777217);
    double output=std::bit_cast<double>(poisonBits);
    std::atomic<int>arrived{0},observed{0};
#pragma omp parallel num_threads(3)
    {
        if(omp_get_thread_num()==2)std::this_thread::sleep_for(std::chrono::milliseconds(10));
        arrived.fetch_add(1,std::memory_order_release);
        rmsnorm_compute_row_sumsq_int32_vectorized(source.data(),1,source.size(),&output,{.allow_parallel=false});
        if(omp_get_thread_num()==0)
        {
            observed.store(arrived.load(std::memory_order_acquire));
            output=std::bit_cast<double>(poisonBits); // Retire/reuse immediately after collective return.
        }
    }
    EXPECT_EQ(observed.load(),3);EXPECT_EQ(std::bit_cast<std::uint64_t>(output),poisonBits);
}

TEST(CPUInt32RMSNorm,AllPositiveWorkersThroughTenRetainOriginalSummaries)
{
    requireNativeIntegerSummary();
    SummaryTeamPolicy policy;
    for(std::size_t columns:{1u,3u,4u,7u,8u,9u,16u,17u,31u,32u,33u,127u,128u,129u,257u,4096u,8192u})
        for(std::size_t rows:{1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u,12u,13u,14u,15u,16u,31u,33u})
        {
            const auto source=integers(rows*columns,19);const auto original=source;
            std::vector<double>expected(rows),actual(rows);
            for(std::size_t row=0;row<rows;++row)
            {
                expected[row]=std::bit_cast<double>(poisonBits);
                rmsnorm_compute_row_sumsq_int32_vectorized(source.data()+row*columns,1,columns,expected.data()+row,
                    {.allow_parallel=false});
                ASSERT_TRUE(std::isfinite(expected[row]));
                const auto isa=activeISALevel();
                if(isa==ISALevel::NEON||isa==ISALevel::NEONDotProd||isa==ISALevel::AVX2)
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(expected[row]),
                        std::bit_cast<std::uint64_t>(eightIntegerSquares(std::span(source).subspan(row*columns,columns))));
            }
            for(int workers=1;workers<=10;++workers)
            {
                for(std::size_t row=0;row<rows;++row)actual[row]=std::bit_cast<double>(~std::bit_cast<std::uint64_t>(expected[row]));
                int actualWorkers=0;
#pragma omp parallel num_threads(workers)
                {
#pragma omp single
                    actualWorkers=omp_get_num_threads();
                    rmsnorm_compute_row_sumsq_int32_vectorized(source.data(),rows,columns,actual.data(),{.allow_parallel=false});
                }
                ASSERT_EQ(actualWorkers,workers);
                for(std::size_t row=0;row<rows;++row)
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(actual[row]),std::bit_cast<std::uint64_t>(expected[row]));
                EXPECT_EQ(source,original);
            }
        }
}
} // namespace llaminar2::test
