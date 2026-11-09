/**
 * @file Test__CPURMSNormWorkshare.cpp
 * @brief Production-TU RMSNorm ownership, completion and original-row byte regressions.
 *
 * Tests call the real public primitives with ordinary and retained OpenMP teams.
 * The wide all-ones witness separates a wrong lexical partial owner from minor
 * reduction drift. Delayed small/empty participants prove completion, including
 * exact input/output aliases that must be normalized only once. This device-free
 * target does not certify factories, model inference or performance/scaling.
 * The existing scalar summary seam is checked against independently rounded
 * FP64 products/sums, including witnesses that separate serial and lane folds.
 */
#include "kernels/cpu/primitives/RMSNormPrimitives.h"
#include "kernels/cpu/primitives/RMSNormPrimitives_detail.h"
#include "utils/CPUFeatures.h"
#include "tensors/FP16Utils.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <type_traits>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace llaminar2::test
{
using namespace primitives;

/** @test Explicit scalar FP32 summaries retain increasing-column FP64 bytes, not a SIMD lane fold. */
TEST(CPURMSNormWorkshare,ScalarFP32SummaryRetainsIncreasingColumnFMA)
{
    EXPECT_EQ(std::bit_cast<std::uint64_t>(primitives::detail::compute_sumsq_scalar(nullptr,0)),0u);
    std::vector<std::size_t>widths;
    for(std::size_t columns=1;columns<=130;++columns)widths.push_back(columns);
    for(std::size_t columns:{255u,256u,257u,511u,512u,513u,896u,1024u,1025u,2047u,4096u})widths.push_back(columns);
    constexpr std::array<std::uint32_t,8>boundaryWords{
        0x00000001u,0x007fffffu,0x00800000u,0x3f800001u,0x7f7fffffu,0xff7fffffu,0x00000000u,0x80000000u};
    constexpr std::size_t guard=7,suffix=9;
    for(const auto columns:widths)for(const auto anchor:std::array<std::size_t,3>{0,columns/2,columns-1})
        for(const bool boundaries:{false,true})
        {
            SCOPED_TRACE(::testing::Message()<<"K="<<columns<<" anchor="<<anchor<<" boundaries="<<boundaries);
            std::vector<float>input(guard+columns+suffix,919.25f);
            for(std::size_t c=0;c<columns;++c)
                input[guard+c]=boundaries?std::bit_cast<float>(boundaryWords[c%boundaryWords.size()]):1.0f;
            input[guard+anchor]=0x1p27f;const auto original=input;
            double expected=0;
            for(std::size_t c=0;c<columns;++c)
            {
                const double value=double(input[guard+c]);
                // Each original FP32 square fits exactly in FP64; volatile
                // addition independently retains every increasing-column edge.
                volatile double square=value*value;
                volatile double rounded=expected+square;expected=rounded;
            }
            if(columns==17&&anchor==0&&!boundaries)
            {
                // At2^54, ULP=4: sixteen serial +1 terms are individually lost.
                // A lane-partitioned reduction retains some before its fold.
                EXPECT_EQ(std::bit_cast<std::uint64_t>(expected),std::bit_cast<std::uint64_t>(0x1p54));
            }
            const auto actual=primitives::detail::compute_sumsq_scalar(input.data()+guard,columns);
            EXPECT_EQ(std::bit_cast<std::uint64_t>(actual),std::bit_cast<std::uint64_t>(expected));
            EXPECT_EQ(std::memcmp(input.data(),original.data(),input.size()*sizeof(float)),0);
        }
}

namespace
{
/** @brief Keep requested retained teams exact without leaking caller OpenMP policy. */
class RMSApplicationTeamPolicy final
{
public:
    /** @brief Retain caller policy and disable dynamic shrinking for this fixture. */
    RMSApplicationTeamPolicy():threads_(omp_get_max_threads()),dynamic_(omp_get_dynamic()){omp_set_dynamic(0);}
    /** @brief Restore the original caller ICVs even on an assertion failure. */
    ~RMSApplicationTeamPolicy(){omp_set_num_threads(threads_);omp_set_dynamic(dynamic_);}
private:
    int threads_,dynamic_;
};
} // namespace

/** @test Admitted native FP32 application preserves rounded edges, tails, exact aliases and guards. */
TEST(CPURMSNormWorkshare,FP32ApplicationRetainsOriginalRoundedEdges)
{
    const auto isa=activeISALevel();
    if(isa==ISALevel::Scalar||!isNativeCPUExecutionISAPair(compiledCPUISALevel(),isa))
        throw std::runtime_error("CPU FP32 RMS application requires an admitted native profile");
    const RMSApplicationTeamPolicy policy;
    std::vector<std::size_t>widths;
    for(std::size_t c=1;c<=65;++c)widths.push_back(c);
    for(std::size_t c:{127u,128u,129u,255u,256u,257u,511u,512u,513u})widths.push_back(c);
    constexpr std::array<std::uint32_t,8>weights{
        0x00000001u,0x80000001u,0x007fffffu,0x807fffffu,0x3f800001u,0xbf400003u,0x00000000u,0x80000000u};
    constexpr std::size_t guard=7,suffix=9;
    const float inverse=float(1.0/std::sqrt(12.0));
    for(const std::size_t rows:{1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u,12u,13u,14u,15u,16u,31u,33u})
        for(const auto columns:widths)for(const bool weighted:{false,true})for(const bool scalar:{false,true})
            for(const bool alias:{false,true})for(const bool retained:{false,true})
            {
                if(retained&&columns!=1&&columns!=17&&columns!=129&&columns!=513)continue;
                SCOPED_TRACE(::testing::Message()<<"M="<<rows<<" K="<<columns<<" gamma="<<weighted<<" scalar="<<scalar<<" alias="<<alias<<" retained="<<retained);
                std::vector<float>source(guard+(rows+2)*columns+suffix,919.25f),gamma(guard+columns+suffix,719.5f);
                for(std::size_t i=0;i<rows*columns;++i)source[guard+i]=(i%2)?-3.0f:3.0f;
                for(std::size_t c=0;c<columns;++c)gamma[guard+c]=std::bit_cast<float>(weights[c%weights.size()]);
                const auto original=source,originalGamma=gamma;
                auto actual=alias?source:std::vector<float>(source.size(),-19.5f);auto expected=actual;
                std::size_t reorderedDifferences=0;
                for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<columns;++c)
                {
                    const auto i=guard+r*columns+c;
                    // Every square is9, so all admitted summary trees yield the
                    // same exact mean9 and staged FP64 inverse1/sqrt(9+3).
                    volatile float scaled=source[i]*inverse;
                    volatile float value=weighted?scaled*gamma[guard+c]:float(scaled);expected[i]=value;
                    if(weighted)
                    {
                        volatile float premature=inverse*gamma[guard+c];
                        volatile float reordered=source[i]*premature;
                        reorderedDifferences+=std::bit_cast<std::uint32_t>(float(reordered))!=std::bit_cast<std::uint32_t>(float(value));
                    }
                }
                if(weighted)ASSERT_GT(reorderedDifferences,0u);
                const RMSNormExecOptions options{.allow_parallel=false,.force_scalar=scalar};
                const auto verify=[&]()
                {
                    EXPECT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),0);
                    EXPECT_EQ(std::memcmp(source.data(),original.data(),source.size()*sizeof(float)),0);
                    EXPECT_EQ(std::memcmp(gamma.data(),originalGamma.data(),gamma.size()*sizeof(float)),0);
                };
                const auto call=[&](){rmsnorm_fused_vectorized(alias?actual.data()+guard:source.data()+guard,
                    weighted?gamma.data()+guard:nullptr,actual.data()+guard,rows,columns,3.0f,options);};
                if(retained)
                {
                    int actualTeam=0;
#pragma omp parallel num_threads(3)
                    {
                        if(omp_get_thread_num()==0)actualTeam=omp_get_num_threads();
                        call();if(omp_get_thread_num()==0)verify();
                    }
                    ASSERT_EQ(actualTeam,3);
                }
                else{call();verify();}
            }
}

/** @test A retained wide M=1 invocation uses the complete row, not each worker's private partial. */
TEST(CPURMSNormWorkshare, WideRetainedRowUsesCompleteOriginalSummary)
{
    constexpr std::size_t columns = 12288;
    std::vector<float> input(columns,1.0f),gamma(columns,1.0f),expected(columns),actual(columns,-19.0f);
    RMSNormExecOptions serial{.allow_parallel=false};
    rmsnorm_fused_vectorized(input.data(),gamma.data(),expected.data(),1,columns,1e-5f,serial);
#pragma omp parallel num_threads(3)
    {
        rmsnorm_fused_vectorized(input.data(),gamma.data(),actual.data(),1,columns,1e-5f);
    }
    std::size_t mismatches=0;
    for(std::size_t column=0;column<columns;++column)
        mismatches+=std::bit_cast<std::uint32_t>(actual[column])!=std::bit_cast<std::uint32_t>(expected[column]);
    EXPECT_EQ(mismatches,0u)<<"expected="<<expected[0]<<" actual="<<actual[0];
}

/** @test A retained small exact-alias invocation consumes the original row once and joins on return. */
TEST(CPURMSNormWorkshare, SmallRetainedAliasCompletesExactlyOnce)
{
    constexpr std::size_t columns=17;
    std::array<float,columns> input,gamma,expected;
    for(std::size_t column=0;column<columns;++column)
    {
        input[column]=float(column+1)*0.125f;
        gamma[column]=0.25f+float(column)*0.03125f;
    }
    RMSNormExecOptions serial{.allow_parallel=false};
    rmsnorm_fused_vectorized(input.data(),gamma.data(),expected.data(),1,columns,1e-5f,serial);
    auto actual=input;
    std::atomic<int> arrivals{0},observed{0};
#pragma omp parallel num_threads(3)
    {
        if(omp_get_thread_num()) std::this_thread::sleep_for(std::chrono::milliseconds(2*omp_get_thread_num()));
        arrivals.fetch_add(1,std::memory_order_release);
        rmsnorm_fused_vectorized(actual.data(),gamma.data(),actual.data(),1,columns,1e-5f,serial);
        if(omp_get_thread_num()==0) observed.store(arrivals.load(std::memory_order_acquire));
    }
    EXPECT_EQ(observed.load(),3);
    EXPECT_EQ(actual,expected);
}

/** @test Empty retained calls still complete the team before the original caller can retire storage. */
TEST(CPURMSNormWorkshare, EmptyRetainedInvocationCompletesBeforeReturn)
{
    std::atomic<int> arrivals{0},observed{0};
#pragma omp parallel num_threads(3)
    {
        if(omp_get_thread_num()) std::this_thread::sleep_for(std::chrono::milliseconds(2*omp_get_thread_num()));
        arrivals.fetch_add(1,std::memory_order_release);
        rmsnorm_fused_vectorized(nullptr,nullptr,nullptr,0,0,1e-5f);
        if(omp_get_thread_num()==0) observed.store(arrivals.load(std::memory_order_acquire));
    }
    EXPECT_EQ(observed.load(),3);
}
namespace
{
/** @brief Original physical floating format; no native numerical helper serves as an oracle. */
enum class Format { FP32, FP16, BF16 };
/** @brief Physical fixture word selected by declared original format. */
template<Format F>using Word=std::conditional_t<F==Format::FP32,float,std::uint16_t>;
/** @brief Encode one original fixture through portable scalar storage rules.
 * @tparam F Original format; @param value Original value; @return Physical word.
 */
template<Format F>Word<F> encode(float value)
{
    if constexpr(F==Format::FP32)return value;
    else if constexpr(F==Format::FP16)return fp32_to_fp16(value);
    else
    {
        const auto bits=std::bit_cast<std::uint32_t>(value);
        return std::uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
    }
}
/** @brief Decode one oracle word through the original scalar storage contract.
 * @tparam F Original format; @param value Original word; @return Exact FP32 value.
 */
template<Format F>float decode(Word<F> value)
{
    if constexpr(F==Format::FP32)return value;
    else if constexpr(F==Format::FP16)return fp16_to_fp32(value);
    else return std::bit_cast<float>(std::uint32_t(value)<<16);
}
/** @brief Invoke the real production-TU matrix entrypoint for one original storage kind.
 * @tparam F Original format; @param input Original values; @param gamma Original weights.
 * @param output Original result; @param rows Original M; @param columns Original K.
 * @param opts Explicit execution/arithmetic intent.
 */
template<Format F>void invoke(const Word<F>*input,const float*gamma,Word<F>*output,
    std::size_t rows,std::size_t columns,const RMSNormExecOptions&opts)
{
    if constexpr(F==Format::FP32)rmsnorm_fused_vectorized(input,gamma,output,rows,columns,1e-5f,opts);
    else if constexpr(F==Format::FP16)rmsnorm_fused_fp16_vectorized(input,gamma,output,rows,columns,1e-5f,opts);
    else rmsnorm_fused_bf16_vectorized(input,gamma,output,rows,columns,1e-5f,opts);
}
/** @brief Sweep actual matrices against independent formulas and actual M=1 calls.
 * @tparam F Original format; @param workers Explicit retained team size, or zero for standalone.
 * @param t5 Whether to exercise the distinct FP32 accumulation program.
 */
template<Format F>void sweep(std::size_t workers,bool t5=false)
{
    for(std::size_t columns:{1u,3u,4u,5u,7u,8u,9u,17u,31u,32u,33u,65u,129u,257u})
        for(std::size_t rows:{1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u,12u,13u,14u,15u,16u,31u,33u})
        {
            std::vector<Word<F>> input(rows*columns+23),actual(input.size(),encode<F>(-19.0f)),expected=actual;
            std::vector<float> gamma(columns);
            for(std::size_t c=0;c<columns;++c)
                gamma[c]=c%7==0?0x1p-20f:1.0f+float(int(c%31)-15)*0x1p-12f;
            for(std::size_t i=0;i<input.size();++i)input[i]=encode<F>(float(int((i*17+101)%257)-128)*0.03125f);
            const auto before=input;
            RMSNormExecOptions opts{.allow_parallel=false,.t5_compat_mode=t5};
            for(std::size_t row=0;row<rows;++row)
                invoke<F>(input.data()+5+row*columns,gamma.data(),expected.data()+5+row*columns,1,columns,opts);
            if(workers)
            {
#pragma omp parallel num_threads(workers)
                {invoke<F>(input.data()+5,gamma.data(),actual.data()+5,rows,columns,opts);}
            }
            else invoke<F>(input.data()+5,gamma.data(),actual.data()+5,rows,columns,opts);
            EXPECT_EQ(actual,expected)<<"format="<<int(F)<<" M="<<rows<<" K="<<columns<<" workers="<<workers;
            EXPECT_EQ(input,before);
#if defined(__aarch64__) && defined(__ARM_NEON)
            for(std::size_t row=0;row<rows;++row)
            {
                double sum=0.0;float t5_sum=0.0f;
                for(std::size_t c=0;c<columns;++c)
                {
                    const auto value=decode<F>(input[5+row*columns+c]);
                    sum=std::fma(double(value),double(value),sum);
                    t5_sum=std::fma(value,value,t5_sum);
                }
                if(t5)sum=double(t5_sum);
                float inverse;
                if constexpr(F==Format::FP32)inverse=float(1.0/std::sqrt(sum/double(columns)+double(1e-5f)));
                else inverse=1.0f/std::sqrt(float(sum/double(columns))+1e-5f);
                for(std::size_t c=0;c<columns;++c)
                {
                    const auto value=(decode<F>(input[5+row*columns+c])*inverse)*gamma[c];
                    const auto golden=encode<F>(value);
                    const auto got=actual[5+row*columns+c];
                    if constexpr(F==Format::FP32)EXPECT_EQ(std::bit_cast<std::uint32_t>(got),std::bit_cast<std::uint32_t>(golden));
                    else EXPECT_EQ(got,golden);
                }
            }
#endif
        }
}
} // namespace

/** @test Original floating matrices match actual M=1 bytes, metadata and independent native formulas. */
TEST(CPURMSNormWorkshare,FloatingMatricesPreserveOriginalRowsAndSerialBytes)
{
    for(std::size_t workers:{0u,3u})
    {
        sweep<Format::FP32>(workers);sweep<Format::FP16>(workers);sweep<Format::BF16>(workers);
        sweep<Format::FP32>(workers,true);
    }
}

/** @test Every positive worker count through10 retains actual M=1 publication bytes. */
TEST(CPURMSNormWorkshare,AllPositiveWorkersThroughTenRetainSerialBytes)
{
    for(std::size_t workers=1;workers<=10;++workers)
    {
        sweep<Format::FP32>(workers);sweep<Format::FP16>(workers);sweep<Format::BF16>(workers);
        sweep<Format::FP32>(workers,true);
    }
}

/** @test Complete original banks and immutable gamma/options reject before publication. */
TEST(CPURMSNormWorkshare,OriginalBankAndPolicyAliasesRejectBeforeMutation)
{
    std::array<float,17> input,output,gamma;
    input.fill(1.0f);output.fill(-19.0f);gamma.fill(1.0f);
    const auto before=input,untouched=output;
    EXPECT_THROW(rmsnorm_fused_vectorized(input.data(),gamma.data(),input.data()+1,1,16,1e-5f),std::invalid_argument);
    EXPECT_EQ(input,before);
    EXPECT_THROW(rmsnorm_fused_vectorized(input.data(),output.data(),output.data(),1,17,1e-5f),std::invalid_argument);
    EXPECT_THROW(rmsnorm_fused_vectorized(input.data(),gamma.data(),output.data(),1,17,-1.0f),std::invalid_argument);
    EXPECT_THROW(rmsnorm_fused_vectorized(input.data(),gamma.data(),output.data(),std::size_t(-1),17,1e-5f),std::invalid_argument);
    EXPECT_EQ(output,untouched);
}
/** @test Half normal ties use canonical RNE in every bulk/tail lane, including explicit scalar intent. */
TEST(CPURMSNormWorkshare,PortableHalfNormalRoundingIsLaneIndependent)
{
    for(std::size_t columns:{1u,7u,8u,9u,15u,16u,17u,31u,32u,33u})
        for(bool scalar:{false,true})
        {
            std::vector<std::uint16_t> input(columns,0x3c00),output(columns,0xa55a);
            std::vector<float> gamma(columns,1.0f+3.0f*0x1p-12f);
            RMSNormExecOptions opts{.allow_parallel=false,.force_scalar=scalar};
            rmsnorm_fused_fp16_vectorized(input.data(),gamma.data(),output.data(),1,columns,1e-5f,opts);
            const auto expected=fp32_to_fp16((1.0f/std::sqrt(1.0f+1e-5f))*gamma[0]);
            ASSERT_EQ(expected,0x3c01);
            for(auto word:output)EXPECT_EQ(word,expected)<<"K="<<columns<<" scalar="<<scalar;
        }
}

#if defined(__aarch64__) && defined(__ARM_NEON)
/** @test Staged and per-row inverse programs remain distinct, including a negative zero publication. */
TEST(CPURMSNormWorkshare,NativePerRowAndStagedInverseProgramsRemainDistinct)
{
    const float input=-std::numeric_limits<float>::max(),gamma=1.0f;
    float per_row=19.0f,staged=19.0f;
    rmsnorm_fused_row_neon(&input,&gamma,&per_row,1,1e-5f);
    rmsnorm_fused_vectorized(&input,&gamma,&staged,1,1,1e-5f);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(per_row),0x80000000u);
    EXPECT_LT(staged,-0.99f);
}
#endif
} // namespace llaminar2::test
