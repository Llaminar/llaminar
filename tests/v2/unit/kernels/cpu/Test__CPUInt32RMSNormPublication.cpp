/**
 * @file Test__CPUInt32RMSNormPublication.cpp
 * @brief Actual-TU original-INT32 row quantization and fused-owner regressions.
 *
 * Independent pointwise FP32 products, declared vector-RNE/scalar-away rounding,
 * original dynamic row scales, poisoned writers, bank exclusions and retained
 * completion are tested without decoding a row in production. Native M1 is the
 * grouped oracle for each ISA's distinct FP64 summary. This is not a model,
 * factory, PMA, NUMA-placement or performance qualification.
 */
#include "kernels/cpu/primitives/RMSNormPrimitives.h"
#include "utils/CPUFeatures.h"
#include <gtest/gtest.h>
#include <algorithm>
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
/** @brief Independently filtered numeric receipts must contain a genuine native program.
 * @throws std::runtime_error For scalar/foreign/empty native admission.
 */
void requireNativeInt32Publication()
{
    const auto active=activeISALevel();
    if(active==ISALevel::Scalar||!isNativeCPUExecutionISAPair(compiledCPUISALevel(),active))
        throw std::runtime_error("INT32 RMS publication conformance requires a genuine native program");
}
/** @brief Observe an independent FP32 storage edge, not production rounding helpers.
 * @param value Original pointwise result; @return Rounded binary32 value.
 */
float rounded(float value){volatile float stored=value;return stored;}
/** @brief Independently reproduce signed nearest-even without changing the FP environment.
 * @param value Bounded finite scaled value; @return Exact nearest-even integer.
 */
int nearestEven(float value)
{
    const double magnitude=std::fabs(double(value));const int base=int(std::floor(magnitude));
    const double remainder=magnitude-double(base);
    const int result=base+int(remainder>0.5||(remainder==0.5&&(base%2)!=0));
    return value<0?-result:result;
}
/** @brief A complete independently authored signed row and FP32 scale. */
struct Reference{std::vector<std::int8_t>codes;float scale;};
/** @brief Independently spell the original row-INT8 program, not block-Q8 quantization.
 * @param source Original integers; @param gamma Original optional weights.
 * @param inverse Original inverse; @param scalar Explicit diagnostic program.
 * @return Complete reference payload and scale, including vector/tail tie distinction.
 */
Reference reference(std::span<const std::int32_t>source,const float*gamma,float inverse,bool scalar=false)
{
    Reference result{std::vector<std::int8_t>(source.size()),1};
    if(inverse==0)return result;
    std::vector<float>values(source.size());float maximum=0;
    for(std::size_t c=0;c<source.size();++c)
    {
        values[c]=rounded(float(source[c])*inverse);
        if(gamma)values[c]=rounded(values[c]*gamma[c]);
        if(!std::isfinite(values[c]))throw std::runtime_error("Invalid independent INT32 RMS fixture");
        maximum=std::max(maximum,std::fabs(values[c]));
    }
    result.scale=maximum>0?maximum/127.0f:1.0f;const auto reciprocal=1.0f/result.scale;
    const auto width=activeISALevel()==ISALevel::AVX512?16u:8u;
    const auto complete=scalar?0:(source.size()/width)*width;
    for(std::size_t c=0;c<source.size();++c)
    {
        const auto value=std::clamp(rounded(values[c]*reciprocal),-127.0f,127.0f);
        result.codes[c]=std::int8_t(c<complete?nearestEven(value):int(std::round(value)));
    }
    return result;
}
/** @brief Poison every payload/scale writer relative to its independent expected result.
 * @param codes Complete output; @param scale Matching output scale; @param expected Complete reference.
 */
void poison(std::span<std::int8_t>codes,float&scale,const Reference&expected)
{
    for(std::size_t c=0;c<codes.size();++c)codes[c]=std::bit_cast<std::int8_t>(std::uint8_t(std::bit_cast<std::uint8_t>(expected.codes[c])^0x80u));
    scale=std::bit_cast<float>(~std::bit_cast<std::uint32_t>(expected.scale));
}
/** @brief Generate original full-width integer words without a floating conversion.
 * @param count Complete original geometry; @return Deterministic signed fixture.
 */
std::vector<std::int32_t>originals(std::size_t count)
{
    std::vector<std::int32_t>values(count);std::uint32_t state=13;
    for(auto &value:values){state=state*1664525u+1013904223u;value=std::bit_cast<std::int32_t>(state);}
    return values;
}
/** @brief Keep actual team size explicit without leaking test runtime changes. */
struct TeamPolicy
{
    int dynamic=omp_get_dynamic();
    /** @brief Request the explicitly exercised positive team sizes. */
    TeamPolicy(){omp_set_dynamic(0);}
    /** @brief Restore the incoming OpenMP dynamic-team policy. */
    ~TeamPolicy(){omp_set_dynamic(dynamic);}
};
} // namespace

TEST(CPUInt32RMSNormPublication,OriginalNativeRNEFragmentsAndAwayTails)
{
    requireNativeInt32Publication();
    std::array<std::int32_t,17>source;source.fill(1);
    std::array<float,17>gamma;gamma.fill(0.5f);gamma[0]=127;gamma[7]=-0.5f;gamma[15]=-0.5f;
    for(bool scalar:{false,true})
    {
        const auto expected=reference(source,gamma.data(),1,scalar);std::array<std::int8_t,17>output;float scale;
        poison(output,scale,expected);const float inverse=1;
        rmsnorm_apply_int32_to_int8_vectorized(source.data(),gamma.data(),&inverse,1,17,output.data(),&scale,
            {.allow_parallel=false,.force_scalar=scalar});
        EXPECT_TRUE(std::equal(output.begin(),output.end(),expected.codes.begin()));
        EXPECT_EQ(std::bit_cast<std::uint32_t>(scale),std::bit_cast<std::uint32_t>(expected.scale));
        EXPECT_EQ(output[1],scalar?1:0);EXPECT_EQ(output[16],1);
    }
}

TEST(CPUInt32RMSNormPublication,ExplicitScalarFusedUsesSerialSummaryAndOriginalAwayPublication)
{
    requireNativeInt32Publication();
    for(std::size_t columns:{1u,7u,8u,9u,16u,17u,127u,257u})
    {
        const auto source=originals(columns);std::vector<float>gamma(columns,0.625f);
        double sum=0;for(auto word:source){const auto value=double(word);sum=std::fma(value,value,sum);}
        const float inverse=float(1.0/std::sqrt(sum/double(columns)+double(1e-5f)));
        const auto expected=reference(source,gamma.data(),inverse,true);
        std::vector<std::int8_t>output(columns);float scale;poison(output,scale,expected);
        rmsnorm_fused_int32_to_int8_vectorized(source.data(),gamma.data(),output.data(),&scale,1,columns,1e-5f,
            {.allow_parallel=false,.force_scalar=true});
        EXPECT_EQ(output,expected.codes);EXPECT_EQ(std::bit_cast<std::uint32_t>(scale),std::bit_cast<std::uint32_t>(expected.scale));
    }
}

TEST(CPUInt32RMSNormPublication,OriginalScaleHasNoForeignFloorAndZeroInverseDoesNotReadGamma)
{
    requireNativeInt32Publication();
    std::array<std::int32_t,17>source;source.fill(1);std::array<float,17>gamma;gamma.fill(1e-12f);
    std::array<std::int8_t,17>output;float scale;float inverse=1;
    auto expected=reference(source,gamma.data(),inverse);poison(output,scale,expected);
    rmsnorm_apply_int32_to_int8_vectorized(source.data(),gamma.data(),&inverse,1,17,output.data(),&scale,{.allow_parallel=false});
    EXPECT_LT(scale,1e-8f);EXPECT_EQ(std::bit_cast<std::uint32_t>(scale),std::bit_cast<std::uint32_t>(expected.scale));
    EXPECT_TRUE(std::equal(output.begin(),output.end(),expected.codes.begin()));
    inverse=0;gamma.fill(std::numeric_limits<float>::quiet_NaN());output.fill(41);scale=-19;
    rmsnorm_apply_int32_to_int8_vectorized(source.data(),gamma.data(),&inverse,1,17,output.data(),&scale,{.allow_parallel=false});
    EXPECT_EQ(scale,1);for(auto value:output)EXPECT_EQ(value,0);
}

TEST(CPUInt32RMSNormPublication,CompleteOriginalBanksAndPolicyRejectBeforePublication)
{
    alignas(float) std::array<std::int32_t,32>source;source.fill(11);const auto sourceBefore=source;
    std::array<float,32>gamma;gamma.fill(1);const auto gammaBefore=gamma;
    std::array<float,2>inverse{0.25f,0.5f};const auto inverseBefore=inverse;
    alignas(float) std::array<std::int8_t,64>output;output.fill(39);const auto outputBefore=output;
    std::array<float,2>scales{-31,-31};const auto scalesBefore=scales;
    const auto reject=[&](std::int8_t*codes,float*metadata)
    {
        EXPECT_THROW(rmsnorm_apply_int32_to_int8_vectorized(source.data(),gamma.data(),inverse.data(),2,8,codes,metadata,
            {.allow_parallel=false}),std::invalid_argument);
    };
    reject(reinterpret_cast<std::int8_t*>(source.data()+8),scales.data());
    reject(output.data(),reinterpret_cast<float*>(source.data()+8));
    reject(reinterpret_cast<std::int8_t*>(gamma.data()),scales.data());reject(output.data(),gamma.data());
    reject(reinterpret_cast<std::int8_t*>(inverse.data()),scales.data());reject(output.data(),inverse.data());
    reject(output.data(),reinterpret_cast<float*>(output.data()+4));
    RMSNormExecOptions options{.allow_parallel=false,.parallel_threshold_elems=31};
    EXPECT_THROW(rmsnorm_apply_int32_to_int8_vectorized(source.data(),gamma.data(),inverse.data(),1,8,
        reinterpret_cast<std::int8_t*>(&options),scales.data(),options),std::invalid_argument);
    EXPECT_THROW(rmsnorm_fused_int32_to_int8_vectorized(source.data(),gamma.data(),output.data(),
        reinterpret_cast<float*>(&options.parallel_threshold_elems),1,8,1e-5f,options),std::invalid_argument);
    EXPECT_EQ(options.parallel_threshold_elems,31u);EXPECT_FALSE(options.allow_parallel);EXPECT_FALSE(options.force_scalar);
    EXPECT_THROW(rmsnorm_fused_int32_to_int8_vectorized(source.data(),gamma.data(),output.data(),scales.data(),2,8,-1,
        {.allow_parallel=false}),std::invalid_argument);
    EXPECT_THROW(rmsnorm_fused_int32_to_int8_vectorized(source.data(),gamma.data(),output.data(),scales.data(),2,8,0,
        {.allow_parallel=false,.t5_compat_mode=true}),std::invalid_argument);
    EXPECT_THROW(rmsnorm_apply_int32_to_int8_vectorized(source.data(),gamma.data(),nullptr,2,8,output.data(),scales.data()),std::invalid_argument);
    EXPECT_THROW(rmsnorm_fused_int32_to_int8_vectorized(source.data(),gamma.data(),output.data(),scales.data(),
        std::numeric_limits<std::size_t>::max(),8,0),std::invalid_argument);
    EXPECT_EQ(source,sourceBefore);EXPECT_EQ(gamma,gammaBefore);EXPECT_EQ(inverse,inverseBefore);
    EXPECT_EQ(output,outputBefore);EXPECT_EQ(scales,scalesBefore);
}

TEST(CPUInt32RMSNormPublication,EmptyStagedAndFusedCallsCompleteRetainedTeam)
{
    TeamPolicy policy;
    for(bool fused:{false,true})for(const auto geometry:{std::array<std::size_t,2>{0,0},{0,17},{3,0}})
    {
        std::atomic<int>arrived{0},early{0};
#pragma omp parallel num_threads(3)
        {
            if(omp_get_thread_num()==2)std::this_thread::sleep_for(std::chrono::milliseconds(10));
            arrived.fetch_add(1,std::memory_order_release);
            if(fused)rmsnorm_fused_int32_to_int8_vectorized(nullptr,nullptr,nullptr,nullptr,geometry[0],geometry[1],0,{.allow_parallel=false});
            else rmsnorm_apply_int32_to_int8_vectorized(nullptr,nullptr,nullptr,geometry[0],geometry[1],nullptr,nullptr,{.allow_parallel=false});
            if(arrived.load(std::memory_order_acquire)!=3)early.fetch_add(1,std::memory_order_relaxed);
        }
        EXPECT_EQ(early.load(),0);
    }
}

TEST(CPUInt32RMSNormPublication,NonemptyStagedAndFusedCompleteBeforeImmediateReuse)
{
    requireNativeInt32Publication();TeamPolicy policy;
    std::array<std::int32_t,17>source;source.fill(1);std::array<float,17>gamma;gamma.fill(1);
    const float inverse=0.03125f;
    for(const bool fused:{false,true})
    {
        std::array<std::int8_t,17>output;output.fill(43);float scale=-19;
        std::atomic<int>arrived{0},observed{0};int actualWorkers=0;
#pragma omp parallel num_threads(3)
        {
            if(omp_get_thread_num()==2)std::this_thread::sleep_for(std::chrono::milliseconds(10));
            arrived.fetch_add(1,std::memory_order_release);
            if(fused)rmsnorm_fused_int32_to_int8_vectorized(source.data(),gamma.data(),output.data(),&scale,1,17,1e-5f,{.allow_parallel=false});
            else rmsnorm_apply_int32_to_int8_vectorized(source.data(),gamma.data(),&inverse,1,17,output.data(),&scale,{.allow_parallel=false});
            if(omp_get_thread_num()==0)
            {
                actualWorkers=omp_get_num_threads();observed.store(arrived.load(std::memory_order_acquire));
                // One row/tile belongs only to participant0, so even a nowait
                // countercontrol has no competing payload writer during reuse.
                output.fill(77);scale=-9;
            }
        }
        EXPECT_EQ(actualWorkers,3);EXPECT_EQ(observed.load(),3)<<(fused?"fused":"staged");
        for(auto code:output)EXPECT_EQ(code,77);EXPECT_EQ(scale,-9);
    }
}

TEST(CPUInt32RMSNormPublication,NonfiniteAndUnrepresentableRowsAreFatalNotZeroed)
{
    requireNativeInt32Publication();
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    std::array<std::int32_t,33>source;source.fill(1);std::array<float,33>gamma;gamma.fill(1);
    std::array<std::int8_t,33>output;float scale;float inverse=std::numeric_limits<float>::quiet_NaN();
    EXPECT_DEATH(rmsnorm_apply_int32_to_int8_vectorized(source.data(),gamma.data(),&inverse,1,33,output.data(),&scale,{.allow_parallel=false}),"INT32 RMS row publication");
    inverse=1;gamma[1]=std::numeric_limits<float>::infinity();
    EXPECT_DEATH(rmsnorm_apply_int32_to_int8_vectorized(source.data(),gamma.data(),&inverse,1,33,output.data(),&scale,{.allow_parallel=false}),"INT32 RMS row publication");
    // NaN is inside the first complete native8/16 fragment; later complete
    // finite fragments must not hide it through x86 maximum instructions.
    gamma.fill(1);gamma[1]=std::numeric_limits<float>::quiet_NaN();
    EXPECT_DEATH(rmsnorm_apply_int32_to_int8_vectorized(source.data(),gamma.data(),&inverse,1,33,output.data(),&scale,{.allow_parallel=false}),"INT32 RMS row publication");
    gamma.fill(1);gamma.back()=std::numeric_limits<float>::quiet_NaN();
    EXPECT_DEATH(rmsnorm_apply_int32_to_int8_vectorized(source.data(),gamma.data(),&inverse,1,33,output.data(),&scale,{.allow_parallel=false}),"INT32 RMS row publication");
    gamma.fill(std::numeric_limits<float>::denorm_min());
    EXPECT_DEATH(rmsnorm_apply_int32_to_int8_vectorized(source.data(),gamma.data(),&inverse,1,33,output.data(),&scale,{.allow_parallel=false}),"INT32 RMS row publication");
}

TEST(CPUInt32RMSNormPublication,AllPositiveWorkersThroughTenRetainNativeStagedFusedBytesAndGuards)
{
    requireNativeInt32Publication();TeamPolicy policy;
    for(std::size_t columns:{1u,3u,4u,7u,8u,9u,15u,16u,17u,31u,32u,33u,63u,64u,65u,127u,128u,129u,255u,256u,257u,1024u,4096u,8192u})
        for(std::size_t rows:{1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u,12u,13u,14u,15u,16u,31u,33u})
        {
            const auto source=originals(rows*columns);const auto before=source;
            std::vector<float>gamma(columns);for(std::size_t c=0;c<columns;++c)gamma[c]=float(int(c%17)-8)/7.0f;
            const auto gammaBefore=gamma;
            for(bool withGamma:{false,true})
            {
                const float*weights=withGamma?gamma.data():nullptr;
                std::vector<float>inverses(rows),expectedScales(rows);std::vector<std::int8_t>expected(rows*columns);
                for(std::size_t row=0;row<rows;++row)
                {
                    double summary=std::numeric_limits<double>::quiet_NaN();
                    rmsnorm_compute_row_sumsq_int32_vectorized(source.data()+row*columns,1,columns,&summary,{.allow_parallel=false});
                    ASSERT_TRUE(std::isfinite(summary));
                    inverses[row]=float(1.0/std::sqrt(summary/double(columns)+double(1e-5f)));
                    const auto independent=reference(std::span(source).subspan(row*columns,columns),weights,inverses[row]);
                    auto codes=std::span(expected).subspan(row*columns,columns);poison(codes,expectedScales[row],independent);
                    rmsnorm_fused_int32_to_int8_vectorized(source.data()+row*columns,weights,codes.data(),expectedScales.data()+row,1,columns,1e-5f,{.allow_parallel=false});
                    ASSERT_TRUE(std::equal(codes.begin(),codes.end(),independent.codes.begin()));
                    ASSERT_EQ(std::bit_cast<std::uint32_t>(expectedScales[row]),std::bit_cast<std::uint32_t>(independent.scale));
                }
                const auto invBefore=inverses;
                for(int workers=1;workers<=10;++workers)for(bool fused:{false,true})
                {
                    constexpr std::size_t prefix=7,suffix=9;
                    std::vector<std::int8_t>actual(prefix+expected.size()+suffix,53);
                    std::vector<float>scales(prefix+rows+suffix,-17);
                    for(std::size_t row=0;row<rows;++row)
                        poison(std::span(actual).subspan(prefix+row*columns,columns),scales[prefix+row],
                            Reference{std::vector<std::int8_t>(expected.begin()+row*columns,expected.begin()+(row+1)*columns),expectedScales[row]});
                    int actualWorkers=0;
#pragma omp parallel num_threads(workers)
                    {
#pragma omp single
                        actualWorkers=omp_get_num_threads();
                        if(fused)rmsnorm_fused_int32_to_int8_vectorized(source.data(),weights,actual.data()+prefix,scales.data()+prefix,rows,columns,1e-5f,{.allow_parallel=false});
                        else rmsnorm_apply_int32_to_int8_vectorized(source.data(),weights,inverses.data(),rows,columns,actual.data()+prefix,scales.data()+prefix,{.allow_parallel=false});
                    }
                    ASSERT_EQ(actualWorkers,workers);
                    ASSERT_TRUE(std::equal(expected.begin(),expected.end(),actual.begin()+prefix));
                    for(std::size_t row=0;row<rows;++row)ASSERT_EQ(std::bit_cast<std::uint32_t>(scales[prefix+row]),std::bit_cast<std::uint32_t>(expectedScales[row]));
                    for(std::size_t c=0;c<prefix;++c){EXPECT_EQ(actual[c],53);EXPECT_EQ(scales[c],-17);}
                    for(std::size_t c=prefix+expected.size();c<actual.size();++c)EXPECT_EQ(actual[c],53);
                    for(std::size_t c=prefix+rows;c<scales.size();++c)EXPECT_EQ(scales[c],-17);
                }
                EXPECT_EQ(inverses,invBefore);
            }
            EXPECT_EQ(source,before);EXPECT_EQ(gamma,gammaBefore);
        }
}
} // namespace llaminar2::test
