/**
 * @file Test__CPUSoftmax.cpp
 * @brief Production CPU softmax storage, native arithmetic and retained-team regressions.
 *
 * Independent references may materialize diagnostic exponentials; production
 * APIs must stream original storage. Native exact references use observable
 * scalar rounding edges and a separate double-exp accuracy oracle. Same-backend
 * public serial rows, not a diagnostic dispatch override, certify grouped bytes.
 */
#include "kernels/cpu/primitives/SoftmaxPrimitives_New.h"
#include "utils/CPUExplicitRoundingOracle.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <span>
#include <thread>
#include <type_traits>
#include <vector>

namespace llaminar2::test
{
namespace
{
using primitives::softmax_detail::Storage;
using primitives::softmax_detail::Unit;
static_assert(!std::is_default_constructible_v<primitives::softmax_detail::Row<Storage::FP32>>);
static_assert(!std::is_default_constructible_v<primitives::softmax_detail::Batch<Storage::Q8_1>>);

/** @brief Call the actual public native-dispatched row API.
 * @tparam Format Original storage; @param row Original bank; @param units Physical count.
 * @param causal Original prefix policy; @param scale Finite multiplier; @param index Original coordinate.
 */
template<Storage Format> void one(Unit<Format>*row,int units,bool causal,float scale,int index)
{
    if constexpr(Format==Storage::FP32)primitives::softmax_row_fp32(row,units,causal,scale,index);
    else if constexpr(Format==Storage::BFloat16)primitives::softmax_row_bf16(row,units,causal,scale,index);
    else if constexpr(Format==Storage::Half)primitives::softmax_row_fp16(row,units,causal,scale,index);
    else primitives::softmax_row_q8_1(row,units,causal,scale,index);
}
/** @brief Call the actual public collective row-major API.
 * @tparam Format Original storage; @param row Whole original bank; @param rows Original rows.
 * @param units Physical units per row; @param causal Prefix policy; @param scale Finite multiplier.
 * @param parallel Standalone team intent, not permission to split an active team.
 */
template<Storage Format> void many(Unit<Format>*row,int rows,int units,bool causal,float scale,bool parallel)
{
    if constexpr(Format==Storage::FP32)primitives::softmax_row_major_fp32(row,rows,units,causal,scale,parallel);
    else if constexpr(Format==Storage::BFloat16)primitives::softmax_row_major_bf16(row,rows,units,causal,scale,parallel);
    else if constexpr(Format==Storage::Half)primitives::softmax_row_major_fp16(row,rows,units,causal,scale,parallel);
    else primitives::softmax_row_major_q8_1(row,rows,units,causal,scale,parallel);
}
/** @brief Independently narrow ordinary BF16 words with ties-even.
 * @param value Original FP32 value; @return Canonical BF16 bits.
 */
std::uint16_t bf16(float value)
{const auto bits=std::bit_cast<std::uint32_t>(value);return std::uint16_t((bits+0x7fffu+((bits>>16)&1))>>16);}
/** @brief Prepare original physical storage before invocation, retaining trailing poison.
 * @tparam Format Original storage; @param units Physical count; @param seed Diagnostic identity.
 * @return Original bank plus two untouched physical canaries.
 */
template<Storage Format>std::vector<Unit<Format>> source(int units,int seed)
{
    std::vector<Unit<Format>> result(std::size_t(units)+2);
    std::memset(result.data(),0x5d,result.size()*sizeof(Unit<Format>));
    for(int i=0;i<units;++i)
    {
        if constexpr(Format==Storage::Q8_1)
        {
            auto&block=result[i];block.d=fp32_to_fp16(float((i+seed)%7+1)*0.01731f*((i+seed)%3==0?-1.0f:1.0f));
            block.sum_qs=0;
            for(int lane=0;lane<32;++lane){block.qs[lane]=std::int8_t((i*131+lane*37+seed*17)%256-128);block.sum_qs+=block.qs[lane];}
        }
        else
        {
            const float value=float((i*131+seed*17)%257-128)*0.13731f;
            if constexpr(Format==Storage::FP32)result[i]=value;
            else if constexpr(Format==Storage::Half)result[i]=fp32_to_fp16(value);
            else result[i]=bf16(value);
        }
    }
    return result;
}
/** @brief Read the authoritative declared source storage, independently of production readers.
 * @tparam Format Original physical storage; @param row Original bank; @param col Original coordinate.
 * @param scale Original multiplier; @return Separately rounded score.
 */
template<Storage Format>float score(const Unit<Format>*row,std::size_t col,float scale)
{
    if constexpr(Format==Storage::Q8_1)
    {const auto&block=row[col/32];return memoryRoundedMultiply(float(block.qs[col%32]),memoryRoundedMultiply(fp16_to_fp32(block.d),scale));}
    else if constexpr(Format==Storage::FP32)return memoryRoundedMultiply(row[col],scale);
    else if constexpr(Format==Storage::Half)return memoryRoundedMultiply(fp16_to_fp32(row[col]),scale);
    else return memoryRoundedMultiply(std::bit_cast<float>(std::uint32_t(row[col])<<16),scale);
}
/** @brief Compare complete output bytes, including retained poisoned canaries.
 * @tparam T Original storage unit; @param actual Production bank; @param expected Diagnostic bank.
 */
template<class T>void same(const std::vector<T>&actual,const std::vector<T>&expected)
{ASSERT_EQ(actual.size(),expected.size());EXPECT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(T)),0);}

#if LLAMINAR_COMPILED_WITH_NEON
/** @brief Independently retain each native exponential arithmetic edge through memory.
 * @param value Original nonpositive shifted score; @return Exact native-program reference word.
 */
float referenceExp(float value)
{
    if(value<=-104.0f)return 0.0f;
    const float scaled=memoryRoundedMultiply(value,0x1.715476p+0f);
    const int exponent=int(std::nearbyint(scaled));
    volatile float first=std::fma(-float(exponent),0x1.62e400p-1f,value);
    volatile float remainder=std::fma(-float(exponent),0x1.7f7d1cp-20f,float(first));
    constexpr float coefficients[]{0x1.a01a02p-16f,0x1.a01a02p-13f,0x1.6c16c2p-10f,0x1.111112p-7f,
        0x1.555556p-5f,0x1.555556p-3f,0x1.000000p-1f,1.0f,1.0f};
    float polynomial=coefficients[0];
    for(std::size_t i=1;i<std::size(coefficients);++i)
    {volatile float step=std::fma(polynomial,float(remainder),coefficients[i]);polynomial=step;}
    const int normal_exponent=std::max(exponent,-126);
    const auto normal=std::bit_cast<float>(std::uint32_t(normal_exponent+127)<<23);
    const auto residual=std::bit_cast<float>(std::uint32_t(exponent-normal_exponent+127)<<23);
    return memoryRoundedMultiply(memoryRoundedMultiply(polynomial,residual),normal);
}
/** @brief Diagnostic original-row reference with fixed coordinate-modulo-four double chains.
 * @tparam Format Original physical storage; @param original Original bank plus canaries.
 * @param units Physical count; @param causal Original policy; @param scale Finite multiplier.
 * @param index Original causal coordinate; @return Complete independently authored expected bank.
 */
template<Storage Format>std::vector<Unit<Format>> reference(const std::vector<Unit<Format>>&original,int units,bool causal,float scale,int index)
{
    auto result=original;
    const std::size_t columns=std::size_t(units)*(Format==Storage::Q8_1?32:1);
    const auto live=causal?std::min(columns,std::size_t(std::max<std::int64_t>(0,std::int64_t(index)+1))):columns;
    float peak=-std::numeric_limits<float>::infinity();
    for(std::size_t i=0;i<live;++i)peak=std::max(peak,score<Format>(original.data(),i,scale));
    std::vector<float> exponential(live);
    double chains[4]{};
    for(std::size_t i=0;i<live;++i)
    {
        exponential[i]=referenceExp(memoryRoundedAdd(score<Format>(original.data(),i,scale),-peak));
        volatile double next=chains[i%4]+double(exponential[i]);chains[i%4]=next;
    }
    volatile double lo=chains[0]+chains[1],hi=chains[2]+chains[3];
    volatile double sum=double(lo)+double(hi);
    volatile float inverse=live&&peak!=-std::numeric_limits<float>::infinity()?float(1.0/double(sum)):1.0f;
    if constexpr(Format==Storage::Q8_1)
    {
        volatile float d=std::max(float(inverse)/127.0f,0x1p-24f);
        volatile float inv_d=1.0f/float(d);
        for(int block=0;block<units;++block)
        {
            result[block].d=fp32_to_fp16(d);result[block].sum_qs=0;
            for(int lane=0;lane<32;++lane)
            {
                const auto col=std::size_t(block)*32+lane;
                const float probability=col<live&&peak!=-std::numeric_limits<float>::infinity()?memoryRoundedMultiply(exponential[col],inverse):0.0f;
                const int code=std::clamp(int(std::nearbyint(memoryRoundedMultiply(probability,inv_d))),0,127);
                result[block].qs[lane]=std::int8_t(code);result[block].sum_qs+=code;
            }
        }
    }
    else
        for(std::size_t col=0;col<columns;++col)
        {
            const float probability=col<live&&peak!=-std::numeric_limits<float>::infinity()?memoryRoundedMultiply(exponential[col],inverse):0.0f;
            if constexpr(Format==Storage::FP32)result[col]=probability;
            else if constexpr(Format==Storage::Half)result[col]=fp32_to_fp16(probability);
            else result[col]=bf16(probability);
        }
    return result;
}
/** @brief Sweep complete original row geometries without a production row-sized temporary.
 * @tparam Format Original physical storage.
 */
template<Storage Format>void nativeRows()
{
    std::vector<int> sizes;
    for(int n=0;n<=65;++n)sizes.push_back(n);
    for(int n:{127,128,129,257,1024})sizes.push_back(n);
    for(int units:sizes)
        for(int seed=0;seed<4;++seed)
            for(float scale:{0.0f,0.13f,1.0f,-0.75f})
                for(int index:{-1,0,1,2,15,16,31,32,std::numeric_limits<int>::max()})
                {
                    const auto original=source<Format>(units,seed);
                    auto actual=original;
                    const auto expected=reference<Format>(original,units,true,scale,index);
                    one<Format>(actual.data(),units,true,scale,index);same(actual,expected);
                }
    for(int units:{4097,32769})
    {
        const auto original=source<Format>(units,3);auto actual=original;
        const auto expected=reference<Format>(original,units,false,0.17f,0);
        one<Format>(actual.data(),units,false,0.17f,0);same(actual,expected);
    }
}
/** @test Exact native arithmetic remains the same complete original-row program at every fringe. */
TEST(NEONSoftmax, OrderedOriginalRowsAllStorage)
{nativeRows<Storage::FP32>();nativeRows<Storage::BFloat16>();nativeRows<Storage::Half>();nativeRows<Storage::Q8_1>();}

/** @test Native exponential accuracy retains actual subnormal probabilities, not a clipped ordinary path. */
TEST(NEONSoftmax, ExponentialAccuracyAndSubnormalProgram)
{
    std::vector<float> inputs{0.0f,-0.0f,-std::numeric_limits<float>::infinity(),-104.0f,-103.97208f,-90.0f,-87.0f};
    for(int i=0;i<=100000;++i)inputs.push_back(-104.0f*float(i)/100000.0f);
    for(int n=0;n<=150;++n)
    {
        const float center=-float(n)*0.6931471805599453f;
        inputs.push_back(center);inputs.push_back(std::nextafter(center,0.0f));inputs.push_back(std::nextafter(center,-std::numeric_limits<float>::infinity()));
    }
    while(inputs.size()%4)inputs.push_back(0.0f);
    for(std::size_t i=0;i<inputs.size();i+=4)
    {
        float actual[4];vst1q_f32(actual,primitives::neon_softmax::exponential(vld1q_f32(inputs.data()+i)));
        for(int lane=0;lane<4;++lane)
        {
            const float word=inputs[i+lane];EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[lane]),std::bit_cast<std::uint32_t>(referenceExp(word)));
            const float exact=float(std::exp(double(word)));
            EXPECT_LE(std::abs(double(actual[lane])-exact),double(exact)*2e-6+std::numeric_limits<float>::denorm_min());
            if(exact>0)EXPECT_GT(actual[lane],0.0f);
        }
    }
}

/** @test Invalid active numeric state cannot publish even an early partial output row. */
TEST(NEONSoftmax, ActiveNumericFailuresRejectBeforeMutation)
{
    for(int col=0;col<65;++col)
        for(float invalid:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()})
        {
            auto values=source<Storage::FP32>(65,1);values[col]=invalid;const auto original=values;
            EXPECT_THROW(one<Storage::FP32>(values.data(),65,false,1.0f,0),std::domain_error);same(values,original);
        }
    const auto invalid16=[]<Storage Format>(std::array<std::uint16_t,3> words){
        for(int col=0;col<65;++col)for(auto invalid:words)
        {
            auto values=source<Format>(65,1);values[col]=invalid;const auto original=values;
            EXPECT_THROW(one<Format>(values.data(),65,false,1.0f,0),std::domain_error);same(values,original);
        }
    };
    invalid16.template operator()<Storage::Half>({0x7c00,0x7e01,0x7d01});
    invalid16.template operator()<Storage::BFloat16>({0x7f80,0x7fc1,0x7f81});
    for(int block=0;block<9;++block)
        for(std::uint16_t invalid:{std::uint16_t(0x7c00),std::uint16_t(0xfc00),std::uint16_t(0x7e01)})
        {
            auto values=source<Storage::Q8_1>(9,1);values[block].d=invalid;const auto original=values;
            EXPECT_THROW(one<Storage::Q8_1>(values.data(),9,false,1.0f,0),std::domain_error);same(values,original);
        }
    auto masked=source<Storage::FP32>(65,1);
    std::fill(masked.begin(),masked.begin()+65,std::numeric_limits<float>::quiet_NaN());
    one<Storage::FP32>(masked.data(),65,true,1.0f,-1);
    for(int col=0;col<65;++col)EXPECT_EQ(masked[col],0.0f);
    auto negative_infinite=source<Storage::FP32>(65,1);std::fill(negative_infinite.begin(),negative_infinite.begin()+65,-std::numeric_limits<float>::infinity());
    one<Storage::FP32>(negative_infinite.data(),65,false,1.0f,0);
    for(int col=0;col<65;++col)EXPECT_EQ(negative_infinite[col],0.0f);
}
#endif

/** @test The actual compiled exponential polynomial contains the full constant and linear Horner stages. */
TEST(CPUSoftmax, ExponentialPolynomialLinearTermAndAccuracy)
{
    const auto check=[](const float*actual,const float*input,int count){
        for(int lane=0;lane<count;++lane)
        {
            const double exact=std::exp(double(input[lane]));
            EXPECT_LE(std::abs(double(actual[lane])-exact),exact*2e-6);
        }
    };
    for(int sample=0;sample<=10000;++sample)
    {
        float input[16],actual[16];
        for(int lane=0;lane<16;++lane)input[lane]=-80.0f*float(sample)/10000.0f-float(lane)*0.003f;
#if LLAMINAR_COMPILED_WITH_NEON
        vst1q_f32(actual,primitives::neon_softmax::exponential(vld1q_f32(input)));check(actual,input,4);
#endif
#if defined(__SSE2__)
        _mm_storeu_ps(actual,primitives::detail::exp128_ps_poly(_mm_loadu_ps(input)));check(actual,input,4);
#endif
#if defined(__AVX2__)
        _mm256_storeu_ps(actual,primitives::detail::exp256_ps_poly(_mm256_loadu_ps(input)));check(actual,input,8);
#endif
#if defined(__AVX512F__)
        _mm512_storeu_ps(actual,primitives::detail::exp512_ps_poly(_mm512_loadu_ps(input)));check(actual,input,16);
#endif
    }
}

/** @test All public routes use Q8_1Block's actual FP16 metadata, including -128 and signed coefficients. */
TEST(CPUSoftmax, Q8FP16MetadataAndSignedEndpoint)
{
    using Function=void(*)(Q8_1Block*,int,bool,float,int);
    const std::vector<Function> functions{
        primitives::softmax_row_q8_1_scalar,primitives::softmax_row_q8_1,
#if defined(__AVX2__)
        primitives::softmax_row_q8_1_avx2,
#endif
#if defined(__AVX512F__)
        primitives::softmax_row_q8_1_avx512,
#endif
    };
    for(auto function:functions)
        for(int code:{-128,-1,1,127})
            for(float original_scale:{-1.0f,1.0f})
                for(float multiplier:{-1.0f,1.0f})
                {
                    Q8_1Block block{};block.d=fp32_to_fp16(original_scale);block.sum_qs=std::int16_t(code*32);std::fill(std::begin(block.qs),std::end(block.qs),std::int8_t(code));
                    function(&block,1,false,multiplier,0);
                    EXPECT_EQ(block.d,fp32_to_fp16((1.0f/32.0f)/127.0f));EXPECT_EQ(block.sum_qs,32*127);
                    for(auto q:block.qs){EXPECT_EQ(q,127);EXPECT_NEAR(float(q)*fp16_to_fp32(block.d),1.0f/32.0f,1e-5f);}
                    function(&block,1,true,1.0f,-1);
                    EXPECT_EQ(block.d,fp32_to_fp16(1.0f/127.0f));EXPECT_EQ(block.sum_qs,0);
                    for(auto q:block.qs)EXPECT_EQ(q,0);
                }
    for(auto function:functions)for(float original_scale:{-1.0f,1.0f})
    {
        Q8_1Block block{};block.d=fp32_to_fp16(original_scale);
        std::fill(std::begin(block.qs),std::end(block.qs),std::int8_t(127));block.qs[0]=-128;
        block.sum_qs=std::int16_t(-128+31*127);
        function(&block,1,false,-original_scale,0);
        EXPECT_EQ(block.d,fp32_to_fp16(1.0f/127.0f));EXPECT_EQ(block.qs[0],127);EXPECT_EQ(block.sum_qs,127);
        for(int lane=1;lane<32;++lane)EXPECT_EQ(block.qs[lane],0);
    }
}

/** @test A representable long uniform distribution cannot disappear through FP16 scale underflow. */
TEST(CPUSoftmax, Q8FP16MinimumScaleRetainsLongUniformRows)
{
    using Function=void(*)(Q8_1Block*,int,bool,float,int);
    const std::vector<Function> functions{
        primitives::softmax_row_q8_1_scalar,primitives::softmax_row_q8_1,
#if defined(__AVX2__)
        primitives::softmax_row_q8_1_avx2,
#endif
#if defined(__AVX512F__)
        primitives::softmax_row_q8_1_avx512,
#endif
    };
    for(auto function:functions)
    {
        std::vector<Q8_1Block> bank(8192);
        for(auto&block:bank){block.d=fp32_to_fp16(1.0f);block.sum_qs=32;std::fill(std::begin(block.qs),std::end(block.qs),std::int8_t(1));}
        function(bank.data(),int(bank.size()),false,1.0f,0);
        for(const auto&block:bank)
        {
            EXPECT_EQ(block.d,1);EXPECT_EQ(block.sum_qs,32*64);
            for(auto code:block.qs)EXPECT_EQ(code,64);
        }
    }
    for(float invalid:{0.0f,-1.0f,2.0f,0x1p-26f,0x1p-25f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})
        EXPECT_THROW(primitives::softmax_detail::q8OutputScale(invalid),std::domain_error);
    EXPECT_EQ(primitives::softmax_detail::q8OutputScale(std::nextafter(0x1p-25f,1.0f)),0x1p-24f);
}

/** @brief Compare actual published probabilities against independently normalized double exponentials.
 * @tparam Format Original physical storage.
 */
template<Storage Format>void probabilityAccuracy()
{
    for(int units:{1,2,3,7,8,15,16,31,32,33,63,64,65,127,128,129,257,1025})
        for(float multiplier:{0.0f,0.13f,1.0f,-0.75f})for(int index:{-1,0,2,31,std::numeric_limits<int>::max()})
        {
            const auto original=source<Format>(units,11);auto actual=original;
            const auto columns=std::size_t(units)*(Format==Storage::Q8_1?32:1);
            const auto live=std::min(columns,std::size_t(std::max<std::int64_t>(0,std::int64_t(index)+1)));
            double peak=-std::numeric_limits<double>::infinity();
            for(std::size_t col=0;col<live;++col)peak=std::max(peak,double(score<Format>(original.data(),col,multiplier)));
            std::vector<double> expected(columns,0.0);double total=0.0;
            for(std::size_t col=0;col<live;++col){expected[col]=std::exp(double(score<Format>(original.data(),col,multiplier))-peak);total+=expected[col];}
            double maximum_probability=0.0;
            if(total)for(auto&probability:expected){probability/=total;maximum_probability=std::max(maximum_probability,probability);}
            one<Format>(actual.data(),units,true,multiplier,index);
            for(std::size_t col=0;col<columns;++col)
            {
                const double value=score<Format>(actual.data(),col,1.0f),exact=expected[col];
                double bound;
                if constexpr(Format==Storage::FP32)bound=3e-7+5e-6*exact;
                else if constexpr(Format==Storage::BFloat16)bound=1e-7+0.004*exact;
                else if constexpr(Format==Storage::Half)bound=1e-7+0.0005*exact;
                else bound=maximum_probability/254.0+0.001*maximum_probability+127.0*std::ldexp(1.0,-24)+1e-7;
                EXPECT_LE(std::abs(value-exact),bound);
                if(col>=live)EXPECT_EQ(value,0.0);
            }
        }
}
/** @test Every physical storage satisfies independent probability accuracy, not only its own arithmetic model. */
TEST(CPUSoftmax, ProbabilityAccuracyAgainstDoubleExponentials)
{
    probabilityAccuracy<Storage::FP32>();probabilityAccuracy<Storage::BFloat16>();
    probabilityAccuracy<Storage::Half>();probabilityAccuracy<Storage::Q8_1>();
}

/** @brief Reject malformed complete geometry before original banks can mutate.
 * @tparam Format Original physical storage.
 */
template<Storage Format>void geometry()
{
    auto data=source<Format>(65,1);const auto original=data;
    EXPECT_THROW(one<Format>(data.data(),-1,false,1.0f,0),std::invalid_argument);
    EXPECT_THROW(one<Format>(nullptr,1,false,1.0f,0),std::invalid_argument);
    EXPECT_THROW(one<Format>(data.data(),65,false,std::numeric_limits<float>::quiet_NaN(),0),std::invalid_argument);
    EXPECT_THROW(many<Format>(data.data(),-1,65,false,1.0f,false),std::invalid_argument);
    EXPECT_THROW(many<Format>(data.data(),1,-1,false,1.0f,false),std::invalid_argument);
    EXPECT_THROW(many<Format>(nullptr,1,1,false,1.0f,false),std::invalid_argument);
    if constexpr(sizeof(Unit<Format>)>=4)
    {
        EXPECT_THROW(many<Format>(data.data(),std::numeric_limits<int>::max(),std::numeric_limits<int>::max(),false,1.0f,false),std::invalid_argument);
    }
    auto*wrapping=reinterpret_cast<Unit<Format>*>(std::numeric_limits<std::uintptr_t>::max()-1);
    EXPECT_THROW(many<Format>(wrapping,2,1,false,1.0f,false),std::invalid_argument);
    EXPECT_NO_THROW(one<Format>(nullptr,0,false,1.0f,0));
    EXPECT_NO_THROW(many<Format>(nullptr,0,1,false,1.0f,false));
    EXPECT_NO_THROW(many<Format>(nullptr,1,0,false,1.0f,false));same(data,original);
}
/** @test Causal predicates retain original columns across INT_MAX without borrowing fictitious capacity. */
TEST(CPUSoftmax, OriginalColumnPredicateRetainsINTMAXBoundary)
{
    using primitives::softmax_detail::CausalPrefix;
    const auto first_past=std::size_t(std::numeric_limits<int>::max())+1;
    const auto total=first_past+32;
    for(int index:{std::numeric_limits<int>::min(),-1,0,1,std::numeric_limits<int>::max()})
    {
        const auto prefix=CausalPrefix::bind(total,true,index);
        for(auto col:{std::size_t(0),std::size_t(1),first_past-1,first_past,first_past+1,total-1,total})
        {
            const bool expected=std::int64_t(col)<=std::int64_t(index)&&col<total;
            EXPECT_EQ(prefix.contains(col),expected);
        }
    }
    EXPECT_FALSE(CausalPrefix::bind(total,true,-1).contains(first_past));
    EXPECT_TRUE(CausalPrefix::bind(total,true,std::numeric_limits<int>::max()).contains(first_past-1));
    EXPECT_FALSE(CausalPrefix::bind(total,true,std::numeric_limits<int>::max()).contains(first_past));
    EXPECT_TRUE(CausalPrefix::bind(total,false,-1).contains(first_past));
}
/** @test Valid FP16 banks/local probability fragments require natural alignment only. */
TEST(CPUSoftmax, UnalignedFP16OriginalAndProbabilityFragments)
{
    alignas(32)std::array<std::uint16_t,1024> bank;
    for(int units:{0,1,3,7,8,9,31,32,33,65,255,256,257})
    {
        auto expected=source<Storage::Half>(units,7);
        bank.fill(0x5d5d);std::copy_n(expected.data(),units,bank.data()+1);
        ASSERT_EQ(reinterpret_cast<std::uintptr_t>(bank.data()+1)%32,2);
        one<Storage::Half>(expected.data(),units,false,0.17f,0);
        one<Storage::Half>(bank.data()+1,units,false,0.17f,0);
        EXPECT_EQ(std::memcmp(bank.data()+1,expected.data(),std::size_t(units)*sizeof(std::uint16_t)),0);
        EXPECT_EQ(bank[0],0x5d5d);EXPECT_EQ(bank[std::size_t(units)+1],0x5d5d);EXPECT_EQ(bank[std::size_t(units)+2],0x5d5d);
    }
#if defined(__AVX2__)
    alignas(32)std::array<float,9> probabilities{0.0f,0.125f,0.25f,0.375f,0.5f,0.625f,0.75f,0.875f,1.0f};
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(probabilities.data()+1)%32,4);
    const auto actual=primitives::detail::loadFP16SoftmaxProbabilities(std::span<const float,8>(probabilities.data()+1,8));
    float stored[8];_mm256_storeu_ps(stored,actual);
    EXPECT_EQ(std::memcmp(stored,probabilities.data()+1,sizeof(stored)),0);
#endif
}

/** @test All physical storages share complete checked geometry, including empty banks. */
TEST(CPUSoftmax, OriginalGeometryRejectsBeforeMutation)
{geometry<Storage::FP32>();geometry<Storage::BFloat16>();geometry<Storage::Half>();geometry<Storage::Q8_1>();}

/** @brief Restore actual OpenMP caller policy after retained-team proofs. */
class TeamPolicy final
{
public:
    /** @brief Retain caller worker capacity and disable dynamic team shrinking. */
    TeamPolicy():threads_(omp_get_max_threads()),dynamic_(omp_get_dynamic()){omp_set_dynamic(0);}
    /** @brief Restore caller policy even on assertion exits. */
    ~TeamPolicy(){omp_set_num_threads(threads_);omp_set_dynamic(dynamic_);}
    /** @return Original actual positive worker budget. */
    int threads()const{return threads_;}
private:int threads_,dynamic_;
};
/** @brief Compare actual serial rows against standalone and retained public batch APIs.
 * @tparam Format Original storage; @param maximum_workers Real positive caller worker budget.
 */
template<Storage Format>void groupedRows(int maximum_workers)
{
    for(int units:{3,17,65})for(int rows:{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31,33})for(bool causal:{false,true})
    {
        const int count=rows*units;const auto original=source<Format>(count,7);auto expected=original;
        for(int row=0;row<rows;++row)one<Format>(expected.data()+std::size_t(row)*units,units,causal,0.17f,row);
        for(bool parallel:{false,true})
        {
            auto actual=original;many<Format>(actual.data(),rows,units,causal,0.17f,parallel);same(actual,expected);
            for(int workers=1;workers<=maximum_workers;++workers)
            {
                actual=original;int observed=0;
#pragma omp parallel num_threads(workers)
                {
#pragma omp single
                    observed=omp_get_num_threads();
                    many<Format>(actual.data(),rows,units,causal,0.17f,parallel);
                }
                ASSERT_EQ(observed,workers);same(actual,expected);
            }
        }
    }
}
/** @test All original M1..16/31/33 rows retain actual production serial bytes at every positive worker count. */
TEST(CPUSoftmax, AllPositiveWorkersRetainOriginalSerialRowsFP32)
{
    const TeamPolicy policy;
    groupedRows<Storage::FP32>(policy.threads());
}
/** @test Same whole-team sweep for BF16 storage; one storage per test keeps each within budget. */
TEST(CPUSoftmax, AllPositiveWorkersRetainOriginalSerialRowsBF16)
{
    const TeamPolicy policy;
    groupedRows<Storage::BFloat16>(policy.threads());
}
/** @test Same whole-team sweep for FP16 storage. */
TEST(CPUSoftmax, AllPositiveWorkersRetainOriginalSerialRowsFP16)
{
    const TeamPolicy policy;
    groupedRows<Storage::Half>(policy.threads());
}
/** @test Same whole-team sweep for Q8_1 storage. */
TEST(CPUSoftmax, AllPositiveWorkersRetainOriginalSerialRowsQ8_1)
{
    const TeamPolicy policy;
    groupedRows<Storage::Q8_1>(policy.threads());
}
/** @brief Observe an empty collective's completion before its enclosing team join.
 * @tparam Format Original storage.
 */
template<Storage Format>void emptyCompletion()
{
    for(bool hint:{false,true})for(int rows:{0,1})
    {
        std::atomic<int> arrived{0};int observed=0;
#pragma omp parallel num_threads(3)
        {
            const int rank=omp_get_thread_num();
            if(rank==2)std::this_thread::sleep_for(std::chrono::milliseconds(3));
            arrived.fetch_add(1,std::memory_order_release);
            many<Format>(nullptr,rows,rows?0:1,false,1.0f,hint);
            if(rank==0)observed=arrived.load(std::memory_order_acquire);
        }
        EXPECT_EQ(observed,3);
    }
}
/** @test Empty retained row-major APIs have a real release edge even with a false parallel hint. */
TEST(CPUSoftmax, EmptyRetainedWorkshareCompletesBeforeRelease)
{
    const TeamPolicy policy;
    emptyCompletion<Storage::FP32>();emptyCompletion<Storage::BFloat16>();emptyCompletion<Storage::Half>();emptyCompletion<Storage::Q8_1>();
}
} // namespace
} // namespace llaminar2::test
