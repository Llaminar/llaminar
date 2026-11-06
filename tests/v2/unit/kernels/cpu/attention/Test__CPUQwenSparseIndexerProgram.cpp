/**
 * @file Test__CPUQwenSparseIndexerProgram.cpp
 * @brief Independent public-indexer representation/RMS/partial-RoPE byte oracle.
 *
 * The reference quantizes finite values with binary-grid arithmetic in FP64,
 * not a production half/BF16 converter. Every declared FP32 multiply/add/divide
 * is rounded through volatile storage; the 128-coordinate reduction retains
 * the original butterfly and both rotary products precede their rounded add.
 * Original query/weight/table/raw-key bytes, exact in-place query reuse and
 * complete output guards are authenticated outside the production invocation.
 * Independent volatile public-libm function pointers prevent the oracle from
 * silently choosing a paired sincos ABI with different last bits. The ranking
 * regression separately treats produced queries/pools as score-phase inputs,
 * checks independent per-ISA FMA/head programs, and authenticates every original
 * score/selection/count guard; it does not replace the preparation oracle.
 * The integration totality cells derive all stages from original inputs, with
 * position/pair-varying supplied rotary planes and bitwise source-bank checks.
 */
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "kernels/cpu/attention/CPUQwenSparseRepresentation.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace llaminar2::qwen4exp::test
{
namespace
{
/** @brief Round one original FP32 product explicitly.
 * @param a Original first operand; @param b Original second operand; @return Rounded product.
 */
float multiply(float a,float b){volatile float x=a*b;return x;}
/** @brief Round one original FP32 sum explicitly.
 * @param a Original first operand; @param b Original second operand; @return Rounded sum.
 */
float add(float a,float b){volatile float x=a+b;return x;}
/** @brief Round one original FP32 quotient explicitly.
 * @param a Original numerator; @param b Original denominator; @return Rounded quotient.
 */
float divide(float a,float b){volatile float x=a/b;return x;}
/** @brief Independently score one produced query head against one original pool.
 * @param query Original produced128-coordinate head; @param pool Original pool.
 * @param isa Already-admitted native program; @return Its exact QSA dot word.
 * @details Volatile stores retain each intentional FMA and butterfly edge;
 * this reference never calls a production dot/fold or a generic attention fold.
 */
float rankingDot(const float*query,const float*pool,ISALevel isa)
{
    const unsigned width=isa==ISALevel::AVX512?16:8;std::array<float,16>lanes{};
    for(unsigned c=0;c<128;++c)
    {volatile float value=std::fma(query[c],pool[c],lanes[c%width]);lanes[c%width]=value;}
    for(unsigned step=width/2;step;step/=2)
        for(unsigned lane=0;lane<step;++lane)lanes[lane]=add(lanes[lane],lanes[lane+step]);
    return lanes[0];
}
/** @brief Call the declared public FP32 elementaries independently of production helpers.
 * @param angle Original separately rounded angle; @return Public cosf and sinf words.
 * @details Volatile function-pointer loads make both SDK calls observable, so
 * neither compiler profile may substitute a different paired libm ABI here.
 */
std::pair<float,float>elementaries(float angle)
{
    float(*volatile cosine)(float)=::cosf;float(*volatile sine)(float)=::sinf;
    return{cosine(angle),sine(angle)};
}
/** @brief Independently round a finite original FP32 value onto a binary representation grid.
 * @param input Original finite value; @param representation Declared representation.
 * @return Exact FP32 container after full ties-even half/BF16 rounding.
 */
float representation(float input,QwenSparseScalar representation)
{
    if(representation==QwenSparseScalar::FP32)return input;
    const auto bits=std::bit_cast<std::uint32_t>(input);
    if((bits&0x7fffffffu)>0x7f800000u)return std::bit_cast<float>(representation==QwenSparseScalar::FP16?(bits&0x80000000u)|0x7fc00000u:(bits&0xffff0000u)|0x00400000u);
    if((bits&0x7fffffffu)==0x7f800000u)return input;
    const double magnitude=std::fabs(double(input));if(magnitude==0)return input;
    int exponent=0;std::frexp(magnitude,&exponent);
    const int fraction=representation==QwenSparseScalar::FP16?10:7;
    const int minimum=representation==QwenSparseScalar::FP16?-24:-133;
    const int maximumExponent=representation==QwenSparseScalar::FP16?16:128;
    if(exponent>maximumExponent)return std::copysign(INFINITY,input);
    const int step=std::max(minimum,exponent-fraction-1);const double units=std::ldexp(magnitude,-step);
    const double floor=std::floor(units),remainder=units-floor;
    const double nearest=floor+(remainder>0.5||(remainder==0.5&&std::fmod(floor,2.0)!=0.0)?1.0:0.0);
    const auto result=std::ldexp(nearest,step);
    if(result>=std::ldexp(1.0,maximumExponent))return std::copysign(INFINITY,input);
    return std::copysign(float(result),input);
}
/** @brief Independently normalize one original128-coordinate head with fixed separate FP32 edges.
 * @param input Original FP32 source; @param weight Original converted FP32 weights.
 * @param rep Declared representation; @param epsilon Original FP32 epsilon.
 * @return Exact produced reference cast-boundary stage, independent of rotary geometry.
 */
std::array<float,128>normalize(const float*input,const float*weight,QwenSparseScalar rep,float epsilon)
{
    std::array<float,128>rounded{},squares{},normalized{};
    for(unsigned c=0;c<128;++c){rounded[c]=representation(input[c],rep);squares[c]=multiply(rounded[c],rounded[c]);}
    for(unsigned width=64;width;width/=2)for(unsigned c=0;c<width;++c)squares[c]=add(squares[c],squares[c+width]);
    const auto inverse=divide(1.0f,std::sqrt(add(divide(squares[0],128.0f),epsilon)));
    for(unsigned c=0;c<128;++c)normalized[c]=representation(multiply(multiply(rounded[c],inverse),weight[c]),rep);
    return normalized;
}
/** @brief Independently rotate one produced reference stage without recomputing invariant RMS work.
 * @param normalized Independent produced128-coordinate cast stage.
 * @param cosine Original cosine row; @param sine Original sine row.
 * @param rotaryDim Original even width; @param rep Original representation.
 * @return Exact independent reference output.
 */
std::array<float,128>rotate(const std::array<float,128>&normalized,const float*cosine,const float*sine,int rotaryDim,QwenSparseScalar rep)
{
    std::array<float,128>output{};const unsigned half=unsigned(rotaryDim/2);
    for(unsigned c=0;c<128;++c)
    {
        if(c>=unsigned(rotaryDim)){output[c]=normalized[c];continue;}
        const auto co=representation(cosine[c%half],rep),si=representation(sine[c%half],rep);
        const auto partner=c<half?-normalized[c+half]:normalized[c-half];
        output[c]=representation(add(representation(multiply(normalized[c],co),rep),representation(multiply(partner,si),rep)),rep);
    }
    return output;
}
/** @brief Compare complete result/guard words, not a same-path serial output or tolerance.
 * @param actual Original complete output; @param gold Independent reference.
 */
void authenticate(const std::vector<float>&actual,const std::vector<float>&gold)
{
    ASSERT_EQ(actual.size(),gold.size());for(std::size_t i=0;i<actual.size();++i)
    {ASSERT_TRUE(std::isfinite(actual[i]));ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[i]),std::bit_cast<std::uint32_t>(gold[i]))<<"word="<<i;}
}
/** @brief Authenticate one native representation's entire original source-word corpus.
 * @tparam Rep Original closed QSA representation.
 * @param inputs Diagnostic original FP32 words, never production decoded storage.
 */
template<QwenSparseScalar Rep>void authenticateCast(const std::vector<float>&inputs)
{
    constexpr std::size_t prefix=7,suffix=9;constexpr std::uint32_t guard=0x45b639deu;
    for(std::size_t base=0;base<inputs.size();base+=8)
    {
        std::array<float,8>original{};std::array<std::uint32_t,8>expected{};std::array<float,prefix+8+suffix>output;
        output.fill(std::bit_cast<float>(guard));
        for(std::size_t lane=0;lane<8;++lane){original[lane]=inputs[(base+lane)%inputs.size()];expected[lane]=std::bit_cast<std::uint32_t>(representation(original[lane],Rep));output[prefix+lane]=std::bit_cast<float>(~expected[lane]);}
        const auto unchanged=original;
        const auto result=::llaminar2::cpu::qsa_native::roundRepresentation8<Rep>(::llaminar2::cpu::qsa_native::load8(original.data()));
#if LLAMINAR_COMPILED_WITH_NEON
        vst1q_f32(output.data()+prefix,result.low);vst1q_f32(output.data()+prefix+4,result.high);
#else
        _mm256_storeu_ps(output.data()+prefix,result);
#endif
        for(std::size_t lane=0;lane<8;++lane)ASSERT_EQ(std::bit_cast<std::uint32_t>(output[prefix+lane]),expected[lane])<<"base="<<base<<" lane="<<lane<<" input="<<std::bit_cast<std::uint32_t>(original[lane]);
        for(std::size_t i=0;i<output.size();++i)if(i<prefix||i>=prefix+8)ASSERT_EQ(std::bit_cast<std::uint32_t>(output[i]),guard);
        ASSERT_EQ(std::memcmp(original.data(),unchanged.data(),sizeof(original)),0);
    }
}
/** @brief Memoize only invariant diagnostic work, authenticated against every original input.
 * The kernel still runs every geometry; these owned expected stages avoid repeating
 * the identical FP64-grid reference normalization64 times in strict Debug.
 */
struct ReferenceNormalization
{
    std::vector<float>query,raw,norm;
    std::vector<std::array<float,128>>queries,pools;
};
/** @brief Restore the original caller OpenMP policy. */
struct TeamPolicy final
{
    int threads=omp_get_max_threads(),dynamic=omp_get_dynamic();
    /** @brief Admit one independent caller without changing host affinity. */
    TeamPolicy(){omp_set_dynamic(0);omp_set_num_threads(1);}
    /** @brief Restore original policy on normal and failed assertions. */
    ~TeamPolicy(){omp_set_num_threads(threads);omp_set_dynamic(dynamic);}
};
} // namespace
/** @test Native casts retain full half-subnormal RNE, BF16 ties and explicit special-word programs. */
TEST(CPUQwenSparseIndexerProgram,NativeRepresentationCastsRetainIndependentIEEEWordBoundaries)
{
    ::llaminar2::cpu::qsa_native::requireNativeProgram(activeISALevel());
    std::vector<float>inputs;inputs.reserve(500000);
    for(unsigned code=0;code<65536;++code)
    {
        const unsigned exponent=(code>>10)&31,fraction=code&1023;float half;
        if(exponent==31)half=std::bit_cast<float>(((code&0x8000u)<<16)|0x7f800000u|(fraction<<13));
        else half=std::copysign(std::ldexp(float(exponent?1024+fraction:fraction),exponent?int(exponent)-25:-24),code&0x8000u?-1.0f:1.0f);
        inputs.push_back(half);inputs.push_back(std::bit_cast<float>(code<<16));
        if(code<0x7bffu)
        {
            const unsigned next=code+1,e=(next>>10)&31,f=next&1023;
            const float upper=std::ldexp(float(e?1024+f:f),e?int(e)-25:-24);
            const float middle=float((double(half)+double(upper))*0.5);
            for(float x:{std::nextafter(middle,-INFINITY),middle,std::nextafter(middle,INFINITY)}){inputs.push_back(x);inputs.push_back(-x);}
        }
    }
    std::uint32_t state=0x9e3779b9u;for(unsigned i=0;i<65536;++i){state^=state<<13;state^=state>>17;state^=state<<5;inputs.push_back(std::bit_cast<float>(state));}
    // Terminal RNE carries are not adjacent finite half/BF16 grid words: pin
    // both signs and both neighboring FP32 words independently of random data.
    for(auto rep:{QwenSparseScalar::FP16,QwenSparseScalar::BF16})for(std::uint32_t sign:{0u,0x80000000u})for(int offset:{-1,0,1})
    {
        const std::uint32_t midpoint=rep==QwenSparseScalar::FP16?0x477ff000u:0x7f7f8000u;
        const std::uint32_t finite=rep==QwenSparseScalar::FP16?0x477fe000u:0x7f7f0000u;
        const auto source=(midpoint+std::uint32_t(offset))|sign,expected=(offset<0?finite:0x7f800000u)|sign;
        const auto value=std::bit_cast<float>(source);ASSERT_EQ(std::bit_cast<std::uint32_t>(representation(value,rep)),expected);inputs.push_back(value);
    }
    authenticateCast<QwenSparseScalar::FP32>(inputs);authenticateCast<QwenSparseScalar::FP16>(inputs);authenticateCast<QwenSparseScalar::BF16>(inputs);
}
/** @brief Prove query/pool preparation retains independently rounded bytes across rotary
 * widths, rows and exact alias for one original representation. Each representation is
 * its own test so the strict Debug build stays inside the per-test budget. */
void originalRepresentationAndPartialRope(QwenSparseScalar rep)
{
    ::llaminar2::cpu::qsa_native::requireNativeProgram(activeISALevel());TeamPolicy team;
    constexpr std::size_t guard=8;constexpr float canary=919.25f,epsilon=1.0e-6f;
    std::map<std::pair<QwenSparseScalar,int>,ReferenceNormalization>normalizations;
    for(int rotary=2;rotary<=128;rotary+=2)for(int rows:{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31,33})for(bool analytic:{false,true})
    {
        if(analytic&&rotary!=64)continue;
        SCOPED_TRACE(::testing::Message()<<"representation="<<int(rep)<<" rotary="<<rotary<<" rows="<<rows<<" analytic="<<analytic);
        const int blocks=(rows+3)/4,half=rotary/2;QwenSparseReplay replay{0,rows,0};
        std::vector<float>query(guard+std::size_t(rows)*4*128+9,canary),raw(std::size_t(rows)*128),partial(4*128,canary),norm(128),cosine(std::size_t(rows+1)*half),sine(cosine.size());
        for(std::size_t c=guard;c<query.size()-9;++c)query[c]=float(int((c*7331+17)%257)-128)*0.017321f+0.00001f;
        for(std::size_t c=0;c<raw.size();++c)raw[c]=float(int((c*47+73)%251)-125)*0.021731f-0.00001f;
        for(unsigned c=0;c<128;++c)norm[c]=1.0f+float(int((c*11)%31)-15)*0.00731f;
        const auto frequency=makeQwenSparsePowerFrequencies(10000.0f,64);
        for(int position=0;position<=rows;++position)for(int pair=0;pair<half;++pair)
        {const float angle=analytic?multiply(float(position),frequency[pair]):float(position+1)*float((pair*13)%31+1)*0.01713f;const auto[co,si]=elementaries(angle);cosine[std::size_t(position)*half+pair]=co;sine[std::size_t(position)*half+pair]=si;}
        auto [found,inserted]=normalizations.try_emplace({rep,rows});auto&reference=found->second;
        if(inserted)
        {
            reference.query=query;reference.raw=raw;reference.norm=norm;
            for(int rowHead=0;rowHead<rows*4;++rowHead)reference.queries.push_back(normalize(query.data()+guard+std::size_t(rowHead)*128,norm.data(),rep,epsilon));
            for(int block=0;block<rows/4;++block)
            {
                std::array<float,128>pool{};for(int c=0;c<128;++c){float sum=representation(raw[std::size_t(block)*4*128+c],rep);for(int r=1;r<4;++r)sum=add(sum,representation(raw[(std::size_t(block)*4+r)*128+c],rep));pool[c]=representation(multiply(sum,0.25f),rep);}
                reference.pools.push_back(normalize(pool.data(),norm.data(),rep,epsilon));
            }
        }
        ASSERT_EQ(reference.query,query);ASSERT_EQ(reference.raw,raw);ASSERT_EQ(reference.norm,norm);
        std::vector<float>goldQuery(query.size(),canary),goldPools(guard+std::size_t(blocks)*128+9,canary);
        for(int row=0;row<rows;++row)for(int head=0;head<4;++head)
        {const auto offset=guard+(std::size_t(row)*4+head)*128;const auto gold=rotate(reference.queries[std::size_t(row)*4+head],cosine.data()+std::size_t(row)*half,sine.data()+std::size_t(row)*half,rotary,rep);std::copy(gold.begin(),gold.end(),goldQuery.begin()+offset);}
        for(int block=0;block<rows/4;++block)
        {const int position=block*4;const auto gold=rotate(reference.pools[block],cosine.data()+std::size_t(position)*half,sine.data()+std::size_t(position)*half,rotary,rep);std::copy(gold.begin(),gold.end(),goldPools.begin()+guard+std::size_t(block)*128);}
        const auto originalQuery=query,originalRaw=raw,originalNorm=norm,originalCosine=cosine,originalSine=sine;
        for(bool alias:{false,true})
        {
            SCOPED_TRACE(alias?"exact-query-alias":"disjoint-query");auto actual=alias?query:goldQuery;auto pools=goldPools;
            for(std::size_t c=guard;c<actual.size()-9;++c){if(alias)ASSERT_NE(std::bit_cast<std::uint32_t>(actual[c]),std::bit_cast<std::uint32_t>(goldQuery[c]));else actual[c]=std::bit_cast<float>(std::bit_cast<std::uint32_t>(goldQuery[c])^0x80000000u);}
            for(std::size_t c=guard;c<guard+std::size_t(rows/4)*128;++c)pools[c]=std::bit_cast<float>(std::bit_cast<std::uint32_t>(goldPools[c])^0x80000000u);
            std::vector<float>scores(std::size_t(rows)*blocks,canary);std::vector<std::int32_t>selected(std::size_t(rows)*QwenSparseSelected,-99),counts(rows,-99);
            const QwenSparseIndexer p{.replay=&replay,.query=alias?actual.data()+guard:query.data()+guard,.raw_keys=raw.data(),.partial_keys=partial.data(),.pooled_keys=pools.data()+guard,.query_norm=norm.data(),.key_norm=norm.data(),.cosine=analytic?nullptr:cosine.data(),.sine=analytic?nullptr:sine.data(),.prepared_query=actual.data()+guard,.scores=scores.data(),.selected=selected.data(),.selected_count=counts.data(),.row_capacity=rows,.token_capacity=rows,.block_capacity=blocks,.position_capacity=rows+1,.rotary_dim=rotary,.norm_epsilon=epsilon,.representation=rep,.rotary_source=analytic?QwenSparseRotarySource::AnalyticTheta:QwenSparseRotarySource::TablePlanes,.rope_theta=10000.0f,.inverse_frequencies=frequency};
            cpuQwenSparseIndexer(p);
            {SCOPED_TRACE("prepared-query-publication");authenticate(actual,goldQuery);}
            {SCOPED_TRACE("prepared-pool-publication");authenticate(pools,goldPools);}
        }
        EXPECT_EQ(query,originalQuery);EXPECT_EQ(raw,originalRaw);EXPECT_EQ(norm,originalNorm);EXPECT_EQ(cosine,originalCosine);EXPECT_EQ(sine,originalSine);
    }
}
/** @test FP32 query/pool preparation bytes. */
TEST(CPUQwenSparseIndexerProgram,OriginalRepresentationAndPartialRopeMatchIndependentBytesFP32){originalRepresentationAndPartialRope(QwenSparseScalar::FP32);}
/** @test FP16 query/pool preparation bytes. */
TEST(CPUQwenSparseIndexerProgram,OriginalRepresentationAndPartialRopeMatchIndependentBytesFP16){originalRepresentationAndPartialRope(QwenSparseScalar::FP16);}
/** @test BF16 query/pool preparation bytes. */
TEST(CPUQwenSparseIndexerProgram,OriginalRepresentationAndPartialRopeMatchIndependentBytesBF16){originalRepresentationAndPartialRope(QwenSparseScalar::BF16);}
/** @test Actual FP32 mean/RoPE edges do not become cross-statement FMA after GNU inlining. */
TEST(CPUQwenSparseIndexerProgram,FP32MeanAndRopeCastEdgesCannotContractAfterInlining)
{
    ::llaminar2::cpu::qsa_native::requireNativeProgram(activeISALevel());TeamPolicy team;
    constexpr std::size_t guard=8;constexpr float canary=919.25f;
    for(bool rotaryCancellation:{false,true})for(int rotary:{2,16,18})
    {
        SCOPED_TRACE(rotaryCancellation?"rounded-products-before-RoPE-add":"subnormal-mean-before-epsilon");
        SCOPED_TRACE(::testing::Message()<<"rotary="<<rotary);const int half=rotary/2;
        QwenSparseReplay replay{0,1,0};std::array<float,128>norm;norm.fill(1.5f);
        std::vector<float>query(guard+512+9,canary),raw(128,1),partial(512,1),pools(128,canary),scores(1,canary);
        std::array<std::int32_t,QwenSparseSelected>selected{};std::int32_t count=-1;
        const float epsilon=rotaryCancellation?0.5f:std::bit_cast<float>(1u);
        const float coefficient=std::bit_cast<float>(0x3f3504f3u);
        const std::vector<float>cosine(half,rotaryCancellation?coefficient:1.0f),sine(half,rotaryCancellation?coefficient:0.0f);
        for(int head=0;head<4;++head)for(int c=0;c<128;++c)
            query[guard+std::size_t(head)*128+c]=rotaryCancellation?(c<64?1.0f:std::ldexp(1.0f,-40)):(c==0?std::bit_cast<float>(0x1bb504f3u):std::ldexp(1.0f,-100));
        if(rotaryCancellation)
        {
            const float value=std::bit_cast<float>(0x3f800001u);
            // Pin paired coordinates in scalar, fully native and native+fringe
            // shapes, rather than certifying native RoPE through a scalar pair.
            std::fill_n(norm.begin(),rotary,value);
            const auto first=multiply(value,coefficient);
            ASSERT_EQ(std::bit_cast<std::uint32_t>(add(first,multiply(-value,coefficient))),0u);
            ASSERT_NE(std::bit_cast<std::uint32_t>(std::fma(-value,coefficient,first)),0u);
        }
        else
        {
            const auto summary=multiply(query[guard],query[guard]);ASSERT_EQ(std::bit_cast<std::uint32_t>(summary),64u);
            ASSERT_EQ(std::bit_cast<std::uint32_t>(add(divide(summary,128.0f),epsilon)),1u);
            ASSERT_EQ(std::bit_cast<std::uint32_t>(std::fma(summary,0.0078125f,epsilon)),2u);
        }
        std::vector<float>gold(query.size(),canary);
        for(int head=0;head<4;++head)
        {const auto offset=guard+std::size_t(head)*128;const auto normalized=normalize(query.data()+offset,norm.data(),QwenSparseScalar::FP32,epsilon);const auto expected=rotate(normalized,cosine.data(),sine.data(),rotary,QwenSparseScalar::FP32);std::copy(expected.begin(),expected.end(),gold.begin()+offset);}
        if(rotaryCancellation)for(int head=0;head<4;++head)for(int pair=0;pair<half;++pair)
            ASSERT_EQ(std::bit_cast<std::uint32_t>(gold[guard+std::size_t(head)*128+pair]),0u);
        const auto originalQuery=query;const auto originalNorm=norm;
        for(bool alias:{false,true})
        {
            SCOPED_TRACE(alias?"exact-query-alias":"disjoint-query");auto actual=alias?query:gold;
            for(std::size_t i=guard;i<guard+512;++i)
            {if(alias)ASSERT_NE(std::bit_cast<std::uint32_t>(actual[i]),std::bit_cast<std::uint32_t>(gold[i]));else actual[i]=std::bit_cast<float>(std::bit_cast<std::uint32_t>(gold[i])^0x80000000u);}
            const QwenSparseIndexer p{.replay=&replay,.query=alias?actual.data()+guard:query.data()+guard,.raw_keys=raw.data(),.partial_keys=partial.data(),.pooled_keys=pools.data(),.query_norm=norm.data(),.key_norm=norm.data(),.cosine=cosine.data(),.sine=sine.data(),.prepared_query=actual.data()+guard,.scores=scores.data(),.selected=selected.data(),.selected_count=&count,.row_capacity=1,.token_capacity=1,.block_capacity=1,.position_capacity=1,.rotary_dim=rotary,.norm_epsilon=epsilon,.representation=QwenSparseScalar::FP32};
            cpuQwenSparseIndexer(p);authenticate(actual,gold);
        }
        EXPECT_EQ(query,originalQuery);EXPECT_EQ(norm,originalNorm);
    }
}
/** @test Public analytic publication uses separate SDK elementaries, not a compiler-selected paired ABI. */
TEST(CPUQwenSparseIndexerProgram,IndexerAnalyticElementariesCannotMerge)
{
    ::llaminar2::cpu::qsa_native::requireNativeProgram(activeISALevel());TeamPolicy team;
    constexpr std::size_t guard=8;constexpr float canary=919.25f;constexpr int half=32;
    // Pin the primitive at a known divergent SDK angle, using public unary
    // functions as the independent authority rather than a private sincos ABI.
    const auto angle=std::bit_cast<float>(0x3a83ef22u);const auto expectedWords=elementaries(angle);
    float actualCosine=std::bit_cast<float>(std::bit_cast<std::uint32_t>(expectedWords.first)^0x80000000u);
    float actualSine=std::bit_cast<float>(std::bit_cast<std::uint32_t>(expectedWords.second)^0x80000000u);
    detail::hostSeparateSinCos(angle,actualCosine,actualSine);
    ASSERT_EQ(std::bit_cast<std::uint32_t>(actualCosine),std::bit_cast<std::uint32_t>(expectedWords.first));
    ASSERT_EQ(std::bit_cast<std::uint32_t>(actualSine),std::bit_cast<std::uint32_t>(expectedWords.second));
    for(int position:{1,4,8,16,32})
    {
    SCOPED_TRACE(::testing::Message()<<"absolute-position="<<position);
    QwenSparseReplay replay{0,1,position};std::array<float,128>norm{};
    std::fill_n(norm.begin(),half,1.0f);const auto frequency=makeQwenSparsePowerFrequencies(10000.0f,64);
    std::array<float,half>cosine{},sine{};
    for(int pair=0;pair<half;++pair)
    {const auto[co,si]=elementaries(multiply(float(position),frequency[pair]));cosine[pair]=co;sine[pair]=si;}
    std::vector<float>query(guard+512+9,canary),raw(128,1),partial(512,1),pools(128,canary),scores(1,canary);
    for(int head=0;head<4;++head)for(int c=0;c<128;++c)query[guard+std::size_t(head)*128+c]=c<64?2.0f:std::ldexp(1.0f,-40);
    // Mean2 + epsilon2 gives inverse1/2 exactly. Original input2 differs
    // from every cosf/sinf output, including legitimate cosf(angle)==1.
    std::vector<float>gold(query.size(),canary);
    for(int head=0;head<4;++head)
    {
        const auto offset=guard+std::size_t(head)*128;
        const auto normalized=normalize(query.data()+offset,norm.data(),QwenSparseScalar::FP32,2.0f);
        ASSERT_EQ(normalized[1],1.0f);ASSERT_EQ(normalized[1+half],0.0f);
        const auto expected=rotate(normalized,cosine.data(),sine.data(),64,QwenSparseScalar::FP32);
        ASSERT_EQ(std::bit_cast<std::uint32_t>(expected[1]),std::bit_cast<std::uint32_t>(cosine[1]));
        ASSERT_EQ(std::bit_cast<std::uint32_t>(expected[1+half]),std::bit_cast<std::uint32_t>(sine[1]));
        std::copy(expected.begin(),expected.end(),gold.begin()+offset);
    }
    const auto originalQuery=query;const auto originalNorm=norm;const auto originalFrequency=frequency;
    const auto originalRaw=raw;const auto originalPartial=partial;
    for(bool alias:{false,true})
    {
        SCOPED_TRACE(alias?"exact-query-alias":"disjoint-query");auto actual=alias?query:gold;
        for(std::size_t i=guard;i<guard+512;++i)
        {if(alias)ASSERT_NE(std::bit_cast<std::uint32_t>(actual[i]),std::bit_cast<std::uint32_t>(gold[i]));else actual[i]=std::bit_cast<float>(std::bit_cast<std::uint32_t>(gold[i])^0x80000000u);}
        std::vector<std::int32_t>selected(guard+QwenSparseSelected+9,-91),count(guard+1+9,-71);
        const QwenSparseIndexer p{.replay=&replay,.query=alias?actual.data()+guard:query.data()+guard,.raw_keys=raw.data(),.partial_keys=partial.data(),.pooled_keys=pools.data(),.query_norm=norm.data(),.key_norm=norm.data(),.prepared_query=actual.data()+guard,.scores=scores.data(),.selected=selected.data()+guard,.selected_count=count.data()+guard,.row_capacity=1,.token_capacity=1,.block_capacity=1,.position_capacity=position+1,.rotary_dim=64,.norm_epsilon=2.0f,.representation=QwenSparseScalar::FP32,.rotary_source=QwenSparseRotarySource::AnalyticTheta,.rope_theta=10000.0f,.inverse_frequencies=frequency};
        cpuQwenSparseIndexer(p);authenticate(actual,gold);
        EXPECT_EQ(p.inverse_frequencies,originalFrequency);EXPECT_EQ(count[guard],1);EXPECT_EQ(selected[guard],0);
        for(std::size_t i=0;i<count.size();++i)if(i!=guard)EXPECT_EQ(count[i],-71);
        for(std::size_t i=0;i<selected.size();++i)if(i!=guard)EXPECT_EQ(selected[i],-91);
        for(float value:pools)EXPECT_EQ(value,canary);for(float value:scores)EXPECT_EQ(value,canary);
    }
    EXPECT_EQ(query,originalQuery);EXPECT_EQ(norm,originalNorm);EXPECT_EQ(frequency,originalFrequency);
    EXPECT_EQ(raw,originalRaw);EXPECT_EQ(partial,originalPartial);
    }
}
/** @test Real ranking preserves native head order across threshold rows and all output guards.
 * Produced query/pool stages are inputs to this focused score oracle; other
 * tests independently certify their preparation. Inactive/nonranking score
 * poison is never consumed, and the public selector must retain exact ties.
 */
TEST(CPUQwenSparseIndexerProgram,NativeRankingScoresRetainOriginalHeadProgram)
{
    const auto isa=activeISALevel();::llaminar2::cpu::qsa_native::requireNativeProgram(isa);
    const TeamPolicy team;constexpr std::size_t guard=7,suffix=9;
    constexpr std::uint32_t poison=0x7fc54321u;const auto canary=std::bit_cast<float>(0x4a613257u);
    std::size_t authoredScores=0,cutoffWitnesses=0,headOrderWitnesses=0;
    for(auto rep:{QwenSparseScalar::FP32,QwenSparseScalar::FP16,QwenSparseScalar::BF16})
    for(auto demand:{QwenSparseQueryDemand::AllLive,QwenSparseQueryDemand::RankingOnly})
    for(int rows:{1,3,16,33})
    {
        // M1 must really rank; grouped cases straddle the same threshold and
        // leave row0's score poison unused before ranking starts at row1.
        const int prefix=QwenSparseFirstRankingLogicalRow-(rows==1?0:1);
        SCOPED_TRACE(::testing::Message()<<"rep="<<int(rep)<<" demand="<<int(demand)<<" rows="<<rows<<" prefix="<<prefix);
        const int capacity=rows+2,tokens=prefix+capacity,blocks=(tokens+3)/4,positions=tokens+1;
        QwenSparseReplay replay;replay.prefix_tokens=prefix;replay.live_rows=rows;
        std::vector<float>query(guard+std::size_t(capacity)*512+suffix,canary),raw(std::size_t(capacity)*128),partial(512),norm(128,1);
        for(std::size_t i=guard;i<query.size()-suffix;++i)query[i]=float(int((i*13)%97)-48)*0.015625f;
        for(std::size_t i=0;i<raw.size();++i)raw[i]=float(int((i*7)%59)-29)*0.03125f;
        for(std::size_t i=0;i<partial.size();++i)partial[i]=float(int((i*11)%61)-30)*0.03125f;
        std::vector<float>pools(guard+std::size_t(blocks)*128+suffix,canary),prepared(query.size(),canary);
        for(std::size_t i=guard;i<pools.size()-suffix;++i)pools[i]=float(int((i*17)%127)-63)*0.015625f;
        const bool witness=demand==QwenSparseQueryDemand::AllLive&&rows==1;
        if(witness)
        {
            // Four orthogonal query axes yield X, X*2^-24, X*2^-24, 0
            // against every old pool. X is near sqrt(128), so reversing head
            // addition changes the final scaled word in every representation.
            std::fill(query.begin()+guard,query.end()-suffix,0);
            for(int row=0;row<capacity;++row)for(int head=0;head<4;++head)
                query[guard+std::size_t(row)*512+head*128+head]=1;
            std::fill(raw.begin(),raw.end(),0);std::fill(partial.begin(),partial.end(),0);
            for(int row=0;row<capacity;++row)raw[std::size_t(row)*128]=1;
            for(int row=0;row<4;++row)partial[std::size_t(row)*128]=1;
            for(int block=0;block<prefix/4;++block)
            {
                auto*pool=pools.data()+guard+std::size_t(block)*128;std::fill_n(pool,128,0);
                pool[0]=1;pool[1]=pool[2]=0x1p-24f;
            }
        }
        std::vector<float>cosine(positions,1),sine(positions,0);
        std::vector<float>scores(guard+std::size_t(capacity)*blocks+suffix,std::bit_cast<float>(poison));
        std::vector<std::int32_t>selected(guard+std::size_t(capacity)*QwenSparseSelected+suffix,-91),count(guard+capacity+suffix,-71);
        auto goldScores=scores;auto goldSelected=selected;auto goldCount=count;
        const auto originalQuery=query,originalRaw=raw,originalPartial=partial,originalNorm=norm,originalPools=pools;
        const auto originalCosine=cosine,originalSine=sine;
        const QwenSparseIndexer p{.replay=&replay,.query=query.data()+guard,.raw_keys=raw.data(),.partial_keys=partial.data(),.pooled_keys=pools.data()+guard,.query_norm=norm.data(),.key_norm=norm.data(),.cosine=cosine.data(),.sine=sine.data(),.prepared_query=prepared.data()+guard,.scores=scores.data()+guard,.selected=selected.data()+guard,.selected_count=count.data()+guard,.row_capacity=capacity,.token_capacity=tokens,.block_capacity=blocks,.position_capacity=positions,.rotary_dim=2,.norm_epsilon=1.0e-4f,.representation=rep,.query_demand=demand};
        cpuQwenSparseIndexer(p);
        for(int row=0;row<capacity;++row)
        {
            if(row>=rows){goldCount[guard+row]=0;continue;}
            const int visible=prefix+row+1,complete=visible/4,keep=std::min(complete,QwenSparseBlocks);
            if(complete>QwenSparseBlocks)
                for(int block=0;block<complete;++block)
                {
                    float score=0;std::array<float,4>contribution{};
                    for(int head=0;head<4;++head)
                    {
                        const auto value=rankingDot(prepared.data()+guard+std::size_t(row)*512+head*128,pools.data()+guard+std::size_t(block)*128,isa);
                        ASSERT_TRUE(std::isfinite(value));contribution[head]=std::max(0.0f,value);score=add(score,contribution[head]);
                    }
                    const auto expected=multiply(score,0.08838834764831844055f);ASSERT_TRUE(std::isfinite(expected));
                    if(witness&&block==0)
                    {
                        float reverse=0;for(int head=3;head>=0;--head)reverse=add(reverse,contribution[head]);
                        ASSERT_NE(std::bit_cast<std::uint32_t>(score),std::bit_cast<std::uint32_t>(reverse));
                        ASSERT_NE(std::bit_cast<std::uint32_t>(expected),std::bit_cast<std::uint32_t>(multiply(reverse,0.08838834764831844055f)));
                        ++headOrderWitnesses;
                    }
                    goldScores[guard+std::size_t(row)*blocks+block]=expected;++authoredScores;
                }
            std::vector<int>chosen(complete);for(int block=0;block<complete;++block)chosen[block]=block;
            if(complete>QwenSparseBlocks)
            {
                const auto*expected=goldScores.data()+guard+std::size_t(row)*blocks;
                std::sort(chosen.begin(),chosen.end(),[&](int a,int b){return expected[a]>expected[b]||(expected[a]==expected[b]&&a<b);});
                chosen.resize(keep);std::sort(chosen.begin(),chosen.end());
                if(witness)
                {
                    // The newly completed basis pool wins; 511 of the512
                    // identical old pools must be retained by smaller ordinal.
                    ASSERT_GT(expected[QwenSparseBlocks],expected[0]);
                    ASSERT_EQ(std::bit_cast<std::uint32_t>(expected[QwenSparseBlocks-1]),std::bit_cast<std::uint32_t>(expected[0]));
                    ASSERT_EQ(chosen[QwenSparseBlocks-2],QwenSparseBlocks-2);
                    ASSERT_EQ(chosen[QwenSparseBlocks-1],QwenSparseBlocks);++cutoffWitnesses;
                }
            }
            for(int block=0;block<keep;++block)for(int member=0;member<4;++member)
                goldSelected[guard+std::size_t(row)*QwenSparseSelected+block*4+member]=chosen[block]*4+member;
            for(int tail=0;tail<visible%4;++tail)
                goldSelected[guard+std::size_t(row)*QwenSparseSelected+keep*4+tail]=complete*4+tail;
            goldCount[guard+row]=keep*4+visible%4;
        }
        // Check every word, but report once per cell without aborting later
        // representations: a failing control must exercise all format witnesses.
        std::size_t mismatches=0,first=0;
        for(std::size_t i=0;i<scores.size();++i)
            if(std::bit_cast<std::uint32_t>(scores[i])!=std::bit_cast<std::uint32_t>(goldScores[i]))
            {if(mismatches==0)first=i;++mismatches;}
        EXPECT_EQ(mismatches,0u)<<"first score word="<<first
            <<" actual bits="<<std::bit_cast<std::uint32_t>(scores[first])
            <<" gold bits="<<std::bit_cast<std::uint32_t>(goldScores[first]);
        EXPECT_EQ(selected,goldSelected);EXPECT_EQ(count,goldCount);
        EXPECT_EQ(query,originalQuery);EXPECT_EQ(raw,originalRaw);EXPECT_EQ(partial,originalPartial);EXPECT_EQ(norm,originalNorm);
        EXPECT_EQ(cosine,originalCosine);EXPECT_EQ(sine,originalSine);
        for(std::size_t i=0;i<pools.size();++i)
            if(i<guard||i>=guard+std::size_t(blocks)*128||(i-guard)/128<std::size_t(prefix/4)||(i-guard)/128>=std::size_t((prefix+rows)/4))
                EXPECT_EQ(std::bit_cast<std::uint32_t>(pools[i]),std::bit_cast<std::uint32_t>(originalPools[i]));
        for(std::size_t i=0;i<prepared.size();++i)
            if(i<guard||i>=guard+std::size_t(rows)*512||
                (demand==QwenSparseQueryDemand::RankingOnly&&(prefix+int((i-guard)/512)+1)/4<=QwenSparseBlocks))
                EXPECT_EQ(std::bit_cast<std::uint32_t>(prepared[i]),std::bit_cast<std::uint32_t>(canary));
        EXPECT_EQ(replay.prefix_tokens,prefix);EXPECT_EQ(replay.live_rows,rows);
    }
    EXPECT_GT(authoredScores,0u);EXPECT_EQ(cutoffWitnesses,3u);EXPECT_EQ(headOrderWitnesses,3u);
}
namespace
{
/** @brief Closed original representation and accepted-tail geometry for totality cells. */
struct IndexerTotalityCase
{
    QwenSparseScalar representation; ///< Original FP32-container cast program.
    int tail; ///< Original prefix remainder0..3, never a worker-dependent partition.
};
constexpr std::size_t totalityGuard=7,totalitySuffix=9;
constexpr float totalityCanary=std::bit_cast<float>(0x4a613257u);
constexpr float totalityScorePoison=std::bit_cast<float>(0x7fc54321u);
/** @brief Authenticate every float/guard word, including intentional unconsumed NaN poison.
 * @param actual Complete original result storage; @param expected Independent complete gold.
 * @param role Original result role for a bounded diagnostic.
 */
void expectIndexerWords(std::span<const float>actual,std::span<const float>expected,const char*role)
{
    ASSERT_EQ(actual.size(),expected.size());std::size_t mismatches=0,first=0;
    for(std::size_t i=0;i<actual.size();++i)
        if(std::bit_cast<std::uint32_t>(actual[i])!=std::bit_cast<std::uint32_t>(expected[i]))
        {if(mismatches==0)first=i;++mismatches;}
    EXPECT_EQ(mismatches,0u)<<role<<" first word="<<first;
}
/** @brief Persistent diagnostic output banks; reset changes bytes, never pointer topology. */
struct IndexerTotalityBanks
{
    std::vector<float>prepared,pools,scores;
    std::vector<std::int32_t>selected,count;
    /** @brief Allocate one fixed fixture before invoking any kernel or retained team.
     * @param capacity Original row envelope; @param blocks Original pool stride.
     * @param initialPools Original old pools, unused capacity and all guard words.
     */
    IndexerTotalityBanks(int capacity,int blocks,const std::vector<float>&initialPools)
        :prepared(totalityGuard+std::size_t(capacity)*512+totalitySuffix,totalityCanary),pools(initialPools),
         scores(totalityGuard+std::size_t(capacity)*blocks+totalitySuffix,totalityScorePoison),
         selected(totalityGuard+std::size_t(capacity)*QwenSparseSelected+totalitySuffix,-91),
         count(totalityGuard+std::size_t(capacity)+totalitySuffix,-71){}
    /** @brief Re-poison all writers and restore original pools without reallocating.
     * @param initialPools Same-sized original source/guard words retained by the fixture.
     */
    void reset(const std::vector<float>&initialPools)
    {
        ASSERT_EQ(pools.size(),initialPools.size());std::copy(initialPools.begin(),initialPools.end(),pools.begin());
        std::fill(prepared.begin(),prepared.end(),totalityCanary);std::fill(scores.begin(),scores.end(),totalityScorePoison);
        std::fill(selected.begin(),selected.end(),-91);std::fill(count.begin(),count.end(),-71);
    }
    /** @brief Check complete produced stages/selection and every inactive/unused guard.
     * @param expected Independently produced original-source result banks.
     */
    void expect(const IndexerTotalityBanks&expected)const
    {
        expectIndexerWords(prepared,expected.prepared,"prepared");expectIndexerWords(pools,expected.pools,"pools");
        expectIndexerWords(scores,expected.scores,"scores");EXPECT_EQ(selected,expected.selected);EXPECT_EQ(count,expected.count);
    }
};
/** @brief Original host-owned replay/projections and independent whole-indexer oracle. */
struct IndexerTotalityFixture
{
    static constexpr int origin=17,rotary=16,half=rotary/2;
    QwenSparseScalar representation;ISALevel isa;
    int rows,capacity,prefix,tokens,blocks,positions;
    QwenSparseReplay replay;
    std::vector<float>query,raw,partial,queryNorm,keyNorm,cosine,sine,initialPools;
    /** @brief Bind finite original sources at every tail and with inactive row capacity.
     * @param rep Original cast program; @param tail Original accepted-prefix remainder0..3.
     * @param liveRows Original live rows; @param program Already-admitted native dot program.
     */
    IndexerTotalityFixture(QwenSparseScalar rep,int tail,int liveRows,ISALevel program)
        :representation(rep),isa(program),rows(liveRows),capacity(rows+2),prefix(QwenSparseBlocks*4+tail),
         tokens(prefix+capacity),blocks((tokens+3)/4),positions(origin+tokens+1),
         replay{.prefix_tokens=prefix,.live_rows=rows,.position_origin=origin},
         query(totalityGuard+std::size_t(capacity)*512+totalitySuffix,totalityCanary),
         raw(totalityGuard+std::size_t(capacity)*128+totalitySuffix,totalityCanary),
         partial(totalityGuard+512+totalitySuffix,totalityCanary),
         queryNorm(totalityGuard+128+totalitySuffix,totalityCanary),keyNorm(queryNorm),
         cosine(totalityGuard+std::size_t(positions)*half+totalitySuffix,totalityCanary),sine(cosine),
         initialPools(totalityGuard+std::size_t(blocks)*128+totalitySuffix,totalityCanary)
    {
        for(std::size_t i=totalityGuard;i<query.size()-totalitySuffix;++i)query[i]=float(int((i*13)%97)-48)*0.015625f;
        for(std::size_t i=totalityGuard;i<raw.size()-totalitySuffix;++i)raw[i]=float(int((i*7)%59)-29)*0.03125f;
        for(std::size_t i=totalityGuard;i<partial.size()-totalitySuffix;++i)partial[i]=float(int((i*11)%61)-30)*0.03125f;
        for(int c=0;c<128;++c)
        {queryNorm[totalityGuard+c]=1.0f+float(c%17-8)*0.03125f;keyNorm[totalityGuard+c]=1.0f+float(c%13-6)*0.015625f;}
        // Supplied finite dyadic planes distinguish original positions and pairs;
        // identity coefficients would hide an ignored origin or wrong address.
        for(int position=0;position<positions;++position)for(int pair=0;pair<half;++pair)
        {
            cosine[totalityGuard+std::size_t(position)*half+pair]=float((position*17+pair*7)%251+1)*0.00390625f;
            sine[totalityGuard+std::size_t(position)*half+pair]=float((position*11+pair*13)%127-63)*0.0078125f;
        }
        for(int block=0;block<prefix/4;++block)for(int c=0;c<128;++c)
            initialPools[totalityGuard+std::size_t(block)*128+c]=float((block*17+c*13)%127-63)*0.015625f;
    }
    /** @brief Produce independent cast/RMS/RoPE/scores/frontiers from original sources once.
     * @param demand Original all-live or ranking-only authorship policy.
     * @param gold Same-envelope poisoned banks receiving independent expected words.
     * @details No production kernel, produced stage, native dot/fold or converter
     * supplies gold. Original old pools remain authored inputs; every new pool
     * uses prior-tail/current rows0..3 and the original block-start position.
     */
    void reference(QwenSparseQueryDemand demand,IndexerTotalityBanks&gold)const
    {
        const auto store=[&](const float*input,const float*weight,int position,float*output)
        {
            const auto normalized=normalize(input,weight,representation,1.0e-4f);
            const auto result=rotate(normalized,cosine.data()+totalityGuard+std::size_t(position)*half,
                sine.data()+totalityGuard+std::size_t(position)*half,rotary,representation);
            std::copy(result.begin(),result.end(),output);
        };
        for(int block=prefix/4;block<(prefix+rows)/4;++block)
        {
            std::array<float,128>pooled{};
            for(int c=0;c<128;++c)
            {
                float sum=0;
                for(int member=0;member<4;++member)
                {
                    const int logical=block*4+member;
                    const auto*source=logical<prefix?partial.data()+totalityGuard+std::size_t(logical-(prefix/4)*4)*128:
                        raw.data()+totalityGuard+std::size_t(logical-prefix)*128;
                    const auto value=test::representation(source[c],representation);
                    sum=member==0?value:add(sum,value);
                }
                pooled[c]=test::representation(multiply(sum,0.25f),representation);
            }
            store(pooled.data(),keyNorm.data()+totalityGuard,origin+block*4,gold.pools.data()+totalityGuard+std::size_t(block)*128);
        }
        for(int row=0;row<rows;++row)
        {
            const int visible=prefix+row+1,complete=visible/4,keep=std::min(complete,QwenSparseBlocks);
            if(demand==QwenSparseQueryDemand::AllLive||complete>QwenSparseBlocks)
                for(int head=0;head<4;++head)
                    store(query.data()+totalityGuard+std::size_t(row)*512+head*128,queryNorm.data()+totalityGuard,
                        origin+prefix+row,gold.prepared.data()+totalityGuard+std::size_t(row)*512+head*128);
            if(complete>QwenSparseBlocks)
                for(int block=0;block<complete;++block)
                {
                    float score=0;
                    for(int head=0;head<4;++head)
                    {
                        const auto value=rankingDot(gold.prepared.data()+totalityGuard+std::size_t(row)*512+head*128,
                            gold.pools.data()+totalityGuard+std::size_t(block)*128,isa);
                        ASSERT_TRUE(std::isfinite(value));score=add(score,std::max(0.0f,value));
                    }
                    gold.scores[totalityGuard+std::size_t(row)*blocks+block]=multiply(score,0.08838834764831844055f);
                }
            std::vector<int>chosen(complete);for(int block=0;block<complete;++block)chosen[block]=block;
            if(complete>QwenSparseBlocks)
            {
                const auto*score=gold.scores.data()+totalityGuard+std::size_t(row)*blocks;
                std::sort(chosen.begin(),chosen.end(),[&](int a,int b){return score[a]>score[b]||(score[a]==score[b]&&a<b);});
                chosen.resize(keep);std::sort(chosen.begin(),chosen.end());
            }
            for(int block=0;block<keep;++block)for(int member=0;member<4;++member)
                gold.selected[totalityGuard+std::size_t(row)*QwenSparseSelected+block*4+member]=chosen[block]*4+member;
            for(int tail=0;tail<visible%4;++tail)
                gold.selected[totalityGuard+std::size_t(row)*QwenSparseSelected+keep*4+tail]=complete*4+tail;
            gold.count[totalityGuard+row]=keep*4+visible%4;
        }
        for(int row=rows;row<capacity;++row)gold.count[totalityGuard+row]=0;
    }
    /** @brief Bind unchanged original pointers and one persistent output-bank topology.
     * @param output Persistent fixture banks, reset only between completed invocations.
     * @param demand Original authored query-work mode.
     * @return Original-input descriptor validated by the called public entrypoint.
     */
    QwenSparseIndexer bind(IndexerTotalityBanks&output,QwenSparseQueryDemand demand)const
    {
        return{.replay=&replay,.query=query.data()+totalityGuard,.raw_keys=raw.data()+totalityGuard,
            .partial_keys=partial.data()+totalityGuard,.pooled_keys=output.pools.data()+totalityGuard,
            .query_norm=queryNorm.data()+totalityGuard,.key_norm=keyNorm.data()+totalityGuard,
            .cosine=cosine.data()+totalityGuard,.sine=sine.data()+totalityGuard,
            .prepared_query=output.prepared.data()+totalityGuard,.scores=output.scores.data()+totalityGuard,
            .selected=output.selected.data()+totalityGuard,.selected_count=output.count.data()+totalityGuard,
            .row_capacity=capacity,.token_capacity=tokens,.block_capacity=blocks,.position_capacity=positions,
            .rotary_dim=rotary,.norm_epsilon=1.0e-4f,.representation=representation,.query_demand=demand};
    }
};
/** @brief Give each real format/tail integration cell its own unchanged timeout budget. */
class CPUQwenSparseIndexerTotality:public ::testing::TestWithParam<IndexerTotalityCase>{};
constexpr std::array<IndexerTotalityCase,12>indexerTotalityCases{{
    {QwenSparseScalar::FP32,0},{QwenSparseScalar::FP32,1},{QwenSparseScalar::FP32,2},{QwenSparseScalar::FP32,3},
    {QwenSparseScalar::FP16,0},{QwenSparseScalar::FP16,1},{QwenSparseScalar::FP16,2},{QwenSparseScalar::FP16,3},
    {QwenSparseScalar::BF16,0},{QwenSparseScalar::BF16,1},{QwenSparseScalar::BF16,2},{QwenSparseScalar::BF16,3}}};
/** @brief Give one closed format/tail tuple a stable independently filterable test name.
 * @param info Original parameter; @return Exact source-owned CTest filter suffix.
 */
std::string indexerTotalityName(const ::testing::TestParamInfo<IndexerTotalityCase>&info)
{
    const char*name=nullptr;
    switch(info.param.representation){case QwenSparseScalar::FP32:name="FP32";break;case QwenSparseScalar::FP16:name="FP16";break;case QwenSparseScalar::BF16:name="BF16";break;}
    if(!name)throw std::invalid_argument("Indexer totality case has no declared representation");
    return std::string(name)+"_Tail"+std::to_string(info.param.tail);
}
} // namespace
/** @test Every native row/team envelope matches original-source gold before outer-team completion. */
TEST_P(CPUQwenSparseIndexerTotality,NativeRowsAndWorkers)
{
    const auto isa=activeISALevel();::llaminar2::cpu::qsa_native::requireNativeProgram(isa);
    const TeamPolicy policy;const int maximum=policy.threads;ASSERT_GT(maximum,0);
    for(int rows:{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31,33})
    {
        const auto parameter=GetParam();IndexerTotalityFixture f(parameter.representation,parameter.tail,rows,isa);
        const auto query=f.query,raw=f.raw,partial=f.partial,queryNorm=f.queryNorm,keyNorm=f.keyNorm,cosine=f.cosine,sine=f.sine;
        for(auto demand:{QwenSparseQueryDemand::AllLive,QwenSparseQueryDemand::RankingOnly})
        {
            IndexerTotalityBanks expected(f.capacity,f.blocks,f.initialPools),actual(f.capacity,f.blocks,f.initialPools);
            ASSERT_NO_FATAL_FAILURE(f.reference(demand,expected));const auto p=f.bind(actual,demand);
            const auto verify=[&]()
            {
                actual.expect(expected);expectIndexerWords(f.query,query,"original query");expectIndexerWords(f.raw,raw,"original raw keys");
                expectIndexerWords(f.partial,partial,"original partial keys");expectIndexerWords(f.queryNorm,queryNorm,"original query norm");
                expectIndexerWords(f.keyNorm,keyNorm,"original key norm");expectIndexerWords(f.cosine,cosine,"original cosine");
                expectIndexerWords(f.sine,sine,"original sine");
                EXPECT_EQ(f.replay.prefix_tokens,f.prefix);EXPECT_EQ(f.replay.live_rows,rows);EXPECT_EQ(f.replay.position_origin,f.origin);
            };
            for(int workers=1;workers<=maximum;++workers)
            {
                SCOPED_TRACE(::testing::Message()<<"rep="<<int(parameter.representation)<<" tail="<<parameter.tail<<" M="<<rows<<" demand="<<int(demand)<<" workers="<<workers);
                omp_set_num_threads(workers);actual.reset(f.initialPools);cpuQwenSparseIndexer(p);verify();
                actual.reset(f.initialPools);int actualTeam=0;bool checkedInside=false;
#pragma omp parallel num_threads(workers)
                {
                    if(omp_get_thread_num()==0)actualTeam=omp_get_num_threads();
                    cpuQwenSparseIndexer(p);
                    // No test barrier may repair a missing kernel completion edge.
                    if(omp_get_thread_num()==0){verify();checkedInside=true;}
                }
                EXPECT_EQ(actualTeam,workers);EXPECT_TRUE(checkedInside);
            }
        }
    }
}
INSTANTIATE_TEST_SUITE_P(OriginalInputs,CPUQwenSparseIndexerTotality,
    ::testing::ValuesIn(indexerTotalityCases),indexerTotalityName);
} // namespace llaminar2::qwen4exp::test
