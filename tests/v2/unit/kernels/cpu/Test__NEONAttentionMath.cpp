/**
 * @file Test__NEONAttentionMath.cpp
 * @brief Original noncontiguous-cache arithmetic and admission proofs for native attention leaves.
 *
 * These tests certify register programs and synchronous borrowed envelopes, not
 * FA2 scheduling, cache leases, request state or serving. Grouped dots/updates
 * compare actual native single-row calls and independent rounded arithmetic;
 * no production row replay or decoded cache/head mirror is introduced.
 */
#include "kernels/cpu/attention/NEONAttentionMath.h"
#include "kernels/cpu/primitives/VectorPrimitives.h"
#include "utils/CPUExplicitRoundingOracle.h"
#include "utils/CPUFeatures.h"
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <csignal>
#include <cstdio>
#include <exception>
#include <bit>
#include <cstring>
#include <vector>

namespace llaminar2::cpu::neon_attention
{
/** @brief Focused access to the private row programs' own fatal guards. */
struct AdmittedRowsTestAccess
{
    /** @brief Run the private integer row program with an explicit width. */
    static std::int32_t integerDotRows(const Q16ReductionBound&bound,const std::int16_t*query,const std::int16_t*key,int width)
    { return AdmittedRows::integerDotRows<1>(bound,query,{key},width)[0]; }
};
} // namespace llaminar2::cpu::neon_attention

namespace llaminar2::test
{
namespace
{
namespace native=cpu::neon_attention;

/// Whether code outside the owning kernel can name a private floating row program.
template<class Rows>concept ReachesFloatingRows=requires(const float*query,std::array<const float*,1>keys)
{Rows::template dotRows<native::Storage::FP32,1>(query,keys,1);};
/// Whether code outside the owning kernel can name the private weighted update.
template<class Rows>concept ReachesUpdateRows=requires(float*output,std::array<const float*,1>values,std::array<float,1>weights)
{Rows::template accumulateRows<native::Storage::FP32,1>(output,values,weights,1);};
/// Whether code outside the owning kernel can name the private integer row program.
template<class Rows>concept ReachesIntegerRows=requires(const cpu::Q16ReductionBound&bound,const std::int16_t*query,
    std::array<const std::int16_t*,1>keys){Rows::template integerDotRows<1>(bound,query,keys,1);};
/// Positive control: the checked public overload is reachable through the same form.
template<class Rows>concept ReachesCheckedDot=requires(std::span<const float>query,std::array<std::span<const float>,1>keys)
{Rows::template dot<native::Storage::FP32,1>(query,keys);};
static_assert(ReachesCheckedDot<native::AdmittedRows>);
// The row programs that trust admitted geometry are not part of the public surface.
static_assert(!ReachesFloatingRows<native::AdmittedRows>&&!ReachesUpdateRows<native::AdmittedRows>&&
    !ReachesIntegerRows<native::AdmittedRows>);
using native::Storage;
/** @brief Authenticate the complete compiler envelope before direct native leaf tests. */
class NEONAttentionMath : public ::testing::Test
{
protected:
    /** @brief Reject unavailable compiler/runtime instructions rather than skipping a leaf. */
    static void SetUpTestSuite(){(void)activeISALevel();}
};
/** @brief Encode original cache storage before invocation.
 * @tparam Format Original physical encoding; @param value Diagnostic source value.
 * @return Original physical word.
 */
template<Storage Format>native::Unit<Format> encode(float value)
{
    if constexpr(Format==Storage::FP32)return value;
    else if constexpr(Format==Storage::FP16)return fp32_to_fp16(value);
    else if constexpr(Format==Storage::I16)return std::int16_t(value*113.0f);
    else{const auto bits=std::bit_cast<std::uint32_t>(value);return std::uint16_t((bits+0x7fffu+((bits>>16)&1))>>16);}
}
/** @brief Independently decode original storage without the production reader.
 * @tparam Format Original physical encoding; @param value Original word.
 * @return Canonical arithmetic value.
 */
template<Storage Format>float decode(native::Unit<Format>value)
{
    if constexpr(Format==Storage::FP32||Format==Storage::I16)return float(value);
    else if constexpr(Format==Storage::FP16)return fp16_to_fp32(value);
    else return std::bit_cast<float>(std::uint32_t(value)<<16);
}
/** @brief Independent memory-rounded original native dot tree.
 * @tparam Format Original cache encoding; @param query Original query; @param key Original key.
 * @return Same-backend two-chain/pair-fold result.
 */
template<Storage Format>float reference(std::span<const float>query,std::span<const native::Unit<Format>>key)
{
    float lo[4]{},hi[4]{};std::size_t col=0;
    for(;col+8<=query.size();col+=8)for(int lane=0;lane<4;++lane)
    {
        volatile float a=std::fma(query[col+lane],decode<Format>(key[col+lane]),lo[lane]);lo[lane]=a;
        volatile float b=std::fma(query[col+4+lane],decode<Format>(key[col+4+lane]),hi[lane]);hi[lane]=b;
    }
    float sum[4];for(int lane=0;lane<4;++lane)sum[lane]=memoryRoundedAdd(lo[lane],hi[lane]);
    float total=memoryRoundedAdd(memoryRoundedAdd(sum[0],sum[1]),memoryRoundedAdd(sum[2],sum[3]));
    for(;col<query.size();++col){volatile float next=std::fma(query[col],decode<Format>(key[col]),total);total=next;}
    return total;
}
/** @brief Cover actual independent cache allocations without deriving one row pointer from another.
 * @tparam Format Original physical cache encoding; @tparam Rows Native register row count.
 */
template<Storage Format,std::size_t Rows>void floating()
{
    std::vector<int> widths;for(int n=0;n<=33;++n)widths.push_back(n);
    for(int n:{63,64,65,127,128,129,255,256,257})widths.push_back(n);
    for(int n:widths)for(int seed=0;seed<16;++seed)
    {
        std::vector<float> query(n);for(int i=0;i<n;++i)query[i]=float((i*73+seed*37)%257-128)*0.01731f;
        std::array<std::vector<native::Unit<Format>>,Rows> banks;
        std::array<std::span<const native::Unit<Format>>,Rows> keys;
        for(std::size_t row=0;row<Rows;++row)
        {
            banks[row].resize(n+2);
            std::memset(banks[row].data(),0x5d,banks[row].size()*sizeof(native::Unit<Format>));
            for(int i=0;i<n;++i)banks[row][i]=encode<Format>(float(int((std::size_t(i)*31+row*71+seed*13)%257)-128)*0.13731f);
            keys[row]=std::span(banks[row]).first(n);
        }
        const auto original=banks;
        const auto grouped=native::dot<Format,Rows>(query,keys);
        for(std::size_t row=0;row<Rows;++row)
        {
            const auto serial=native::dot<Format,1>(query,{keys[row]});
            EXPECT_EQ(std::bit_cast<std::uint32_t>(grouped[row]),std::bit_cast<std::uint32_t>(serial[0]));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(grouped[row]),std::bit_cast<std::uint32_t>(reference<Format>(query,keys[row])));
            if constexpr(Format==Storage::FP32)
                EXPECT_EQ(std::bit_cast<std::uint32_t>(grouped[row]),std::bit_cast<std::uint32_t>(primitives::vec_dot(query.data(),keys[row].data(),n)));
        }
        std::array<float,Rows> weights;for(std::size_t row=0;row<Rows;++row)weights[row]=float(int(row)-2)*0.1731f;
        std::vector<float> expected(n+2,919.25f),serial=expected,actual=expected;
        for(int i=0;i<n;++i)expected[i]=serial[i]=actual[i]=float((i*17)%73-36)*0.013731f;
        for(int i=0;i<n;++i)for(std::size_t row=0;row<Rows;++row)
        {volatile float next=std::fma(decode<Format>(keys[row][i]),weights[row],expected[i]);expected[i]=next;}
        for(std::size_t row=0;row<Rows;++row)native::accumulate<Format,1>(std::span(serial).first(n),{keys[row]},{weights[row]});
        native::accumulate<Format,Rows>(std::span(actual).first(n),keys,weights);
        EXPECT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),0);
        EXPECT_EQ(std::memcmp(actual.data(),serial.data(),actual.size()*sizeof(float)),0);
        for(std::size_t row=0;row<Rows;++row)EXPECT_EQ(std::memcmp(banks[row].data(),original[row].data(),banks[row].size()*sizeof(native::Unit<Format>)),0);
    }
}
/** @test Shared native cache-row arithmetic equals actual M1 and independent original-order programs. */
TEST_F(NEONAttentionMath, FloatingOriginalCacheRowsRetainSerialBytes)
{
    floating<Storage::FP32,1>();floating<Storage::FP32,2>();floating<Storage::FP32,4>();
    floating<Storage::FP16,1>();floating<Storage::FP16,2>();floating<Storage::FP16,4>();
    floating<Storage::BF16,1>();floating<Storage::BF16,2>();floating<Storage::BF16,4>();
    floating<Storage::I16,1>();floating<Storage::I16,2>();floating<Storage::I16,4>();
}
/** @brief Verify exact int32 arithmetic under the typed reduction bound, including cancellation.
 * @tparam Rows Native one/two/four-row register program.
 */
template<std::size_t Rows>void integer()
{
    for(int n:{0,1,2,3,4,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257})
        for(int seed=0;seed<16;++seed)
        {
            const auto bound=cpu::Q16ReductionBound::forTerms(std::max(n,1));
            const int limit=bound.queryLimit();
            std::vector<std::int16_t> query(n);
            std::array<std::vector<std::int16_t>,Rows> banks;
            std::array<std::span<const std::int16_t>,Rows> keys;
            std::array<std::int64_t,Rows> expected{};
            for(int i=0;i<n;++i)query[i]=std::int16_t((i*131+seed*37)%(2*limit+1)-limit);
            for(std::size_t row=0;row<Rows;++row)
            {
                banks[row].resize(n);for(int i=0;i<n;++i){banks[row][i]=std::int16_t(int((std::size_t(i)*71+row*31+seed)%65536)-32768);expected[row]+=std::int64_t(query[i])*banks[row][i];}
                keys[row]=banks[row];
            }
            const auto actual=native::integerDot<Rows>(bound,query,keys);
            for(std::size_t row=0;row<Rows;++row){EXPECT_EQ(actual[row],expected[row]);EXPECT_EQ(actual[row],native::integerDot<1>(bound,query,{keys[row]})[0]);}
        }
    const auto bound=cpu::Q16ReductionBound::forTerms(8);
    std::array<std::int16_t,8> query;query.fill(std::int16_t(bound.queryLimit()));
    std::array<std::int16_t,8> key{32767,32767,32767,32767,-32767,-32767,-32767,-32767};
    std::array<std::span<const std::int16_t>,Rows> keys;keys.fill(key);
    for(auto value:native::integerDot<Rows>(bound,query,keys))EXPECT_EQ(value,0);
}
/** @test Int32 accumulation under the typed bound equals the exact int64 sum for every row program. */
TEST_F(NEONAttentionMath, IntegerOriginalRowsAccumulateExactlyUnderTheTypedBound)
{integer<1>();integer<2>();integer<4>();}

/**
 * @test Worst-case witnesses at the bound's closed form: every sign pattern of
 * |q| = limit and |k| = 32768 stays exact in int32 for head widths 32..256,
 * including all-negative keys at the full -32768 magnitude.
 */
TEST_F(NEONAttentionMath, Q16BoundWorstCaseWitnessesStayExact)
{
    for(int terms:{32,64,128,256})
    {
        const auto bound=cpu::Q16ReductionBound::forTerms(terms);
        ASSERT_LE(bound.worstMagnitude(),std::int64_t(std::numeric_limits<std::int32_t>::max()));
        for(int pattern=0;pattern<4;++pattern)
        {
            std::vector<std::int16_t> query(terms),key(terms);
            std::int64_t expected=0;
            for(int i=0;i<terms;++i)
            {
                const int q_sign=pattern&1?((i&1)?-1:1):1, k_sign=pattern&2?1:-1;
                query[i]=std::int16_t(q_sign*bound.queryLimit());
                key[i]=k_sign<0?std::int16_t(-32768):std::int16_t(32767);
                expected+=std::int64_t(query[i])*key[i];
            }
            const std::array<std::span<const std::int16_t>,4> keys{key,key,key,key};
            for(const auto value:native::integerDot<4>(bound,query,keys))EXPECT_EQ(value,expected) << terms << "/" << pattern;
        }
    }
    // T=256 at the forbidden limit 256 would reach 2^31: the bound forbids it.
    EXPECT_EQ(cpu::Q16ReductionBound::forTerms(256).queryLimit(),255);
    EXPECT_GT(std::int64_t(256)*256*32768,std::int64_t(std::numeric_limits<std::int32_t>::max()));
}

/** @test The bound rejects wider reductions and is fatal for query codes beyond its limit. */
TEST_F(NEONAttentionMath, Q16BoundRejectsUnprovenReductions)
{
    const auto bound=cpu::Q16ReductionBound::forTerms(64);
    std::vector<std::int16_t> query(65,1),key(65,1);
    EXPECT_THROW((native::integerDot<1>(bound,query,{std::span<const std::int16_t>(key)})),std::invalid_argument);
    query.resize(64);key.resize(64);
    for(const int position:{0,7,63})
    {
        auto beyond=query;beyond[position]=std::int16_t(bound.queryLimit()+1);
        EXPECT_DEATH((native::integerDot<1>(bound,beyond,{std::span<const std::int16_t>(key)})),"outside its int32 reduction bound");
        beyond[position]=std::int16_t(-32768);
        EXPECT_DEATH((native::integerDot<1>(bound,beyond,{std::span<const std::int16_t>(key)})),"outside its int32 reduction bound");
    }
}

/**
 * @test Tail-bearing adversarial input: n = 65 so one code is handled by the
 * scalar tail after a vector fold already at 2,113,929,216. The invalid tail
 * code must be rejected before any int32 tail arithmetic (which would exceed
 * INT32_MAX), for every row program and for invalid codes in the vector part.
 */
TEST_F(NEONAttentionMath, Q16BoundRejectsInvalidTailCodesBeforeTailArithmetic)
{
    GTEST_FLAG_SET(death_test_style,"fast");
    const auto bound=cpu::Q16ReductionBound::forTerms(65);
    ASSERT_EQ(bound.queryLimit(),1008);
    std::vector<std::int16_t> query(65,std::int16_t(-1008)),key(65,std::int16_t(-32768));
    const std::span<const std::int16_t> k(key);
    const std::array<std::span<const std::int16_t>,1> keys1{k};
    const std::array<std::span<const std::int16_t>,2> keys2{k,k};
    const std::array<std::span<const std::int16_t>,4> keys4{k,k,k,k};
    // Valid: 65 * 1008 * 32768 = 2,146,959,360 <= INT32_MAX, exact in int32 for every row program.
    for(const auto value:native::integerDot<1>(bound,query,keys1))EXPECT_EQ(value,std::int32_t(2146959360));
    for(const auto value:native::integerDot<2>(bound,query,keys2))EXPECT_EQ(value,std::int32_t(2146959360));
    for(const auto value:native::integerDot<4>(bound,query,keys4))EXPECT_EQ(value,std::int32_t(2146959360));
    // Explicit abort from the diagnostic: an escaped exception or terminate exits 77.
    const auto rejects=[&](const std::vector<std::int16_t>&invalid,const char*where){
        const auto expect=[&]<std::size_t Rows>(const std::array<std::span<const std::int16_t>,Rows>&keys){
            EXPECT_EXIT(({
                std::set_terminate([]{std::fputs("escaped exception or terminate\n",stderr);std::_Exit(77);});
                (void)native::integerDot<Rows>(bound,invalid,keys);
                std::_Exit(0);
            }),::testing::KilledBySignal(SIGABRT),"Q16 query code outside its int32 reduction bound") << where << " rows=" << Rows;
        };
        expect(keys1);expect(keys2);expect(keys4);
    };
    for(const std::int16_t bad:{std::int16_t(-32768),std::int16_t(1009),std::int16_t(-1009),std::int16_t(32767)})
    {
        auto invalid=query;invalid[64]=bad;
        rejects(invalid,"tail");
        invalid[64]=std::int16_t(-1008);invalid[3]=bad;
        rejects(invalid,"vector");
    }
}

/** @test Query quantization retains max/limit, inverse, threshold, RNE and bounded original tails. */
TEST_F(NEONAttentionMath, OriginalQueryQuantizationProgram)
{
    for(int n=0;n<=257;++n)for(int limit:{1,127,255,1023,2047,32767})for(float magnitude:{0.0f,1e-12f,std::nextafter(1e-12f,1.0f),0.01731f,113.75f})
    {
        std::vector<float> source(n);float maximum=0.0f;
        for(int i=0;i<n;++i){source[i]=float((i*73)%257-128)*magnitude;maximum=std::max(maximum,std::abs(source[i]));}
        std::vector<std::int16_t> actual(n+2,0x5151),expected=actual;
        float scale=0.0f;
        if(maximum>1e-12f)
        {
            volatile float retained_scale=maximum/float(limit),inverse=1.0f/float(retained_scale);scale=retained_scale;
            for(int i=0;i<n;++i)expected[i]=std::int16_t(std::clamp(int(std::lrint(memoryRoundedMultiply(source[i],inverse))),-limit,limit));
        }
        else std::fill_n(expected.begin(),n,std::int16_t(0));
        const float result=native::quantizeQuery(source,std::span(actual).first(n),limit);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(result),std::bit_cast<std::uint32_t>(scale));
        EXPECT_EQ(actual,expected);
    }
}
/** @test Malformed original geometry, cache/policy aliases and invalid data reject before output writes. */
TEST_F(NEONAttentionMath, OriginalStorageAndDomainAdmission)
{
    std::array<float,8> query{1,2,3,4,5,6,7,8},value=query,output=query;const auto original=output;
    std::array<std::span<const float>,4> values;values.fill(value);
    const std::array<float,4> weights{1,2,3,4};
    EXPECT_THROW((native::dot<Storage::FP32,4>(query,{std::span(value).first(7),value,value,value})),std::invalid_argument);
    EXPECT_THROW((native::accumulate<Storage::FP32,4>(std::span(output).first(7),values,weights)),std::invalid_argument);
    values[3]=output;
    EXPECT_THROW((native::accumulate<Storage::FP32,4>(output,values,weights)),std::invalid_argument);
    values.fill(std::span(value).first(4));std::array<float,4> aliased_weights{1,2,3,4};
    EXPECT_THROW((native::accumulate<Storage::FP32,4>(aliased_weights,values,aliased_weights)),std::invalid_argument);
    EXPECT_EQ(output,original);EXPECT_EQ(aliased_weights,weights);
    std::array<std::int16_t,8> destination;destination.fill(0x5151);const auto initial=destination;
    EXPECT_THROW(native::quantizeQuery(query,destination,0),std::invalid_argument);
    EXPECT_THROW(native::quantizeQuery(query,std::span(destination).first(7),127),std::invalid_argument);
    for(int index=0;index<8;++index)
    {
        auto invalid=query;invalid[index]=std::numeric_limits<float>::quiet_NaN();
        EXPECT_THROW(native::quantizeQuery(invalid,destination,127),std::domain_error);EXPECT_EQ(destination,initial);
    }
}
/**
 * @brief Prove one nonfinite-weight rejection: SIGABRT from the diagnostic, not an
 * escaped exception or terminate, with no byte of the child-visible numerator written.
 * @tparam Rows One/two/four-row program. @param numerator Shared (MAP_SHARED) numerator.
 */
template<std::size_t Rows>void nonfiniteIsFatalBeforeWrites(std::span<float>numerator)
{
    std::array<float,8> value{1,2,3,4,5,6,7,8};
    std::array<std::span<const float>,Rows> values;values.fill(value);
    std::vector<float> before(numerator.begin(),numerator.end());
    for(const float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()})
        for(std::size_t row=0;row<Rows;++row)
        {
            std::array<float,Rows> weights;weights.fill(0.5f);weights[row]=bad;
            EXPECT_EXIT(({
                // An escaped exception or terminate would exit 77, not SIGABRT.
                std::set_terminate([]{std::fputs("escaped exception or terminate\n",stderr);std::_Exit(77);});
                native::accumulate<Storage::FP32,Rows>(numerator,values,weights);
                std::_Exit(0);
            }),::testing::KilledBySignal(SIGABRT),"consumed invalid nonfinite attention weight") << "rows=" << Rows << " row=" << row;
            EXPECT_EQ(std::memcmp(numerator.data(),before.data(),before.size()*sizeof(float)),0) << "rows=" << Rows << " row=" << row;
        }
}
} // namespace

/**
 * @test A nonfinite weight is fatal before any numerator write, for every row
 * program. The numerator is shared with the death-test child, and a
 * write-before-reject negative control proves the witness sees child writes.
 */
/** @test The private integer row program rejects a width beyond its bound's term count, and any negative width. */
TEST_F(NEONAttentionMath, PrivateIntegerRowsRejectUnprovenWidths)
{
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    const auto bound=cpu::Q16ReductionBound::forTerms(8);
    std::vector<std::int16_t> query(16,1),key(16,1);
    EXPECT_EQ(native::AdmittedRowsTestAccess::integerDotRows(bound,query.data(),key.data(),8),8);
    EXPECT_EQ(native::AdmittedRowsTestAccess::integerDotRows(bound,query.data(),key.data(),0),0);
    EXPECT_DEATH((void)native::AdmittedRowsTestAccess::integerDotRows(bound,query.data(),key.data(),9),
        "reduction width outside its int32 term bound");
    EXPECT_DEATH((void)native::AdmittedRowsTestAccess::integerDotRows(bound,query.data(),key.data(),-1),
        "reduction width outside its int32 term bound");
}

TEST_F(NEONAttentionMath, NonfiniteWeightIsFatalBeforeAnySharedNumeratorWrite)
{
    GTEST_FLAG_SET(death_test_style,"fast"); // The child must inherit the shared mapping.
    void*mapping=mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
    ASSERT_NE(mapping,MAP_FAILED);
    std::span<float> numerator(static_cast<float*>(mapping),8);
    for(std::size_t i=0;i<numerator.size();++i)numerator[i]=float(i)*0.25f-1.0f;
    const std::vector<float> original(numerator.begin(),numerator.end());

    // Negative control: a child that writes first and then rejects is detected.
    EXPECT_EXIT(({numerator[3]=99.0f;native::rejectAttentionData("nonfinite attention weight");}),
        ::testing::KilledBySignal(SIGABRT),"consumed invalid nonfinite attention weight");
    EXPECT_EQ(numerator[3],99.0f);
    numerator[3]=original[3];

    nonfiniteIsFatalBeforeWrites<1>(numerator);
    nonfiniteIsFatalBeforeWrites<2>(numerator);
    nonfiniteIsFatalBeforeWrites<4>(numerator);
    EXPECT_EQ(std::memcmp(numerator.data(),original.data(),original.size()*sizeof(float)),0);
    munmap(mapping,4096);
}
} // namespace llaminar2::test
