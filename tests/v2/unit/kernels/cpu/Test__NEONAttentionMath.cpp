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
#include <bit>
#include <cstring>
#include <vector>

namespace llaminar2::test
{
namespace
{
namespace native=cpu::neon_attention;
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
/** @brief Verify exact widened integer arithmetic, including cancellation after large partials.
 * @tparam Rows Native one/two/four-row register program.
 */
template<std::size_t Rows>void integer()
{
    for(int n:{0,1,2,3,4,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257})
        for(int seed=0;seed<16;++seed)
        {
            std::vector<std::int16_t> query(n);
            std::array<std::vector<std::int16_t>,Rows> banks;
            std::array<std::span<const std::int16_t>,Rows> keys;
            std::array<std::int64_t,Rows> expected{};
            for(int i=0;i<n;++i)query[i]=std::int16_t((i*131+seed*37)%2047-1023);
            for(std::size_t row=0;row<Rows;++row)
            {
                banks[row].resize(n);for(int i=0;i<n;++i){banks[row][i]=std::int16_t(int((std::size_t(i)*71+row*31+seed)%65536)-32768);expected[row]+=std::int64_t(query[i])*banks[row][i];}
                keys[row]=banks[row];
            }
            const bool fits=std::all_of(expected.begin(),expected.end(),[](auto x){return x>=std::numeric_limits<std::int32_t>::min()&&x<=std::numeric_limits<std::int32_t>::max();});
            if(!fits){EXPECT_THROW((native::integerDot<Rows>(query,keys)),std::overflow_error);continue;}
            const auto actual=native::integerDot<Rows>(query,keys);
            for(std::size_t row=0;row<Rows;++row){EXPECT_EQ(actual[row],expected[row]);EXPECT_EQ(actual[row],native::integerDot<1>(query,{keys[row]})[0]);}
        }
    std::array<std::int16_t,8> query;query.fill(32767);
    std::array<std::int16_t,8> key{32767,32767,32767,32767,-32767,-32767,-32767,-32767};
    std::array<std::span<const std::int16_t>,Rows> keys;keys.fill(key);
    for(auto value:native::integerDot<Rows>(query,keys))EXPECT_EQ(value,0);
    query.fill(-32768);key.fill(-32768);
    EXPECT_THROW((native::integerDot<Rows>(query,keys)),std::overflow_error);
}
/** @test Exact int64 intermediates prevent wrapped/saturated native grouped integer dots. */
TEST_F(NEONAttentionMath, IntegerOriginalRowsWidenBeforeAccumulation)
{integer<1>();integer<2>();integer<4>();}

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
} // namespace
} // namespace llaminar2::test
