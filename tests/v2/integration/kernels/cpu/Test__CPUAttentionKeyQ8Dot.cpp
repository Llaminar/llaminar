/**
 * @file Test__CPUAttentionKeyQ8Dot.cpp
 * @brief Original AQ8 register dots versus the existing declared-ISA decode/dot components.
 *
 * Whole decoded heads exist only in this diagnostic oracle. Production tiles
 * consume immutable compressed blocks/anchors, share fragments across queries,
 * and must retain the same ISA's serial attention fold. Different native ISA
 * programs are compared separately, not treated as one universal byte oracle.
 * NEON calls its production FP32 dot helper; x86 transcribes the existing class
 * loops. The old ARM AQ8 adapter was unreachable: this leaf proof does not
 * qualify its newly compiled entry, cache/arena leases or full FA2 execution.
 */
#include "kernels/cpu/attention/CPUAttentionKeyQ8Dot.h"
#include "../../../utils/CPUAttentionFP32DotReference.h"
#include <gtest/gtest.h>
#include <bit>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <type_traits>
#include <vector>

namespace
{
using namespace llaminar2;
namespace aq8=cpu::attention_key_q8;

/** @return All genuinely compiled/native codec programs on this host.
 * @throws std::runtime_error If this explicit native suite has no arithmetic program.
 */
std::vector<aq8::ISA> dotISAs()
{
    std::vector<aq8::ISA> result;
#if LLAMINAR_COMPILED_WITH_NEON
    if(cpu_supports_neon())result.push_back(aq8::ISA::NEON);
#endif
#if defined(__AVX2__)
    if(cpu_supports_avx2())result.push_back(aq8::ISA::AVX2);
#endif
#if defined(__AVX512F__)
    if(cpu_supports_avx512())result.push_back(aq8::ISA::AVX512);
#endif
    if(result.empty())throw std::runtime_error("AQ8 dot conformance requires an admitted native ISA; an empty sweep is not a pass");
    return result;
}

/** @brief Sweep original strided query banks, all physical query counts and native ISA programs.
 * @tparam D Complete head width; @tparam Rows Bound shared-query width.
 * @details Prefix/stride gaps remain in the original bank; no production input
 * is repacked. Separate actual M1 and old decode/dot oracles check every word.
 */
template<int D,std::size_t Rows>
void checkOriginalDots()
{
    constexpr std::size_t stride=D+9,prefix=7,count=33;
    std::vector<float> original(prefix+count*stride+11,-91.25f);
    std::array<float,D> anchor{},decoded{};
    for(unsigned seed=1;seed<=12;++seed)
    {
        for(std::size_t row=0;row<count;++row)
            for(int c=0;c<D;++c)original[prefix+row*stride+c]=float(int((row*31+c*17+seed)%113)-56)*0.003f;
        for(int c=0;c<D;++c)anchor[c]=float(int((c*7+seed)%29)-14)*0.125f;
        for(const int exponent:{-100,-30,-1,0,12,40})
        {
            AttentionKeyQ8Block<D> block{};block.quadratic_scale=std::ldexp(1.0f,exponent);
            for(int c=0;c<D;++c)block.codes[c]=std::int8_t(int((c*17+seed*23)%255)-127);
            const auto saved=block;const auto saved_anchor=anchor;const auto saved_queries=original;
            for(const auto isa:dotISAs())
            {
                aq8::dequantize<D>(block,anchor,decoded,isa);
                for(std::size_t first=0;first+Rows<=count;first+=Rows)
                {
                    std::array<std::span<const float>,Rows> queries;
                    for(std::size_t row=0;row<Rows;++row)queries[row]={original.data()+prefix+(first+row)*stride,D};
                    const auto tile=aq8::QueryDotTile<D,Rows>::bind(queries,anchor,isa);
                    ASSERT_EQ(tile.isa(),isa);
                    const auto actual=tile.dot(block);
                    for(std::size_t row=0;row<Rows;++row)
                    {
                        const float serial=aq8::QueryDotTile<D,1>::bind({queries[row]},anchor,isa).dot(block)[0];
                        const float old=test::cpuAttentionFP32DotReference<D>(queries[row],decoded,isa);
                        ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[row]),std::bit_cast<std::uint32_t>(serial));
                        ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[row]),std::bit_cast<std::uint32_t>(old))
                            <<"D="<<D<<" rows="<<Rows<<" seed="<<seed<<" exponent="<<exponent<<" first="<<first;
                    }
                }
            }
            EXPECT_EQ(std::memcmp(&block,&saved,sizeof(block)),0);
            EXPECT_EQ(std::memcmp(anchor.data(),saved_anchor.data(),sizeof(anchor)),0);
            EXPECT_EQ(std::memcmp(original.data(),saved_queries.data(),original.size()*sizeof(float)),0);
        }
    }
}

/** @brief Retain independent memory-rounded decode edges and non-power-of-two published scales.
 * @tparam D Complete head dimension; @tparam Rows Original independent query count.
 * @details The RN witness separates the required two products/add from both
 * reassociation and an outer FMA. Published keys are independently decoded;
 * an old codec/compiler difference must not redefine the expected program.
 */
template<int D,std::size_t Rows>
void checkDecodeRounding()
{
    std::array<std::array<float,D>,Rows> bank{};
    std::array<std::span<const float>,Rows> queries;
    std::array<float,D> anchor{},source{},old_decoded{};
    anchor[0]=-9.0f;source[0]=std::ldexp(1.0f,-19);source[1]=16130.25f;
    for(std::size_t row=0;row<Rows;++row){bank[row][0]=1.0f;queries[row]=bank[row];}
    for(const auto isa:dotISAs())
    {
        AttentionKeyQ8Block<D> witness{};
        witness.quadratic_scale=std::bit_cast<float>(0x3f800001u);witness.codes[0]=3;
        volatile float code=3.0f,scale=witness.quadratic_scale;
        volatile float product=scale*code;
        volatile float residual=product*code;
        volatile float correct=residual+anchor[0];
        volatile float code_squared=code*code;
        volatile float wrong_product=scale*code_squared;
        volatile float reassociated=wrong_product+anchor[0];
        const float fused=std::fma(float(product),float(code),anchor[0]);
        ASSERT_EQ(std::bit_cast<std::uint32_t>(float(correct)),0x36000000u);
        ASSERT_EQ(std::bit_cast<std::uint32_t>(float(reassociated)),0x35800000u);
        ASSERT_EQ(std::bit_cast<std::uint32_t>(fused),0x35c00000u);
        ASSERT_NE(float(correct),float(reassociated));ASSERT_NE(float(correct),fused);
        const auto program=aq8::QueryDotTile<D,Rows>::bind(queries,anchor,isa);
        for(const float value:program.dot(witness))
            EXPECT_EQ(std::bit_cast<std::uint32_t>(value),0x36000000u)<<"D="<<D<<" Rows="<<Rows;

        AttentionKeyQ8Block<D> published{};
        aq8::quantize<D>(source,anchor,published,isa);
        ASSERT_NE(std::bit_cast<std::uint32_t>(published.quadratic_scale)&0x007fffffu,0u);
        ASSERT_EQ(published.codes[0],3);
        volatile float published_code=float(published.codes[0]);
        volatile float published_product=published.quadratic_scale*published_code;
        volatile float published_residual=published_product*std::abs(float(published_code));
        volatile float expected=published_residual+anchor[0];
        aq8::dequantize<D>(published,anchor,old_decoded,isa);
        for(const float value:program.dot(published))
            EXPECT_EQ(std::bit_cast<std::uint32_t>(value),std::bit_cast<std::uint32_t>(float(expected)))
                <<"D="<<D<<" Rows="<<Rows<<" old_codec="<<std::hex<<std::bit_cast<std::uint32_t>(old_decoded[0]);
    }
}
} // namespace

/** @test Every admitted head/query tile retains original serial ISA words without a head mirror. */
TEST(CPUAttentionKeyQ8Dot, OriginalQueriesMatchSerialAndPreviousPipeline)
{
    checkOriginalDots<64,1>();checkOriginalDots<64,2>();checkOriginalDots<64,3>();checkOriginalDots<64,4>();
    checkOriginalDots<128,1>();checkOriginalDots<128,2>();checkOriginalDots<128,3>();checkOriginalDots<128,4>();
    checkOriginalDots<256,1>();checkOriginalDots<256,2>();checkOriginalDots<256,3>();checkOriginalDots<256,4>();
}

/** @test Decode rounding cannot be reassociated or fused; real published scales remain authoritative. */
TEST(CPUAttentionKeyQ8Dot, SeparateDecodeRoundingAndPublishedScales)
{
    checkDecodeRounding<64,1>();checkDecodeRounding<64,2>();checkDecodeRounding<64,3>();checkDecodeRounding<64,4>();
    checkDecodeRounding<128,1>();checkDecodeRounding<128,2>();checkDecodeRounding<128,3>();checkDecodeRounding<128,4>();
    checkDecodeRounding<256,1>();checkDecodeRounding<256,2>();checkDecodeRounding<256,3>();checkDecodeRounding<256,4>();
}

/** @test The actual Perf__ authentication protocol rejects no-op/partial and nonfinite banks. */
TEST(CPUAttentionKeyQ8Dot, PoisonedAuthenticationRejectsNoOpAndPartialWriters)
{
    std::array<float,4> expected{},actual{};
    test::poisonAttentionDotOutput(actual);
    EXPECT_FALSE(test::authenticateAttentionDotOutput(actual,expected));
    actual[0]=expected[0];
    EXPECT_FALSE(test::authenticateAttentionDotOutput(actual,expected));
    actual=expected;
    EXPECT_TRUE(test::authenticateAttentionDotOutput(actual,expected));
    actual[1]=std::numeric_limits<float>::quiet_NaN();expected[1]=actual[1];
    EXPECT_FALSE(test::authenticateAttentionDotOutput(actual,expected));
    EXPECT_FALSE(test::authenticateAttentionDotOutput({},{}));
    EXPECT_FALSE(test::authenticateAttentionDotOutput(actual,std::span<const float>(expected).first(3)));
}

/** @test An unbound/foreign program cannot be represented as a successful dot tile. */
TEST(CPUAttentionKeyQ8Dot, OriginalHeadGeometryAndUnavailableProgramsReject)
{
    using Tile=aq8::QueryDotTile<64,2>;
    static_assert(!std::is_default_constructible_v<Tile>);
    std::array<float,64> query{},anchor{};
    const std::array<std::span<const float>,2> valid{query,query};
    EXPECT_THROW((Tile::bind(valid,{})),std::invalid_argument);
    EXPECT_THROW((Tile::bind({query,std::span<const float>(query).first(63)},anchor)),std::invalid_argument);
    EXPECT_THROW((Tile::bind(valid,anchor,static_cast<aq8::ISA>(255))),std::invalid_argument);
    const auto isas=dotISAs();
    ASSERT_FALSE(isas.empty())<<"this explicit native suite requires an admitted native ISA";
    for(const auto isa:isas)EXPECT_NO_THROW((Tile::bind(valid,anchor,isa)));
}
