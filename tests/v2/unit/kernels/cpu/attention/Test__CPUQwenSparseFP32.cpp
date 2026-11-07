/**
 * @file Test__CPUQwenSparseFP32.cpp
 * @brief Original QSA native FP32 trees, independent FMA references and ISA admission.
 *
 * These are exact shared arithmetic-leaf proofs, not execution of the full
 * sparse indexer/cache/factory/replay path. QSA's butterfly is explicitly kept
 * distinct from the generic NEON attention fold; no unavailable-ISA substitute
 * or cross-ISA output-proximity oracle is used.
 */
#include "kernels/cpu/attention/CPUQwenSparseFP32Program.h"
#include "utils/CPUExplicitRoundingOracle.h"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>
namespace llaminar2::test
{
namespace
{
/** @brief Require an actual admitted native program for each filtered numeric receipt.
 * @return Original active ISA; @throws std::invalid_argument For unavailable/Scalar/foreign execution.
 */
ISALevel nativeProgram()
{
    const auto isa=activeISALevel();cpu::qsa_native::requireNativeProgram(isa);return isa;
}
/** @brief Independently transcribe the original per-ISA QSA FMA chains and butterfly.
 * @param a Complete original left row; @param b Complete original right row.
 * @param isa Declared native tree; @return Memory-rounded original program's score.
 */
float reference(std::span<const float>a,std::span<const float>b,ISALevel isa)
{
    const auto width=isa==ISALevel::AVX512?16u:8u;std::array<float,16>lanes{};
    for(std::size_t c=0;c<a.size();++c)
    {volatile float value=std::fma(a[c],b[c],lanes[c%width]);lanes[c%width]=value;}
    for(unsigned step=width/2;step;step/=2)
        for(unsigned lane=0;lane<step;++lane)lanes[lane]=memoryRoundedAdd(lanes[lane],lanes[lane+step]);
    return lanes[0];
}
/** @brief Exercise original strided query/rotation rows without padding or input mutation.
 * @tparam Width Exact QSA logical width; @param isa Original declared native program.
 */
template<std::size_t Width>
void originalRows(ISALevel isa)
{
    constexpr std::size_t prefix=7,gap=9;
    for(std::size_t rows:{1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u,12u,13u,14u,15u,16u,31u,33u})
    {
        std::vector<float>a(prefix+rows*(Width+gap),919.25f),b(a.size(),-719.5f);
        std::uint32_t seed=std::uint32_t(rows*13+Width);
        for(std::size_t row=0;row<rows;++row)for(std::size_t c=0;c<Width;++c)
        {
            seed=seed*1664525u+1013904223u;const auto offset=prefix+row*(Width+gap)+c;
            a[offset]=std::bit_cast<float>((seed&0x807fffffu)|((110u+(seed%28))<<23));
            seed=seed*1664525u+1013904223u;
            b[offset]=std::bit_cast<float>((seed&0x807fffffu)|((110u+(seed%28))<<23));
        }
        const auto aBefore=a,bBefore=b;
        for(std::size_t row=0;row<rows;++row)
        {
            const auto offset=prefix+row*(Width+gap);
            const auto left=std::span(a).subspan(offset,Width),right=std::span(b).subspan(offset,Width);
            const auto expected=reference(left,right,isa);ASSERT_TRUE(std::isfinite(expected));
            const auto actual=cpu::qsa_native::dot<Width>(left.data(),right.data(),isa);
            ASSERT_TRUE(std::isfinite(actual));EXPECT_EQ(std::bit_cast<std::uint32_t>(actual),std::bit_cast<std::uint32_t>(expected));
        }
        EXPECT_EQ(a,aBefore);EXPECT_EQ(b,bBefore);
    }
}
} // namespace

TEST(CPUQwenSparseFP32,OriginalRowsRetainDeclaredNativeFMAAndButterfly)
{
    const auto isa=nativeProgram();originalRows<128>(isa);originalRows<256>(isa);
}
TEST(CPUQwenSparseFP32,QsaButterflyIsNotGenericAttentionFold)
{
    const auto isa=nativeProgram();std::array<float,128>a,b{};a.fill(1);
    const auto epsilon=std::bit_cast<float>(0x33800000u);b[0]=1;b[1]=epsilon;b[2]=-1;b[3]=epsilon;
    const auto expected=reference(a,b,isa);
    ASSERT_EQ(std::bit_cast<std::uint32_t>(expected),0x34000000u);
    const auto generic=memoryRoundedAdd(memoryRoundedAdd(1,epsilon),memoryRoundedAdd(-1,epsilon));
    ASSERT_EQ(std::bit_cast<std::uint32_t>(generic),0x33800000u);ASSERT_NE(generic,expected);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(cpu::qsa_native::dot<128>(a.data(),b.data(),isa)),0x34000000u);
}
TEST(CPUQwenSparseFP32,ScalarAndForeignProgramsAreNotAdvertised)
{
    EXPECT_THROW(cpu::qsa_native::requireNativeProgram(ISALevel::Scalar),std::invalid_argument);
#if LLAMINAR_COMPILED_WITH_NEON
    EXPECT_THROW(cpu::qsa_native::requireNativeProgram(ISALevel::AVX2),std::invalid_argument);
    EXPECT_THROW(cpu::qsa_native::requireNativeProgram(ISALevel::AVX512),std::invalid_argument);
#else
    EXPECT_THROW(cpu::qsa_native::requireNativeProgram(ISALevel::NEON),std::invalid_argument);
    EXPECT_THROW(cpu::qsa_native::requireNativeProgram(ISALevel::NEONDotProd),std::invalid_argument);
#endif
}
} // namespace llaminar2::test
