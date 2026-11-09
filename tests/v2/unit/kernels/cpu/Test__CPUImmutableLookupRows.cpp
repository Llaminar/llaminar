/**
 * @file Test__CPUImmutableLookupRows.cpp
 * @brief Original immutable-lookup source geometry, native payload and team-completion regressions.
 *
 * These model-free tests compile the actual production TU. Independent integer
 * half/BF16/IQ4 controls protect conversion and nibble arithmetic; actual M1
 * publications protect every other source's established byte program. All
 * output banks are poisoned before each separately asserted writer. Fixtures
 * are requested live rows, not prepared-weight/cache mirrors or PMA admission.
 */
#include "kernels/qwen4exp/ImmutableLookupKernels.h"
#include "kernels/qwen4exp/ImmutableLookupDecode.h"
#include "tensors/IQQuantTables.h"
#include "utils/CPUFeatures.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <limits>
#include <span>
#include <thread>
#include <vector>
#include <omp.h>

namespace llaminar2::test
{
namespace
{
using qwen4exp::decodeCPUImmutableLookupRows;
constexpr std::uint32_t poisonWord=0x7fdabcdeu;
constexpr float canary=-12345.25f;

/** @brief Require the actual selected native program, not just a SIMD-capable binary.
 * @throws std::runtime_error If an independently filtered numeric receipt selects
 * scalar or a foreign/uncompiled ISA. Valid AVX512-codegen/AVX2-runtime remains admitted.
 */
void requireNativeLookup()
{
    const auto active=activeISALevel();
    if(active==ISALevel::Scalar||!isNativeCPUExecutionISAPair(compiledCPUISALevel(),active))
        throw std::runtime_error("Immutable lookup native payload conformance requires an admitted non-Scalar ISA");
}
/** @brief Authenticate every original prefix/suffix guard word, including adjacent fringes.
 * @param storage Complete owned test bank; @param prefix Guard words before live output.
 * @param live Number of writable output words.
 * @return False for malformed geometry or any altered guard, never a partial boundary check.
 */
bool guardsMatch(std::span<const float>storage,std::size_t prefix,std::size_t live)
{
    if(prefix>storage.size()||live>storage.size()-prefix)return false;
    const auto untouched=[](float value){return std::bit_cast<std::uint32_t>(value)==std::bit_cast<std::uint32_t>(canary);};
    return std::all_of(storage.begin(),storage.begin()+prefix,untouched)&&
        std::all_of(storage.begin()+prefix+live,storage.end(),untouched);
}

/** @brief Restore the caller's OpenMP configuration after one bounded test budget. */
class LookupWorkers
{
public:
    /** @param count Positive test team size; this is not production placement evidence. */
    explicit LookupWorkers(int count):previous_(omp_get_max_threads()),dynamic_(omp_get_dynamic())
    {omp_set_dynamic(0);omp_set_num_threads(count);}
    /** @brief Restore both settings without changing process affinity. */
    ~LookupWorkers(){omp_set_num_threads(previous_);omp_set_dynamic(dynamic_);}
    LookupWorkers(const LookupWorkers&)=delete;
    LookupWorkers&operator=(const LookupWorkers&)=delete;
private:
    int previous_; ///< Original process preference, not discovered physical capacity.
    int dynamic_; ///< Original dynamic-team policy.
};

/** @brief Independently widen one original IEEE half word using integer fields.
 * @param word Original physical word, including signed zeros/subnormals/NaN payloads.
 * @return Exact FP32 bits; no hardware half conversion or production decoder is called.
 */
std::uint32_t halfBits(std::uint16_t word)
{
    const auto sign=std::uint32_t(word&0x8000u)<<16;
    const auto exponent=(word>>10)&31u;
    std::uint32_t mantissa=word&1023u;
    if(exponent==31)return sign|0x7f800000u|(mantissa<<13);
    if(exponent)return sign|((exponent+112u)<<23)|(mantissa<<13);
    if(!mantissa)return sign;
    int shift=0;
    while(!(mantissa&1024u)){mantissa<<=1;++shift;}
    return sign|(std::uint32_t(113-shift)<<23)|((mantissa&1023u)<<13);
}
/** @brief Poison an entire independent publication, including NaN payload discriminators.
 * @param values Complete live output bank, excluding its canary prefix/suffix.
 */
void poison(std::span<float>values)
{std::fill(values.begin(),values.end(),std::bit_cast<float>(poisonWord));}
/** @brief Compare every authored word, including nonfinite values, without NaN-blind arithmetic.
 * @param actual Complete writer output; @param expected Original independent/serial words.
 */
void expectWords(std::span<const float>actual,std::span<const float>expected)
{
    ASSERT_EQ(actual.size(),expected.size());
    for(std::size_t i=0;i<actual.size();++i)
        ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[i]),std::bit_cast<std::uint32_t>(expected[i]))<<"coordinate "<<i;
}
/** @brief Obtain complete original row bytes, never an approximate source-width ratio.
 * @param format Original source; @param width Positive logical row elements.
 * @return Exact canonical physical row extent, including a partial final source block.
 */
std::size_t strideFor(TensorType format,int width)
{
    if(format==TensorType::FP32)return std::size_t(width)*4;
    if(format==TensorType::FP16||format==TensorType::BF16)return std::size_t(width)*2;
    return NativeEmbeddingLayout::forSource(format,1,width).row_stride_bytes;
}
/** @brief Poison every word differently from its known complete reference.
 * @param values Complete independent writer bank; @param expected Complete original reference.
 * @details Complemented bits also prevent a coincidentally matching NaN poison
 * from accepting an unwritten coordinate in the all-format random-byte sweep.
 */
void poisonRelative(std::span<float>values,std::span<const float>expected)
{
    ASSERT_EQ(values.size(),expected.size());
    for(std::size_t i=0;i<values.size();++i)values[i]=std::bit_cast<float>(~std::bit_cast<std::uint32_t>(expected[i]));
}
/** @brief Evaluate the unchanged original source component, independently of the CPU bulk writer.
 * @param type Original representation; @param width Logical row elements.
 * @param stride Exact complete physical row bytes; @param rows Immutable input rows.
 * @param count Original live rows.
 * @return Complete component reference, not an independent quantizer/codebook arithmetic proof.
 */
std::vector<float>componentReference(TensorType type,int width,std::size_t stride,const std::byte*rows,std::size_t count)
{
    const NativeEmbeddingIQTables grids{iq3s_grid,iq3xxs_grid,iq2s_grid,iq2xs_grid,iq2xxs_grid,iq1s_grid};
    const auto decode=[&]<TensorType Format>()
    {
        std::vector<float>result(count*width);
        for(std::size_t row=0;row<count;++row)
            for(int column=0;column<width;++column)
                result[row*width+column]=qwen4exp::lookup_detail::decode<Format>(rows+row*stride,column,grids);
        return result;
    };
    switch(type)
    {
#define LOOKUP_COMPONENT_CASE(F) case TensorType::F:return decode.template operator()<TensorType::F>();
        LOOKUP_COMPONENT_CASE(FP32) LOOKUP_COMPONENT_CASE(FP16) LOOKUP_COMPONENT_CASE(BF16)
        QFN_LOOKUP_QUANT_FORMATS(LOOKUP_COMPONENT_CASE)
#undef LOOKUP_COMPONENT_CASE
    default:throw std::invalid_argument("Immutable lookup test has no original source component");
    }
}
} // namespace

TEST(CPUImmutableLookupRows,FloatingAllOriginalPayloads)
{
    requireNativeLookup();
    constexpr int width=257;
    constexpr std::size_t count=256,size=count*width;
    for(auto type:{TensorType::FP16,TensorType::BF16,TensorType::FP32})
    {
        const auto stride=strideFor(type,width);
        std::vector<std::byte>native(count*stride);
        std::vector<float>expected(size),storage(size+14,canary);
        auto output=std::span(storage).subspan(7,size);
        for(std::size_t i=0;i<size;++i)
        {
            const auto word=std::uint16_t(i);
            if(type==TensorType::FP32)
            {
                const std::uint32_t bits=(std::uint32_t(word)<<16)|0x1235u;
                std::memcpy(native.data()+i*4,&bits,4);expected[i]=std::bit_cast<float>(bits);
            }
            else
            {
                std::memcpy(native.data()+i*2,&word,2);
                expected[i]=std::bit_cast<float>(type==TensorType::FP16?halfBits(word):std::uint32_t(word)<<16);
            }
        }
        const auto original=native;
        for(int workers:{1,3,10})
        {
            LookupWorkers team(workers);poison(output);
            decodeCPUImmutableLookupRows(type,width,stride,native.data(),count,output.data());
            expectWords(output,expected);EXPECT_EQ(native,original);
            EXPECT_TRUE(guardsMatch(storage,7,size));
        }
    }
}

TEST(CPUImmutableLookupRows,FloatingIndependentNonfiniteFringes)
{
    requireNativeLookup();
    constexpr std::array<std::uint16_t,18>halfWords{1,0x8001,0x7c01,0xfc35,0x8000,0,0x3ff,0x83ff,
        0x400,0x8400,0x3c01,0xbc01,0x7bff,0xfbff,0x7e35,0xfe35,0x7c00,0xfc00};
    constexpr std::array<std::uint16_t,18>bf16Words{1,0x8001,0x7f81,0xffb5,0x8000,0,0x7f,0x807f,
        0x80,0x8080,0x3f81,0xbf81,0x7f7f,0xff7f,0x7fc1,0xffc1,0x7f80,0xff80};
    constexpr std::array<std::uint32_t,18>fp32Words{1,0x80000001u,0x7f800001u,0xff800035u,0x80000000u,0,
        0x7fffffu,0x807fffffu,0x800000u,0x80800000u,0x3f800001u,0xbf800001u,0x7f7fffffu,0xff7fffffu,
        0x7fc12345u,0xffc12345u,0x7f800000u,0xff800000u};
    for(auto type:{TensorType::FP16,TensorType::BF16,TensorType::FP32})
    {
        bool sawNaNTail=false,sawSubnormalTail=false,sawNegativeZeroTail=false;
        for(int width:{1,2,3,5,6,7,31,32,33,255,257,258,259})
        {
            const auto stride=strideFor(type,width);
            std::vector<std::byte>source(stride);
            std::vector<float>expected(width),storage(width+14,canary);
            auto output=std::span(storage).subspan(7,width);
            for(int i=0;i<width;++i)
            {
                const auto word=type==TensorType::BF16?bf16Words[i%bf16Words.size()]:halfWords[i%halfWords.size()];
                const auto bits=type==TensorType::FP32?fp32Words[i%fp32Words.size()]:
                    (type==TensorType::BF16?std::uint32_t(word)<<16:halfBits(word));
                expected[i]=std::bit_cast<float>(bits);
                if(type==TensorType::FP32)std::memcpy(source.data()+i*4,&bits,4);
                else std::memcpy(source.data()+i*2,&word,2);
                // Widths1..3 prove these exact source classes reach bounded
                // tails, rather than merely appearing elsewhere in a full bank.
                if(i>=width-width%4)
                {
                    sawNaNTail|=(bits&0x7f800000u)==0x7f800000u&&(bits&0x7fffffu)!=0;
                    const auto exponent=type==TensorType::FP16?(word&0x7c00u):
                        (type==TensorType::BF16?(word&0x7f80u):(bits&0x7f800000u));
                    const auto mantissa=type==TensorType::FP16?(word&0x3ffu):
                        (type==TensorType::BF16?(word&0x7fu):(bits&0x7fffffu));
                    sawSubnormalTail|=exponent==0&&mantissa!=0;
                    sawNegativeZeroTail|=bits==0x80000000u;
                }
            }
            const auto original=source;
            for(int workers:{1,3,10})
            {
                LookupWorkers team(workers);poisonRelative(output,expected);
                decodeCPUImmutableLookupRows(type,width,stride,source.data(),1,output.data());
                expectWords(output,expected);EXPECT_EQ(source,original);
                EXPECT_TRUE(guardsMatch(storage,7,width));
            }
        }
        EXPECT_TRUE(sawNaNTail);EXPECT_TRUE(sawSubnormalTail);EXPECT_TRUE(sawNegativeZeroTail);
    }
}

TEST(CPUImmutableLookupRows,GuardAuthenticationRejectsAdjacentOverwrites)
{
    std::array<float,19>storage;storage.fill(canary);
    EXPECT_TRUE(guardsMatch(storage,7,5));
    for(std::size_t i=0;i<storage.size();++i)
    {
        if(i>=7&&i<12)continue;
        storage[i]=canary+1.0f;
        EXPECT_FALSE(guardsMatch(storage,7,5))<<"guard "<<i;
        storage[i]=canary;
    }
    EXPECT_FALSE(guardsMatch(storage,20,0));EXPECT_FALSE(guardsMatch(storage,7,13));
    EXPECT_TRUE(guardsMatch(storage,7,5));
}

TEST(CPUImmutableLookupRows,IQ4OriginalCodebookScalesAndPartialRows)
{
    requireNativeLookup();
    constexpr std::array<int,16>grid{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    constexpr std::array<std::uint16_t,12>scales{0,0x8000,1,0x3ff,0x400,0x3555,0x3c01,0x7bff,0xbc01,0x7c01,0x7e35,0xfc01};
    LookupWorkers team(3);
    for(int width=1;width<=65;++width)
        for(auto scale:scales)
        {
            constexpr std::size_t count=5;
            const auto stride=strideFor(TensorType::IQ4_NL,width),blocks=stride/sizeof(IQ4_NLBlock);
            std::vector<IQ4_NLBlock>native(count*blocks);
            std::vector<float>expected(count*width),storage(count*width+14,canary);
            auto output=std::span(storage).subspan(7,count*width);
            for(std::size_t b=0;b<native.size();++b)
            {
                native[b].d=scale;
                for(int i=0;i<16;++i)native[b].qs[i]=std::uint8_t(b*29+i*17);
            }
            for(std::size_t row=0;row<count;++row)
                for(int column=0;column<width;++column)
                {
                    const auto &block=native[row*blocks+column/32];
                    const int code=(block.qs[column%16]>>(4*((column%32)/16)))&15;
                    volatile float product=std::bit_cast<float>(halfBits(scale))*float(grid[code]);
                    expected[row*width+column]=product;
                }
            const auto original=native;poison(output);
            decodeCPUImmutableLookupRows(TensorType::IQ4_NL,width,stride,reinterpret_cast<const std::byte*>(native.data()),count,output.data());
            expectWords(output,expected);EXPECT_EQ(0,std::memcmp(native.data(),original.data(),native.size()*sizeof(IQ4_NLBlock)));
            EXPECT_TRUE(guardsMatch(storage,7,count*width));
        }
}

TEST(CPUImmutableLookupRows,EverySourceAndPositiveTeamRetainSerialOriginalRows)
{
    requireNativeLookup();
    const std::array formats{
        TensorType::FP32,TensorType::FP16,TensorType::BF16,
#define LOOKUP_TEST_FORMAT(F) TensorType::F,
        QFN_LOOKUP_QUANT_FORMATS(LOOKUP_TEST_FORMAT)
#undef LOOKUP_TEST_FORMAT
    };
    for(auto type:formats)
        for(int width:{1,31,32,33,63,64,65,255,256,257})
        {
            constexpr std::size_t count=5;
            const auto stride=strideFor(type,width);
            std::vector<std::byte>native(count*stride);
            std::uint32_t state=0x13579bdu;
            for(auto &b:native){state=state*1664525u+1013904223u;b=std::byte(state>>24);}
            const auto original=native;
            const auto component=componentReference(type,width,stride,native.data(),count);
            std::vector<float>serial(count*width),storage(count*width+14,canary);
            auto output=std::span(storage).subspan(7,count*width);
            {
                LookupWorkers team(1);
                for(std::size_t row=0;row<count;++row)
                {
                    auto destination=std::span(serial).subspan(row*width,width);
                    const auto reference=std::span(component).subspan(row*width,width);poisonRelative(destination,reference);
                    decodeCPUImmutableLookupRows(type,width,stride,native.data()+row*stride,1,destination.data());
                    expectWords(destination,reference);
                }
            }
            for(int workers=1;workers<=10;++workers)
            {
                LookupWorkers team(workers);poisonRelative(output,serial);
                decodeCPUImmutableLookupRows(type,width,stride,native.data(),count,output.data());
                expectWords(output,serial);EXPECT_EQ(native,original);
                EXPECT_TRUE(guardsMatch(storage,7,count*width));
            }
        }
}

TEST(CPUImmutableLookupRows,GeometryAndOriginalOutputIntersectionsRejectBeforeStores)
{
    std::array<std::uint32_t,128>storage;storage.fill(poisonWord);const auto original=storage;
    auto *source=reinterpret_cast<const std::byte*>(storage.data());auto *output=reinterpret_cast<float*>(storage.data()+64);
    for(auto width:{0,-1})EXPECT_THROW(decodeCPUImmutableLookupRows(TensorType::FP16,width,64,source,1,output),std::invalid_argument);
    EXPECT_THROW(decodeCPUImmutableLookupRows(TensorType::FP16,32,63,source,1,output),std::invalid_argument);
    EXPECT_THROW(decodeCPUImmutableLookupRows(TensorType::FP16,32,64,source,std::numeric_limits<std::size_t>::max(),output),std::invalid_argument);
    EXPECT_THROW(decodeCPUImmutableLookupRows(TensorType::FP16,32,64,nullptr,1,output),std::invalid_argument);
    EXPECT_THROW(decodeCPUImmutableLookupRows(TensorType::FP16,32,64,source,1,nullptr),std::invalid_argument);
    EXPECT_THROW(decodeCPUImmutableLookupRows(TensorType::FP16,32,64,source,1,reinterpret_cast<float*>(storage.data()+1)),std::invalid_argument);
    EXPECT_THROW(decodeCPUImmutableLookupRows(TensorType::FP16,32,64,source+4,1,reinterpret_cast<float*>(storage.data())),std::invalid_argument);
    EXPECT_THROW(decodeCPUImmutableLookupRows(TensorType::FP16,32,64,reinterpret_cast<const std::byte*>(std::numeric_limits<std::uintptr_t>::max()-16),1,output),std::invalid_argument);
    EXPECT_THROW(decodeCPUImmutableLookupRows(TensorType::FP16,32,64,source,1,reinterpret_cast<float*>(std::numeric_limits<std::uintptr_t>::max()-16)),std::invalid_argument);
    EXPECT_THROW(decodeCPUImmutableLookupRows(static_cast<TensorType>(0xffff),32,64,source,0,output),std::invalid_argument);
    EXPECT_EQ(storage,original);
}

TEST(CPUImmutableLookupRows,RetainedNativePublicationCompletesBeforeOutputReuse)
{
    requireNativeLookup();
    LookupWorkers team(3);constexpr int width=33;constexpr std::size_t count=3;
    const auto stride=strideFor(TensorType::IQ4_NL,width);
    std::vector<IQ4_NLBlock>source(count*stride/sizeof(IQ4_NLBlock));
    for(std::size_t b=0;b<source.size();++b)
    {source[b].d=0x3555;for(int i=0;i<16;++i)source[b].qs[i]=std::uint8_t(b*29+i*17);}
    const auto *bytes=reinterpret_cast<const std::byte*>(source.data());
    const auto expected=componentReference(TensorType::IQ4_NL,width,stride,bytes,count);
    std::vector<float>output(count*width);std::atomic<int>mismatches{0};
#pragma omp parallel num_threads(3)
    {
        for(int iteration=0;iteration<20;++iteration)
        {
#pragma omp single
            {for(std::size_t i=0;i<output.size();++i)output[i]=std::bit_cast<float>(~std::bit_cast<std::uint32_t>(expected[i]));}
            decodeCPUImmutableLookupRows(TensorType::IQ4_NL,width,stride,bytes,count,output.data());
#pragma omp single
            {
                if(std::memcmp(output.data(),expected.data(),output.size()*sizeof(float)))mismatches.fetch_add(1,std::memory_order_relaxed);
                // The selected owner may retire/reuse the publication as soon
                // as the kernel returns; no late writer may follow this edge.
                std::fill(output.begin(),output.end(),canary);
            }
        }
    }
    EXPECT_EQ(mismatches.load(),0);
}

TEST(CPUImmutableLookupRows,EmptyRetainedCallsCompleteBeforeImmediateReuse)
{
    LookupWorkers team(3);std::atomic<int>arrived{0},early{0};
#pragma omp parallel num_threads(3)
    {
        for(int iteration=0;iteration<20;++iteration)
        {
            if(omp_get_thread_num()==2)std::this_thread::sleep_for(std::chrono::milliseconds(1));
            arrived.fetch_add(1,std::memory_order_release);
            decodeCPUImmutableLookupRows(TensorType::FP16,7,14,nullptr,0,nullptr);
            if(arrived.load(std::memory_order_acquire)<3*(iteration+1))early.fetch_add(1,std::memory_order_relaxed);
#pragma omp barrier
        }
    }
    EXPECT_EQ(arrived.load(),60);EXPECT_EQ(early.load(),0);
}
} // namespace llaminar2::test
