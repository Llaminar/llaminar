/**
 * @file Test__QwenSparseNativeBank.cpp
 * @brief Device-free QSA original-bank layout, address-domain and caller admission.
 *
 * These metadata/interface tests run on any ISA, without loading a model or
 * dereferencing fabricated addresses. Genuine fixture storage owns all positive
 * cases; the checked descriptor never substitutes for capacity or lifetime.
 * The actual complete common validator TU and StateIO dependency are linked,
 * not a duplicate test validator or SDK stub. Native numerical/cache integration
 * is a separate obligation and is not certified by these geometry assertions.
 */
#include "kernels/qwen4exp/QwenSparseNativeBankContract.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
namespace llaminar2::qwen4exp::test
{
namespace
{
using native_bank::Binding;using native_bank::Range;
/** @brief Own original bytes/tables throughout each synchronous checked borrow. */
struct Fixture final
{
    std::vector<std::uint32_t>storage;
    std::array<float,256>centroids{};std::array<float,512>anchor{};
    std::vector<float>rotations=std::vector<float>(2*256*256,0.125f);
    QwenSparseNativeBank bank{};std::size_t bytes=0,total=0;
    /** @brief Construct independently sized genuine token/head-major test storage.
     * @param codec Original cache codec; @param heads Original local heads.
     * @param tokens Physical rows; @param layout Original physical ordering.
     * @param q16 Independently selected Q16 native block width.
     */
    Fixture(TensorType codec,int heads,int tokens,QwenSparseNativeLayout layout,int q16=32)
    {
        switch(codec)
        {
        case TensorType::FP32:bytes=256*4;break;
        case TensorType::FP16:case TensorType::BF16:bytes=256*2;break;
        case TensorType::Q8_1:bytes=8*36;break;
        case TensorType::Q16_1:bytes=std::size_t(256/q16)*(8+2*q16);break;
        case TensorType::AQ8:bytes=260;break;
        case TensorType::TQ4:bytes=136;break;
        case TensorType::TQ8:bytes=264;break;
        default:throw std::invalid_argument("Unsupported fixture codec");
        }
        bank.codec=codec;bank.layout=layout;bank.q16_block_size=q16;
        bank.head_stride_bytes=bytes+12;bank.token_stride_bytes=std::size_t(heads)*bank.head_stride_bytes+20;
        if(layout==QwenSparseNativeLayout::HeadMajor)
        {bank.token_stride_bytes=bytes+20;bank.head_stride_bytes=std::size_t(tokens)*bank.token_stride_bytes+12;}
        total=std::size_t(tokens-1)*bank.token_stride_bytes+std::size_t(heads-1)*bank.head_stride_bytes+bytes;
        storage.assign((total+3)/4+7+9,0x4ab7d31eu);bank.data=storage.data()+7;
        bank.centroids=centroids.data();bank.anchor=anchor.data();
        bank.rotations[0]=rotations.data();bank.rotations[1]=rotations.data()+256*256;
    }
};
} // namespace

/** @test Every native physical layout resolves original logical256 slices, not stride padding. */
TEST(QwenSparseNativeBank,EveryCodecRoleAndLayoutRetainsOriginalSlices)
{
    for(auto codec:{TensorType::FP32,TensorType::FP16,TensorType::BF16,TensorType::Q8_1,
            TensorType::Q16_1,TensorType::AQ8,TensorType::TQ4,TensorType::TQ8})
        for(auto layout:{QwenSparseNativeLayout::TokenMajor,QwenSparseNativeLayout::HeadMajor})
            for(int heads:{1,2})for(int tokens:{1,3,5})for(int q16:{32,64,128})
                for(auto role:{detail::BankRole::Key,detail::BankRole::Value})
            {
                if((codec!=TensorType::Q16_1&&q16!=32)||(codec==TensorType::AQ8&&role==detail::BankRole::Value))continue;
                Fixture f(codec,heads,tokens,layout,q16);const auto before=f.storage;
                const auto b=Binding::bind(f.bank,role,heads,tokens);
                ASSERT_EQ(b.heads(),heads);EXPECT_EQ(b.role(),role);EXPECT_EQ(b.range().bytes(),f.total);
                for(int token=0;token<tokens;++token)for(int head=0;head<heads;++head)
                {
                    const auto slice=b.head(token,head);
                    EXPECT_EQ(slice.size(),f.bytes);EXPECT_EQ(slice.data(),static_cast<const unsigned char*>(f.bank.data)+
                        std::size_t(token)*f.bank.token_stride_bytes+std::size_t(head)*f.bank.head_stride_bytes);
                }
                EXPECT_EQ(f.storage,before);EXPECT_EQ(b.original().data,f.bank.data);
            }
}
/** @test Root/stride/table overflow and alignment reject without reading a wrapped pointer. */
TEST(QwenSparseNativeBank,WrappingAndUnalignedOriginalAddressDomainsReject)
{
    Fixture f(TensorType::FP32,2,3,QwenSparseNativeLayout::TokenMajor);
    const auto before=f.storage;auto bad=f.bank;bad.token_stride_bytes=std::numeric_limits<std::size_t>::max()-3;
    EXPECT_THROW((void)Binding::bind(bad,detail::BankRole::Key,2,3),std::invalid_argument);
    bad=f.bank;bad.head_stride_bytes=std::numeric_limits<std::size_t>::max()-3;
    EXPECT_THROW((void)Binding::bind(bad,detail::BankRole::Key,2,1),std::invalid_argument);
    bad=f.bank;bad.data=reinterpret_cast<const void*>(std::numeric_limits<std::uintptr_t>::max()-3);
    EXPECT_THROW((void)Binding::bind(bad,detail::BankRole::Key,1,1),std::invalid_argument);
    bad=f.bank;bad.data=static_cast<const unsigned char*>(bad.data)+1;
    EXPECT_THROW((void)Binding::bind(bad,detail::BankRole::Key,2,3),std::invalid_argument);
    Fixture tq(TensorType::TQ8,2,3,QwenSparseNativeLayout::TokenMajor);bad=tq.bank;
    bad.centroids=reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max()-3);
    EXPECT_THROW((void)Binding::bind(bad,detail::BankRole::Key,2,3),std::invalid_argument);
    bad=tq.bank;bad.rotations[1]=reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max()-3);
    EXPECT_THROW((void)Binding::bind(bad,detail::BankRole::Key,2,3),std::invalid_argument);
    EXPECT_EQ(f.storage,before);
}
/** @test Half rows genuinely admit two-byte alignment while every four-byte codec rejects it. */
TEST(QwenSparseNativeBank,ExactPhysicalElementAlignmentIsNotSIMDOrUniversalAlignment)
{
    for(auto codec:{TensorType::FP16,TensorType::BF16})
    {
        Fixture f(codec,2,3,QwenSparseNativeLayout::TokenMajor);const auto before=f.storage;
        auto bank=f.bank;bank.data=static_cast<const unsigned char*>(bank.data)+2;
        bank.head_stride_bytes+=2;bank.token_stride_bytes+=2;
        // The fixture's owned suffix accommodates these extra eight bytes;
        // no descriptor is being used to claim an allocation's capacity.
        EXPECT_NO_THROW((void)Binding::bind(bank,detail::BankRole::Key,2,3));
        auto bad=bank;bad.data=static_cast<const unsigned char*>(bad.data)+1;
        EXPECT_THROW((void)Binding::bind(bad,detail::BankRole::Key,2,3),std::invalid_argument);
        bad=bank;++bad.head_stride_bytes;
        EXPECT_THROW((void)Binding::bind(bad,detail::BankRole::Key,2,3),std::invalid_argument);
        EXPECT_EQ(f.storage,before);
    }
    for(auto codec:{TensorType::FP32,TensorType::Q8_1,TensorType::Q16_1,TensorType::AQ8,TensorType::TQ4,TensorType::TQ8})
    {
        Fixture f(codec,2,3,QwenSparseNativeLayout::TokenMajor);auto bad=f.bank;
        bad.data=static_cast<const unsigned char*>(bad.data)+2;
        EXPECT_THROW((void)Binding::bind(bad,detail::BankRole::Key,2,3),std::invalid_argument);
        bad=f.bank;bad.head_stride_bytes+=2;
        EXPECT_THROW((void)Binding::bind(bad,detail::BankRole::Key,2,3),std::invalid_argument);
        bad=f.bank;bad.token_stride_bytes+=2;
        EXPECT_THROW((void)Binding::bind(bad,detail::BankRole::Key,2,3),std::invalid_argument);
    }
}
/** @test Every original bank/table/anchor byte remains excluded from prospective publications. */
TEST(QwenSparseNativeBank,CompleteOriginalCapacityAndMetadataExclusions)
{
    std::array<float,256>output{};
    for(auto codec:{TensorType::FP32,TensorType::AQ8,TensorType::TQ4,TensorType::TQ8})
    {
        Fixture f(codec,2,5,QwenSparseNativeLayout::HeadMajor);
        const auto b=Binding::bind(f.bank,detail::BankRole::Key,2,5);
        EXPECT_NO_THROW(b.exclude(Range::bind(output.data(),sizeof(output),alignof(float))));
        const auto last=b.head(4,1);EXPECT_THROW(b.exclude(Range::bind(last.data()+last.size()-4,4,4)),std::invalid_argument);
        if(codec==TensorType::AQ8)EXPECT_THROW(b.exclude(Range::bind(f.anchor.data()+511,4,4)),std::invalid_argument);
        if(codec==TensorType::TQ4||codec==TensorType::TQ8)
        {
            const auto end=codec==TensorType::TQ4?15:255;
            EXPECT_THROW(b.exclude(Range::bind(f.centroids.data()+end,4,4)),std::invalid_argument);
            EXPECT_THROW(b.exclude(Range::bind(f.rotations.data()+2*256*256-1,4,4)),std::invalid_argument);
        }
    }
}
/** @test Closed role/layout/codec metadata cannot create a partially admitted reader. */
TEST(QwenSparseNativeBank,UnknownOrIncompleteNativePoliciesReject)
{
    Fixture f(TensorType::AQ8,2,3,QwenSparseNativeLayout::TokenMajor);
    EXPECT_THROW((void)Binding::bind(f.bank,detail::BankRole::Value,2,3),std::invalid_argument);
    EXPECT_THROW((void)Binding::bind(f.bank,detail::BankRole(255),2,3),std::invalid_argument);
    EXPECT_THROW((void)Binding::bind(f.bank,detail::BankRole::Key,3,3),std::invalid_argument);
    auto bad=f.bank;bad.layout=QwenSparseNativeLayout(255);
    EXPECT_THROW((void)Binding::bind(bad,detail::BankRole::Key,2,3),std::invalid_argument);
    Fixture q16(TensorType::Q16_1,2,3,QwenSparseNativeLayout::TokenMajor);
    bad=q16.bank;bad.q16_block_size=48;
    EXPECT_THROW((void)Binding::bind(bad,detail::BankRole::Key,2,3),std::invalid_argument);
    Fixture tq(TensorType::TQ4,2,3,QwenSparseNativeLayout::TokenMajor);bad=tq.bank;bad.rotations[1]=nullptr;
    EXPECT_THROW((void)Binding::bind(bad,detail::BankRole::Key,2,3),std::invalid_argument);
    bad=tq.bank;bad.centroids=nullptr;EXPECT_THROW((void)Binding::bind(bad,detail::BankRole::Key,2,3),std::invalid_argument);
}
/** @test The actual common entrypoint uses the checked original bank, not its retired predicate. */
TEST(QwenSparseNativeBank,ActualCommonValidatorRejectsWrappingOriginalBank)
{
    Fixture keys(TensorType::FP32,1,3,QwenSparseNativeLayout::TokenMajor),values(TensorType::FP32,1,3,QwenSparseNativeLayout::TokenMajor);
    QwenSparseReplay replay{};std::array<float,256>query{},output{};std::array<std::int32_t,QwenSparseSelected>selected{};std::int32_t count=1;
    QwenSparseAttention p{.replay=&replay,.query=query.data(),.kv={.keys=keys.bank,.values=values.bank},
        .selected=selected.data(),.selected_count=&count,.output=output.data(),.row_capacity=1,.token_capacity=3,.query_heads=1,.kv_heads=1};
    EXPECT_NO_THROW(validateQwenSparseAttention(p));
    p.kv.keys.token_stride_bytes=std::numeric_limits<std::size_t>::max()-3;
    EXPECT_THROW(validateQwenSparseAttention(p),std::invalid_argument);
    EXPECT_EQ(output,(std::array<float,256>{}));
}
/** @test Corrupted post-admission row/head coordinates abort without crossing a collective exception edge. */
TEST(QwenSparseNativeBank,PostAdmissionCoordinateCorruptionIsFatal)
{
    Fixture f(TensorType::FP32,2,3,QwenSparseNativeLayout::TokenMajor);
    const auto b=Binding::bind(f.bank,detail::BankRole::Key,2,3);
    EXPECT_DEATH((void)b.head(-1,0),"lost its admitted original coordinates");
    EXPECT_DEATH((void)b.head(3,0),"lost its admitted original coordinates");
    EXPECT_DEATH((void)b.head(0,2),"lost its admitted original coordinates");
}
/** @test The actual caller rejects integer/FP-to-integer overflow before native grid or rotary arithmetic. */
TEST(QwenSparseNativeBank,ActualCommonValidatorRejectsUnrepresentableGQAAndRotaryPolicies)
{
    Fixture keys(TensorType::FP32,1,3,QwenSparseNativeLayout::TokenMajor),values(TensorType::FP32,1,3,QwenSparseNativeLayout::TokenMajor);
    QwenSparseReplay replay{};std::array<float,512>query{},output{};std::array<std::int32_t,QwenSparseSelected>selected{};std::int32_t count=1;
    const QwenSparseAttention original{.replay=&replay,.query=query.data(),.kv={.keys=keys.bank,.values=values.bank},
        .selected=selected.data(),.selected_count=&count,.output=output.data(),.row_capacity=1,.token_capacity=3,.query_heads=1,.kv_heads=1};
    auto p=original;p.query_head_start=std::numeric_limits<int>::max();EXPECT_THROW(validateQwenSparseAttention(p),std::invalid_argument);
    p=original;p.kv_head_start=std::numeric_limits<int>::max();EXPECT_THROW(validateQwenSparseAttention(p),std::invalid_argument);
    p=original;p.query_heads=std::numeric_limits<int>::max();EXPECT_THROW(validateQwenSparseAttention(p),std::invalid_argument);
    p=original;p.query_heads=2;p.row_capacity=std::numeric_limits<int>::max();EXPECT_THROW(validateQwenSparseAttention(p),std::invalid_argument);
    std::array<float,32>cosine{},sine{};p=original;
    p.kv.key_policy.encoding=attention::AttentionKeyCacheEncoding::PreRotaryDeviceTransform;
    p.kv.key_cosine=cosine.data();p.kv.key_sine=sine.data();p.kv.key_position_capacity=1;
    p.kv.key_policy.partial_rotary_factor=0.25f;EXPECT_NO_THROW(validateQwenSparseAttention(p));
    for(float factor:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::max(),-1.0f,0.0f,1.5f})
    {p.kv.key_policy.partial_rotary_factor=factor;EXPECT_THROW(validateQwenSparseAttention(p),std::invalid_argument);}
}
} // namespace llaminar2::qwen4exp::test
