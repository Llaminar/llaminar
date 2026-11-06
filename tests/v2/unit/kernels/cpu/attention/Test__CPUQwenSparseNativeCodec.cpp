/**
 * @file Test__CPUQwenSparseNativeCodec.cpp
 * @brief Native original QSA codec fragments and independent separately rounded references.
 *
 * Fixtures own real original physical banks and supplied tables. Every result
 * word is poisoned and authenticated outside timing; source/guards remain exact.
 * These are codec-leaf proofs, not execution of the sparse attention public
 * entrypoint, query/selection/softmax/inverse-basis lifecycle or a performance
 * certificate. Invalid numerical data must abort, never become a repaired zero.
 */
#include "kernels/cpu/attention/CPUQwenSparseNativeCodec.h"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
namespace llaminar2::test
{
namespace
{
using qwen4exp::detail::BankRole;using qwen4exp::QwenSparseNativeBank;using qwen4exp::QwenSparseNativeLayout;
/** @brief Admit an actual native program independently for each filtered numerical receipt. */
void requireNative(){cpu::qsa_native::requireNativeProgram(activeISALevel());}
/** @brief Independently retain a completed FP32 product in diagnostic storage.
 * @param a First operand; @param b Second operand; @return Separately rounded product.
 */
float product(float a,float b){volatile float out=a*b;return out;}
/** @brief Independently retain a completed FP32 sum in diagnostic storage.
 * @param a First operand; @param b Second operand; @return Separately rounded sum.
 */
float sum(float a,float b){volatile float out=a+b;return out;}
/** @brief Copy an original typed word without inferring packed-source alignment.
 * @tparam T Original stored word; @param p Original bytes; @return Exact stored word.
 */
template<class T>T word(const unsigned char*p){T value;std::memcpy(&value,p,sizeof(value));return value;}
/** @brief Publish a fixture word only during setup.
 * @tparam T Original stored word; @param p Exclusive fixture byte address; @param value Setup word.
 */
template<class T>void put(unsigned char*p,T value){std::memcpy(p,&value,sizeof(value));}
/** @brief Independently reconstruct a finite half payload, including original subnormals.
 * @param h Original half word; @return Exact finite binary32 representation.
 */
float half(std::uint16_t h)
{
    const float magnitude=(h&0x7c00u)?std::ldexp(float(1024+(h&1023)),int((h>>10)&31)-25):std::ldexp(float(h&1023),-24);
    return std::copysign(magnitude,(h&0x8000u)?-1.0f:1.0f);
}
/** @brief Own genuine physical rows and original noncanonical codebooks throughout each borrow. */
struct Fixture final
{
    QwenSparseNativeBank bank{};std::size_t bytes=0;
    std::vector<std::uint32_t>storage;std::array<float,256>book;std::array<float,512>anchor;
    std::vector<float>rotations=std::vector<float>(2*256*256,0);
    /** @brief Construct independent original physical codecs, never production-decoded fixtures.
     * @param codec Original encoding; @param q16 Native block width.
     * @param layout Original physical token/head ordering.
     */
    Fixture(TensorType codec,int q16,QwenSparseNativeLayout layout)
    {
        switch(codec){case TensorType::FP32:bytes=1024;break;case TensorType::FP16:case TensorType::BF16:bytes=512;break;
        case TensorType::Q8_1:bytes=288;break;case TensorType::Q16_1:bytes=std::size_t(256/q16)*(8+2*q16);break;
        case TensorType::AQ8:bytes=260;break;case TensorType::TQ4:bytes=136;break;case TensorType::TQ8:bytes=264;break;
        default:throw std::invalid_argument("bad fixture codec");}
        bank.codec=codec;bank.q16_block_size=q16;bank.layout=layout;
        bank.head_stride_bytes=bytes+12;bank.token_stride_bytes=2*bank.head_stride_bytes+20;
        if(layout==QwenSparseNativeLayout::HeadMajor){bank.token_stride_bytes=bytes+20;bank.head_stride_bytes=2*bank.token_stride_bytes+12;}
        const auto total=bank.head_stride_bytes+bank.token_stride_bytes+bytes;
        storage.assign((total+3)/4+16,0x4a37d12eu);bank.data=storage.data()+7;
        bank.centroids=book.data();bank.anchor=anchor.data();bank.rotations[0]=rotations.data();bank.rotations[1]=rotations.data()+256*256;
        for(unsigned i=0;i<book.size();++i)book[i]=product(float(int(i)-127),std::bit_cast<float>(0x3d8f5c29u));
        for(unsigned i=0;i<anchor.size();++i)anchor[i]=i%3?-9.0f:std::bit_cast<float>(0x3f800001u);
        for(int row=0;row<2;++row)for(int head=0;head<2;++head)
        {
            auto*p=data(row,head);
            if(codec==TensorType::TQ4||codec==TensorType::TQ8)
            {put(p,row?1e-31f:3.25f);put(p+4,row?std::numeric_limits<float>::quiet_NaN():0.7f);}
            if(codec==TensorType::AQ8)put(p,std::bit_cast<float>(0x3f800001u));
            for(int c=0;c<256;++c)
            {
                const auto code=std::int8_t((c+head*73)%256-128);
                if(codec==TensorType::FP32)put(p+c*4,product(float(c-127),0.017321f));
                if(codec==TensorType::FP16){constexpr std::uint16_t h[]={0,0x8000,1,0x03ff,0x0400,0x3555,0x3c01,0x7bff,0xfbff};put(p+c*2,h[c%9]);}
                if(codec==TensorType::BF16){constexpr std::uint16_t h[]={0,0x8000,1,0x007f,0x0080,0x3f81,0xbf81,0x7f7f,0xff7f};put(p+c*2,h[c%9]);}
                if(codec==TensorType::Q8_1){auto*b=p+(c/32)*36;put(b,std::uint16_t(c/32%2?0x0001:0x3c01));put(b+4+c%32,code);}
                if(codec==TensorType::Q16_1){auto*b=p+(c/q16)*(8+2*q16);put(b,std::bit_cast<float>(0x3f800001u));put(b+8+2*(c%q16),std::int16_t((c*257)%65536-32768));}
                if(codec==TensorType::AQ8)put(p+4+c,code);
                if(codec==TensorType::TQ8)p[8+c]=std::uint8_t(c+head*73);
            }
            if(codec==TensorType::TQ4)for(int group=0;group<32;++group)
            {std::uint8_t low[8],high=0;for(int lane=0;lane<8;++lane){const auto index=(group*8+lane+head*11)%16;low[lane]=std::uint8_t(index&7);high|=std::uint8_t((index>>3)<<lane);}tq3_pack_8(low,p+8+group*3);p[104+group]=high;}
        }
    }
    /** @brief Resolve owned fixture storage independently of the production binding.
     * @param row Physical token; @param head Physical local KV head; @return Original fixture root.
     */
    unsigned char*data(int row,int head){return reinterpret_cast<unsigned char*>(storage.data()+7)+std::size_t(row)*bank.token_stride_bytes+std::size_t(head)*bank.head_stride_bytes;}
    /** @brief Independent declared-source coordinate program with explicit memory rounding.
     * @param row Original row; @param head Original local head; @param c Original coordinate.
     * @param role Exact key/source or value/fitted radial role; @return Finite original decoded coordinate.
     */
    float reference(int row,int head,int c,BankRole role)
    {
        const auto*p=data(row,head);
        switch(bank.codec)
        {
        case TensorType::FP32:return word<float>(p+4*c);
        case TensorType::FP16:return half(word<std::uint16_t>(p+2*c));
        case TensorType::BF16:return std::bit_cast<float>(std::uint32_t(word<std::uint16_t>(p+2*c))<<16);
        case TensorType::Q8_1:{const auto*b=p+(c/32)*36;return product(half(word<std::uint16_t>(b)),float(word<std::int8_t>(b+4+c%32)));}
        case TensorType::Q16_1:{const auto n=bank.q16_block_size;const auto*b=p+(c/n)*(8+2*n);return product(word<float>(b),float(word<std::int16_t>(b+8+2*(c%n))));}
        case TensorType::AQ8:{const auto code=float(word<std::int8_t>(p+4+c));return sum(product(product(word<float>(p),code),std::abs(code)),anchor[head*256+c]);}
        case TensorType::TQ4:case TensorType::TQ8:
        {
            const auto source=word<float>(p);if(source<1e-30f)return 0;
            unsigned index=0;
            if(bank.codec==TensorType::TQ8)index=p[8+c];
            else{const unsigned bit=unsigned(c*3),byte=bit/8,shift=bit%8;unsigned low=p[8+byte]>>shift;if(shift>5)low|=unsigned(p[9+byte])<<(8-shift);index=(low&7)|(((p[104+c/8]>>(c%8))&1)<<3);}
            return product(book[index],product(role==BankRole::Key?source:word<float>(p+4),0.0625f));
        }
        default:throw std::invalid_argument("bad reference codec");
        }
    }
};
/** @brief Authenticate every native fragment against independent original coordinates.
 * @tparam Encoding Original admitted codec; @param f Original owned bank.
 * @param role Original radial policy.
 */
template<TensorType Encoding>void check(Fixture&f,BankRole role)
{
    const auto before=f.storage;const auto oldBook=f.book;const auto oldAnchor=f.anchor;
    const auto binding=qwen4exp::native_bank::Binding::bind(f.bank,role,2,2);
    for(int h=0;h<2;++h)
    {
        const auto decoder=cpu::qsa_native::Decoder<Encoding>::bind(binding,h);
        for(int r=0;r<2;++r){const auto row=decoder.row(r);for(int c=0;c<256;c+=8)
        {
            std::array<float,24>out;out.fill(919.25f);std::array<std::uint32_t,8>expected;
            for(int lane=0;lane<8;++lane){expected[lane]=std::bit_cast<std::uint32_t>(f.reference(r,h,c+lane,role));out[7+lane]=std::bit_cast<float>(~expected[lane]);}
            cpu::qsa_native::store8(out.data()+7,row.read(c));
            for(int lane=0;lane<8;++lane)EXPECT_EQ(std::bit_cast<std::uint32_t>(out[7+lane]),expected[lane]);
            for(int i=0;i<24;++i)if(i<7||i>=15)EXPECT_EQ(out[i],919.25f);
        }}
    }
    EXPECT_EQ(f.storage,before);EXPECT_EQ(f.book,oldBook);EXPECT_EQ(f.anchor,oldAnchor);
}
} // namespace
/** @test Every original native codec/role/layout matches an independent separately rounded program. */
TEST(CPUQwenSparseNativeCodec,OriginalFragmentsAndRadialRolesRetainFiniteBytes)
{
    requireNative();
    for(auto layout:{QwenSparseNativeLayout::TokenMajor,QwenSparseNativeLayout::HeadMajor})for(auto role:{BankRole::Key,BankRole::Value})
    {
#define QSA_CHECK(codec) {Fixture f(TensorType::codec,32,layout);check<TensorType::codec>(f,role);}
        QSA_CHECK(FP32);QSA_CHECK(FP16);QSA_CHECK(BF16);QSA_CHECK(Q8_1);QSA_CHECK(TQ4);QSA_CHECK(TQ8);
#undef QSA_CHECK
        for(int n:{32,64,128}){Fixture f(TensorType::Q16_1,n,layout);check<TensorType::Q16_1>(f,role);}
        if(role==BankRole::Key){Fixture f(TensorType::AQ8,32,layout);check<TensorType::AQ8>(f,role);}
    }
}
/** @test Radius-first TQ scaling has observable overflow and subnormal rounding boundaries. */
TEST(CPUQwenSparseNativeCodec,OriginalRadialScaleOrderCannotBeReassociated)
{
    requireNative();
    const auto checkOrder=[]<TensorType Encoding>()
    {
        {
            Fixture f(Encoding,32,QwenSparseNativeLayout::TokenMajor);f.book.fill(2);
            put(f.data(0,0),std::numeric_limits<float>::max());
            const auto binding=qwen4exp::native_bank::Binding::bind(f.bank,BankRole::Key,2,2);
            const auto decoder=cpu::qsa_native::Decoder<Encoding>::bind(binding,0);
            std::array<float,8>output;output.fill(-919.25f);
            cpu::qsa_native::store8(output.data(),decoder.row(0).read(0));
            for(auto value:output)EXPECT_EQ(std::bit_cast<std::uint32_t>(value),0x7dffffffu);
            EXPECT_TRUE(std::isinf(product(2,std::numeric_limits<float>::max())));
        }
        {
            Fixture f(Encoding,32,QwenSparseNativeLayout::TokenMajor);f.book.fill(16);
            put(f.data(0,0),1.0f);put(f.data(0,0)+4,std::bit_cast<float>(8u));
            const auto binding=qwen4exp::native_bank::Binding::bind(f.bank,BankRole::Value,2,2);
            const auto decoder=cpu::qsa_native::Decoder<Encoding>::bind(binding,0);
            std::array<float,8>output;output.fill(-919.25f);
            cpu::qsa_native::store8(output.data(),decoder.row(0).read(0));
            for(auto value:output)EXPECT_EQ(std::bit_cast<std::uint32_t>(value),0u);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(product(product(16,std::bit_cast<float>(8u)),0.0625f)),8u);
        }
    };
    checkOrder.template operator()<TensorType::TQ4>();checkOrder.template operator()<TensorType::TQ8>();
}
/** @test Nonfinite native coordinates and invalid radial metadata cannot be repaired into finite data. */
TEST(CPUQwenSparseNativeCodec,InvalidOriginalNumericDataIsFatalBeforeFragmentPublication)
{
    requireNative();Fixture f(TensorType::FP32,32,QwenSparseNativeLayout::TokenMajor);
    const auto binding=qwen4exp::native_bank::Binding::bind(f.bank,BankRole::Key,2,2);
    const auto decoder=cpu::qsa_native::Decoder<TensorType::FP32>::bind(binding,0);
    put(f.data(0,0)+4,std::numeric_limits<float>::quiet_NaN());
    EXPECT_DEATH((void)decoder.row(0).read(0),"nonfinite numeric data");
    put(f.data(0,0)+4,1.0f);put(f.data(0,0)+7*4,std::numeric_limits<float>::quiet_NaN());
    EXPECT_DEATH((void)decoder.row(0).read(0),"nonfinite numeric data");
    put(f.data(0,0)+7*4,std::numeric_limits<float>::infinity());
    EXPECT_DEATH((void)decoder.row(0).read(0),"nonfinite numeric data");
    Fixture tq(TensorType::TQ4,32,QwenSparseNativeLayout::TokenMajor);put(tq.data(0,0),-1.0f);
    const auto tb=qwen4exp::native_bank::Binding::bind(tq.bank,BankRole::Key,2,2);
    const auto td=cpu::qsa_native::Decoder<TensorType::TQ4>::bind(tb,0);
    EXPECT_DEATH((void)td.row(0),"nonfinite numeric data");
    const auto fittedNegative=[]<TensorType Encoding>()
    {
        Fixture value(Encoding,32,QwenSparseNativeLayout::TokenMajor);put(value.data(0,0)+4,-0.5f);
        const auto vb=qwen4exp::native_bank::Binding::bind(value.bank,BankRole::Value,2,2);
        const auto vd=cpu::qsa_native::Decoder<Encoding>::bind(vb,0);
        EXPECT_DEATH((void)vd.row(0),"nonfinite numeric data");
    };
    fittedNegative.template operator()<TensorType::TQ4>();fittedNegative.template operator()<TensorType::TQ8>();
}
} // namespace llaminar2::test
