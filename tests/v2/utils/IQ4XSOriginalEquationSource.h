/**
 * @file IQ4XSOriginalEquationSource.h
 * @brief Independent literal original-source IQ4_XS and IQ4_NL projection oracles.
 *
 * Expected factors use original source words and a literal integer grid. Tests
 * do not prescribe a replacement packed representation. One live Q8 block
 * fixes dot and reduction association while all source scale codes are swept.
 */
#pragma once
#include "tensors/TensorClasses.h"
#include "tensors/FP16Utils.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
#include <array>
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

namespace llaminar2::test::iq4xs_original_equation
{
/** @brief Original quantized source type; IQ4_NL is an unchanged alias compatibility positive. */
enum class Format { IQ4XS, IQ4NL };
constexpr int k=288, first=3;
constexpr std::array<uint16_t,20> bases{
    0x0000,0x8000,0x0001,0x8001,0x03ff,0x83ff,0x0400,0x8400,
    0x1401,0x9401,0x3555,0xb555,0x3bff,0xbbff,0x3c00,0xbc00,
    0x3c01,0xbc01,0x7bff,0xfbff};
constexpr std::array<int,16> grid{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
/** @brief Original blocks and view lifetime; expected values never use prepared state. */
struct Source
{
    Format format;
    int n;
    std::vector<IQ4_XSBlock> xs;
    std::vector<IQ4_NLBlock> nl;
    std::vector<uint8_t> bytes;
    std::shared_ptr<TensorBase> parent,view;
};
/**
 * @brief Create all signed scale codes and exceptional finite source bases.
 * @param format Explicit source type, never inferred from shared execution codebook4.
 * @return Original tensor parent/view with source-row offset and logical K tail.
 */
inline Source source(Format format)
{
    Source result{format,format==Format::IQ4XS?int(64*bases.size()+1):67};
    const int rows=result.n+first;
    if(format==Format::IQ4XS)
    {
        result.xs.resize(rows*2);
        for(int n=0;n<rows;++n) for(int sb=0;sb<2;++sb)
        {
            auto &b=result.xs[n*2+sb]; const int logical=std::max(0,n-first);
            b.d=bases[(logical/64+sb)%bases.size()];
            for(int sub=0;sub<8;++sub)
            {
                const unsigned code=(logical+sub*7+sb*13)%64;
                b.scales_l[sub/2]|=uint8_t((code&15)<<(4*(sub%2)));
                b.scales_h|=uint16_t((code>>4)<<(2*sub));
                for(int j=0;j<16;++j) b.qs[sub*16+j]=uint8_t(n*17+sb*37+sub*53+j*71);
            }
        }
        result.bytes.resize(result.xs.size()*sizeof(IQ4_XSBlock));
        std::memcpy(result.bytes.data(),result.xs.data(),result.bytes.size());
        result.parent=std::make_shared<IQ4_XSTensor>(std::vector<size_t>{size_t(rows),k},result.bytes);
    }
    else
    {
        result.nl.resize(rows*(k/32));
        for(int n=0;n<rows;++n) for(int kb=0;kb<k/32;++kb)
        {
            auto &b=result.nl[n*(k/32)+kb]; b.d=bases[(n+kb)%bases.size()];
            for(int j=0;j<16;++j) b.qs[j]=uint8_t(n*17+kb*53+j*71);
        }
        result.bytes.resize(result.nl.size()*sizeof(IQ4_NLBlock));
        std::memcpy(result.bytes.data(),result.nl.data(),result.bytes.size());
        result.parent=std::make_shared<IQ4_NLTensor>(std::vector<size_t>{size_t(rows),k},result.bytes);
    }
    result.view=result.parent->create_view({size_t(result.n),k},size_t(first)*k);
    return result;
}
/**
 * @brief Recover the original scale as an exact FP32 product.
 * @param source Immutable source blocks.
 * @param column Logical view output column.
 * @param block Logical32-value K block.
 * @return Original factor, never narrowed to a prepared half.
 */
inline float factor(const Source &source,int column,int block)
{
    if(source.format==Format::IQ4NL)
        return fp16_to_fp32(source.nl[(column+first)*(k/32)+block].d);
    const auto &b=source.xs[(column+first)*2+block/8]; const int sub=block%8;
    const int code=((b.scales_l[sub/2]>>(4*(sub%2)))&15)|(((b.scales_h>>(2*sub))&3)<<4);
    return float(double(fp16_to_fp32(b.d))*double(code-32));
}
/**
 * @brief Decode a source grid integer using a literal canonical table.
 * @param source Immutable source blocks. @param column Logical view column.
 * @param block Logical32-value K block. @param lane Position0..31 in that block.
 * @return Signed IQ4 grid integer before its scale.
 */
inline int integer(const Source &source,int column,int block,int lane)
{
    const uint8_t q=source.format==Format::IQ4XS
        ?source.xs[(column+first)*2+block/8].qs[(block%8)*16+lane%16]
        :source.nl[(column+first)*(k/32)+block].qs[lane%16];
    return grid[(q>>(lane<16?0:4))&15];
}
/**
 * @brief Evaluate one unchanged Q8 contribution with only one live integer dot.
 * @param source Original source matrix. @param column Logical output column.
 * @param active_block Only nonzero Q8 block. @param sign Activation sign +/-1.
 * @return Same single-scale FP32 multiply/multiply/add boundaries as production.
 */
inline float expected(const Source &source,int column,int active_block,int sign)
{
    const int dot=sign*(integer(source,column,active_block,0)+127*integer(source,column,active_block,31));
    const float term=device_fp32_contract::multiply(device_fp32_contract::multiply(
        1.0f,factor(source,column,active_block)),float(dot));
    return device_fp32_contract::add(0.0f,term);
}
} // namespace llaminar2::test::iq4xs_original_equation
