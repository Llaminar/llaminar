/**
 * @file IQ3SOriginalEquationSource.h
 * @brief Independent original-source IQ3_S factors and literal-grid dot oracles.
 *
 * Expected factors read only original half bases and source nibble words. No
 * prepared scale, new metadata helper or production weight decoder is an oracle.
 * One live Q8 block fixes dot/reduction association; source rows retain a K tail.
 */
#pragma once
#include "tensors/TensorClasses.h"
#include "tensors/FP16Utils.h"
#include "tensors/IQQuantTables.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
#include <array>
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

namespace llaminar2::test::iq3s_original_equation
{
constexpr int k=288, first=3;
constexpr std::array<uint16_t,20> bases{
    0x0000,0x8000,0x0001,0x8001,0x03ff,0x83ff,0x0400,0x8400,
    0x1401,0x9401,0x3555,0xb555,0x3bff,0xbbff,0x3c00,0xbc00,
    0x3c01,0xbc01,0x7bff,0xfbff};
/** @brief Original blocks and parent/view lifetimes; no prepared values retained. */
struct Source
{
    int n=16*int(bases.size())+1;
    std::vector<IQ3_SBlock> blocks;
    std::vector<uint8_t> bytes;
    std::shared_ptr<TensorBase> parent,view;
};
/**
 * @brief Create all16 source coefficients across20 finite half-base domains.
 * @return Original source parent/view with row offset3 and K288 logical tail.
 */
inline Source source()
{
    Source result;const int rows=result.n+first;
    result.blocks.resize(rows*2);
    for(int n=0;n<rows;++n) for(int sb=0;sb<2;++sb)
    {
        auto &b=result.blocks[n*2+sb];const int logical=std::max(0,n-first);
        b.d=bases[(logical/16+sb)%bases.size()];
        for(int sub=0;sub<8;++sub)
        {
            const unsigned nibble=(logical+sub*7+sb*13)%16;
            b.scales[sub/2]|=uint8_t(nibble<<(4*(sub%2)));
            for(int group=0;group<8;++group)
            {
                const unsigned index=(logical*13+group*37+sub*71+sb*113)%512;
                b.qs[sub*8+group]=uint8_t(index);
                b.qh[sub]|=uint8_t((index>>8)<<group);
                const unsigned signs=(logical+group*3+sub*5+sb*7)&15;
                b.signs[sub*4+group/2]|=uint8_t(signs<<(4*(group%2)));
            }
        }
    }
    // A separate ordinary literal produces384.375 versus legacy-half384.5.
    // Preserve the complete coefficient/domain sweep above in the other rows.
    auto &literal=result.blocks[(rows-1)*2];literal={};literal.d=0x3c01u;
    for(int sub=0;sub<8;++sub)literal.scales[sub/2]|=uint8_t(1u<<(4*(sub%2)));
    result.bytes.resize(result.blocks.size()*sizeof(IQ3_SBlock));
    std::memcpy(result.bytes.data(),result.blocks.data(),result.bytes.size());
    result.parent=std::make_shared<IQ3_STensor>(std::vector<size_t>{size_t(rows),k},result.bytes);
    result.view=result.parent->create_view({size_t(result.n),k},size_t(first)*k);
    return result;
}
/**
 * @brief Recover the exact original half-times-odd factor in FP32.
 * @param source Immutable original blocks. @param column Logical view column.
 * @param block Logical32-value K block. @return Original FP32 factor, never half-rounded.
 */
inline float factor(const Source &source,int column,int block)
{
    const auto &b=source.blocks[(column+first)*2+block/8];const int sub=block%8;
    const int nibble=(b.scales[sub/2]>>(4*(sub%2)))&15;
    return float(double(fp16_to_fp32(b.d))*double(1+2*nibble));
}
/**
 * @brief Decode original signed grid integers without any prepared resolver.
 * @param source Original source blocks. @param column Logical view column.
 * @param block Logical K32 block. @param lane Original position0..31.
 * @return Original signed integer selected by nine-bit grid index and sign nibble.
 */
inline int integer(const Source &source,int column,int block,int lane)
{
    const auto &b=source.blocks[(column+first)*2+block/8];const int sub=block%8,group=lane/4;
    const unsigned index=b.qs[sub*8+group]|(((b.qh[sub]>>group)&1u)<<8);
    const int magnitude=int((iq3s_grid[index]>>(8*(lane%4)))&255u);
    const unsigned signs=b.signs[sub*4+group/2]>>(4*(group%2));
    return (signs&(1u<<(lane%4)))?-magnitude:magnitude;
}
/**
 * @brief Evaluate the unchanged Q8 contribution with exactly one live dot.
 * @param source Original matrix. @param column Logical output column.
 * @param active_block Nonzero Q8 block. @param sign Activation sign +/-1.
 * @return Canonical multiply/multiply/add result using only original source factors.
 */
inline float expected(const Source &source,int column,int active_block,int sign)
{
    const int dot=sign*(integer(source,column,active_block,0)+127*integer(source,column,active_block,31));
    const float term=device_fp32_contract::multiply(device_fp32_contract::multiply(
        1.0f,factor(source,column,active_block)),float(dot));
    return device_fp32_contract::add(0.0f,term);
}
} // namespace llaminar2::test::iq3s_original_equation
