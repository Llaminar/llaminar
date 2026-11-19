/**
 * @file BoundedCenteredQ4KSource.h
 * @brief Explicit bounded native GGUF source fixture, independent of engine preparation.
 *
 * Each original32-value group contains complementary nibble pairs. Exact dyadic
 * source scales/mins make their decoded sum zero and bound |w| by30/4096.
 * This authors source blocks once; no prepared weight or activation is rescaled.
 */
#pragma once
#include "tensors/TensorClasses.h"
#include <array>
#include <cstring>
#include <memory>
#include <limits>
#include <stdexcept>
namespace llaminar2::test {
/**
 * @brief Author one valid original256-element Q4_K block.
 * @param ordinal Source block coordinate; independent rows/experts retain distinct values.
 * @param seed Immutable role-specific fixture seed.
 * @return Raw GGUF block with original low/high-half packing and six-bit fields.
 */
inline Q4_KBlock boundedCenteredQ4KBlock(std::size_t ordinal,std::uint32_t seed) {
    Q4_KBlock b{};b.d=0x0c00;b.dmin=0x0800; // half2^-12 and2^-13
    std::array<unsigned,8> scale{},minimum{};
    for(unsigned group=0;group<8;++group) {
        scale[group]=1+unsigned((ordinal*7+seed+group*3)%4);
        minimum[group]=15*scale[group];
    }
    for(unsigned i=0;i<4;++i) {
        b.scales[i]=std::uint8_t(scale[i]|((scale[i+4]>>4)<<6));
        b.scales[i+4]=std::uint8_t(minimum[i]|((minimum[i+4]>>4)<<6));
        b.scales[i+8]=std::uint8_t((scale[i+4]&15)|((minimum[i+4]&15)<<4));
    }
    for(unsigned group=0;group<8;++group)for(unsigned pair=0;pair<16;++pair) {
        const unsigned q=unsigned((pair*5+ordinal*11+seed+group*3)%16);
        for(unsigned side=0;side<2;++side) {
            const unsigned lane=pair*2+side,value=side?15-q:q;
            const unsigned index=(group/2)*32+lane,shift=(group%2)*4;
            b.qs[index]|=std::uint8_t(value<<shift);
        }
    }
    return b;
}
/**
 * @brief Create the sole authored source consumed identically by both engines.
 * @param shape Original[N,K] matrix. @param seed Immutable role seed.
 * @return Native Q4_K tensor with no FP32 source conversion or GPU representation.
 */
inline std::unique_ptr<TensorBase> makeBoundedCenteredQ4K(const std::vector<size_t>& shape,std::uint32_t seed) {
    if(shape.size()!=2||!shape[0]||!shape[1]||shape[1]%256)
        throw std::invalid_argument("Bounded Q4_K source requires complete original256-column groups");
    if(shape[0]>std::numeric_limits<size_t>::max()/(shape[1]/256)/sizeof(Q4_KBlock))
        throw std::invalid_argument("Bounded Q4_K source byte extent overflows");
    const auto blocks=shape[0]*(shape[1]/256);
    AlignedVector<std::uint8_t> bytes(blocks*sizeof(Q4_KBlock));
    for(std::size_t i=0;i<blocks;++i) {
        const auto block=boundedCenteredQ4KBlock(i,seed);
        std::memcpy(bytes.data()+i*sizeof(block),&block,sizeof(block));
    }
    return std::make_unique<Q4_KTensor>(shape,std::move(bytes));
}
}
