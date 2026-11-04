/**
 * @file ExpertTierEncodedColumnOracle.h
 * @brief Independent test inspection of every live native data and metadata bit.
 *
 * This does not call production gather/scatter/copy or decode floating values.
 * Original 64-output physical planes are inspected per logical column. Shared
 * Q6 words contribute only that column's actual bits; dirty inactive columns
 * never become a weight-equivalence or communication extent.
 */
#pragma once
#include "kernels/cpu/gemm/CPUNativeVNNIPreparedFootprint.h"
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>
namespace llaminar2::test {
/**
 * @brief Inspect one original logical native column/block independently of movement code.
 * @param root Genuine complete physical native root, validated by the owning fixture.
 * @param layout Actual K/data/block stride and native encoding of that root.
 * @param n Original local output column, including padding only for a dirty-neighbor oracle.
 * @param kb Actual native32 K-block coordinate.
 * @return Every data/comp/scale/min/grid bit for this column, excluding neighbors.
 */
template<class Layout>
std::vector<uint8_t> encodedTierColumn(std::span<const uint8_t> root,
    const Layout& layout,size_t n,size_t kb)
{
    const size_t unit=((n/64)*(layout.cpu_K/32)+kb)*layout.cpu_block_stride,column=n%64;
    const bool q6=layout.cpu_encoding==cpu::native_vnni::CPUNativeVNNIEncoding::Q6KNativeDualScale;
    std::vector<uint8_t> result;
    for(size_t byte=0;byte<(q6?16:layout.cpu_data_stride/64);++byte)
        result.push_back(root[unit+(byte/4)*256+column*4+byte%4]);
    if(q6)for(size_t group=0;group<8;++group) {
        uint64_t low,high;const size_t offset=unit+1024+group*64+(column/16)*16;
        std::memcpy(&low,root.data()+offset,8);std::memcpy(&high,root.data()+offset+8,8);
        const auto bit=(column%16)*4;
        result.push_back(static_cast<uint8_t>(((low>>bit)&15)|(((high>>bit)&15)<<4)));
    }
    const bool compact=layout.cpu_encoding==cpu::native_vnni::CPUNativeVNNIEncoding::CompactMultiScale;
    for(size_t plane=0;plane<(compact?2:(layout.cpu_block_stride-layout.cpu_data_stride)/128);++plane)
        for(size_t byte=0;byte<2;++byte)result.push_back(root[unit+layout.cpu_data_stride+plane*128+column*2+byte]);
    if(compact)for(size_t byte=0;byte<4;++byte)result.push_back(root[unit+layout.cpu_data_stride+256+column*4+byte]);
    return result;
}
} // namespace llaminar2::test
