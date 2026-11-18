/** @file ROCmIQ3XXSLookupBatch.h
 * @brief Issue the eight original IQ3_XXS grid words before consuming any.
 */
#pragma once
#include <cstdint>

namespace llaminar2::rocm {
/** @brief Fetch eight four-byte codebook entries from two packed index words.
 * @pre The caller holds the original immutable 256-entry IQ3_XXS table.
 * The joint operand constraint lets all lane-indexed loads issue before sign
 * application and the first dependent dot. Entry order remains unchanged.
 */
__device__ __forceinline__ void loadIQ3XXSGridBatch(
    std::uint32_t low,std::uint32_t high,const std::uint32_t* table,std::uint32_t (&entry)[8])
{
#pragma unroll
    for(int i=0;i<8;++i)
        entry[i]=table[((i<4?low:high)>>(8*(i&3)))&255];
#if defined(__gfx906__)
    asm volatile("" : "+v"(entry[0]),"+v"(entry[1]),"+v"(entry[2]),"+v"(entry[3]),
                       "+v"(entry[4]),"+v"(entry[5]),"+v"(entry[6]),"+v"(entry[7]));
#endif
}
}
