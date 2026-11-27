/**
 * @file CPUNativeNEONRowDecomposition.h
 * @brief Exact ordered projection keys executed as two existing physical row programs.
 *
 * Row decomposition changes only payload reuse, never a row's K partitions,
 * integer products, scale operations or FP fold order. It is a static execution
 * policy, not an error recovery path. Logical batch scheduling remains unchanged.
 */
#pragma once

#include "CPUNativeQuantisedMultiScaleKernels.h"
#include <array>

namespace llaminar2::cpu::native_quantised
{
    /** @brief One exact source program and the instruction family of its two-row leaf. */
    struct NEONRowDecompositionKey
    {
        int rows;
        multi_scale::NEONByteDot family;
        CPUNativeQuantisedEncoding encoding;
        int map;
        bool minimum;
        multi_scale::NEONByteDot pair_family;
    };

    using NEONRowDot = multi_scale::NEONByteDot;
    using NEONRowEncoding = CPUNativeQuantisedEncoding;

    /** @brief Ordered-only source keys; absent keys retain their original physical tile. */
    inline constexpr NEONRowDecompositionKey kNEONRowDecompositions[] = {
        {3, NEONRowDot::ASIMD, NEONRowEncoding::CompactMultiScale, 11, false, NEONRowDot::ASIMD},
        {3, NEONRowDot::ASIMD, NEONRowEncoding::NibbleLUTSourceScale, 2, false, NEONRowDot::ASIMD},
        {4, NEONRowDot::ASIMD, NEONRowEncoding::CompactMultiScale, 11, false, NEONRowDot::ASIMD},
        {4, NEONRowDot::ASIMD, NEONRowEncoding::CompactMultiScale, 7, true, NEONRowDot::ASIMD},
        {4, NEONRowDot::ASIMD, NEONRowEncoding::Q4KSuperblockCoded, 1, true, NEONRowDot::ASIMD},
        {3, NEONRowDot::DotProd, NEONRowEncoding::CompactMultiScale, 11, false, NEONRowDot::DotProd},
        {3, NEONRowDot::DotProd, NEONRowEncoding::CompactMultiScale, 7, true, NEONRowDot::DotProd},
        {3, NEONRowDot::DotProd, NEONRowEncoding::NibbleLUTSourceScale, 2, false, NEONRowDot::DotProd},
        {3, NEONRowDot::DotProd, NEONRowEncoding::Q4KSuperblockCoded, 1, true, NEONRowDot::DotProd},
        {4, NEONRowDot::DotProd, NEONRowEncoding::CompactMultiScale, 11, false, NEONRowDot::DotProd},
        {4, NEONRowDot::DotProd, NEONRowEncoding::CompactMultiScale, 7, true, NEONRowDot::DotProd},
        {4, NEONRowDot::DotProd, NEONRowEncoding::Q4KSuperblockCoded, 1, true, NEONRowDot::DotProd},
        {4, NEONRowDot::I8MM, NEONRowEncoding::CompactMultiScale, 11, false, NEONRowDot::DotProd},
    };

    /** @brief Compile-time selection; backend-native arithmetic is never decomposed. */
    template <int Rows, NEONRowDot Family, CPUProjectionNumericalPolicy Policy,
              NEONRowEncoding Encoding, int Map, bool Minimum>
    consteval bool decomposeNEONRows()
    {
        if constexpr (Policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
            for (const auto &key : kNEONRowDecompositions)
                if (key.rows == Rows && key.family == Family && key.encoding == Encoding &&
                    key.map == Map && key.minimum == Minimum)
                    return true;
        return false;
    }
}
