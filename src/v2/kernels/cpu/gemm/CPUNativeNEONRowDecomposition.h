/**
 * @file CPUNativeNEONRowDecomposition.h
 * @brief Exact projection keys routed to admitted existing physical row programs.
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
        CPUProjectionNumericalPolicy policy;
        CPUNativeQuantisedEncoding encoding;
        int map;
        bool minimum;
        multi_scale::NEONByteDot pair_family;
        multi_scale::NEONByteDot tail_family;
    };

    using NEONRowDot = multi_scale::NEONByteDot;
    using NEONRowPolicy = CPUProjectionNumericalPolicy;
    using NEONRowEncoding = CPUNativeQuantisedEncoding;

    /** @brief Exact source keys; rows 1/2 select one same-row leaf, rows 3/4 select two live slices. */
    inline constexpr NEONRowDecompositionKey kNEONRowDecompositions[] = {
        {3, NEONRowDot::ASIMD, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::CompactMultiScale, 11, false, NEONRowDot::ASIMD, NEONRowDot::ASIMD},
        {3, NEONRowDot::ASIMD, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::NibbleLUTSourceScale, 2, false, NEONRowDot::ASIMD, NEONRowDot::ASIMD},
        {4, NEONRowDot::ASIMD, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::CompactMultiScale, 11, false, NEONRowDot::ASIMD, NEONRowDot::ASIMD},
        {4, NEONRowDot::ASIMD, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::CompactMultiScale, 7, true, NEONRowDot::ASIMD, NEONRowDot::ASIMD},
        {4, NEONRowDot::ASIMD, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::Q4KSuperblockCoded, 1, true, NEONRowDot::ASIMD, NEONRowDot::ASIMD},
        {3, NEONRowDot::DotProd, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::CompactMultiScale, 11, false, NEONRowDot::DotProd, NEONRowDot::ASIMD},
        {3, NEONRowDot::DotProd, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::CompactMultiScale, 7, true, NEONRowDot::DotProd, NEONRowDot::DotProd},
        {3, NEONRowDot::DotProd, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::NibbleLUTSourceScale, 2, false, NEONRowDot::DotProd, NEONRowDot::DotProd},
        {3, NEONRowDot::DotProd, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::Q4KSuperblockCoded, 1, true, NEONRowDot::DotProd, NEONRowDot::DotProd},
        {4, NEONRowDot::DotProd, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::CompactMultiScale, 11, false, NEONRowDot::DotProd, NEONRowDot::DotProd},
        {4, NEONRowDot::DotProd, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::CompactMultiScale, 7, true, NEONRowDot::DotProd, NEONRowDot::DotProd},
        {4, NEONRowDot::DotProd, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::Q4KSuperblockCoded, 1, true, NEONRowDot::DotProd, NEONRowDot::DotProd},
        {4, NEONRowDot::I8MM, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::CompactMultiScale, 11, false, NEONRowDot::DotProd, NEONRowDot::DotProd},
        {1, NEONRowDot::DotProd, NEONRowPolicy::BackendNative, NEONRowEncoding::Q5NativeSingleScale, 0, false, NEONRowDot::ASIMD, NEONRowDot::ASIMD},
        {1, NEONRowDot::DotProd, NEONRowPolicy::BackendNative, NEONRowEncoding::Q5NativeSingleScale, 1, true, NEONRowDot::ASIMD, NEONRowDot::ASIMD},
        {1, NEONRowDot::DotProd, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::CompactMultiScale, 11, false, NEONRowDot::ASIMD, NEONRowDot::ASIMD},
        {1, NEONRowDot::DotProd, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::Q5NativeSingleScale, 0, false, NEONRowDot::ASIMD, NEONRowDot::ASIMD},
        {1, NEONRowDot::DotProd, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::Q5NativeSingleScale, 1, true, NEONRowDot::ASIMD, NEONRowDot::ASIMD},
        {2, NEONRowDot::I8MM, NEONRowPolicy::GPUAlignedExpert, NEONRowEncoding::CompactMultiScale, 11, false, NEONRowDot::DotProd, NEONRowDot::DotProd},
    };

    static_assert([] {
        for (const auto &source : kNEONRowDecompositions)
            for (const auto &other : kNEONRowDecompositions)
                if (source.policy == other.policy && source.encoding == other.encoding &&
                    source.map == other.map && source.minimum == other.minimum)
                {
                    const int first = source.rows <= 2 ? source.rows : 2;
                    if (other.rows == first && other.family == source.pair_family)
                        return false;
                    if (source.rows > 2 && other.rows == source.rows - 2 && other.family == source.tail_family)
                        return false;
                }
        return true;
    }(), "NEON row routing must select terminal physical leaves without chaining");

    /** @brief Compile-time exact-key selection; unlisted arithmetic programs retain their physical tile. */
    template <int Rows, NEONRowDot Family, CPUProjectionNumericalPolicy Policy,
              NEONRowEncoding Encoding, int Map, bool Minimum>
    consteval bool decomposeNEONRows()
    {
        for (const auto &key : kNEONRowDecompositions)
            if (key.rows == Rows && key.family == Family && key.policy == Policy && key.encoding == Encoding &&
                key.map == Map && key.minimum == Minimum)
                return true;
        return false;
    }
    /** @brief Resolve the pair instruction family from the same exact routing authority. */
    template <int Rows, NEONRowDot Family, CPUProjectionNumericalPolicy Policy,
              NEONRowEncoding Encoding, int Map, bool Minimum, bool Tail = false>
    consteval NEONRowDot decomposedNEONPairFamily()
    {
        static_assert(decomposeNEONRows<Rows, Family, Policy, Encoding, Map, Minimum>());
        for (const auto &key : kNEONRowDecompositions)
            if (key.rows == Rows && key.family == Family && key.policy == Policy && key.encoding == Encoding &&
                key.map == Map && key.minimum == Minimum)
                return Tail ? key.tail_family : key.pair_family;
        throw "missing exact NEON row decomposition";
    }

}
