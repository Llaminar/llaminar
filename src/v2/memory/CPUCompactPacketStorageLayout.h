/**
 * @file CPUCompactPacketStorageLayout.h
 * @brief Shared physical declaration of one independently coherent CPU expert packet.
 *
 * Per-layer stages retain one packet; serial arenas retain explicit families of
 * packets. Both price and materialize these same four root declarations. The
 * type owns immutable geometry, never allocation capacity or a live byte ledger.
 * Per-layer CPU packet roots retain their own exact backing allocation claims.
 */
#pragma once
#include "memory/CPUStorageRoot.h"
#include <limits>
#include <initializer_list>
namespace llaminar2
{
/** @brief Four independent native FP32 roots; top-k expands metadata, not hidden rows. */
struct CPUCompactPacketStorageLayout
{
    const size_t rows; ///< Immutable token capacity.
    CPUStorageRoot<float> hidden,indices,weights,output;
    /** @brief Compare complete root geometry and policy.
     * @param other Independently declared packet.
     * @return Whether token capacity and all four native root declarations match.
     */
    bool operator==(const CPUCompactPacketStorageLayout& other) const = default;
    /**
     * @brief Declare one packet without allocating its backing storage.
     * @param rows Positive flattened token-row capacity.
     * @param hidden_width Positive model compute width.
     * @param top_k Positive metadata route width per token.
     * @param allocator Exact physical CPU allocator.
     * @return Immutable declarations used by both BOM and construction.
     * @throws std::invalid_argument For invalid dimensions or allocator.
     * @throws std::overflow_error For unrepresentable root capacities.
     */
    [[nodiscard]] static CPUCompactPacketStorageLayout declare(size_t rows,int hidden_width,int top_k,DeviceId allocator)
    {
        if(!rows || hidden_width<=0 || top_k<=0 || !allocator.is_cpu())
            throw std::invalid_argument("CPU compact packet requires exact positive CPU geometry");
        const auto multiply=[](size_t a,size_t b) {
            if(a && b>std::numeric_limits<size_t>::max()/a)
                throw std::overflow_error("CPU compact packet root extent overflow");
            return a*b;
        };
        const auto root=[allocator](size_t count) {
            return CPUStorageRoot<float>(count,PhysicalMemoryOwner::ActivationArena,allocator);
        };
        const auto hidden_elements=multiply(rows,static_cast<size_t>(hidden_width));
        const auto route_elements=multiply(rows,static_cast<size_t>(top_k));
        return {rows,root(hidden_elements),root(route_elements),root(route_elements),root(hidden_elements)};
    }
    /** @return Exact independently rounded backing bytes of all four roots. */
    [[nodiscard]] size_t physicalBytes() const
    {
        size_t bytes=0;
        for(const auto value:{hidden.physicalBytes(),indices.physicalBytes(),weights.physicalBytes(),output.physicalBytes()})
        {
            if(value>std::numeric_limits<size_t>::max()-bytes)
                throw std::overflow_error("CPU compact packet byte sum overflow");
            bytes+=value;
        }
        return bytes;
    }
};
}
