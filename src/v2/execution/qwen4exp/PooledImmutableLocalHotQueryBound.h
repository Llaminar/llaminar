/**
 * @file PooledImmutableLocalHotQueryBound.h
 * @brief Immutable compact launch extent borrowed from the actual canonical query.
 *
 * Storage retains its original admitted worst-case size. This value only bounds
 * one captured invocation: every consumer's authenticated misses are a subset
 * of the source canonical rows, additionally bounded by its actual local root.
 * No device count is read, no capacity is allocated and no protocol is issued.
 */
#pragma once
#include "kernels/common/DeviceRowRange.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotTransportProtocol.h"
#include <array>
#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>
namespace llaminar2::qwen4exp {
/** @brief Exact cold canonical-row and per-consumer compact upper bounds. */
class PooledImmutableLocalHotQueryBound final {
public:
    /**
     * @brief Bound each real consumer by its original root and canonical subset.
     * @param tokens Actual source capture's complete original token-row descriptor.
     * @param consumers Immutable original coordinate-ordered consumer declarations.
     * @param storedUnionCapacity Existing admitted union storage bound, unchanged.
     * @return Capture-only bound with checked sum; live empty consumers stay legal.
     * @throws std::invalid_argument For foreign/partial rows, roster or storage bounds.
     */
    static PooledImmutableLocalHotQueryBound forCanonicalQuery(DeviceRowRange tokens,
        std::span<const local_hot_transport::ConsumerGeometry> consumers,std::uint32_t storedUnionCapacity)
    {
        const auto rows=tokens.captureIdentity();
        if(rows.first_physical_row || rows.repeats_per_row!=1 || rows.physical_rows!=rows.count_capacity ||
           tokens.capacity()>std::numeric_limits<int>::max()/16 || consumers.empty() || consumers.size()>8)
            throw std::invalid_argument("Compact query launch bound requires complete original token rows and consumer roster");
        const auto original=static_cast<std::uint32_t>(tokens.capacity())*16;
        std::array<std::uint32_t,8> maxima{};std::uint64_t total=0;
        for(std::size_t i=0;i<consumers.size();++i){
            const auto& consumer=consumers[i];
            if(!local_hot_transport::valid(consumer) || consumer.coordinate!=i || consumer.consumers!=consumers.size())
                throw std::invalid_argument("Compact query launch bound changed original consumer coordinates");
            maxima[i]=std::min(original,consumer.maximum_misses);total+=maxima[i];
        }
        if(!storedUnionCapacity || !total || total>storedUnionCapacity || total>std::uint32_t{INT32_MAX})
            throw std::invalid_argument("Compact query launch bound exceeds its original admitted union");
        return {rows,original,maxima,static_cast<std::uint32_t>(consumers.size()),static_cast<std::uint32_t>(total)};
    }
    /** @return Canonical source original-ID upper bound (token rows times sixteen). */
    std::uint32_t canonicalOriginal()const noexcept{return canonical_original_;}
    /** @return Checked sum of per-consumer subset bounds, never a live count. */
    std::uint32_t maximumOriginal()const noexcept{return union_original_;}
    /** @return Actual immutable consumer population. */
    std::uint32_t consumers()const noexcept{return consumers_;}
    /** @return One coordinate's declared compact bound; rejects foreign coordinates. */
    std::uint32_t consumerMaximum(std::uint32_t coordinate)const
    {if(coordinate>=consumers_)throw std::out_of_range("Compact query bound consumer is foreign");return maxima_[coordinate];}
    /** @return Complete original token descriptor identity, including its exact count producer. */
    const DeviceRowRange::CaptureIdentity& tokenIdentity()const noexcept{return rows_;}
    /** @brief Value equality includes every embedded pointer and declared launch geometry. */
    bool operator==(const PooledImmutableLocalHotQueryBound&)const=default;
private:
    /** @brief Only checked canonical-subset construction may form this cold value. */
    PooledImmutableLocalHotQueryBound(DeviceRowRange::CaptureIdentity rows,std::uint32_t canonical,
        std::array<std::uint32_t,8> maxima,std::uint32_t consumers,std::uint32_t total)
        :rows_(rows),canonical_original_(canonical),maxima_(maxima),consumers_(consumers),union_original_(total){}
    DeviceRowRange::CaptureIdentity rows_;
    std::uint32_t canonical_original_;
    std::array<std::uint32_t,8> maxima_;
    std::uint32_t consumers_,union_original_;
};
} // namespace llaminar2::qwen4exp
