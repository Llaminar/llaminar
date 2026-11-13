/**
 * @file PooledImmutableLocalHotTransportGeometry.h
 * @brief Exact optional compact-reader transport geometry contributed to the sole PMA.
 *
 * Engaged local-hot transport REPLACES the replicated expanded-FP32 reader BOM.
 * Original reader geometry remains the exact roster/full-query identity. Local
 * miss keys, inverse maps and response staging belong to the actual L1 roots;
 * this declaration prices only additional source scratch and wire metadata.
 */
#pragma once
#include "PooledImmutableReplicatedReaderGeometry.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotTransportProtocol.h"
#include "loaders/ImmutableRowTableExtent.h"
#include <array>
#include <limits>
#include <optional>

namespace llaminar2::qwen4exp {
/** @brief Checked cold byte geometry; no live count, slot selection or allocation decision. */
class PooledImmutableLocalHotTransportGeometry final {
public:
    /**
     * @brief Project the original concatenated miss capacity for construction and terminal validation.
     * @param maximumOriginal Exact original sixteen-head query capacity per consumer.
     * @param consumers Actual compute population; storage-only owners do not multiply input rows.
     * @return The same checked N bound used by the original coordinator, never a live transfer count.
     * @throws std::invalid_argument For an incomplete query, absent/excess consumers or unrepresentable N.
     */
    static std::uint32_t compactOriginalCapacity(std::uint32_t maximumOriginal,std::size_t consumers)
    {
        if(!maximumOriginal || maximumOriginal%16 || !consumers || consumers>8)
            throw std::invalid_argument("Compact miss capacity requires its complete original query and consumer roster");
        const auto n=product(maximumOriginal,consumers);
        if(n>adaptive_lookup::MaximumSlots)
            throw std::invalid_argument("Compact reader union exceeds original pooled index geometry");
        return static_cast<std::uint32_t>(n);
    }
    /**
     * @brief Preserve original roster and native source extent while declaring compact scratch.
     * @param readers Exact existing source/reader placement, not inferred GPU ordinals.
     * @param source Original loader-authenticated native extent (never FP32 row width).
     * @return Geometry whose N=sum reader miss maxima, while original U remains unchanged.
     * @throws std::invalid_argument For unsupported/foreign or nonrepresentable geometry.
     */
    static PooledImmutableLocalHotTransportGeometry declare(
        PooledImmutableReplicatedReaderGeometry readers,ImmutableRowTableExtent source)
    {
        if(readers.consumers().size()>8 || !source.rows() || !source.rowBytes())
            throw std::invalid_argument("Compact reader transport requires its original bounded roster/native source");
        const auto n=compactOriginalCapacity(readers.maximumOriginal(),readers.consumers().size());
        const auto coordinator=readers.coordinator();
        const auto maximumOriginal=readers.maximumOriginal();
        std::vector<PhysicalMemoryAllocatorIdentity> consumers(readers.consumers().begin(),readers.consumers().end());
        return {coordinator,std::move(consumers),maximumOriginal,std::move(readers),source,static_cast<std::uint32_t>(n)};
    }
    /**
     * @brief Declare the genuine one-consumer local path without a fake replicated reader.
     * @param consumer Exact original source/compute allocator. @param maximumOriginal Full query envelope.
     * @param source Original native extent. @return No remote transport population; original source-self metadata remains explicit.
     */
    static PooledImmutableLocalHotTransportGeometry singleton(PhysicalMemoryAllocatorIdentity consumer,
        std::uint32_t maximumOriginal,ImmutableRowTableExtent source)
    {
        if(consumer.world_rank<0 || !consumer.device.is_gpu() || !maximumOriginal || maximumOriginal%16 ||
            maximumOriginal>adaptive_lookup::MaximumSlots)
            throw std::invalid_argument("Singleton local-hot transport requires its actual complete source consumer");
        return {consumer,{consumer},maximumOriginal,std::nullopt,source,compactOriginalCapacity(maximumOriginal,1)};
    }
    /** @return Actual original consumer population, including a genuine singleton. */
    std::span<const PhysicalMemoryAllocatorIdentity> consumers()const noexcept{return consumers_;}
    /** @return The actual source coordinator, never an inferred first ordinal. */
    PhysicalMemoryAllocatorIdentity coordinator()const noexcept{return coordinator_;}
    /** @return Same original reader roster/full sixteen-head query envelope. */
    const auto& originalReaders()const noexcept{return readers_;}
    /** @return Exact original native format/row extent; physical materialization authenticates its source. */
    const auto& sourceExtent()const noexcept{return source_;}
    /** @return Distinct source rows bounded by the original model query, never multiplied by readers. */
    std::uint32_t maximumPoolUniqueRows()const noexcept{return maximum_original_;}
    /** @return Concatenated compact miss envelope for original coordinator N scratch only. */
    std::uint32_t maximumPoolOriginalRows()const noexcept{return union_rows_;}
    /** @return Native metadata for exactly the original consumer coordinate. */
    local_hot_transport::ConsumerGeometry consumer(std::size_t coordinate,std::uint64_t originalConsumerStorageIdentity)const
    {
        if(coordinate>=consumers_.size() || !originalConsumerStorageIdentity)throw std::invalid_argument("Compact transport names a foreign original reader");
        return {static_cast<std::uint32_t>(coordinate),static_cast<std::uint32_t>(consumers_.size()),
            maximum_original_,maximum_original_,source_.rows(),source_.rowBytes(),
            static_cast<std::uint32_t>(source_.format()),originalConsumerStorageIdentity};
    }
    /** @return Only live keys are communicated; this is their admitted per-reader capacity. */
    std::size_t keyCapacityBytes()const{return product(maximum_original_,sizeof(std::uint64_t));}
    /** @return Exact maximum compact native response, never an FP32/model output span. */
    std::size_t resultCapacityBytes()const{return product(maximum_original_,source_.rowBytes());}
    /** @return Source-owned merged key root, separate from original coordinator sort/inverse scratch. */
    std::size_t mergedKeyBytes()const{return consumers_.size()==1?0:product(union_rows_,sizeof(std::uint64_t));}
    /** @return Remote ingress windows only; source-self keys are borrowed from its actual L1 root. */
    std::size_t incomingKeyBytes()const{return product(consumers_.size()-1,keyCapacityBytes());}
    /** @return Remote packed native results only; source-self packs directly into its actual L1 staging root. */
    std::size_t outgoingNativeBytes()const{return product(consumers_.size()-1,resultCapacityBytes());}
    /**
     * @brief Replace old reader result/ACK BOM with compact source scratch and endpoint control.
     * @param host Original mapped transport host observation.
     * @param devices Exact original local consumer observations, no free-memory probes.
     * @return Typed mandatory contribution, excluding L1 keys/result roots and separately priced native operands.
     */
    PhysicalMemoryPlan completeBOM(PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> devices)const
    {
        if(host.world_rank!=coordinator_.world_rank || host.device!=DeviceId::cpu())
            throw std::invalid_argument("Compact transport lost its original host allocator");
        const auto resource=[&](PhysicalMemoryAllocatorIdentity id){
            const PhysicalMemoryResource* found=nullptr;
            for(const auto& r:devices)if(r.world_rank==id.world_rank && r.device==id.device){
                if(found)throw std::invalid_argument("Duplicate compact reader PMA resource");found=&r;
            }
            if(!found)throw std::invalid_argument("Missing compact reader PMA resource");return *found;
        };
        auto source=resource(coordinator_);PhysicalMemoryPlanBuilder out;
        // Source-self must author/retain its actual Request88 even for a singleton.
        out.add(source,PhysicalMemoryOwner::ActivationTransportStaging,sizeof(local_hot_transport::Control));
        out.add(source,PhysicalMemoryOwner::ActivationTransportStaging,sizeof(local_hot_transport::UnionControl));
        if(consumers_.size()==1)return out.build();
        out.add(source,PhysicalMemoryOwner::ActivationTransportStaging,mergedKeyBytes());
        out.add(source,PhysicalMemoryOwner::ActivationTransportStaging,incomingKeyBytes());
        out.add(source,PhysicalMemoryOwner::ActivationTransportStaging,outgoingNativeBytes());
        for(const auto id:consumers_){
            const auto peer=resource(id);if(id==coordinator_)continue;
            out.add(source,PhysicalMemoryOwner::ActivationTransportStaging,sizeof(local_hot_transport::Control));
            out.add(peer,PhysicalMemoryOwner::ActivationTransportStaging,sizeof(local_hot_transport::Control));

        }
        return out.build();
    }
private:
    /** @brief Construct only after the exact original roster/native products are checked. */
    PooledImmutableLocalHotTransportGeometry(PhysicalMemoryAllocatorIdentity coordinator,
        std::vector<PhysicalMemoryAllocatorIdentity> consumers,std::uint32_t maximumOriginal,
        std::optional<PooledImmutableReplicatedReaderGeometry> readers,ImmutableRowTableExtent source,std::uint32_t rows)
        :coordinator_(coordinator),consumers_(std::move(consumers)),maximum_original_(maximumOriginal),
        readers_(std::move(readers)),source_(source),union_rows_(rows){}
    /** @return Checked cold byte product shared by declaration and actual creator bounds. */
    static std::size_t product(std::size_t a,std::size_t b){
        if(b && a>std::numeric_limits<std::size_t>::max()/b)throw std::overflow_error("Compact native transport byte geometry overflows");return a*b;
    }
    PhysicalMemoryAllocatorIdentity coordinator_;
    std::vector<PhysicalMemoryAllocatorIdentity> consumers_;
    std::uint32_t maximum_original_;
    std::optional<PooledImmutableReplicatedReaderGeometry> readers_;
    ImmutableRowTableExtent source_;
    std::uint32_t union_rows_;
};
} // namespace llaminar2::qwen4exp
