/**
 * @file PooledImmutableLocalHotGeometry.cpp
 * @brief Checked local-cache root offsets and sole-PMA floor/optional contributions.
 *
 * One bounded native response bank remains independent of optional cache slots.
 * Capturable all-hit readiness uses the original mapped timeline allocation
 * and one bounded response window. The original native parent owns the probe
 * fork dependency; no separately allocated probe event or count mirror exists.
 */
#include "PooledImmutableLocalHotGeometry.h"
#include "loaders/ImmutableRowTable.h"
#include "PooledImmutableLocalHotState.h"
#include "ImmutableLookupDecode.h"
#include "transfer/TransferEngine.h"
#include <limits>
#include <stdexcept>
namespace llaminar2::qwen4exp {
namespace {
/** @return Representable byte sum before exposing geometry. */
std::size_t plus(std::size_t a,std::size_t b)
{if(a>SIZE_MAX-b)throw std::overflow_error("Local hot cache byte sum overflow");return a+b;}
/** @return Representable byte product before exposing geometry. */
std::size_t times(std::size_t a,std::size_t b)
{if(a && b>SIZE_MAX/a)throw std::overflow_error("Local hot cache byte product overflow");return a*b;}
/** @return Exact requested root alignment, not allocator rounding. */
std::size_t aligned(std::size_t x){return plus(x,63)/64*64;}
/** @brief Append a named admitted range; absent arrays require no alignment padding. */
void append(std::size_t& end,std::size_t& offset,std::size_t bytes)
{if(!bytes){offset=0;return;}offset=aligned(end);end=plus(offset,bytes);}
/** @brief Merge complete immutable BOM contributions without accounting arithmetic. */
void merge(PhysicalMemoryPlanBuilder& builder,const PhysicalMemoryPlan& plan)
{for(const auto& bom:plan.resources())builder.add(bom);}
}
PooledImmutableLocalHotGeometry PooledImmutableLocalHotGeometry::declare(
    const ImmutableRowTableSource& source,PhysicalMemoryAllocatorIdentity consumer,
    std::uint32_t maximum_original_ids,std::uint32_t slots,adaptive_lookup::Policy policy)
{
    namespace a=adaptive_lookup;
    if(consumer.world_rank<0 || !consumer.device.is_gpu() || !consumer.device.is_valid() ||
        !maximum_original_ids || maximum_original_ids%16 || maximum_original_ids>a::MaximumSlots ||
        slots>a::MaximumSlots || slots>source.rows || !source.row_elements ||
        source.row_elements>static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        !policy.aging_accesses || !policy.protected_denominator ||
        policy.protected_numerator>policy.protected_denominator)
        throw std::invalid_argument("Local hot cache requires original GPU/source/full-query geometry and existing policy");
    const auto format=ggufToTensorType(source.format);
    switch(format){
#define QFN_LOCAL_HOT_FORMAT(F) case TensorType::F: break;
        QFN_LOCAL_HOT_FORMAT(FP32) QFN_LOCAL_HOT_FORMAT(FP16) QFN_LOCAL_HOT_FORMAT(BF16)
        QFN_LOOKUP_QUANT_FORMATS(QFN_LOCAL_HOT_FORMAT)
#undef QFN_LOCAL_HOT_FORMAT
    default:throw std::invalid_argument("Local hot cache source format has no original native decoder");
    }
    lookup_detail::validateGeometry(format,static_cast<int>(source.row_elements),source.rowBytes());
    PooledImmutableLocalHotGeometry g(ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()));
    g.consumer_=consumer;g.row_elements_=source.row_elements;g.maximum_=maximum_original_ids;
    g.slots_=slots;g.policy_=policy;g.stride_=plus(g.rowBytes(),3)/4*4;
    g.width_=source.rows<=0x100000000ULL?a::RowKeyWidth::UInt32:a::RowKeyWidth::UInt64;
    if(slots){
        g.lane_width_=1;while(g.lane_width_<slots)g.lane_width_<<=1;
        g.protected_=static_cast<std::uint32_t>(std::uint64_t(slots)*policy.protected_numerator/policy.protected_denominator);
        std::size_t end=0;
        append(end,g.cache_.control,sizeof(a::OwnerControl));
        append(end,g.cache_.slots,times(slots,g.width_==a::RowKeyWidth::UInt32?sizeof(a::Slot32):sizeof(a::Slot64)));
        append(end,g.cache_.nodes,times(slots-1,sizeof(a::RadixNode)));
        append(end,g.cache_.epochs,times(times(4,g.lane_width_),sizeof(std::uint32_t)));
        append(end,g.cache_.counts,times(times(4,g.lane_width_),sizeof(std::uint16_t)));
        append(end,g.cache_.payload,times(slots,g.stride_));g.cache_.bytes=aligned(end);
    }
    std::size_t end=0;
    append(end,g.frame_.control,sizeof(a::FrameControl));
    append(end,g.frame_.identity,sizeof(local_hot::FrameIdentity));
    append(end,g.frame_.counters,sizeof(local_hot::Counters));
    append(end,g.frame_.unique,times(g.maximum_,sizeof(a::UniqueRecord)));
    append(end,g.frame_.original_to_unique,times(g.maximum_,sizeof(std::uint32_t)));
    append(end,g.frame_.miss_keys,times(g.maximum_,sizeof(std::uint64_t)));
    append(end,g.frame_.response,times(g.maximum_,g.rowBytes()));
    append(end,g.frame_.result_header,sizeof(local_hot_transport::ResultHeader));
    append(end,g.frame_.received_header_bytes,sizeof(std::uint64_t));
    append(end,g.frame_.received_payload_bytes,sizeof(std::uint64_t));g.frame_.bytes=aligned(end);
    end=0;g.scratch_.radix_blocks=(g.maximum_+255)/256;
    append(end,g.scratch_.keys_a,times(g.maximum_,sizeof(a::KeyIndex)));
    append(end,g.scratch_.keys_b,times(g.maximum_,sizeof(a::KeyIndex)));
    append(end,g.scratch_.radix_counts,times(times(g.scratch_.radix_blocks,16),sizeof(std::uint32_t)));
    append(end,g.scratch_.radix_offsets,times(times(g.scratch_.radix_blocks,16),sizeof(std::uint32_t)));
    append(end,g.scratch_.scan,times(g.maximum_,sizeof(std::uint32_t)));g.scratch_.bytes=aligned(end);
    return g;
}
void PooledImmutableLocalHotGeometry::require(PhysicalMemoryResource consumer,PhysicalMemoryResource host)const
{
    if(!consumer.valid() || !host.valid() || consumer.world_rank!=consumer_.world_rank ||
        consumer.device!=consumer_.device || host.world_rank!=consumer_.world_rank || host.device!=DeviceId::cpu())
        throw std::invalid_argument("Local hot cache changed original CPU/GPU allocator observations");
}
PhysicalMemoryPlan PooledImmutableLocalHotGeometry::mandatoryBOM(
    PhysicalMemoryResource consumer,PhysicalMemoryResource host)const
{return mandatoryBOM(consumer,host,NativeOrderingEventBudgetProjection::installed());}
PhysicalMemoryPlan PooledImmutableLocalHotGeometry::mandatoryBOM(PhysicalMemoryResource consumer,
    PhysicalMemoryResource host,const NativeOrderingEventBudgetProjection& projection)const
{
    require(consumer,host);PhysicalMemoryPlanBuilder builder;
    builder.add(consumer,PhysicalMemoryOwner::ImmutableLookupStaging,plus(frame_.bytes,scratch_.bytes));
    builder.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,TransferEngine::mappedHostRegionAllocationBytes(sizeof(std::uint64_t)));
    merge(builder,projection.bom(NativeOrderingEventProgram::pooledLocalHot(false),host,consumer));
    return builder.build();
}
PhysicalMemoryPlan PooledImmutableLocalHotGeometry::slotDependentBOM(
    PhysicalMemoryResource consumer,PhysicalMemoryResource host)const
{return slotDependentBOM(consumer,host,NativeOrderingEventBudgetProjection::installed());}
PhysicalMemoryPlan PooledImmutableLocalHotGeometry::slotDependentBOM(PhysicalMemoryResource consumer,
    PhysicalMemoryResource host,const NativeOrderingEventBudgetProjection& projection)const
{
    require(consumer,host);PhysicalMemoryPlanBuilder builder;
    if(cache_.bytes){builder.add(consumer,PhysicalMemoryOwner::ImmutableTableStorage,cache_.bytes);
        merge(builder,projection.growthBOM(NativeOrderingEventProgram::pooledLocalHot(false),
            NativeOrderingEventProgram::pooledLocalHot(true),host,consumer));}
    return builder.build();
}
} // namespace llaminar2::qwen4exp
