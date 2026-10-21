/**
 * @file ImmutableFixedHeadLookupKernels.cpp
 * @brief Setup-only fixed-head projections and complete capture operand checks.
 *
 * These checks never dereference device payload or live counters. Placement,
 * physical admission and source lifetime remain with their existing owners.
 * Native capacity is distinct from the validated per-replay transport extent.
 */
#include "ImmutableFixedHeadLookupKernels.h"
#include <array>
#include <limits>

namespace llaminar2::qwen4exp
{
ImmutableFixedHeadLookupPlan ImmutableFixedHeadLookupPlan::bind(ImmutableResidentTableRangeView table,
    NgramHashPolicy policy,int participants,int participant)
{
    policy.validate();
    if((participants!=2&&participants!=4&&participants!=8)||participant<0||participant>=participants)
        throw std::invalid_argument("Fixed-head lookup requires an ordered two/four/eight participant cohort");
    const int heads=kNgramHeads/participants,first=participant*heads;
    for(int h=0;h<kNgramHeads;++h)
        if(policy.offsets[h]>table.sourceRowCount()||policy.moduli[h]>table.sourceRowCount()-policy.offsets[h])
            throw std::invalid_argument("Fixed-head lookup source does not cover original hash vocabulary");
    const auto begin=participant==0?0:policy.offsets[first];
    const auto end=participant+1==participants?table.sourceRowCount():policy.offsets[first+heads];
    if(table.globalRowBegin()!=begin||table.rowCount()!=end-begin)
        throw std::invalid_argument("Fixed-head lookup range differs from original gap-preserving head ownership");
    return ImmutableFixedHeadLookupPlan(table,policy,participants,participant);
}

bool ImmutableFixedHeadLookupPlan::operator==(const ImmutableFixedHeadLookupPlan &o) const noexcept
{
    return table_==o.table_&&participants_==o.participants_&&participant_==o.participant_&&
        policy_.multipliers==o.policy_.multipliers&&policy_.offsets==o.policy_.offsets&&
        policy_.moduli==o.policy_.moduli&&policy_.eosToken==o.policy_.eosToken;
}

namespace fixed_head_lookup_detail
{
std::size_t product(std::size_t a,std::size_t b)
{
    if(b&&a>std::numeric_limits<std::size_t>::max()/b)
        throw std::overflow_error("Fixed-head lookup admitted byte geometry overflows");
    return a*b;
}

ImmutableFixedHeadLookupCaptureIdentity validate(const ImmutableFixedHeadLookupPlan &p,
    const ImmutableFixedHeadLookupArgs &a,void *stream)
{
    if(!stream||reinterpret_cast<std::uintptr_t>(stream)<=2)
        throw std::invalid_argument("Fixed-head lookup requires exact non-default stream");
    const auto rows=a.tokenRows.captureIdentity();
    const auto canonical=rows.count?DeviceRowRange::deviceCounted(rows.physical_rows,rows.count):
        DeviceRowRange::fullyActive(rows.physical_rows);
    if(rows!=canonical.captureIdentity())
        throw std::invalid_argument("Fixed-head lookup requires complete original token-row count authority");
    const auto ids=product(static_cast<std::size_t>(rows.physical_rows),kNgramHeads);
    const auto local=product(product(static_cast<std::size_t>(rows.physical_rows),p.heads()),p.table().rowBytes());
    const auto gathered=product(local,p.participants());
    const auto elements=product(ids,p.table().rowElements());
    if(a.idCapacity!=ids||a.headerCapacity!=static_cast<std::size_t>(p.participants())||
       a.localNativeBytes!=local||a.gatheredNativeBytes!=gathered||a.outputElements!=elements)
        throw std::invalid_argument("Fixed-head lookup requires exact admitted operand capacities");
    using resident_lookup_detail::range;
    const auto nativeAlignment=p.table().format()==TensorType::FP32||p.table().format()==TensorType::Q8_K?
        alignof(float):alignof(std::uint16_t);
    const std::array ranges{
        range(p.table().nativeData(),p.table().nativeBytes(),nativeAlignment),
        range(a.ids,product(ids,sizeof(std::int64_t)),alignof(std::int64_t)),
        range(a.localHeader,sizeof(FixedHeadValidationHeader),alignof(FixedHeadValidationHeader)),
        range(a.gatheredHeaders,product(a.headerCapacity,sizeof(FixedHeadValidationHeader)),alignof(FixedHeadValidationHeader)),
        range(a.transportRows,sizeof(std::int32_t),alignof(std::int32_t)),
        range(a.localNative,local,nativeAlignment),range(a.gatheredNative,gathered,nativeAlignment),
        range(a.output,product(elements,sizeof(float)),alignof(float)),
        range(a.healthy,sizeof(int),alignof(int)),range(a.error,sizeof(int),alignof(int))};
    for(std::size_t i=0;i<ranges.size();++i)
        for(std::size_t j=0;j<i;++j)resident_lookup_detail::disjoint(ranges[i],ranges[j]);
    if(rows.count)
    {
        const auto count=range(rows.count,sizeof(std::int32_t),alignof(std::int32_t));
        for(const auto &other:ranges)resident_lookup_detail::disjoint(count,other);
    }
    if(a.counters.bound()) {
        if(!a.counters.valid() || a.healthy!=a.counters.control+sampling_math::kDeviceGenerationControlOk ||
            a.error!=a.counters.control+sampling_math::kDeviceGenerationControlErrorCode ||
            !validQwenFixedHeadCounterLifetime(rows.physical_rows,p.heads(),p.table().rowBytes()))
            throw std::invalid_argument("Fixed-head observations require the same original canonical health/error row and bounded lifetime");
        const auto observations=range(a.counters.control+sampling_math::kDeviceGenerationControlFixedHeadCallsLow,
            6*sizeof(std::uint64_t),alignof(std::uint64_t));
        for(const auto &other:ranges)resident_lookup_detail::disjoint(observations,other);
        if(rows.count)resident_lookup_detail::disjoint(observations,range(rows.count,sizeof(std::int32_t),alignof(std::int32_t)));
    }
    return {p,rows,a.ids,a.idCapacity,a.localHeader,a.gatheredHeaders,a.headerCapacity,a.transportRows,
        a.localNative,a.localNativeBytes,a.gatheredNative,a.gatheredNativeBytes,a.output,a.outputElements,
        a.healthy,a.error,stream,{},a.counters};
}
}
}
