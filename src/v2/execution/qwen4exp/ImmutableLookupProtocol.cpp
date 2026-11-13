/**
 * @file ImmutableLookupProtocol.cpp
 * @brief Checked storage geometry for the immutable lookup service boundary.
 *
 * This file contributes only typed byte geometry. PhysicalMemoryAuthority
 * admits and leases those bytes; geometry never queries or subtracts memory.
 */
#include "ImmutableLookupProtocol.h"
#include <limits>
#include <stdexcept>
namespace llaminar2::qwen4exp
{
ImmutableLookupLayout ImmutableLookupLayout::create(std::size_t maximumIds,std::size_t rowBytes)
{
    if(maximumIds==0||rowBytes==0)throw std::invalid_argument("Immutable lookup requires positive ID capacity and native row bytes");
    constexpr auto max=std::numeric_limits<std::size_t>::max();
    const auto multiply=[&](std::size_t a,std::size_t b){
        if(a>max/b)throw std::overflow_error("Immutable lookup payload geometry overflows bytes");return a*b;
    };
    const auto add=[&](std::size_t a,std::size_t b){
        if(a>max-b)throw std::overflow_error("Immutable lookup mapped geometry overflows bytes");return a+b;
    };
    const auto align=[&](std::size_t value){return add(value,63)&~std::size_t{63};};
    const auto ids=sizeof(ImmutableLookupRequest)+sizeof(ImmutableLookupCompletion)+sizeof(ImmutableLookupAcknowledgement);
    const auto payload=align(add(ids,multiply(maximumIds,sizeof(std::int64_t))));
    const auto bytes=align(add(payload,multiply(maximumIds,rowBytes)));
    return {maximumIds,rowBytes,ids,payload,bytes};
}
ImmutableLookupLayout ImmutableLookupLayout::createSubordinate(std::size_t maximumIds,std::size_t rowBytes)
{
    auto result=create(maximumIds,rowBytes);
    constexpr auto max=std::numeric_limits<std::size_t>::max();
    const auto multiply=[&](std::size_t n,std::size_t element) {
        if(n>max/element)throw std::overflow_error("Subordinate immutable metadata array overflows");
        return n*element;
    };
    const auto add=[&](std::size_t a,std::size_t b) {
        if(a>max-b)throw std::overflow_error("Subordinate immutable channel overflows");
        return a+b;
    };
    const auto align=[&](std::size_t n){return add(n,63)&~std::size_t{63};};
    result.protocol_=kImmutableLookupSubordinateProtocol;
    result.intentsOffset_=align(add(result.idsOffset_,multiply(maximumIds,sizeof(std::int64_t))));
    result.invalidationsOffset_=align(add(result.intentsOffset_,multiply(maximumIds,sizeof(ImmutableRowSubordinateIntent))));
    result.payloadOffset_=align(add(result.invalidationsOffset_,multiply(maximumIds,sizeof(ImmutableRowInvalidationAcknowledgement))));
    result.bytes_=align(add(result.payloadOffset_,multiply(maximumIds,rowBytes)));
    return result;
}
std::size_t ImmutableLookupLayout::intentsOffset() const
{
    if(!subordinate())throw std::logic_error("Ordinary immutable channel has no subordinate intent frame");
    return intentsOffset_;
}
std::size_t ImmutableLookupLayout::invalidationsOffset() const
{
    if(!subordinate())throw std::logic_error("Ordinary immutable channel has no subordinate invalidation frame");
    return invalidationsOffset_;
}
} // namespace llaminar2::qwen4exp
