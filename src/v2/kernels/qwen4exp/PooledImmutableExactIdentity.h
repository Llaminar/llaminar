/**
 * @file PooledImmutableExactIdentity.h
 * @brief Exact bounded identity validation over already admitted owner scratch.
 *
 * Both algorithms read only the live prefix. Small sets avoid fixed global
 * radix-bin work; larger sets retain stable integer radix validation. Neither
 * algorithm mutates physical slots, grants or accepted plans.
 */
#pragma once
#include "AdaptiveImmutableRowCacheState.h"
namespace llaminar2::qwen4exp::adaptive_lookup::pooled {
/** @brief Only unassigned allocation slots may repeat the None sentinel. */
enum class DuplicateIdentityPolicy { AllValues, IgnoreUnassignedSlot };
/** @brief Largest set whose quadratic comparisons stay below four-pass bin work. */
inline constexpr std::uint32_t SmallIdentityLimit=64;
/**
 * @brief Validate all exact keys before the caller commits any physical mutation.
 * @param first Admitted live key prefix; radix may reorder its contents.
 * @param second Disjoint admitted scratch of at least count keys.
 * @param bins Disjoint admitted 256-bin scratch, untouched for small sets.
 * @param count Actual combined identities, including every retained grant.
 * @param bytes Four or eight significant bytes, as proven by the caller's key type.
 * @param policy Exact duplicate-sentinel contract for this key domain.
 * @return False on an exact forbidden duplicate, true otherwise.
 */
QFN_ADAPTIVE_HD inline bool exactDistinctIdentities(std::uint64_t* first,std::uint64_t* second,
    std::uint32_t* bins,std::uint32_t count,unsigned bytes,DuplicateIdentityPolicy policy)
{
    if(count<=SmallIdentityLimit){
        // At64, at most2016 comparisons replace at least2560 radix iterations.
        // Field/interval validation remains the caller's prerequisite; only
        // scratch equality work changes, with no accepted-prefix trust.
        for(std::uint32_t i=0;i<count;++i){const auto key=first[i];
            if(policy==DuplicateIdentityPolicy::IgnoreUnassignedSlot && key==None)continue;
            for(std::uint32_t j=0;j<i;++j)if(key==first[j])return false;
        }
        return true;
    }
    auto* input=first;auto* output=second;
    for(unsigned pass=0;pass<bytes;++pass){
        for(unsigned bin=0;bin<256;++bin)bins[bin]=0;
        for(std::uint32_t i=0;i<count;++i)++bins[(input[i]>>(pass*8))&255];
        std::uint32_t cursor=0;
        for(unsigned bin=0;bin<256;++bin){const auto size=bins[bin];bins[bin]=cursor;cursor+=size;}
        for(std::uint32_t i=0;i<count;++i)output[bins[(input[i]>>(pass*8))&255]++]=input[i];
        auto* swap=input;input=output;output=swap;
    }
    for(std::uint32_t i=1;i<count;++i)
        if(input[i]==input[i-1] && !(policy==DuplicateIdentityPolicy::IgnoreUnassignedSlot && input[i]==None))return false;
    return true;
}
} // namespace llaminar2::qwen4exp::adaptive_lookup::pooled
