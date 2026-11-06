/**
 * @file PooledImmutableLocalHotOps.h
 * @brief Allocation-free native local-cache frame mechanics over unchanged cache Operations.
 *
 * The frame owns one stable deduplicated query and bounded native response.
 * Probe retains initial hits before any policy operation. Only the actual
 * model-consumer endpoint may call consumeDemand; speculative fetch/response
 * never earns frequency, reserves slots or evicts rows. No method reads host
 * state, performs I/O, or creates a source/materialization authority.
 */
#pragma once
#include "PooledImmutableLocalHotState.h"
#include "AdaptiveImmutableRowCacheOps.h"
namespace llaminar2::qwen4exp::local_hot {
namespace a=adaptive_lookup;
namespace t=local_hot_transport;
/** @brief Decoded exact native root aliases; ownership remains with the prepared storage. */
template<class Slot> struct View {
    a::Metadata<Slot> metadata{};
    FrameIdentity* identity=nullptr;
    Counters* counters=nullptr;
    a::UniqueRecord* unique=nullptr;
    std::uint32_t* original_to_unique=nullptr;
    std::uint64_t* miss_keys=nullptr;
    const a::KeyIndex* canonical_sorted=nullptr;
    std::byte* cache_payload=nullptr;
    std::byte* response=nullptr;
    std::uint32_t maximum_original=0;
    std::size_t row_bytes=0,slot_stride=0;
    std::uint64_t table=0,source_generation=0,consumer_storage_identity=0;
};
/** @brief Exactly one original bounded frame lifecycle; no independent replacement algorithm exists. */
template<class Slot> class Operations final {
public:
    /** @brief Borrow original admitted aliases; this creates no allocation or live ledger. */
    QFN_ADAPTIVE_HD explicit Operations(View<Slot> view):v_(view),cache_(view.metadata){}
    /**
     * @brief Accept only the actual new request publication after physical frame retirement.
     * @param source Same local original-admission producer, acquired by the request graph edge.
     * @return True without clearing warmed slots, policy, query ordinal or counters.
     * Sticky source/ABI/overflow failure cannot be turned into a reusable request.
     */
    QFN_ADAPTIVE_HD bool resetRequest(const t::SourceAdmissionHeader& source)
    {
        if(!v_.identity || !v_.metadata.frame ||
            v_.identity->terminal_error!=a::Status::Success)return false;
        const auto& frame=*v_.metadata.frame;
        if(!t::validSourceAdmission(source,v_.table,v_.source_generation) ||
            source.source_invocation<=v_.identity->source_request_invocation ||
            v_.identity->disposition!=Disposition::Idle || frame.phase!=a::Phase::Idle || frame.held_pins ||
            (frame.status!=a::Status::Success && frame.status!=a::Status::Cancelled) ||
            (v_.metadata.capacity && (!v_.metadata.owner || v_.metadata.owner->flags!=a::OwnerInitialized)))
            return fatal(a::Status::InvalidPhase);
        v_.identity->source_request_invocation=source.source_invocation;
        return true;
    }
    /** @brief Begin only after the preceding consumed/discarded frame's actual readers retired. */
    QFN_ADAPTIVE_HD bool begin(PooledImmutableLocalHotAccessIntent intent,std::uint32_t original_ids)
    {
        if(!v_.identity || !v_.metadata.frame || !v_.counters)return false;
        if(v_.identity->terminal_error!=a::Status::Success)return false;
        if(!v_.unique || !v_.original_to_unique || !v_.miss_keys || !v_.canonical_sorted ||
            !v_.row_bytes || v_.slot_stride<v_.row_bytes || !v_.table || !v_.source_generation ||
            !v_.consumer_storage_identity || !v_.maximum_original || v_.maximum_original%16 ||
            (v_.metadata.capacity && !v_.cache_payload) || !validLocalHotAccessIntent(intent) ||
            original_ids>v_.maximum_original || original_ids%16 ||
            !v_.identity->source_request_invocation || v_.identity->disposition!=Disposition::Idle)
            return fatal(a::Status::InvalidPhase);
        const auto prior=v_.identity->query;
        const auto request_invocation=v_.identity->source_request_invocation;
        if(prior.sequence && (prior.table!=v_.table ||
            prior.source_storage_generation!=v_.source_generation ||
            prior.consumer_storage_identity!=v_.consumer_storage_identity))
            return fatal(a::Status::Invariant);
        if(prior.sequence==UINT64_MAX)return fatal(a::Status::Exhausted);
        if(!cache_.begin(original_ids))return false;
        *v_.identity=FrameIdentity{};
        v_.identity->source_request_invocation=request_invocation;
        v_.identity->query={v_.table,v_.source_generation,v_.consumer_storage_identity,prior.sequence+1};
        v_.identity->intent=intent;
        if(intent==PooledImmutableLocalHotAccessIntent::FutureReadAhead)v_.identity->flags|=FutureOrigin;
        v_.identity->disposition=Disposition::Probing;return true;
    }
    /** @brief Accept the actual stable coalescer's bounds before any original cache pins. */
    QFN_ADAPTIVE_HD bool coalesced(std::uint32_t unique_rows)
    {
        auto& f=*v_.metadata.frame;
        if(v_.identity->disposition!=Disposition::Probing || f.status!=a::Status::Success ||
            unique_rows>f.original_ids || unique_rows>v_.maximum_original ||
            ((unique_rows==0)!=(f.original_ids==0)))
            return fail(a::Status::Invariant);
        f.unique_rows=unique_rows;return true;
    }
    /** @brief Read-only independent directory probe; native adapters parallelize unique rows. */
    QFN_ADAPTIVE_HD bool probeUnique(std::uint32_t index)
    {
        const auto& f=*v_.metadata.frame;
        if(v_.identity->disposition!=Disposition::Probing || index>=f.unique_rows || f.status!=a::Status::Success)
            return false;
        auto& r=v_.unique[index];
        if(r.row>=v_.metadata.source_rows || r.first_original_coordinate>=f.original_ids)
            return false;
        const auto slot=cache_.find(r.row);
        r.slot=slot;r.expected_slot_generation=slot==a::None?0:v_.metadata.slots[slot].generation;
        r.flags=slot==a::None?0:a::InitialHit;r.response_coordinate=a::None;return true;
    }
    /** @brief Seal every genuine initial pin before compact misses or later policy are visible. */
    QFN_ADAPTIVE_HD bool pinAndCompact()
    {
        auto& f=*v_.metadata.frame;
        if(v_.identity->disposition!=Disposition::Probing || f.status!=a::Status::Success)
            return fail(a::Status::InvalidPhase);
        std::uint32_t misses=0;
        for(std::uint32_t i=0;i<f.unique_rows;++i){
            auto& r=v_.unique[i];
            if(r.flags&a::InitialHit){
                if(!cache_.pinInitial(r.slot,r.expected_slot_generation))return false;
            }else{
                if(r.slot!=a::None || misses>=v_.maximum_original)return fail(a::Status::Invariant);
                r.response_coordinate=misses;v_.miss_keys[misses++]=r.row;
            }
        }
        f.source_rows=misses;v_.identity->disposition=Disposition::Probed;
        if(!misses)v_.identity->disposition=Disposition::DataReady;
        return true;
    }
    /** @brief Authenticate real completed native bytes before the auxiliary branch publishes data-ready. */
    QFN_ADAPTIVE_HD bool acceptNativeResult(const t::RequestHeader& request,t::ConsumerGeometry geometry,
        t::DeviceResultView result)
    {
        auto& f=*v_.metadata.frame;
        if(v_.identity->disposition!=Disposition::Probed || f.status!=a::Status::Success ||
            !t::same(request.query,v_.identity->query) || request.intent!=v_.identity->intent ||
            request.original_ids!=f.original_ids ||
            request.miss_rows!=f.source_rows || !result.header || !result.received_header_bytes ||
            !result.received_payload_bytes || result.native_rows!=v_.response ||
            f.source_rows>result.native_capacity/v_.row_bytes ||
            !t::acceptResult(*result.header,request,geometry,*result.received_header_bytes,*result.received_payload_bytes))
            return fail(a::Status::SourceFailure);
        if((v_.identity->flags&LateAdoption) && !addCounter(v_.counters->future_late_native_rows,f.source_rows))return false;
        v_.identity->retained_native_bytes=*result.received_payload_bytes;
        v_.identity->disposition=Disposition::DataReady;return true;
    }
    /** @return Same pinned native slot or private completed response, independent of optional admission. */
    QFN_ADAPTIVE_HD const std::byte* readableUnique(std::uint32_t index)const
    {
        const auto& f=*v_.metadata.frame;
        if(f.status!=a::Status::Success || v_.identity->disposition!=Disposition::DataReady || index>=f.unique_rows)
            return nullptr;
        const auto& r=v_.unique[index];
        if(r.flags&a::InitialHit){
            if(r.slot>=v_.metadata.capacity)return nullptr;
            const auto& slot=v_.metadata.slots[r.slot];
            if(a::PhysicalSlot<Slot>::state(slot)!=a::SlotState::Ready ||
                !a::PhysicalSlot<Slot>::pinned(slot) || slot.generation!=r.expected_slot_generation ||
                std::uint64_t(slot.row)!=r.row)return nullptr;
            return v_.cache_payload+std::size_t(r.slot)*v_.slot_stride;
        }
        if(r.response_coordinate>=f.source_rows ||
            f.source_rows>v_.identity->retained_native_bytes/v_.row_bytes)return nullptr;
        return v_.response+std::size_t(r.response_coordinate)*v_.row_bytes;
    }
    /** @return Native row for one complete original coordinate; the main LocalRead alone writes model output. */
    QFN_ADAPTIVE_HD const std::byte* readableOriginal(std::uint32_t index)const
    {
        if(index>=v_.metadata.frame->original_ids)return nullptr;
        return readableUnique(v_.original_to_unique[index]);
    }
    /** @brief Record completion only after the SAME LocalRead native output producer finished. */
    QFN_ADAPTIVE_HD bool completeRead(t::QueryIdentity query)
    {
        const auto& f=*v_.metadata.frame;
        if(!t::same(query,v_.identity->query) || f.status!=a::Status::Success ||
            v_.identity->disposition!=Disposition::DataReady ||
            v_.identity->intent!=PooledImmutableLocalHotAccessIntent::CurrentInvocation || (v_.identity->flags&ReadCompleted))
            return fatal(a::Status::InvalidPhase);
        std::uint64_t hits=0;
        for(std::uint32_t i=0;i<f.original_ids;++i){
            const auto u=v_.original_to_unique[i];if(u>=f.unique_rows)return fatal(a::Status::Invariant);
            if(v_.unique[u].flags&a::InitialHit)++hits;
        }
        if(!addCounter(v_.counters->read_original_ids,f.original_ids) ||
            !addCounter(v_.counters->local_hit_original_ids,hits) ||
            (!f.source_rows && !addCounter(v_.counters->all_hit_wait_avoided_original_ids,f.original_ids)))return false;
        v_.identity->flags|=ReadCompleted;return true;
    }
    /** @brief Adopt the SAME retained future frame without another probe/source lookup or policy credit. */
    QFN_ADAPTIVE_HD bool adopt(t::QueryIdentity query)
    {
        if(!t::same(query,v_.identity->query) ||
            v_.identity->intent!=PooledImmutableLocalHotAccessIntent::FutureReadAhead ||
            (v_.identity->disposition!=Disposition::Probed && v_.identity->disposition!=Disposition::DataReady))
            return fail(a::Status::InvalidPhase);
        if(v_.identity->disposition==Disposition::DataReady){
            if(!addCounter(v_.counters->future_on_time_native_rows,v_.metadata.frame->source_rows))return false;
        }else v_.identity->flags|=LateAdoption;
        v_.identity->intent=PooledImmutableLocalHotAccessIntent::CurrentInvocation;return true;
    }
    /**
     * @brief Credit actual demand once in true original order, then reserve optional slots.
     * @note Only an authenticated original model-consumed graph edge calls this
     * operation. L2 receipts and intent alone never authorize this transition.
     */
    QFN_ADAPTIVE_HD bool consumeDemand(t::QueryIdentity query)
    {
        auto& f=*v_.metadata.frame;
        if(!t::same(query,v_.identity->query) || f.status!=a::Status::Success ||
            v_.identity->intent!=PooledImmutableLocalHotAccessIntent::CurrentInvocation ||
            v_.identity->disposition!=Disposition::DataReady || !(v_.identity->flags&ReadCompleted) ||
            !cache_.beginPolicy(f.unique_rows))return fail(a::Status::InvalidPhase);
        for(std::uint32_t i=0;i<f.original_ids;++i){
            const auto u=v_.original_to_unique[i];if(u>=f.unique_rows)return fail(a::Status::Invariant);
            auto& r=v_.unique[u];const a::Decision prior{r.slot,r.expected_slot_generation};
            const auto beforeEvictions=f.evictions;
            const auto decision=cache_.observe(r.row,prior);
            // Observe is the original sole victim owner. Preserve completed
            // erasures even if a subsequent reservation or frame fails.
            if(!addCounter(v_.counters->evictions,f.evictions-beforeEvictions))return false;
            if(f.status!=a::Status::Success)return false;
            r.slot=decision.slot;r.expected_slot_generation=decision.generation;
            if(decision.slot!=a::None && !(r.flags&a::InitialHit))r.flags|=a::ReservedFill;
        }
        if(!cache_.prepare())return false;
        if(f.source_rows){
            if(!cache_.transition(a::Phase::Prepared,a::Phase::Published) ||
                !cache_.transition(a::Phase::Published,a::Phase::ResponseCopied) ||
                !cache_.transition(a::Phase::ResponseCopied,a::Phase::Validated))return false;
        }else if(!cache_.transition(a::Phase::Prepared,a::Phase::Validated))return false;
        v_.identity->disposition=Disposition::Consuming;return true;
    }
    /** @return Exact native byte source for an admitted miss; copies never include slot padding. */
    QFN_ADAPTIVE_HD const std::byte* fillSource(std::uint32_t unique,std::size_t byte)const
    {
        if(v_.identity->disposition!=Disposition::Consuming || v_.metadata.frame->status!=a::Status::Success ||
            unique>=v_.metadata.frame->unique_rows || byte>=v_.row_bytes)return nullptr;
        const auto& r=v_.unique[unique];
        if(!(r.flags&a::ReservedFill) || r.response_coordinate>=v_.metadata.frame->source_rows)return nullptr;
        return v_.response+std::size_t(r.response_coordinate)*v_.row_bytes+byte;
    }
    /** @return Same reservation's native byte destination, never a Ready publication. */
    QFN_ADAPTIVE_HD std::byte* fillDestination(std::uint32_t unique,std::size_t byte)const
    {
        if(!fillSource(unique,byte))return nullptr;const auto& r=v_.unique[unique];
        if(r.slot>=v_.metadata.capacity)return nullptr;
        const auto& slot=v_.metadata.slots[r.slot];
        if(slot.generation!=r.expected_slot_generation || a::PhysicalSlot<Slot>::state(slot)!=a::SlotState::Reserved ||
            !a::PhysicalSlot<Slot>::pinned(slot))return nullptr;
        return v_.cache_payload+std::size_t(r.slot)*v_.slot_stride+byte;
    }
    /** @return Actual unique coordinate for one stable compact miss, using the retained original row sort. */
    QFN_ADAPTIVE_HD std::uint32_t uniqueForMiss(std::uint32_t miss)const
    {
        const auto& f=*v_.metadata.frame;
        if(miss>=f.source_rows || !f.original_ids)return a::None;
        const auto row=v_.miss_keys[miss];std::uint32_t begin=0,end=std::uint32_t(f.original_ids);
        while(begin<end){const auto middle=begin+(end-begin)/2;
            if(v_.canonical_sorted[middle].row<row)begin=middle+1;else end=middle;}
        if(begin>=f.original_ids || v_.canonical_sorted[begin].row!=row)return a::None;
        const auto first=v_.canonical_sorted[begin].original_coordinate;
        if(first>=f.original_ids)return a::None;
        const auto unique=v_.original_to_unique[first];
        if(unique>=f.unique_rows || v_.unique[unique].row!=row || v_.unique[unique].response_coordinate!=miss)return a::None;
        return unique;
    }
    /** @brief Publish Ready only after every actual native copy producer has completed. */
    QFN_ADAPTIVE_HD bool completeFills()
    {
        if(v_.identity->disposition!=Disposition::Consuming || v_.metadata.frame->status!=a::Status::Success)
            return fail(a::Status::InvalidPhase);
        for(std::uint32_t miss=0;miss<v_.metadata.frame->source_rows;++miss){
            const auto unique=uniqueForMiss(miss);if(unique==a::None)return fatal(a::Status::Invariant);
            const auto& r=v_.unique[unique];
            if(r.flags&a::ReservedFill){
                const auto beforePromotions=v_.metadata.frame->promotions;
                const bool ready=cache_.makeReady({r.slot,r.expected_slot_generation});
                if(!addCounter(v_.counters->promotions,v_.metadata.frame->promotions-beforePromotions) || !ready)return false;
            }}
        if(!addCounter(v_.counters->consumed_original_ids,v_.metadata.frame->original_ids) ||
            ((v_.identity->flags&FutureOrigin) && !addCounter(v_.counters->future_useful_native_rows,v_.metadata.frame->source_rows)))return false;
        v_.identity->disposition=Disposition::Consumed;return true;
    }
    /** @brief Discard unconsumed work without calling policy, reserving slots or altering hot row protection. */
    QFN_ADAPTIVE_HD bool discard(t::QueryIdentity query)
    {
        if(!t::same(query,v_.identity->query) || v_.metadata.frame->status!=a::Status::Success ||
            (v_.identity->disposition!=Disposition::Probed && v_.identity->disposition!=Disposition::DataReady))
            return fail(a::Status::InvalidPhase);
        if((v_.identity->flags&FutureOrigin) && !addCounter(v_.counters->future_wasted_native_rows,
            v_.identity->retained_native_bytes/v_.row_bytes))return false;
        cache_.fail(a::Status::Cancelled);v_.identity->disposition=Disposition::Discarded;return true;
    }
    /** @brief Preserve absorbing cancellation/failure even during partial reservations/fills. */
    QFN_ADAPTIVE_HD bool cancel(t::QueryIdentity query,a::Status cause=a::Status::Cancelled)
    {
        if(!t::same(query,v_.identity->query) || v_.identity->disposition==Disposition::Idle ||
            cause==a::Status::Success)return fatal(a::Status::InvalidPhase);
        cache_.fail(cause);
        if(cause!=a::Status::Cancelled && v_.identity->terminal_error==a::Status::Success)
            v_.identity->terminal_error=v_.metadata.frame->status;
        v_.identity->disposition=Disposition::Cancelled;return true;
    }
    /** @brief Physical drain is distinct from semantic successful consumption. */
    struct Retirement {
        enum class Physical : std::uint32_t { Pending,Retired } physical=Physical::Pending;
        enum class Outcome : std::uint32_t { Consumed,Discarded,Cancelled,Failed } outcome=Outcome::Failed;
        a::Status status=a::Status::Invariant;
    };
    /** @brief Release only actual matching held pins after final native readers/transport retired. */
    QFN_ADAPTIVE_HD Retirement retire(t::QueryIdentity query)
    {
        auto& f=*v_.metadata.frame;
        const auto disposition=v_.identity->disposition;
        if(!t::same(query,v_.identity->query) ||
            (disposition!=Disposition::Consumed && disposition!=Disposition::Discarded && disposition!=Disposition::Cancelled) ||
            f.unique_rows>v_.maximum_original){fatal(a::Status::InvalidPhase);return {};}
        const bool cancelling=disposition==Disposition::Cancelled;
        for(std::uint32_t i=0;i<f.unique_rows;++i){
            const auto& r=v_.unique[i];
            if(r.slot==a::None)continue;
            if(r.slot>=v_.metadata.capacity){fatal(a::Status::Invariant);return {};}
            const auto& slot=v_.metadata.slots[r.slot];
            if(!a::PhysicalSlot<Slot>::pinned(slot)){
                if(cancelling)continue;
                fatal(a::Status::Invariant);return {};
            }
            if(slot.generation!=r.expected_slot_generation || std::uint64_t(slot.row)!=r.row ||
                !cache_.release({r.slot,r.expected_slot_generation})){fatal(a::Status::Invariant);return {};}
        }
        const auto status=f.status;
        const bool success=cache_.finish();
        if(f.phase!=a::Phase::Idle){fatal(a::Status::InvalidPhase);return {};}
        v_.identity->disposition=Disposition::Idle;
        if(success && disposition==Disposition::Consumed)
            return {Retirement::Physical::Retired,Retirement::Outcome::Consumed,a::Status::Success};
        if(status==a::Status::Cancelled && disposition==Disposition::Discarded)
            return {Retirement::Physical::Retired,Retirement::Outcome::Discarded,status};
        if(status==a::Status::Cancelled && cancelling)
            return {Retirement::Physical::Retired,Retirement::Outcome::Cancelled,status};
        if(v_.identity->terminal_error==a::Status::Success)v_.identity->terminal_error=status;
        return {Retirement::Physical::Retired,Retirement::Outcome::Failed,status};
    }
    /** @brief Preserve the first native frame error. */
    QFN_ADAPTIVE_HD bool fail(a::Status status){return cache_.fail(status);}
    /** @brief Make malformed lifetime/overflow permanently fail closed across request reset. */
    QFN_ADAPTIVE_HD bool fatal(a::Status status)
    {
        if(v_.identity && v_.identity->terminal_error==a::Status::Success)
            v_.identity->terminal_error=status;
        cache_.poison();return cache_.fail(status);
    }
private:
    /** @brief One transition owns each counter; overflow fails before wrap rather than weakening diagnostics. */
    QFN_ADAPTIVE_HD bool addCounter(std::uint64_t& counter,std::uint64_t value)
    {
        if(value>UINT64_MAX-counter)return fatal(a::Status::Exhausted);
        counter+=value;return true;
    }
    View<Slot> v_;
    a::Operations<Slot> cache_;
};
} // namespace llaminar2::qwen4exp::local_hot
