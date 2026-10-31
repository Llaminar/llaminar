/**
 * @file PooledImmutableCoordinatorOps.h
 * @brief Global stable coalescing, frozen owner waves and one shared pooled admission policy.
 *
 * One exclusive CPU/device coordinator parses original inputs. Frequency counts
 * follow exact original positions. Victims use the frozen prior-frame queue
 * frontier after actual initial pins and free completions; after actual grants,
 * recency updates replay original positions exactly once. No physical state is
 * inferred from queue membership or plan intent. Native transport and source
 * binding must retain/authenticate the original producer and event edges.
 */
#pragma once
#include "PooledImmutableCoordinatorState.h"
#include "PooledImmutableDirectoryLookup.h"
namespace llaminar2::qwen4exp::adaptive_lookup::pooled
{
/** @brief Allocation-free exclusive borrower of the sole coordinator authority and exact admitted roots. */
class CoordinatorOperations final
{
public:
    /** @brief Bind existing roots; decoded pointers and records do not mint a runtime capability. */
    QFN_ADAPTIVE_HD explicit CoordinatorOperations(CoordinatorView view) : v_(view) {}
    /**
     * @brief Initialize virgin policy and frame roots once, without touching physical owners.
     * @return False on missing/aliased roots, inconsistent exact capacities or repeated setup.
     */
    QFN_ADAPTIVE_HD bool initialize()
    {
        if (!valid() || v_.frame->phase != LookupPhase::Uninitialized || v_.frame->ordinal ||
            (capacity() && v_.policy->flags)) return fail(Status::InvalidPhase);
        for (std::uint32_t i=0;i<capacity();++i) initializeEntry(i);
        for (std::uint32_t i=0;i+1<capacity();++i) initializeNode(i);
        for (std::uint64_t i=0;i<std::uint64_t{4}*v_.parameters.laneWidth();++i) initializeFrequency(i);
        return initializeControls();
    }

    /**
     * @brief Begin one original lookup interval and globally coalesce IDs in stable first-occurrence order.
     * @param admission Original admitted request generation acquired from its actual producer.
     * @param originalRows Exact original row prefix, including interleaved duplicates; no source-home partition.
     * @param count Actual original IDs, bounded by the independently admitted N geometry.
     * @return True after retaining original-position-to-global-unique permutation and immutable unique rows.
     * @note Every successful U0/all-hit invocation advances the lookup ordinal; source sequences never supply it.
     */
    template<class Row> QFN_ADAPTIVE_HD bool begin(std::uint64_t admission, const Row *originalRows, std::uint32_t count)
    {
        static_assert(std::is_same_v<Row,std::uint32_t> || std::is_same_v<Row,std::uint64_t>);
        if (!openOriginal(admission,originalRows,count)) return false;
        auto &f=*v_.frame;
        for (std::uint32_t i=0;i<count;++i)
            if (originalRows[i]>=v_.parameters.sourceRows()) return fail(Status::Invariant);
        for (std::uint32_t i = 0; i < count; ++i) v_.sort_a[i] = {originalRows[i],i,0};
        sort(v_.sort_a,v_.sort_b,count,false);
        std::uint32_t unique = 0;
        for (std::uint32_t at = 0; at < count;)
        {
            const auto first = at++;
            while (at < count && v_.sort_a[at].row == v_.sort_a[first].row) ++at;
            if (unique == v_.parameters.maximumUnique()) return fail(Status::Invariant);
            v_.sort_b[unique] = {v_.sort_a[first].row,v_.sort_a[first].position,unique};
            for (auto i = first; i < at; ++i) v_.original_to_unique[v_.sort_a[i].position] = unique;
            ++unique;
        }
        // Four stable integer passes reorder distinct groups by their first
        // original position. The other scratch bank then stores sorted→stable.
        sort(v_.sort_b,v_.sort_a,unique,true);
        for (std::uint32_t i = 0; i < unique; ++i)
        {
            const auto occurrence = v_.sort_b[i];
            v_.sort_a[occurrence.unique].unique = i;
            auto &r = v_.unique[i];
            r = {};
            r.row = occurrence.row;
            r.first_position = occurrence.position;
            r.entry = find(r.row);
            if (f.status != Status::Success) return false;
            r.phase = r.entry == None ? UniquePhase::Missing : UniquePhase::NeedsPin;
        }
        for (std::uint32_t i = 0; i < count; ++i)
            v_.original_to_unique[i] = v_.sort_a[v_.original_to_unique[i]].unique;
        f.unique_rows = unique;
        return finishOriginal();
    }
    /** @return Actual original frame identity, independent of source protocol counters. */
    QFN_ADAPTIVE_HD Interval interval() const { return v_.frame ? v_.frame->interval : Interval{}; }
    /**
     * @brief Emit one directed initial-pin command only for a previously published locator.
     * @param unique Stable global unique coordinate.
     * @param command Output original full locator and explicit monotonic dispatch ordinal.
     * @return False without a command for a miss; malformed/repeated hit issuance poisons the frame.
     */
    QFN_ADAPTIVE_HD bool pinCommand(std::uint32_t unique, OwnerCommand &command)
    {
        if (!record(unique,LookupPhase::Pinning)) return false;
        auto &r = v_.unique[unique];
        if (r.phase == UniquePhase::Missing) return false;
        if (r.phase != UniquePhase::NeedsPin) return fail(Status::InvalidPhase);
        const auto &l = v_.entries[r.entry].locator;
        command = {{interval(),l.row,l.generation,0,unique,l.owner,l.slot,GrantKind::Read},
            r.row,CommandKind::PinInitial,0,nextCommand()};
        if (!command.ordinal.value) return false;
        r.pin = command;
        r.phase = UniquePhase::PinPending;
        ++v_.frame->outstanding;
        return true;
    }
    /**
     * @brief Acquire the exact actual owner read grant before enabling any allocation planning.
     * @param unique Original stable unique coordinate whose pin command was issued.
     * @param grant Actual owner reply, authenticated by the eventual directed transport binding.
     * @return False on missing/stale/foreign/repeated replies before marking any policy entry observed.
     */
    QFN_ADAPTIVE_HD bool acceptPin(std::uint32_t unique,const Grant &grant)
    {
        if (!record(unique,LookupPhase::Pinning)) return false;
        auto &r = v_.unique[unique];
        if (r.phase != UniquePhase::PinPending || !sameTarget(r.pin.target,grant) ||
            grant.kind != GrantKind::Read || grant.destination_token || !grant.generation || !v_.frame->outstanding)
            return fail(Status::Invariant);
        r.grant = grant;
        r.phase = UniquePhase::Read;
        v_.entries[r.entry].observed_frame = v_.frame->ordinal;
        --v_.frame->outstanding;
        return true;
    }
    /**
     * @brief Acquire the complete actual all-owner pin frontier, including every C0 owner.
     * @param headers Exactly P ordered immutable actual owner pin headers.
     * @param count Actual declared owner count, never the consumer count.
     * @return True only after every hit has its actual read grant and every original owner agrees.
     */
    QFN_ADAPTIVE_HD bool sealPins(const PinHeader *headers,std::uint32_t count)
    {
        if (!healthy(LookupPhase::Pinning) || v_.frame->outstanding || !headers || count != v_.parameters.participants())
            return fail(Status::InvalidPhase);
        for (std::uint32_t i = 0; i < count; ++i)
            if (!same(headers[i].interval,interval()) || headers[i].owner != i || headers[i].status != Status::Success)
                return fail(Status::Invariant);
        for (std::uint32_t i = 0; i < v_.frame->unique_rows; ++i)
            if (v_.unique[i].phase != UniquePhase::Missing && v_.unique[i].phase != UniquePhase::Read)
                return fail(Status::InvalidPhase);
        for (unsigned i = 0; i < 3; ++i) v_.frame->eligible_cursor[i] = capacity() ? v_.policy->head[i] : None;
        v_.frame->phase = LookupPhase::Availability;
        return true;
    }
    /**
     * @brief Retain one exact immutable availability lease from each actual declared owner and create free plans.
     * @param leases Exactly P acquired owner-authored leases in original owner order.
     * @param count Actual owner count.
     * @return True after stable first-occurrence assignment and compact per-owner free-plan construction.
     * @note Counts are directed plan extents; no coordinator occupancy/free-chain authority is introduced.
     */
    QFN_ADAPTIVE_HD bool planFree(const AvailabilityLease *leases,std::uint32_t count)
    { return planFreeImpl(leases,count,nullptr); }
    /**
     * @brief Read the exact directed immutable free plan for one original owner.
     * @param owner Original owner coordinate.
     * @return Typed live plan view; empty owners receive only their zero-count protocol metadata.
     */
    QFN_ADAPTIVE_HD FreeAllocationPlan freePlan(std::uint32_t owner)
    {
        if (!healthy(LookupPhase::FreePlanned) || owner >= v_.parameters.participants())
        { fail(Status::InvalidPhase); return {}; }
        const auto &p = v_.owner_plans[owner];
        return {p.lease,v_.decisions+p.free_offset,p.free_count};
    }
    /**
     * @brief Acquire one exact actual reservation reply from the free or victim wave.
     * @param unique Stable global source coordinate, independent of local dispatch and generation.
     * @param grant Actual immutable owner reservation grant.
     * @return False before metadata mutation for any stale/foreign/duplicate/absent grant.
     */
    QFN_ADAPTIVE_HD bool acceptReservation(std::uint32_t unique,const Grant &grant)
    {
        if (!valid() || unique >= v_.frame->unique_rows) return fail(Status::InvalidPhase);
        auto &r = v_.unique[unique];
        const bool free = v_.frame->phase == LookupPhase::FreePlanned && r.phase == UniquePhase::FreePlanned;
        const bool victimWave = v_.frame->phase == LookupPhase::VictimPlanned && r.phase == UniquePhase::VictimPlanned;
        const auto expectedOwner = r.allocation_owner;
        const auto at = index(grant.owner,grant.slot);
        if ((!free && !victimWave) || v_.frame->status != Status::Success || !same(grant.interval,interval()) ||
            grant.kind != GrantKind::Reservation || grant.unique != unique || grant.row != r.row ||
            grant.owner != expectedOwner || !grant.generation || grant.destination_token != std::uint64_t{unique}+1 ||
            at == None || !v_.frame->outstanding) return fail(Status::Invariant);
        auto &e = v_.entries[at];
        if (free ? (e.present || e.stream != Stream::None) :
            (!e.present || !sameLocator(e.locator,r.decision.victim) ||
                e.observed_frame == v_.frame->ordinal || grant.generation <= e.locator.generation))
            return fail(Status::Invariant);
        if (e.present && (!erase(at) || !unlink(at))) return false;
        e = {{r.row,grant.generation,grant.owner,grant.slot},0,v_.frame->ordinal,None,None,Stream::None,0};
        r.entry = at;
        r.grant = grant;
        r.phase = free ? UniquePhase::FreeGranted : UniquePhase::VictimGranted;
        auto &owner = v_.owner_plans[grant.owner];
        if (free) ++owner.free_received; else ++owner.victim_received;
        --v_.frame->outstanding;
        return true;
    }
    /**
     * @brief Acquire all actual free completions; cancellation may close only an unexecuted whole owner plan.
     * @param receipts Exactly P original owner-authored free completion records.
     * @param count Actual declared owner count.
     * @return True after reconciling actual granted prefixes; unused or cancelled quota vetoes every victim.
     */
    QFN_ADAPTIVE_HD bool completeFree(const FreeAllocationReceipt *receipts,std::uint32_t count)
    { return complete(receipts,count,false); }
    /**
     * @brief Select compact second-wave victims only after complete actual global free exhaustion.
     * @return True after creating all directed victim plans; rows not admitted remain explicit source-only work.
     * @note Policy cursors advance over the frozen queue snapshot, without planned pin/occupancy mirrors.
     */
    QFN_ADAPTIVE_HD bool planVictims()
    {
        if (!healthy(LookupPhase::FreeCompleted)) return fail(Status::InvalidPhase);
        for (std::uint32_t i = 0; i < v_.frame->unique_rows; ++i)
        {
            auto &r = v_.unique[i];
            if (r.phase != UniquePhase::Missing && r.phase != UniquePhase::SourceOnly) continue;
            const auto at = v_.frame->free_exhausted && !v_.frame->cancelled_completion ? victim() : None;
            if (v_.frame->status != Status::Success) return false;
            if (at == None || frequency(r.row,false) <= frequency(v_.entries[at].locator.row,false))
            { r.phase = UniquePhase::SourceOnly; continue; }
            const auto &e = v_.entries[at];
            r.decision = {r.row,i,CommandKind::ReserveVictim,e.locator,nextCommand()};
            r.phase = UniquePhase::VictimPlanned;
            r.allocation_owner = e.locator.owner;
            v_.frame->eligible_cursor[static_cast<unsigned>(e.stream)] = e.next;
            ++v_.owner_plans[e.locator.owner].victim_count;
            ++v_.frame->victim_planned;
        }
        buildPlans(true);
        v_.frame->outstanding = v_.frame->victim_planned;
        v_.frame->phase = LookupPhase::VictimPlanned;
        return v_.frame->status == Status::Success;
    }
    /**
     * @brief Read the exact immutable victim plan for one original owner.
     * @param owner Original owner coordinate, independent of consumers/coordinator.
     * @return Typed directed live view; the original free-plan prefix remains unchanged.
     */
    QFN_ADAPTIVE_HD VictimAllocationPlan victimPlan(std::uint32_t owner)
    {
        if (!healthy(LookupPhase::VictimPlanned) || owner >= v_.parameters.participants())
        { fail(Status::InvalidPhase); return {}; }
        const auto &p = v_.owner_plans[owner];
        return {p.lease,v_.decisions+p.victim_offset,p.victim_count};
    }
    /**
     * @brief Acquire all actual second-wave completions before any source window can be published.
     * @param receipts Exactly P original owner-authored victim completion records.
     * @param count Actual declared owner count, including empty plans and C0.
     * @return True after exact reconciliation and one original-order recency pass.
     */
    QFN_ADAPTIVE_HD bool completeVictims(const VictimAllocationReceipt *receipts,std::uint32_t count)
    { return complete(receipts,count,true) && applyQueues(); }
    /**
     * @brief Seal one actual source window after both complete owner allocation waves.
     * @param source Original source transaction, or zero only for an all-hit/U0 frame.
     * @return False on incomplete/cancelled waves or incompatible source identity.
     */
    QFN_ADAPTIVE_HD bool sealSource(SourceWindow source)
    {
        if (!healthy(LookupPhase::VictimCompleted) || v_.frame->outstanding || v_.frame->cancelled_completion)
            return fail(Status::InvalidPhase);
        std::uint32_t misses = 0;
        for (std::uint32_t i = 0; i < v_.frame->unique_rows; ++i) misses += v_.unique[i].phase != UniquePhase::Read;
        if (misses ? (source.table != v_.table || !source.generation || !source.sequence) :
            (source.table || source.generation || source.sequence)) return fail(Status::Invariant);
        v_.frame->source = source;
        v_.frame->phase = LookupPhase::Source;
        return true;
    }
    /**
     * @brief Publish locators only after actual all-owner Ready completion and exact source retirement.
     * @param headers Exactly P acquired actual Ready producers, distinct from fill-validation headers.
     * @param count Actual declared owner count.
     * @param retirement Actual source retirement disposition.
     * @param consumed Original consumed source window, or zero for unpublished source work.
     * @return True after publishing every actual new reservation locator.
     */
    QFN_ADAPTIVE_HD bool publish(const ReadyHeader *headers,std::uint32_t count,
        SourceRetirement retirement,SourceWindow consumed = {})
    {
        if (!healthy(LookupPhase::Source) || !headers || count != v_.parameters.participants() || retirement==SourceRetirement::Aborted || !retired(retirement,consumed))
            return fail(Status::SourceFailure);
        for (std::uint32_t i = 0; i < count; ++i)
            if (!same(headers[i].interval,interval()) || !sameSource(headers[i].source,v_.frame->source) ||
                headers[i].owner != i || headers[i].status != Status::Success) return fail(Status::Invariant);
        for (std::uint32_t i = 0; i < v_.frame->unique_rows; ++i)
        {
            const auto &r = v_.unique[i];
            if (r.phase != UniquePhase::FreeGranted && r.phase != UniquePhase::VictimGranted) continue;
            v_.entries[r.entry].present = 1;
            if (!insert(r.entry)) return false;
        }
        v_.frame->phase = LookupPhase::Ready;
        return true;
    }
    /**
     * @brief Reconcile actual release receipts after accepted waves drain and source/readers retire.
     * @param retirement Actual source retirement disposition.
     * @param consumed Exact original consumed source identity, or zero before publication.
     * @param receipts Complete increasing-global-unique actual releaseWithReceipt prefix.
     * @param count Exact actual acquired grant count, excluding unaccepted plans and source-only rows.
     * @return True only after complete receipt prevalidation; unknown accepted/source state poisons the frame.
     */
    QFN_ADAPTIVE_HD bool cancel(SourceRetirement retirement,SourceWindow consumed,
        const ReleaseReceipt *receipts,std::uint32_t count)
    {
        if (!valid() || v_.frame->phase == LookupPhase::Idle || v_.frame->phase == LookupPhase::Uninitialized ||
            v_.frame->status != Status::Success || v_.frame->outstanding || !retired(retirement,consumed))
            return fail(Status::SourceFailure);
        std::uint32_t expected = 0;
        for (std::uint32_t i = 0; i < v_.frame->unique_rows; ++i) expected += acquired(v_.unique[i].phase);
        if (expected != count || (count && !receipts)) return fail(Status::Invariant);
        std::uint32_t cursor = 0;
        for (std::uint32_t i = 0; i < v_.frame->unique_rows; ++i)
        {
            const auto &r = v_.unique[i];
            if (!acquired(r.phase)) continue;
            const auto &receipt = receipts[cursor++];
            if (!sameTarget(r.grant,receipt.original) || r.grant.kind != receipt.original.kind ||
                r.grant.destination_token != receipt.original.destination_token || receipt.reserved ||
                (receipt.disposition != ReleaseDisposition::RetainedReady && receipt.disposition != ReleaseDisposition::RecycledEmpty) ||
                ((r.phase == UniquePhase::Read || v_.entries[r.entry].present) &&
                    receipt.disposition != ReleaseDisposition::RetainedReady)) return fail(Status::Invariant);
        }
        if (!applyQueues()) return false;
        cursor = 0;
        for (std::uint32_t i = 0; i < v_.frame->unique_rows; ++i)
        {
            const auto &r = v_.unique[i];
            if (!acquired(r.phase)) continue;
            const auto disposition = receipts[cursor++].disposition;
            if (r.phase == UniquePhase::Read || v_.entries[r.entry].present) continue;
            if (disposition == ReleaseDisposition::RetainedReady)
            { v_.entries[r.entry].present = 1; if (!insert(r.entry)) return false; }
            else if (v_.entries[r.entry].stream != Stream::None && !unlink(r.entry)) return false;
        }
        v_.frame->status = Status::Cancelled;
        v_.frame->phase = LookupPhase::Draining;
        return true;
    }
    /**
     * @brief Retire the original frame only after every actual owner confirms physical source/reader retirement.
     * @param headers Exactly P acquired actual retired-owner records.
     * @param count Actual declared owner count.
     * @return True on healthy completion or clean cancellation; original ordinal and policy persist.
     */
    QFN_ADAPTIVE_HD bool finish(const RetiredHeader *headers,std::uint32_t count)
    {
        if (!valid() || !headers || count != v_.parameters.participants() ||
            (v_.frame->phase != LookupPhase::Ready && v_.frame->phase != LookupPhase::Draining) ||
            (v_.frame->status != Status::Success && v_.frame->status != Status::Cancelled)) return fail(Status::InvalidPhase);
        for (std::uint32_t i = 0; i < count; ++i)
            if (!same(headers[i].interval,interval()) || !sameSource(headers[i].source,v_.frame->source) ||
                headers[i].owner != i || (headers[i].status != Status::Success &&
                    (v_.frame->status != Status::Cancelled || headers[i].status != Status::Cancelled)))
                return fail(Status::Invariant);
        v_.frame->phase = LookupPhase::Idle;
        v_.frame->status = Status::Success;
        return true;
    }
    /** @return One bounded radix lookup; only Ready-published locators are visible. */
    QFN_ADAPTIVE_HD std::uint32_t find(std::uint64_t row)
    {
        if (!valid() || (capacity() && (v_.policy->flags != OwnerInitialized || !policyMatches()))) return failIndex();
        const auto found=lookupDirectory(v_.entries,v_.nodes,capacity(),capacity()?v_.policy->directory_root:None,row);
        return found.error?failIndex():found.entry;
    }

private:
    friend struct CoordinatorCoalescingAccess;
    /** @brief Execute the canonical free-plan policy using actual scalar observations or the native stable run producer. */
    QFN_ADAPTIVE_HD bool planFreeImpl(const AvailabilityLease *leases,std::uint32_t count,const OriginalObservationRuns* runs)
    {
        if (!healthy(LookupPhase::Availability) || !leases || count != v_.parameters.participants() ||
            aliases(leases,std::size_t{count}*sizeof(AvailabilityLease)))
            return fail(Status::InvalidPhase);
        for (std::uint32_t i = 0; i < count; ++i)
            if (!same(leases[i].interval,interval()) || leases[i].owner != i || leases[i].free_slots > v_.owners[i].slots)
                return fail(Status::Invariant);
        for (std::uint32_t i = 0; i < count; ++i) v_.owner_plans[i].lease = leases[i];
        if(runs){
            if(!observeRuns(*runs))return false;
        }else{
        for (std::uint32_t pos = 0; pos < v_.frame->original_ids; ++pos)
        {
            const auto unique = v_.original_to_unique[pos];
            if (unique >= v_.frame->unique_rows) return fail(Status::Invariant);
            if (capacity())
            {
                ++v_.policy->logical_clock;
                frequency(v_.unique[unique].row,true);
                if (v_.frame->status != Status::Success) return false;
            }
        }
        }
        v_.frame->observed = true;
        std::uint32_t cursor = 0;
        for (std::uint32_t i = 0; i < v_.frame->unique_rows; ++i)
        {
            auto &r = v_.unique[i];
            if (r.phase == UniquePhase::Read) continue;
            while (cursor < count && v_.owner_plans[cursor].free_count == leases[cursor].free_slots) ++cursor;
            if (cursor == count) continue;
            r.decision = {r.row,i,CommandKind::ReserveFree,{},nextCommand()};
            r.allocation_owner = cursor;
            r.phase = UniquePhase::FreePlanned;
            ++v_.owner_plans[cursor].free_count;
            ++v_.frame->free_planned;
        }
        buildPlans(false);
        v_.frame->outstanding = v_.frame->free_planned;
        v_.frame->phase = LookupPhase::FreePlanned;
        return v_.frame->status == Status::Success;
    }

    /**
     * @brief Validate every authored stable run before mutating a frequency cell or policy clock.
     * @note Equal-row repeats commute only within one aging epoch. Hash-colliding
     * distinct rows retain their exact original run order, including partition boundaries.
     */
    QFN_ADAPTIVE_HD bool observeRuns(OriginalObservationRuns runs)
    {
        if(runs.records!=v_.sort_a || runs.partitions!=v_.sort_b)return fail(Status::Invariant);
        constexpr auto width=OriginalObservationRuns::PartitionRows;
        const auto original=v_.frame->original_ids;
        const auto partitions=(std::uint64_t{original}+width-1)/width;
        for(std::uint32_t part=0;part<partitions;++part){
            const auto first=part*width,end=original-first<width?original:first+width;
            const auto count=runs.partitions[part].unique;
            if(!count || count>end-first)return fail(Status::Invariant);
            auto position=first;
            for(std::uint32_t i=0;i<count;++i){const auto& run=runs.records[first+i];
                if(run.position!=position || !run.unique || run.unique>end-position || run.row>=v_.parameters.sourceRows())return fail(Status::Invariant);
                position+=run.unique;
            }
            if(position!=end)return fail(Status::Invariant);
        }
        if(!capacity())return true;
        for(std::uint32_t part=0;part<partitions;++part){
            const auto first=part*width,count=runs.partitions[part].unique;
            for(std::uint32_t i=0;i<count;++i){const auto& run=runs.records[first+i];auto remaining=run.unique;
                while(remaining){
                    // The first observation increments the clock before computing
                    // its epoch. Stop exactly before the next original epoch.
                    const auto first_clock=v_.policy->logical_clock+1;
                    const auto room=v_.parameters.agingAccesses()-first_clock%v_.parameters.agingAccesses();
                    const auto observed=room<remaining?static_cast<std::uint32_t>(room):remaining;
                    v_.policy->logical_clock+=observed;
                    frequency(run.row,true,observed);
                    if(v_.frame->status!=Status::Success)return false;
                    remaining-=observed;
                }
            }
        }
        return true;
    }

    /** @brief Initialize one independent setup directory record; native callers parallelize these disjoint writers. */
    QFN_ADAPTIVE_HD void initializeEntry(std::uint32_t i)
    { v_.entries[i]={{},0,0,None,None,Stream::None,0}; }
    /** @brief Initialize one independent setup free radix node. */
    QFN_ADAPTIVE_HD void initializeNode(std::uint32_t i)
    { v_.nodes[i]={{i+2<capacity()?i+1:None,None},None}; }
    /** @brief Initialize one independent setup frequency cell. */
    QFN_ADAPTIVE_HD void initializeFrequency(std::uint64_t i)
    { v_.frequency_epochs[i]=v_.frequency_counts[i]=0; }

    /** @brief Seal virgin controls only after all actual independent setup producers have completed. */
    QFN_ADAPTIVE_HD bool initializeControls()
    {
        if (!valid() || v_.frame->phase != LookupPhase::Uninitialized || v_.frame->ordinal ||
            (capacity() && v_.policy->flags)) return fail(Status::InvalidPhase);
        std::uint32_t expected = 0;
        for (std::uint32_t owner=0;owner<v_.parameters.participants();++owner)
        {
            const auto &g=v_.owners[owner];
            if (g.first_entry!=expected || expected>capacity() || g.slots>capacity()-expected)
                return fail(Status::InvalidGeometry);
            expected+=g.slots;
        }
        if (expected!=capacity()) return fail(Status::InvalidGeometry);
        if (capacity())
        {
            *v_.policy = {};
            v_.policy->directory_root = None;
            v_.policy->node_free = capacity() > 1 ? 0 : None;
            for (unsigned i = 0; i < 3; ++i) v_.policy->head[i] = v_.policy->tail[i] = None;
            v_.policy->aging_accesses = v_.parameters.agingAccesses();
            v_.policy->lane_width = v_.parameters.laneWidth();
            v_.policy->protected_limit = v_.parameters.protectedLimit();
            v_.policy->flags = OwnerInitialized;
        }
        *v_.frame = {};
        v_.frame->phase = LookupPhase::Idle;
        return true;
    }
    /** @brief Open original identity/bounds once; only canonical coalescers may publish its data and commit. */
    template<class Row> QFN_ADAPTIVE_HD bool openOriginal(std::uint64_t admission,const Row* originalRows,std::uint32_t count)
    {
        static_assert(std::is_same_v<Row,std::uint32_t> || std::is_same_v<Row,std::uint64_t>);
        if (!valid()) return fail(Status::InvalidGeometry);
        auto &f = *v_.frame;
        if (!admission || f.phase != LookupPhase::Idle || f.status != Status::Success ||
            admission < f.interval.admission || f.ordinal == UINT64_MAX ||
            (f.interval.table && (f.interval.table != v_.table || f.source_storage_generation != v_.source_storage_generation)) ||
            count > v_.parameters.maximumOriginal() ||
            (count && (!originalRows || aliases(originalRows,std::size_t{count}*sizeof(Row)))) ||
            (capacity() && (v_.policy->flags != OwnerInitialized || !policyMatches() ||
                UINT64_MAX-v_.policy->logical_clock < count ||
                (v_.policy->logical_clock+count)/v_.parameters.agingAccesses() > UINT32_MAX)))
            return fail(Status::InvalidPhase);
        const auto ordinal = f.ordinal + 1;
        f = {};
        f.ordinal = ordinal;
        f.interval = {v_.table,admission,ordinal};
        f.source_storage_generation = v_.source_storage_generation;
        f.initial_clock = capacity() ? v_.policy->logical_clock : 0;
        f.original_ids = count;
        for (std::uint32_t i = 0; i < v_.parameters.participants(); ++i) v_.owner_plans[i] = {};
        f.phase=LookupPhase::Coalescing;
        return true;
    }

    /** @brief Commit only an actual complete coalescing producer, after every disjoint write is ordered. */
    QFN_ADAPTIVE_HD bool finishOriginal()
    {
        if (!healthy(LookupPhase::Coalescing) || v_.frame->unique_rows>v_.parameters.maximumUnique() ||
            v_.frame->unique_rows>v_.frame->original_ids ||
            ((v_.frame->original_ids==0)!=(v_.frame->unique_rows==0))) return fail(Status::Invariant);
        v_.frame->phase=LookupPhase::Pinning;return true;
    }

    /** @brief One byte range in the exact native root declaration; never a physical allocation ledger. */
    struct Range { const void *data; std::size_t bytes; };
    /** @brief Describe all exact immutable/mutable banks in constant work for alias validation. */
    QFN_ADAPTIVE_HD void ranges(Range (&out)[15]) const
    {
        const auto k = std::size_t{v_.parameters.maximumUnique()}, n = std::size_t{v_.parameters.maximumOriginal()};
        const auto p = std::size_t{v_.parameters.participants()}, c = std::size_t{capacity()};
        const auto lanes = std::size_t{4}*v_.parameters.laneWidth();
        out[0] = {v_.frame,sizeof(LookupFrame)};
        out[1] = {v_.unique,k*sizeof(CoordinatorUnique)};
        out[2] = {v_.sort_a,n*sizeof(CoordinatorOccurrence)};
        out[3] = {v_.sort_b,n*sizeof(CoordinatorOccurrence)};
        out[4] = {v_.radix_counts,256*sizeof(std::uint32_t)};
        out[5] = {v_.original_to_unique,n*sizeof(std::uint32_t)};
        out[6] = {v_.owner_plans,p*sizeof(CoordinatorOwnerPlan)};
        out[7] = {v_.decisions,k*sizeof(AllocationDecision)};
        out[8] = {v_.owners,p*sizeof(CoordinatorOwnerGeometry)};
        out[9] = {v_.policy,c?sizeof(CoordinatorPolicyControl):0};
        out[10] = {v_.entries,c*sizeof(DirectoryEntry)};
        out[11] = {v_.nodes,(c?c-1:0)*sizeof(RadixNode)};
        out[12] = {v_.frequency_epochs,lanes*sizeof(std::uint32_t)};
        out[13] = {v_.frequency_counts,lanes*sizeof(std::uint16_t)};
        out[14] = {nullptr,0};
    }
    /** @return True only for a complete nonwrapping native range. */
    QFN_ADAPTIVE_HD static bool bounded(Range r)
    { return !r.bytes || (r.data && reinterpret_cast<std::uintptr_t>(r.data) <= ~std::uintptr_t{0}-r.bytes); }
    /** @return Exact byte-range intersection, independent of payload values. */
    QFN_ADAPTIVE_HD static bool overlap(Range a,Range b)
    {
        if (!a.bytes || !b.bytes) return false;
        if (!bounded(a) || !bounded(b)) return true;
        const auto x = reinterpret_cast<std::uintptr_t>(a.data), y = reinterpret_cast<std::uintptr_t>(b.data);
        return x < y+b.bytes && y < x+a.bytes;
    }
    /** @return Whether an incoming decoded prefix aliases any retained native root. */
    QFN_ADAPTIVE_HD bool aliases(const void *data,std::size_t bytes) const
    {
        const Range candidate{data,bytes};
        if (!bounded(candidate)) return true;
        Range banks[15];ranges(banks);
        for (const auto bank:banks) if (overlap(candidate,bank)) return true;
        return false;
    }
    /** @return Exact static slot→entry coordinate; it does not imply occupancy, pin or read permission. */
    QFN_ADAPTIVE_HD std::uint32_t index(std::uint32_t owner,std::uint32_t slot) const
    {
        if (owner >= v_.parameters.participants() || slot >= v_.owners[owner].slots) return None;
        const auto first = v_.owners[owner].first_entry;
        return first < capacity() && slot < capacity()-first ? first+slot : None;
    }
    /** @return Whether a unique record contains an actual owner lifetime rather than plan intent. */
    QFN_ADAPTIVE_HD static bool acquired(UniquePhase phase)
    { return phase == UniquePhase::Read || phase == UniquePhase::FreeGranted || phase == UniquePhase::VictimGranted; }
    /** @brief Fixed stable integer radix sort over the live prefix; original input/output banks never overlap. */
    QFN_ADAPTIVE_HD void sort(CoordinatorOccurrence *input,CoordinatorOccurrence *output,std::uint32_t count,bool positions)
    {
        const auto passes = positions?4u:8u;
        for (unsigned pass = 0; pass < passes; ++pass)
        {
            for (unsigned bin = 0; bin < 256; ++bin) v_.radix_counts[bin] = 0;
            for (std::uint32_t i = 0; i < count; ++i)
            {
                const auto key = positions?input[i].position:input[i].row;
                ++v_.radix_counts[(key>>(8*pass))&255];
            }
            std::uint32_t cursor = 0;
            for (unsigned bin = 0; bin < 256; ++bin)
            { const auto size=v_.radix_counts[bin];v_.radix_counts[bin]=cursor;cursor+=size; }
            for (std::uint32_t i = 0; i < count; ++i)
            {
                const auto key = positions?input[i].position:input[i].row;
                output[v_.radix_counts[(key>>(8*pass))&255]++] = input[i];
            }
            auto *swap=input;input=output;output=swap;
        }
    }
    /** @brief Compact one wave into append-only per-owner plan slices without altering unique/grant order. */
    QFN_ADAPTIVE_HD void buildPlans(bool victims)
    {
        std::uint32_t offset = victims?v_.frame->free_planned:0;
        for (std::uint32_t owner = 0; owner < v_.parameters.participants(); ++owner)
        {
            auto &p = v_.owner_plans[owner];
            if (victims) { p.victim_offset=offset;offset+=p.victim_count; }
            else { p.free_offset=offset;offset+=p.free_count; }
            p.plan_write=0;
        }
        if (offset > v_.parameters.maximumUnique()) { fail(Status::Invariant);return; }
        for (std::uint32_t i = 0; i < v_.frame->unique_rows; ++i)
        {
            auto &r = v_.unique[i];
            if (r.phase != (victims?UniquePhase::VictimPlanned:UniquePhase::FreePlanned)) continue;
            auto &p = v_.owner_plans[r.allocation_owner];
            r.decision_index = (victims?p.victim_offset:p.free_offset)+p.plan_write++;
            v_.decisions[r.decision_index] = r.decision;
        }
        for (std::uint32_t owner = 0; owner < v_.parameters.participants(); ++owner)
            v_.owner_plans[owner].plan_write=0;
    }
    /** @brief Reconcile exact actual owner completions before committing any wave phase or closing unaccepted plans. */
    template<class Receipt> QFN_ADAPTIVE_HD bool complete(const Receipt *receipts,std::uint32_t count,bool victims)
    {
        if (!healthy(victims?LookupPhase::VictimPlanned:LookupPhase::FreePlanned) || !receipts ||
            count != v_.parameters.participants() || aliases(receipts,std::size_t{count}*sizeof(Receipt)))
            return fail(Status::InvalidPhase);
        bool exhausted = true, cancelled = false;
        for (std::uint32_t owner = 0; owner < count; ++owner)
        {
            const auto &p = v_.owner_plans[owner];
            const auto &r = receipts[owner];
            const auto planned = victims?p.victim_count:p.free_count;
            const auto received = victims?p.victim_received:p.free_received;
            if (!same(r.lease.interval,interval()) || r.lease.owner != owner || r.lease.free_slots != p.lease.free_slots ||
                (r.status != Status::Success && r.status != Status::Cancelled) || r.granted != received ||
                (r.status == Status::Success ? r.granted != planned : (r.granted != 0 && r.granted != planned)))
                return fail(Status::Invariant);
            cancelled = cancelled || r.status == Status::Cancelled;
            exhausted = exhausted && r.status == Status::Success && r.granted == r.lease.free_slots;
        }
        const auto pending = victims?UniquePhase::VictimPlanned:UniquePhase::FreePlanned;
        std::uint32_t closed = 0;
        for (std::uint32_t i = 0; i < v_.frame->unique_rows; ++i)
        {
            const auto &r = v_.unique[i];
            if (r.phase != pending) continue;
            const auto owner = r.allocation_owner;
            if (receipts[owner].status != Status::Cancelled || receipts[owner].granted) return fail(Status::Invariant);
            ++closed;
        }
        if (closed != v_.frame->outstanding) return fail(Status::Invariant);
        for (std::uint32_t i = 0; i < v_.frame->unique_rows; ++i)
            if (v_.unique[i].phase == pending) v_.unique[i].phase = UniquePhase::SourceOnly;
        v_.frame->outstanding = 0;
        v_.frame->cancelled_completion = v_.frame->cancelled_completion || cancelled;
        if (!victims) v_.frame->free_exhausted = exhausted;
        v_.frame->phase = victims?LookupPhase::VictimCompleted:LookupPhase::FreeCompleted;
        return true;
    }
    /** @brief Apply exact original-position recency once, only to actual acquired owner lifetimes. */
    QFN_ADAPTIVE_HD bool applyQueues()
    {
        auto &f = *v_.frame;
        if (f.queues_applied || !f.observed) return true;
        for (std::uint32_t pos=0;pos<f.original_ids;)
        {
            const auto unique=v_.original_to_unique[pos];
            if(unique>=f.unique_rows)return fail(Status::Invariant);
            auto end=pos+1;
            while(end<f.original_ids && v_.original_to_unique[end]==unique)++end;
            auto& r=v_.unique[unique];
            if(acquired(r.phase)){
                if(r.phase!=UniquePhase::Read && !r.queue_started){
                    v_.entries[r.entry].touched=f.initial_clock+pos+1;
                    if(!append(r.entry,Stream::NewProbation))return false;
                    r.queue_started=true;
                    ++pos;
                }
                // One touch at the final repeat has exactly the same promotion,
                // possible demotion, list order/cursors and final clock. Further
                // identical touches cannot intervene with another row or policy
                // writer, even when protectedLimit is zero and this row demotes itself.
                if(pos<end && !touch(r.entry,f.initial_clock+end))return false;
            }
            pos=end;
        }
        f.queues_applied = true;
        return true;
    }
    CoordinatorView v_;
    /** @return Checked coordinator device dispatch order, unrelated to source unique/token/generation. */
    QFN_ADAPTIVE_HD CommandOrdinal nextCommand()
    {
        if (v_.frame->command_ordinal == UINT64_MAX) { fail(Status::Exhausted); return {0}; }
        return {++v_.frame->command_ordinal};
    }
    /** @return Sole declared metadata count, not an occupancy or free count. */
    QFN_ADAPTIVE_HD std::uint32_t capacity() const { return v_.parameters.entries(); }
    /** @return Exact immutable protocol identities. */
    QFN_ADAPTIVE_HD static bool same(Interval a, Interval b)
    { return a.table == b.table && a.admission == b.admission && a.sequence == b.sequence; }
    /** @return Exact original source retirement tuple. */
    QFN_ADAPTIVE_HD static bool sameSource(SourceWindow a, SourceWindow b)
    { return a.table == b.table && a.generation == b.generation && a.sequence == b.sequence; }
    /** @return Complete original locator identity, including full owner-local generation. */
    QFN_ADAPTIVE_HD static bool sameLocator(Locator a, Locator b)
    { return a.row == b.row && a.generation == b.generation && a.owner == b.owner && a.slot == b.slot; }
    /** @return Exact initial pin response, independent of physical slot generation issuance. */
    QFN_ADAPTIVE_HD static bool sameTarget(const Grant &a, const Grant &b)
    { return same(a.interval, b.interval) && a.row == b.row && a.generation == b.generation &&
             a.unique == b.unique && a.owner == b.owner && a.slot == b.slot; }
    /** @brief Preserve the first error and fail closed; no physical owner state is mutated. */
    QFN_ADAPTIVE_HD bool fail(Status status)
    {
        if (v_.frame && v_.frame->status == Status::Success)
            v_.frame->status = status;
        if (capacity() && v_.policy)
            v_.policy->flags |= OwnerFailed;
        return false;
    }
    /** @return None after recording a bounded structural error. */
    QFN_ADAPTIVE_HD std::uint32_t failIndex() { fail(Status::Invariant); return None; }
    /** @return Exact typed disjoint native roots in constant work; static owner offsets are setup-authenticated. */
    QFN_ADAPTIVE_HD bool valid() const
    {
        const auto &p = v_.parameters;
        if (!v_.frame || !v_.table || !v_.source_storage_generation || !p.sourceRows() ||
            !p.maximumUnique() || p.maximumUnique()>MaximumSlots || p.maximumOriginal()<p.maximumUnique() ||
            p.maximumOriginal()>MaximumSlots || !p.participants() || p.participants()>MaximumSlots ||
            v_.unique_capacity!=p.maximumUnique() || v_.original_capacity!=p.maximumOriginal() ||
            v_.owner_capacity!=p.participants() || !p.agingAccesses() || capacity()>MaximumSlots ||
            (capacity() && (!p.laneWidth() || (p.laneWidth() & (p.laneWidth()-1))))) return false;
        Range banks[15];ranges(banks);
        for (unsigned i=0;i<15;++i)
        {
            if (!bounded(banks[i])) return false;
            for (unsigned j=0;j<i;++j) if (overlap(banks[i],banks[j])) return false;
        }
        return true;
    }
    /** @return Exact installed arithmetic policy identity; changing scalars requires fresh setup rather than reinterpreting old epochs. */
    QFN_ADAPTIVE_HD bool policyMatches() const
    { return !capacity() || (v_.policy->aging_accesses==v_.parameters.agingAccesses() &&
        v_.policy->lane_width==v_.parameters.laneWidth() && v_.policy->protected_limit==v_.parameters.protectedLimit()); }
    /** @return Current exact phase and healthy coordinator. */
    QFN_ADAPTIVE_HD bool healthy(LookupPhase phase) const
    { return valid() && v_.frame->source_storage_generation == v_.source_storage_generation &&
             v_.frame->interval.table == v_.table && v_.frame->phase == phase && v_.frame->status == Status::Success &&
             (!capacity() || (v_.policy->flags == OwnerInitialized && policyMatches())); }
    /** @return Bounded already-defined unique record. */
    QFN_ADAPTIVE_HD bool record(std::uint32_t unique, LookupPhase phase)
    {
        return healthy(phase) && unique < v_.frame->unique_rows ? true : fail(Status::InvalidPhase);
    }
    /** @return The zero-source and consumed-source cases stay distinct from lookup ordinals. */
    QFN_ADAPTIVE_HD bool retired(SourceRetirement value, SourceWindow consumed) const
    {
        const auto source = v_.frame->source;
        return source.sequence ? (value == SourceRetirement::Consumed || value == SourceRetirement::Aborted) && sameSource(source, consumed) :
            value == SourceRetirement::Unpublished && !consumed.table && !consumed.generation && !consumed.sequence;
    }
    /** @return MSB-first row bit for the bounded binary Patricia directory. */
    QFN_ADAPTIVE_HD static unsigned bit(std::uint64_t row, unsigned at) { return unsigned((row >> (63 - at)) & 1); }
    /** @return First differing bit, or64 for a forbidden duplicate. */
    QFN_ADAPTIVE_HD static unsigned differing(std::uint64_t value)
    { unsigned n = 0; while (n < 64 && !(value & (std::uint64_t{1} << (63 - n)))) ++n; return n; }
    /** @brief Bounded parentless deletion; free-node reuse never grows tombstones or allocates. */
    QFN_ADAPTIVE_HD bool erase(std::uint32_t slot)
    {
        auto *edge = &v_.policy->directory_root;
        std::uint32_t *parentEdge = nullptr, parent = None, side = 0;
        const auto row = static_cast<std::uint64_t>(v_.entries[slot].locator.row);
        for (unsigned steps = 0; steps <= 64; ++steps)
        {
            const auto ref = *edge;
            if (ref == None)
                return fail(Status::Invariant);
            if (ref & LeafTag)
            {
                if ((ref & IndexMask) != slot)
                    return fail(Status::Invariant);
                if (parent == None)
                    *edge = None;
                else
                {
                    *parentEdge = v_.nodes[parent].child[1 - side];
                    v_.nodes[parent] = {{v_.policy->node_free, None}, None};
                    v_.policy->node_free = parent;
                }
                return true;
            }
            if (ref >= capacity() - 1 || v_.nodes[ref].split_bit >= 64)
                return fail(Status::Invariant);
            parentEdge = edge;
            parent = ref;
            side = bit(row, v_.nodes[ref].split_bit);
            edge = &v_.nodes[ref].child[side];
        }
        return fail(Status::Invariant);
    }
    /** @brief Insert one completely ready row with at most two bounded radix walks. */
    QFN_ADAPTIVE_HD bool insert(std::uint32_t slot)
    {
        auto &root = v_.policy->directory_root;
        const auto leaf = LeafTag | slot;
        const auto row = static_cast<std::uint64_t>(v_.entries[slot].locator.row);
        if (root == None)
        {
            root = leaf;
            return true;
        }
        auto ref = root;
        unsigned steps = 0;
        while (!(ref & LeafTag))
        {
            if (++steps > 64 || ref >= capacity() - 1 || v_.nodes[ref].split_bit >= 64)
                return fail(Status::Invariant);
            ref = v_.nodes[ref].child[bit(row, v_.nodes[ref].split_bit)];
            if (ref == None)
                return fail(Status::Invariant);
        }
        const auto other = ref & IndexMask;
        if (other >= capacity())
            return fail(Status::Invariant);
        const auto split = differing(row ^ static_cast<std::uint64_t>(v_.entries[other].locator.row));
        if (split >= 64)
            return fail(Status::Invariant);
        auto *edge = &root;
        steps = 0;
        while (!(*edge & LeafTag) && v_.nodes[*edge].split_bit < split)
        {
            if (++steps > 64)
                return fail(Status::Invariant);
            edge = &v_.nodes[*edge].child[bit(row, v_.nodes[*edge].split_bit)];
        }
        const auto node = v_.policy->node_free;
        if (node == None || node >= capacity() - 1)
            return fail(Status::Invariant);
        v_.policy->node_free = v_.nodes[node].child[0];
        const auto side = bit(row, split);
        v_.nodes[node].child[side] = leaf;
        v_.nodes[node].child[1 - side] = *edge;
        v_.nodes[node].split_bit = split;
        *edge = node;
        return true;
    }
    /** @brief Remove one policy-list member; cursor progress is not a physical eligibility permission. */
    QFN_ADAPTIVE_HD bool unlink(std::uint32_t at)
    {
        auto &e = v_.entries[at];
        const auto list = static_cast<unsigned>(e.stream);
        if (list >= 3 || (e.previous != None && e.previous >= capacity()) ||
            (e.next != None && e.next >= capacity()))
            return fail(Status::Invariant);
        auto &o = *v_.policy;
        if ((e.previous == None ? o.head[list] != at : v_.entries[e.previous].next != at) ||
            (e.next == None ? o.tail[list] != at : v_.entries[e.next].previous != at))
            return fail(Status::Invariant);
        if (v_.frame->eligible_cursor[list] == at)
            v_.frame->eligible_cursor[list] = e.next;
        if (e.previous == None) o.head[list] = e.next;
        else v_.entries[e.previous].next = e.next;
        if (e.next == None) o.tail[list] = e.previous;
        else v_.entries[e.next].previous = e.previous;
        if (list == 2)
        {
            if (!o.protected_count) return fail(Status::Invariant);
            --o.protected_count;
        }
        e.previous = e.next = None;
        e.stream = Stream::None;
        return true;
    }
    /** @brief Append one complete list member and maintain its exact full queue order. */
    QFN_ADAPTIVE_HD bool append(std::uint32_t at, Stream stream)
    {
        const auto list = static_cast<unsigned>(stream);
        if (list >= 3 || at >= capacity()) return fail(Status::Invariant);
        auto &e = v_.entries[at];
        auto &o = *v_.policy;
        const auto tail = o.tail[list];
        if (e.stream != Stream::None || e.previous != None || e.next != None ||
            (tail == None ? o.head[list] != None :
                (tail >= capacity() || v_.entries[tail].next != None ||
                 v_.entries[tail].stream != stream || less(at, tail))))
            return fail(Status::Invariant);
        e.previous = tail;
        e.next = None;
        e.stream = stream;
        if (tail == None) o.head[list] = at;
        else v_.entries[tail].next = at;
        o.tail[list] = at;
        if (list == 2) ++o.protected_count;
        if (v_.frame->eligible_cursor[list] == None && e.observed_frame != v_.frame->ordinal)
            v_.frame->eligible_cursor[list] = at;
        return true;
    }
    /** @brief Count one original access; a promotion/demotion changes policy only. */
    QFN_ADAPTIVE_HD bool touch(std::uint32_t at, std::uint64_t clock)
    {
        if (!unlink(at)) return false;
        v_.entries[at].touched = clock;
        if (!append(at, Stream::Protected)) return false;
        if (v_.policy->protected_count > v_.parameters.protectedLimit())
        {
            const auto demote = v_.policy->head[2];
            if (demote == None || !unlink(demote) || !append(demote, Stream::DemotedProbation))
                return false;
        }
        return true;
    }
    /** @return Fixed logical-time, original-row, directory-index tie order. */
    QFN_ADAPTIVE_HD bool less(std::uint32_t a, std::uint32_t b) const
    {
        const auto &x = v_.entries[a], &y = v_.entries[b];
        return x.touched != y.touched ? x.touched < y.touched :
            x.locator.row != y.locator.row ? x.locator.row < y.locator.row : a < b;
    }
    /** @return Next policy candidate not covered by an actual initial-pin reply in this frame. */
    QFN_ADAPTIVE_HD std::uint32_t eligible(unsigned list)
    {
        auto &at = v_.frame->eligible_cursor[list];
        for (std::uint64_t steps = 0; at != None && steps <= capacity(); ++steps)
        {
            if (at >= capacity()) return failIndex();
            const auto &e = v_.entries[at];
            if (e.observed_frame != v_.frame->ordinal)
                return e.present ? at : failIndex();
            at = e.next;
        }
        return at == None ? None : failIndex();
    }
    /** @return Oldest eligible probation across both queues, then protected; actual owner may still reject. */
    QFN_ADAPTIVE_HD std::uint32_t victim()
    {
        if (!capacity()) return None;
        const auto a = eligible(0), b = eligible(1);
        if (v_.frame->status != Status::Success) return None;
        if (a != None && b != None) return less(a, b) ? a : b;
        if (a != None) return a;
        if (b != None) return b;
        return eligible(2);
    }
    /** @return Canonical four-lane avalanche hash; source row high bits participate. */
    QFN_ADAPTIVE_HD static std::uint64_t hash(std::uint64_t n)
    {
        n ^= n >> 30;
        n *= 0xbf58476d1ce4e5b9ULL;
        n ^= n >> 27;
        n *= 0x94d049bb133111ebULL;
        return n ^ (n >> 31);
    }
    /** @return Four-lane lazy-aged frequency, with fixed saturating integer arithmetic. */
    QFN_ADAPTIVE_HD std::uint16_t frequency(std::uint64_t row, bool observe,std::uint32_t observations=1)
    {
        if (!capacity()) return 0;
        const auto epoch = static_cast<std::uint32_t>(v_.policy->logical_clock / v_.parameters.agingAccesses());
        std::uint16_t result = 0xffffU;
        for (std::uint64_t lane = 0; lane < 4; ++lane)
        {
            const auto at = lane * v_.parameters.laneWidth() +
                (hash(row + lane * 0x9e3779b97f4a7c15ULL) & (v_.parameters.laneWidth() - 1));
            if (v_.frequency_epochs[at] > epoch) { fail(Status::Invariant); return 0; }
            const auto elapsed = epoch - v_.frequency_epochs[at];
            auto count = static_cast<std::uint16_t>(v_.frequency_counts[at] >> (elapsed < 16 ? elapsed : 16));
            if (observe){
                const auto sum=std::uint64_t{count}+observations;
                count=static_cast<std::uint16_t>(sum<0xffffU?sum:0xffffU);
            }
            v_.frequency_epochs[at] = epoch;
            v_.frequency_counts[at] = count;
            if (count < result) result = count;
        }
        return result;
    }
};
} // namespace llaminar2::qwen4exp::adaptive_lookup::pooled
