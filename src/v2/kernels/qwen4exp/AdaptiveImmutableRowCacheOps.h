/**
 * @file AdaptiveImmutableRowCacheOps.h
 * @brief Allocation-free CPU/device integer operations for the bounded GPU row owner.
 *
 * One ordered metadata writer owns updates. Parallel probes finish and every
 * initial hit is pinned before policy mutation. Three sorted recency streams
 * preserve the existing probation/protected victim equation without a heap.
 * These operations issue no source I/O, native allocation or communication.
 */
#pragma once
#include "AdaptiveImmutableRowCacheState.h"
#include "AdaptiveImmutablePhysicalSlot.h"
#include <limits>
namespace llaminar2::qwen4exp::adaptive_lookup
{
/** @brief Actual shared metadata algorithm; native and device-free tests use the same implementation. */
template <class Slot> class Operations final
{
public:
    /** @brief Borrow one already-owned metadata view; no storage is materialized. */
    QFN_ADAPTIVE_HD explicit Operations(Metadata<Slot> view) : v_(view) {}
    /** @return Original state bits, independent of region/pin flags. */
    QFN_ADAPTIVE_HD static SlotState state(const Slot &s)
    {
        return PhysicalSlot<Slot>::state(s);
    }
    /** @return Full logical list membership, including pending Reserved rows. */
    QFN_ADAPTIVE_HD static Stream stream(const Slot &s)
    {
        return static_cast<Stream>((s.flags & StreamMask) >> 2);
    }
    /** @return Whether this frame retains the row's native backing. */
    QFN_ADAPTIVE_HD static bool pinned(const Slot &s)
    {
        return PhysicalSlot<Slot>::pinned(s);
    }
    /** @return Canonical integer avalanche hash shared with the CPU cache equation. */
    QFN_ADAPTIVE_HD static std::uint64_t hash(std::uint64_t n)
    {
        n ^= n >> 30;
        n *= 0xbf58476d1ce4e5b9ULL;
        n ^= n >> 27;
        n *= 0x94d049bb133111ebULL;
        return n ^ (n >> 31);
    }
    /** @brief Initialize one virgin physical slot during setup only; payload remains untouched. */
    QFN_ADAPTIVE_HD void initializeSlot(std::uint32_t i)
    {
        if (i < v_.capacity)
        {
            PhysicalSlot<Slot>::initialize(v_.slots[i]);
        }
    }
    /** @brief Initialize one pre-admitted directory node during setup only. */
    QFN_ADAPTIVE_HD void initializeNode(std::uint32_t i)
    {
        if (v_.capacity > 1 && i < v_.capacity - 1)
        {
            v_.nodes[i] = {{i + 1 < v_.capacity - 1 ? i + 1 : None, None}, None};
        }
    }
    /** @brief Initialize model-lifetime control once; request reset never calls this operation. */
    QFN_ADAPTIVE_HD void initializeOwner()
    {
        if (v_.capacity)
        {
            auto &o = *v_.owner;
            o = OwnerControl{};
            o.directory_root = None;
            o.node_free = v_.capacity > 1 ? 0 : None;
            o.slot_free = None;
            for (unsigned j = 0; j < 3; ++j)
                o.head[j] = o.tail[j] = None;
            o.flags = OwnerInitialized;
        }
    }
    /** @brief Initialize the sole mandatory frame even for a zero-slot domain. */
    QFN_ADAPTIVE_HD void initializeFrame()
    {
        *v_.frame = FrameControl{};
        v_.frame->phase = Phase::Idle;
        for (unsigned j = 0; j < 3; ++j)
            v_.frame->eligible_cursor[j] = None;
    }
    /** @brief Validate lifetime budgets before any pin/policy mutation and enter the one actual frame. */
    QFN_ADAPTIVE_HD bool begin(std::uint64_t originalIds)
    {
        auto &f = *v_.frame;
        if (f.phase != Phase::Idle)
            return fail(Status::InvalidPhase);
        if (originalIds > MaximumSlots || v_.capacity > MaximumSlots || !v_.source_rows ||
            (sizeof(v_.slots[0].row) == 4 && v_.source_rows > 0x100000000ULL) ||
            (v_.capacity && (!v_.lane_width || (v_.lane_width & (v_.lane_width - 1)) || !v_.slots ||
                             !v_.frequency_epochs || !v_.frequency_counts || (v_.capacity > 1 && !v_.nodes))))
            return fail(Status::InvalidGeometry);
        if (v_.capacity && (!v_.owner || v_.owner->flags != OwnerInitialized || !v_.aging_accesses ||
                            originalIds > ~std::uint64_t{0} - v_.owner->logical_clock ||
                            originalIds > ~std::uint64_t{0} - v_.owner->next_slot_generation ||
                            (v_.owner->logical_clock + originalIds) / v_.aging_accesses > 0xffffffffULL))
            return fail(Status::Exhausted);
        f = FrameControl{};
        f.original_ids = originalIds;
        f.phase = Phase::Building;
        for (unsigned j = 0; j < 3; ++j)
            f.eligible_cursor[j] = None;
        return true;
    }
    /** @return Exact Ready slot for this original row; bounded by the original source key width. */
    QFN_ADAPTIVE_HD std::uint32_t find(std::uint64_t row) const
    {
        if (!v_.capacity)
            return None;
        auto ref = v_.owner->directory_root;
        for (unsigned steps = 0; steps <= 64; ++steps)
        {
            if (ref == None)
                return None;
            if (ref & LeafTag)
            {
                const auto s = ref & IndexMask;
                if (s >= v_.capacity || state(v_.slots[s]) != SlotState::Ready ||
                    (v_.slots[s].flags & ~KnownSlotFlags))
                    return badFind();
                return static_cast<std::uint64_t>(v_.slots[s].row) == row ? s : None;
            }
            if (ref >= v_.capacity - 1 || v_.nodes[ref].split_bit >= 64)
                return badFind();
            ref = v_.nodes[ref].child[bit(row, v_.nodes[ref].split_bit)];
        }
        return badFind();
    }
    /** @brief Pin each unique initial Ready row before any reservation can evict it. */
    QFN_ADAPTIVE_HD bool pinInitial(std::uint32_t slot, std::uint64_t generation)
    {
        if (v_.frame->phase != Phase::Building || v_.frame->policy_started || slot >= v_.capacity)
            return fail(Status::InvalidPhase);
        auto &s = v_.slots[slot];
        if (state(s) != SlotState::Ready || s.generation != generation || pinned(s))
            return fail(Status::Invariant);
        PhysicalSlot<Slot>::pin(s);
        ++v_.frame->held_pins;
        return true;
    }
    /** @brief Start owner policy only after the complete initial-hit pin frontier. */
    QFN_ADAPTIVE_HD bool beginPolicy(std::uint64_t uniqueRows)
    {
        if (v_.frame->phase != Phase::Building || v_.frame->policy_started ||
            (uniqueRows > v_.frame->original_ids || uniqueRows < v_.frame->held_pins))
            return fail(Status::InvalidPhase);
        v_.frame->unique_rows = uniqueRows;
        v_.frame->policy_started = 1;
        if (v_.capacity)
            for (unsigned j = 0; j < 3; ++j)
                v_.frame->eligible_cursor[j] = v_.owner->head[j];
        return true;
    }
    /**
     * @brief Observe one original coordinate in order and optionally reserve its source row.
     * @param row Valid original source row.
     * @param prior Existing initial hit or earlier reservation for the same unique row.
     * @return Exact pinned slot/generation, or absence when optional admission declines.
     */
    QFN_ADAPTIVE_HD Decision observe(std::uint64_t row, Decision prior = {})
    {
        if (v_.frame->phase != Phase::Building || !v_.frame->policy_started ||
            v_.frame->observed_ids >= v_.frame->original_ids || row >= v_.source_rows)
        {
            fail(Status::InvalidPhase);
            return {};
        }
        ++v_.frame->observed_ids;
        if (!v_.capacity)
            return {};
        if (prior.slot != None)
        {
            if (prior.slot >= v_.capacity)
            {
                fail(Status::Invariant);
                return {};
            }
            const auto &s = v_.slots[prior.slot];
            if (!pinned(s) || s.generation != prior.generation || static_cast<std::uint64_t>(s.row) != row ||
                state(s) == SlotState::Empty)
            {
                fail(Status::Invariant);
                return {};
            }
        }
        auto &o = *v_.owner;
        ++o.logical_clock;
        const auto candidate = frequency(row, true);
        if (v_.frame->status != Status::Success)
            return {};
        if (prior.slot != None)
        {
            if (!touch(prior.slot, o.logical_clock))
                return {};
            return prior;
        }
        std::uint32_t slot = None;
        bool virgin = false;
        if (o.slot_free != None)
            slot = o.slot_free;
        else if (o.filled < v_.capacity)
        {
            slot = o.filled;
            virgin = true;
        }
        else
        {
            slot = victim();
            if (slot == None)
                return {};
            const auto resident = frequency(static_cast<std::uint64_t>(v_.slots[slot].row), false);
            if (v_.frame->status != Status::Success || candidate <= resident)
                return {};
        }
        if (o.next_slot_generation == ~std::uint64_t{0})
        {
            fail(Status::Exhausted);
            return {};
        }
        auto &s = v_.slots[slot];
        if (state(s) == SlotState::Ready)
        {
            if (v_.frame->evictions == 0xffffffffU)
            {
                fail(Status::Exhausted);
                return {};
            }
            if (pinned(s) || !erase(slot) || !unlink(slot))
            {
                fail(Status::Invariant);
                return {};
            }
            ++v_.frame->evictions;
        }
        else if (!virgin)
        {
            if (state(s) != SlotState::Empty)
            {
                fail(Status::Invariant);
                return {};
            }
            o.slot_free = s.previous;
        }
        else
            ++o.filled;
        const auto generation = ++o.next_slot_generation;
        PhysicalSlot<Slot>::reserve(s, row, generation, o.logical_clock);
        if (!append(slot, Stream::NewProbation))
            return {};
        ++v_.frame->held_pins;
        return {slot, generation};
    }
    /** @brief Seal owner decisions before exact source-frame compaction/publication. */
    QFN_ADAPTIVE_HD bool prepare()
    {
        if (v_.frame->phase != Phase::Building || !v_.frame->policy_started ||
            v_.frame->observed_ids != v_.frame->original_ids)
            return fail(Status::InvalidPhase);
        v_.frame->phase = Phase::Prepared;
        return v_.frame->status == Status::Success;
    }
    /** @brief Publish only an already-filled matching reservation after device ACK/admission validation. */
    QFN_ADAPTIVE_HD bool makeReady(Decision d)
    {
        if (v_.frame->status != Status::Success || v_.frame->phase != Phase::Validated ||
            d.slot >= v_.capacity || !v_.owner || v_.owner->flags != OwnerInitialized)
            return fail(Status::InvalidPhase);
        auto &s = v_.slots[d.slot];
        if (state(s) != SlotState::Reserved || !pinned(s) || s.generation != d.generation)
            return fail(Status::Invariant);
        if (v_.frame->promotions == 0xffffffffU)
            return fail(Status::Exhausted);
        PhysicalSlot<Slot>::ready(s);
        if (!insert(d.slot))
        {
            PhysicalSlot<Slot>::reserved(s);
            return false;
        }
        ++v_.frame->promotions;
        return true;
    }
    /** @brief Release one unique frame pin after its final native reader; cancel removes only still-Reserved
     * bytes. */
    QFN_ADAPTIVE_HD bool release(Decision d)
    {
        if (d.slot == None)
            return true;
        if (d.slot >= v_.capacity || !v_.frame->held_pins)
            return fail(Status::Invariant);
        auto &s = v_.slots[d.slot];
        if (s.generation != d.generation || !pinned(s))
            return fail(Status::Invariant);
        if (state(s) == SlotState::Reserved)
        {
            if (!unlink(d.slot))
                return false;
            PhysicalSlot<Slot>::recycle(s, v_.owner->slot_free);
            v_.owner->slot_free = d.slot;
        }
        else if (state(s) == SlotState::Ready)
            PhysicalSlot<Slot>::unpin(s);
        else
            return fail(Status::Invariant);
        --v_.frame->held_pins;
        return true;
    }
    /** @brief Finish after every actual pin/copy reader retires; caller cannot reset an active topology. */
    QFN_ADAPTIVE_HD bool finish()
    {
        if (v_.frame->phase == Phase::Idle || v_.frame->held_pins ||
            (v_.frame->status == Status::Success && v_.frame->phase != Phase::Validated))
            return fail(Status::InvalidPhase);
        v_.frame->phase = Phase::Idle;
        return v_.frame->status == Status::Success;
    }
    /** @brief Advance only the actual ordered frame frontier; used after native protocol validation. */
    QFN_ADAPTIVE_HD bool transition(Phase expected, Phase next)
    {
        if (v_.frame->phase != expected ||
            !((expected == Phase::Prepared &&
               (next == Phase::Published || (next == Phase::Validated && !v_.frame->source_rows))) ||
              (expected == Phase::Published && next == Phase::ResponseCopied) ||
              (expected == Phase::ResponseCopied && next == Phase::Validated)))
            return fail(Status::InvalidPhase);
        v_.frame->phase = next;
        return true;
    }
    /** @brief Make a source/invariant failure sticky in the existing optional owner. */
    QFN_ADAPTIVE_HD void poison()
    {
        if (v_.capacity)
            v_.owner->flags |= OwnerFailed;
    }
    /** @brief Reject an impossible operation while retaining the first bounded failure. */
    QFN_ADAPTIVE_HD bool fail(Status status)
    {
        if (v_.frame->status == Status::Success)
            v_.frame->status = status;
        return false;
    }

private:
    Metadata<Slot> v_;
    /** @return Absence only after recording a corrupt directory as a fatal owner error. */
    QFN_ADAPTIVE_HD std::uint32_t badFind() const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        atomicCAS(reinterpret_cast<unsigned *>(&v_.frame->status), static_cast<unsigned>(Status::Success),
                  static_cast<unsigned>(Status::Invariant));
#else
        if (v_.frame->status == Status::Success)
            v_.frame->status = Status::Invariant;
#endif
        return None;
    }
    /** @return Exact bit in source-MSB order; no source identity is truncated. */
    QFN_ADAPTIVE_HD static unsigned bit(std::uint64_t row, unsigned at)
    {
        return static_cast<unsigned>((row >> (63 - at)) & 1);
    }
    /** @return First differing source bit, with duplicates rejected by the caller. */
    QFN_ADAPTIVE_HD static unsigned differing(std::uint64_t x)
    {
        unsigned n = 0;
        while (n < 64 && !(x & (std::uint64_t{1} << (63 - n))))
            ++n;
        return n;
    }
    /** @brief Bounded parentless deletion; free-node reuse never grows tombstones or allocates. */
    QFN_ADAPTIVE_HD bool erase(std::uint32_t slot)
    {
        auto *edge = &v_.owner->directory_root;
        std::uint32_t *parentEdge = nullptr, parent = None, side = 0;
        const auto row = static_cast<std::uint64_t>(v_.slots[slot].row);
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
                    v_.nodes[parent] = {{v_.owner->node_free, None}, None};
                    v_.owner->node_free = parent;
                }
                return true;
            }
            if (ref >= v_.capacity - 1 || v_.nodes[ref].split_bit >= 64)
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
        auto &root = v_.owner->directory_root;
        const auto leaf = LeafTag | slot;
        const auto row = static_cast<std::uint64_t>(v_.slots[slot].row);
        if (root == None)
        {
            root = leaf;
            return true;
        }
        auto ref = root;
        unsigned steps = 0;
        while (!(ref & LeafTag))
        {
            if (++steps > 64 || ref >= v_.capacity - 1 || v_.nodes[ref].split_bit >= 64)
                return fail(Status::Invariant);
            ref = v_.nodes[ref].child[bit(row, v_.nodes[ref].split_bit)];
            if (ref == None)
                return fail(Status::Invariant);
        }
        const auto other = ref & IndexMask;
        if (other >= v_.capacity)
            return fail(Status::Invariant);
        const auto split = differing(row ^ static_cast<std::uint64_t>(v_.slots[other].row));
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
        const auto node = v_.owner->node_free;
        if (node == None || node >= v_.capacity - 1)
            return fail(Status::Invariant);
        v_.owner->node_free = v_.nodes[node].child[0];
        const auto side = bit(row, split);
        v_.nodes[node].child[side] = leaf;
        v_.nodes[node].child[1 - side] = *edge;
        v_.nodes[node].split_bit = split;
        *edge = node;
        return true;
    }
    /** @brief Remove a full-list member and keep its monotonic eligible cursor valid. */
    QFN_ADAPTIVE_HD bool unlink(std::uint32_t slot)
    {
        auto &s = v_.slots[slot];
        const auto list = static_cast<unsigned>(stream(s));
        if (list >= 3)
            return fail(Status::Invariant);
        auto &o = *v_.owner;
        if (v_.frame->eligible_cursor[list] == slot)
            v_.frame->eligible_cursor[list] = s.next;
        if (s.previous == None)
            o.head[list] = s.next;
        else
            v_.slots[s.previous].next = s.next;
        if (s.next == None)
            o.tail[list] = s.previous;
        else
            v_.slots[s.next].previous = s.previous;
        if (list == 2)
        {
            if (!o.protected_count)
                return fail(Status::Invariant);
            --o.protected_count;
        }
        s.previous = s.next = None;
        return true;
    }
    /** @brief Append to one proven sorted stream; a pinned append never rewinds an exhausted cursor. */
    QFN_ADAPTIVE_HD bool append(std::uint32_t slot, Stream streamId)
    {
        const auto list = static_cast<unsigned>(streamId);
        if (list >= 3)
            return fail(Status::Invariant);
        auto &s = v_.slots[slot];
        auto &o = *v_.owner;
        const auto tail = o.tail[list];
        if (tail != None && less(slot, tail))
            return fail(Status::Invariant);
        s.previous = tail;
        s.next = None;
        s.flags = (s.flags & ~StreamMask) | (list << 2);
        if (tail != None)
            v_.slots[tail].next = slot;
        else
            o.head[list] = slot;
        o.tail[list] = slot;
        if (list == 2)
            ++o.protected_count;
        if (v_.frame->eligible_cursor[list] == None && state(s) == SlotState::Ready && !pinned(s))
            v_.frame->eligible_cursor[list] = slot;
        return true;
    }
    /** @brief Assign actual logical access time, then enforce the complete protected limit including pinned
     * rows. */
    QFN_ADAPTIVE_HD bool touch(std::uint32_t slot, std::uint64_t clock)
    {
        if (!unlink(slot))
            return false;
        v_.slots[slot].touched = clock;
        if (!append(slot, Stream::Protected))
            return false;
        if (v_.owner->protected_count > v_.protected_limit)
        {
            const auto demote = v_.owner->head[2];
            if (demote == None || !unlink(demote))
                return fail(Status::Invariant);
            if (!append(demote, Stream::DemotedProbation))
                return false;
        }
        return true;
    }
    /** @return Exact recency/row/slot tie order within a region. */
    QFN_ADAPTIVE_HD bool less(std::uint32_t a, std::uint32_t b) const
    {
        const auto &x = v_.slots[a], &y = v_.slots[b];
        return x.touched != y.touched ? x.touched < y.touched : x.row != y.row ? x.row < y.row : a < b;
    }
    /** @return Next eligible entry; total pinned skips are bounded by actual frame membership transitions. */
    QFN_ADAPTIVE_HD std::uint32_t eligible(unsigned list)
    {
        auto &cursor = v_.frame->eligible_cursor[list];
        std::uint64_t skipped = 0;
        while (cursor != None)
        {
            if (cursor >= v_.capacity)
            {
                fail(Status::Invariant);
                return None;
            }
            const auto &s = v_.slots[cursor];
            if (state(s) == SlotState::Ready && !pinned(s))
                return cursor;
            if (++skipped > v_.frame->unique_rows)
            {
                fail(Status::Invariant);
                return None;
            }
            cursor = s.next;
        }
        return None;
    }
    /** @return Oldest eligible probation across both streams, then protected; never a pinned victim. */
    QFN_ADAPTIVE_HD std::uint32_t victim()
    {
        const auto a = eligible(0), b = eligible(1);
        if (v_.frame->status != Status::Success)
            return None;
        if (a != None && b != None)
            return less(a, b) ? a : b;
        if (a != None)
            return a;
        if (b != None)
            return b;
        return eligible(2);
    }
    /** @return Four-lane lazy-aged estimate, with the CPU policy's exact saturation/aging order. */
    QFN_ADAPTIVE_HD std::uint16_t frequency(std::uint64_t row, bool observe)
    {
        const auto epoch = static_cast<std::uint32_t>(v_.owner->logical_clock / v_.aging_accesses);
        std::uint16_t result = 0xffffU;
        for (std::uint64_t lane = 0; lane < 4; ++lane)
        {
            const auto at =
                lane * v_.lane_width + (hash(row + lane * 0x9e3779b97f4a7c15ULL) & (v_.lane_width - 1));
            if (v_.frequency_epochs[at] > epoch)
            {
                fail(Status::Invariant);
                return 0;
            }
            const auto elapsed = epoch - v_.frequency_epochs[at];
            auto count = static_cast<std::uint16_t>(v_.frequency_counts[at] >> (elapsed < 16 ? elapsed : 16));
            if (observe && count != 0xffffU)
                ++count;
            v_.frequency_epochs[at] = epoch;
            v_.frequency_counts[at] = count;
            if (count < result)
                result = count;
        }
        return result;
    }
};
} // namespace llaminar2::qwen4exp::adaptive_lookup
