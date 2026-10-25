/**
 * @file CPUSharedGDNSnapshots.h
 * @brief Arena-backed immutable CPU GDN epoch checkpoints and raw-update prefixes.
 *
 * A candidate names a checkpoint and a log prefix. Only fixed epoch boundaries
 * copy matrices; acceptance serializes bytes without executing arithmetic.
 * BufferArena/PMA owns every byte. Views borrow that storage and never allocate,
 * resize, release, or maintain a second capacity ledger. Three uint32 protocol
 * words encode initial phase, published row count, and planned row count using
 * memcpy so fast-math cannot reinterpret lifecycle metadata as floating point.
 * begin() retires publication, append()/crossed() fill admitted storage, and
 * complete() publishes the finished head. Stage generation authenticates the
 * lifetime of an explicit transaction independently of shared-kernel rebinding.
 */
#pragma once
#include "CPUGDNStateLayout.h"
#include <cstring>
#include <algorithm>
#include <functional>
#include <utility>
namespace llaminar2 {
/**
 * @brief Explicit transaction borrowed from its graph stage's retained arena.
 *
 * The stage and its generation authority must outlive publication. A backend
 * clear/rebind does not retire this view: the stage increments its generation
 * on execution attempts, reset, and workspace replacement. Width describes
 * physical storage; live_width describes accepted flattened row indices.
 */
struct CPUSharedGDNTransaction {
    float *slots; ///< Arena-owned checkpoint/raw-log payload, immutable after publication.
    size_t slot_floats; ///< Logical admitted payload extent, excluding alignment padding.
    int width; ///< Fixed admitted row capacity per request.
    int requests; ///< Fixed number of request partitions in this graph.
    int live_width; ///< Flattened row-index stride; independent of admitted physical width.
    int heads; ///< Participant-local value-head count.
    int dk; ///< Key width of each checkpoint and raw update.
    int dv; ///< Value width of each checkpoint and raw update.
    const uint64_t *generation_authority; ///< Stage-owned lifetime, independent of kernel rebinding.
    uint64_t generation; ///< Exact stage transaction generation observed at publication.
};
/**
 * @brief One accepted row borrowed from a completed stage-owned transaction.
 *
 * Planning retains only this immutable identity. The publisher validates every
 * transaction and source/live span before creating copy/zero/phase tasks; no
 * mutable work bank is populated on the planning thread.
 */
struct CPUSharedGDNAcceptedPrefix {
    CPUSharedGDNTransaction transaction; ///< Stage-owned slots, geometry, and generation authority.
    int row; ///< Selected live row within the single request represented by this plan.
};
/**
 * @brief Immutable logical row admission, independent of physical alignment padding.
 *
 * This typed BOM input carries the explicitly declared verifier capacity from
 * graph setup through binding. Rounded physical bytes never advertise extra
 * rows, even when adjacent small geometries occupy the same aligned head span.
 */
class CPUSharedGDNLogicalCapacity final {
    int flat_rows_; ///< Declared complete flattened row capacity.
    int requests_; ///< Fixed number of independently owned request partitions.
public:
    /**
     * @brief Validate a whole per-request logical verifier capacity.
     * @param flat_rows Explicitly declared flattened row ceiling, not live rows.
     * @param requests Fixed graph request count.
     * @throws std::invalid_argument for absent, fractional, or overflowing geometry.
     */
    CPUSharedGDNLogicalCapacity(int flat_rows,int requests)
        : flat_rows_(flat_rows),requests_(requests) {
        if (flat_rows<=0 || requests<=0 || flat_rows%requests ||
            flat_rows/requests>std::numeric_limits<int>::max()-8)
            throw std::invalid_argument("CPU shared GDN requires explicit whole per-request logical capacity");
    }
    /** @return Exact declared flattened row ceiling. */
    int flatRows() const noexcept { return flat_rows_; }
    /** @return Fixed request partition count. */
    int requestCount() const noexcept { return requests_; }
    /** @return Declared per-request row ceiling, unaffected by live replay width. */
    int rowsPerRequest() const noexcept { return flat_rows_/requests_; }
    /**
     * @brief Compute the aligned physical contribution of this logical capacity.
     * @param layout Validated complete-bank geometry including local value heads.
     * @return Exact FP32-sized slot extent across all declared request/head partitions.
     * @throws std::overflow_error if partition multiplication exceeds size_t.
     * The caller's PMA/arena authority owns allocation; this value is only a BOM input.
     */
    size_t slotFloats(const CPUGDNStateLayout &layout) const;
};
/** @brief Typed candidate identity within one retained head transaction. */
struct CPUGDNCandidate {
    int checkpoint; ///< Fixed epoch containing the candidate's rounded checkpoint.
    int prefix; ///< Exclusive raw-log endpoint from the initial epoch's start.
    int phase; ///< Number of pending records in that candidate's current epoch.
};
/**
 * @brief Borrowed per-head arena view with an explicit build/publish lifecycle.
 *
 * Callers admit headFloats() elements before construction and serialize writes
 * to a head. Read-only selection follows complete(); retiring or rebuilding a
 * head invalidates its candidates. Geometry and storage remain fixed across
 * smaller live replays. No operation changes the canonical arithmetic boundary.
 */
class CPUSharedGDNSnapshots final {
    CPUGDNStateLayout layout_; ///< Immutable canonical state geometry.
    float *base_; ///< Borrowed arena payload; this view never owns an allocation.
    int rows_; ///< Admitted per-request capacity, independent of live row count.
    /** @brief Local build permission; published reader views use persisted metadata. */
    enum class BuildLifecycle : uint8_t {
        ReadOnly, Retired, PendingInitialReadout, Appending, Published
    };
    mutable BuildLifecycle build_lifecycle_=BuildLifecycle::ReadOnly;
    /**
     * @brief Read lifecycle metadata without floating-point conversions.
     * @param index Protocol word index, restricted internally to 0..2.
     * @return Stored uint32 bits; validation belongs to the consuming operation.
     */
    uint32_t metadata(int index) const {
        uint32_t v;
        std::memcpy(&v,base_+index,sizeof(v));
        return v;
    }
    /**
     * @brief Store lifecycle metadata without changing allocation ownership.
     * @param index Protocol word index, restricted internally to 0..2.
     * @param v Exact uint32 protocol value to publish.
     */
    void metadata(int index,uint32_t v) const {
        std::memcpy(base_+index,&v,sizeof(v));
    }
    /**
     * @brief Retire prior publication and prepare validated metadata/live history.
     * @param initial Immutable disjoint canonical input head.
     * @param live_rows Planned candidate count in 0..admitted capacity.
     * @throws std::logic_error if an initial-readout callback attempts reentry.
     * @throws std::invalid_argument for absent storage/input, invalid phase, or extent.
     * The checkpoint remains unpublished and untouched until its owning begin
     * path either copies it immediately or completes the synchronous first readout.
     * A failed admission leaves every prior candidate unpublished.
     */
    void beginMetadataAndHistory(const float *initial,int live_rows) const {
        if (build_lifecycle_==BuildLifecycle::PendingInitialReadout)
            throw std::logic_error("CPU shared GDN cannot restart during its initial readout");
        build_lifecycle_=BuildLifecycle::Retired;
        if (!base_) throw std::invalid_argument("CPU shared GDN begin requires retained storage");
        metadata(1,0); metadata(2,0);
        if (!initial || live_rows<0 || live_rows>rows_ ||
            std::bit_cast<uint32_t>(initial[0])>0x40e00000u || initial[0]!=static_cast<float>(static_cast<int>(initial[0])))
            throw std::invalid_argument("CPU shared GDN initial phase or live extent is invalid");
        const int phase=static_cast<int>(initial[0]);
        metadata(0,static_cast<uint32_t>(phase));
        metadata(2,static_cast<uint32_t>(live_rows));
        std::memcpy(record(0),layout_.record(initial,0),phase*layout_.recordFloats()*sizeof(float));
    }
    /**
     * @brief Reject writes from a reader, retired, or already published view.
     * @throws std::logic_error unless this view owns a pending/appending build.
     */
    void requireBuilding() const {
        if (build_lifecycle_!=BuildLifecycle::PendingInitialReadout &&
            build_lifecycle_!=BuildLifecycle::Appending)
            throw std::logic_error("CPU shared GDN mutation requires an exclusive unpublished build");
    }
public:
    static constexpr size_t kHeaderFloats=16; ///< One cache line retains three protocol words and named alignment padding.
    static constexpr size_t kHeadAlignmentFloats=16; ///< Independent heads start on cache-line boundaries.
    /**
     * @brief Borrow a head's already admitted persistent transaction storage.
     * @param layout Validated canonical checkpoint/history geometry.
     * @param base Payload with at least headFloats(layout, rows) elements; a null
     *        inactive view may be constructed but must not be mutated.
     * @param rows Fixed positive admitted per-request row capacity.
     * Construction performs no allocation or publication and preserves payloads.
     */
    CPUSharedGDNSnapshots(CPUGDNStateLayout layout, float *base, int rows)
        : layout_(layout), base_(base), rows_(rows) {}
    /** @brief Do not duplicate local build permission; construct fresh metadata readers. */
    CPUSharedGDNSnapshots(const CPUSharedGDNSnapshots &)=delete;
    /** @brief Do not move a pending readout permission away from its synchronous owner. */
    CPUSharedGDNSnapshots(CPUSharedGDNSnapshots &&)=delete;
    /** @brief Prevent replacing a pending lifecycle with another view's permissions. */
    CPUSharedGDNSnapshots &operator=(const CPUSharedGDNSnapshots &)=delete;
    /** @brief Prevent replacing a pending lifecycle through move assignment. */
    CPUSharedGDNSnapshots &operator=(CPUSharedGDNSnapshots &&)=delete;
    /**
     * @brief Size immutable checkpoints for every possible starting phase.
     * @param rows Positive admitted per-request row capacity.
     * @return Initial checkpoint plus every potentially crossed fixed epoch.
     * @throws std::invalid_argument if rows cannot fit checked prefix arithmetic.
     */
    static size_t checkpoints(int rows) {
        if (rows<=0 || rows>std::numeric_limits<int>::max()-8)
            throw std::invalid_argument("CPU shared GDN row extent exceeds checked prefix ABI");
        return 1 + (static_cast<size_t>(rows)+7)/8;
    }
    /**
     * @brief Compute the exact per-head payload contribution to the typed BOM.
     * @param l Validated canonical checkpoint/history geometry.
     * @param rows Positive admitted per-request row capacity.
     * @return FP32-sized elements for a cache-line header containing three protocol
     *         words, immutable checkpoints, at most seven initial raw records, one
     *         record per new row, and explicit cache-line head-stride alignment.
     * @throws std::invalid_argument if rows cannot fit checked prefix arithmetic.
     * The caller's admission authority multiplies heads/requests and owns bytes.
     */
    static size_t headFloats(const CPUGDNStateLayout &l, int rows) {
        const size_t payload=kHeaderFloats+checkpoints(rows)*l.matrixFloats()+
            (static_cast<size_t>(rows)+7)*l.recordFloats();
        return (payload+kHeadAlignmentFloats-1)&~(kHeadAlignmentFloats-1);
    }
    /**
     * @brief Reject overlapping publication buffers before any state is mutated.
     * @param left First borrowed buffer address.
     * @param left_bytes Full logical extent of the first buffer.
     * @param right Second borrowed buffer address.
     * @param right_bytes Full logical extent of the second buffer.
     * @return True for overlap or overflowing address ranges; false for disjoint
     *         checked spans. This check neither probes nor changes payload bytes.
     */
    static bool buffersOverlap(const void *left,size_t left_bytes,
        const void *right,size_t right_bytes) {
        const auto l=reinterpret_cast<uintptr_t>(left);
        const auto r=reinterpret_cast<uintptr_t>(right);
        if (left_bytes>std::numeric_limits<uintptr_t>::max()-l ||
            right_bytes>std::numeric_limits<uintptr_t>::max()-r) return true;
        return l<r+right_bytes && r<l+left_bytes;
    }
    /**
     * @brief Locate one admitted fixed-epoch checkpoint.
     * @param epoch Zero-based checkpoint identity within this retained head.
     * @return Borrowed matrix payload; writes belong only to begin()/crossed()/initial-readout retention
     *         before publication, and consumers retain no independent ownership.
     * @throws std::out_of_range if epoch is outside admitted checkpoint storage.
     * @throws std::invalid_argument if the view has invalid admitted row geometry.
     */
    float *checkpoint(int epoch) const {
        if (epoch<0 || static_cast<size_t>(epoch)>=checkpoints(rows_))
            throw std::out_of_range("CPU shared GDN checkpoint identity is invalid");
        return base_+kHeaderFloats+static_cast<size_t>(epoch)*layout_.matrixFloats();
    }
    /**
     * @brief Locate raw history without copying a candidate-specific prefix.
     * @param prefix Record index from the initial epoch's start; the exclusive
     *        endpoint rows+7 is permitted only for zero-length boundary copies.
     * @return Borrowed record storage, or its one-past endpoint for an empty copy.
     * @throws std::out_of_range if prefix is outside the admitted raw-log extent.
     * @throws std::invalid_argument if the view has invalid admitted row geometry.
     */
    float *record(int prefix) const {
        if (prefix<0 || prefix>rows_+7)
            throw std::out_of_range("CPU shared GDN raw prefix identity is invalid");
        return base_+kHeaderFloats+checkpoints(rows_)*layout_.matrixFloats()+static_cast<size_t>(prefix)*layout_.recordFloats();
    }
    /**
     * @brief Start an unpublished transaction before mutable state advances.
     * @param initial Complete canonical head whose immutable bytes remain valid
     *        throughout this call; it must not overlap the destination payload.
     * @param live_rows Planned rows in 0..admitted capacity for this head.
     * @throws std::invalid_argument for null input, absent storage, invalid phase, or extent.
     * @throws std::logic_error if invoked during a pending initial-readout callback.
     * The initial checkpoint and live history are copied once. Candidate reads
     * remain invalid until complete(); beginning again retires prior publication.
     */
    void begin(const float *initial,int live_rows) const {
        beginMetadataAndHistory(initial,live_rows);
        std::memcpy(checkpoint(0),initial+1,layout_.matrixFloats()*sizeof(float));
        build_lifecycle_=BuildLifecycle::Appending;
    }
    /**
     * @brief Retain the initial checkpoint as part of a synchronous first readout.
     * @tparam InitialReadout Callable accepting immutable source and retained target matrices.
     * @param initial Disjoint canonical head borrowed only during this operation.
     * @param live_rows Positive planned candidate count within admitted capacity.
     * @param readout Synchronous first-group operation that stores every source
     *        matrix element into the retained target while computing its readout.
     * @throws std::invalid_argument for absent storage/input, phase, or row extent.
     * @throws std::logic_error for reentry or an attempted publication/retirement
     *         while the initial readout owns the retention dependency.
     * @throws ... Propagates any callback exception after retiring publication.
     *
     * Initial history and metadata are prepared before invoking readout. The
     * checkpoint is fully retained when the callback returns; only then can the
     * builder append later groups, advance epochs, or call complete(). Neither
     * source pointer nor callback is retained after this synchronous boundary.
     * New reader views can later materialize persisted metadata independently of
     * this local build permission; no additional persisted bytes are required.
     */
    template<class InitialReadout>
    void beginWithInitialReadout(const float *initial,int live_rows,InitialReadout &&readout) const {
        beginMetadataAndHistory(initial,live_rows);
        if (live_rows==0)
            throw std::invalid_argument("CPU shared GDN initial readout requires a live candidate");
        build_lifecycle_=BuildLifecycle::PendingInitialReadout;
        try {
            std::invoke(std::forward<InitialReadout>(readout),initial+1,checkpoint(0));
            build_lifecycle_=BuildLifecycle::Appending;
        } catch (...) {
            metadata(1,0);
            build_lifecycle_=BuildLifecycle::Retired;
            throw;
        }
    }

    /**
     * @brief Validate a candidate against complete integer protocol metadata.
     * @param row Zero-based candidate within this request's live row sequence.
     * @return True only for a completed head and a row in its published extent.
     * No payload or lifecycle state is changed on an invalid candidate.
     */
    bool validRow(int row) const {
        return base_ && metadata(0)<8u && metadata(2)<=static_cast<uint32_t>(rows_) && metadata(1)==metadata(2) &&
            row>=0 && static_cast<uint32_t>(row)<metadata(1);
    }
    /**
     * @brief Retire every candidate while retaining arena capacity and payloads.
     * @throws std::logic_error if invoked during a pending initial-readout callback.
     * @throws std::invalid_argument for an inactive view without storage.
     * The owning stage uses this before an execution attempt, reset, or rebind;
     * an empty participant publishes no candidate or stale prior-request header.
     */
    void retire() const {
        if (build_lifecycle_==BuildLifecycle::PendingInitialReadout)
            throw std::logic_error("CPU shared GDN cannot retire during its initial readout");
        if (!base_) throw std::invalid_argument("CPU shared GDN retirement requires retained storage");
        build_lifecycle_=BuildLifecycle::Retired;
        metadata(0,0);
        metadata(1,0);
        metadata(2,0);
    }
    /**
     * @brief Publish the planned candidate extent after all head writes complete.
     * @throws std::logic_error for an unbegun reader, pending initial readout,
     *         retired build, or already published build.
     * The single head owner calls this only after every append()/crossed() edge;
     * readers reject the unpublished state between begin() and this operation.
     */
    void complete() const {
        if (build_lifecycle_!=BuildLifecycle::Appending)
            throw std::logic_error("CPU shared GDN publication requires a completed initial retention");
        metadata(1,metadata(2));
        build_lifecycle_=BuildLifecycle::Published;
    }
    /**
     * @brief Derive an epoch/prefix identity without replaying recurrence arithmetic.
     * @param row Zero-based row in this head's planned transaction.
     * @return Checkpoint identity, exclusive raw-log prefix, and canonical phase.
     * @throws std::out_of_range if metadata or row is not a valid planned candidate.
     * Builders may resolve a planned identity before complete(); materialization
     * additionally requires that the entire head has been published.
     */
    CPUGDNCandidate candidate(int row) const {
        if (!base_ || metadata(0)>=8 || metadata(2)>static_cast<uint32_t>(rows_) || row<0 || static_cast<uint32_t>(row)>=metadata(2))
            throw std::out_of_range("CPU shared GDN candidate is not planned");
        const int prefix=static_cast<int>(metadata(0))+row+1;
        return {prefix/8,prefix,prefix%8};
    }
    /**
     * @brief Retain one prepared raw update before a working epoch can flush.
     * @param row Zero-based planned candidate owning this new update.
     * @param raw Readable key/value/decay/beta record of layout.recordFloats()
     *        elements, disjoint from its retained destination.
     * @throws std::out_of_range if row or planned metadata is invalid.
     * @throws std::logic_error if this view does not own an unpublished build.
     * The head owner appends each row once, preserving canonical serial order.
     */
    void append(int row, const float *raw) const {
        requireBuilding();
        if (!base_ || metadata(0)>=8 || metadata(2)>static_cast<uint32_t>(rows_) || row<0 || static_cast<uint32_t>(row)>=metadata(2))
            throw std::out_of_range("CPU shared GDN update is not planned");
        std::memcpy(record(static_cast<int>(metadata(0))+row),raw,layout_.recordFloats()*sizeof(float));
    }
    /**
     * @brief Retain a checkpoint produced by one complete eight-update flush.
     * @param row Planned candidate ending exactly at a fixed epoch boundary.
     * @param working Complete mutable canonical head immediately after the flush;
     *        its checkpoint is disjoint from retained immutable storage.
     * @throws std::out_of_range for an invalid candidate/checkpoint identity.
     * @throws std::logic_error if this view does not own an unpublished build.
     * The caller invokes this only at the canonical boundary; no arithmetic or
     * partial-history materialization occurs in snapshot publication.
     */
    void crossed(int row, const float *working) const {
        requireBuilding();
        std::memcpy(checkpoint(candidate(row).checkpoint),working+1,layout_.matrixFloats()*sizeof(float));
    }
    /**
     * @brief Serialize one completed candidate without recurrence arithmetic.
     * @param row Zero-based row in the published live extent.
     * @param target Complete writable canonical head, disjoint from retained
     *        checkpoints/raw log; its allocation remains owned by the caller.
     * @throws std::out_of_range if the candidate is unpublished or invalid.
     * Copies the named checkpoint and exact pending prefix, then clears inactive
     * history to canonical zero bytes. Retained candidate payloads remain intact.
     */
    void materialize(int row,float *target) const {
        if (!validRow(row)) throw std::out_of_range("CPU shared GDN candidate is unpublished");
        const auto c=candidate(row);
        target[0]=static_cast<float>(c.phase);
        std::memcpy(target+1,checkpoint(c.checkpoint),layout_.matrixFloats()*sizeof(float));
        std::memcpy(layout_.record(target,0),record(c.prefix-c.phase),c.phase*layout_.recordFloats()*sizeof(float));
        std::fill(layout_.record(target,c.phase),target+layout_.headFloats(),0.0f);
    }
};

inline size_t CPUSharedGDNLogicalCapacity::slotFloats(const CPUGDNStateLayout &layout) const {
    const size_t heads=static_cast<size_t>(layout.floats()/layout.headFloats());
    const size_t head_floats=CPUSharedGDNSnapshots::headFloats(layout,rowsPerRequest());
    const size_t requests=static_cast<size_t>(requests_);
    if (heads>std::numeric_limits<size_t>::max()/requests ||
        head_floats>std::numeric_limits<size_t>::max()/(heads*requests))
        throw std::overflow_error("CPU shared GDN logical capacity exceeds physical slot extent");
    return heads*requests*head_floats;
}
}
