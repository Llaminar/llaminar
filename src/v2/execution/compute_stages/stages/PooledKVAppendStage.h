/**
 * @file PooledKVAppendStage.h
 * @brief Native pooled-page KV append stage for CPU single-row decode.
 *
 * This is the P2 production-caller append: one compute stage writes ONE
 * decode row's K/V into a pooled table page and publishes the immutable
 * publication ref into a caller-owned ref cell that the attention stage
 * consumes in the same graph. It is the pooled counterpart of
 * `KVCacheAppendStage` and is deliberately fail-closed: only a single-row
 * CPU FP32 decode is supported, the table/pool must be the SAME admitted
 * instance, and every failure refuses before mutating any page.
 *
 * The stage performs no allocation, no host shadow and no full-history
 * gather. Page bytes come from the caller-admitted `PhysicalPagePool::Grant`;
 * the final immutable publication is a `CPUFP32PagedKVTable::Snapshot` whose
 * exact typed ref is minted by the table owner and written into `ref_out`
 * before the stage returns. The graph orders this stage before attention.
 */
#pragma once
#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "kernels/PagedKVSnapshotRef.h"
#include "planning/PhysicalPagePool.h"
#include "kernels/cpu/attention/CPUFP32PagedKVTable.h"

namespace llaminar2
{
/**
 * @brief Caller-owned live pooled publication (ref + retained Snapshot).
 *
 * The append stage mints the exact ref AND retains the published Snapshot in
 * the same cell; the attention stage resolves the ref while the Snapshot is
 * still alive. The cell is the ownership unit: both writers/readers see one
 * publication. A snappshot is move-only, so a cell is move-only too.
 */
class PooledPublicationCell final
{
public:
    PooledPublicationCell() = default;
    ~PooledPublicationCell() = default;
    PooledPublicationCell(PooledPublicationCell &&) = default;
    PooledPublicationCell &operator=(PooledPublicationCell &&) = default;
    PooledPublicationCell(const PooledPublicationCell &) = delete;
    PooledPublicationCell &operator=(const PooledPublicationCell &) = delete;
    PagedKVSnapshotRef ref;                              ///< Exact typed ref of the live publication.
    CPUFP32PagedKVTable::Snapshot snapshot;              ///< Retained publication (keeps the bank alive).
    bool live() const noexcept { return ref.valid(); }
};

class PooledKVAppendStage final : public IComputeStage
{
public:
    struct Params
    {
        STAGE_PARAMS_COMMON_FIELDS;
        const ITensor *K = nullptr; ///< Key row(s) to append (CPU FP32).
        const ITensor *V = nullptr; ///< Value row(s) to append (CPU FP32).
        /** Exact admitted pool/table under ONE PMA instance (never a foreign pair). */
        PhysicalPagePool *pool = nullptr;
        CPUFP32PagedKVTable *table = nullptr;
        /** Caller-owned live publication cell written by this stage on success. */
        PooledPublicationCell *ref_out = nullptr;
        /** Per-request release level: request record identity for binding. */
        std::uint64_t request_id = 0;
        /** Per-request slot (identity metadata, never a page). */
        int slot = 0;
        int layer_idx = -1;
        int kv_heads = 0;
        int head_dim = 0;
        int page_tokens = 0;
        /** Live rows this graph appends (decode always 1; buffer capacity is larger). */
        int live_rows = 1;
        /** Prior publication of the SAME request/layer, or null on the first row.
         * Every prior page is shared into the new bank (immutable sharing);
         * only the new row page is freshly written (private COW tail). */
        const PooledPublicationCell *prior = nullptr;
    };

    explicit PooledKVAppendStage(Params params);
    bool execute(IDeviceContext *ctx) override;
    ComputeStageType type() const override { return ComputeStageType::KV_CACHE_APPEND; }
    bool supportsBackend(ComputeBackendType backend) const override { return backend == ComputeBackendType::CPU; }
    std::string name() const override { return "PooledKVAppend"; }
    StageDumpInfo buildDumpInfoImpl() const override;

private:
    Params params_;
};
} // namespace llaminar2
