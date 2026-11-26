/**
 * @file PooledKVAppendStage.h
 * @brief Native pooled-page KV append stage for CPU single-row decode.
 *
 * This is the P2 production-caller append: one compute stage writes a decode
 * row's K/V into a pooled table page and publishes the immutable publication
 * ref into a caller-owned ref cell that the attention stage consumes in the
 * same graph. It is the pooled counterpart of `KVCacheAppendStage` and is
 * deliberately fail-closed: only a single-row CPU FP32 decode is supported,
 * the table/pool must be the SAME admitted instance, and every failure refuses
 * before mutating any page.
 *
 * Ownership contract (R24 M1/M6):
 * - `prior_out` is a distinct cell from `ref_out`; the stage never clears it.
 *   The prior publication is read at execute time and kept alive until the
 *   new publication commits; only the caller's `advanceStep()` releases it.
 * - All page frames come from the caller's `grant`, admitted BEFORE the graph
 *   runs for the complete step demand (`PooledKVDecodeBinding::prepareStep`).
 *   The stage never admits pages itself and never allocates: K/V rows are
 *   written in the native per-row interleaved layout directly from the K/V
 *   tensors at the `KVNativeStorageLayout` stride.
 *
 * The final immutable publication is a `CPUFP32PagedKVTable::Snapshot` whose
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
        /** Caller-owned next publication cell written by this stage on success. */
        PooledPublicationCell *ref_out = nullptr;
        /**
         * @brief Prior publication of the SAME request/layer, a DISTINCT cell.
         *
         * Never the same object as `ref_out`. Read at execute time; the prior
         * stays alive until the caller's `advanceStep()` after the forward.
         * Null on the first row. Every prior page is shared into the new bank
         * (immutable sharing); only the last page can be COW-extended.
         */
        const PooledPublicationCell *prior = nullptr;
        /**
         * @brief Pre-admitted step grant (all layers, complete demand).
         *
         * The stage consumes frames from this grant only; it never calls
         * `tryAdmit`. Null or an exhausted grant refuses before mutation.
         */
        PhysicalPagePool::Grant *grant = nullptr;
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
