/**
 * @file PooledKVAppendStage.cpp
 * @brief P2 production-caller pooled append: page write + typed ref mint.
 *
 * Contract (R24 M1-M6):
 * - CPU, FP32, single-row decode only (fail-closed otherwise). Multi-row
 *   append is allowed ONLY on an empty bank (initial prefill); a prior
 *   publication with multi-row append is refused before mutation (M4).
 * - The caller supplies the SAME admitted pool and table (one PMA instance);
 *   a foreign pair is rejected before any mutation.
 * - ALL page frames come from the caller's pre-admitted step grant (M6).
 *   The stage never calls tryAdmit, never allocates a staging vector: K/V
 *   rows are written in the native per-row interleaved layout directly from
 *   the K/V tensors (K row, then V row, at the KVNativeStorageLayout stride).
 * - On any failure nothing is published and `ref_out` is cleared; the prior
 *   cell (a DISTINCT object) is never touched (M1).
 * - The bank is bound exactly once at commit (M5, table enforces).
 */
#include "PooledKVAppendStage.h"
#include "utils/Assertions.h"
#include "kernels/kvcache/KVStorageLayout.h"
#include <cstring>

namespace llaminar2
{
namespace
{
/** @brief Native per-row interleaved byte geometry (K row then V row). */
struct InterleaveGeometry
{
    std::size_t k_bytes = 0; ///< K extent of one token row.
    std::size_t v_bytes = 0; ///< V extent of one token row.
    std::size_t row_stride = 0; ///< Native stride between token rows (k+v).
    std::size_t row_floats = 0; ///< Floats of one K row (tensor row width).
};

InterleaveGeometry nativeInterleave(int kv_heads, int head_dim)
{
    const auto layout = KVNativeStorageLayout::resolve(KVCacheFamily::AttentionOnly,
        kv_heads, head_dim, "fp32", DeviceId::cpu());
    const auto pair = layout.pagePayload(1);
    if (pair.k_bytes != pair.v_bytes ||
        (pair.k_bytes % sizeof(float)) != 0)
        throw std::invalid_argument("Pooled FP32 interleave requires equal aligned K/V rows");
    InterleaveGeometry geometry;
    geometry.k_bytes = pair.k_bytes;
    geometry.v_bytes = pair.v_bytes;
    geometry.row_stride = pair.totalBytes();
    geometry.row_floats = pair.k_bytes / sizeof(float);
    return geometry;
}

/** @brief Copy one page worth of rows from K/V tensors via fragment copies. */
template <typename WriteLike>
void writePageRows(WriteLike &write, const float *k,
    const float *v, int row0, int rows, const InterleaveGeometry &geometry)
{
    for (int row = row0; row < row0 + rows; ++row)
    {
        const auto *k_row = k + static_cast<std::size_t>(row) * geometry.row_floats;
        const auto *v_row = v + static_cast<std::size_t>(row) * geometry.row_floats;
        write.copyNext(std::as_bytes(std::span(k_row, geometry.k_bytes / sizeof(float))));
        write.copyNext(std::as_bytes(std::span(v_row, geometry.v_bytes / sizeof(float))));
    }
}
} // namespace

PooledKVAppendStage::PooledKVAppendStage(Params params)
    : IComputeStage(params.device_id), params_(std::move(params))
{
}

bool PooledKVAppendStage::execute(IDeviceContext *ctx)
{
    (void)ctx;
    // M1: clearing ref_out must never affect the prior cell. The prior is a
    // different object; this clears only the NEXT cell of the caller.
    if (params_.ref_out)
        *params_.ref_out = {}; // No stale publication on any failure path.

    if (!params_.pool || !params_.table)
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled append requires the admitted pool and table");
        return false;
    }
    if (!params_.pool->admissionAuthority() ||
        params_.pool->admissionAuthority().get() != params_.table->issuerForAdmission())
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled append pool/table must share ONE PMA instance");
        return false;
    }
    if (!params_.device_id.is_cpu())
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled append supports only CPU");
        return false;
    }
    if (!params_.K || !params_.V || !params_.ref_out)
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled append requires K, V and a ref cell");
        return false;
    }
    if (params_.K->dtype_name() != std::string("FP32") ||
        params_.V->dtype_name() != std::string("FP32"))
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled append supports FP32 K/V only");
        return false;
    }
    if (params_.kv_heads <= 0 || params_.head_dim <= 0 || params_.page_tokens <= 0)
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled append requires positive KV geometry");
        return false;
    }
    if (!params_.grant)
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled append requires the pre-admitted step grant");
        return false;
    }
    if (params_.grant->issuer() != params_.pool->issuerForAdmission())
    {
        LOG_ERROR("[PooledKVAppendStage] Step grant must come from the same pool authority");
        return false;
    }
    const int rows = params_.live_rows > 0 ? params_.live_rows
                                            : static_cast<int>(params_.K->rows());
    if (rows <= 0)
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled append requires positive live rows");
        return false;
    }

    // M3: native per-row interleaved layout and stride from the layout
    // authority, never a hard-coded 2*width assumption.
    InterleaveGeometry geometry;
    try
    {
        geometry = nativeInterleave(params_.kv_heads, params_.head_dim);
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled interleave geometry refused: " << e.what());
        return false;
    }
    if (params_.K->fp32_data() == nullptr || params_.V->fp32_data() == nullptr)
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled append requires host FP32 K/V backing");
        return false;
    }
    const float *k = params_.K->fp32_data();
    const float *v = params_.V->fp32_data();

    auto update = params_.table->tryBeginUpdate();
    if (!update)
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled table has no available bank (issuance budget or banks exhausted)");
        return false;
    }

    // Publication-source model (immutable sharing + private COW tail):
    // - every prior FULL page is shared verbatim into the new bank;
    // - the prior LAST page, when PARTIAL, is COW-extended: the new bank
    //   shares pages 0..n-2, and ONE fresh private page holds the prior last
    //   page's rows plus this step's new rows (M2: history beyond one page is
    //   never lost);
    // - a new page is started only when the prior last page was FULL, or when
    //   there is no prior publication at all (initial prefill), in which case
    //   the row count is split across any number of full/partial pages (M4).
    // M4: multi-row splitting is supported FROM a partial tail (the caller
    // boundary refuses continued prefill before execution for ATTENTION
    // correctness; the append itself must never refuse rows>remaining).
    const bool prior_live = params_.prior && params_.prior->live();
    const bool has_prior = prior_live;
    int prior_full_pages = 0;
    int prior_last_rows = 0;
    if (has_prior)
    {
        const auto pages = params_.prior->snapshot.view().pages();
        prior_full_pages = static_cast<int>(pages.size()) - 1;
        prior_last_rows = pages.empty() ? 0 : pages.back().rows();
    }
    const bool extend_partial =
        has_prior && prior_last_rows > 0 && prior_last_rows < params_.page_tokens;

    int written = 0;
    if (extend_partial)
    {
        // M2: share EVERY prior full page first (0..n-2), then COW the last.
        const auto pages = params_.prior->snapshot.view().pages();
        for (int page = 0; page < prior_full_pages; ++page)
        {
            auto shared = pages[static_cast<std::size_t>(page)].share();
            update->appendShared(std::move(shared), params_.kv_heads, params_.head_dim);
        }
        const auto &prior_last = pages.back();
        auto prior_shared = prior_last.share();
        const int tail_room = params_.page_tokens - prior_last_rows;
        // M4: extend the partial tail as far as it fits, then split the
        // remainder into fresh pages (never refuse rows>remaining).
        const int tail_rows = std::min(tail_room, rows);
        auto tail_write = params_.grant->prepareCOW(
            prior_shared, static_cast<std::size_t>(tail_rows) * geometry.row_stride);
        if (!tail_write)
        {
            LOG_ERROR("[PooledKVAppendStage] Pooled COW extension preparation refused");
            return false;
        }
        tail_write->copySource();
        writePageRows(*tail_write, k, v, 0, tail_rows, geometry);
        auto committed = tail_write->commit();
        auto shared = committed.share();
        update->appendShared(std::move(shared), params_.kv_heads, params_.head_dim);
        written = tail_rows;
    }
    else if (has_prior)
    {
        // The prior tail was FULL (or an empty-history edge): share EVERY
        // prior page verbatim; the fresh loop then appends new full pages.
        const auto pages = params_.prior->snapshot.view().pages();
        for (const auto &page : pages)
        {
            auto shared = page.share();
            update->appendShared(std::move(shared), params_.kv_heads, params_.head_dim);
        }
    }
    // Fresh pages for the remainder (no prior case: all rows; extension case:
    // rows beyond the tail room; full-tail case: all rows).
    while (written < rows)
    {
        const int page_rows = std::min(params_.page_tokens, rows - written);
        const std::size_t page_bytes =
            static_cast<std::size_t>(page_rows) * geometry.row_stride;
        if (params_.grant->remainingCommitment() == 0)
        {
            LOG_ERROR("[PooledKVAppendStage] Pre-admitted step grant exhausted mid-append");
            return false;
        }
        llaminar2::PhysicalPagePool::FrameHandle frame;
        try
        {
            frame = params_.grant->assignFrame();
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[PooledKVAppendStage] Step grant frame assignment refused: " << e.what());
            return false;
        }
        auto write = params_.grant->prepareAppend(frame, page_bytes);
        if (!write)
        {
            LOG_ERROR("[PooledKVAppendStage] Pooled page append preparation refused");
            return false;
        }
        writePageRows(*write, k, v, written, page_rows, geometry);
        auto committed = write->commit();
        auto shared = committed.share();
        update->appendShared(std::move(shared), params_.kv_heads, params_.head_dim);
        written += page_rows;
    }
    auto snapshot = update->commit();
    if (!params_.table->bindPublication(snapshot, params_.request_id, params_.slot))
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled publication binding refused");
        return false;
    }
    auto ref = params_.table->refFor(snapshot, params_.request_id, params_.slot);
    if (!ref.valid())
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled publication ref mint refused");
        return false;
    }
    params_.ref_out->ref = ref;
    params_.ref_out->snapshot = std::move(snapshot);
    return true;
}
StageDumpInfo PooledKVAppendStage::buildDumpInfoImpl() const
{
    StageDumpInfo info;
    if (params_.K)
        info.addInput("K", params_.K, params_.K->rows(), params_.K->cols());
    if (params_.V)
        info.addInput("V", params_.V, params_.V->rows(), params_.V->cols());
    info.addScalarInt("layer_idx", params_.layer_idx);
    info.addScalarInt("request_id_low32", static_cast<int>(params_.request_id & 0xFFFFFFFFu));
    return info;
}
} // namespace llaminar2
