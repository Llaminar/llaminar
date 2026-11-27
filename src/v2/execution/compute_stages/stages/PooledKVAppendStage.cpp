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
#include "PooledKVDecodeBinding.h"  // LIVE tail plan of the binding (single authority)
#include "utils/Assertions.h"
#include "utils/PerfStatsCollector.h"
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

    // Resolve the LIVE tail plan at execute time. CPU forward graphs are
    // rebuilt per forward but the builder materializes stage parameters at
    // graph-build time; the stage params then point at binding state whose
    // VALUES change between steps (mode, rolled frames, row counts). The
    // binding is the single authority: read it NOW, never replay a baked
    // bool from a previous step's build.
    bool exclusive_tail = params_.exclusive_tail;
    bool tail_finishes_last_page = params_.tail_finishes_last_page;
    PhysicalPagePool::Grant *tail_grant = params_.tail_grant;
    const PhysicalPagePool::FrameHandle *tail_frame = params_.tail_frame;
    const PhysicalPagePool::FrameHandle *fresh_tail_frame = params_.fresh_tail_frame;
    const int live_rows = params_.live_rows > 0 ? params_.live_rows
                                                : static_cast<int>(params_.K->rows());
    const int rows = live_rows;
    if (rows <= 0)
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled append requires positive live rows");
        return false;
    }
    if (params_.kv_heads <= 0 || params_.head_dim <= 0 || params_.page_tokens <= 0)
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled append requires positive KV geometry");
        return false;
    }
    if (params_.binding && params_.kv_heads > 0 && params_.head_dim > 0)
    {
        const auto mode = params_.binding->stepTailMode(params_.layer_idx, live_rows);
        exclusive_tail = mode != PooledKVDecodeBinding::StepTailMode::None;
        tail_finishes_last_page = params_.binding->stepFinishesOnTailFrame(
            params_.layer_idx, live_rows);
        tail_grant = params_.binding->tailGrant();
        tail_frame = params_.binding->tailFrame(params_.layer_idx);
        fresh_tail_frame = params_.binding->freshTailFrame(params_.layer_idx);
    }

    // The exclusive-tail path needs the persistent tail grant; a tail-only
    // step admits no step grant. Both cannot be absent.
    if (exclusive_tail && (!tail_grant || !tail_frame))
    {
        LOG_ERROR("[PooledKVAppendStage] Exclusive tail step requires the persistent tail grant/frame");
        return false;
    }
    // A non-exclusive step ALWAYS requires the pre-admitted step grant.
    if (!params_.grant)
    {
        if (!exclusive_tail)
        {
            LOG_ERROR("[PooledKVAppendStage] Pooled append requires the pre-admitted step grant");
            return false;
        }
    }
    else if (params_.grant->issuer() != params_.pool->issuerForAdmission())
    {
        LOG_ERROR("[PooledKVAppendStage] Step grant must come from the same pool authority");
        return false;
    }
    // N2: the request's accepted-work envelope owns the version. Each layer
    // publication consumes one reserved issuance; normal-admission fallback
    // exists only for direct stage tests.
    std::optional<CPUFP32PagedKVTable::Update> update;
    if (params_.issuance)
        update = params_.table->tryBeginReserved(*params_.issuance);
    else
        update = params_.table->tryBeginUpdate();
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
    // EXCLUSIVE TAIL (P2 economy, design line 43 "every writer owns an
    // exclusive tail or COW destination"): when the prior last page is the
    // request's OWN current private publication of its PERSISTENT tail frame,
    // the step extends THAT frame in place — no copy of the prior tail rows is
    // performed. The prior bank page (in the previous bank's snapshot) is then
    // pin-relinquished so the frame is not double-unpinned when that bank is
    // dropped. When the tail becomes full (or the step starts fresh) and ends
    // with a partial page, the final partial page is written into a FRESH tail
    // frame (rolled at prepareStep); its rows are not copied either.
    // M4: multi-row splitting is supported FROM a partial tail (the caller
    // boundary refuses continued prefill before execution for ATTENTION
    // correctness; the append itself must never refuse rows>remaining).
    const bool prior_live = params_.prior && params_.prior->live();
    const bool has_prior = prior_live;
    int prior_full_pages = 0;
    int prior_last_rows = 0;
    const CPUFP32KVPage *prior_last = nullptr;
    if (has_prior)
    {
        const auto &pages = params_.prior->snapshot.view().pages();
        prior_full_pages = static_cast<int>(pages.size()) - 1;
        prior_last_rows = pages.empty() ? 0 : pages.back().rows();
        prior_last = pages.empty() ? nullptr : &pages.back();
    }
    const bool extend_partial =
        has_prior && prior_last_rows > 0 && prior_last_rows < params_.page_tokens;
    const bool in_place =
        exclusive_tail && prior_last && prior_last->privatePublication() &&
        extend_partial;

    int written = 0;
    if (in_place)
    {
        // M2: share EVERY prior full page first (0..n-2), then extend the
        // request's own tail frame in place. The old prior page is pin
        // relinquished AFTER the commit so this bank never double-unpins.
        const auto &pages = params_.prior->snapshot.view().pages();
        for (int page = 0; page < prior_full_pages; ++page)
        {
            auto shared = pages[static_cast<std::size_t>(page)].share();
            update->appendShared(std::move(shared), params_.kv_heads, params_.head_dim);
        }
        const int tail_room = params_.page_tokens - prior_last_rows;
        // M4: extend the partial tail as far as it fits, then split the
        // remainder into fresh pages (never refuse rows>remaining).
        const int tail_rows = std::min(tail_room, rows);
        auto tail_write = tail_grant->prepareExclusiveTail(
            *tail_frame, *prior_last->privatePublication(),
            static_cast<std::size_t>(tail_rows) * geometry.row_stride);
        if (!tail_write)
        {
            LOG_ERROR("[PooledKVAppendStage] Pooled in-place tail extension refused");
            return false;
        }
        writePageRows(*tail_write, k, v, 0, tail_rows, geometry);
        // P2 economy: recorded in-place extension copies ONLY the fresh rows
        // (never the prior tail bytes — the whole point of the private tail).
        PerfStatsCollector::addCounter(
            "pooled_kv_decode", "in_place_tail_bytes_per_step",
            static_cast<double>(tail_rows) * geometry.row_stride,
            "decode", params_.device_id.toString(),
            {{"layer", std::to_string(params_.layer_idx)},
             {"rows", std::to_string(rows)}});
        PerfStatsCollector::addCounter(
            "pooled_kv_decode", "in_place_tail_bytes_per_token",
            (static_cast<double>(tail_rows) * geometry.row_stride) /
                static_cast<double>(rows),
            "decode", params_.device_id.toString(),
            {{"layer", std::to_string(params_.layer_idx)}});
        auto committed = tail_write->commit();
        // Two-pin model: the prior publication keeps its own pin; the new
        // publication holds the plan's pin. When the prior bank snapshot is
        // dropped at advanceStep, its unpin leaves exactly one pin — the new
        // publication's — so no double-unpin exists and no relinquish is
        // needed. The prior page (same frame) only ever reads its ORIGINAL
        // cached rows (a strict prefix), so old snapshots remain valid.
        update->append(CPUFP32KVPage(std::move(committed), params_.kv_heads, params_.head_dim));
        written = tail_rows;
    }
    else if (extend_partial)
    {
        // COW path (shared/foreign tail): copy prior last page + new rows.
        const auto &pages = params_.prior->snapshot.view().pages();
        for (int page = 0; page < prior_full_pages; ++page)
        {
            auto shared = pages[static_cast<std::size_t>(page)].share();
            update->appendShared(std::move(shared), params_.kv_heads, params_.head_dim);
        }
        auto prior_shared = pages.back().share();
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
        // R25 economy: the private COW tail copies the prior last page bytes
        // plus the fresh rows. Record the exact bytes copied per token so the
        // copy cost is measurable (maxim 21/26) instead of silent.
        PerfStatsCollector::addCounter(
            "pooled_kv_decode", "cow_tail_bytes_per_step",
            static_cast<double>(pages.back().storage().size()) +
                static_cast<double>(tail_rows) * geometry.row_stride,
            "decode", params_.device_id.toString(),
            {{"layer", std::to_string(params_.layer_idx)},
             {"rows", std::to_string(rows)}});
        PerfStatsCollector::addCounter(
            "pooled_kv_decode", "cow_tail_bytes_per_token",
            (static_cast<double>(pages.back().storage().size()) +
                static_cast<double>(tail_rows) * geometry.row_stride) /
                static_cast<double>(rows),
            "decode", params_.device_id.toString(),
            {{"layer", std::to_string(params_.layer_idx)}});
        auto committed = tail_write->commit();
        auto shared = committed.share();
        update->appendShared(std::move(shared), params_.kv_heads, params_.head_dim);
        written = tail_rows;
    }
    else if (has_prior)
    {
        // The prior tail was FULL (or an empty-history edge): share EVERY
        // prior page verbatim; the fresh loop then appends new full pages.
        const auto &pages = params_.prior->snapshot.view().pages();
        for (const auto &page : pages)
        {
            auto shared = page.share();
            update->appendShared(std::move(shared), params_.kv_heads, params_.head_dim);
        }
    }
    // Fresh pages for the remainder. With an exclusive tail, the step's LAST
    // partial page is written into the (rolled) TAIL frame — never shared
    // history — so the next step can extend it in place again.
    while (written < rows)
    {
        const bool final_partial =
            tail_finishes_last_page &&
            (rows - written) < params_.page_tokens;
        const int page_rows = final_partial
            ? (rows - written)
            : std::min(params_.page_tokens, rows - written);
        const std::size_t page_bytes =
            static_cast<std::size_t>(page_rows) * geometry.row_stride;
        if (final_partial)
        {
            const auto *destination = fresh_tail_frame
                ? fresh_tail_frame : tail_frame;
            auto tail_write = tail_grant->prepareFreshWrite(*destination, page_bytes);
            if (!tail_write)
            {
                LOG_ERROR("[PooledKVAppendStage] Pooled fresh tail frame preparation refused");
                return false;
            }
            writePageRows(*tail_write, k, v, written, page_rows, geometry);
            PerfStatsCollector::addCounter(
                "pooled_kv_decode", "in_place_tail_bytes_per_step",
                static_cast<double>(page_bytes),
                "decode", params_.device_id.toString(),
                {{"layer", std::to_string(params_.layer_idx)},
                 {"rows", std::to_string(rows)}});
            auto committed = tail_write->commit();
            update->append(CPUFP32KVPage(std::move(committed), params_.kv_heads, params_.head_dim));
            written += page_rows;
            continue;
        }
        if (!params_.grant || params_.grant->remainingCommitment() == 0)
        {
            LOG_ERROR("[PooledKVAppendStage] Pre-admitted step grant exhausted mid-append"
                << " (written=" << written << " rows=" << rows
                << " exclusive=" << exclusive_tail
                << " tail_last=" << tail_finishes_last_page
                << " prior_rows=" << prior_last_rows
                << " has_prior=" << has_prior << ")");
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
