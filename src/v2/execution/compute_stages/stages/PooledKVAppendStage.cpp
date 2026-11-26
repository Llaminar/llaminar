/**
 * @file PooledKVAppendStage.cpp
 * @brief P2 production-caller pooled append: page write + typed ref mint.
 *
 * Contract:
 * - ONE decode row, CPU, FP32 K/V only (fail-closed otherwise).
 * - The caller supplies the SAME admitted pool and table (one PMA instance);
 *   a foreign pair is rejected before any mutation.
 * - The stage admits one page from the pool, copies the interleaved K/V row
 *   into the page frame, appends it to the table update, commits the
 *   publication, binds the request and mints the exact typed ref into
 *   `ref_out`. On ANY failure nothing is published and `ref_out` is cleared.
 * - No allocation, no host shadow, no full-history gather.
 */
#include "PooledKVAppendStage.h"
#include "utils/Assertions.h"
#include <cstring>

namespace llaminar2
{
PooledKVAppendStage::PooledKVAppendStage(Params params)
    : IComputeStage(params.device_id), params_(std::move(params))
{
}

bool PooledKVAppendStage::execute(IDeviceContext *ctx)
{
    (void)ctx;
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
    const int rows = static_cast<int>(params_.K->rows());
    if (rows != 1)
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled append supports exactly one decode row");
        return false;
    }

    // One page holds page_tokens interleaved K/V rows. A decode step appends
    // exactly one row into the current page; when the page is full the caller
    // (graph family) opens the next page. This stage writes the SINGLE row.
    const std::size_t width = static_cast<std::size_t>(params_.kv_heads) *
        static_cast<std::size_t>(params_.head_dim);
    const std::size_t payload_bytes = width * 2 * sizeof(float); // K then V interleaved.

    auto update = params_.table->tryBeginUpdate();
    if (!update)
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled table has no available bank (issuance budget or banks exhausted)");
        return false;
    }
    // Immutable sharing: every page of the prior publication becomes a SHARED
    // page in the new bank (no bytes are copied; the pool retains one frame
    // because the shared reference keeps it alive). A request's history is
    // never regathered into contiguous scratch.
    if (params_.prior && params_.prior->live())
    {
        const auto pages = params_.prior->snapshot.view().pages();
        for (const auto &page : pages)
        {
            if (page.sharedBacked())
                update->appendShared(page.share(), page.heads(), page.dimension());
            else
            {
                // Private page from an earlier step: convert once into a shared
                // ref so the new bank and the old publication both retain it.
                auto shared = page.share();
                update->appendShared(std::move(shared), page.heads(), page.dimension());
            }
        }
    }
    auto grant = params_.pool->tryAdmit(1);
    if (!grant)
    {
        LOG_ERROR("[PooledKVAppendStage] Pool has no free page for the append");
        return false;
    }
    const auto frame = grant->assignFrame();
    auto write = grant->prepareAppend(frame, payload_bytes);
    if (!write)
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled page append preparation refused");
        return false;
    }
    const float *k = params_.K->fp32_data();
    const float *v = params_.V->fp32_data();
    std::vector<float> payload(width * 2);
    std::memcpy(payload.data(), k, width * sizeof(float));
    std::memcpy(payload.data() + width, v, width * sizeof(float));
    write->copyFrom(std::as_bytes(std::span(payload)));
    auto committed = write->commit();
    // Convert immediately to an immutable SHARED page: only shared frames
    // survive the request grant's release, so a private committed page would
    // be reassigned while the bank still references it. The shared reference
    // keeps the exact frame incarnation alive for every later step.
    auto shared = committed.share();
    update->appendShared(std::move(shared), params_.kv_heads, params_.head_dim);
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
