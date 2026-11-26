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
    const int rows = params_.live_rows > 0 ? params_.live_rows
                                            : static_cast<int>(params_.K->rows());
    if (rows <= 0)
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled append requires positive live rows");
        return false;
    }
    const std::size_t width = static_cast<std::size_t>(params_.kv_heads) *
        static_cast<std::size_t>(params_.head_dim);
    const std::size_t row_bytes = width * 2 * sizeof(float); // K then V interleaved per row.
    const float *k = params_.K->fp32_data();
    const float *v = params_.V->fp32_data();

    auto update = params_.table->tryBeginUpdate();
    if (!update)
    {
        LOG_ERROR("[PooledKVAppendStage] Pooled table has no available bank (issuance budget or banks exhausted)");
        return false;
    }

    // Publication-source model (immutable sharing + private COW tail):
    // - every page of the prior publication is kept;
    // - the prior LAST page is extended by COW copy only when it is PARTIAL
    //   (a decode step appending into the current page): the new bank shares
    //   all FULL prior pages, and one fresh private page holds the prior last
    //   page's rows + this step's new row(s);
    // - a new page is started only when the prior last page was FULL.
    const CPUFP32KVPage *prior_pages = nullptr;
    int prior_last_rows = 0;
    int prior_page_count = 0;
    if (params_.prior && params_.prior->live())
    {
        const auto pages = params_.prior->snapshot.view().pages();
        prior_page_count = static_cast<int>(pages.size());
        prior_last_rows = pages.empty() ? 0 : pages.back().rows();
        prior_pages = pages.data();
    }
    const int final_page_rows = prior_page_count > 0 ? prior_last_rows : 0;
    const bool extend_partial = prior_page_count > 0 && final_page_rows < params_.page_tokens;

    if (!extend_partial && prior_pages)
    {
        // All prior pages are full: share them verbatim.
        const auto pages = params_.prior->snapshot.view().pages();
        for (const auto &page : pages)
        {
            auto shared = page.share();
            update->appendShared(std::move(shared), page.heads(), page.dimension());
        }
    }

    if (extend_partial)
    {
        // COW the prior last page and append this step's rows after it.
        const auto &prior_last = params_.prior->snapshot.view().pages().back();
        auto prior_shared = prior_last.share();
        const std::size_t source_rows = prior_last.rows();
        const std::size_t total_rows = static_cast<std::size_t>(source_rows) + rows;
        if (total_rows > static_cast<std::size_t>(params_.page_tokens))
        {
            LOG_ERROR("[PooledKVAppendStage] Pooled decode exceeds the page capacity at this step");
            return false;
        }
        auto grant = params_.pool->tryAdmit(1);
        if (!grant)
        {
            LOG_ERROR("[PooledKVAppendStage] Pool has no free frame for the COW extension");
            return false;
        }
        auto cow = grant->prepareCOW(prior_shared, static_cast<std::size_t>(rows) * row_bytes);
        if (!cow)
        {
            LOG_ERROR("[PooledKVAppendStage] Pooled COW extension preparation refused");
            return false;
        }
        cow->copySource();
        std::vector<float> fresh(static_cast<std::size_t>(rows) * width * 2);
        std::memcpy(fresh.data(), k, static_cast<std::size_t>(rows) * width * sizeof(float));
        std::memcpy(fresh.data() + static_cast<std::size_t>(rows) * width, v,
            static_cast<std::size_t>(rows) * width * sizeof(float));
        cow->copyNext(std::as_bytes(std::span(fresh)));
        auto committed = cow->commit();
        auto shared = committed.share();
        update->appendShared(std::move(shared), params_.kv_heads, params_.head_dim);
    }
    else
    {
        // Fresh page(s): prefill of an empty bank, or a decode step starting a
        // new full page. Every page except the last is written full; the last
        // may be partial.
        int written = 0;
        while (written < rows)
        {
            const int page_rows = std::min(params_.page_tokens, rows - written);
            const std::size_t page_bytes = static_cast<std::size_t>(page_rows) * row_bytes;
            auto grant = params_.pool->tryAdmit(1);
            if (!grant)
            {
                LOG_ERROR("[PooledKVAppendStage] Pool has no free page for the append");
                return false;
            }
            const auto frame = grant->assignFrame();
            auto write = grant->prepareAppend(frame, page_bytes);
            if (!write)
            {
                LOG_ERROR("[PooledKVAppendStage] Pooled page append preparation refused");
                return false;
            }
            std::vector<float> payload(static_cast<std::size_t>(page_rows) * width * 2);
            std::memcpy(payload.data(), k + static_cast<std::size_t>(written) * width,
                static_cast<std::size_t>(page_rows) * width * sizeof(float));
            std::memcpy(payload.data() + static_cast<std::size_t>(page_rows) * width,
                v + static_cast<std::size_t>(written) * width,
                static_cast<std::size_t>(page_rows) * width * sizeof(float));
            write->copyFrom(std::as_bytes(std::span(payload)));
            auto committed = write->commit();
            auto shared = committed.share();
            update->appendShared(std::move(shared), params_.kv_heads, params_.head_dim);
            written += page_rows;
        }
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
