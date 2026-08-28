#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "loaders/gpu_pipeline/DeviceLoadPipeline.h"
#include "loaders/GPUVramPreflight.h"
#include "backends/IBackend.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include "utils/VramBillOfMaterials.h"
#include "utils/WeightLoadingProfiler.h"

/**
 * @file LoadOrchestrator.cpp
 * @brief Implementation of the GPU weight loading orchestration lifecycle.
 *
 * The orchestrator separates model-weight lifetime from temporary upload staging:
 * registered GEMM kernels keep this object alive for persistent pool pointers, while
 * finalize() drops staging VRAM and pinned host rings once the pipeline has drained.
 */

// Backend-specific kernel headers (linked conditionally)
#ifdef HAVE_ROCM
#include "kernels/rocm/repack/VnniRepackKernels.h"
#endif

#ifdef HAVE_CUDA
#include "kernels/cuda/repack/CUDAVnniRepackKernels.h"
#endif

#include <algorithm>
#include <stdexcept>

namespace llaminar2
{

    namespace
    {
        std::string formatMiB(size_t bytes)
        {
            return std::to_string(bytes / (1024 * 1024)) + " MiB";
        }

        bool vramBudgetPreflight(IBackend *backend,
                                 int device_id,
                                 size_t planned_weight_bytes,
                                 size_t staging_slot_bytes,
                                 int staging_stream_count)
        {
            if (!backend)
                return true;

            if (planned_weight_bytes == 0 && staging_slot_bytes == 0)
                return true;

            const size_t free_vram_bytes = backend->deviceMemoryFree(device_id);
            auto policy = configuredGPUWeightLoadMemoryPolicy();
            /*
             * allocate() already receives the exact slot selected by the
             * caller's load BOM. Unlimited staging here means "price this
             * exact slot"; applying the configured cap a second time could
             * silently describe a different allocation.
             */
            policy.staging_stream_count = staging_stream_count;
            policy.staging_budget_bytes = 0;
            const auto load_bom = gpuWeightLoadMemoryBOM(
                planned_weight_bytes,
                staging_slot_bytes,
                free_vram_bytes,
                policy);
            const size_t required_vram_bytes = load_bom.load_bytes;
            const size_t staging_bytes = load_bom.staging_bytes;
            if (load_bom.fits())
            {
                logVramBomLine(
                    "weight_preflight",
                    "source=LoadOrchestrator status=pass backend=" + backend->backendName() +
                        " device_id=" + std::to_string(device_id) +
                        " required_bytes=" + std::to_string(required_vram_bytes) +
                        " required_mib=" + vramBomMiB(required_vram_bytes) +
                        " planned_weights_bytes=" + std::to_string(planned_weight_bytes) +
                        " planned_weights_mib=" + vramBomMiB(planned_weight_bytes) +
                        " staging_bytes=" + std::to_string(staging_bytes) +
                        " staging_mib=" + vramBomMiB(staging_bytes) +
                        " free_bytes=" + std::to_string(free_vram_bytes) +
                        " free_mib=" + vramBomMiB(free_vram_bytes));
                LOG_DEBUG("LoadOrchestrator: VRAM preflight passed for device " << device_id
                                                                                << " required=" << formatMiB(required_vram_bytes)
                                                                                << " planned_weights=" << formatMiB(planned_weight_bytes)
                                                                                << " staging=" << formatMiB(staging_bytes)
                                                                                << " free=" << formatMiB(free_vram_bytes));
                return true;
            }

            logVramBomLine(
                "weight_preflight",
                "source=LoadOrchestrator status=fail backend=" + backend->backendName() +
                    " device_id=" + std::to_string(device_id) +
                    " required_bytes=" + std::to_string(required_vram_bytes) +
                    " required_mib=" + vramBomMiB(required_vram_bytes) +
                    " planned_weights_bytes=" + std::to_string(planned_weight_bytes) +
                    " planned_weights_mib=" + vramBomMiB(planned_weight_bytes) +
                    " staging_bytes=" + std::to_string(staging_bytes) +
                    " staging_mib=" + vramBomMiB(staging_bytes) +
                    " free_bytes=" + std::to_string(free_vram_bytes) +
                    " free_mib=" + vramBomMiB(free_vram_bytes));
            LOG_ERROR("LoadOrchestrator: VRAM preflight failed for device " << device_id
                                                                            << ": required=" << formatMiB(required_vram_bytes)
                                                                            << " free=" << formatMiB(free_vram_bytes)
                                                                            << " planned_weights=" << formatMiB(planned_weight_bytes)
                                                                            << " staging=" << formatMiB(staging_bytes)
                                                                            << ". Mitigations: set LLAMINAR_WEIGHT_STREAMING=1, use a smaller model, "
                                                                            << "reduce context/KV cache pressure, or reduce resident experts.");
            return false;
        }
    } // namespace

    LoadOrchestrator::LoadOrchestrator(IBackend *backend)
        : backend_(backend)
    {
    }

    LoadOrchestrator::~LoadOrchestrator() { release(); }

    void LoadOrchestrator::addDevice(int device_id)
    {
        if (findDevice(device_id))
        {
            LOG_ERROR("LoadOrchestrator: device " << device_id << " already added");
            throw std::runtime_error("LoadOrchestrator: duplicate device id " +
                                     std::to_string(device_id));
        }

        DeviceContext ctx;
        ctx.device_id = device_id;
        ctx.pool = std::make_unique<WeightVRAMPool>();
        devices_.push_back(std::move(ctx));

        LOG_DEBUG("LoadOrchestrator: added device " << device_id);
    }

    void LoadOrchestrator::planWeight(int device_id, const std::string &name,
                                      int N, int K, int payload_bytes_per_block,
                                      bool is_asymmetric, bool has_emins,
                                      size_t raw_gguf_bytes)
    {
        auto *ctx = findDevice(device_id);
        if (!ctx)
        {
            LOG_ERROR("LoadOrchestrator: unknown device " << device_id);
            throw std::runtime_error("LoadOrchestrator: unknown device " +
                                     std::to_string(device_id));
        }

        ctx->pool->planWeight(name, N, K, payload_bytes_per_block, is_asymmetric,
                              has_emins, raw_gguf_bytes);
    }

    void LoadOrchestrator::planRawWeight(int device_id, const std::string &name,
                                         int N, int K, size_t raw_bytes)
    {
        auto *ctx = findDevice(device_id);
        if (!ctx)
        {
            LOG_ERROR("LoadOrchestrator: unknown device " << device_id);
            throw std::runtime_error("LoadOrchestrator: unknown device " +
                                     std::to_string(device_id));
        }

        ctx->pool->planRawWeight(name, N, K, raw_bytes);
    }

    void LoadOrchestrator::allocate(size_t pinned_slot_size, int num_h2d_streams)
    {
        ScopedWeightLoadDetailTimer alloc_timer("gpu_pipeline.allocate");

        for (auto &ctx : devices_)
        {
            /**
             * A non-zero pinned slot means the later H2D pipeline will need at
             * least one upload stream and a pinned-ring slot. Silently allowing
             * zero streams here creates a half-allocated orchestrator: the pool
             * exists, but load() cannot stage any raw bytes. Fail at allocation
             * time so tests and callers see the real contract violation.
             */
            if (pinned_slot_size > 0 && num_h2d_streams <= 0)
            {
                throw std::runtime_error("LoadOrchestrator: pinned staging requested with no H2D streams for device " +
                                         std::to_string(ctx.device_id));
            }

            const int staging_slots = std::max(0, num_h2d_streams);
            const size_t planned_weight_bytes = ctx.pool ? ctx.pool->totalPlannedBytes() : 0;
            if (!vramBudgetPreflight(
                    backend_,
                    ctx.device_id,
                    planned_weight_bytes,
                    pinned_slot_size,
                    staging_slots))
            {
                throw std::runtime_error("LoadOrchestrator: VRAM budget preflight failed for device " +
                                         std::to_string(ctx.device_id));
            }

            // Allocate VRAM pool with staging slots
            if (!ctx.pool->allocate(backend_, ctx.device_id, num_h2d_streams,
                                    pinned_slot_size))
            {
                throw std::runtime_error("LoadOrchestrator: failed to allocate pool for device " +
                                         std::to_string(ctx.device_id));
            }

            // Allocate pinned ring buffer
            if (pinned_slot_size > 0 && num_h2d_streams > 0)
            {
                ctx.pinned_ring = std::make_unique<PinnedRingBuffer>(pinned_slot_size, num_h2d_streams);
                if (!ctx.pinned_ring->allocate(backend_, ctx.device_id))
                {
                    throw std::runtime_error("LoadOrchestrator: failed to allocate pinned ring for device " +
                                             std::to_string(ctx.device_id));
                }
            }
        }

        LOG_DEBUG("LoadOrchestrator: allocated " << devices_.size() << " device(s)");
    }

    WeightVRAMPool *LoadOrchestrator::getPool(int device_id)
    {
        auto *ctx = findDevice(device_id);
        return ctx ? ctx->pool.get() : nullptr;
    }

    const WeightVRAMPool *LoadOrchestrator::getPool(int device_id) const
    {
        auto *ctx = findDevice(device_id);
        return ctx ? ctx->pool.get() : nullptr;
    }

    size_t LoadOrchestrator::numDevices() const { return devices_.size(); }

    void LoadOrchestrator::addWeightJob(int device_id, const WeightJob &job)
    {
        auto *ctx = findDevice(device_id);
        if (!ctx)
        {
            LOG_ERROR("LoadOrchestrator::addWeightJob: unknown device " << device_id);
            throw std::runtime_error("LoadOrchestrator: unknown device " +
                                     std::to_string(device_id));
        }
        if (!ctx->pinned_ring || !ctx->pinned_ring->isAllocated())
        {
            throw std::runtime_error(
                "LoadOrchestrator::addWeightJob requires allocated pinned staging for device " +
                std::to_string(device_id));
        }

        /*
         * Both mapped-file pread and heap-backed memcpy write the exact payload
         * into the persistent pinned slot. There is no alignment padding and no
         * second staging allocation, so the complete configured slot remains
         * available for row-bounded chunks.
         */
        const size_t usable_slot_bytes =
            ctx->pinned_ring->slotSize();
        if (usable_slot_bytes == 0)
        {
            throw std::runtime_error(
                "LoadOrchestrator: staging slot has no usable payload space for weight '" +
                job.name + "'");
        }

        if (job.raw_bytes <= usable_slot_bytes)
        {
            WeightJob whole = job;
            whole.full_N = whole.full_N > 0 ? whole.full_N : whole.N;
            whole.full_K = whole.full_K > 0 ? whole.full_K : whole.K;
            ctx->pending_jobs.push_back(std::move(whole));
            return;
        }

        if (job.N <= 0 || job.raw_bytes % static_cast<size_t>(job.N) != 0)
        {
            throw std::runtime_error(
                "LoadOrchestrator: oversized weight '" + job.name +
                "' cannot be row-chunked because raw bytes are not divisible by N");
        }

        if (!job.host_raw_data)
        {
            throw std::runtime_error(
                "LoadOrchestrator: oversized weight '" + job.name +
                "' has a null host source");
        }

        const auto *source = static_cast<const uint8_t *>(job.host_raw_data);
        const size_t bytes_per_row = job.raw_bytes / static_cast<size_t>(job.N);
        const size_t rows_per_chunk = usable_slot_bytes / bytes_per_row;
        if (rows_per_chunk == 0)
        {
            throw std::runtime_error(
                "LoadOrchestrator: staging slot is smaller than one raw row for weight '" +
                job.name + "'");
        }

        size_t chunk_count = 0;
        for (int row = 0; row < job.N;)
        {
            const int chunk_rows = static_cast<int>(std::min(
                rows_per_chunk, static_cast<size_t>(job.N - row)));
            WeightJob chunk = job;
            chunk.host_raw_data = source + static_cast<size_t>(row) * bytes_per_row;
            chunk.raw_bytes = static_cast<size_t>(chunk_rows) * bytes_per_row;
            chunk.row_offset = job.row_offset + row;
            chunk.full_N = job.full_N > 0 ? job.full_N : job.N;
            chunk.full_K = job.full_K > 0 ? job.full_K : job.K;
            chunk.N = chunk_rows;
            ctx->pending_jobs.push_back(std::move(chunk));
            ++chunk_count;
            row += chunk_rows;
        }

        LOG_DEBUG("LoadOrchestrator: split oversized weight '" << job.name
                                                                << "' raw_bytes=" << job.raw_bytes
                                                                << " into " << chunk_count
                                                                << " row-chunk jobs"
                                                                << " (slot_bytes="
                                                                << usable_slot_bytes
                                                                << ")");
    }

    RepackKernels LoadOrchestrator::createRepackKernels() const
    {
        if (!backend_)
        {
            throw std::runtime_error("LoadOrchestrator: no backend set, cannot create repack kernels");
        }

        RepackKernels kernels{};
        const auto name = backend_->backendName();

#ifdef HAVE_CUDA
        if (name == "CUDA")
        {
            kernels.vnniRepack = launchVnniRepackCUDA;
            return kernels;
        }
#endif

#ifdef HAVE_ROCM
        if (name == "ROCm")
        {
            kernels.vnniRepack = launchVnniRepack;
            return kernels;
        }
#endif

        throw std::runtime_error("LoadOrchestrator: unsupported backend: " + name);
    }

    size_t LoadOrchestrator::totalPendingBytes(int device_id) const
    {
        const auto *ctx = findDevice(device_id);
        if (!ctx)
            return 0;
        size_t total = 0;
        for (const auto &job : ctx->pending_jobs)
            total += job.raw_bytes;
        return total;
    }

    size_t LoadOrchestrator::pendingJobCount(int device_id) const
    {
        const auto *ctx = findDevice(device_id);
        return ctx ? ctx->pending_jobs.size() : 0;
    }

    void LoadOrchestrator::load(DeviceLoadPipeline::ProgressCallback progress_cb)
    {
        if (devices_.empty())
        {
            return; // Nothing to do
        }

        if (!backend_)
        {
            throw std::runtime_error("LoadOrchestrator::load: no backend set");
        }

        ScopedWeightLoadDetailTimer total_timer("gpu_pipeline.total");

        const auto kernels = createRepackKernels();

        for (auto &ctx : devices_)
        {
            if (ctx.pending_jobs.empty())
            {
                LOG_DEBUG("LoadOrchestrator::load: no jobs for device " << ctx.device_id);
                continue;
            }

            if (!ctx.pool || !ctx.pool->isAllocated())
            {
                throw std::runtime_error("LoadOrchestrator::load: pool not allocated for device " +
                                         std::to_string(ctx.device_id));
            }

            if (!ctx.pinned_ring || !ctx.pinned_ring->isAllocated())
            {
                throw std::runtime_error("LoadOrchestrator::load: pinned ring not allocated for device " +
                                         std::to_string(ctx.device_id) +
                                         " pending_jobs=" + std::to_string(ctx.pending_jobs.size()) +
                                         " pool_allocated=" + std::string((ctx.pool && ctx.pool->isAllocated()) ? "true" : "false"));
            }

            const int num_streams = ctx.pinned_ring->numSlots();
            const size_t source_backward_jumps =
                orderWeightJobsForSequentialHostAccess(ctx.pending_jobs);
            if (PerfStatsCollector::isEnabled())
            {
                PerfStatsCollector::addCounter(
                    "weight_loading",
                    "gpu_pipeline_source_backward_jumps",
                    static_cast<double>(source_backward_jumps),
                    "load",
                    "gpu:" + std::to_string(ctx.device_id));
            }

            DeviceLoadPipeline pipeline(*backend_, ctx.device_id, *ctx.pool,
                                        *ctx.pinned_ring, kernels, num_streams);

            if (!pipeline.initialize())
            {
                throw std::runtime_error("LoadOrchestrator::load: pipeline init failed for device " +
                                         std::to_string(ctx.device_id));
            }

            LOG_DEBUG("LoadOrchestrator::load: processing " << ctx.pending_jobs.size()
                                                            << " source-ordered weights on device "
                                                            << ctx.device_id
                                                            << " (removed " << source_backward_jumps
                                                            << " backward mmap transitions)");

            if (!pipeline.processJobs(ctx.pending_jobs, progress_cb))
            {
                throw std::runtime_error("LoadOrchestrator::load: pipeline failed for device " +
                                         std::to_string(ctx.device_id));
            }

            ctx.pending_jobs.clear();
        }
    }

    void LoadOrchestrator::finalize()
    {
        for (auto &ctx : devices_)
        {
            if (ctx.pool)
                ctx.pool->releaseStaging();
            if (ctx.pinned_ring)
                ctx.pinned_ring->release();
        }
    }

    void LoadOrchestrator::release()
    {
        for (auto &ctx : devices_)
        {
            if (ctx.pinned_ring)
                ctx.pinned_ring->release();
            if (ctx.pool)
                ctx.pool->release();
        }
        devices_.clear();
    }

    LoadOrchestrator::DeviceContext *LoadOrchestrator::findDevice(int device_id)
    {
        for (auto &ctx : devices_)
        {
            if (ctx.device_id == device_id)
                return &ctx;
        }
        return nullptr;
    }

    const LoadOrchestrator::DeviceContext *LoadOrchestrator::findDevice(int device_id) const
    {
        for (const auto &ctx : devices_)
        {
            if (ctx.device_id == device_id)
                return &ctx;
        }
        return nullptr;
    }

} // namespace llaminar2
