#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "loaders/gpu_pipeline/DeviceLoadPipeline.h"
#include "backends/IBackend.h"
#include "utils/Logger.h"
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
                                 size_t staging_bytes)
        {
            if (!backend)
                return true;

            const size_t required_vram_bytes = planned_weight_bytes + staging_bytes;
            if (required_vram_bytes == 0)
                return true;

            const size_t free_vram_bytes = backend->deviceMemoryFree(device_id);
            const size_t total_vram_bytes = backend->deviceMemoryTotal(device_id);
            if (free_vram_bytes == 0)
                return true;

            const size_t safety_margin_bytes = std::max<size_t>(512ULL * 1024ULL * 1024ULL,
                                                                total_vram_bytes / 20ULL);
            if (required_vram_bytes + safety_margin_bytes <= free_vram_bytes)
            {
                LOG_DEBUG("LoadOrchestrator: VRAM preflight passed for device " << device_id
                                                                                << " required=" << formatMiB(required_vram_bytes)
                                                                                << " planned_weights=" << formatMiB(planned_weight_bytes)
                                                                                << " staging=" << formatMiB(staging_bytes)
                                                                                << " free=" << formatMiB(free_vram_bytes)
                                                                                << " safety_margin=" << formatMiB(safety_margin_bytes));
                return true;
            }

            LOG_ERROR("LoadOrchestrator: VRAM preflight failed for device " << device_id
                                                                            << ": required=" << formatMiB(required_vram_bytes)
                                                                            << " available_after_margin="
                                                                            << formatMiB(free_vram_bytes > safety_margin_bytes ? free_vram_bytes - safety_margin_bytes : 0)
                                                                            << " free=" << formatMiB(free_vram_bytes)
                                                                            << " total=" << formatMiB(total_vram_bytes)
                                                                            << " planned_weights=" << formatMiB(planned_weight_bytes)
                                                                            << " staging=" << formatMiB(staging_bytes)
                                                                            << " safety_margin=" << formatMiB(safety_margin_bytes)
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
            const int staging_slots = std::max(0, num_h2d_streams);
            const size_t planned_weight_bytes = ctx.pool ? ctx.pool->totalPlannedBytes() : 0;
            const size_t staging_bytes = pinned_slot_size * static_cast<size_t>(staging_slots);
            if (!vramBudgetPreflight(backend_, ctx.device_id, planned_weight_bytes, staging_bytes))
            {
                throw std::runtime_error("LoadOrchestrator: VRAM budget preflight failed for device " +
                                         std::to_string(ctx.device_id));
            }

            // Allocate VRAM pool with staging slots
            if (!ctx.pool->allocate(backend_, ctx.device_id, num_h2d_streams))
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
        ctx->pending_jobs.push_back(job);
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
                                         std::to_string(ctx.device_id));
            }

            const int num_streams = ctx.pinned_ring->numSlots();
            DeviceLoadPipeline pipeline(*backend_, ctx.device_id, *ctx.pool,
                                        *ctx.pinned_ring, kernels, num_streams);

            if (!pipeline.initialize())
            {
                throw std::runtime_error("LoadOrchestrator::load: pipeline init failed for device " +
                                         std::to_string(ctx.device_id));
            }

            LOG_DEBUG("LoadOrchestrator::load: processing " << ctx.pending_jobs.size()
                                                            << " weights on device " << ctx.device_id);

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
