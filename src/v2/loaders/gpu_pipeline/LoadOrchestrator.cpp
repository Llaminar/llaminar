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
#include <limits>
#include <stdexcept>

namespace llaminar2
{

    LoadOrchestrator::LoadOrchestrator(
        IBackend *backend,
        std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
        PhysicalMemoryOwner persistent_owner)
        : backend_(backend),
          memory_authority_(std::move(memory_authority)),
          persistent_owner_(persistent_owner),
          allocation_authority_kind_(
              AllocationAuthorityKind::Production)
    {
        if (!backend_ || !memory_authority_ ||
            persistent_owner_ == PhysicalMemoryOwner::Count ||
            persistent_owner_ == PhysicalMemoryOwner::WeightLoadStaging)
        {
            throw std::invalid_argument(
                "LoadOrchestrator production construction requires a backend, physical-memory authority, and persistent owner");
        }
    }

    LoadOrchestrator::LoadOrchestrator(
        IBackend *backend,
        TestOnlyUnadmittedGPUAllocation test_only)
        : backend_(backend),
          allocation_authority_kind_(
              AllocationAuthorityKind::ExplicitTest)
    {
        (void)test_only;
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
        planWeightForOwner(
            device_id,
            name,
            N,
            K,
            payload_bytes_per_block,
            is_asymmetric,
            has_emins,
            raw_gguf_bytes,
            persistent_owner_);
    }

    void LoadOrchestrator::planWeightForOwner(
        int device_id,
        const std::string &name,
        int N,
        int K,
        int payload_bytes_per_block,
        bool is_asymmetric,
        bool has_emins,
        size_t raw_gguf_bytes,
        PhysicalMemoryOwner owner)
    {
        auto *ctx = findDevice(device_id);
        if (!ctx)
        {
            LOG_ERROR("LoadOrchestrator: unknown device " << device_id);
            throw std::runtime_error("LoadOrchestrator: unknown device " +
                                     std::to_string(device_id));
        }

        const size_t bytes_before = ctx->pool->totalPlannedBytes();
        ctx->pool->planWeight(name, N, K, payload_bytes_per_block, is_asymmetric,
                              has_emins, raw_gguf_bytes);
        recordPersistentOwnerGrowth(
            *ctx,
            owner,
            bytes_before,
            ctx->pool->totalPlannedBytes());
    }

    void LoadOrchestrator::planRawWeight(int device_id, const std::string &name,
                                         int N, int K, size_t raw_bytes)
    {
        planRawWeightForOwner(
            device_id,
            name,
            N,
            K,
            raw_bytes,
            persistent_owner_);
    }

    void LoadOrchestrator::planRawWeightForOwner(
        int device_id,
        const std::string &name,
        int N,
        int K,
        size_t raw_bytes,
        PhysicalMemoryOwner owner)
    {
        auto *ctx = findDevice(device_id);
        if (!ctx)
        {
            LOG_ERROR("LoadOrchestrator: unknown device " << device_id);
            throw std::runtime_error("LoadOrchestrator: unknown device " +
                                     std::to_string(device_id));
        }

        const size_t bytes_before = ctx->pool->totalPlannedBytes();
        ctx->pool->planRawWeight(name, N, K, raw_bytes);
        recordPersistentOwnerGrowth(
            *ctx,
            owner,
            bytes_before,
            ctx->pool->totalPlannedBytes());
    }

    void LoadOrchestrator::recordPersistentOwnerGrowth(
        DeviceContext &ctx,
        PhysicalMemoryOwner owner,
        size_t bytes_before,
        size_t bytes_after)
    {
        if (bytes_after < bytes_before)
        {
            throw std::logic_error(
                "LoadOrchestrator persistent plan moved backwards");
        }
        ctx.persistent_planned_bytes = bytes_after;
        if (allocation_authority_kind_ == AllocationAuthorityKind::Production &&
            (owner == PhysicalMemoryOwner::Count ||
             owner == PhysicalMemoryOwner::WeightLoadStaging))
        {
            throw std::invalid_argument(
                "LoadOrchestrator persistent GPU bytes require a non-staging physical-memory owner");
        }
        if (owner == PhysicalMemoryOwner::Count)
        {
            /* Explicit allocator tests deliberately carry no production BOM. */
            return;
        }
        const size_t index = static_cast<size_t>(owner);
        if (index >= ctx.persistent_owner_bytes.size())
            throw std::invalid_argument("LoadOrchestrator received an invalid owner");
        const size_t growth = bytes_after - bytes_before;
        if (growth > std::numeric_limits<size_t>::max() -
                         ctx.persistent_owner_bytes[index])
        {
            throw std::overflow_error(
                "LoadOrchestrator persistent owner byte count overflowed");
        }
        ctx.persistent_owner_bytes[index] += growth;
        ctx.last_persistent_owner = owner;
    }

    size_t LoadOrchestrator::plannedPersistentBytes(
        int device_id,
        PhysicalMemoryOwner owner) const
    {
        const auto *ctx = findDevice(device_id);
        if (!ctx || owner == PhysicalMemoryOwner::Count)
            return 0u;
        const size_t index = static_cast<size_t>(owner);
        if (index >= ctx->persistent_owner_bytes.size())
            throw std::invalid_argument(
                "LoadOrchestrator received an invalid persistent owner");
        return resolvedPersistentOwnerBytes(*ctx)[index];
    }

    std::array<std::size_t, PhysicalMemoryBOM::ownerCount()>
    LoadOrchestrator::resolvedPersistentOwnerBytes(
        const DeviceContext &ctx) const
    {
        auto owner_bytes = ctx.persistent_owner_bytes;
        const size_t planned_weight_bytes = ctx.persistent_planned_bytes;
        const size_t persistent_bytes =
            alignGPUWeightLoadAllocation(planned_weight_bytes);
        if (persistent_bytes > planned_weight_bytes)
        {
            if (ctx.last_persistent_owner == PhysicalMemoryOwner::Count)
            {
                if (allocation_authority_kind_ ==
                    AllocationAuthorityKind::Production)
                {
                    throw std::logic_error(
                        "LoadOrchestrator has persistent alignment bytes but no owning weight");
                }
                return owner_bytes;
            }

            const size_t owner_index = static_cast<size_t>(
                ctx.last_persistent_owner);
            if (owner_index >= owner_bytes.size())
            {
                throw std::logic_error(
                    "LoadOrchestrator final alignment owner is invalid");
            }
            const size_t tail = persistent_bytes - planned_weight_bytes;
            if (tail > std::numeric_limits<size_t>::max() -
                           owner_bytes[owner_index])
            {
                throw std::overflow_error(
                    "LoadOrchestrator final owner-alignment charge overflowed");
            }
            owner_bytes[owner_index] += tail;
        }
        return owner_bytes;
    }

    void LoadOrchestrator::allocate(size_t pinned_slot_size, int num_h2d_streams)
    {
        ScopedWeightLoadDetailTimer alloc_timer("gpu_pipeline.allocate");

        if (backend_ &&
            allocation_authority_kind_ ==
                AllocationAuthorityKind::PlanningOnly)
        {
            throw std::logic_error(
                "LoadOrchestrator cannot allocate through a backend without an explicit production authority or test-only token");
        }

        try
        {
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
            const size_t planned_weight_bytes = ctx.persistent_planned_bytes;
            const size_t maximum_source_bytes = ctx.pool
                                                    ? ctx.pool->maximumPlannedStagingBytes()
                                                    : 0u;
            const size_t admitted_source_bytes =
                staging_slots > 0
                    ? (pinned_slot_size > 0
                           ? std::min(
                                 maximum_source_bytes,
                                 pinned_slot_size)
                           : maximum_source_bytes)
                    : 0u;
            const GPUWeightLoadMemoryPolicy exact_policy{
                .staging_stream_count = staging_slots,
                .staging_budget_bytes = 0u,
            };
            const auto geometry = resolveGPUWeightLoadMemoryGeometry(
                admitted_source_bytes, exact_policy);
            const size_t persistent_bytes =
                alignGPUWeightLoadAllocation(planned_weight_bytes);
            const DeviceId device(
                backend_ ? backend_->backendDeviceType()
                         : DeviceType::CPU,
                backend_ ? ctx.device_id : 0);

            const auto owner_bytes = resolvedPersistentOwnerBytes(ctx);

            size_t attributed_persistent_bytes = 0u;
            for (const size_t bytes : owner_bytes)
            {
                if (bytes > std::numeric_limits<size_t>::max() -
                                attributed_persistent_bytes)
                {
                    throw std::overflow_error(
                        "LoadOrchestrator persistent owner aggregate overflowed");
                }
                attributed_persistent_bytes += bytes;
            }
            if (allocation_authority_kind_ == AllocationAuthorityKind::Production &&
                attributed_persistent_bytes != persistent_bytes)
            {
                throw std::logic_error(
                    "LoadOrchestrator persistent allocation is not completely attributed to physical-memory owners");
            }

            /*
             * Acquire every ledger claim into local RAII tokens first. A later
             * owner failure therefore rolls back earlier claims before any
             * device allocation can escape this transaction.
             */
            std::vector<PhysicalMemoryAllocationLease> weight_leases;
            std::optional<PhysicalMemoryAllocationLease>
                device_staging_lease;
            std::optional<PhysicalMemoryAllocationLease>
                host_staging_lease;
            if (allocation_authority_kind_ == AllocationAuthorityKind::Production)
            {
                for (size_t owner_index = 0;
                     owner_index < owner_bytes.size();
                     ++owner_index)
                {
                    if (owner_bytes[owner_index] == 0u)
                        continue;
                    weight_leases.emplace_back(
                        memory_authority_->claimNewAllocation(
                            device,
                            PhysicalMemoryBOM::ownerAt(owner_index),
                            owner_bytes[owner_index]));
                }
                if (geometry.staging_bytes != 0u)
                {
                    device_staging_lease.emplace(
                        memory_authority_->claimNewAllocation(
                            device,
                            PhysicalMemoryOwner::WeightLoadStaging,
                            geometry.staging_bytes));
                }
                if (geometry.host_staging_bytes != 0u)
                {
                    host_staging_lease.emplace(
                        memory_authority_->claimNewAllocation(
                            DeviceId::cpu(),
                            PhysicalMemoryOwner::WeightLoadStaging,
                            geometry.host_staging_bytes));
                }
            }

            // Allocate VRAM pool with staging slots
            if (!ctx.pool->allocate(backend_, ctx.device_id, num_h2d_streams,
                                    geometry.staging_slot_bytes))
            {
                throw std::runtime_error("LoadOrchestrator: failed to allocate pool for device " +
                                         std::to_string(ctx.device_id));
            }

            // Allocate pinned ring buffer
            if (geometry.staging_slot_bytes > 0 && num_h2d_streams > 0)
            {
                ctx.pinned_ring = std::make_unique<PinnedRingBuffer>(
                    geometry.staging_slot_bytes, num_h2d_streams);
                if (!ctx.pinned_ring->allocate(backend_, ctx.device_id))
                {
                    ctx.pool->release();
                    throw std::runtime_error("LoadOrchestrator: failed to allocate pinned ring for device " +
                                             std::to_string(ctx.device_id));
                }
            }

            if (ctx.pool->totalPlannedBytes() !=
                persistent_bytes + geometry.staging_bytes)
            {
                if (ctx.pinned_ring)
                    ctx.pinned_ring->release();
                ctx.pool->release();
                throw std::logic_error(
                    "LoadOrchestrator materialized a GPU weight pool whose bytes differ from its canonical allocation geometry");
            }

            /* Allocation and all three claims now enter their shared lifetime. */
            ctx.weight_leases = std::move(weight_leases);
            ctx.device_staging_lease = std::move(device_staging_lease);
            ctx.host_staging_lease = std::move(host_staging_lease);
            }
        }
        catch (...)
        {
            /*
             * Allocation is a topology transaction, not one transaction per
             * device. A later-device failure must not strand earlier VRAM or
             * make the live ledger disagree with physical ownership.
             */
            rollbackMaterializedAllocations();
            throw;
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
            if (PerfStatsCollector::isDomainEnabled("weight_loading"))
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
            ctx.host_staging_lease.reset();
            ctx.device_staging_lease.reset();
        }
    }

    void LoadOrchestrator::release()
    {
        rollbackMaterializedAllocations();
        devices_.clear();
    }

    void LoadOrchestrator::rollbackMaterializedAllocations() noexcept
    {
        for (auto &ctx : devices_)
        {
            if (ctx.pinned_ring)
                ctx.pinned_ring->release();
            if (ctx.pool)
                ctx.pool->release();
            ctx.host_staging_lease.reset();
            ctx.device_staging_lease.reset();
            ctx.weight_leases.clear();
        }
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
