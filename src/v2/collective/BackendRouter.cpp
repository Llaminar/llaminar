/**
 * @file BackendRouter.cpp
 * @brief BackendRouter implementation for collective backend selection
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "BackendRouter.h"
#include "backends/HostBackend.h"
#include "backends/MPIBackend.h"
#include "backends/UPIBackend.h"
#include "../config/TPDomain.h"
#ifdef HAVE_NCCL
#include "backends/NCCLBackend.h"
#endif
#ifdef HAVE_RCCL
#include "backends/RCCLBackend.h"
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#endif
#include "../utils/Logger.h"

namespace llaminar2
{

    // =========================================================================
    // BackendRouter Implementation
    // =========================================================================

    BackendRouter::BackendRouter(
        std::shared_ptr<IMPIContext> mpi_ctx,
        const ClusterInventory &cluster_inventory,
        std::unique_ptr<IBackendFactory> factory)
        : mpi_ctx_(std::move(mpi_ctx)),
          cluster_inventory_(cluster_inventory),
          factory_(factory ? std::move(factory) : std::make_unique<DefaultBackendFactory>())
    {
        // =====================================================================
        // Pre-initialize GPU backends with their coordinators
        // =====================================================================
        // CRITICAL: This must happen BEFORE any tensor allocations occur.
        // NCCL/RCCL use GPU contexts internally during ncclCommInitRank/rcclCommInitRank,
        // which can corrupt events/streams if initialized after tensors are created.
        // By initializing here, we ensure coordinators are ready before any GPU work.
        // =====================================================================
        preInitializeNCCLBackend();
        preInitializeRCCLBackend();
    }

    void BackendRouter::preInitializeNCCLBackend()
    {
#ifdef HAVE_NCCL
        // Check if NCCL backend is available
        if (!factory_->isAvailable(CollectiveBackendType::NCCL))
        {
            LOG_DEBUG("[BackendRouter] NCCL backend not available, skipping pre-initialization");
            return;
        }

        // Create NCCL backend if not already created
        auto *backend = getOrCreateBackend(CollectiveBackendType::NCCL);
        if (!backend)
        {
            LOG_DEBUG("[BackendRouter] Failed to create NCCL backend for pre-initialization");
            return;
        }

        // If already initialized, nothing to do
        if (backend->isInitialized())
        {
            LOG_DEBUG("[BackendRouter] NCCL backend already initialized");
            return;
        }

        // Get local CUDA devices from cluster inventory
        // Use rank 0 for single-process, or mpi_ctx_->rank() for multi-process
        const int my_rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
        const auto &rank_inv = cluster_inventory_.getRank(my_rank);

        // Count CUDA devices in this rank's inventory
        std::vector<int> cuda_ordinals;
        for (const auto &gpu : rank_inv.gpus)
        {
            if (gpu.type == DeviceType::CUDA)
            {
                cuda_ordinals.push_back(gpu.local_device_id);
            }
        }

        if (cuda_ordinals.empty())
        {
            LOG_DEBUG("[BackendRouter] No CUDA devices in cluster inventory, skipping NCCL pre-initialization");
            return;
        }

        // Build a device group with ALL available CUDA devices from inventory
        // This triggers the NCCL backend's copy communicator initialization
        DeviceGroupBuilder builder;
        builder.setName("nccl_copy_preinit")
            .setScope(CollectiveScope::LOCAL);

        for (int ordinal : cuda_ordinals)
        {
            builder.addDevice(DeviceId::cuda(ordinal));
        }

        DeviceGroup group = builder.build();

        if (!backend->initialize(group))
        {
            LOG_WARN("[BackendRouter] Failed to pre-initialize NCCL backend - "
                     "CUDA-to-CUDA copy may not work correctly");
            return;
        }

        LOG_INFO("[BackendRouter] Pre-initialized NCCL backend with copy communicators");
#else
        LOG_DEBUG("[BackendRouter] NCCL not available (HAVE_NCCL not defined)");
#endif
    }

    void BackendRouter::preInitializeRCCLBackend()
    {
#ifdef HAVE_RCCL
        // Check if RCCL backend is available
        if (!factory_->isAvailable(CollectiveBackendType::RCCL))
        {
            LOG_DEBUG("[BackendRouter] RCCL backend not available, skipping pre-initialization");
            return;
        }

        // Create RCCL backend if not already created
        auto *backend = getOrCreateBackend(CollectiveBackendType::RCCL);
        if (!backend)
        {
            LOG_DEBUG("[BackendRouter] Failed to create RCCL backend for pre-initialization");
            return;
        }

        // If already initialized, nothing to do
        if (backend->isInitialized())
        {
            LOG_DEBUG("[BackendRouter] RCCL backend already initialized");
            return;
        }

        // Get local ROCm devices from cluster inventory
        // Use rank 0 for single-process, or mpi_ctx_->rank() for multi-process
        const int my_rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
        const auto &rank_inv = cluster_inventory_.getRank(my_rank);

        // Count ROCm devices in this rank's inventory
        std::vector<int> rocm_ordinals;
        for (const auto &gpu : rank_inv.gpus)
        {
            if (gpu.type == DeviceType::ROCm)
            {
                rocm_ordinals.push_back(gpu.local_device_id);
            }
        }

        if (rocm_ordinals.empty())
        {
            LOG_DEBUG("[BackendRouter] No ROCm devices in cluster inventory, skipping RCCL pre-initialization");
            return;
        }

        // Build a device group with ALL available ROCm devices from inventory
        // This triggers the RCCL backend's copy communicator initialization
        DeviceGroupBuilder builder;
        builder.setName("rccl_copy_preinit")
            .setScope(CollectiveScope::LOCAL);

        for (int ordinal : rocm_ordinals)
        {
            builder.addDevice(DeviceId::rocm(ordinal));
        }

        DeviceGroup group = builder.build();

        if (!backend->initialize(group))
        {
            LOG_WARN("[BackendRouter] Failed to pre-initialize RCCL backend - "
                     "ROCm-to-ROCm copy may not work correctly");
            return;
        }

        LOG_INFO("[BackendRouter] Pre-initialized RCCL backend with copy communicators");
#else
        LOG_DEBUG("[BackendRouter] RCCL not available (HAVE_RCCL not defined)");
#endif
    }

    BackendRouter::~BackendRouter()
    {
        shutdown();
    }

    ICollectiveBackend *BackendRouter::getBackend(const DeviceGroup &group)
    {
        // Check cache first
        std::string key = makeGroupKey(group);
        auto it = group_backend_cache_.find(key);
        if (it != group_backend_cache_.end())
        {
            return it->second;
        }

        // Select and initialize backend
        CollectiveBackendType type = selectBackendType(group);

        // Log backend selection with group composition
        LOG_INFO("BackendRouter: Selected " << toString(type) << " for group '" << group.name
                                            << "' (cuda_count=" << group.cuda_count << ", rocm_count=" << group.rocm_count
                                            << ", heterogeneous=" << (group.isHeterogeneous() ? "true" : "false") << ")");

        ICollectiveBackend *backend = getOrCreateBackend(type);

        if (backend && !backend->isInitialized())
        {
            if (!backend->initialize(group))
            {
                LOG_ERROR("Failed to initialize " << toString(type) << " backend for group " << group.name);
                return nullptr;
            }
        }

        group_backend_cache_[key] = backend;
        return backend;
    }

    ICollectiveBackend *BackendRouter::getBackend(CollectiveBackendType type)
    {
        return getOrCreateBackend(type);
    }

    BackendSelection BackendRouter::selectBackend(const DeviceGroup &group) const
    {
        BackendSelection selection;
        selection.type = selectBackendType(group);

        // Generate reason string based on selection type and group composition
        // Note: NCCL/RCCL support cross-rank (GLOBAL scope) communication via
        // ncclCommInitRank/rcclCommInitRank, so we prefer them for homogeneous GPU groups.
        if (group.allCUDA() && selection.type == CollectiveBackendType::NCCL)
        {
            if (group.isGlobal())
            {
                selection.reason = "All CUDA devices (cross-rank) - using NCCL";
            }
            else
            {
                selection.reason = "All CUDA devices - using NCCL";
            }
        }
        else if (group.allROCm() && selection.type == CollectiveBackendType::RCCL)
        {
            if (group.isGlobal())
            {
                selection.reason = "All ROCm devices (cross-rank) - using RCCL";
            }
            else
            {
                selection.reason = "All ROCm devices - using RCCL";
            }
        }
        else if (group.isGlobal() && selection.type == CollectiveBackendType::MPI)
        {
            selection.reason = "Global scope (heterogeneous or CPU-only) - using MPI";
        }
        else if (group.isHeterogeneous())
        {
            selection.reason = "Heterogeneous group - using Host backend (CPU staging)";
            selection.requires_multi_phase = true;
        }
        else
        {
            selection.reason = "Fallback to Host backend";
        }

        return selection;
    }

    ICollectiveBackend *BackendRouter::getBackendForCopy(DeviceId src, DeviceId dst)
    {
        // Same device = no transfer needed, but return a backend anyway
        if (src == dst)
        {
            if (src.is_cuda())
            {
                auto *backend = getOrCreateBackend(CollectiveBackendType::NCCL);
                if (backend && backend->supportsCopy(src, dst))
                    return backend;
            }
            else if (src.is_rocm())
            {
                auto *backend = getOrCreateBackend(CollectiveBackendType::RCCL);
                if (backend && backend->supportsCopy(src, dst))
                    return backend;
            }
            else if (src.is_cpu())
            {
                auto *backend = getOrCreateBackend(CollectiveBackendType::HOST);
                if (backend && backend->supportsCopy(src, dst))
                    return backend;
            }
            return nullptr;
        }

        // Cross-vendor (CUDA↔ROCm): use HOST backend (host-staged copy)
        if ((src.is_cuda() && dst.is_rocm()) || (src.is_rocm() && dst.is_cuda()))
        {
            auto *backend = getOrCreateBackend(CollectiveBackendType::HOST);
            if (backend && backend->supportsCopy(src, dst))
                return backend;
            return nullptr;
        }

        // Same CUDA vendor: use NCCL (supports P2P or pinned staging fallback)
        if (src.is_cuda() && dst.is_cuda())
        {
            auto *backend = getOrCreateBackend(CollectiveBackendType::NCCL);
            if (backend)
            {
                // NCCL backend needs to be initialized before copy() works.
                // The copy communicator is created during initialize() to span all CUDA devices.
                if (!backend->isInitialized())
                {
                    DeviceGroupBuilder builder;
                    builder.setName("nccl_copy_" + src.toString() + "_to_" + dst.toString())
                        .setScope(CollectiveScope::LOCAL)
                        .addDevice(src)
                        .addDevice(dst);
                    DeviceGroup group = builder.build();

                    if (!backend->initialize(group))
                    {
                        LOG_ERROR("[getBackendForCopy] Failed to initialize NCCL backend for "
                                  << src.toString() << " -> " << dst.toString());
                        return nullptr;
                    }
                }

                if (backend->supportsCopy(src, dst))
                {
                    return backend;
                }
            }
        }

        // Same ROCm vendor: use RCCL (supports P2P or pinned staging fallback)
        if (src.is_rocm() && dst.is_rocm())
        {
            auto *backend = getOrCreateBackend(CollectiveBackendType::RCCL);
            if (backend)
            {
                // RCCL backend needs to be initialized before copy() works.
                // The copy communicator is created during initialize() to span all ROCm devices.
                if (!backend->isInitialized())
                {
                    DeviceGroupBuilder builder;
                    builder.setName("rccl_copy_" + src.toString() + "_to_" + dst.toString())
                        .setScope(CollectiveScope::LOCAL)
                        .addDevice(src)
                        .addDevice(dst);
                    DeviceGroup group = builder.build();

                    if (!backend->initialize(group))
                    {
                        LOG_ERROR("[getBackendForCopy] Failed to initialize RCCL backend for "
                                  << src.toString() << " -> " << dst.toString());
                        return nullptr;
                    }
                }

                if (backend->supportsCopy(src, dst))
                {
                    return backend;
                }
            }
        }

        // CPU-to-CPU: use Host backend
        if (src.is_cpu() && dst.is_cpu())
        {
            auto *backend = getOrCreateBackend(CollectiveBackendType::HOST);
            if (backend && backend->supportsCopy(src, dst))
            {
                return backend;
            }
        }

        return nullptr; // No suitable backend
    }

    bool BackendRouter::isAvailable(CollectiveBackendType type) const
    {
        return factory_->isAvailable(type);
    }

    bool BackendRouter::initializeBackend(CollectiveBackendType type, const DeviceGroup &group)
    {
        ICollectiveBackend *backend = getOrCreateBackend(type);
        if (!backend)
        {
            return false;
        }

        if (backend->isInitialized())
        {
            return true;
        }

        return backend->initialize(group);
    }

    std::vector<CollectiveBackendType> BackendRouter::availableBackends() const
    {
        std::vector<CollectiveBackendType> result;
        if (factory_->isAvailable(CollectiveBackendType::MPI))
            result.push_back(CollectiveBackendType::MPI);
        if (factory_->isAvailable(CollectiveBackendType::NCCL))
            result.push_back(CollectiveBackendType::NCCL);
        if (factory_->isAvailable(CollectiveBackendType::RCCL))
            result.push_back(CollectiveBackendType::RCCL);
        if (factory_->isAvailable(CollectiveBackendType::HOST))
            result.push_back(CollectiveBackendType::HOST);
        return result;
    }

    void BackendRouter::shutdown()
    {
        group_backend_cache_.clear();

        if (mpi_backend_)
        {
            mpi_backend_->shutdown();
            mpi_backend_.reset();
        }
        if (nccl_backend_)
        {
            nccl_backend_->shutdown();
            nccl_backend_.reset();
        }
        if (rccl_backend_)
        {
            rccl_backend_->shutdown();
            rccl_backend_.reset();
        }
        if (host_backend_)
        {
            host_backend_->shutdown();
            host_backend_.reset();
        }
        if (upi_backend_)
        {
            upi_backend_->shutdown();
            upi_backend_.reset();
        }
    }

    bool BackendRouter::executeHeterogeneousAllReduce(
        const DeviceGroup &group,
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        // Placeholder implementation - multi-phase heterogeneous AllReduce
        // TODO: Implement proper multi-phase execution
        LOG_WARN("Heterogeneous AllReduce not yet implemented - falling back to Host");

        ICollectiveBackend *host = getOrCreateBackend(CollectiveBackendType::HOST);
        if (!host)
        {
            return false;
        }

        if (!host->isInitialized())
        {
            if (!host->initialize(group))
            {
                return false;
            }
        }

        return host->allreduce(buffer, count, dtype, op);
    }

    bool BackendRouter::executeHeterogeneousAllGather(
        const DeviceGroup &group,
        const void *send_buf,
        void *recv_buf,
        size_t send_count,
        CollectiveDataType dtype)
    {
        // Placeholder implementation - multi-phase heterogeneous AllGather
        LOG_WARN("Heterogeneous AllGather not yet implemented - falling back to Host");

        ICollectiveBackend *host = getOrCreateBackend(CollectiveBackendType::HOST);
        if (!host)
        {
            return false;
        }

        if (!host->isInitialized())
        {
            if (!host->initialize(group))
            {
                return false;
            }
        }

        return host->allgather(send_buf, recv_buf, send_count, dtype);
    }

    std::string BackendRouter::diagnostics() const
    {
        std::string result = "BackendRouter diagnostics:\n";
        result += "  MPI backend: " + std::string(mpi_backend_ ? "created" : "not created") + "\n";
        result += "  NCCL backend: " + std::string(nccl_backend_ ? "created" : "not created") + "\n";
        result += "  RCCL backend: " + std::string(rccl_backend_ ? "created" : "not created") + "\n";
        result += "  Host backend: " + std::string(host_backend_ ? "created" : "not created") + "\n";
        result += "  UPI backend: " + std::string(upi_backend_ ? "registered" : "not registered") + "\n";
        result += "  Cached groups: " + std::to_string(group_backend_cache_.size()) + "\n";
        result += "  Domain support: " + std::string(hasDomainSupport() ? "yes" : "no") + "\n";
        return result;
    }

    // =========================================================================
    // Domain-Aware Backend Selection
    // =========================================================================

    ICollectiveBackend *BackendRouter::selectBackendForDomain(const TPDomain *domain)
    {
        // No domain specified - use MPI world backend as fallback
        if (!domain)
        {
            LOG_DEBUG("selectBackendForDomain: No domain specified, using MPI backend");
            return getOrCreateBackend(CollectiveBackendType::MPI);
        }

        // Trivial domain (size <= 1) - no communication needed
        if (domain->isTrivial())
        {
            LOG_DEBUG("selectBackendForDomain: Trivial domain '" << domain->name << "', no backend needed");
            return nullptr;
        }

        switch (domain->type)
        {
        case TPDomainType::GPU_INTRA_RANK:
        {
            // GPU intra-rank: check device composition
            if (hasHeterogeneousGPUs(domain->devices))
            {
                // CUDA + ROCm mix: use HETEROGENEOUS backend for cross-vendor collective
                LOG_DEBUG("selectBackendForDomain: GPU domain '" << domain->name
                                                                 << "' has heterogeneous GPUs, using HETEROGENEOUS backend");
                return getOrCreateBackend(CollectiveBackendType::HETEROGENEOUS);
            }
            else if (allCUDA(domain->devices))
            {
                // All CUDA: use NCCL if available
                if (factory_->isAvailable(CollectiveBackendType::NCCL))
                {
                    LOG_DEBUG("selectBackendForDomain: GPU domain '" << domain->name
                                                                     << "' is all CUDA, using NCCL backend");
                    return getOrCreateBackend(CollectiveBackendType::NCCL);
                }
                LOG_DEBUG("selectBackendForDomain: GPU domain '" << domain->name
                                                                 << "' is all CUDA but NCCL unavailable, falling back to MPI");
                return getOrCreateBackend(CollectiveBackendType::MPI);
            }
            else if (allROCm(domain->devices))
            {
                // All ROCm: use RCCL if available
                if (factory_->isAvailable(CollectiveBackendType::RCCL))
                {
                    LOG_DEBUG("selectBackendForDomain: GPU domain '" << domain->name
                                                                     << "' is all ROCm, using RCCL backend");
                    return getOrCreateBackend(CollectiveBackendType::RCCL);
                }
                LOG_DEBUG("selectBackendForDomain: GPU domain '" << domain->name
                                                                 << "' is all ROCm but RCCL unavailable, falling back to MPI");
                return getOrCreateBackend(CollectiveBackendType::MPI);
            }
            // Mixed or unknown GPU types - fallback to MPI
            LOG_DEBUG("selectBackendForDomain: GPU domain '" << domain->name
                                                             << "' has unknown GPU mix, falling back to MPI");
            return getOrCreateBackend(CollectiveBackendType::MPI);
        }

        case TPDomainType::CPU_CROSS_RANK:
        {
            // CPU cross-rank: use UPI backend if registered and valid
            if (upi_backend_ && upi_backend_->isValid())
            {
                LOG_DEBUG("selectBackendForDomain: CPU domain '" << domain->name
                                                                 << "' using UPI backend (MPI over UPI ~50 GB/s)");
                return upi_backend_.get();
            }
            // Fallback to MPI backend
            LOG_DEBUG("selectBackendForDomain: CPU domain '" << domain->name
                                                             << "' has no UPI backend, falling back to MPI");
            return getOrCreateBackend(CollectiveBackendType::MPI);
        }
        }

        // Unknown domain type - fallback to MPI
        LOG_WARN("selectBackendForDomain: Unknown domain type, falling back to MPI");
        return getOrCreateBackend(CollectiveBackendType::MPI);
    }

    void BackendRouter::registerUPIBackend(std::unique_ptr<UPICollectiveBackend> backend)
    {
        if (backend)
        {
            LOG_INFO("BackendRouter: Registering UPI backend (domain rank=" << backend->domainRank()
                                                                            << ", domain size=" << backend->domainSize() << ")");
        }
        upi_backend_ = std::move(backend);
    }

    bool BackendRouter::hasDomainSupport() const
    {
        return upi_backend_ != nullptr;
    }

    // =========================================================================
    // Domain-Aware Selection Helpers
    // =========================================================================

    bool BackendRouter::hasHeterogeneousGPUs(const std::vector<DeviceId> &devices) const
    {
        bool has_cuda = false;
        bool has_rocm = false;
        for (const auto &dev : devices)
        {
            if (dev.type == DeviceType::CUDA)
                has_cuda = true;
            if (dev.type == DeviceType::ROCm)
                has_rocm = true;
        }
        return has_cuda && has_rocm;
    }

    bool BackendRouter::allCUDA(const std::vector<DeviceId> &devices) const
    {
        if (devices.empty())
            return false;
        return std::all_of(devices.begin(), devices.end(),
                           [](const DeviceId &d)
                           { return d.type == DeviceType::CUDA; });
    }

    bool BackendRouter::allROCm(const std::vector<DeviceId> &devices) const
    {
        if (devices.empty())
            return false;
        return std::all_of(devices.begin(), devices.end(),
                           [](const DeviceId &d)
                           { return d.type == DeviceType::ROCm; });
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    ICollectiveBackend *BackendRouter::getOrCreateBackend(CollectiveBackendType type)
    {
        switch (type)
        {
        case CollectiveBackendType::MPI:
            if (!mpi_backend_ && factory_->isAvailable(type))
            {
                mpi_backend_ = factory_->createBackend(type, mpi_ctx_);
            }
            return mpi_backend_.get();

        case CollectiveBackendType::NCCL:
            if (!nccl_backend_ && factory_->isAvailable(type))
            {
                nccl_backend_ = factory_->createBackend(type, mpi_ctx_);
            }
            return nccl_backend_.get();

        case CollectiveBackendType::RCCL:
            if (!rccl_backend_ && factory_->isAvailable(type))
            {
                rccl_backend_ = factory_->createBackend(type, mpi_ctx_);
            }
            return rccl_backend_.get();

        case CollectiveBackendType::HOST:
            if (!host_backend_ && factory_->isAvailable(type))
            {
                host_backend_ = factory_->createBackend(type, mpi_ctx_);
            }
            return host_backend_.get();

        case CollectiveBackendType::HETEROGENEOUS:
            if (!heterogeneous_backend_ && factory_->isAvailable(type))
            {
                heterogeneous_backend_ = factory_->createBackend(type, mpi_ctx_);
            }
            return heterogeneous_backend_.get();

        case CollectiveBackendType::AUTO:
            // AUTO should be resolved before calling this
            LOG_ERROR("getOrCreateBackend called with AUTO type");
            return nullptr;
        }

        return nullptr;
    }

    CollectiveBackendType BackendRouter::selectBackendType(const DeviceGroup &group) const
    {
        // =========================================================================
        // Native GPU Collectives (NCCL/RCCL)
        // =========================================================================
        // Both NCCL and RCCL support cross-rank (MPI) communication via
        // ncclCommInitRank/rcclCommInitRank. They are preferred for GPU collectives
        // regardless of scope (LOCAL or GLOBAL) when the group is homogeneous.
        //
        // This enables GPU-native AllReduce even for tensor-parallel inference
        // across MPI ranks, avoiding GPU→CPU→GPU transfers through MPI.
        // =========================================================================

        if (group.allCUDA() && factory_->isAvailable(CollectiveBackendType::NCCL))
        {
            return CollectiveBackendType::NCCL;
        }

        if (group.allROCm() && factory_->isAvailable(CollectiveBackendType::RCCL))
        {
            return CollectiveBackendType::RCCL;
        }

        // For GLOBAL scope (cross-rank MPI), heterogeneous groups MUST use MPI.
        if (group.isGlobal())
        {
            return CollectiveBackendType::MPI;
        }

        // Heterogeneous GPU mix or CPU-only: use Host backend
        return CollectiveBackendType::HOST;
    }

    std::string BackendRouter::makeGroupKey(const DeviceGroup &group) const
    {
        // Create a unique key for the group based on its composition
        return group.name + "_" + std::to_string(static_cast<int>(group.scope));
    }

    // =========================================================================
    // DefaultBackendFactory Implementation
    // =========================================================================

    std::unique_ptr<ICollectiveBackend> DefaultBackendFactory::createBackend(
        CollectiveBackendType type,
        std::shared_ptr<IMPIContext> mpi_ctx)
    {
        switch (type)
        {
        case CollectiveBackendType::HOST:
            return std::make_unique<HostBackend>();

        case CollectiveBackendType::MPI:
            return std::make_unique<MPIBackend>(mpi_ctx);

        case CollectiveBackendType::NCCL:
#ifdef HAVE_NCCL
            return std::make_unique<NCCLBackend>(mpi_ctx);
#else
            LOG_INFO("DefaultBackendFactory::createBackend - NCCL not available (HAVE_NCCL not defined)");
            return nullptr;
#endif

        case CollectiveBackendType::RCCL:
#ifdef HAVE_RCCL
            return std::make_unique<RCCLBackend>(mpi_ctx);
#else
            LOG_INFO("DefaultBackendFactory::createBackend - RCCL not available (HAVE_RCCL not defined)");
            return nullptr;
#endif

        case CollectiveBackendType::AUTO:
            // AUTO should be resolved before calling createBackend
            LOG_ERROR("DefaultBackendFactory::createBackend - AUTO type should be resolved first");
            return nullptr;
        }

        return nullptr;
    }

    bool DefaultBackendFactory::isAvailable(CollectiveBackendType type) const
    {
        switch (type)
        {
        case CollectiveBackendType::MPI:
            return true; // MPI is always available (we require it)

        case CollectiveBackendType::HOST:
            return true; // Host backend is always available

        case CollectiveBackendType::NCCL:
#ifdef HAVE_NCCL
            return true;
#else
            return false;
#endif

        case CollectiveBackendType::RCCL:
#ifdef HAVE_RCCL
            return true;
#else
            return false;
#endif

        case CollectiveBackendType::AUTO:
            return true; // AUTO is always available (it's a selection mode)
        }

        return false;
    }

    // =========================================================================
    // GlobalBackendRouter Implementation
    // =========================================================================

    std::unique_ptr<BackendRouter> GlobalBackendRouter::instance_;

    void GlobalBackendRouter::init(
        std::shared_ptr<IMPIContext> mpi_ctx,
        const ClusterInventory &cluster_inventory)
    {
        instance_ = std::make_unique<BackendRouter>(std::move(mpi_ctx), cluster_inventory);
    }

    bool GlobalBackendRouter::initForTests()
    {
        // Already initialized?
        if (instance_)
        {
            return true;
        }

        // Create cluster inventory populated with local devices from DeviceManager
        // This is the ONE place where we access the DeviceManager singleton for tests
        ClusterInventory test_inventory;
        test_inventory.world_size = 1;
        test_inventory.node_count = 1;

        // Create rank 0 inventory with actual devices
        RankInventory rank0;
        rank0.rank = 0;
        rank0.node_id = 0;
        rank0.local_rank = 0;

        // Populate GPU list from DeviceManager
        const auto &dm = DeviceManager::instance();
        const int cuda_count = dm.cuda_device_count();
        const int rocm_count = dm.rocm_device_count();

        for (int i = 0; i < cuda_count; ++i)
        {
            DeviceInfo info;
            info.type = DeviceType::CUDA;
            info.local_device_id = i;
            rank0.gpus.push_back(info);
        }

        for (int i = 0; i < rocm_count; ++i)
        {
            DeviceInfo info;
            info.type = DeviceType::ROCm;
            info.local_device_id = i;
            rank0.gpus.push_back(info);
        }

        test_inventory.ranks.push_back(rank0);
        test_inventory.total_gpus = cuda_count + rocm_count;

        // No MPI context needed for tests
        instance_ = std::make_unique<BackendRouter>(nullptr, test_inventory);

        LOG_INFO("[GlobalBackendRouter] Initialized for testing (no MPI)");
        return instance_ != nullptr;
    }

    BackendRouter *GlobalBackendRouter::get()
    {
        return instance_.get();
    }

    void GlobalBackendRouter::shutdown()
    {
        instance_.reset();
        // Drain the RCCL coordinator pool after all backends are destroyed.
        // This ensures ncclCommDestroy only happens once at process shutdown,
        // avoiding the ROCm CLR state accumulation bug from repeated cycles.
        RCCLBackend::drainCoordinatorPool();
    }

    // =========================================================================
    // CollectiveBackendFactory Implementation (static factory)
    // =========================================================================

    std::unique_ptr<ICollectiveBackend> CollectiveBackendFactory::create(CollectiveBackendType type)
    {
        DefaultBackendFactory factory;
        return factory.createBackend(type, nullptr);
    }

    bool CollectiveBackendFactory::isAvailable(CollectiveBackendType type)
    {
        DefaultBackendFactory factory;
        return factory.isAvailable(type);
    }

    std::vector<CollectiveBackendType> CollectiveBackendFactory::availableBackends()
    {
        std::vector<CollectiveBackendType> result;
        DefaultBackendFactory factory;

        if (factory.isAvailable(CollectiveBackendType::MPI))
            result.push_back(CollectiveBackendType::MPI);
        if (factory.isAvailable(CollectiveBackendType::NCCL))
            result.push_back(CollectiveBackendType::NCCL);
        if (factory.isAvailable(CollectiveBackendType::RCCL))
            result.push_back(CollectiveBackendType::RCCL);
        if (factory.isAvailable(CollectiveBackendType::HOST))
            result.push_back(CollectiveBackendType::HOST);

        return result;
    }

} // namespace llaminar2
