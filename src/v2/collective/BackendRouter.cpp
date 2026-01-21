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
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#include "backends/PCIeBARBackend.h"
#endif
#include "../utils/Logger.h"

namespace llaminar2
{

    // =========================================================================
    // BackendRouter Implementation
    // =========================================================================

    BackendRouter::BackendRouter(
        std::shared_ptr<MPIContext> mpi_ctx,
        const ClusterInventory &cluster_inventory,
        std::unique_ptr<IBackendFactory> factory)
        : mpi_ctx_(std::move(mpi_ctx)),
          cluster_inventory_(cluster_inventory),
          factory_(factory ? std::move(factory) : std::make_unique<DefaultBackendFactory>())
    {
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

        // Generate reason string
        if (group.isGlobal())
        {
            selection.reason = "Global scope requires MPI";
        }
        else if (group.allCUDA())
        {
            selection.reason = "All CUDA devices - using NCCL";
        }
        else if (group.allROCm())
        {
            selection.reason = "All ROCm devices - using RCCL";
        }
        else if (group.isHeterogeneous() && group.cuda_count > 0 && group.rocm_count > 0 &&
                 selection.type == CollectiveBackendType::PCIE_BAR)
        {
            selection.reason = "CUDA + ROCm mix - using direct PCIe BAR P2P (~2.65 GB/s)";
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
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (factory_->isAvailable(CollectiveBackendType::PCIE_BAR))
            result.push_back(CollectiveBackendType::PCIE_BAR);
#endif
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
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (pcie_bar_backend_)
        {
            pcie_bar_backend_->shutdown();
            pcie_bar_backend_.reset();
        }
#endif
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
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        result += "  PCIe_BAR backend: " + std::string(pcie_bar_backend_ ? "created" : "not created") + "\n";
#else
        result += "  PCIe_BAR backend: not available (requires HAVE_CUDA && HAVE_ROCM)\n";
#endif
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
                // CUDA + ROCm mix: prefer PCIe BAR for direct P2P (~25μs latency)
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
                if (factory_->isAvailable(CollectiveBackendType::PCIE_BAR))
                {
                    LOG_DEBUG("selectBackendForDomain: GPU domain '" << domain->name
                                                                     << "' has heterogeneous GPUs, using PCIe BAR backend");
                    return getOrCreateBackend(CollectiveBackendType::PCIE_BAR);
                }
#endif
                LOG_DEBUG("selectBackendForDomain: GPU domain '" << domain->name
                                                                 << "' has heterogeneous GPUs but PCIe BAR unavailable, falling back to MPI");
                return getOrCreateBackend(CollectiveBackendType::MPI);
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

        case CollectiveBackendType::PCIE_BAR:
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            if (!pcie_bar_backend_ && factory_->isAvailable(type))
            {
                pcie_bar_backend_ = factory_->createBackend(type, mpi_ctx_);
            }
            return pcie_bar_backend_.get();
#else
            LOG_WARN("PCIE_BAR requested but HAVE_CUDA && HAVE_ROCM not defined");
            return nullptr;
#endif

        case CollectiveBackendType::HOST:
            if (!host_backend_ && factory_->isAvailable(type))
            {
                host_backend_ = factory_->createBackend(type, mpi_ctx_);
            }
            return host_backend_.get();

        case CollectiveBackendType::AUTO:
            // AUTO should be resolved before calling this
            LOG_ERROR("getOrCreateBackend called with AUTO type");
            return nullptr;
        }

        return nullptr;
    }

    CollectiveBackendType BackendRouter::selectBackendType(const DeviceGroup &group) const
    {
        // Global scope always uses MPI
        if (group.isGlobal())
        {
            return CollectiveBackendType::MPI;
        }

        // Local scope: prefer native GPU backend if homogeneous
        if (group.allCUDA() && factory_->isAvailable(CollectiveBackendType::NCCL))
        {
            return CollectiveBackendType::NCCL;
        }

        if (group.allROCm() && factory_->isAvailable(CollectiveBackendType::RCCL))
        {
            return CollectiveBackendType::RCCL;
        }

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        // CUDA + ROCm mix: prefer PCIe BAR if available (direct P2P)
        if (group.isHeterogeneous() && group.cuda_count > 0 && group.rocm_count > 0)
        {
            if (factory_->isAvailable(CollectiveBackendType::PCIE_BAR))
            {
                return CollectiveBackendType::PCIE_BAR;
            }
        }
#endif

        // CPU-only or heterogeneous without P2P: use Host backend
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
        std::shared_ptr<MPIContext> mpi_ctx)
    {
        switch (type)
        {
        case CollectiveBackendType::HOST:
            return std::make_unique<HostBackend>();

        case CollectiveBackendType::MPI:
            return std::make_unique<MPIBackend>(mpi_ctx);

        case CollectiveBackendType::NCCL:
            // NCCL backend not yet implemented
            LOG_INFO("DefaultBackendFactory::createBackend - NCCL backend not yet implemented");
            return nullptr;

        case CollectiveBackendType::RCCL:
            // RCCL backend not yet implemented
            LOG_INFO("DefaultBackendFactory::createBackend - RCCL backend not yet implemented");
            return nullptr;

        case CollectiveBackendType::PCIE_BAR:
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            return std::make_unique<PCIeBARBackend>();
#else
            LOG_INFO("DefaultBackendFactory::createBackend - PCIE_BAR requires HAVE_CUDA && HAVE_ROCM");
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

        case CollectiveBackendType::PCIE_BAR:
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        {
            // Check runtime availability (BAR P2P requires both CUDA and ROCm GPUs + permissions)
            auto caps = DirectP2PEngine::probeCapabilities();
            return caps.canDoPCIeBarP2P();
        }
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
        std::shared_ptr<MPIContext> mpi_ctx,
        const ClusterInventory &cluster_inventory)
    {
        instance_ = std::make_unique<BackendRouter>(std::move(mpi_ctx), cluster_inventory);
    }

    BackendRouter *GlobalBackendRouter::get()
    {
        return instance_.get();
    }

    void GlobalBackendRouter::shutdown()
    {
        instance_.reset();
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
