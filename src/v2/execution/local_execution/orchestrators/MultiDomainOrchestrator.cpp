/**
 * @file MultiDomainOrchestrator.cpp
 * @brief Multi-domain tensor parallelism wrapper implementation
 * @author David Sanftenberg
 * @date January 2026
 */

#include "MultiDomainOrchestrator.h"
#include "DeviceGraphOrchestrator.h"
#include "../../../utils/Logger.h"

namespace llaminar2
{

    // =============================================================================
    // Constructor / Destructor
    // =============================================================================

    MultiDomainOrchestrator::MultiDomainOrchestrator()
        : stats_{}, initialized_(false)
    {
    }

    MultiDomainOrchestrator::~MultiDomainOrchestrator()
    {
        // tp_config_ destructor will clean up MPI communicators
        LOG_DEBUG("MultiDomainOrchestrator destroyed");
    }

    // =============================================================================
    // Factory Methods
    // =============================================================================

    std::unique_ptr<MultiDomainOrchestrator> MultiDomainOrchestrator::create(
        MultiDomainOrchestratorConfig config,
        IMPIContext *mpi_ctx)
    {
        auto orchestrator = std::unique_ptr<MultiDomainOrchestrator>(new MultiDomainOrchestrator());

        if (!orchestrator->initialize(std::move(config), mpi_ctx))
        {
            LOG_ERROR("Failed to initialize MultiDomainOrchestrator");
            return nullptr;
        }

        return orchestrator;
    }

    std::unique_ptr<MultiDomainOrchestrator> MultiDomainOrchestrator::createForTest(
        std::unique_ptr<IInferenceRunner> inner,
        std::unique_ptr<MultiDomainTPConfig> tp_config)
    {
        if (!inner)
        {
            LOG_ERROR("createForTest: inner orchestrator cannot be null");
            return nullptr;
        }

        auto orchestrator = std::unique_ptr<MultiDomainOrchestrator>(new MultiDomainOrchestrator());

        orchestrator->inner_runner_ = std::move(inner);
        orchestrator->tp_config_ = std::move(tp_config);
        orchestrator->initialized_ = true;

        LOG_DEBUG("MultiDomainOrchestrator created for testing");
        return orchestrator;
    }

    // =============================================================================
    // Initialization
    // =============================================================================

    bool MultiDomainOrchestrator::initialize(
        MultiDomainOrchestratorConfig config,
        IMPIContext *mpi_ctx)
    {
        config_ = std::move(config);

        // Detect system topology
        topology_ = std::make_unique<NodeTopology>(NodeTopology::detect());
        LOG_INFO("Detected topology: " << topology_->toString());

        // Use provided TP config or create one based on topology
        if (config_.tp_config)
        {
            tp_config_ = std::move(config_.tp_config);
        }
        else
        {
            // Auto-configure TP domains based on topology
            // For now, create empty config - full implementation will detect GPUs/CPUs
            // and create appropriate domains
            LOG_DEBUG("No TP config provided, will use default (single domain)");
        }

        // Create inner DeviceGraphOrchestrator
        // Note: Full implementation would create QwenStandardGraph and pass to DeviceGraphOrchestrator
        // For now, we require model loading which will be implemented in integration
        if (config_.model_path.empty())
        {
            LOG_ERROR("Model path is required for initialization");
            return false;
        }

        // TODO: Create inner orchestrator from model path
        // This requires ModelLoader and QwenStandardGraph which are integration-level dependencies
        // For unit testing, use createForTest() with a mock orchestrator
        LOG_WARN("Full initialization not yet implemented - use createForTest() for unit tests");

        initialized_ = false;
        return false;
    }

    // =============================================================================
    // IInferenceRunner Interface - Delegation to Inner Orchestrator
    // =============================================================================

    bool MultiDomainOrchestrator::forward(const int *tokens, int seq_len)
    {
        if (!initialized_ || !inner_runner_)
        {
            LOG_ERROR("MultiDomainOrchestrator not initialized");
            return false;
        }

        return inner_runner_->forward(tokens, seq_len);
    }

    const float *MultiDomainOrchestrator::logits() const
    {
        if (!initialized_ || !inner_runner_)
        {
            return nullptr;
        }

        return inner_runner_->logits();
    }

    int MultiDomainOrchestrator::vocab_size() const
    {
        if (!initialized_ || !inner_runner_)
        {
            return 0;
        }

        return inner_runner_->vocab_size();
    }

    void MultiDomainOrchestrator::clear_cache()
    {
        if (initialized_ && inner_runner_)
        {
            inner_runner_->clear_cache();
        }
    }

    int MultiDomainOrchestrator::get_position() const
    {
        if (!initialized_ || !inner_runner_)
        {
            return 0;
        }

        return inner_runner_->get_position();
    }

    ExecutionPath MultiDomainOrchestrator::executionPath() const
    {
        return ExecutionPath::GRAPH;
    }

    const char *MultiDomainOrchestrator::architecture() const
    {
        if (!initialized_ || !inner_runner_)
        {
            return "unknown";
        }

        return inner_runner_->architecture();
    }

    // =============================================================================
    // Domain Access
    // =============================================================================

    const MultiDomainTPConfig *MultiDomainOrchestrator::getTPConfig() const
    {
        return tp_config_.get();
    }

    const TPDomain *MultiDomainOrchestrator::getGPUDomain() const
    {
        if (!tp_config_)
        {
            return nullptr;
        }

        return tp_config_->gpuDomain();
    }

    const TPDomain *MultiDomainOrchestrator::getCPUDomain() const
    {
        if (!tp_config_)
        {
            return nullptr;
        }

        return tp_config_->cpuDomain();
    }

    const NodeTopology *MultiDomainOrchestrator::getTopology() const
    {
        return topology_.get();
    }

    IInferenceRunner *MultiDomainOrchestrator::getInnerRunner() const
    {
        return inner_runner_.get();
    }

    DeviceGraphOrchestrator *MultiDomainOrchestrator::getInnerOrchestrator() const
    {
        // Dynamic cast to DeviceGraphOrchestrator - returns nullptr for mocks in tests
        return dynamic_cast<DeviceGraphOrchestrator *>(inner_runner_.get());
    }

    // =============================================================================
    // Statistics
    // =============================================================================

    MultiDomainOrchestrator::DomainStats MultiDomainOrchestrator::getDomainStats() const
    {
        return stats_;
    }

    void MultiDomainOrchestrator::resetStats()
    {
        stats_ = DomainStats{};
    }

    // =============================================================================
    // Status
    // =============================================================================

    bool MultiDomainOrchestrator::isReady() const
    {
        return initialized_ && inner_runner_ != nullptr;
    }

    std::string MultiDomainOrchestrator::getModelInfo() const
    {
        std::string info = "MultiDomainOrchestrator";

        if (!config_.model_path.empty())
        {
            info += " model=" + config_.model_path;
        }

        if (tp_config_)
        {
            if (auto *gpu_domain = tp_config_->gpuDomain())
            {
                info += " gpu_domain_size=" + std::to_string(gpu_domain->domain_size);
            }
            if (auto *cpu_domain = tp_config_->cpuDomain())
            {
                info += " cpu_domain_size=" + std::to_string(cpu_domain->domain_size);
            }
        }

        if (topology_)
        {
            info += " sockets=" + std::to_string(topology_->numSockets());
        }

        return info;
    }

} // namespace llaminar2
