/**
 * @file abstract_pipeline.cpp
 * @brief Implementation of multi-architecture pipeline factory scaffolding.
 */
#include "abstract_pipeline.h"
#include "pipeline_snapshot_manager.h"
#include "logger.h"

namespace llaminar
{
    // ==================== AbstractPipeline Default Implementations ====================

    void AbstractPipeline::captureStageSnapshot(
        PipelineStage stage,
        int layer_index,
        const float *data,
        int seq_len,
        int feature_dim)
    {
        // Default implementation delegates to the snapshot manager
        // In release builds, this compiles to a no-op
        // In debug builds, this checks LLAMINAR_PARITY_CAPTURE environment variable
        PipelineSnapshotManager::instance().capture(
            stage,
            layer_index,
            data,
            seq_len,
            feature_dim,
            "llaminar");
    }

    bool AbstractPipeline::isParityEnabled() const
    {
        // Default implementation checks the snapshot manager
        // In release builds, always returns false (optimized away)
        // In debug builds, returns true if LLAMINAR_PARITY_CAPTURE=1
        return PipelineSnapshotManager::instance().isEnabled();
    }

    // ==================== PipelineFactory ====================

    PipelineFactory &PipelineFactory::instance()
    {
        static PipelineFactory inst;
        return inst;
    }

    void PipelineFactory::registerCreator(const std::string &arch, CreateFn fn)
    {
        if (!fn)
        {
            LOG_WARN("PipelineFactory: attempted to register null creator for arch=" << arch);
            return;
        }
        for (const auto &p : creators_)
        {
            if (p.first == arch)
            {
                LOG_WARN("PipelineFactory: creator already registered for arch=" << arch);
                return; // idempotent
            }
        }
        creators_.push_back({arch, fn});
        LOG_INFO("PipelineFactory: registered pipeline arch='" << arch << "'");
    }

    std::unique_ptr<AbstractPipeline> PipelineFactory::create(const ModelConfig &cfg) const
    {
        for (const auto &p : creators_)
        {
            if (p.first == cfg.architecture)
            {
                return p.second(cfg);
            }
        }
        LOG_ERROR("PipelineFactory: no creator registered for arch='" << cfg.architecture << "'");
        return nullptr;
    }
} // namespace llaminar
