/**
 * @file abstract_pipeline.cpp
 * @brief Implementation of multi-architecture pipeline factory scaffolding.
 */
#include "abstract_pipeline.h"
#include "logger.h"

namespace llaminar
{
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
