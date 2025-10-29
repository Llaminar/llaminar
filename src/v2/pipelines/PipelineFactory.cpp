/**
 * @file PipelineFactory.cpp
 * @brief Factory implementation for architecture-specific pipelines
 * @author David Sanftenberg
 */

#include "PipelineFactory.h"
#include "../utils/Logger.h"
#include <iostream>

namespace llaminar2
{

    PipelineFactory &PipelineFactory::instance()
    {
        static PipelineFactory inst;
        return inst;
    }

    void PipelineFactory::registerCreator(const std::string &architecture, CreateFn creator)
    {
        if (!creator)
        {
            LOG_ERROR("[PipelineFactory] Warning: Attempted to register null creator for architecture '" << architecture << "'");
            return;
        }

        auto it = creators_.find(architecture);
        if (it != creators_.end())
        {
            LOG_ERROR("[PipelineFactory] Warning: Creator already registered for architecture '" << architecture << "' (ignoring duplicate)");
            return;
        }

        creators_[architecture] = creator;
        LOG_INFO("[PipelineFactory] Registered pipeline for architecture '" << architecture << "'");
    }

    std::unique_ptr<PipelineBase> PipelineFactory::create(
        const std::string &architecture,
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        int device_idx,
        const PipelineConfig &config) const
    {
        auto it = creators_.find(architecture);
        if (it == creators_.end())
        {
            LOG_ERROR("[PipelineFactory] Error: No pipeline registered for architecture '" << architecture << "'");
            std::string supported_list = "[PipelineFactory] Supported architectures: ";
            bool first = true;
            for (const auto &kv : creators_)
            {
                if (!first)
                    supported_list += ", ";
                supported_list += kv.first;
                first = false;
            }
            LOG_ERROR(supported_list);
            return nullptr;
        }

        return it->second(model_ctx, mpi_ctx, device_idx, config);
    }

    bool PipelineFactory::isSupported(const std::string &architecture) const
    {
        return creators_.find(architecture) != creators_.end();
    }

    std::vector<std::string> PipelineFactory::supportedArchitectures() const
    {
        std::vector<std::string> result;
        result.reserve(creators_.size());
        for (const auto &kv : creators_)
        {
            result.push_back(kv.first);
        }
        return result;
    }

} // namespace llaminar2
