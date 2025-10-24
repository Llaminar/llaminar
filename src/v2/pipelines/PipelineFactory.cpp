/**
 * @file PipelineFactory.cpp
 * @brief Factory implementation for architecture-specific pipelines
 * @author David Sanftenberg
 */

#include "PipelineFactory.h"
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
            std::cerr << "[PipelineFactory] Warning: Attempted to register null creator for architecture '"
                      << architecture << "'" << std::endl;
            return;
        }

        auto it = creators_.find(architecture);
        if (it != creators_.end())
        {
            std::cerr << "[PipelineFactory] Warning: Creator already registered for architecture '"
                      << architecture << "' (ignoring duplicate)" << std::endl;
            return;
        }

        creators_[architecture] = creator;
        std::cout << "[PipelineFactory] Registered pipeline for architecture '" << architecture << "'" << std::endl;
    }

    std::unique_ptr<PipelineBase> PipelineFactory::create(
        const std::string &architecture,
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        int device_idx) const
    {
        auto it = creators_.find(architecture);
        if (it == creators_.end())
        {
            std::cerr << "[PipelineFactory] Error: No pipeline registered for architecture '"
                      << architecture << "'" << std::endl;
            std::cerr << "[PipelineFactory] Supported architectures: ";
            bool first = true;
            for (const auto &kv : creators_)
            {
                if (!first)
                    std::cerr << ", ";
                std::cerr << kv.first;
                first = false;
            }
            std::cerr << std::endl;
            return nullptr;
        }

        return it->second(model_ctx, mpi_ctx, device_idx);
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
