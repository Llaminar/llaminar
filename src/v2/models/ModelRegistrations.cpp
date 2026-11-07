/**
 * @file ModelRegistrations.cpp
 * @brief Registers all built-in model architectures
 *
 * This is the single registration authority for built-in graph/schema pairs.
 * Architecture-owned source admission and configuration live in models/<arch>;
 * shared loading/planning contracts remain responsible for consumed operands.
 * Registration preserves source identity even when models share native stages.
 */

#include "ModelRegistrations.h"
#include "qwen/QwenStandardGraph.h"
#include "qwen/Qwen2Schema.h"
#include "qwen3/Qwen3Schema.h"
#include "qwen35/Qwen35Graph.h"
#include "qwen35/Qwen35Schema.h"
#include "qwen35moe/Qwen35MoEGraph.h"
#include "qwen35moe/Qwen35MoESchema.h"
#include "qwen4exp/Qwen4ExpGraph.h"
#include "qwen4exp/Qwen4ExpSchema.h"
#include "clover/CloverMoEGraph.h"
#include "clover/CloverMoESchema.h"
#include "../execution/local_execution/graph/GraphBuilderRegistry.h"
#include "../execution/local_execution/graph/SchemaFactoryRegistry.h"

#include <mutex>

namespace llaminar2
{

    /** @brief Install the matching schema and graph owners for each built-in architecture. */
    static void doRegisterBuiltinModels()
    {
        // =================================================================
        // Qwen2 (also serves Qwen3 which reuses QwenStandardGraph)
        // =================================================================
        GraphBuilderRegistry::registerFactory("qwen2",
                                              [](const GraphConfig &cfg, std::shared_ptr<IMPIContext> mpi)
                                              {
                                                  return std::make_shared<QwenStandardGraph>(cfg, std::move(mpi));
                                              });

        GraphBuilderRegistry::registerFactory("qwen3",
                                              [](const GraphConfig &cfg, std::shared_ptr<IMPIContext> mpi)
                                              {
                                                  return std::make_shared<QwenStandardGraph>(cfg, std::move(mpi));
                                              });

        SchemaFactoryRegistry::registerFactory("qwen2",
                                               []()
                                               { return std::make_unique<Qwen2SchemaFactory>(); });

        SchemaFactoryRegistry::registerFactory("qwen3",
                                               []()
                                               { return std::make_unique<Qwen3SchemaFactory>(); });

        // =================================================================
        // Qwen3.5 Dense (hybrid GDN + Full Attention)
        // =================================================================
        GraphBuilderRegistry::registerFactory("qwen35",
                                              [](const GraphConfig &cfg, std::shared_ptr<IMPIContext> mpi)
                                              {
                                                  return std::make_shared<Qwen35Graph>(cfg, std::move(mpi));
                                              });

        SchemaFactoryRegistry::registerFactory("qwen35",
                                               []()
                                               { return std::make_unique<Qwen35SchemaFactory>(); });

        // =================================================================
        // Qwen3.5 MoE (hybrid GDN + Full Attention + Mixture of Experts)
        // =================================================================
        GraphBuilderRegistry::registerFactory("qwen35moe",
                                              [](const GraphConfig &cfg, std::shared_ptr<IMPIContext> mpi)
                                              {
                                                  return std::make_shared<Qwen35MoEGraph>(cfg, std::move(mpi));
                                              });

        SchemaFactoryRegistry::registerFactory("qwen35moe",
                                               []()
                                               { return std::make_unique<Qwen35MoESchemaFactory>(); });

        GraphBuilderRegistry::registerFactory("clover_moe",
            [](const GraphConfig &config, std::shared_ptr<IMPIContext> mpi)
            { return std::make_shared<CloverMoEGraph>(config, std::move(mpi)); });
        SchemaFactoryRegistry::registerFactory("clover_moe",
            []() { return std::make_unique<CloverMoESchemaFactory>(); });

        GraphBuilderRegistry::registerFactory("qwen4exp",
            [](const GraphConfig &config,std::shared_ptr<IMPIContext> mpi)
            { return std::make_shared<Qwen4ExpGraph>(config,std::move(mpi)); });
        SchemaFactoryRegistry::registerFactory("qwen4exp",
            [](){return std::make_unique<Qwen4ExpSchemaFactory>();});

        // =================================================================
        // Future models: add registration calls here
        // =================================================================
        // GraphBuilderRegistry::registerFactory("llama", ...);
        // SchemaFactoryRegistry::registerFactory("llama", ...);
    }

    void registerBuiltinModels()
    {
        static std::once_flag s_flag;
        std::call_once(s_flag, doRegisterBuiltinModels);
    }

} // namespace llaminar2
