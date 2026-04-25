/**
 * @file ModelRegistrations.cpp
 * @brief Registers all built-in model architectures
 *
 * This is the SINGLE place to add a new model architecture.
 * Adding support for a new model requires:
 *   1. Creating files under models/<arch>/
 *   2. Adding registration calls here
 *
 * No edits to execution/, config/, loaders/, or kernels/ needed.
 */

#include "ModelRegistrations.h"
#include "qwen/QwenStandardGraph.h"
#include "qwen/Qwen2Schema.h"
#include "qwen3/Qwen3Schema.h"
#include "qwen35/Qwen35Graph.h"
#include "qwen35/Qwen35Schema.h"
#include "qwen35moe/Qwen35MoEGraph.h"
#include "qwen35moe/Qwen35MoESchema.h"
#include "../execution/local_execution/graph/GraphBuilderRegistry.h"
#include "../execution/local_execution/graph/SchemaFactoryRegistry.h"

#include <mutex>

namespace llaminar2
{

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
