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
#include "qwen/Qwen2Graph.h"
#include "qwen/Qwen2Schema.h"
#include "qwen3/Qwen3Schema.h"
#include "qwen35/Qwen35Graph.h"
#include "qwen35/Qwen35Schema.h"
#include "../execution/local_execution/graph/GraphBuilderRegistry.h"
#include "../execution/local_execution/graph/SchemaFactoryRegistry.h"

namespace llaminar2
{

    void registerBuiltinModels()
    {
        // =================================================================
        // Qwen2 (also serves Qwen3 which reuses Qwen2Graph)
        // =================================================================
        GraphBuilderRegistry::registerFactory("qwen2",
                                              [](const GraphConfig &cfg, std::shared_ptr<MPIContext> mpi)
                                              {
                                                  return std::make_shared<Qwen2Graph>(cfg, std::move(mpi));
                                              });

        GraphBuilderRegistry::registerFactory("qwen3",
                                              [](const GraphConfig &cfg, std::shared_ptr<MPIContext> mpi)
                                              {
                                                  return std::make_shared<Qwen2Graph>(cfg, std::move(mpi));
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
                                              [](const GraphConfig &cfg, std::shared_ptr<MPIContext> mpi)
                                              {
                                                  return std::make_shared<Qwen35Graph>(cfg, std::move(mpi));
                                              });

        SchemaFactoryRegistry::registerFactory("qwen35",
                                               []()
                                               { return std::make_unique<Qwen35SchemaFactory>(); });

        // =================================================================
        // Future models: add registration calls here
        // =================================================================
        // GraphBuilderRegistry::registerFactory("llama", ...);
        // SchemaFactoryRegistry::registerFactory("llama", ...);
    }

} // namespace llaminar2
