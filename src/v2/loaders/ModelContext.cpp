/**
 * @file ModelContext.cpp
 * @brief Model context implementation
 * @author David Sanftenberg
 */

#include "ModelContext.h"
#include "../utils/Logger.h"
#include <exception>
#include <iostream>
#include <limits>

namespace llaminar2
{

    /**
     * @brief Internal wrapper to expose ModelLoader (by reference) as IModelLoader shared_ptr
     *
     * This wrapper allows ModelContext to return IModelLoader interface while the actual
     * ModelLoader is owned by value within ModelContext. The wrapper holds a reference
     * and relies on the ModelContext outliving any use of the interface pointer.
     */
    class ModelLoaderInterfaceWrapper : public IModelLoader
    {
    public:
        explicit ModelLoaderInterfaceWrapper(ModelLoader &loader) : loader_(loader) {}

        bool isLoaded() const override { return loader_.isLoaded(); }

        std::shared_ptr<TensorBase> loadTensor(
            const std::string &name,
            DeviceId device,
            WeightPrecision weight_precision) override
        {
            return loader_.loadTensor(name, device, weight_precision);
        }

        std::shared_ptr<TensorBase> loadTensorRowSlice(
            const std::string &name,
            size_t row_start, size_t row_end,
            DeviceId device,
            WeightPrecision weight_precision) override
        {
            return loader_.loadTensorRowSlice(name, row_start, row_end, device, weight_precision);
        }

        std::shared_ptr<TensorBase> loadTensorColumnSlice(
            const std::string &name,
            size_t col_start, size_t col_end,
            DeviceId device,
            WeightPrecision weight_precision) override
        {
            return loader_.loadTensorColumnSlice(name, col_start, col_end, device, weight_precision);
        }

        std::shared_ptr<TensorBase> loadTensorExpertSlice(
            const std::string &name,
            size_t expert_start, size_t expert_end,
            DeviceId device,
            WeightPrecision weight_precision) override
        {
            return loader_.loadTensorExpertSlice(name, expert_start, expert_end, device, weight_precision);
        }

        bool hasTensor(const std::string &name) const override { return loader_.hasTensor(name); }
        std::vector<std::string> tensorNames() const override { return loader_.tensorNames(); }
        std::string architecture() const override { return loader_.architecture(); }
        size_t tensorCount() const override { return loader_.tensorCount(); }
        size_t totalBytes() const override { return loader_.totalBytes(); }

        int getInt(const std::string &key, int default_val) const override
        {
            return loader_.getInt(key, default_val);
        }
        uint64_t getUInt64(const std::string &key, uint64_t default_val) const override
        {
            return loader_.getUInt64(key, default_val);
        }
        float getFloat(const std::string &key, float default_val) const override
        {
            return loader_.getFloat(key, default_val);
        }
        std::string getString(const std::string &key, const std::string &default_val) const override
        {
            return loader_.getString(key, default_val);
        }

        uint64_t blockCount() const override { return loader_.blockCount(); }
        uint64_t embeddingLength() const override { return loader_.embeddingLength(); }
        uint64_t headCount() const override { return loader_.headCount(); }
        uint64_t headCountKV() const override { return loader_.headCountKV(); }
        uint64_t vocabSize() const override { return loader_.vocabSize(); }
        uint64_t contextLength() const override { return loader_.contextLength(); }
        uint64_t feedForwardLength() const override { return loader_.feedForwardLength(); }
        uint64_t keyLength() const override { return loader_.keyLength(); }
        float ropeTheta() const override { return loader_.ropeTheta(); }
        float rmsNormEps() const override { return loader_.rmsNormEps(); }

        std::optional<std::vector<size_t>> getTensorShape(const std::string &name) const override
        {
            return loader_.getTensorShape(name);
        }

    private:
        ModelLoader &loader_;
    };

    ModelContext::ModelContext(const std::string &model_path,
                               std::shared_ptr<IMPIContext> mpi_ctx,
                               std::shared_ptr<WeightPlacementMap> placement_map,
                               TensorFactory *factory,
                               WeightDistributionStrategy strategy)
        : model_path_(model_path), loader_(factory)
    {
        // WeightManager will be created after model is loaded (see create())
        // placement_map is passed through to WeightManager
    }

    std::shared_ptr<ModelContext> ModelContext::create(
        const std::string &model_path,
        std::shared_ptr<IMPIContext> mpi_ctx,
        std::shared_ptr<WeightPlacementMap> placement_map,
        TensorFactory *factory,
        WeightDistributionStrategy strategy,
        WeightPrecision weight_precision)
    {
        // Create TensorFactory from MPI context if not provided
        // This ensures proper NUMA-aware allocation and avoids the ModelLoader
        // creating an internal single-rank MPI context
        std::unique_ptr<TensorFactory> owned_factory;
        if (!factory && mpi_ctx)
        {
            owned_factory = std::make_unique<TensorFactory>(*mpi_ctx);
            factory = owned_factory.get();
        }

        auto ctx = std::shared_ptr<ModelContext>(
            new ModelContext(model_path, mpi_ctx, placement_map, factory, strategy));

        // Store owned factory so it lives as long as the context
        if (owned_factory)
        {
            ctx->owned_test_factory_ = std::move(owned_factory);
        }

        // Load model metadata
        try
        {
            ctx->loader_.loadModel(model_path);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[ModelContext] Failed to load model: " << model_path << " (" << e.what() << ")");
            return nullptr;
        }

        // Create WeightManager with loaded model and placement map
        ctx->weight_manager_ = std::make_shared<WeightManager>(
            ctx->loader_, mpi_ctx, placement_map, strategy, weight_precision);

        // Set up interface wrappers
        ctx->loader_interface_ = std::make_shared<ModelLoaderInterfaceWrapper>(ctx->loader_);

        return ctx;
    }

    std::shared_ptr<ModelContext> ModelContext::create(
        const std::string &model_path,
        const ModelContextConfig &config)
    {
        // Validate config
        auto errors = config.validate();
        if (!errors.empty())
        {
            for (const auto &e : errors)
            {
                LOG_ERROR("[ModelContext] Config error: " << e);
            }
            return nullptr;
        }

        // Determine strategy - auto-select if layer partitioned
        WeightDistributionStrategy strategy = config.strategy;
        if (config.isLayerPartitioned() && strategy == WeightDistributionStrategy::REPLICATED)
        {
            strategy = WeightDistributionStrategy::LAYER_PARTITIONED;
        }

        // Create TensorFactory from MPI context if not provided
        std::unique_ptr<TensorFactory> owned_factory;
        TensorFactory *factory = config.factory;
        if (!factory && config.mpi_ctx)
        {
            owned_factory = std::make_unique<TensorFactory>(*config.mpi_ctx);
            factory = owned_factory.get();
        }

        // Create context
        auto ctx = std::shared_ptr<ModelContext>(
            new ModelContext(model_path, config.mpi_ctx, config.placement_map, factory, strategy));

        // Store owned factory so it lives as long as the context
        if (owned_factory)
        {
            ctx->owned_test_factory_ = std::move(owned_factory);
        }

        // Configure mmap before loading (must precede loadModel)
        ctx->loader_.setUseMmap(config.use_mmap);
        ctx->loader_.setSkipMmapCacheEviction(config.skip_mmap_cache_eviction);

        // Load model metadata
        try
        {
            ctx->loader_.loadModel(model_path);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[ModelContext] Failed to load model: " << model_path << " (" << e.what() << ")");
            return nullptr;
        }

        // Create WeightManager
        ctx->weight_manager_ = std::make_shared<WeightManager>(
            ctx->loader_, config.mpi_ctx, config.placement_map, strategy, config.weight_precision);

        // Configure layer range if partitioned
        if (strategy == WeightDistributionStrategy::LAYER_PARTITIONED)
        {
            // ModelContextConfig uses inclusive last_layer (-1 = all layers)
            // WeightManager uses exclusive last_layer [first, last)
            if (config.last_layer >= 0)
            {
                ctx->weight_manager_->setLayerRange(config.first_layer, config.last_layer + 1);
            }
            else if (config.first_layer > 0)
            {
                // Has offset but no upper limit — shouldn't happen, but handle gracefully
                ctx->weight_manager_->setLayerRange(config.first_layer, std::numeric_limits<int>::max());
            }
            // If first_layer==0 and last_layer==-1, skip setLayerRange (all layers pass through)
            ctx->weight_manager_->setHasEmbedding(config.has_embedding);
            ctx->weight_manager_->setHasLmHead(config.has_lm_head);
        }

        // Configure sharding if needed (TODO: implement setShardInfo in WeightManager)
        // if (config.isSharded()) {
        //     ctx->weight_manager_->setShardInfo(
        //         config.shard_index, config.total_shards, config.work_fraction);
        // }

        // Set up interface wrappers
        ctx->loader_interface_ = std::make_shared<ModelLoaderInterfaceWrapper>(ctx->loader_);

        LOG_DEBUG("[ModelContext] Created with " << config.toString());
        return ctx;
    }

    std::shared_ptr<ModelContext> ModelContext::createForPPStage(
        const std::string &model_path,
        int first_layer,
        int last_layer,
        bool has_embedding,
        bool has_lm_head,
        std::shared_ptr<IMPIContext> mpi_ctx,
        WeightPrecision weight_precision)
    {
        // Create context with LAYER_PARTITIONED strategy
        auto ctx = std::shared_ptr<ModelContext>(
            new ModelContext(model_path, mpi_ctx, nullptr, nullptr, WeightDistributionStrategy::LAYER_PARTITIONED));

        // Load model metadata
        try
        {
            ctx->loader_.loadModel(model_path);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[ModelContext] Failed to load model: " << model_path << " (" << e.what() << ")");
            return nullptr;
        }

        // Create WeightManager with LAYER_PARTITIONED strategy
        ctx->weight_manager_ = std::make_shared<WeightManager>(
            ctx->loader_, mpi_ctx, nullptr, WeightDistributionStrategy::LAYER_PARTITIONED, weight_precision);

        // Configure layer range for filtering
        ctx->weight_manager_->setLayerRange(first_layer, last_layer, has_embedding, has_lm_head);

        // Set up interface wrappers
        ctx->loader_interface_ = std::make_shared<ModelLoaderInterfaceWrapper>(ctx->loader_);

        // CRITICAL: Override blockCount() to return the layer count for THIS stage
        // This ensures nested MDO (TP within PP) builds graphs with correct layer count
        ctx->pp_block_count_override_ = last_layer - first_layer;

        LOG_INFO("[ModelContext] Created PP stage context for layers [" << first_layer << ", " << last_layer
                                                                        << "), embedding=" << (has_embedding ? "yes" : "no")
                                                                        << ", lm_head=" << (has_lm_head ? "yes" : "no"));

        return ctx;
    }

    std::shared_ptr<ModelContext> ModelContext::createForTesting(
        const std::string &model_path,
        std::shared_ptr<IMPIContext> mpi_ctx,
        uint32_t block_count)
    {
        // Create TensorFactory from MPI context (if provided) to prevent ModelLoader
        // from creating internal MPI_COMM_NULL context that conflicts with test's MPI_COMM_WORLD
        std::unique_ptr<TensorFactory> owned_factory;
        TensorFactory *factory = nullptr;
        if (mpi_ctx)
        {
            owned_factory = std::make_unique<TensorFactory>(*mpi_ctx);
            factory = owned_factory.get();
        }

        auto ctx = std::shared_ptr<ModelContext>(
            new ModelContext(model_path, mpi_ctx, nullptr, factory, WeightDistributionStrategy::REPLICATED));

        // Store owned factory so it lives as long as the context
        if (owned_factory)
        {
            ctx->owned_test_factory_ = std::move(owned_factory);
        }

        // Initialize minimal valid model structure to prevent accessing uninitialized memory
        ctx->loader_.initializeTestModel(block_count);

        // Set up interface wrappers (no WeightManager for test contexts by default)
        ctx->loader_interface_ = std::make_shared<ModelLoaderInterfaceWrapper>(ctx->loader_);
        // weight_manager_interface_ remains nullptr - tests should use MockModelContext instead

        return ctx;
    }

} // namespace llaminar2
