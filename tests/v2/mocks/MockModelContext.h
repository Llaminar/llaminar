/**
 * @file MockModelContext.h
 * @brief Mock model context for unit testing without GGUF files
 *
 * This mock enables:
 * - Testing inference pipelines without actual model files
 * - Complete model configuration with composed mocks
 * - Presets for common model architectures (Qwen2-0.5B, Qwen2-7B, etc.)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "interfaces/IModelContext.h"
#include "MockModelLoader.h"
#include "MockWeightManager.h"
#include <memory>
#include <string>
#include <stdexcept>

namespace llaminar2::test
{

    // Forward declaration
    class MockModelContextBuilder;

    /**
     * @brief Model preset configurations for quick setup
     */
    enum class ModelPreset
    {
        MINIMAL,   // 1 layer, 128 hidden dim - smallest valid transformer
        QWEN2_05B, // 24 layers, 896 hidden dim
        QWEN2_7B,  // 28 layers, 3584 hidden dim
        LLAMA3_8B  // 32 layers, 4096 hidden dim
    };

    /**
     * @brief Mock model context for unit testing
     *
     * Composes MockModelLoader and MockWeightManager to provide a complete
     * model context for testing inference pipelines without GGUF files.
     *
     * Features:
     * - Configurable model hyperparameters
     * - Pre-built tensors and weights with random or specified data
     * - Support for various tensor types (FP32, Q4_0, Q8_0, IQ4_NL)
     * - Presets for common model configurations
     *
     * Usage:
     * @code
     * // Simple usage with presets
     * auto ctx = MockModelContext::createQwen2_05B();
     * int num_layers = ctx->blockCount();
     * auto embd = ctx->getWeightForDevice("token_embd.weight");
     *
     * // Custom model with builder
     * auto ctx = MockModelContextBuilder()
     *     .setArchitecture("qwen2")
     *     .setBlockCount(24)
     *     .setEmbeddingLength(896)
     *     .addWeight("token_embd.weight", createFP32Tensor({151936, 896}))
     *     .build();
     * @endcode
     */
    class MockModelContext : public IModelContext
    {
    public:
        // =========================================================================
        // Presets for Common Model Configurations
        // =========================================================================

        /**
         * @brief Create mock for Qwen2-0.5B architecture
         *
         * Config: 24 layers, 896 hidden dim, 14 heads, 2 KV heads, 4864 FFN dim
         * Includes minimal weights for basic testing.
         */
        static std::shared_ptr<MockModelContext> createQwen2_05B();

        /**
         * @brief Create mock for Qwen2-7B architecture
         *
         * Config: 28 layers, 3584 hidden dim, 28 heads, 4 KV heads, 18944 FFN dim
         */
        static std::shared_ptr<MockModelContext> createQwen2_7B();

        /**
         * @brief Create mock for Llama-3-8B architecture
         *
         * Config: 32 layers, 4096 hidden dim, 32 heads, 8 KV heads, 14336 FFN dim
         */
        static std::shared_ptr<MockModelContext> createLlama3_8B();

        /**
         * @brief Create minimal mock for testing (single layer)
         *
         * Config: 1 layer, 128 hidden dim, 4 heads, 2 KV heads, 512 FFN dim
         * Smallest valid transformer for fast unit tests.
         */
        static std::shared_ptr<MockModelContext> createMinimal();

        /**
         * @brief Create mock from preset
         * @param preset Model preset configuration
         */
        static std::shared_ptr<MockModelContext> createFromPreset(ModelPreset preset);

        // =========================================================================
        // Construction
        // =========================================================================

        MockModelContext();
        ~MockModelContext() override = default;

        // =========================================================================
        // IModelContext Implementation
        // =========================================================================

        const std::string &path() const override { return model_path_; }
        const std::string &architecture() const override;

        std::shared_ptr<IModelLoader> loader() override;
        std::shared_ptr<IWeightManager> weightManager() override;

        int blockCount() const override;
        int embeddingLength() const override;
        int headCount() const override;
        int headCountKV() const override;
        int vocabSize() const override;
        int contextLength() const override;
        int feedForwardLength() const override;
        int keyLength() const override;

        std::shared_ptr<TensorBase> getWeightForDevice(
            const std::string &name,
            DeviceId device = DeviceId::cpu()) override;

        bool hasTensor(const std::string &name) const override;

        // =========================================================================
        // Test Configuration API (for use by Builder and tests)
        // =========================================================================

        /**
         * @brief Set model file path (for display purposes)
         */
        void setPath(const std::string &path) { model_path_ = path; }

        /**
         * @brief Get mutable reference to underlying MockModelLoader
         */
        MockModelLoader &mockLoader() { return *mock_loader_; }

        /**
         * @brief Get mutable reference to underlying MockWeightManager
         */
        MockWeightManager &mockWeightManager() { return *mock_weight_manager_; }

        /**
         * @brief Get shared pointer to MockModelLoader
         */
        std::shared_ptr<MockModelLoader> mockLoaderPtr() { return mock_loader_; }

        /**
         * @brief Get shared pointer to MockWeightManager
         */
        std::shared_ptr<MockWeightManager> mockWeightManagerPtr() { return mock_weight_manager_; }

        // =========================================================================
        // Direct Configuration Helpers (delegate to MockModelLoader)
        // =========================================================================

        void setArchitecture(const std::string &arch);
        void setBlockCount(uint64_t count);
        void setEmbeddingLength(uint64_t length);
        void setHeadCount(uint64_t count);
        void setHeadCountKV(uint64_t count);
        void setVocabSize(uint64_t size);
        void setContextLength(uint64_t length);
        void setFeedForwardLength(uint64_t length);
        void setRopeTheta(float theta);
        void setRmsNormEps(float eps);

        // =========================================================================
        // Test Inspection API
        // =========================================================================

        /**
         * @brief Get total number of getWeight calls
         */
        size_t getWeightCallCount() const;

        /**
         * @brief Get total number of loadTensor calls
         */
        size_t loadTensorCallCount() const;

        /**
         * @brief Reset all counters
         */
        void resetCounters();

    private:
        std::string model_path_ = "mock_model.gguf";
        std::shared_ptr<MockModelLoader> mock_loader_;
        std::shared_ptr<MockWeightManager> mock_weight_manager_;

        // Cached architecture string (from loader)
        mutable std::string cached_architecture_;
    };

    /**
     * @brief Fluent builder for MockModelContext
     *
     * Example:
     * @code
     * auto ctx = MockModelContextBuilder()
     *     .setArchitecture("qwen2")
     *     .setBlockCount(24)
     *     .setEmbeddingLength(896)
     *     .setHeadCount(14)
     *     .setHeadCountKV(2)
     *     .setVocabSize(151936)
     *     .addWeight("token_embd.weight", create_fp32_tensor({151936, 896}))
     *     .addShardedWeight("blk.0.attn_q.weight", create_q4_0_tensor({896, 896}), ShardingMode::COLUMN_PARALLEL)
     *     .build();
     * @endcode
     */
    class MockModelContextBuilder
    {
    public:
        MockModelContextBuilder();

        // =========================================================================
        // Preset Configuration
        // =========================================================================

        /**
         * @brief Use a model preset as base configuration
         *
         * Sets architecture, hyperparameters, and basic weights.
         * Additional weights can be added after calling this.
         */
        MockModelContextBuilder &usePreset(ModelPreset preset);

        // =========================================================================
        // Model Configuration (delegates to MockModelLoader)
        // =========================================================================

        MockModelContextBuilder &setPath(const std::string &path);
        MockModelContextBuilder &setArchitecture(const std::string &arch);
        MockModelContextBuilder &setBlockCount(uint64_t count);
        MockModelContextBuilder &setEmbeddingLength(uint64_t length);
        MockModelContextBuilder &setHeadCount(uint64_t count);
        MockModelContextBuilder &setHeadCountKV(uint64_t count);
        MockModelContextBuilder &setVocabSize(uint64_t size);
        MockModelContextBuilder &setContextLength(uint64_t length);
        MockModelContextBuilder &setFeedForwardLength(uint64_t length);
        MockModelContextBuilder &setRopeTheta(float theta);
        MockModelContextBuilder &setRmsNormEps(float eps);

        // =========================================================================
        // Tensor Addition (adds to both loader and weight manager)
        // =========================================================================

        /**
         * @brief Add a pre-built tensor/weight
         * @param name Tensor name
         * @param tensor Tensor to add
         */
        MockModelContextBuilder &addTensor(const std::string &name, std::shared_ptr<TensorBase> tensor);

        /**
         * @brief Add a weight with specified sharding mode
         */
        MockModelContextBuilder &addShardedWeight(
            const std::string &name,
            std::shared_ptr<TensorBase> tensor,
            ShardingMode mode);

        /**
         * @brief Add FP32 tensor with random data
         */
        MockModelContextBuilder &addFP32RandomTensor(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);

        /**
         * @brief Add FP32 tensor filled with zeros
         */
        MockModelContextBuilder &addFP32ZerosTensor(const std::string &name, const std::vector<size_t> &shape);

        /**
         * @brief Add FP32 tensor filled with ones
         */
        MockModelContextBuilder &addFP32OnesTensor(const std::string &name, const std::vector<size_t> &shape);

        /**
         * @brief Add Q4_0 tensor with random quantized data
         */
        MockModelContextBuilder &addQ4_0RandomTensor(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);

        /**
         * @brief Add Q8_0 tensor with random quantized data
         */
        MockModelContextBuilder &addQ8_0RandomTensor(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);

        /**
         * @brief Add IQ4_NL tensor with random quantized data
         */
        MockModelContextBuilder &addIQ4_NLRandomTensor(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);

        // =========================================================================
        // Weight Sharding Configuration (delegates to MockWeightManager)
        // =========================================================================

        MockModelContextBuilder &setStrategy(WeightDistributionStrategy strategy);
        MockModelContextBuilder &setSharded(const std::string &name, ShardingMode mode);
        MockModelContextBuilder &setReplicated(const std::string &name);
        MockModelContextBuilder &setColumnParallel(const std::string &name);
        MockModelContextBuilder &setRowParallel(const std::string &name);
        MockModelContextBuilder &setInputParallel(const std::string &name);
        MockModelContextBuilder &setNonGemm(const std::string &name);

        // =========================================================================
        // Layer Presets (adds all tensors for a complete layer)
        // =========================================================================

        /**
         * @brief Add embedding layer weights
         * Uses vocab_size and embedding_length from configuration.
         */
        MockModelContextBuilder &addEmbeddingLayer();

        /**
         * @brief Add a complete transformer layer
         * @param layer_idx Layer index (0-based)
         */
        MockModelContextBuilder &addTransformerLayer(int layer_idx);

        /**
         * @brief Add output layer (final norm + LM head)
         */
        MockModelContextBuilder &addOutputLayer();

        /**
         * @brief Add all layers (embedding + transformers + output)
         * Adds block_count transformer layers.
         */
        MockModelContextBuilder &addAllLayers();

        // =========================================================================
        // Build
        // =========================================================================

        /**
         * @brief Build the MockModelContext
         * @return Shared pointer to configured mock
         */
        std::shared_ptr<MockModelContext> build();

    private:
        std::shared_ptr<MockModelContext> mock_;

        // Track configuration for layer presets
        std::string path_ = "mock_model.gguf";
        std::string architecture_ = "qwen2";
        uint64_t block_count_ = 1;
        uint64_t embedding_length_ = 128;
        uint64_t head_count_ = 4;
        uint64_t head_count_kv_ = 2;
        uint64_t vocab_size_ = 32000;
        uint64_t context_length_ = 2048;
        uint64_t ffn_dim_ = 512; // Computed from embedding_length for Qwen2
        float rope_theta_ = 10000.0f;
        float rms_norm_eps_ = 1e-6f;

        void applyPreset(ModelPreset preset);
        void configureQwen2_05B();
        void configureQwen2_7B();
        void configureLlama3_8B();
        void configureMinimal();
    };

    // =============================================================================
    // IMPLEMENTATION
    // =============================================================================

    inline MockModelContext::MockModelContext()
        : mock_loader_(std::make_shared<MockModelLoader>()),
          mock_weight_manager_(std::make_shared<MockWeightManager>())
    {
        // Set loader as loaded by default
        mock_loader_->setLoaded(true);
    }

    inline const std::string &MockModelContext::architecture() const
    {
        cached_architecture_ = mock_loader_->architecture();
        return cached_architecture_;
    }

    inline std::shared_ptr<IModelLoader> MockModelContext::loader()
    {
        return mock_loader_;
    }

    inline std::shared_ptr<IWeightManager> MockModelContext::weightManager()
    {
        return mock_weight_manager_;
    }

    inline int MockModelContext::blockCount() const
    {
        return static_cast<int>(mock_loader_->blockCount());
    }

    inline int MockModelContext::embeddingLength() const
    {
        return static_cast<int>(mock_loader_->embeddingLength());
    }

    inline int MockModelContext::headCount() const
    {
        return static_cast<int>(mock_loader_->headCount());
    }

    inline int MockModelContext::headCountKV() const
    {
        return static_cast<int>(mock_loader_->headCountKV());
    }

    inline int MockModelContext::vocabSize() const
    {
        return static_cast<int>(mock_loader_->vocabSize());
    }

    inline int MockModelContext::contextLength() const
    {
        return static_cast<int>(mock_loader_->contextLength());
    }

    inline int MockModelContext::feedForwardLength() const
    {
        return static_cast<int>(mock_loader_->feedForwardLength());
    }

    inline int MockModelContext::keyLength() const
    {
        return static_cast<int>(mock_loader_->keyLength());
    }

    inline std::shared_ptr<TensorBase> MockModelContext::getWeightForDevice(
        const std::string &name,
        DeviceId device)
    {
        return mock_weight_manager_->getWeightForDevice(name, device);
    }

    inline bool MockModelContext::hasTensor(const std::string &name) const
    {
        return mock_loader_->hasTensor(name);
    }

    inline void MockModelContext::setArchitecture(const std::string &arch)
    {
        mock_loader_->setArchitecture(arch);
    }

    inline void MockModelContext::setBlockCount(uint64_t count)
    {
        mock_loader_->setBlockCount(count);
    }

    inline void MockModelContext::setEmbeddingLength(uint64_t length)
    {
        mock_loader_->setEmbeddingLength(length);
    }

    inline void MockModelContext::setHeadCount(uint64_t count)
    {
        mock_loader_->setHeadCount(count);
    }

    inline void MockModelContext::setHeadCountKV(uint64_t count)
    {
        mock_loader_->setHeadCountKV(count);
    }

    inline void MockModelContext::setVocabSize(uint64_t size)
    {
        mock_loader_->setVocabSize(size);
    }

    inline void MockModelContext::setContextLength(uint64_t length)
    {
        mock_loader_->setContextLength(length);
    }

    inline void MockModelContext::setFeedForwardLength(uint64_t length)
    {
        mock_loader_->setFeedForwardLength(length);
    }

    inline void MockModelContext::setRopeTheta(float theta)
    {
        mock_loader_->setRopeTheta(theta);
    }

    inline void MockModelContext::setRmsNormEps(float eps)
    {
        mock_loader_->setRmsNormEps(eps);
    }

    inline size_t MockModelContext::getWeightCallCount() const
    {
        return mock_weight_manager_->getWeightCallCount();
    }

    inline size_t MockModelContext::loadTensorCallCount() const
    {
        return mock_loader_->loadTensorCallCount();
    }

    inline void MockModelContext::resetCounters()
    {
        mock_loader_->resetCounters();
        mock_weight_manager_->resetCounters();
    }

    // =============================================================================
    // PRESET FACTORY METHODS
    // =============================================================================

    inline std::shared_ptr<MockModelContext> MockModelContext::createQwen2_05B()
    {
        // Metadata-only preset - no tensors created for fast unit tests
        // Use MockModelContextBuilder().usePreset(...).addAllLayers().build() for tests needing tensors
        return MockModelContextBuilder()
            .usePreset(ModelPreset::QWEN2_05B)
            .build();
    }

    inline std::shared_ptr<MockModelContext> MockModelContext::createQwen2_7B()
    {
        // Metadata-only preset - no tensors created for fast unit tests
        return MockModelContextBuilder()
            .usePreset(ModelPreset::QWEN2_7B)
            .build();
    }

    inline std::shared_ptr<MockModelContext> MockModelContext::createLlama3_8B()
    {
        // Metadata-only preset - no tensors created for fast unit tests
        return MockModelContextBuilder()
            .usePreset(ModelPreset::LLAMA3_8B)
            .build();
    }

    inline std::shared_ptr<MockModelContext> MockModelContext::createMinimal()
    {
        // Metadata-only preset - minimal vocab so tensors are small if needed
        return MockModelContextBuilder()
            .usePreset(ModelPreset::MINIMAL)
            .build();
    }

    inline std::shared_ptr<MockModelContext> MockModelContext::createFromPreset(ModelPreset preset)
    {
        // Metadata-only preset - no tensors created for fast unit tests
        return MockModelContextBuilder()
            .usePreset(preset)
            .build();
    }

    // =============================================================================
    // BUILDER IMPLEMENTATION
    // =============================================================================

    inline MockModelContextBuilder::MockModelContextBuilder()
        : mock_(std::make_shared<MockModelContext>()) {}

    inline MockModelContextBuilder &MockModelContextBuilder::usePreset(ModelPreset preset)
    {
        applyPreset(preset);
        return *this;
    }

    inline void MockModelContextBuilder::applyPreset(ModelPreset preset)
    {
        switch (preset)
        {
        case ModelPreset::QWEN2_05B:
            configureQwen2_05B();
            break;
        case ModelPreset::QWEN2_7B:
            configureQwen2_7B();
            break;
        case ModelPreset::LLAMA3_8B:
            configureLlama3_8B();
            break;
        case ModelPreset::MINIMAL:
            configureMinimal();
            break;
        }
    }

    inline void MockModelContextBuilder::configureQwen2_05B()
    {
        architecture_ = "qwen2";
        block_count_ = 24;
        embedding_length_ = 896;
        head_count_ = 14;
        head_count_kv_ = 2;
        vocab_size_ = 151936;
        context_length_ = 32768;
        ffn_dim_ = 4864;
        rope_theta_ = 1000000.0f;
        rms_norm_eps_ = 1e-6f;

        setArchitecture(architecture_);
        setBlockCount(block_count_);
        setEmbeddingLength(embedding_length_);
        setHeadCount(head_count_);
        setHeadCountKV(head_count_kv_);
        setVocabSize(vocab_size_);
        setContextLength(context_length_);
        setFeedForwardLength(ffn_dim_);
        setRopeTheta(rope_theta_);
        setRmsNormEps(rms_norm_eps_);
    }

    inline void MockModelContextBuilder::configureQwen2_7B()
    {
        architecture_ = "qwen2";
        block_count_ = 28;
        embedding_length_ = 3584;
        head_count_ = 28;
        head_count_kv_ = 4;
        vocab_size_ = 152064;
        context_length_ = 131072;
        ffn_dim_ = 18944;
        rope_theta_ = 1000000.0f;
        rms_norm_eps_ = 1e-6f;

        setArchitecture(architecture_);
        setBlockCount(block_count_);
        setEmbeddingLength(embedding_length_);
        setHeadCount(head_count_);
        setHeadCountKV(head_count_kv_);
        setVocabSize(vocab_size_);
        setContextLength(context_length_);
        setFeedForwardLength(ffn_dim_);
        setRopeTheta(rope_theta_);
        setRmsNormEps(rms_norm_eps_);
    }

    inline void MockModelContextBuilder::configureLlama3_8B()
    {
        architecture_ = "llama";
        block_count_ = 32;
        embedding_length_ = 4096;
        head_count_ = 32;
        head_count_kv_ = 8;
        vocab_size_ = 128256;
        context_length_ = 8192;
        ffn_dim_ = 14336;
        rope_theta_ = 500000.0f;
        rms_norm_eps_ = 1e-5f;

        setArchitecture(architecture_);
        setBlockCount(block_count_);
        setEmbeddingLength(embedding_length_);
        setHeadCount(head_count_);
        setHeadCountKV(head_count_kv_);
        setVocabSize(vocab_size_);
        setContextLength(context_length_);
        setFeedForwardLength(ffn_dim_);
        setRopeTheta(rope_theta_);
        setRmsNormEps(rms_norm_eps_);
    }

    inline void MockModelContextBuilder::configureMinimal()
    {
        architecture_ = "qwen2";
        block_count_ = 1;
        embedding_length_ = 128;
        head_count_ = 4;
        head_count_kv_ = 2;
        vocab_size_ = 1000;
        context_length_ = 512;
        ffn_dim_ = 512;
        rope_theta_ = 10000.0f;
        rms_norm_eps_ = 1e-6f;

        setArchitecture(architecture_);
        setBlockCount(block_count_);
        setEmbeddingLength(embedding_length_);
        setHeadCount(head_count_);
        setHeadCountKV(head_count_kv_);
        setVocabSize(vocab_size_);
        setContextLength(context_length_);
        setFeedForwardLength(ffn_dim_);
        setRopeTheta(rope_theta_);
        setRmsNormEps(rms_norm_eps_);
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setPath(const std::string &path)
    {
        path_ = path;
        mock_->setPath(path);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setArchitecture(const std::string &arch)
    {
        architecture_ = arch;
        mock_->setArchitecture(arch);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setBlockCount(uint64_t count)
    {
        block_count_ = count;
        mock_->setBlockCount(count);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setEmbeddingLength(uint64_t length)
    {
        embedding_length_ = length;
        mock_->setEmbeddingLength(length);
        // Update FFN dim for Qwen2-style (intermediate_size = hidden_size * 3 * 2 / 3 / 2 ~= 5.5x)
        if (architecture_ == "qwen2")
        {
            ffn_dim_ = length * 11 / 2; // ~5.5x hidden dim
        }
        else
        {
            ffn_dim_ = length * 4; // Standard 4x for other architectures
        }
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setHeadCount(uint64_t count)
    {
        head_count_ = count;
        mock_->setHeadCount(count);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setHeadCountKV(uint64_t count)
    {
        head_count_kv_ = count;
        mock_->setHeadCountKV(count);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setVocabSize(uint64_t size)
    {
        vocab_size_ = size;
        mock_->setVocabSize(size);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setContextLength(uint64_t length)
    {
        context_length_ = length;
        mock_->setContextLength(length);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setFeedForwardLength(uint64_t length)
    {
        ffn_dim_ = length;
        mock_->setFeedForwardLength(length);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setRopeTheta(float theta)
    {
        rope_theta_ = theta;
        mock_->setRopeTheta(theta);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setRmsNormEps(float eps)
    {
        rms_norm_eps_ = eps;
        mock_->setRmsNormEps(eps);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::addTensor(
        const std::string &name,
        std::shared_ptr<TensorBase> tensor)
    {
        mock_->mockLoader().addTensor(name, tensor);
        mock_->mockWeightManager().addWeight(name, tensor);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::addShardedWeight(
        const std::string &name,
        std::shared_ptr<TensorBase> tensor,
        ShardingMode mode)
    {
        mock_->mockLoader().addTensor(name, tensor);
        mock_->mockWeightManager().addWeight(name, tensor);
        mock_->mockWeightManager().setShardingMode(name, mode);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::addFP32RandomTensor(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        // Create tensor once in the loader, then share it with weight manager
        // This avoids duplicating large tensors (critical for vocab-sized tensors)
        mock_->mockLoader().addFP32RandomTensor(name, shape, min, max, seed);
        auto tensor = mock_->mockLoader().loadTensor(name);
        if (tensor)
        {
            mock_->mockWeightManager().addWeight(name, tensor);
        }
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::addFP32ZerosTensor(
        const std::string &name,
        const std::vector<size_t> &shape)
    {
        // Create tensor once in the loader, then share it with weight manager
        mock_->mockLoader().addFP32ZerosTensor(name, shape);
        auto tensor = mock_->mockLoader().loadTensor(name);
        if (tensor)
        {
            mock_->mockWeightManager().addWeight(name, tensor);
        }
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::addFP32OnesTensor(
        const std::string &name,
        const std::vector<size_t> &shape)
    {
        // Create tensor once in the loader, then share it with weight manager
        mock_->mockLoader().addFP32OnesTensor(name, shape);
        auto tensor = mock_->mockLoader().loadTensor(name);
        if (tensor)
        {
            mock_->mockWeightManager().addWeight(name, tensor);
        }
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::addQ4_0RandomTensor(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        // Create tensor once in the loader, then share it with weight manager
        mock_->mockLoader().addQ4_0RandomTensor(name, shape, min, max, seed);
        auto tensor = mock_->mockLoader().loadTensor(name);
        if (tensor)
        {
            mock_->mockWeightManager().addWeight(name, tensor);
        }
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::addQ8_0RandomTensor(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        // Create tensor once in the loader, then share it with weight manager
        mock_->mockLoader().addQ8_0RandomTensor(name, shape, min, max, seed);
        auto tensor = mock_->mockLoader().loadTensor(name);
        if (tensor)
        {
            mock_->mockWeightManager().addWeight(name, tensor);
        }
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::addIQ4_NLRandomTensor(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        mock_->mockLoader().addIQ4_NLRandomTensor(name, shape, min, max, seed);
        // MockWeightManager doesn't have IQ4_NL method, use loader tensor
        auto tensor = mock_->mockLoader().loadTensor(name);
        if (tensor)
        {
            mock_->mockWeightManager().addWeight(name, tensor);
        }
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setStrategy(WeightDistributionStrategy strategy)
    {
        mock_->mockWeightManager().setStrategy(strategy);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setSharded(const std::string &name, ShardingMode mode)
    {
        mock_->mockWeightManager().setShardingMode(name, mode);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setReplicated(const std::string &name)
    {
        mock_->mockWeightManager().setShardingMode(name, ShardingMode::REPLICATE);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setColumnParallel(const std::string &name)
    {
        mock_->mockWeightManager().setShardingMode(name, ShardingMode::COLUMN_PARALLEL);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setRowParallel(const std::string &name)
    {
        mock_->mockWeightManager().setShardingMode(name, ShardingMode::ROW_PARALLEL);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setInputParallel(const std::string &name)
    {
        mock_->mockWeightManager().setShardingMode(name, ShardingMode::INPUT_PARALLEL);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::setNonGemm(const std::string &name)
    {
        mock_->mockWeightManager().setGemmWeight(name, false);
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::addEmbeddingLayer()
    {
        // Token embeddings
        addFP32RandomTensor("token_embd.weight", {vocab_size_, embedding_length_});
        setReplicated("token_embd.weight");
        setNonGemm("token_embd.weight");
        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::addTransformerLayer(int layer_idx)
    {
        std::string prefix = "blk." + std::to_string(layer_idx) + ".";
        size_t head_dim = embedding_length_ / head_count_;
        size_t kv_dim = head_dim * head_count_kv_;

        // Attention norms
        addFP32RandomTensor(prefix + "attn_norm.weight", {embedding_length_});
        setReplicated(prefix + "attn_norm.weight");
        setNonGemm(prefix + "attn_norm.weight");

        // QKV projections (column parallel for tensor parallelism)
        addFP32RandomTensor(prefix + "attn_q.weight", {embedding_length_, embedding_length_});
        setColumnParallel(prefix + "attn_q.weight");

        addFP32RandomTensor(prefix + "attn_k.weight", {kv_dim, embedding_length_});
        setColumnParallel(prefix + "attn_k.weight");

        addFP32RandomTensor(prefix + "attn_v.weight", {kv_dim, embedding_length_});
        setColumnParallel(prefix + "attn_v.weight");

        // Output projection (row parallel)
        addFP32RandomTensor(prefix + "attn_output.weight", {embedding_length_, embedding_length_});
        setRowParallel(prefix + "attn_output.weight");

        // FFN norm
        addFP32RandomTensor(prefix + "ffn_norm.weight", {embedding_length_});
        setReplicated(prefix + "ffn_norm.weight");
        setNonGemm(prefix + "ffn_norm.weight");

        // FFN projections
        addFP32RandomTensor(prefix + "ffn_gate.weight", {ffn_dim_, embedding_length_});
        setColumnParallel(prefix + "ffn_gate.weight");

        addFP32RandomTensor(prefix + "ffn_up.weight", {ffn_dim_, embedding_length_});
        setColumnParallel(prefix + "ffn_up.weight");

        addFP32RandomTensor(prefix + "ffn_down.weight", {embedding_length_, ffn_dim_});
        setInputParallel(prefix + "ffn_down.weight");

        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::addOutputLayer()
    {
        // Final norm
        addFP32RandomTensor("output_norm.weight", {embedding_length_});
        setReplicated("output_norm.weight");
        setNonGemm("output_norm.weight");

        // LM head (output projection)
        addFP32RandomTensor("output.weight", {vocab_size_, embedding_length_});
        setColumnParallel("output.weight");

        return *this;
    }

    inline MockModelContextBuilder &MockModelContextBuilder::addAllLayers()
    {
        addEmbeddingLayer();
        for (uint64_t i = 0; i < block_count_; ++i)
        {
            addTransformerLayer(static_cast<int>(i));
        }
        addOutputLayer();
        return *this;
    }

    inline std::shared_ptr<MockModelContext> MockModelContextBuilder::build()
    {
        return mock_;
    }

} // namespace llaminar2::test
