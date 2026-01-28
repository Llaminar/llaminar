/**
 * @file MockModelLoader.h
 * @brief Mock model loader for unit testing without GGUF files
 *
 * This mock enables:
 * - Testing weight loading logic without actual model files
 * - Creating in-memory tensors with controlled data
 * - Simulating different model configurations
 * - Presets for common model architectures (Qwen2-0.5B, Qwen2-7B, etc.)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "interfaces/IModelLoader.h"
#include "tensors/Tensors.h"
#include "backends/DeviceId.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <random>
#include <functional>
#include <stdexcept>

namespace llaminar2::test
{

    /**
     * @brief Builder for MockModelLoader
     *
     * Provides a fluent interface for constructing mock model loaders:
     * @code
     * auto mock = MockModelLoaderBuilder()
     *     .setArchitecture("qwen2")
     *     .setBlockCount(24)
     *     .setEmbeddingLength(896)
     *     .setHeadCount(14)
     *     .setHeadCountKV(2)
     *     .setVocabSize(151936)
     *     .addFP32Tensor("token_embd.weight", {151936, 896})
     *     .addQ4_0Tensor("blk.0.attn_q.weight", {896, 896})
     *     .build();
     * @endcode
     */
    class MockModelLoaderBuilder;

    /**
     * @brief Mock model loader for unit testing
     *
     * Provides in-memory tensors without file I/O for testing:
     * - Configurable model hyperparameters
     * - Pre-built tensors with random or specified data
     * - Support for FP32, Q4_0, Q8_0, IQ4_NL tensors
     * - Row/column slice support for tensor parallelism testing
     *
     * Usage:
     * @code
     * // Simple usage with presets
     * auto mock = MockModelLoader::createQwen2_05B();
     * auto embd = mock->loadTensor("token_embd.weight");
     *
     * // Custom model
     * auto mock = MockModelLoaderBuilder()
     *     .setArchitecture("llama")
     *     .setBlockCount(32)
     *     .addFP32RandomTensor("token_embd.weight", {32000, 4096})
     *     .build();
     * @endcode
     */
    class MockModelLoader : public IModelLoader
    {
    public:
        // =========================================================================
        // Presets for Common Model Configurations
        // =========================================================================

        /**
         * @brief Create mock for Qwen2-0.5B architecture
         *
         * Config: 24 layers, 896 hidden dim, 14 heads, 2 KV heads
         * Creates minimal tensors (embedding, first layer attention/FFN)
         */
        static std::shared_ptr<MockModelLoader> createQwen2_05B();

        /**
         * @brief Create mock for Qwen2-7B architecture
         *
         * Config: 28 layers, 3584 hidden dim, 28 heads, 4 KV heads
         */
        static std::shared_ptr<MockModelLoader> createQwen2_7B();

        /**
         * @brief Create mock for Llama-3-8B architecture
         *
         * Config: 32 layers, 4096 hidden dim, 32 heads, 8 KV heads
         */
        static std::shared_ptr<MockModelLoader> createLlama3_8B();

        /**
         * @brief Create minimal mock for testing (single layer)
         *
         * Config: 1 layer, 128 hidden dim, 4 heads, 2 KV heads
         * Smallest valid transformer for unit tests
         */
        static std::shared_ptr<MockModelLoader> createMinimal();

        // =========================================================================
        // Construction
        // =========================================================================

        MockModelLoader();
        ~MockModelLoader() override = default;

        // =========================================================================
        // IModelLoader Implementation
        // =========================================================================

        bool isLoaded() const override { return loaded_; }

        std::shared_ptr<TensorBase> loadTensor(
            const std::string &name,
            DeviceId device = DeviceId::cpu(),
            WeightPrecision weight_precision = WeightPrecision::NATIVE) override;

        std::shared_ptr<TensorBase> loadTensorRowSlice(
            const std::string &name,
            size_t row_start, size_t row_end,
            DeviceId device = DeviceId::cpu(),
            WeightPrecision weight_precision = WeightPrecision::NATIVE) override;

        std::shared_ptr<TensorBase> loadTensorColumnSlice(
            const std::string &name,
            size_t col_start, size_t col_end,
            DeviceId device = DeviceId::cpu(),
            WeightPrecision weight_precision = WeightPrecision::NATIVE) override;

        bool hasTensor(const std::string &name) const override;
        std::optional<std::vector<size_t>> getTensorShape(const std::string &name) const override;
        std::vector<std::string> tensorNames() const override;

        std::string architecture() const override { return architecture_; }
        size_t tensorCount() const override { return tensors_.size(); }
        size_t totalBytes() const override;

        int getInt(const std::string &key, int default_val = 0) const override;
        uint64_t getUInt64(const std::string &key, uint64_t default_val = 0) const override;
        float getFloat(const std::string &key, float default_val = 0.0f) const override;
        std::string getString(const std::string &key, const std::string &default_val = "") const override;

        uint64_t blockCount() const override { return block_count_; }
        uint64_t embeddingLength() const override { return embedding_length_; }
        uint64_t headCount() const override { return head_count_; }
        uint64_t headCountKV() const override { return head_count_kv_; }
        uint64_t vocabSize() const override { return vocab_size_; }
        uint64_t contextLength() const override { return context_length_; }
        uint64_t feedForwardLength() const override { return feed_forward_length_; }
        float ropeTheta() const override { return rope_theta_; }
        float rmsNormEps() const override { return rms_norm_eps_; }

        // =========================================================================
        // Test Configuration API (for use by Builder and tests)
        // =========================================================================

        void setLoaded(bool loaded) { loaded_ = loaded; }
        void setArchitecture(const std::string &arch) { architecture_ = arch; }
        void setBlockCount(uint64_t count) { block_count_ = count; }
        void setEmbeddingLength(uint64_t length) { embedding_length_ = length; }
        void setHeadCount(uint64_t count) { head_count_ = count; }
        void setHeadCountKV(uint64_t count) { head_count_kv_ = count; }
        void setVocabSize(uint64_t size) { vocab_size_ = size; }
        void setContextLength(uint64_t length) { context_length_ = length; }
        void setFeedForwardLength(uint64_t length) { feed_forward_length_ = length; }
        void setRopeTheta(float theta) { rope_theta_ = theta; }
        void setRmsNormEps(float eps) { rms_norm_eps_ = eps; }
        void setIntParam(const std::string &key, int value) { int_params_[key] = value; }
        void setFloatParam(const std::string &key, float value) { float_params_[key] = value; }
        void setStringParam(const std::string &key, const std::string &value) { string_params_[key] = value; }

        /**
         * @brief Add a pre-built tensor to the mock
         * @param name Tensor name
         * @param tensor Tensor to add (takes ownership via shared_ptr)
         */
        void addTensor(const std::string &name, std::shared_ptr<TensorBase> tensor);

        /**
         * @brief Add FP32 tensor with random data
         * @param name Tensor name
         * @param shape Tensor dimensions
         * @param min Minimum random value
         * @param max Maximum random value
         * @param seed Random seed for reproducibility
         */
        void addFP32RandomTensor(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);

        /**
         * @brief Add FP32 tensor filled with zeros
         */
        void addFP32ZerosTensor(const std::string &name, const std::vector<size_t> &shape);

        /**
         * @brief Add FP32 tensor filled with ones
         */
        void addFP32OnesTensor(const std::string &name, const std::vector<size_t> &shape);

        /**
         * @brief Add Q4_0 tensor with random quantized data
         */
        void addQ4_0RandomTensor(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);

        /**
         * @brief Add Q8_0 tensor with random quantized data
         */
        void addQ8_0RandomTensor(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);

        /**
         * @brief Add IQ4_NL tensor with random quantized data
         */
        void addIQ4_NLRandomTensor(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);

        // =========================================================================
        // Test Inspection API
        // =========================================================================

        /**
         * @brief Get number of times loadTensor was called
         */
        size_t loadTensorCallCount() const { return load_tensor_calls_; }

        /**
         * @brief Get names of tensors that were requested but not found
         */
        const std::vector<std::string> &missingTensorRequests() const { return missing_requests_; }

        /**
         * @brief Reset call counts and missing requests
         */
        void resetCounters();

    private:
        // Model configuration
        bool loaded_ = true;
        std::string architecture_ = "qwen2";
        uint64_t block_count_ = 1;
        uint64_t embedding_length_ = 128;
        uint64_t head_count_ = 4;
        uint64_t head_count_kv_ = 2;
        uint64_t vocab_size_ = 32000;
        uint64_t context_length_ = 2048;
        uint64_t feed_forward_length_ = 512; // Default to 4×embedding_length for minimal preset
        float rope_theta_ = 10000.0f;
        float rms_norm_eps_ = 1e-6f;

        // Custom parameters
        std::unordered_map<std::string, int> int_params_;
        std::unordered_map<std::string, float> float_params_;
        std::unordered_map<std::string, std::string> string_params_;

        // Tensor storage
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> tensors_;
        std::unordered_map<std::string, std::vector<size_t>> tensor_shapes_;

        // Call tracking
        mutable size_t load_tensor_calls_ = 0;
        mutable std::vector<std::string> missing_requests_;

        // Helper for creating slices
        std::shared_ptr<TensorBase> createRowSlice(
            const std::shared_ptr<TensorBase> &tensor,
            const std::vector<size_t> &shape,
            size_t row_start, size_t row_end);

        std::shared_ptr<TensorBase> createColumnSlice(
            const std::shared_ptr<TensorBase> &tensor,
            const std::vector<size_t> &shape,
            size_t col_start, size_t col_end);
    };

    /**
     * @brief Fluent builder for MockModelLoader
     *
     * Example:
     * @code
     * auto mock = MockModelLoaderBuilder()
     *     .setArchitecture("qwen2")
     *     .setBlockCount(24)
     *     .setEmbeddingLength(896)
     *     .addFP32RandomTensor("token_embd.weight", {151936, 896})
     *     .addQ4_0RandomTensor("blk.0.attn_q.weight", {896, 896})
     *     .build();
     * @endcode
     */
    class MockModelLoaderBuilder
    {
    public:
        MockModelLoaderBuilder();

        // Architecture config
        MockModelLoaderBuilder &setArchitecture(const std::string &arch);
        MockModelLoaderBuilder &setBlockCount(uint64_t count);
        MockModelLoaderBuilder &setEmbeddingLength(uint64_t length);
        MockModelLoaderBuilder &setHeadCount(uint64_t count);
        MockModelLoaderBuilder &setHeadCountKV(uint64_t count);
        MockModelLoaderBuilder &setVocabSize(uint64_t size);
        MockModelLoaderBuilder &setContextLength(uint64_t length);
        MockModelLoaderBuilder &setFeedForwardLength(uint64_t length);
        MockModelLoaderBuilder &setRopeTheta(float theta);
        MockModelLoaderBuilder &setRmsNormEps(float eps);

        // Custom parameters
        MockModelLoaderBuilder &setInt(const std::string &key, int value);
        MockModelLoaderBuilder &setFloat(const std::string &key, float value);
        MockModelLoaderBuilder &setString(const std::string &key, const std::string &value);

        // Tensor addition
        MockModelLoaderBuilder &addTensor(const std::string &name, std::shared_ptr<TensorBase> tensor);
        MockModelLoaderBuilder &addFP32Tensor(const std::string &name, const std::vector<size_t> &shape);
        MockModelLoaderBuilder &addFP32RandomTensor(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);
        MockModelLoaderBuilder &addFP32ZerosTensor(const std::string &name, const std::vector<size_t> &shape);
        MockModelLoaderBuilder &addFP32OnesTensor(const std::string &name, const std::vector<size_t> &shape);
        MockModelLoaderBuilder &addQ4_0RandomTensor(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);
        MockModelLoaderBuilder &addQ8_0RandomTensor(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);
        MockModelLoaderBuilder &addIQ4_NLRandomTensor(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);

        // Preset layers (adds all tensors for a transformer layer)
        MockModelLoaderBuilder &addEmbeddingLayer();
        MockModelLoaderBuilder &addTransformerLayer(int layer_idx);
        MockModelLoaderBuilder &addOutputLayer();

        // Build
        std::shared_ptr<MockModelLoader> build();

    private:
        std::shared_ptr<MockModelLoader> mock_;
    };

    // =============================================================================
    // IMPLEMENTATION
    // =============================================================================

    inline MockModelLoader::MockModelLoader() = default;

    inline bool MockModelLoader::hasTensor(const std::string &name) const
    {
        return tensors_.find(name) != tensors_.end();
    }

    inline std::optional<std::vector<size_t>> MockModelLoader::getTensorShape(const std::string &name) const
    {
        auto it = tensor_shapes_.find(name);
        if (it == tensor_shapes_.end())
        {
            // Fall back to getting shape from tensor if it exists
            auto tensor_it = tensors_.find(name);
            if (tensor_it == tensors_.end())
            {
                return std::nullopt;
            }
            return tensor_it->second->shape();
        }
        return it->second;
    }

    inline std::vector<std::string> MockModelLoader::tensorNames() const
    {
        std::vector<std::string> names;
        names.reserve(tensors_.size());
        for (const auto &[name, _] : tensors_)
        {
            names.push_back(name);
        }
        return names;
    }

    inline size_t MockModelLoader::totalBytes() const
    {
        size_t total = 0;
        for (const auto &[_, tensor] : tensors_)
        {
            // Approximate bytes based on element count and typical 4-byte elements
            // For quantized tensors this is an approximation, but sufficient for tests
            total += tensor->numel() * sizeof(float);
        }
        return total;
    }

    inline int MockModelLoader::getInt(const std::string &key, int default_val) const
    {
        // Check custom params first
        auto it = int_params_.find(key);
        if (it != int_params_.end())
            return it->second;

        // Check known hyperparameters
        if (key == "block_count")
            return static_cast<int>(block_count_);
        if (key == "head_count")
            return static_cast<int>(head_count_);
        if (key == "head_count_kv")
            return static_cast<int>(head_count_kv_);

        return default_val;
    }

    inline uint64_t MockModelLoader::getUInt64(const std::string &key, uint64_t default_val) const
    {
        // Check known hyperparameters
        if (key == "block_count")
            return block_count_;
        if (key == "embedding_length")
            return embedding_length_;
        if (key == "context_length")
            return context_length_;
        if (key == "head_count")
            return head_count_;
        if (key == "head_count_kv")
            return head_count_kv_;
        if (key == "vocab_size")
            return vocab_size_;

        // Check custom int params (promote to uint64)
        auto it = int_params_.find(key);
        if (it != int_params_.end())
            return static_cast<uint64_t>(it->second);

        return default_val;
    }

    inline float MockModelLoader::getFloat(const std::string &key, float default_val) const
    {
        auto it = float_params_.find(key);
        if (it != float_params_.end())
            return it->second;

        if (key == "rope_theta")
            return rope_theta_;
        if (key == "rms_norm_eps")
            return rms_norm_eps_;

        return default_val;
    }

    inline std::string MockModelLoader::getString(const std::string &key, const std::string &default_val) const
    {
        auto it = string_params_.find(key);
        if (it != string_params_.end())
            return it->second;

        if (key == "architecture")
            return architecture_;

        return default_val;
    }

    inline void MockModelLoader::addTensor(const std::string &name, std::shared_ptr<TensorBase> tensor)
    {
        tensor_shapes_[name] = {tensor->rows(), tensor->cols()};
        tensors_[name] = std::move(tensor);
    }

    inline void MockModelLoader::addFP32RandomTensor(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(min, max);

        size_t numel = 1;
        for (auto s : shape)
            numel *= s;

        auto tensor = std::make_shared<FP32Tensor>(shape);
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < numel; ++i)
        {
            data[i] = dist(rng);
        }

        addTensor(name, tensor);
    }

    inline void MockModelLoader::addFP32ZerosTensor(const std::string &name, const std::vector<size_t> &shape)
    {
        size_t numel = 1;
        for (auto s : shape)
            numel *= s;

        auto tensor = std::make_shared<FP32Tensor>(shape);
        std::memset(tensor->mutable_data(), 0, numel * sizeof(float));
        addTensor(name, tensor);
    }

    inline void MockModelLoader::addFP32OnesTensor(const std::string &name, const std::vector<size_t> &shape)
    {
        size_t numel = 1;
        for (auto s : shape)
            numel *= s;

        auto tensor = std::make_shared<FP32Tensor>(shape);
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < numel; ++i)
        {
            data[i] = 1.0f;
        }
        addTensor(name, tensor);
    }

    inline void MockModelLoader::addQ4_0RandomTensor(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        constexpr size_t BLOCK_SIZE = 32;

        // Q4_0: 32 elements per block, 18 bytes per block (2 bytes scale + 16 bytes quants)
        size_t rows = shape.size() >= 1 ? shape[0] : 1;
        size_t cols = shape.size() >= 2 ? shape[1] : 1;
        size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t total_blocks = rows * blocks_per_row;

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(min, max);

        // Generate random FP32 data first, then quantize
        std::vector<uint8_t> raw_data(total_blocks * 18); // Q4_0 block = 18 bytes

        // Simple Q4_0 quantization: create valid blocks
        for (size_t b = 0; b < total_blocks; ++b)
        {
            // Generate 32 random values
            float block_vals[BLOCK_SIZE];
            float abs_max = 0.0f;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                block_vals[i] = dist(rng);
                abs_max = std::max(abs_max, std::abs(block_vals[i]));
            }

            // Compute scale (half precision stored as uint16)
            float scale = abs_max / 7.0f; // Q4_0 range is [-8, 7]
            uint16_t scale_fp16 = fp32_to_fp16(scale);

            // Store scale
            uint8_t *block_ptr = raw_data.data() + b * 18;
            std::memcpy(block_ptr, &scale_fp16, 2);

            // Quantize and pack (4 bits per value, 2 values per byte)
            float inv_scale = scale > 0 ? 1.0f / scale : 0.0f;
            for (size_t i = 0; i < BLOCK_SIZE / 2; ++i)
            {
                int q0 = std::clamp(static_cast<int>(std::round(block_vals[2 * i] * inv_scale)) + 8, 0, 15);
                int q1 = std::clamp(static_cast<int>(std::round(block_vals[2 * i + 1] * inv_scale)) + 8, 0, 15);
                block_ptr[2 + i] = static_cast<uint8_t>((q1 << 4) | q0);
            }
        }

        auto tensor = std::make_shared<Q4_0Tensor>(shape, raw_data);
        addTensor(name, tensor);
    }

    inline void MockModelLoader::addQ8_0RandomTensor(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        constexpr size_t BLOCK_SIZE = 32;

        size_t rows = shape.size() >= 1 ? shape[0] : 1;
        size_t cols = shape.size() >= 2 ? shape[1] : 1;
        size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t total_blocks = rows * blocks_per_row;

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(min, max);

        // Q8_0: 32 elements per block, 34 bytes per block (2 bytes scale + 32 bytes quants)
        std::vector<uint8_t> raw_data(total_blocks * 34);

        for (size_t b = 0; b < total_blocks; ++b)
        {
            float block_vals[BLOCK_SIZE];
            float abs_max = 0.0f;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                block_vals[i] = dist(rng);
                abs_max = std::max(abs_max, std::abs(block_vals[i]));
            }

            float scale = abs_max / 127.0f;
            uint16_t scale_fp16 = fp32_to_fp16(scale);

            uint8_t *block_ptr = raw_data.data() + b * 34;
            std::memcpy(block_ptr, &scale_fp16, 2);

            float inv_scale = scale > 0 ? 1.0f / scale : 0.0f;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                int8_t q = std::clamp(static_cast<int>(std::round(block_vals[i] * inv_scale)), -128, 127);
                block_ptr[2 + i] = static_cast<uint8_t>(q);
            }
        }

        auto tensor = std::make_shared<Q8_0Tensor>(shape, raw_data);
        addTensor(name, tensor);
    }

    inline void MockModelLoader::addIQ4_NLRandomTensor(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        constexpr size_t BLOCK_SIZE = 32;

        size_t rows = shape.size() >= 1 ? shape[0] : 1;
        size_t cols = shape.size() >= 2 ? shape[1] : 1;
        size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t total_blocks = rows * blocks_per_row;

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(min, max);

        // IQ4_NL: 32 elements per block, 18 bytes per block
        std::vector<uint8_t> raw_data(total_blocks * 18);

        for (size_t b = 0; b < total_blocks; ++b)
        {
            float block_vals[BLOCK_SIZE];
            float abs_max = 0.0f;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                block_vals[i] = dist(rng);
                abs_max = std::max(abs_max, std::abs(block_vals[i]));
            }

            float scale = abs_max / 7.0f;
            uint16_t scale_fp16 = fp32_to_fp16(scale);

            uint8_t *block_ptr = raw_data.data() + b * 18;
            std::memcpy(block_ptr, &scale_fp16, 2);

            // Simple 4-bit packing (not true IQ4_NL lookup, but structurally valid)
            float inv_scale = scale > 0 ? 1.0f / scale : 0.0f;
            for (size_t i = 0; i < BLOCK_SIZE / 2; ++i)
            {
                int q0 = std::clamp(static_cast<int>(std::round(block_vals[2 * i] * inv_scale)) + 8, 0, 15);
                int q1 = std::clamp(static_cast<int>(std::round(block_vals[2 * i + 1] * inv_scale)) + 8, 0, 15);
                block_ptr[2 + i] = static_cast<uint8_t>((q1 << 4) | q0);
            }
        }

        auto tensor = std::make_shared<IQ4_NLTensor>(shape, raw_data);
        addTensor(name, tensor);
    }

    inline std::shared_ptr<TensorBase> MockModelLoader::loadTensor(
        const std::string &name,
        DeviceId /*device*/,
        WeightPrecision /*weight_precision*/)
    {
        ++load_tensor_calls_;

        auto it = tensors_.find(name);
        if (it == tensors_.end())
        {
            missing_requests_.push_back(name);
            return nullptr;
        }

        return it->second;
    }

    inline std::shared_ptr<TensorBase> MockModelLoader::loadTensorRowSlice(
        const std::string &name,
        size_t row_start, size_t row_end,
        DeviceId /*device*/,
        WeightPrecision /*weight_precision*/)
    {
        ++load_tensor_calls_;

        auto it = tensors_.find(name);
        if (it == tensors_.end())
        {
            missing_requests_.push_back(name);
            return nullptr;
        }

        auto shape_it = tensor_shapes_.find(name);
        if (shape_it == tensor_shapes_.end())
            return nullptr;

        return createRowSlice(it->second, shape_it->second, row_start, row_end);
    }

    inline std::shared_ptr<TensorBase> MockModelLoader::loadTensorColumnSlice(
        const std::string &name,
        size_t col_start, size_t col_end,
        DeviceId /*device*/,
        WeightPrecision /*weight_precision*/)
    {
        ++load_tensor_calls_;

        auto it = tensors_.find(name);
        if (it == tensors_.end())
        {
            missing_requests_.push_back(name);
            return nullptr;
        }

        auto shape_it = tensor_shapes_.find(name);
        if (shape_it == tensor_shapes_.end())
            return nullptr;

        return createColumnSlice(it->second, shape_it->second, col_start, col_end);
    }

    inline std::shared_ptr<TensorBase> MockModelLoader::createRowSlice(
        const std::shared_ptr<TensorBase> &tensor,
        const std::vector<size_t> &shape,
        size_t row_start, size_t row_end)
    {
        // For FP32 tensors only (simplest case for testing)
        auto *fp32 = dynamic_cast<FP32Tensor *>(tensor.get());
        if (!fp32)
            return nullptr;

        size_t cols = shape.size() >= 2 ? shape[1] : 1;
        size_t slice_rows = row_end - row_start;

        auto slice = std::make_shared<FP32Tensor>(std::vector<size_t>{slice_rows, cols});
        const float *src = fp32->data() + row_start * cols;
        float *dst = slice->mutable_data();
        std::memcpy(dst, src, slice_rows * cols * sizeof(float));

        return slice;
    }

    inline std::shared_ptr<TensorBase> MockModelLoader::createColumnSlice(
        const std::shared_ptr<TensorBase> &tensor,
        const std::vector<size_t> &shape,
        size_t col_start, size_t col_end)
    {
        // For FP32 tensors only
        auto *fp32 = dynamic_cast<FP32Tensor *>(tensor.get());
        if (!fp32)
            return nullptr;

        size_t rows = shape.size() >= 1 ? shape[0] : 1;
        size_t cols = shape.size() >= 2 ? shape[1] : 1;
        size_t slice_cols = col_end - col_start;

        auto slice = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, slice_cols});
        const float *src = fp32->data();
        float *dst = slice->mutable_data();

        for (size_t r = 0; r < rows; ++r)
        {
            std::memcpy(dst + r * slice_cols, src + r * cols + col_start, slice_cols * sizeof(float));
        }

        return slice;
    }

    inline void MockModelLoader::resetCounters()
    {
        load_tensor_calls_ = 0;
        missing_requests_.clear();
    }

    // =============================================================================
    // PRESET IMPLEMENTATIONS
    // =============================================================================

    inline std::shared_ptr<MockModelLoader> MockModelLoader::createMinimal()
    {
        return MockModelLoaderBuilder()
            .setArchitecture("qwen2")
            .setBlockCount(1)
            .setEmbeddingLength(128)
            .setHeadCount(4)
            .setHeadCountKV(2)
            .setVocabSize(1000)
            .setContextLength(512)
            .addEmbeddingLayer()
            .addTransformerLayer(0)
            .addOutputLayer()
            .build();
    }

    inline std::shared_ptr<MockModelLoader> MockModelLoader::createQwen2_05B()
    {
        // Metadata-only preset - no tensors created for fast unit tests
        // Use MockModelLoaderBuilder().setArchitecture(...).addAllLayers().build() for tests needing tensors
        return MockModelLoaderBuilder()
            .setArchitecture("qwen2")
            .setBlockCount(24)
            .setEmbeddingLength(896)
            .setHeadCount(14)
            .setHeadCountKV(2)
            .setVocabSize(151936)
            .setContextLength(32768)
            .setFeedForwardLength(4864) // Actual FFN intermediate size for Qwen2-0.5B
            .setRopeTheta(1000000.0f)
            .setRmsNormEps(1e-6f)
            .build();
    }

    inline std::shared_ptr<MockModelLoader> MockModelLoader::createQwen2_7B()
    {
        // Metadata-only preset - no tensors created for fast unit tests
        return MockModelLoaderBuilder()
            .setArchitecture("qwen2")
            .setBlockCount(28)
            .setEmbeddingLength(3584)
            .setHeadCount(28)
            .setHeadCountKV(4)
            .setVocabSize(152064)
            .setContextLength(131072)
            .setFeedForwardLength(18944) // Actual FFN intermediate size for Qwen2-7B
            .setRopeTheta(1000000.0f)
            .build();
    }

    inline std::shared_ptr<MockModelLoader> MockModelLoader::createLlama3_8B()
    {
        // Metadata-only preset - no tensors created for fast unit tests
        return MockModelLoaderBuilder()
            .setArchitecture("llama")
            .setBlockCount(32)
            .setEmbeddingLength(4096)
            .setHeadCount(32)
            .setHeadCountKV(8)
            .setVocabSize(128256)
            .setContextLength(8192)
            .setFeedForwardLength(14336) // Actual FFN intermediate size for Llama3-8B
            .setRopeTheta(500000.0f)
            .build();
    }

    // =============================================================================
    // BUILDER IMPLEMENTATION
    // =============================================================================

    inline MockModelLoaderBuilder::MockModelLoaderBuilder()
        : mock_(std::make_shared<MockModelLoader>()) {}

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::setArchitecture(const std::string &arch)
    {
        mock_->setArchitecture(arch);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::setBlockCount(uint64_t count)
    {
        mock_->setBlockCount(count);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::setEmbeddingLength(uint64_t length)
    {
        mock_->setEmbeddingLength(length);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::setHeadCount(uint64_t count)
    {
        mock_->setHeadCount(count);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::setHeadCountKV(uint64_t count)
    {
        mock_->setHeadCountKV(count);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::setVocabSize(uint64_t size)
    {
        mock_->setVocabSize(size);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::setContextLength(uint64_t length)
    {
        mock_->setContextLength(length);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::setFeedForwardLength(uint64_t length)
    {
        mock_->setFeedForwardLength(length);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::setRopeTheta(float theta)
    {
        mock_->setRopeTheta(theta);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::setRmsNormEps(float eps)
    {
        mock_->setRmsNormEps(eps);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::setInt(const std::string &key, int value)
    {
        mock_->setIntParam(key, value);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::setFloat(const std::string &key, float value)
    {
        mock_->setFloatParam(key, value);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::setString(const std::string &key, const std::string &value)
    {
        mock_->setStringParam(key, value);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::addTensor(const std::string &name, std::shared_ptr<TensorBase> tensor)
    {
        mock_->addTensor(name, std::move(tensor));
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::addFP32Tensor(const std::string &name, const std::vector<size_t> &shape)
    {
        mock_->addFP32ZerosTensor(name, shape);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::addFP32RandomTensor(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        mock_->addFP32RandomTensor(name, shape, min, max, seed);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::addFP32ZerosTensor(const std::string &name, const std::vector<size_t> &shape)
    {
        mock_->addFP32ZerosTensor(name, shape);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::addFP32OnesTensor(const std::string &name, const std::vector<size_t> &shape)
    {
        mock_->addFP32OnesTensor(name, shape);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::addQ4_0RandomTensor(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        mock_->addQ4_0RandomTensor(name, shape, min, max, seed);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::addQ8_0RandomTensor(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        mock_->addQ8_0RandomTensor(name, shape, min, max, seed);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::addIQ4_NLRandomTensor(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        mock_->addIQ4_NLRandomTensor(name, shape, min, max, seed);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::addEmbeddingLayer()
    {
        size_t vocab = mock_->vocabSize();
        size_t dim = mock_->embeddingLength();
        mock_->addFP32RandomTensor("token_embd.weight", {vocab, dim}, -0.1f, 0.1f, 1001);
        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::addTransformerLayer(int layer_idx)
    {
        std::string prefix = "blk." + std::to_string(layer_idx) + ".";
        size_t dim = mock_->embeddingLength();
        size_t heads = mock_->headCount();
        size_t kv_heads = mock_->headCountKV();
        size_t head_dim = dim / heads;
        size_t ffn_dim = dim * 4; // Typical FFN intermediate dim

        // Attention norms and projections
        mock_->addFP32RandomTensor(prefix + "attn_norm.weight", {dim}, -0.1f, 0.1f, 2001 + layer_idx);
        mock_->addQ4_0RandomTensor(prefix + "attn_q.weight", {dim, dim}, -0.5f, 0.5f, 3001 + layer_idx);
        mock_->addQ4_0RandomTensor(prefix + "attn_k.weight", {kv_heads * head_dim, dim}, -0.5f, 0.5f, 4001 + layer_idx);
        mock_->addQ4_0RandomTensor(prefix + "attn_v.weight", {kv_heads * head_dim, dim}, -0.5f, 0.5f, 5001 + layer_idx);
        mock_->addQ4_0RandomTensor(prefix + "attn_output.weight", {dim, dim}, -0.5f, 0.5f, 6001 + layer_idx);

        // FFN
        mock_->addFP32RandomTensor(prefix + "ffn_norm.weight", {dim}, -0.1f, 0.1f, 7001 + layer_idx);
        mock_->addQ4_0RandomTensor(prefix + "ffn_gate.weight", {ffn_dim, dim}, -0.5f, 0.5f, 8001 + layer_idx);
        mock_->addQ4_0RandomTensor(prefix + "ffn_up.weight", {ffn_dim, dim}, -0.5f, 0.5f, 9001 + layer_idx);
        mock_->addQ4_0RandomTensor(prefix + "ffn_down.weight", {dim, ffn_dim}, -0.5f, 0.5f, 10001 + layer_idx);

        return *this;
    }

    inline MockModelLoaderBuilder &MockModelLoaderBuilder::addOutputLayer()
    {
        size_t vocab = mock_->vocabSize();
        size_t dim = mock_->embeddingLength();
        mock_->addFP32RandomTensor("output_norm.weight", {dim}, -0.1f, 0.1f, 20001);
        mock_->addQ4_0RandomTensor("output.weight", {vocab, dim}, -0.5f, 0.5f, 20002);
        return *this;
    }

    inline std::shared_ptr<MockModelLoader> MockModelLoaderBuilder::build()
    {
        return std::move(mock_);
    }

} // namespace llaminar2::test
