#pragma once

#include "common.h"
#include "tensors/tensor_base.h"
#include "transformer_config.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <fstream>

// GGUF metadata value types
enum class GGUFValueType : uint32_t
{
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12
};

// GGUF tensor type
enum class GGUFTensorType : uint32_t
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15
};

// GGUF metadata value
struct GGUFValue
{
    GGUFValueType type;
    std::vector<uint8_t> data;

    template <typename T>
    T as() const;

    std::string asString() const;
    std::vector<std::string> asStringArray() const;
};

// GGUF tensor info
struct GGUFTensorInfo
{
    std::string name;
    std::vector<uint64_t> dimensions;
    GGUFTensorType type;
    uint64_t offset;
    size_t size_bytes;

    // Quantization info
    bool isQuantized() const;
    size_t getTypeSize() const;
    size_t getBlockSize() const;
};

// GGUF model structure
struct GGUFModel
{
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    uint64_t data_offset; // Start of tensor data section

    std::unordered_map<std::string, GGUFValue> metadata;
    std::vector<GGUFTensorInfo> tensors;

    // Model-specific metadata
    std::string architecture;
    uint32_t context_length;
    uint32_t embedding_length;
    uint32_t block_count;
    uint32_t feed_forward_length;
    uint32_t head_count;
    uint32_t head_count_kv;
    std::vector<std::string> token_list;

    // Helper methods
    bool hasMetadata(const std::string &key) const;
    template <typename T>
    T getMetadata(const std::string &key, T default_value = T{}) const;

    GGUFTensorInfo *findTensor(const std::string &name);
    const GGUFTensorInfo *findTensor(const std::string &name) const;
};

// Model loader class
class ModelLoader
{
public:
    ModelLoader();
    ~ModelLoader() = default;

    // Main loading interface
    bool loadModel(const std::string &file_path);
    bool isLoaded() const { return loaded_; }

    // Model access
    const GGUFModel &getModel() const { return model_; }

    // Tensor loading
    std::shared_ptr<llaminar::TensorBase> loadTensor(const std::string &tensor_name);
    std::vector<std::shared_ptr<llaminar::TensorBase>> loadAllTensors();

    // Model information
    void printModelInfo() const;
    void printTensorInfo() const;
    std::vector<std::string> getTensorNames() const;

    // TESTING HOOKS: Expose specific dequant routines for unit tests (not part of stable public API).
    // These enable constructing synthetic quantized blocks and verifying decode logic.
    std::vector<float> dequantizeQ4_K(const uint8_t *data, size_t n_elements, GGUFTensorType type, const std::string &tensor_name);
    std::vector<float> dequantizeQ4_0(const uint8_t *data, size_t n_elements);
    // Newly added quantization formats (WIP implementations)
    std::vector<float> dequantizeQ5_0(const uint8_t *data, const GGUFTensorInfo &info);                                                      // block_q5_0 (32 vals)
    std::vector<float> dequantizeQ2_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info); // block_q2_K
    std::vector<float> dequantizeQ3_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info); // block_q3_K
    std::vector<float> dequantizeQ5_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info); // block_q5_K (256 super-block)
    std::vector<float> dequantizeQ6_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info); // block_q6_K

    // Quantization support
    bool supportsQuantization(GGUFTensorType type) const;
    std::vector<float> dequantizeTensor(const GGUFTensorInfo &tensor_info,
                                        const std::vector<uint8_t> &quantized_data,
                                        const std::string &tensor_name);

    // Model configuration extraction
    TransformerLayerConfig createLayerConfig() const;

private:
    bool loaded_;
    std::string file_path_;
    GGUFModel model_;
    std::ifstream file_stream_;

    // GGUF parsing
    bool parseHeader();
    bool parseMetadata();
    bool parseTensorInfo();
    bool validateModel();

    // Helper functions
    template <typename T>
    bool readValue(T &value);
    bool readString(std::string &str);
    bool readArray(GGUFValue &value);
    size_t getFileSize() const;

    // Dequantization helpers
    std::vector<float> dequantizeQ8_0(const uint8_t *data, size_t n_elements, const std::string &tensor_name);
    std::vector<float> dequantizeF16(const uint8_t *data, size_t n_elements);
    // Instrumentation helper for dequantization output statistics
    void logDequantStats(const std::string &tensor_name, GGUFTensorType type, const std::vector<float> &values, size_t max_samples) const;

    // Polymorphic dequantization interface
    struct IDequantizer
    {
        virtual ~IDequantizer() = default;
        virtual std::vector<float> run(const GGUFTensorInfo &info, const std::vector<uint8_t> &data, const std::string &name, ModelLoader &loader) const = 0;
    };

    struct Q8_0Dequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &info, const std::vector<uint8_t> &data, const std::string &name, ModelLoader &loader) const override;
    };
    struct Q4_0Dequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &info, const std::vector<uint8_t> &data, const std::string &name, ModelLoader &loader) const override;
    };
    struct Q4KDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &info, const std::vector<uint8_t> &data, const std::string &name, ModelLoader &loader) const override;
    };

    const IDequantizer *selectDequantizer(GGUFTensorType type) const;

    // Model-specific parsing
    void extractModelMetadata();
};

// ---- Static size checks for ggml K-format block structs ----
// These ensure we notice if the embedded/minimal ggml version changes block layouts.
#ifdef GGML_QKK_MAX
static_assert(sizeof(block_q2_K) == 2 * sizeof(ggml_half) + QK_K / 16 + QK_K / 4, "block_q2_K size diverged from ggml");
static_assert(sizeof(block_q3_K) == sizeof(ggml_half) + QK_K / 4 + QK_K / 8 + 12, "block_q3_K size diverged from ggml");
static_assert(sizeof(block_q5_K) == 2 * sizeof(ggml_half) + K_SCALE_SIZE + QK_K / 2 + QK_K / 8, "block_q5_K size diverged from ggml");
static_assert(sizeof(block_q6_K) == sizeof(ggml_half) + QK_K / 16 + 3 * QK_K / 4, "block_q6_K size diverged from ggml");
#endif