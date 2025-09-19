#pragma once

#include "common.h"
#include "tensor.h"
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
    std::shared_ptr<llaminar::Tensor> loadTensor(const std::string &tensor_name);
    std::vector<std::shared_ptr<llaminar::Tensor>> loadAllTensors();

    // Model information
    void printModelInfo() const;
    void printTensorInfo() const;
    std::vector<std::string> getTensorNames() const;

    // Quantization support
    bool supportsQuantization(GGUFTensorType type) const;
    std::vector<float> dequantizeTensor(const GGUFTensorInfo &tensor_info,
                                        const std::vector<uint8_t> &quantized_data);

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

    // Dequantization helpers
    std::vector<float> dequantizeQ8_0(const uint8_t *data, size_t n_elements);
    std::vector<float> dequantizeQ4_0(const uint8_t *data, size_t n_elements);
    std::vector<float> dequantizeF16(const uint8_t *data, size_t n_elements);

    // Model-specific parsing
    void extractModelMetadata();
};