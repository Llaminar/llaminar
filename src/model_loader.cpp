// Clean, minimal includes and helpers
#include "model_loader.h"
#include "logger.h"
#include "quant_dequant.h"
#include <cstring>
#include <numeric>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include "graph_compute.h"
#include <filesystem>
#include "tensors/simple_tensor.h"
#include <iostream>
#include <cstdio>
#include <cstdint>

#ifndef K_SCALE_SIZE
#define K_SCALE_SIZE 12
#endif

typedef uint16_t ggml_half;

// Mirrors ggml's ggml_compute_fp16_to_fp32 to ensure identical behaviour for subnormals.
static inline float fp32_from_bits(uint32_t w)
{
    union
    {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f)
{
    union
    {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(uint16_t h)
{
    const uint32_t w = static_cast<uint32_t>(h) << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

// Constructor
ModelLoader::ModelLoader() : loaded_(false) {}

size_t ModelLoader::getFileSize() const
{
    // tellg/seekg are non-const; create a copy of underlying fd position using const_cast
    auto &fs = const_cast<std::ifstream &>(file_stream_);
    if (!fs.good())
        return 0;
    std::ifstream::pos_type current = fs.tellg();
    fs.seekg(0, std::ios::end);
    std::ifstream::pos_type end = fs.tellg();
    fs.seekg(current, std::ios::beg);
    if (end < 0)
        return 0;
    return static_cast<size_t>(end);
}

void ModelLoader::logDequantStats(const std::string &tensor_name, GGUFTensorType type, const std::vector<float> &values, size_t max_samples) const
{
    const char *env_stats = std::getenv("LLAMINAR_DEQUANT_STATS");
    if (!env_stats || std::string(env_stats) == "0")
        return;
    if (values.empty())
    {
        LOG_INFO("[DEQUANT STATS] tensor='" << tensor_name << "' (empty)");
        return;
    }
    double min_v = std::numeric_limits<double>::infinity();
    double max_v = -std::numeric_limits<double>::infinity();
    long double sum = 0.0L;
    for (size_t i = 0; i < values.size(); ++i)
    {
        float v = values[i];
        if (v < min_v)
            min_v = v;
        if (v > max_v)
            max_v = v;
        sum += v;
    }
    double mean = static_cast<double>(sum / values.size());
    long double var_acc = 0.0L;
    for (float v : values)
    {
        long double d = v - mean;
        var_acc += d * d;
    }
    double var = static_cast<double>(var_acc / values.size());
    double stddev = std::sqrt(var);
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss.precision(6);
    ss << "[DEQUANT STATS] tensor='" << tensor_name << "' type=" << static_cast<int>(type)
       << " n=" << values.size() << " min=" << min_v << " max=" << max_v << " mean=" << mean << " stddev=" << stddev;
    size_t samples = std::min(max_samples, values.size());
    ss << " samples=";
    for (size_t i = 0; i < samples; ++i)
    {
        if (i)
            ss << ",";
        ss << values[i];
    }
    LOG_INFO(ss.str());
}

// Remove legacy duplicate fp16 helpers later in file (original block with fp32_from_bits etc. has been deleted)
void ModelLoader::printModelInfo() const
{
    if (!loaded_)
    {
        std::cout << "Model not loaded" << std::endl;
        return;
    }

    std::cout << "\n=== Model Information ===" << std::endl;
    std::cout << "File: " << file_path_ << std::endl;
    std::cout << "GGUF Version: " << model_.version << std::endl;
    std::cout << "Architecture: " << model_.architecture << std::endl;
    std::cout << "Context Length: " << model_.context_length << std::endl;
    std::cout << "Embedding Length: " << model_.embedding_length << std::endl;
    std::cout << "Block Count: " << model_.block_count << std::endl;
    std::cout << "Head Count: " << model_.head_count << std::endl;
    std::cout << "Head Count KV: " << model_.head_count_kv << std::endl;
    std::cout << "Tensors: " << model_.tensor_count << std::endl;
    std::cout << "Metadata entries: " << model_.metadata_kv_count << std::endl;
    std::cout << "=========================" << std::endl;
}

// ---------------- GGUF helper method definitions (previously only declared) -----------------
bool GGUFModel::hasMetadata(const std::string &key) const { return metadata.find(key) != metadata.end(); }

GGUFTensorInfo *GGUFModel::findTensor(const std::string &name)
{
    for (auto &t : tensors)
        if (t.name == name)
            return &t;
    return nullptr;
}
const GGUFTensorInfo *GGUFModel::findTensor(const std::string &name) const
{
    for (auto const &t : tensors)
        if (t.name == name)
            return &t;
    return nullptr;
}

bool GGUFTensorInfo::isQuantized() const
{
    switch (type)
    {
    case GGUFTensorType::Q4_0:
    case GGUFTensorType::Q5_0:
    case GGUFTensorType::Q8_0:
    case GGUFTensorType::Q2_K:
    case GGUFTensorType::Q3_K:
    case GGUFTensorType::Q4_K:
    case GGUFTensorType::Q5_K:
    case GGUFTensorType::Q6_K:
    case GGUFTensorType::Q8_K:
    case GGUFTensorType::Q4_K_M:
        return true;
    default:
        return false;
    }
}

size_t GGUFTensorInfo::getTypeSize() const
{
    switch (type)
    {
    case GGUFTensorType::F32:
        return 4;
    case GGUFTensorType::F16:
        return 2;
    case GGUFTensorType::Q4_0:
        return 18; // 2 + 16
    case GGUFTensorType::Q5_0:
        return 22; // 2 + 4 + 16
    case GGUFTensorType::Q8_0:
        return 34; // 2 + 32
    case GGUFTensorType::Q2_K:
        return 84;
    case GGUFTensorType::Q3_K:
        return 110;
    case GGUFTensorType::Q4_K:
        return 144;
    case GGUFTensorType::Q5_K:
        return 176;
    case GGUFTensorType::Q6_K:
        return 210;
    case GGUFTensorType::Q8_K:
        return 288; // placeholder, verify if used
    case GGUFTensorType::Q4_K_M:
        return 144; // alias
    default:
        return 0;
    }
}

size_t GGUFTensorInfo::getBlockSize() const
{
    switch (type)
    {
    case GGUFTensorType::Q4_0:
    case GGUFTensorType::Q5_0:
    case GGUFTensorType::Q8_0:
        return 32;
    case GGUFTensorType::Q2_K:
    case GGUFTensorType::Q3_K:
    case GGUFTensorType::Q4_K:
    case GGUFTensorType::Q5_K:
    case GGUFTensorType::Q6_K:
    case GGUFTensorType::Q8_K:
    case GGUFTensorType::Q4_K_M:
        return 256;
    default:
        return 0;
    }
}

std::string GGUFValue::asString() const
{
    if (type != GGUFValueType::STRING || data.size() < 8)
        return {};
    uint64_t len;
    std::memcpy(&len, data.data(), 8);
    if (8 + len > data.size())
        return {};
    return std::string(reinterpret_cast<const char *>(data.data() + 8), len);
}

std::vector<std::string> GGUFValue::asStringArray() const
{
    // Layout in data: [array_type(4)][array_length(8)] then repeated [len(8)][bytes]
    std::vector<std::string> out;
    if (data.size() < 12)
        return out;
    uint32_t atype;
    std::memcpy(&atype, data.data(), 4);
    if (atype != static_cast<uint32_t>(GGUFValueType::STRING))
        return out;
    uint64_t count;
    std::memcpy(&count, data.data() + 4, 8);
    size_t cursor = 12;
    for (uint64_t i = 0; i < count && cursor + 8 <= data.size(); ++i)
    {
        uint64_t len;
        std::memcpy(&len, data.data() + cursor, 8);
        cursor += 8;
        if (cursor + len > data.size())
            break;
        out.emplace_back(reinterpret_cast<const char *>(data.data() + cursor), len);
        cursor += len;
    }
    return out;
}

// -------------- Model loading high-level API --------------
bool ModelLoader::loadModel(const std::string &file_path)
{
    file_stream_.close();
    file_stream_.clear();
    file_stream_.open(file_path, std::ios::binary);
    if (!file_stream_)
    {
        LOG_ERROR("Failed to open GGUF file: " << file_path);
        loaded_ = false;
        return false;
    }
    file_path_ = file_path;
    if (!parseHeader())
    {
        LOG_ERROR("parseHeader failed");
        return false;
    }
    if (!parseMetadata())
    {
        LOG_ERROR("parseMetadata failed");
        return false;
    }
    if (!parseTensorInfo())
    {
        LOG_ERROR("parseTensorInfo failed");
        return false;
    }
    extractModelMetadata();
    if (!validateModel())
    {
        LOG_ERROR("validateModel failed");
        return false;
    }
    // Compute data offset (current position)
    model_.data_offset = static_cast<uint64_t>(file_stream_.tellg());
    loaded_ = true;
    return true;
}

std::shared_ptr<llaminar::TensorBase> ModelLoader::loadTensor(const std::string &tensor_name)
{
    if (!loaded_)
    {
        LOG_ERROR("Model not loaded");
        return nullptr;
    }
    const GGUFTensorInfo *info = model_.findTensor(tensor_name);
    if (!info)
    {
        LOG_ERROR("Tensor not found: " << tensor_name);
        return nullptr;
    }
    // Seek to tensor offset
    file_stream_.seekg(model_.data_offset + info->offset, std::ios::beg);
    std::vector<uint8_t> raw(info->size_bytes);
    if (!file_stream_.read(reinterpret_cast<char *>(raw.data()), raw.size()))
    {
        LOG_ERROR("Failed to read tensor bytes: " << tensor_name);
        return nullptr;
    }
    size_t n_elems = 1;
    for (auto d : info->dimensions)
        n_elems *= d;
    std::vector<float> data_f32;
    if (info->isQuantized())
    {
        data_f32 = dequantizeTensor(*info, raw, tensor_name);
    }
    else
    {
        // F16/F32 decode
        if (info->type == GGUFTensorType::F32)
        {
            data_f32.resize(n_elems);
            std::memcpy(data_f32.data(), raw.data(), n_elems * sizeof(float));
        }
        else if (info->type == GGUFTensorType::F16)
        {
            data_f32.resize(n_elems);
            for (size_t i = 0; i < n_elems; ++i)
            {
                uint16_t h;
                std::memcpy(&h, raw.data() + 2 * i, 2);
                data_f32[i] = ggml_compute_fp16_to_fp32(h);
            }
        }
        else
        {
            LOG_ERROR("Unsupported non-quantized type enum=" << (int)info->type);
            return nullptr;
        }
    }
    // Allocate simple tensor (row-major) -- placeholder until hybrid tensors integrated
    std::vector<int> dims;
    dims.reserve(info->dimensions.size());
    for (auto d : info->dimensions)
        dims.push_back(static_cast<int>(d));
    auto simple = std::make_shared<llaminar::SimpleTensor>(dims, data_f32);
    LOG_INFO("Loaded tensor '" << tensor_name << "' elements=" << n_elems << " first=" << (data_f32.empty() ? 0 : data_f32[0]));
    return simple;
}

std::vector<std::shared_ptr<llaminar::TensorBase>> ModelLoader::loadAllTensors()
{
    std::vector<std::shared_ptr<llaminar::TensorBase>> tensors;
    if (!loaded_)
        return tensors;
    tensors.reserve(model_.tensors.size());
    for (auto const &ti : model_.tensors)
    {
        tensors.push_back(loadTensor(ti.name));
    }
    return tensors;
}

bool ModelLoader::parseHeader()
{
    // Read GGUF magic number
    char magic[4];
    if (!file_stream_.read(magic, 4) || std::string(magic, 4) != "GGUF")
    {
        return false;
    }

    // Read version
    if (!readValue(model_.version))
        return false;

    // Read tensor count
    if (!readValue(model_.tensor_count))
        return false;

    // Read metadata count
    if (!readValue(model_.metadata_kv_count))
        return false;

    std::cout << "GGUF Header: version=" << model_.version
              << ", tensors=" << model_.tensor_count
              << ", metadata=" << model_.metadata_kv_count << std::endl;

    return true;
}

bool ModelLoader::parseMetadata()
{
    LOG_TRACE("parseMetadata: Starting, metadata_kv_count=" << model_.metadata_kv_count);
    for (uint64_t i = 0; i < model_.metadata_kv_count; ++i)
    {
        LOG_TRACE("parseMetadata: Processing metadata entry " << i << "/" << model_.metadata_kv_count);
        std::string key;
        if (!readString(key))
            return false;
        LOG_TRACE("parseMetadata: Read key: " << key);

        uint32_t value_type;
        if (!readValue(value_type))
            return false;
        LOG_TRACE("parseMetadata: Read value_type=" << value_type);

        GGUFValue value;
        value.type = static_cast<GGUFValueType>(value_type);
        LOG_TRACE("parseMetadata: Cast to GGUFValueType: " << static_cast<int>(value.type));

        if (value.type == GGUFValueType::ARRAY)
        {
            if (!readArray(value))
                return false;
        }
        else
        {
            // Read simple value based on type
            size_t value_size = 0;
            switch (value.type)
            {
            case GGUFValueType::UINT8:
                value_size = 1;
                break;
            case GGUFValueType::INT8:
                value_size = 1;
                break;
            case GGUFValueType::UINT16:
                value_size = 2;
                break;
            case GGUFValueType::INT16:
                value_size = 2;
                break;
            case GGUFValueType::UINT32:
                value_size = 4;
                break;
            case GGUFValueType::INT32:
                value_size = 4;
                break;
            case GGUFValueType::FLOAT32:
                value_size = 4;
                break;
            case GGUFValueType::UINT64:
                value_size = 8;
                break;
            case GGUFValueType::INT64:
                value_size = 8;
                break;
            case GGUFValueType::FLOAT64:
                value_size = 8;
                break;
            case GGUFValueType::BOOL:
                value_size = 1;
                break;
            case GGUFValueType::STRING:
            {
                uint64_t str_len;
                if (!readValue(str_len))
                    return false;
                LOG_TRACE("parseMetadata: STRING type, str_len=" << str_len);
                if (str_len > 1000000)
                { // 1MB sanity check
                    LOG_ERROR("parseMetadata: STRING length too large: " << str_len);
                    return false;
                }
                LOG_TRACE("parseMetadata: Attempting to resize data to " << (8 + str_len));
                value.data.resize(8 + str_len);
                LOG_TRACE("parseMetadata: Resize successful, copying data");
                std::memcpy(value.data.data(), &str_len, 8);
                if (!file_stream_.read(reinterpret_cast<char *>(value.data.data() + 8), str_len))
                {
                    return false;
                }
                break;
            }
            default:
                return false;
            }

            if (value.type != GGUFValueType::STRING)
            {
                LOG_TRACE("parseMetadata: Non-string type, value_size=" << value_size);
                if (value_size > 1000000)
                { // 1MB sanity check
                    LOG_ERROR("parseMetadata: Value size too large: " << value_size);
                    return false;
                }
                LOG_TRACE("parseMetadata: Attempting to resize data to " << value_size);
                value.data.resize(value_size);
                LOG_TRACE("parseMetadata: Resize successful, reading data");
                if (!file_stream_.read(reinterpret_cast<char *>(value.data.data()), value_size))
                {
                    return false;
                }
            }
        }

        model_.metadata[key] = std::move(value);
        LOG_TRACE("parseMetadata: Completed entry " << i << " for key: " << key);
    }
    LOG_TRACE("parseMetadata: All metadata processed successfully");

    // Extract and validate architecture
    auto arch_it = model_.metadata.find("general.architecture");
    if (arch_it != model_.metadata.end() && arch_it->second.type == GGUFValueType::STRING)
    {
        const auto &arch_data = arch_it->second.data;
        if (arch_data.size() >= 8) // String has 8-byte length prefix
        {
            uint64_t str_len;
            std::memcpy(&str_len, arch_data.data(), 8);
            if (arch_data.size() >= 8 + str_len)
            {
                model_.architecture = std::string(reinterpret_cast<const char *>(arch_data.data() + 8), str_len);
                LOG_INFO("parseMetadata: Extracted architecture: " << model_.architecture);
            }
        }
    }

    if (model_.architecture.empty())
    {
        LOG_WARN("parseMetadata: Architecture not found or could not be extracted from metadata");
    }

    return true;
}

bool ModelLoader::parseTensorInfo()
{
    LOG_TRACE("parseTensorInfo: Resizing tensors vector to " << model_.tensor_count);
    model_.tensors.resize(model_.tensor_count);
    LOG_TRACE("parseTensorInfo: Vector resized successfully");

    for (uint64_t i = 0; i < model_.tensor_count; ++i)
    {
        LOG_TRACE("parseTensorInfo: Processing tensor " << i << "/" << model_.tensor_count);
        auto &tensor = model_.tensors[i];

        // Read tensor name
        if (!readString(tensor.name))
            return false;
        LOG_TRACE("parseTensorInfo: Read tensor name: " << tensor.name);

        // Read number of dimensions
        uint32_t n_dims;
        if (!readValue(n_dims))
            return false;

        // Read dimensions
        tensor.dimensions.resize(n_dims);
        for (uint32_t j = 0; j < n_dims; ++j)
        {
            if (!readValue(tensor.dimensions[j]))
                return false;
        }

        // Read tensor type
        uint32_t type_val;
        if (!readValue(type_val))
            return false;
        // Map raw type id -> enum. We have deprecated legacy types (Q4_1=3, Q5_1=7, Q8_1=9)
        if (type_val == 3 || type_val == 7 || type_val == 9)
        {
            LOG_ERROR("parseTensorInfo: Tensor '" << tensor.name << "' uses deprecated quantization type id=" << type_val
                                                  << " (Q4_1/Q5_1/Q8_1 removed). Please convert the model to a supported format.");
            return false;
        }
        tensor.type = static_cast<GGUFTensorType>(type_val);
        LOG_TRACE("parseTensorInfo: Tensor '" << tensor.name << "' raw type_val=" << type_val << ", cast enum value=" << static_cast<int>(tensor.type));

        // Read tensor offset
        if (!readValue(tensor.offset))
            return false;

        // Calculate tensor size
        size_t total_elements = 1;
        for (uint64_t dim : tensor.dimensions)
        {
            total_elements *= dim;
        }

        if (tensor.isQuantized())
        {
            size_t block_size = tensor.getBlockSize();
            size_t type_size = tensor.getTypeSize();
            size_t num_blocks = (total_elements + block_size - 1) / block_size;
            LOG_TRACE("parseTensorInfo: Quantized tensor '" << tensor.name << "' total_elements=" << total_elements
                                                            << ", block_elems=" << block_size << ", type_block_bytes=" << type_size
                                                            << ", num_blocks=" << num_blocks << ", computed_size=" << (num_blocks * type_size));
            tensor.size_bytes = num_blocks * type_size;
        }
        else
        {
            tensor.size_bytes = total_elements * tensor.getTypeSize();
        }
        LOG_TRACE("parseTensorInfo: Final size_bytes for '" << tensor.name << "' = " << tensor.size_bytes);
    }

    return true;
}

template <typename T>
bool ModelLoader::readValue(T &value)
{
    return static_cast<bool>(file_stream_.read(reinterpret_cast<char *>(&value), sizeof(T)));
}

bool ModelLoader::readString(std::string &str)
{
    uint64_t length;
    if (!readValue(length))
        return false;

    LOG_TRACE("readString: Read length=" << length);
    if (length > 10000000)
    { // 10MB sanity check for string length
        LOG_ERROR("readString: String length too large: " << length);
        return false;
    }

    LOG_TRACE("readString: Attempting to resize string to " << length);
    str.resize(length);
    LOG_TRACE("readString: Resize successful, reading string data");
    return static_cast<bool>(file_stream_.read(str.data(), length));
}

void ModelLoader::extractModelMetadata()
{
    model_.architecture = model_.hasMetadata("general.architecture") ? model_.metadata.at("general.architecture").asString() : "unknown";

    model_.context_length = model_.getMetadata<uint32_t>("qwen2.context_length", 0);
    model_.embedding_length = model_.getMetadata<uint32_t>("qwen2.embedding_length", 0);
    model_.block_count = model_.getMetadata<uint32_t>("qwen2.block_count", 0);
    model_.head_count = model_.getMetadata<uint32_t>("qwen2.attention.head_count", 0);
    model_.head_count_kv = model_.getMetadata<uint32_t>("qwen2.attention.head_count_kv", 0);

    if (model_.hasMetadata("tokenizer.ggml.tokens"))
    {
        model_.token_list = model_.metadata.at("tokenizer.ggml.tokens").asStringArray();
    }
}

// Exact copy of llama.cpp's ggml_compute_fp16_to_fp32 function

std::vector<float> ModelLoader::dequantizeQ8_0(const uint8_t *data, size_t n_elements, const std::string &tensor_name)
{
    std::vector<float> result(n_elements);

    const size_t qk = 32;              // QK8_0 from llama.cpp
    const size_t nb = n_elements / qk; // number of blocks

    // Exact copy of llama.cpp's dequantize_row_q8_0 logic
    for (size_t i = 0; i < nb; i++)
    {
        const size_t block_offset = i * 34; // 2 bytes (fp16 scale) + 32 bytes (int8 values)

        // Read scale factor (ggml_half = uint16_t)
        uint16_t scale_bits;
        std::memcpy(&scale_bits, data + block_offset, 2);

        // Convert using llama.cpp's exact function
        const float d = ggml_compute_fp16_to_fp32(scale_bits);

        // Dequantize 32 int8 values exactly like llama.cpp
        for (size_t j = 0; j < qk; ++j)
        {
            int8_t quantized_val = static_cast<int8_t>(data[block_offset + 2 + j]);
            result[i * qk + j] = quantized_val * d;
        }
    }

    return result;
}

bool ModelLoader::validateModel()
{
    // Check if architecture is specified
    if (model_.architecture.empty())
    {
        std::cerr << "Error: No architecture specified" << std::endl;
        return false;
    }

    // Validate that we support this architecture
    if (model_.architecture != "qwen2")
    {
        std::cerr << "Error: Unsupported architecture '" << model_.architecture << "' (expected 'qwen2')" << std::endl;
        return false;
    }

    LOG_INFO("validateModel: Architecture '" << model_.architecture << "' is supported");

    if (model_.tensors.empty())
    {
        std::cerr << "Error: No tensors found" << std::endl;
        return false;
    }

    return true;
}

TransformerLayerConfig ModelLoader::createLayerConfig() const
{
    TransformerLayerConfig cfg{};
    cfg.n_head = static_cast<int>(model_.head_count);
    cfg.n_head_kv = static_cast<int>(model_.head_count_kv ? model_.head_count_kv : model_.head_count);
    cfg.head_dim = (model_.embedding_length && model_.head_count) ? static_cast<int>(model_.embedding_length / model_.head_count) : 0;
    cfg.d_model = static_cast<int>(model_.embedding_length);
    cfg.d_ff = static_cast<int>(model_.feed_forward_length);
    cfg.vocab_size = static_cast<int>(model_.token_list.size());
    cfg.max_seq_len = static_cast<int>(model_.context_length);
    cfg.n_layers = static_cast<int>(model_.block_count);
    cfg.eps = 1e-5f; // default RMSNorm epsilon; TODO: pull from metadata if present
    return cfg;
}

bool ModelLoader::supportsQuantization(GGUFTensorType type) const
{
    switch (type)
    {
    case GGUFTensorType::F32:
    case GGUFTensorType::F16:
    case GGUFTensorType::Q4_0:
    case GGUFTensorType::Q5_0:
    case GGUFTensorType::Q8_0:
    case GGUFTensorType::Q2_K:
    case GGUFTensorType::Q3_K:
    case GGUFTensorType::Q4_K:
    case GGUFTensorType::Q5_K:
    case GGUFTensorType::Q6_K:
    case GGUFTensorType::Q8_K:
    case GGUFTensorType::Q4_K_M:
        return true;
    default:
        return false;
    }
}

std::vector<float> ModelLoader::dequantizeTensor(const GGUFTensorInfo &tensor_info,
                                                 const std::vector<uint8_t> &quantized_data,
                                                 const std::string &tensor_name)
{
    static bool printed_enum_map = false;
    if (!printed_enum_map)
    {
        printed_enum_map = true;
        std::cerr << "[ENUM MAP] GGUFTensorType values: F32=" << static_cast<int>(GGUFTensorType::F32)
                  << " F16=" << static_cast<int>(GGUFTensorType::F16)
                  << " Q4_0=" << static_cast<int>(GGUFTensorType::Q4_0)
                  << " Q5_0=" << static_cast<int>(GGUFTensorType::Q5_0)
                  << " Q8_0=" << static_cast<int>(GGUFTensorType::Q8_0)
                  << " Q2_K=" << static_cast<int>(GGUFTensorType::Q2_K)
                  << " Q3_K=" << static_cast<int>(GGUFTensorType::Q3_K)
                  << " Q4_K=" << static_cast<int>(GGUFTensorType::Q4_K)
                  << " Q5_K=" << static_cast<int>(GGUFTensorType::Q5_K)
                  << " Q6_K=" << static_cast<int>(GGUFTensorType::Q6_K)
                  << " Q8_K=" << static_cast<int>(GGUFTensorType::Q8_K)
                  << " Q4_K_M=" << static_cast<int>(GGUFTensorType::Q4_K_M)
                  << std::endl;
        std::cerr << "[ENUM MAP NOTE] Deprecated ids (not shown, do not reuse): Q4_1=3 Q5_1=7 Q8_1=9" << std::endl;
    }
    std::cerr << "[DEQ ENTRY] tensor='" << tensor_name << "' enum=" << static_cast<int>(tensor_info.type)
              << " bytes=" << quantized_data.size() << std::endl;
    LOG_TRACE("dequantizeTensor: Enter name='" << tensor_name << "' type=" << static_cast<int>(tensor_info.type)
                                               << " quantized_bytes=" << quantized_data.size());
    const IDequantizer *dq = selectDequantizer(tensor_info.type);
    if (!dq)
    {
        std::cerr << "Error: Unsupported quantization type for tensor " << tensor_info.name
                  << " (enum value=" << static_cast<int>(tensor_info.type) << ")" << std::endl;
        LOG_ERROR("dequantizeTensor: No dequantizer for type enum=" << static_cast<int>(tensor_info.type)
                                                                    << " tensor='" << tensor_name << "'");
        return {};
    }
    LOG_TRACE("dequantizeTensor: Found dequantizer for type enum=" << static_cast<int>(tensor_info.type)
                                                                   << " tensor='" << tensor_name << "'");
    return dq->run(tensor_info, quantized_data, tensor_name, *this);
}
// ---- Polymorphic Dequantizers ----
const ModelLoader::IDequantizer *ModelLoader::selectDequantizer(GGUFTensorType type) const
{
    static Q8_0Dequantizer q8_0;
    static Q4_0Dequantizer q4_0;
    static Q4KDequantizer q4k;
    // New format dequantizers (implemented below)
    struct Q5_0Dequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override { return L.dequantizeQ5_0(d.data(), i); }
    };
    struct Q5KDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override { return L.dequantizeQ5_K(d.data(), i.type, n, i); }
    };
    struct Q2KDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override { return L.dequantizeQ2_K(d.data(), i.type, n, i); }
    };
    struct Q3KDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override { return L.dequantizeQ3_K(d.data(), i.type, n, i); }
    };
    struct Q6KDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override { return L.dequantizeQ6_K(d.data(), i.type, n, i); }
    };
    static Q5_0Dequantizer q5_0;
    static Q5KDequantizer q5k;
    static Q2KDequantizer q2k;
    static Q3KDequantizer q3k;
    static Q6KDequantizer q6k;
    switch (type)
    {
    case GGUFTensorType::Q8_0:
        LOG_TRACE("selectDequantizer: Returning Q8_0Dequantizer");
        return &q8_0;
    case GGUFTensorType::Q4_0:
        LOG_TRACE("selectDequantizer: Returning Q4_0Dequantizer");
        return &q4_0;
    case GGUFTensorType::Q4_K: // fallthrough intentional for grouped variants later
        LOG_TRACE("selectDequantizer: Returning Q4KDequantizer (placeholder)");
        return &q4k;
    case GGUFTensorType::Q5_0:
        return &q5_0;
    case GGUFTensorType::Q5_K:
        return &q5k;
    case GGUFTensorType::Q2_K:
        return &q2k;
    case GGUFTensorType::Q3_K:
        return &q3k;
    case GGUFTensorType::Q6_K:
        return &q6k;
    default:
        LOG_TRACE("selectDequantizer: No dequantizer for enum=" << static_cast<int>(type));
        return nullptr;
    }
}

std::vector<float> ModelLoader::Q8_0Dequantizer::run(const GGUFTensorInfo &info, const std::vector<uint8_t> &data, const std::string &name, ModelLoader &loader) const
{
    size_t n = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    return loader.dequantizeQ8_0(data.data(), n, name);
}

std::vector<float> ModelLoader::Q4_0Dequantizer::run(const GGUFTensorInfo &info, const std::vector<uint8_t> &data, const std::string &name, ModelLoader &loader) const
{
    (void)name; // not used yet
    size_t n = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    LOG_TRACE("Q4_0Dequantizer::run: tensor='" << name << "' n_elements=" << n << " byte_size=" << data.size());
    auto out = loader.dequantizeQ4_0(data.data(), n);
    if (!out.empty())
    {
        LOG_TRACE("Q4_0Dequantizer::run: first 8 values: "
                  << out[0] << ", " << (out.size() > 1 ? out[1] : 0) << ", " << (out.size() > 2 ? out[2] : 0)
                  << ", " << (out.size() > 3 ? out[3] : 0) << ", " << (out.size() > 4 ? out[4] : 0)
                  << ", " << (out.size() > 5 ? out[5] : 0) << ", " << (out.size() > 6 ? out[6] : 0)
                  << ", " << (out.size() > 7 ? out[7] : 0));
    }
    return out;
}

std::vector<float> ModelLoader::Q4KDequantizer::run(const GGUFTensorInfo &info, const std::vector<uint8_t> &data, const std::string &name, ModelLoader &loader) const
{
    size_t n = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    return loader.dequantizeQ4_K(data.data(), n, info.type, name);
}

// ---- Q4_0 Dequant (reference implementation based on llama.cpp layout) ----
// Blocks of 32 values: 2 bytes (fp16 scale) + 16 bytes packed 4-bit values
std::vector<float> ModelLoader::dequantizeQ4_0(const uint8_t *data, size_t n_elements)
{
    const size_t qk = 32;
    const size_t block_size = sizeof(uint16_t) + 16;
    size_t n_blocks = (n_elements + qk - 1) / qk;
    std::vector<float> out(n_blocks * qk, 0.0f);
    LOG_TRACE("dequantizeQ4_0: n_elements=" << n_elements << " n_blocks=" << n_blocks << " block_size_bytes=" << block_size);
    if (n_elements == 0 || !data)
        return out;

    llaminar::dequant_q4_0_rows(data, out.data(), n_elements);

    if (n_blocks > 0)
    {
        uint16_t scale_bits = 0;
        std::memcpy(&scale_bits, data, sizeof(uint16_t));
        float d = ggml_compute_fp16_to_fp32(scale_bits);
        LOG_TRACE("dequantizeQ4_0: block0 scale_bits=" << scale_bits << " scale_float=" << d
                                                       << " first_8_vals=" << out[0] << "," << out[1] << "," << out[2] << "," << out[3]
                                                       << "," << out[4] << "," << out[5] << "," << out[6] << "," << out[7]);
    }

    out.resize(n_elements);
    return out;
}

// ---- Q5_0 Dequant (mirrors ggml dequantize_row_q5_0 logic) ----
// Block layout (22 bytes, 32 values):
//   uint16_t d (fp16 scale)
//   uint8_t  qh[4]  (packed high bits for both halves)
//   uint8_t  qs[16] (low/high nibbles -> base 4-bit values for 2 values each)
// Reconstruction for j in [0,15]:
//   xh_0 = ((qh >> (j + 0)) & 1) << 4  (high bit for value j)
//   xh_1 = ((qh >> (j + 12)) & 1) << 4 (high bit for value j+16)
//   q = qs[j]
//   v0 = ((q & 0x0F) | xh_0) - 16   (signed 5-bit)
//   v1 = ((q >> 4)   | xh_1) - 16
//   out[j] = v0 * d; out[j+16] = v1 * d
std::vector<float> ModelLoader::dequantizeQ5_0(const uint8_t *data, const GGUFTensorInfo &info)
{
    size_t n_elements = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    const size_t QK = 32;
    const size_t BLOCK_BYTES = 22; // 2 + 4 + 16
    if (!data || n_elements == 0)
        return {};
    size_t n_blocks = (n_elements + QK - 1) / QK;
    std::vector<float> out(n_blocks * QK, 0.0f);
    for (size_t b = 0; b < n_blocks; ++b)
    {
        const uint8_t *block = data + b * BLOCK_BYTES;
        float *dst = out.data() + b * QK;
        size_t remain = std::min<size_t>(QK, n_elements - b * QK);
        // Fast path: reuse fused helper for full blocks
        if (remain == QK)
        {
            llaminar::dequant_block_q5_0(block, dst, static_cast<int>(remain));
        }
        else
        {
            // Decode into temp then copy subset
            float tmp[QK];
            llaminar::dequant_block_q5_0(block, tmp, QK);
            std::memcpy(dst, tmp, remain * sizeof(float));
        }
        if (b == 0)
        {
            uint16_t d_bits;
            std::memcpy(&d_bits, block, 2);
            float d = ggml_compute_fp16_to_fp32(d_bits);
            LOG_TRACE("dequantizeQ5_0: tensor='" << info.name << "' n_elements=" << n_elements
                                                 << " n_blocks=" << n_blocks << " d_first=" << d
                                                 << " first8=" << dst[0] << ',' << dst[1] << ',' << dst[2] << ',' << dst[3]
                                                 << ',' << dst[4] << ',' << dst[5] << ',' << dst[6] << ',' << dst[7]);
        }
    }
    out.resize(n_elements);
    logDequantStats(info.name, GGUFTensorType::Q5_0, out, 8);
    return out;
}

// ---- Q4_K Dequant (spec implementation) ----
// Follows llama.cpp's ggml-quants.c::dequantize_row_q4_K logic for CPU path.
// block_q4_K layout per 256 (QK_K) elements:
//   uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; total 144 bytes.
// Value reconstruction per 64-value segment uses two scale/min pairs.
std::vector<float> ModelLoader::dequantizeQ4_K(const uint8_t *data, size_t n_elements, GGUFTensorType type, const std::string &tensor_name)
{
    constexpr size_t SCALES_BYTES = 3 * QK_K / 64;                  // 12
    constexpr size_t QS_BYTES = QK_K / 2;                           // 128
    constexpr size_t BLOCK_BYTES = 2 + 2 + SCALES_BYTES + QS_BYTES; // 144
    if (!data)
        return {};
    size_t n_blocks = (n_elements + QK_K - 1) / QK_K;
    std::vector<float> out(n_blocks * QK_K, 0.0f);

    auto fp16 = [](const uint8_t *p)
    { uint16_t h; std::memcpy(&h,p,2); return ggml_compute_fp16_to_fp32(h); };

    // Unpack scales/mins exactly as in ggml q4_K path (see quants.c utmp manipulations)
    auto unpack_scales_mins = [](const uint8_t *packed, uint8_t out_scales[8], uint8_t out_mins[8])
    {
        // packed[0..11]
        uint32_t utmp[4] = {0, 0, 0, 0};
        std::memcpy(utmp, packed, 12); // fill first 3 words + part of 4th
        // Reconstruct as in llama.cpp (see lines around utmp modifications in ggml-cpu/quants.c)
        const uint32_t kmask1 = 0x3f3f3f3f;
        const uint32_t kmask2 = 0x0f0f0f0f;
        const uint32_t kmask3 = 0x03030303;
        // After memcpy we have utmp[0], utmp[1], utmp[2]; build utmp[3] and reshuffle
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        // Now utmp bytes: scales = bytes 0..3 of utmp[0..1]? In original code scales points to utmp[0], mins to utmp[2].
        const uint8_t *sc_src = reinterpret_cast<const uint8_t *>(&utmp[0]); // 8 bytes of scales
        const uint8_t *mn_src = reinterpret_cast<const uint8_t *>(&utmp[2]); // 8 bytes of mins
        for (int i = 0; i < 8; ++i)
        {
            out_scales[i] = sc_src[i];
            out_mins[i] = mn_src[i];
        }
    };

    size_t cursor = 0;
    size_t produced = 0;
    for (size_t b = 0; b < n_blocks; ++b)
    {
        if (cursor + BLOCK_BYTES > n_blocks * BLOCK_BYTES)
        {
            // guard against truncated buffer
            LOG_ERROR("Q4_K decode truncated buffer at block=" << b);
            break;
        }
        float d = fp16(data + cursor);
        float dmin = fp16(data + cursor + 2);
        const uint8_t *scales = data + cursor + 4;
        const uint8_t *qs = data + cursor + 4 + SCALES_BYTES;
        // Extract full 8 scales + 8 mins
        uint8_t sc_arr[8], mn_arr[8];
        unpack_scales_mins(scales, sc_arr, mn_arr);
        const uint8_t *qptr = qs;
        // Four groups of 64 values, each uses two (scale,min) pairs
        for (int g = 0; g < 4; ++g)
        {
            int idx0 = 2 * g + 0;
            int idx1 = 2 * g + 1;
            float d1 = d * sc_arr[idx0];
            float dm1 = dmin * mn_arr[idx0];
            float d2 = d * sc_arr[idx1];
            float dm2 = dmin * mn_arr[idx1];
            // first 32 values: low nibble
            for (int l = 0; l < 32 && produced < out.size(); ++l)
                out[produced++] = d1 * (qptr[l] & 0x0F) - dm1;
            // second 32: high nibble
            for (int l = 0; l < 32 && produced < out.size(); ++l)
                out[produced++] = d2 * (qptr[l] >> 4) - dm2;
            qptr += 32;
        }
        if (b == 0)
        {
            LOG_TRACE("dequantizeQ4_K(spec): tensor='" << tensor_name << "' d=" << d << " dmin=" << dmin
                                                       << " first8=" << out[0] << ',' << out[1] << ',' << out[2] << ',' << out[3]
                                                       << ',' << out[4] << ',' << out[5] << ',' << out[6] << ',' << out[7]);
        }
        cursor += BLOCK_BYTES;
    }
    out.resize(n_elements);
    logDequantStats(tensor_name, type, out, 8);
    return out;
}

// (Removed obsolete Q5_0 duplicate & Q5_1 support region)

// ---------------- Quant shard cache & partial load helpers (minimal implementation) ----------------
const ModelLoader::CachedFullTensor *ModelLoader::getOrCacheFullQuantTensor(const std::string &tensor_name, const GGUFTensorInfo &info)
{
    // Fast path: lookup
    {
        std::lock_guard<std::mutex> lock(quant_cache_mutex_);
        auto it = quant_full_cache_.find(tensor_name);
        if (it != quant_full_cache_.end())
        {
            it->second.last_access = ++quant_cache_clock_;
            quant_cache_hits_++;
            return &it->second;
        }
    }
    quant_cache_misses_++;
    // Simplified: just read declared size; deprecated Q5_1 variant detection removed.
    size_t logical_read_size = info.size_bytes;
    const GGUFTensorInfo &adjusted_info = info; // retained for interface uniformity

    // Re-seek and read selected number of bytes
    file_stream_.seekg(model_.data_offset + info.offset, std::ios::beg);
    std::vector<uint8_t> raw(logical_read_size);
    if (!file_stream_.read(reinterpret_cast<char *>(raw.data()), raw.size()))
    {
        LOG_ERROR("getOrCacheFullQuantTensor: read failed for tensor '" << tensor_name << "' (requested=" << logical_read_size << ")");
        return nullptr;
    }
    size_t n_elems = 1;
    for (auto d : info.dimensions)
        n_elems *= d;
    std::vector<float> decoded;
    if (info.isQuantized())
        decoded = dequantizeTensor(adjusted_info, raw, tensor_name);
    else
    {
        if (info.type == GGUFTensorType::F32)
        {
            decoded.resize(n_elems);
            std::memcpy(decoded.data(), raw.data(), n_elems * sizeof(float));
        }
        else if (info.type == GGUFTensorType::F16)
        {
            decoded.resize(n_elems);
            for (size_t i = 0; i < n_elems; ++i)
            {
                uint16_t h;
                std::memcpy(&h, raw.data() + 2 * i, 2);
                decoded[i] = ggml_compute_fp16_to_fp32(h);
            }
        }
        else
        {
            return nullptr;
        }
    }
    auto cache_bytes_limit = quantShardCacheMaxBytes();
    const size_t new_bytes = decoded.size() * sizeof(float);
    {
        std::lock_guard<std::mutex> lock(quant_cache_mutex_);
        // Simple LRU eviction until space
        while (cache_bytes_limit > 0 && quant_cache_bytes_ + new_bytes > cache_bytes_limit && !quant_full_cache_.empty())
        {
            auto victim_it = std::min_element(quant_full_cache_.begin(), quant_full_cache_.end(), [](auto &a, auto &b)
                                              { return a.second.last_access < b.second.last_access; });
            if (victim_it == quant_full_cache_.end())
                break;
            quant_cache_bytes_ -= victim_it->second.bytes;
            quant_full_cache_.erase(victim_it);
            quant_cache_evictions_++;
        }
        auto &entry = quant_full_cache_[tensor_name];
        entry.data.swap(decoded);
        entry.shape.clear();
        for (auto d : info.dimensions)
            entry.shape.push_back(static_cast<int>(d));
        entry.bytes = new_bytes;
        entry.last_access = ++quant_cache_clock_;
        quant_cache_bytes_ += new_bytes;
        quant_cache_loads_++;
        return &entry;
    }
}

size_t ModelLoader::quantShardCacheMaxBytes() const
{
    const char *env = std::getenv("LLAMINAR_SHARD_CACHE_MAX_MB");
    size_t mb = env ? std::strtoul(env, nullptr, 10) : 512;
    if (mb == 0)
        return 0; // disabled
    return mb * 1024ULL * 1024ULL;
}

void ModelLoader::clearQuantShardCache()
{
    std::lock_guard<std::mutex> lock(quant_cache_mutex_);
    quant_full_cache_.clear();
    quant_cache_bytes_ = 0;
    quant_cache_loads_ = 0;
    quant_cache_hits_ = 0;
    quant_cache_misses_ = 0;
    quant_cache_evictions_ = 0;
}

ModelLoader::QuantShardCacheStats ModelLoader::getQuantShardCacheStats() const
{
    QuantShardCacheStats s;
    s.loads = quant_cache_loads_;
    s.cache_hits = quant_cache_hits_;
    s.cache_misses = quant_cache_misses_;
    s.evictions = quant_cache_evictions_;
    s.bytes_resident = quant_cache_bytes_;
    return s;
}

std::vector<std::string> ModelLoader::getTensorNames() const
{
    std::vector<std::string> names;
    names.reserve(model_.tensors.size());
    for (auto const &t : model_.tensors)
        names.push_back(t.name);
    return names;
}

bool ModelLoader::loadTensorColumnShards(const std::string &tensor_name,
                                         const std::vector<int> &col_offsets,
                                         const std::vector<int> &col_counts,
                                         const std::vector<float *> &dests)
{
    if (col_offsets.size() != col_counts.size() || col_offsets.size() != dests.size())
    {
        LOG_ERROR("loadTensorColumnShards: mismatched shard vector sizes");
        return false;
    }
    const GGUFTensorInfo *info = model_.findTensor(tensor_name);
    if (!info)
    {
        LOG_ERROR("loadTensorColumnShards: tensor not found '" << tensor_name << "'");
        return false;
    }
    if (info->dimensions.size() != 2)
    {
        LOG_WARN("loadTensorColumnShards: only 2D tensors supported; falling back to full");
    }
    size_t rows = info->dimensions[0];
    size_t cols = (info->dimensions.size() > 1) ? info->dimensions[1] : 1;
    // Fallback: use (cached) full tensor and slice
    const CachedFullTensor *full = getOrCacheFullQuantTensor(tensor_name, *info);
    if (!full)
        return false;
    const float *src = full->data.data();
    for (size_t s = 0; s < col_offsets.size(); ++s)
    {
        int off = col_offsets[s];
        int count = col_counts[s];
        float *dst = dests[s];
        if (!dst || off < 0 || count <= 0 || static_cast<size_t>(off + count) > cols)
        {
            LOG_ERROR("loadTensorColumnShards: invalid shard spec index=" << s);
            return false;
        }
        for (size_t r = 0; r < rows; ++r)
        {
            const float *row_src = src + r * cols + off;
            std::memcpy(dst + r * count, row_src, sizeof(float) * count);
        }
    }
    return true;
}

bool ModelLoader::loadTensorRowShard(const std::string &tensor_name,
                                     int row_offset,
                                     int row_count,
                                     float *dest)
{
    if (row_count <= 0 || row_offset < 0)
        return false;
    const GGUFTensorInfo *info = model_.findTensor(tensor_name);
    if (!info)
        return false;
    if (info->dimensions.size() != 2)
        return false;
    size_t rows = info->dimensions[0];
    size_t cols = info->dimensions[1];
    if (static_cast<size_t>(row_offset + row_count) > rows)
        return false;
    const CachedFullTensor *full = getOrCacheFullQuantTensor(tensor_name, *info);
    if (!full)
        return false;
    const float *src = full->data.data() + static_cast<size_t>(row_offset) * cols;
    std::memcpy(dest, src, sizeof(float) * static_cast<size_t>(row_count) * cols);
    return true;
}

// ---- Q2_K Dequant ----
// Layout per super-block (256 values) mirrors ggml's block_q2_K:
//   uint8_t scales[16]; uint8_t qs[64]; ggml_half d; ggml_half dmin
// Each scale byte packs a 4-bit scale and 4-bit min for 16-value groups. Reconstruct values as:
// value = (d * scale_nibble) * q - (dmin * min_nibble) where q is the 2-bit quantized value.
std::vector<float> ModelLoader::dequantizeQ2_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info)
{
    (void)type;
    size_t n_elements = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    if (n_elements == 0 || !data)
        return {};

    size_t n_blocks = (n_elements + QK_K - 1) / QK_K;
    constexpr size_t CANONICAL_BLOCK_BYTES = sizeof(block_q2_K);
    constexpr size_t QS_BYTES = QK_K / 4;     // 64
    constexpr size_t SCALE_BYTES = QK_K / 16; // 16

    size_t block_stride = CANONICAL_BLOCK_BYTES;
    if (info.size_bytes != 0)
    {
        if (info.size_bytes % n_blocks != 0)
        {
            LOG_ERROR("Tensor '" << tensor_name << "' size " << info.size_bytes << " does not align with " << n_blocks << " Q2_K blocks (" << CANONICAL_BLOCK_BYTES << " bytes each)");
            return {};
        }
        block_stride = info.size_bytes / n_blocks;
    }

    const bool compact_header_first = (block_stride < CANONICAL_BLOCK_BYTES) && (block_stride >= (2 * sizeof(uint16_t) + QS_BYTES));
    if (!compact_header_first && block_stride < CANONICAL_BLOCK_BYTES)
    {
        LOG_ERROR("Tensor '" << tensor_name << "' Q2_K block stride " << block_stride << " is smaller than canonical layout and not recognized as compact header-first variant");
        return {};
    }

    std::vector<float> out(n_blocks * QK_K);
    const uint8_t *block_ptr = data;
    for (size_t b = 0; b < n_blocks; ++b)
    {
        block_q2_K blk{};
        if (compact_header_first)
        {
            uint16_t d_bits = 0;
            uint16_t dmin_bits = 0;
            std::memcpy(&d_bits, block_ptr, sizeof(uint16_t));
            std::memcpy(&dmin_bits, block_ptr + sizeof(uint16_t), sizeof(uint16_t));
            blk.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = d_bits;
            blk.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin = dmin_bits;

            const uint8_t *compact_scales = block_ptr + 2 * sizeof(uint16_t);
            size_t compact_scale_bytes = block_stride - (2 * sizeof(uint16_t) + QS_BYTES);
            compact_scale_bytes = std::min(compact_scale_bytes, SCALE_BYTES);
            std::memset(blk.scales, 0, SCALE_BYTES);
            if (compact_scale_bytes > 0)
                std::memcpy(blk.scales, compact_scales, compact_scale_bytes);

            const uint8_t *compact_qs = compact_scales + (block_stride - (2 * sizeof(uint16_t) + QS_BYTES));
            std::memcpy(blk.qs, compact_qs, QS_BYTES);

            if (b == 0)
            {
                std::cout << "[Q2K compact debug] tensor='" << tensor_name
                          << "' stride=" << block_stride
                          << " d_bits=0x" << std::hex << d_bits << std::dec
                          << " dmin_bits=0x" << std::hex << dmin_bits << std::dec
                          << " scale0=" << int(blk.scales[0])
                          << " scale1=" << int(blk.scales[1])
                          << " qs0=0x" << std::hex << int(blk.qs[0]) << std::dec
                          << std::endl;
            }
        }
        else
        {
            std::memcpy(&blk, block_ptr, sizeof(block_q2_K));
        }
        float *dst_block = out.data() + b * QK_K;
        dequantize_row_q2_K(&blk, dst_block, QK_K);
        if (std::isnan(dst_block[0]))
        {
            const float d = ggml_compute_fp16_to_fp32(blk.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
            const float dmin = ggml_compute_fp16_to_fp32(blk.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
            LOG_ERROR("dequantizeQ2_K block" << b << " produced NaN. d=" << d
                                             << " dmin=" << dmin
                                             << " scale0=" << int(blk.scales[0])
                                             << " scale1=" << int(blk.scales[1])
                                             << " q0=" << int(blk.qs[0]) << " q1=" << int(blk.qs[1]));
        }
        block_ptr += block_stride;
    }
    out.resize(n_elements);
    logDequantStats(tensor_name, type, out, 8);
    return out;
}

// ---- Q3_K Dequant ----
// Layout per block_q3_K (256 vals): uint16_t d; uint8_t qs[64]; uint8_t hmask[32]; uint8_t scales[12];
// Reconstruct as in llama.cpp: form 16 groups of 16 values with (scales[g]-32) multiplier and signed low bits via hmask.
std::vector<float> ModelLoader::dequantizeQ3_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info)
{
    (void)type;
    // Mirror ggml::dequantize_row_q3_K logic exactly.
    constexpr int QS_BYTES = QK_K / 4;    // low 2-bit packed
    constexpr int HMASK_BYTES = QK_K / 8; // high bit mask
    constexpr int SCALES_PACKED = 12;     // packed 6-bit scales
    size_t n_elements = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    size_t n_blocks = (n_elements + QK_K - 1) / QK_K;
    std::vector<float> out(n_blocks * QK_K);
    auto fp16 = [](const uint8_t *p)
    { uint16_t h; std::memcpy(&h,p,2); return ggml_compute_fp16_to_fp32(h); };
    const uint8_t *ptr = data;
    size_t out_index = 0;
    for (size_t b = 0; b < n_blocks; ++b)
    {
        const uint8_t *hmask = ptr;                     // QK_K/8
        const uint8_t *qs = hmask + HMASK_BYTES;        // low bits
        const uint8_t *sc_p = qs + QS_BYTES;            // packed scales
        const float d_all = fp16(sc_p + SCALES_PACKED); // d stored last in struct

        // Unpack scales to 16 signed 6-bit values
        uint32_t aux[4] = {0, 0, 0, 0};
        std::memcpy(aux, sc_p, SCALES_PACKED);
        const uint32_t kmask1 = 0x03030303u;
        const uint32_t kmask2 = 0x0f0f0f0fu;
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int8_t scales[16];
        std::memcpy(scales, aux, 16);

        const uint8_t *q = qs;
        const uint8_t *hm = hmask;
        uint8_t m = 1; // bit mask cycling through 8 bits each shift phase
        int is = 0;    // scale index
        for (int n = 0; n < QK_K; n += 128)
        {
            int shift = 0;
            for (int j = 0; j < 4; ++j)
            {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                {
                    int8_t qv = ((q[l + 0] >> shift) & 0x3); // low two bits
                    if ((hm[l + 0] & m) == 0)
                        qv -= 4; // subtract 4 if corresponding high-bit not set
                    out[out_index++] = dl * qv;
                }
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                {
                    int8_t qv = ((q[l + 16] >> shift) & 0x3);
                    if ((hm[l + 16] & m) == 0)
                        qv -= 4;
                    out[out_index++] = dl * qv;
                }
                shift += 2;
                m <<= 1; // advance high-bit plane
            }
            q += 32;
        }
        ptr += HMASK_BYTES + QS_BYTES + SCALES_PACKED + 2; // advance to next block (d already consumed)
    }
    out.resize(n_elements);
    logDequantStats(tensor_name, type, out, 8);
    return out;
}

// ---- Q5_K Dequant ----
// Layout (block_q5_K, 256 vals): uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128];
// Scales/mins unpack identical to Q4_K. Two (scale,min) pairs per 64-value segment.
std::vector<float> ModelLoader::dequantizeQ5_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info)
{
    (void)type;
    // Manual layout interpretation (avoid dependency on ggml block struct):
    // [0..1]=d (fp16), [2..3]=dmin (fp16), [4..4+K_SCALE_SIZE-1]=scales (6-bit packed scale/min pairs),
    // then qh (QK_K/8 bytes), then qs (QK_K/2 bytes). Total block bytes = 4 + K_SCALE_SIZE + QK_K/8 + QK_K/2.
    if (!data)
        return {};
    size_t n_elements = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    size_t n_blocks = (n_elements + QK_K - 1) / QK_K;
    std::vector<float> out(n_blocks * QK_K);

    auto fp16_to_f32 = [](const uint8_t *p)
    { uint16_t h; std::memcpy(&h, p, 2); return ggml_compute_fp16_to_fp32(h); };
    auto get_scale_min_k4_local = [](int j, const uint8_t *q, uint8_t &d_out, uint8_t &m_out)
    {
        if (j < 4)
        {
            d_out = q[j] & 63;
            m_out = q[j + 4] & 63;
        }
        else
        {
            d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
            m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
        }
    };

    const size_t block_size = 4 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2; // bytes per q5_K block
    const uint8_t *ptr = data;
    size_t out_index = 0;
    for (size_t b = 0; b < n_blocks; ++b)
    {
        const uint8_t *block = ptr + b * block_size;
        const float d = fp16_to_f32(block + 0);
        const float dmin = fp16_to_f32(block + 2);
        const uint8_t *scales = block + 4;
        const uint8_t *qh = scales + K_SCALE_SIZE;
        const uint8_t *ql = qh + QK_K / 8;

        // (Debug instrumentation removed after validation; kept minimal for cleanliness.)

        int is = 0;             // scale/min pair index (2 per 64 values)
        uint8_t u1 = 1, u2 = 2; // moving bit planes inside qh
        const uint8_t *ql_iter = ql;
        for (int j = 0; j < QK_K; j += 64)
        {
            uint8_t sc, m;
            get_scale_min_k4_local(is + 0, scales, sc, m);
            const float d1 = d * sc;
            const float m1 = dmin * m;
            get_scale_min_k4_local(is + 1, scales, sc, m);
            const float d2 = d * sc;
            const float m2 = dmin * m;
            // First 32 bytes -> 32 low-nibble + optional high-bit additions (values 0..31)
            for (int l = 0; l < 32; ++l)
            {
                uint8_t v = (ql_iter[l] & 0xF);
                if (qh[l] & u1)
                    v += 16;
                out[out_index++] = d1 * v - m1;
            }
            // Upper nibble of same 32 bytes encodes next 32 values
            for (int l = 0; l < 32; ++l)
            {
                uint8_t v = (ql_iter[l] >> 4);
                if (qh[l] & u2)
                    v += 16;
                out[out_index++] = d2 * v - m2;
            }
            // Removed per-layer debug once correctness established.
            ql_iter += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    out.resize(n_elements);
    logDequantStats(tensor_name, type, out, 8);
    return out;
}

// ---- Q6_K Dequant ----
// Layout: uint16_t d; uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; (total 2+128+64+16=210)
// Reconstruct raw6 in [0..63], center to [-32,31], multiply by d*scale_int.
std::vector<float> ModelLoader::dequantizeQ6_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info)
{
    (void)type;
    if (!data)
        return {};
    size_t n_elements = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    size_t n_blocks = (n_elements + QK_K - 1) / QK_K;
    std::vector<float> out(n_blocks * QK_K);
    auto fp16 = [](const uint8_t *p)
    { uint16_t h; std::memcpy(&h,p,2); return ggml_compute_fp16_to_fp32(h); };
    const uint8_t *ptr = data;
    size_t out_index = 0;
    for (size_t b = 0; b < n_blocks; ++b)
    {
        const uint8_t *ql = ptr;                                                 // lower 4 bits (QK_K/2)
        const uint8_t *qh = ql + QK_K / 2;                                       // upper 2 bits (QK_K/4)
        const int8_t *sc = reinterpret_cast<const int8_t *>(qh + QK_K / 4);      // 16 scales
        const float d = fp16(reinterpret_cast<const uint8_t *>(sc) + QK_K / 16); // d at end
        const uint8_t *ql_iter = ql;
        const uint8_t *qh_iter = qh;
        const int8_t *sc_iter = sc;
        for (int n = 0; n < QK_K; n += 128)
        {
            for (int l = 0; l < 32; ++l)
            {
                int is = l / 16; // scale group offset within this 128 chunk (0 or 1) plus even offsets when advancing sc_iter
                const int8_t q1 = ((ql_iter[l + 0] & 0xF) | (((qh_iter[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = ((ql_iter[l + 32] & 0xF) | (((qh_iter[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = ((ql_iter[l + 0] >> 4) | (((qh_iter[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = ((ql_iter[l + 32] >> 4) | (((qh_iter[l] >> 6) & 3) << 4)) - 32;
                out[out_index + l + 0] = d * sc_iter[is + 0] * q1;
                out[out_index + l + 32] = d * sc_iter[is + 2] * q2;
                out[out_index + l + 64] = d * sc_iter[is + 4] * q3;
                out[out_index + l + 96] = d * sc_iter[is + 6] * q4;
            }
            out_index += 128;
            ql_iter += 64;
            qh_iter += 32;
            sc_iter += 8; // advance to next 128-values portion
        }
        ptr += QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // move to next block
    }
    out.resize(n_elements);
    logDequantStats(tensor_name, type, out, 8);
    return out;
}

bool ModelLoader::readArray(GGUFValue &value)
{
    // Read array type and length
    uint32_t array_type;
    uint64_t array_length;
    if (!readValue(array_type) || !readValue(array_length))
    {
        return false;
    }

    LOG_TRACE("readArray: type=" << array_type << ", length=" << array_length);

    // Reserve space for the array data
    value.data.clear();
    value.data.reserve(12 + array_length * 20); // Estimate

    // Store array type and length at the beginning
    value.data.resize(12);
    std::memcpy(value.data.data(), &array_type, 4);
    std::memcpy(value.data.data() + 4, &array_length, 8);

    // Read array elements based on type
    GGUFValueType element_type = static_cast<GGUFValueType>(array_type);

    switch (element_type)
    {
    case GGUFValueType::STRING:
    {
        // Read string array - this is what we need for vocabulary
        for (uint64_t i = 0; i < array_length; ++i)
        {
            uint64_t str_len;
            if (!readValue(str_len))
            {
                return false;
            }

            // Store string length
            size_t len_offset = value.data.size();
            value.data.resize(value.data.size() + 8);
            std::memcpy(value.data.data() + len_offset, &str_len, 8);

            // Store string data
            size_t str_offset = value.data.size();
            value.data.resize(value.data.size() + str_len);
            if (!file_stream_.read(reinterpret_cast<char *>(value.data.data() + str_offset), str_len))
            {
                LOG_ERROR("readArray: Failed to read string data");
                return false;
            }
        }
        LOG_INFO("readArray: Successfully read string array with " << array_length << " elements");
        break;
    }

    case GGUFValueType::UINT32:
    case GGUFValueType::INT32:
    {
        // Read 32-bit integer array
        for (uint64_t i = 0; i < array_length; ++i)
        {
            uint32_t val;
            if (!readValue(val))
            {
                return false;
            }
            size_t offset = value.data.size();
            value.data.resize(value.data.size() + 4);
            std::memcpy(value.data.data() + offset, &val, 4);
        }
        LOG_INFO("readArray: Successfully read int32 array with " << array_length << " elements");
        break;
    }

    default:
    {
        // For other types, skip the data for now
        LOG_WARN("readArray: Skipping unsupported array type " << array_type);
        size_t element_size = 0;
        switch (element_type)
        {
        case GGUFValueType::UINT8:
        case GGUFValueType::INT8:
        case GGUFValueType::BOOL:
            element_size = 1;
            break;
        case GGUFValueType::UINT16:
        case GGUFValueType::INT16:
            element_size = 2;
            break;
        case GGUFValueType::FLOAT32:
            element_size = 4;
            break;
        case GGUFValueType::UINT64:
        case GGUFValueType::INT64:
        case GGUFValueType::FLOAT64:
            element_size = 8;
            break;
        default:
            LOG_ERROR("readArray: Unknown array element type: " << array_type);
            return false;
        }

        if (element_size > 0)
        {
            size_t bytes_to_skip = array_length * element_size;
            if (!file_stream_.seekg(bytes_to_skip, std::ios::cur))
            {
                LOG_ERROR("readArray: Failed to skip array data");
                return false;
            }
        }
        value.data.clear(); // Clear data since we didn't read it
        break;
    }
    }

    return true;
}

// Template method definitions and explicit instantiations
// These were moved here to ensure linkage after introducing new code sections above.
template <typename T>
T GGUFValue::as() const
{
    if constexpr (std::is_same_v<T, int32_t>)
    {
        int32_t v;
        std::memcpy(&v, data.data(), sizeof(v));
        return v;
    }
    else if constexpr (std::is_same_v<T, uint32_t>)
    {
        uint32_t v;
        std::memcpy(&v, data.data(), sizeof(v));
        return v;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        float v;
        std::memcpy(&v, data.data(), sizeof(v));
        return v;
    }
    else if constexpr (std::is_same_v<T, std::string>)
    {
        return asString();
    }
    else
    {
        static_assert(!sizeof(T *), "Unsupported GGUFValue::as<T>() instantiation");
    }
}

template <typename T>
T GGUFModel::getMetadata(const std::string &key, T default_value) const
{
    auto it = metadata.find(key);
    if (it == metadata.end())
        return default_value;
    return it->second.as<T>();
}

template std::string GGUFModel::getMetadata<std::string>(const std::string &key, std::string default_value) const;
template uint32_t GGUFModel::getMetadata<uint32_t>(const std::string &key, uint32_t default_value) const;
template int32_t GGUFModel::getMetadata<int32_t>(const std::string &key, int32_t default_value) const;
template float GGUFModel::getMetadata<float>(const std::string &key, float default_value) const;

template int32_t GGUFValue::as<int32_t>() const;
template uint32_t GGUFValue::as<uint32_t>() const;
template float GGUFValue::as<float>() const;
template std::string GGUFValue::as<std::string>() const;