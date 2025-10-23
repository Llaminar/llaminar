/**
 * @file ModelLoader.cpp
 * @brief Streamlined GGUF model loader implementation for Llaminar V2
 * @author David Sanftenberg
 */

#include "ModelLoader.h"
#include "../tensors/Tensors.h"
#include "../tensors/TensorFactory.h"
#include <cstring>
#include <iostream>

namespace llaminar2
{

    // =============================================================================
    // GGUF VALUE ACCESSORS
    // =============================================================================

    uint32_t GGUFValue::asUInt32() const
    {
        if (type != GGUFValueType::UINT32 || data.size() < 4)
            return 0;
        uint32_t val;
        std::memcpy(&val, data.data(), 4);
        return val;
    }

    uint64_t GGUFValue::asUInt64() const
    {
        if (type != GGUFValueType::UINT64 || data.size() < 8)
            return 0;
        uint64_t val;
        std::memcpy(&val, data.data(), 8);
        return val;
    }

    float GGUFValue::asFloat32() const
    {
        if (type != GGUFValueType::FLOAT32 || data.size() < 4)
            return 0.0f;
        float val;
        std::memcpy(&val, data.data(), 4);
        return val;
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

    // =============================================================================
    // GGUF TENSOR INFO HELPERS
    // =============================================================================

    bool GGUFTensorInfo::isQuantized() const
    {
        switch (type)
        {
        case GGUFTensorType::Q4_0:
        case GGUFTensorType::Q4_1:
        case GGUFTensorType::Q5_0:
        case GGUFTensorType::Q5_1:
        case GGUFTensorType::Q8_0:
        case GGUFTensorType::Q2_K:
        case GGUFTensorType::Q3_K:
        case GGUFTensorType::Q4_K:
        case GGUFTensorType::Q5_K:
        case GGUFTensorType::Q6_K:
        case GGUFTensorType::Q8_K:
        case GGUFTensorType::IQ2_XXS:
        case GGUFTensorType::IQ2_XS:
        case GGUFTensorType::IQ3_XXS:
        case GGUFTensorType::IQ1_S:
        case GGUFTensorType::IQ4_NL:
        case GGUFTensorType::IQ3_S:
        case GGUFTensorType::IQ2_S:
        case GGUFTensorType::IQ4_XS:
        case GGUFTensorType::IQ1_M:
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
        case GGUFTensorType::BF16:
            return 2;
        case GGUFTensorType::Q4_0:
            return 18;
        case GGUFTensorType::Q4_1:
            return 20;
        case GGUFTensorType::Q5_0:
            return 22;
        case GGUFTensorType::Q5_1:
            return 24;
        case GGUFTensorType::Q8_0:
            return 34;
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
            return 288;
        case GGUFTensorType::IQ2_XXS:
            return 66;
        case GGUFTensorType::IQ2_XS:
            return 74;
        case GGUFTensorType::IQ3_XXS:
            return 98;
        case GGUFTensorType::IQ1_S:
            return 50;
        case GGUFTensorType::IQ4_NL:
            return 18;
        case GGUFTensorType::IQ3_S:
            return 110;
        case GGUFTensorType::IQ2_S:
            return 82;
        case GGUFTensorType::IQ4_XS:
            return 18;
        case GGUFTensorType::IQ1_M:
            return 56;
        default:
            return 0;
        }
    }

    size_t GGUFTensorInfo::getBlockSize() const
    {
        switch (type)
        {
        case GGUFTensorType::Q4_0:
        case GGUFTensorType::Q4_1:
        case GGUFTensorType::Q5_0:
        case GGUFTensorType::Q5_1:
        case GGUFTensorType::Q8_0:
        case GGUFTensorType::IQ4_NL:
        case GGUFTensorType::IQ4_XS:
            return 32;
        case GGUFTensorType::Q2_K:
        case GGUFTensorType::Q3_K:
        case GGUFTensorType::Q4_K:
        case GGUFTensorType::Q5_K:
        case GGUFTensorType::Q6_K:
        case GGUFTensorType::Q8_K:
        case GGUFTensorType::IQ2_XXS:
        case GGUFTensorType::IQ2_XS:
        case GGUFTensorType::IQ3_XXS:
        case GGUFTensorType::IQ1_S:
        case GGUFTensorType::IQ3_S:
        case GGUFTensorType::IQ2_S:
        case GGUFTensorType::IQ1_M:
            return 256;
        default:
            return 0;
        }
    }

    // =============================================================================
    // GGUF MODEL HELPERS
    // =============================================================================

    bool GGUFModel::hasMetadata(const std::string &key) const
    {
        return metadata.find(key) != metadata.end();
    }

    GGUFTensorInfo *GGUFModel::findTensor(const std::string &name)
    {
        for (auto &t : tensors)
        {
            if (t.name == name)
                return &t;
        }
        return nullptr;
    }

    const GGUFTensorInfo *GGUFModel::findTensor(const std::string &name) const
    {
        for (const auto &t : tensors)
        {
            if (t.name == name)
                return &t;
        }
        return nullptr;
    }

    // =============================================================================
    // GGUF LOADER
    // =============================================================================

    ModelLoader::ModelLoader(TensorFactory *factory)
        : factory_(factory), loaded_(false) {}

    bool ModelLoader::loadModel(const std::string &file_path)
    {
        // Close any existing file
        if (file_stream_.is_open())
        {
            file_stream_.close();
        }

        // Open Model file
        file_stream_.open(file_path, std::ios::binary);
        if (!file_stream_)
        {
            std::cerr << "[ModelLoader] Failed to open file: " << file_path << std::endl;
            return false;
        }

        file_path_ = file_path;

        // Parse Model structure
        if (!parseHeader())
        {
            std::cerr << "[ModelLoader] Failed to parse header" << std::endl;
            return false;
        }

        if (!parseMetadata())
        {
            std::cerr << "[ModelLoader] Failed to parse metadata" << std::endl;
            return false;
        }

        if (!parseTensorInfo())
        {
            std::cerr << "[ModelLoader] Failed to parse tensor info" << std::endl;
            return false;
        }

        // Extract model hyperparameters from metadata
        extractModelMetadata();

        // Calculate data offset (32-byte aligned after header/metadata)
        std::streampos pos = file_stream_.tellg();
        uint64_t cur = static_cast<uint64_t>(pos);
        uint64_t align = model_.alignment ? model_.alignment : 32;
        uint64_t aligned = (cur + align - 1) / align * align;

        if (aligned != cur)
        {
            file_stream_.seekg(static_cast<std::streamoff>(aligned), std::ios::beg);
            if (!file_stream_)
            {
                std::cerr << "[ModelLoader] Failed to seek to aligned data offset" << std::endl;
                return false;
            }
        }

        model_.data_offset = aligned;

        std::cout << "[ModelLoader] Loaded " << file_path << std::endl;
        std::cout << "  Architecture: " << model_.architecture << std::endl;
        std::cout << "  Layers: " << model_.block_count << std::endl;
        std::cout << "  Hidden size: " << model_.embedding_length << std::endl;
        std::cout << "  Vocab size: " << model_.vocab_size << std::endl;
        std::cout << "  Heads: " << model_.head_count << " (KV: " << model_.head_count_kv << ")" << std::endl;
        std::cout << "  Tensors: " << model_.tensor_count << std::endl;

        loaded_ = true;
        return true;
    }

    std::shared_ptr<TensorBase> ModelLoader::loadTensor(const std::string &tensor_name, int device_idx)
    {
        if (!loaded_)
        {
            std::cerr << "[ModelLoader] Model not loaded" << std::endl;
            return nullptr;
        }

        // Find tensor metadata
        const GGUFTensorInfo *info = model_.findTensor(tensor_name);
        if (!info)
        {
            std::cerr << "[ModelLoader] Tensor not found: " << tensor_name << std::endl;
            return nullptr;
        }

        // Seek to tensor data
        file_stream_.seekg(model_.data_offset + info->offset, std::ios::beg);
        if (!file_stream_)
        {
            std::cerr << "[ModelLoader] Failed to seek to tensor: " << tensor_name << std::endl;
            return nullptr;
        }

        // Read raw bytes
        std::vector<uint8_t> raw(info->size_bytes);
        if (!file_stream_.read(reinterpret_cast<char *>(raw.data()), raw.size()))
        {
            std::cerr << "[ModelLoader] Failed to read tensor data: " << tensor_name << std::endl;
            return nullptr;
        }

        // Convert dimensions to size_t vector (V2 uses size_t, not int)
        std::vector<size_t> shape;
        for (auto d : info->dimensions)
        {
            shape.push_back(static_cast<size_t>(d));
        }

        // TODO: Use device_idx for device placement when V2 supports it
        (void)device_idx; // Suppress unused parameter warning

        // Create typed tensor based on GGUF type
        std::shared_ptr<TensorBase> tensor;

        switch (info->type)
        {
        case GGUFTensorType::F32:
            // FP32: Copy raw bytes as float array
            if (factory_)
            {
                auto fp32_tensor = factory_->createFP32(shape);
                std::memcpy(fp32_tensor->mutable_data(), raw.data(), raw.size());
                tensor = std::move(fp32_tensor);
            }
            else
            {
                tensor = std::make_shared<FP32Tensor>(shape);
                std::memcpy(tensor->mutable_data(), raw.data(), raw.size());
            }
            break;

        case GGUFTensorType::F16:
            // FP16: Raw data is already in FP16 format (uint16_t)
            if (factory_)
            {
                // Convert raw bytes to uint16_t vector
                std::vector<uint16_t> fp16_data(raw.size() / 2);
                std::memcpy(fp16_data.data(), raw.data(), raw.size());
                tensor = factory_->createFP16(shape, fp16_data);
            }
            else
            {
                std::vector<uint16_t> fp16_data(raw.size() / 2);
                std::memcpy(fp16_data.data(), raw.data(), raw.size());
                tensor = std::make_shared<FP16Tensor>(shape, fp16_data);
            }
            break;

        // IQ formats (4-bit, 2-bit, 3-bit, 1-bit non-linear quantization)
        case GGUFTensorType::IQ4_NL:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ4_NL, shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ4_NLTensor>(shape, raw);
            }
            break;

        case GGUFTensorType::IQ4_XS:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ4_XS, shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ4_XSTensor>(shape, raw);
            }
            break;

        case GGUFTensorType::IQ2_XXS:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ2_XXS, shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ2_XXSTensor>(shape, raw);
            }
            break;

        case GGUFTensorType::IQ2_XS:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ2_XS, shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ2_XSTensor>(shape, raw);
            }
            break;

        case GGUFTensorType::IQ3_XXS:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ3_XXS, shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ3_XXSTensor>(shape, raw);
            }
            break;

        case GGUFTensorType::IQ2_S:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ2_S, shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ2_STensor>(shape, raw);
            }
            break;

        case GGUFTensorType::IQ3_S:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ3_S, shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ3_STensor>(shape, raw);
            }
            break;

        case GGUFTensorType::IQ1_S:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ1_S, shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ1_STensor>(shape, raw);
            }
            break;

        case GGUFTensorType::IQ1_M:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ1_M, shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ1_MTensor>(shape, raw);
            }
            break;

        // Simple quantization formats (8-bit, 4-bit)
        case GGUFTensorType::Q8_0:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q8_0, shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q8_0Tensor>(shape, raw);
            }
            break;

        case GGUFTensorType::Q4_0:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q4_0, shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q4_0Tensor>(shape, raw);
            }
            break;

        case GGUFTensorType::Q4_1:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q4_1, shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q4_1Tensor>(shape, raw);
            }
            break;

        // K-quant formats (6-bit, 5-bit, 3-bit, 2-bit with hierarchical scales)
        case GGUFTensorType::Q6_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q6_K, shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q6_KTensor>(shape, raw);
            }
            break;

        case GGUFTensorType::Q5_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q5_K, shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q5_KTensor>(shape, raw);
            }
            break;

        case GGUFTensorType::Q3_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q3_K, shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q3_KTensor>(shape, raw);
            }
            break;

        case GGUFTensorType::Q2_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q2_K, shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q2_KTensor>(shape, raw);
            }
            break;

        default:
            std::cerr << "[ModelLoader] Unsupported tensor type: "
                      << static_cast<int>(info->type) << " for tensor: " << tensor_name << std::endl;
            return nullptr;
        }

        return tensor;
    }

    // =============================================================================
    // PARSING HELPERS
    // =============================================================================

    bool ModelLoader::parseHeader()
    {
        // Read GGUF magic number
        char magic[4];
        if (!file_stream_.read(magic, 4) || std::string(magic, 4) != "GGUF")
        {
            std::cerr << "[ModelLoader] Invalid magic number (not a GGUF file)" << std::endl;
            return false;
        }

        // Read version
        if (!readValue(model_.version))
        {
            std::cerr << "[ModelLoader] Failed to read version" << std::endl;
            return false;
        }

        // Read tensor count
        if (!readValue(model_.tensor_count))
        {
            std::cerr << "[ModelLoader] Failed to read tensor count" << std::endl;
            return false;
        }

        // Read metadata count
        if (!readValue(model_.metadata_kv_count))
        {
            std::cerr << "[ModelLoader] Failed to read metadata count" << std::endl;
            return false;
        }

        std::cout << "[ModelLoader] Header: version=" << model_.version
                  << ", tensors=" << model_.tensor_count
                  << ", metadata=" << model_.metadata_kv_count << std::endl;

        return true;
    }

    bool ModelLoader::readString(std::string &str)
    {
        uint64_t len;
        if (!readValue(len))
            return false;

        if (len > 1000000)
        { // 1MB sanity check
            std::cerr << "[ModelLoader] String length too large: " << len << std::endl;
            return false;
        }

        std::vector<char> buffer(len);
        if (!file_stream_.read(buffer.data(), len))
            return false;

        str.assign(buffer.data(), len);
        return true;
    }

    bool ModelLoader::readArray(GGUFValue &value)
    {
        // Read array element type
        uint32_t elem_type;
        if (!readValue(elem_type))
            return false;

        // Read array length
        uint64_t array_len;
        if (!readValue(array_len))
            return false;

        // Sanity check on array length
        if (array_len > 1000000)
        {
            std::cerr << "[ModelLoader] Array length too large: " << array_len << std::endl;
            return false;
        }

        // Actually read the array data (don't skip it!)
        value.type = GGUFValueType::ARRAY;

        // Determine element size
        size_t elem_size = 0;
        GGUFValueType elem_value_type = static_cast<GGUFValueType>(elem_type);

        switch (elem_value_type)
        {
        case GGUFValueType::UINT8:
        case GGUFValueType::INT8:
        case GGUFValueType::BOOL:
            elem_size = 1;
            break;
        case GGUFValueType::UINT16:
        case GGUFValueType::INT16:
            elem_size = 2;
            break;
        case GGUFValueType::UINT32:
        case GGUFValueType::INT32:
        case GGUFValueType::FLOAT32:
            elem_size = 4;
            break;
        case GGUFValueType::UINT64:
        case GGUFValueType::INT64:
        case GGUFValueType::FLOAT64:
            elem_size = 8;
            break;
        case GGUFValueType::STRING:
            // String arrays need special handling (variable length)
            // For now, skip them properly by reading each string
            for (uint64_t i = 0; i < array_len; ++i)
            {
                std::string str;
                if (!readString(str))
                    return false;
            }
            return true;
        default:
            std::cerr << "[ModelLoader] Unknown array element type: " << elem_type << std::endl;
            return false;
        }

        // Read fixed-size array elements
        size_t total_bytes = elem_size * array_len;
        value.data.resize(total_bytes);
        if (!file_stream_.read(reinterpret_cast<char *>(value.data.data()), total_bytes))
        {
            std::cerr << "[ModelLoader] Failed to read array data" << std::endl;
            return false;
        }

        return true;
    }

    bool ModelLoader::parseMetadata()
    {
        for (uint64_t i = 0; i < model_.metadata_kv_count; ++i)
        {
            // Read key
            std::string key;
            if (!readString(key))
            {
                std::cerr << "[ModelLoader] Failed to read metadata key " << i << std::endl;
                return false;
            }

            // Read value type
            uint32_t value_type;
            if (!readValue(value_type))
            {
                std::cerr << "[ModelLoader] Failed to read value type for key: " << key << std::endl;
                return false;
            }

            GGUFValue value;
            value.type = static_cast<GGUFValueType>(value_type);

            // Handle different value types
            if (value.type == GGUFValueType::ARRAY)
            {
                if (!readArray(value))
                    return false;
            }
            else if (value.type == GGUFValueType::STRING)
            {
                uint64_t str_len;
                if (!readValue(str_len))
                    return false;

                if (str_len > 1000000)
                {
                    std::cerr << "[ModelLoader] String value too large: " << str_len << std::endl;
                    return false;
                }

                value.data.resize(8 + str_len);
                std::memcpy(value.data.data(), &str_len, 8);
                if (!file_stream_.read(reinterpret_cast<char *>(value.data.data() + 8), str_len))
                {
                    return false;
                }
            }
            else
            {
                // Read fixed-size value
                size_t value_size = 0;
                switch (value.type)
                {
                case GGUFValueType::UINT8:
                case GGUFValueType::INT8:
                case GGUFValueType::BOOL:
                    value_size = 1;
                    break;
                case GGUFValueType::UINT16:
                case GGUFValueType::INT16:
                    value_size = 2;
                    break;
                case GGUFValueType::UINT32:
                case GGUFValueType::INT32:
                case GGUFValueType::FLOAT32:
                    value_size = 4;
                    break;
                case GGUFValueType::UINT64:
                case GGUFValueType::INT64:
                case GGUFValueType::FLOAT64:
                    value_size = 8;
                    break;
                default:
                    std::cerr << "[ModelLoader] Unknown value type: " << value_type << std::endl;
                    return false;
                }

                value.data.resize(value_size);
                if (!file_stream_.read(reinterpret_cast<char *>(value.data.data()), value_size))
                {
                    return false;
                }
            }

            model_.metadata[key] = std::move(value);
        }

        return true;
    }

    bool ModelLoader::parseTensorInfo()
    {
        model_.tensors.resize(model_.tensor_count);

        for (uint64_t i = 0; i < model_.tensor_count; ++i)
        {
            auto &tensor = model_.tensors[i];

            // Read tensor name
            if (!readString(tensor.name))
            {
                std::cerr << "[ModelLoader] Failed to read tensor name " << i << std::endl;
                return false;
            }

            // Read number of dimensions
            uint32_t n_dims;
            if (!readValue(n_dims))
            {
                std::cerr << "[ModelLoader] Failed to read dimensions for: " << tensor.name << std::endl;
                return false;
            }

            // Read dimensions
            tensor.dimensions.resize(n_dims);
            for (uint32_t j = 0; j < n_dims; ++j)
            {
                if (!readValue(tensor.dimensions[j]))
                {
                    std::cerr << "[ModelLoader] Failed to read dimension " << j
                              << " for: " << tensor.name << std::endl;
                    return false;
                }
            }

            // CRITICAL: GGUF dimension quirk - metadata is backwards from actual data
            // For 2D tensors, swap dimensions to match actual file layout
            // See V1 ModelLoader.cpp:1620 for detailed explanation
            if (n_dims == 2)
            {
                std::swap(tensor.dimensions[0], tensor.dimensions[1]);
            }

            // Read tensor type
            uint32_t type_val;
            if (!readValue(type_val))
            {
                std::cerr << "[ModelLoader] Failed to read type for: " << tensor.name << std::endl;
                return false;
            }
            tensor.type = static_cast<GGUFTensorType>(type_val);

            // Read tensor offset
            if (!readValue(tensor.offset))
            {
                std::cerr << "[ModelLoader] Failed to read offset for: " << tensor.name << std::endl;
                return false;
            }

            // Calculate size in bytes
            size_t n_elems = 1;
            for (auto d : tensor.dimensions)
                n_elems *= d;

            if (tensor.type == GGUFTensorType::F32)
            {
                tensor.size_bytes = n_elems * 4;
            }
            else if (tensor.type == GGUFTensorType::F16)
            {
                tensor.size_bytes = n_elems * 2;
            }
            else if (tensor.type == GGUFTensorType::BF16)
            {
                tensor.size_bytes = n_elems * 2;
            }
            else if (tensor.isQuantized())
            {
                size_t block_size = tensor.getBlockSize();
                size_t type_size = tensor.getTypeSize();
                size_t n_blocks = (n_elems + block_size - 1) / block_size;
                tensor.size_bytes = n_blocks * type_size;
            }
            else
            {
                std::cerr << "[ModelLoader] Unknown tensor type: " << type_val
                          << " for: " << tensor.name << std::endl;
                return false;
            }
        }

        return true;
    }

    void ModelLoader::extractModelMetadata()
    {
        // Debug: Print all metadata keys
        std::cout << "[ModelLoader] Available metadata keys:\n";
        for (const auto &kv : model_.metadata)
        {
            std::cout << "  " << kv.first << " (type=" << static_cast<int>(kv.second.type) << ")\n";
        }

        // Extract common hyperparameters from metadata
        auto get_uint = [this](const std::string &key) -> uint64_t
        {
            auto it = model_.metadata.find(key);
            if (it != model_.metadata.end())
            {
                // Try both uint32 and uint64
                if (it->second.type == GGUFValueType::UINT32)
                {
                    return static_cast<uint64_t>(it->second.asUInt32());
                }
                else if (it->second.type == GGUFValueType::UINT64)
                {
                    return it->second.asUInt64();
                }
            }
            return 0;
        };

        auto get_float = [this](const std::string &key) -> float
        {
            auto it = model_.metadata.find(key);
            if (it != model_.metadata.end())
            {
                return it->second.asFloat32();
            }
            return 0.0f;
        };

        auto get_string = [this](const std::string &key) -> std::string
        {
            auto it = model_.metadata.find(key);
            if (it != model_.metadata.end())
            {
                return it->second.asString();
            }
            return "";
        };

        // Extract architecture
        model_.architecture = get_string("general.architecture");

        // Extract hyperparameters using architecture-specific keys
        // Qwen2 uses "qwen2." prefix
        std::string arch_prefix = model_.architecture + ".";

        model_.context_length = get_uint(arch_prefix + "context_length");
        model_.embedding_length = get_uint(arch_prefix + "embedding_length");
        model_.block_count = get_uint(arch_prefix + "block_count");
        model_.head_count = get_uint(arch_prefix + "attention.head_count");
        model_.head_count_kv = get_uint(arch_prefix + "attention.head_count_kv");

        // RoPE theta
        float theta = get_float(arch_prefix + "rope.freq_base");
        if (theta > 0.0f)
        {
            model_.rope_theta = theta;
        }

        // Vocab size from tokenizer metadata
        // Try multiple common locations
        model_.vocab_size = get_uint("tokenizer.ggml.token_count");
        if (model_.vocab_size == 0)
        {
            model_.vocab_size = get_uint("tokenizer.ggml.tokens.length");
        }

        // Debug output
        std::cout << "[ModelLoader] Extracted metadata:\n";
        std::cout << "  architecture: " << model_.architecture << "\n";
        std::cout << "  context_length: " << model_.context_length << "\n";
        std::cout << "  embedding_length: " << model_.embedding_length << "\n";
        std::cout << "  block_count: " << model_.block_count << "\n";
        std::cout << "  head_count: " << model_.head_count << "\n";
        std::cout << "  head_count_kv: " << model_.head_count_kv << "\n";
        std::cout << "  vocab_size: " << model_.vocab_size << "\n";
        std::cout << "  rope_theta: " << model_.rope_theta << "\n";
    }

} // namespace llaminar2
