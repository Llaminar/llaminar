/**
 * @file ModelLoader.cpp
 * @brief Streamlined GGUF model loader implementation for Llaminar V2
 * @author David Sanftenberg
 */

#include "../utils/Logger.h"
#include "ModelLoader.h"
#include "../tensors/Tensors.h"
#include "../tensors/TensorFactory.h"
#include "../utils/MPIContext.h"
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
            return 136; // IQ4_XS uses 256-element super-blocks (136 bytes), not 32-element blocks
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
            return 32;
        case GGUFTensorType::IQ4_XS: // IQ4_XS uses 256-element super-blocks like K-quants
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
        : factory_(factory), loaded_(false)
    {
        // Create default factory if none provided
        // This is only really used to simplify testing. Normally we expect a factory.
        if (!factory_)
        {
            // Create a single-rank MPIContext for the factory
            // (rank=0, world_size=1, no actual MPI comm)
            owned_mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
            owned_factory_ = std::make_unique<TensorFactory>(*owned_mpi_ctx_);
            factory_ = owned_factory_.get();
            LOG_WARN("[ModelLoader] Created internal TensorFactory with single-rank MPI context (testing only!)");
        }
    }

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
            LOG_ERROR("[ModelLoader] Failed to open file: " << file_path);
            return false;
        }

        file_path_ = file_path;

        // Parse Model structure
        if (!parseHeader())
        {
            LOG_ERROR("[ModelLoader] Failed to parse header");
            return false;
        }

        if (!parseMetadata())
        {
            LOG_ERROR("[ModelLoader] Failed to parse metadata");
            return false;
        }

        if (!parseTensorInfo())
        {
            LOG_ERROR("[ModelLoader] Failed to parse tensor info");
            return false;
        }

        // Extract model hyperparameters from metadata
        extractModelMetadata();

        // Check for multi-part GGUF metadata
        auto split_count_it = model_.metadata.find("split.count");
        if (split_count_it != model_.metadata.end())
        {
            model_.split_count = static_cast<uint16_t>(split_count_it->second.asUInt32());
        }

        auto split_no_it = model_.metadata.find("split.no");
        if (split_no_it != model_.metadata.end())
        {
            model_.split_no = static_cast<uint16_t>(split_no_it->second.asUInt32());
        }

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
                LOG_ERROR("[ModelLoader] Failed to seek to aligned data offset");
                return false;
            }
        }

        model_.data_offset = aligned;

        // Load additional split files if this is a multi-part GGUF
        if (model_.split_count > 1)
        {
            if (!loadSplitFiles())
            {
                LOG_ERROR("[ModelLoader] Failed to load split files");
                return false;
            }
        }

        LOG_DEBUG("[ModelLoader] Loaded " << file_path);
        LOG_DEBUG("  Architecture: " << model_.architecture);
        LOG_DEBUG("  Layers: " << model_.block_count);
        LOG_DEBUG("  Hidden size: " << model_.embedding_length);
        LOG_DEBUG("  Vocab size: " << model_.vocab_size);
        LOG_DEBUG("  Heads: " << model_.head_count << " (KV: " << model_.head_count_kv << ")");
        LOG_DEBUG("  Tensors: " << model_.tensor_count);
        if (model_.split_count > 1)
        {
            LOG_DEBUG("  Split files: " << model_.split_count);
        }

        loaded_ = true;
        return true;
    }

    void ModelLoader::initializeTestModel(uint32_t block_count)
    {
        // Initialize minimal valid GGUFModel structure for testing
        model_.version = 3;
        model_.tensor_count = 0;
        model_.metadata_kv_count = 0;
        model_.alignment = 32;
        model_.architecture = "test";
        model_.context_length = 2048;
        model_.embedding_length = 512;
        model_.block_count = block_count;
        model_.head_count = 8;
        model_.head_count_kv = 8;
        model_.vocab_size = 1000;
        model_.rope_theta = 10000.0f;
        model_.data_offset = 0;
        model_.split_count = 1;
        model_.split_no = 0;
        model_.split_paths.resize(1, "test.gguf");
        model_.split_data_offsets.resize(1, 0);

        loaded_ = true; // Mark as "loaded" so tests don't try to load actual file

        LOG_DEBUG("[ModelLoader] Initialized test model (no file loaded)");
    }

    bool ModelLoader::hasTensor(const std::string &tensor_name) const
    {
        if (!loaded_)
        {
            return false;
        }
        return model_.findTensor(tensor_name) != nullptr;
    }

    std::optional<std::vector<size_t>> ModelLoader::getTensorShape(const std::string &name) const
    {
        if (!loaded_)
        {
            return std::nullopt;
        }
        const GGUFTensorInfo *info = model_.findTensor(name);
        if (!info)
        {
            return std::nullopt;
        }
        return info->dimensions;
    }

    std::shared_ptr<TensorBase> ModelLoader::loadTensor(const std::string &tensor_name,
                                                        DeviceId device,
                                                        WeightPrecision weight_precision)
    {
        if (!loaded_)
        {
            LOG_ERROR("[ModelLoader] Model not loaded");
            return nullptr;
        }

        // Find tensor metadata
        const GGUFTensorInfo *info = model_.findTensor(tensor_name);
        if (!info)
        {
            LOG_ERROR("[ModelLoader] Tensor not found: " << tensor_name);
            return nullptr;
        }

        // Read raw bytes from file (thread-safe section)
        std::vector<uint8_t> raw;
        {
            std::lock_guard<std::mutex> lock(file_mutex_);

            // Select correct file stream based on split index
            std::ifstream *stream = &file_stream_;
            uint64_t data_offset = model_.data_offset;

            if (model_.split_count > 1)
            {
                if (info->split_idx == 0)
                {
                    // Tensor in main file
                    stream = &file_stream_;
                    data_offset = model_.split_data_offsets[0];
                }
                else if (info->split_idx < model_.split_count)
                {
                    // Tensor in split file
                    stream = &split_streams_[info->split_idx - 1];
                    data_offset = model_.split_data_offsets[info->split_idx];
                }
                else
                {
                    LOG_ERROR("[ModelLoader] Invalid split index " << info->split_idx << " for tensor: " << tensor_name);
                    return nullptr;
                }
            }

            // Seek to tensor data
            stream->seekg(data_offset + info->offset, std::ios::beg);
            if (!(*stream))
            {
                LOG_ERROR("[ModelLoader] Failed to seek to tensor: " << tensor_name);
                return nullptr;
            }

            // Read raw bytes
            raw.resize(info->size_bytes);
            if (!stream->read(reinterpret_cast<char *>(raw.data()), raw.size()))
            {
                LOG_ERROR("[ModelLoader] Failed to read tensor data: " << tensor_name);
                return nullptr;
            }
        } // Release file_mutex_ - tensor creation can proceed in parallel

        // Convert dimensions to size_t vector (V2 uses size_t, not int)
        std::vector<size_t> shape;
        for (auto d : info->dimensions)
        {
            shape.push_back(static_cast<size_t>(d));
        }

        // TODO: Use device for device placement when V2 supports it
        (void)device; // Suppress unused parameter warning

        // WEIGHT PRECISION HANDLING:
        // - NATIVE (default): Keep weights in original GGUF format, no conversion
        // - CONVERT_TO_FP32/BF16/FP16/INT8: Convert all weights to target format at load

        bool should_convert = (weight_precision != WeightPrecision::NATIVE) && info->isQuantized();

        if (should_convert)
        {
            switch (weight_precision)
            {
            case WeightPrecision::CONVERT_TO_INT8:
                return dequantizeToINT8(info, shape, raw);
            case WeightPrecision::CONVERT_TO_FP32:
                return dequantizeToFP32(info, shape, raw);
            case WeightPrecision::CONVERT_TO_BF16:
                // TODO: Implement dequantizeToBF16()
                LOG_WARN("[ModelLoader] BF16 dequantization not yet implemented, keeping quantized");
                break;
            case WeightPrecision::CONVERT_TO_FP16:
                // TODO: Implement dequantizeToFP16()
                LOG_WARN("[ModelLoader] FP16 dequantization not yet implemented, keeping quantized");
                break;
            default:
                break;
            }
        }

        // NATIVE MODE: Create typed tensor based on GGUF type (no conversion)
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

        case GGUFTensorType::BF16:
            // BF16: Raw data is in BF16 format (uint16_t, different from FP16)
            if (factory_)
            {
                // Convert raw bytes to uint16_t vector
                std::vector<uint16_t> bf16_data(raw.size() / 2);
                std::memcpy(bf16_data.data(), raw.data(), raw.size());
                tensor = factory_->createBF16(shape, bf16_data);
            }
            else
            {
                std::vector<uint16_t> bf16_data(raw.size() / 2);
                std::memcpy(bf16_data.data(), raw.data(), raw.size());
                tensor = std::make_shared<BF16Tensor>(shape, bf16_data);
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

        case GGUFTensorType::Q5_0:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q5_0, shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q5_0Tensor>(shape, raw);
            }
            break;

        case GGUFTensorType::Q5_1:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q5_1, shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q5_1Tensor>(shape, raw);
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

        case GGUFTensorType::Q4_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q4_K, shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q4_KTensor>(shape, raw);
            }
            break;

        case GGUFTensorType::Q8_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q8_K, shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q8_KTensor>(shape, raw);
            }
            break;

        default:
            LOG_ERROR("[ModelLoader] Unsupported tensor type: "
                      << static_cast<int>(info->type) << " for tensor: " << tensor_name);
            return nullptr;
        }

        return tensor;
    }

    // =============================================================================
    // ROW-SLICED TENSOR LOADING (Memory-Efficient Tensor Parallelism)
    // =============================================================================

    std::shared_ptr<TensorBase> ModelLoader::loadTensorRowSlice(
        const std::string &tensor_name,
        size_t row_start, size_t row_end,
        DeviceId device,
        WeightPrecision weight_precision)
    {
        if (!loaded_)
        {
            LOG_ERROR("[ModelLoader] Model not loaded");
            return nullptr;
        }

        // Find tensor metadata
        const GGUFTensorInfo *info = model_.findTensor(tensor_name);
        if (!info)
        {
            LOG_ERROR("[ModelLoader] Tensor not found: " << tensor_name);
            return nullptr;
        }

        // Validate tensor is 2D
        if (info->dimensions.size() != 2)
        {
            LOG_ERROR("[ModelLoader] Row slicing requires 2D tensor, got "
                      << info->dimensions.size() << "D for: " << tensor_name);
            return nullptr;
        }

        size_t total_rows = info->dimensions[0];
        size_t cols = info->dimensions[1];

        // Validate row range
        if (row_start >= total_rows || row_end > total_rows || row_start >= row_end)
        {
            LOG_ERROR("[ModelLoader] Invalid row range [" << row_start << ", " << row_end
                                                          << ") for tensor with " << total_rows << " rows: " << tensor_name);
            return nullptr;
        }

        size_t slice_rows = row_end - row_start;

        // Calculate bytes per row based on tensor type
        size_t bytes_per_row = 0;
        size_t block_size = info->getBlockSize();
        size_t type_size = info->getTypeSize();

        if (block_size > 0)
        {
            // Quantized tensor: cols must be divisible by block_size
            if (cols % block_size != 0)
            {
                LOG_ERROR("[ModelLoader] Columns (" << cols << ") not divisible by block size ("
                                                    << block_size << ") for: " << tensor_name);
                return nullptr;
            }
            size_t blocks_per_row = cols / block_size;
            bytes_per_row = blocks_per_row * type_size;
        }
        else
        {
            // Non-quantized tensor (F32, F16, BF16)
            bytes_per_row = cols * type_size;
        }

        // Calculate slice byte range
        size_t slice_offset = row_start * bytes_per_row;
        size_t slice_bytes = slice_rows * bytes_per_row;

        LOG_TRACE("[ModelLoader] Row slice " << tensor_name << ": rows [" << row_start << ", " << row_end
                                             << "), " << slice_bytes << " bytes (of " << info->size_bytes << " total)");

        // Read only the slice bytes (thread-safe section)
        std::vector<uint8_t> raw;
        {
            std::lock_guard<std::mutex> lock(file_mutex_);

            // Select correct file stream based on split index
            std::ifstream *stream = &file_stream_;
            uint64_t data_offset = model_.data_offset;

            if (model_.split_count > 1)
            {
                if (info->split_idx == 0)
                {
                    stream = &file_stream_;
                    data_offset = model_.split_data_offsets[0];
                }
                else if (info->split_idx < model_.split_count)
                {
                    stream = &split_streams_[info->split_idx - 1];
                    data_offset = model_.split_data_offsets[info->split_idx];
                }
                else
                {
                    LOG_ERROR("[ModelLoader] Invalid split index for tensor: " << tensor_name);
                    return nullptr;
                }
            }

            // Seek to slice start within tensor data
            stream->seekg(data_offset + info->offset + slice_offset, std::ios::beg);
            if (!(*stream))
            {
                LOG_ERROR("[ModelLoader] Failed to seek to row slice for: " << tensor_name);
                return nullptr;
            }

            // Read only the slice bytes
            raw.resize(slice_bytes);
            if (!stream->read(reinterpret_cast<char *>(raw.data()), raw.size()))
            {
                LOG_ERROR("[ModelLoader] Failed to read row slice data for: " << tensor_name);
                return nullptr;
            }
        } // Release file_mutex_ - tensor creation can proceed in parallel

        // Create shape for sliced tensor
        std::vector<size_t> slice_shape = {slice_rows, cols};

        // Handle weight precision conversion (same as loadTensor)
        bool should_convert = (weight_precision != WeightPrecision::NATIVE) && info->isQuantized();

        if (should_convert)
        {
            switch (weight_precision)
            {
            case WeightPrecision::CONVERT_TO_FP32:
                return dequantizeToFP32(info, slice_shape, raw);
            case WeightPrecision::CONVERT_TO_INT8:
                return dequantizeToINT8(info, slice_shape, raw);
            default:
                LOG_WARN("[ModelLoader] Unsupported conversion for sliced tensor, keeping native");
                break;
            }
        }

        // Create tensor with sliced shape and data
        // Use factory_ when available for NUMA-aware allocation and device placement
        std::shared_ptr<TensorBase> tensor;

        switch (info->type)
        {
        case GGUFTensorType::F32:
            if (factory_)
            {
                auto fp32_tensor = factory_->createFP32(slice_shape, device);
                std::memcpy(fp32_tensor->mutable_data(), raw.data(), raw.size());
                tensor = std::move(fp32_tensor);
            }
            else
            {
                tensor = std::make_shared<FP32Tensor>(slice_shape);
                std::memcpy(tensor->mutable_data(), raw.data(), raw.size());
            }
            break;

        case GGUFTensorType::F16:
        {
            std::vector<uint16_t> fp16_data(raw.size() / 2);
            std::memcpy(fp16_data.data(), raw.data(), raw.size());
            if (factory_)
            {
                tensor = factory_->createFP16(slice_shape, fp16_data);
            }
            else
            {
                tensor = std::make_shared<FP16Tensor>(slice_shape, fp16_data);
            }
            break;
        }

        case GGUFTensorType::BF16:
        {
            std::vector<uint16_t> bf16_data(raw.size() / 2);
            std::memcpy(bf16_data.data(), raw.data(), raw.size());
            if (factory_)
            {
                tensor = factory_->createBF16(slice_shape, bf16_data);
            }
            else
            {
                tensor = std::make_shared<BF16Tensor>(slice_shape, bf16_data);
            }
            break;
        }

        case GGUFTensorType::Q4_0:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q4_0, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q4_0Tensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q4_1:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q4_1, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q4_1Tensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q5_0:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q5_0, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q5_0Tensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q5_1:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q5_1, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q5_1Tensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q8_0:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q8_0, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q8_0Tensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q2_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q2_K, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q2_KTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q3_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q3_K, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q3_KTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q4_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q4_K, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q4_KTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q5_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q5_K, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q5_KTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q6_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q6_K, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q6_KTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q8_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q8_K, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q8_KTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ4_NL:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ4_NL, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ4_NLTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ4_XS:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ4_XS, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ4_XSTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ3_S:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ3_S, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ3_STensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ3_XXS:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ3_XXS, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ3_XXSTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ2_S:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ2_S, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ2_STensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ2_XS:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ2_XS, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ2_XSTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ2_XXS:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ2_XXS, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ2_XXSTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ1_S:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ1_S, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ1_STensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ1_M:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ1_M, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ1_MTensor>(slice_shape, raw);
            }
            break;

        default:
            LOG_ERROR("[ModelLoader] Unsupported tensor type for row slicing: "
                      << static_cast<int>(info->type));
            return nullptr;
        }

        return tensor;
    }

    std::shared_ptr<TensorBase> ModelLoader::loadTensorColumnSlice(
        const std::string &tensor_name,
        size_t col_start, size_t col_end,
        DeviceId device,
        WeightPrecision weight_precision)
    {
        if (!loaded_)
        {
            LOG_ERROR("[ModelLoader] Model not loaded");
            return nullptr;
        }

        // Find tensor metadata
        const GGUFTensorInfo *info = model_.findTensor(tensor_name);
        if (!info)
        {
            LOG_ERROR("[ModelLoader] Tensor not found: " << tensor_name);
            return nullptr;
        }

        // Validate tensor is 2D
        if (info->dimensions.size() != 2)
        {
            LOG_ERROR("[ModelLoader] Column slicing requires 2D tensor, got "
                      << info->dimensions.size() << "D for: " << tensor_name);
            return nullptr;
        }

        size_t total_rows = info->dimensions[0];
        size_t total_cols = info->dimensions[1];

        // Validate column range
        if (col_start >= total_cols || col_end > total_cols || col_start >= col_end)
        {
            LOG_ERROR("[ModelLoader] Invalid column range [" << col_start << ", " << col_end
                                                             << ") for tensor with " << total_cols << " columns: " << tensor_name);
            return nullptr;
        }

        size_t slice_cols = col_end - col_start;
        size_t block_size = info->getBlockSize();
        size_t type_size = info->getTypeSize();

        // For quantized tensors, validate block alignment
        if (block_size > 0)
        {
            if (col_start % block_size != 0)
            {
                LOG_ERROR("[ModelLoader] Column start (" << col_start
                                                         << ") must be aligned to block size (" << block_size
                                                         << ") for: " << tensor_name);
                return nullptr;
            }
            if (col_end % block_size != 0)
            {
                LOG_ERROR("[ModelLoader] Column end (" << col_end
                                                       << ") must be aligned to block size (" << block_size
                                                       << ") for: " << tensor_name);
                return nullptr;
            }
            if (total_cols % block_size != 0)
            {
                LOG_ERROR("[ModelLoader] Total columns (" << total_cols
                                                          << ") not divisible by block size (" << block_size
                                                          << ") for: " << tensor_name);
                return nullptr;
            }
        }

        // Calculate byte layout
        size_t bytes_per_row = 0;
        size_t bytes_per_slice_row = 0;
        size_t block_offset_bytes = 0;

        if (block_size > 0)
        {
            // Quantized tensor
            size_t blocks_per_row = total_cols / block_size;
            size_t slice_blocks_per_row = slice_cols / block_size;
            size_t start_block = col_start / block_size;

            bytes_per_row = blocks_per_row * type_size;
            bytes_per_slice_row = slice_blocks_per_row * type_size;
            block_offset_bytes = start_block * type_size;
        }
        else
        {
            // Non-quantized tensor (F32, F16, BF16)
            bytes_per_row = total_cols * type_size;
            bytes_per_slice_row = slice_cols * type_size;
            block_offset_bytes = col_start * type_size;
        }

        size_t slice_bytes = total_rows * bytes_per_slice_row;

        LOG_TRACE("[ModelLoader] Column slice " << tensor_name << ": cols [" << col_start << ", " << col_end
                                                << "), " << slice_bytes << " bytes (of " << info->size_bytes << " total)"
                                                << ", reading " << bytes_per_slice_row << " bytes from each of " << total_rows << " rows");

        // Allocate buffer for sliced data
        std::vector<uint8_t> raw(slice_bytes);

        // Read row by row, extracting only the needed column range (thread-safe section)
        {
            std::lock_guard<std::mutex> lock(file_mutex_);

            // Select correct file stream based on split index
            std::ifstream *stream = &file_stream_;
            uint64_t data_offset = model_.data_offset;

            if (model_.split_count > 1)
            {
                if (info->split_idx == 0)
                {
                    stream = &file_stream_;
                    data_offset = model_.split_data_offsets[0];
                }
                else if (info->split_idx < model_.split_count)
                {
                    stream = &split_streams_[info->split_idx - 1];
                    data_offset = model_.split_data_offsets[info->split_idx];
                }
                else
                {
                    LOG_ERROR("[ModelLoader] Invalid split index for tensor: " << tensor_name);
                    return nullptr;
                }
            }

            uint64_t tensor_start = data_offset + info->offset;
            for (size_t row = 0; row < total_rows; ++row)
            {
                // Seek to the start of this row's slice
                uint64_t row_offset = tensor_start + (row * bytes_per_row) + block_offset_bytes;
                stream->seekg(row_offset, std::ios::beg);
                if (!(*stream))
                {
                    LOG_ERROR("[ModelLoader] Failed to seek to column slice for row " << row
                                                                                      << " of: " << tensor_name);
                    return nullptr;
                }

                // Read the slice bytes for this row
                uint8_t *dst = raw.data() + (row * bytes_per_slice_row);
                if (!stream->read(reinterpret_cast<char *>(dst), bytes_per_slice_row))
                {
                    LOG_ERROR("[ModelLoader] Failed to read column slice data for row " << row
                                                                                        << " of: " << tensor_name);
                    return nullptr;
                }
            }
        } // Release file_mutex_ - tensor creation can proceed in parallel

        // Create shape for sliced tensor
        std::vector<size_t> slice_shape = {total_rows, slice_cols};

        // Handle weight precision conversion (same as loadTensor)
        bool should_convert = (weight_precision != WeightPrecision::NATIVE) && info->isQuantized();

        if (should_convert)
        {
            switch (weight_precision)
            {
            case WeightPrecision::CONVERT_TO_FP32:
                return dequantizeToFP32(info, slice_shape, raw);
            case WeightPrecision::CONVERT_TO_INT8:
                return dequantizeToINT8(info, slice_shape, raw);
            default:
                LOG_WARN("[ModelLoader] Unsupported conversion for sliced tensor, keeping native");
                break;
            }
        }

        // Create tensor with sliced shape and data (same switch as loadTensorRowSlice)
        std::shared_ptr<TensorBase> tensor;

        switch (info->type)
        {
        case GGUFTensorType::F32:
            if (factory_)
            {
                auto fp32_tensor = factory_->createFP32(slice_shape, device);
                std::memcpy(fp32_tensor->mutable_data(), raw.data(), raw.size());
                tensor = std::move(fp32_tensor);
            }
            else
            {
                tensor = std::make_shared<FP32Tensor>(slice_shape);
                std::memcpy(tensor->mutable_data(), raw.data(), raw.size());
            }
            break;

        case GGUFTensorType::F16:
        {
            std::vector<uint16_t> fp16_data(raw.size() / 2);
            std::memcpy(fp16_data.data(), raw.data(), raw.size());
            if (factory_)
            {
                tensor = factory_->createFP16(slice_shape, fp16_data);
            }
            else
            {
                tensor = std::make_shared<FP16Tensor>(slice_shape, fp16_data);
            }
            break;
        }

        case GGUFTensorType::BF16:
        {
            std::vector<uint16_t> bf16_data(raw.size() / 2);
            std::memcpy(bf16_data.data(), raw.data(), raw.size());
            if (factory_)
            {
                tensor = factory_->createBF16(slice_shape, bf16_data);
            }
            else
            {
                tensor = std::make_shared<BF16Tensor>(slice_shape, bf16_data);
            }
            break;
        }

        case GGUFTensorType::Q4_0:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q4_0, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q4_0Tensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q4_1:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q4_1, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q4_1Tensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q5_0:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q5_0, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q5_0Tensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q5_1:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q5_1, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q5_1Tensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q8_0:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q8_0, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q8_0Tensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q2_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q2_K, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q2_KTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q3_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q3_K, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q3_KTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q4_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q4_K, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q4_KTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q5_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q5_K, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q5_KTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q6_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q6_K, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q6_KTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::Q8_K:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::Q8_K, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<Q8_KTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ4_NL:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ4_NL, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ4_NLTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ4_XS:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ4_XS, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ4_XSTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ3_S:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ3_S, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ3_STensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ3_XXS:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ3_XXS, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ3_XXSTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ2_S:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ2_S, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ2_STensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ2_XS:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ2_XS, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ2_XSTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ2_XXS:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ2_XXS, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ2_XXSTensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ1_S:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ1_S, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ1_STensor>(slice_shape, raw);
            }
            break;

        case GGUFTensorType::IQ1_M:
            if (factory_)
            {
                tensor = factory_->createQuantized(TensorType::IQ1_M, slice_shape, raw);
            }
            else
            {
                tensor = std::make_shared<IQ1_MTensor>(slice_shape, raw);
            }
            break;

        default:
            LOG_ERROR("[ModelLoader] Unsupported tensor type for column slicing: "
                      << static_cast<int>(info->type));
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
            LOG_ERROR("[ModelLoader] Invalid magic number (not a GGUF file)");
            return false;
        }

        // Read version
        if (!readValue(model_.version))
        {
            LOG_ERROR("[ModelLoader] Failed to read version");
            return false;
        }

        // Read tensor count
        if (!readValue(model_.tensor_count))
        {
            LOG_ERROR("[ModelLoader] Failed to read tensor count");
            return false;
        }

        // Read metadata count
        if (!readValue(model_.metadata_kv_count))
        {
            LOG_ERROR("[ModelLoader] Failed to read metadata count");
            return false;
        }

        LOG_DEBUG("[ModelLoader] Header: version=" << model_.version
                                                   << ", tensors=" << model_.tensor_count
                                                   << ", metadata=" << model_.metadata_kv_count);

        return true;
    }

    bool ModelLoader::readString(std::string &str)
    {
        uint64_t len;
        if (!readValue(len))
            return false;

        if (len > 1000000)
        { // 1MB sanity check
            LOG_ERROR("[ModelLoader] String length too large: " << len);
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
            LOG_ERROR("[ModelLoader] Array length too large: " << array_len);
            return false;
        }

        // Actually read the array data (don't skip it!)
        value.type = GGUFValueType::ARRAY;
        value.array_length = array_len; // Store array length

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
            // Store them in string_array_value for tokenizer access
            value.string_array_value.reserve(array_len);
            for (uint64_t i = 0; i < array_len; ++i)
            {
                std::string str;
                if (!readString(str))
                    return false;
                value.string_array_value.push_back(std::move(str));
            }
            return true;
        default:
            LOG_ERROR("[ModelLoader] Unknown array element type: " << elem_type);
            return false;
        }

        // Read fixed-size array elements
        size_t total_bytes = elem_size * array_len;
        value.data.resize(total_bytes);
        if (!file_stream_.read(reinterpret_cast<char *>(value.data.data()), total_bytes))
        {
            LOG_ERROR("[ModelLoader] Failed to read array data");
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
                LOG_ERROR("[ModelLoader] Failed to read metadata key " << i);
                return false;
            }

            // Read value type
            uint32_t value_type;
            if (!readValue(value_type))
            {
                LOG_ERROR("[ModelLoader] Failed to read value type for key: " << key);
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
                    LOG_ERROR("[ModelLoader] String value too large: " << str_len);
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
                    LOG_ERROR("[ModelLoader] Unknown value type: " << value_type);
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
                LOG_ERROR("[ModelLoader] Failed to read tensor name " << i);
                return false;
            }

            // Read number of dimensions
            uint32_t n_dims;
            if (!readValue(n_dims))
            {
                LOG_ERROR("[ModelLoader] Failed to read dimensions for: " << tensor.name);
                return false;
            }

            // Read dimensions
            tensor.dimensions.resize(n_dims);
            for (uint32_t j = 0; j < n_dims; ++j)
            {
                if (!readValue(tensor.dimensions[j]))
                {
                    LOG_ERROR("[ModelLoader] Failed to read dimension " << j
                                                                        << " for: " << tensor.name);
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
                LOG_ERROR("[ModelLoader] Failed to read type for: " << tensor.name);
                return false;
            }
            tensor.type = static_cast<GGUFTensorType>(type_val);

            // Read tensor offset
            if (!readValue(tensor.offset))
            {
                LOG_ERROR("[ModelLoader] Failed to read offset for: " << tensor.name);
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
                LOG_ERROR("[ModelLoader] Unknown tensor type: " << type_val
                                                                << " for: " << tensor.name);
                return false;
            }

            // Initialize split index to 0 (main file) - will be updated by loadSplitFiles() if needed
            tensor.split_idx = 0;
        }

        return true;
    }

    void ModelLoader::extractModelMetadata()
    {
        // Debug: Print all metadata keys
        LOG_TRACE("[ModelLoader] Available metadata keys:\n");
        for (const auto &kv : model_.metadata)
        {
            LOG_TRACE("  " << kv.first << " (type=" << static_cast<int>(kv.second.type) << ")\n");
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

        // RMSNorm epsilon (default 1e-6 for Qwen2/LLaMA)
        float eps = get_float(arch_prefix + "attention.layer_norm_rms_epsilon");
        if (eps > 0.0f)
        {
            model_.rms_norm_eps = eps;
        }

        // Vocab size from tokenizer metadata
        // Try multiple common locations
        model_.vocab_size = get_uint("tokenizer.ggml.token_count");
        if (model_.vocab_size == 0)
        {
            model_.vocab_size = get_uint("tokenizer.ggml.tokens.length");
        }

        // Fallback: Read array length from tokenizer.ggml.tokens (most common case)
        if (model_.vocab_size == 0)
        {
            auto it = model_.metadata.find("tokenizer.ggml.tokens");
            if (it != model_.metadata.end() && it->second.type == GGUFValueType::ARRAY)
            {
                model_.vocab_size = it->second.asArrayLength();
                LOG_DEBUG("[ModelLoader] Extracted vocab_size from tokenizer.ggml.tokens array: " << model_.vocab_size);
            }
        }

        // Debug output
        LOG_DEBUG("[ModelLoader] Extracted metadata:\n");
        LOG_DEBUG("  architecture: " << model_.architecture);
        LOG_DEBUG("  context_length: " << model_.context_length);
        LOG_DEBUG("  embedding_length: " << model_.embedding_length);
        LOG_DEBUG("  block_count: " << model_.block_count);
        LOG_DEBUG("  head_count: " << model_.head_count);
        LOG_DEBUG("  head_count_kv: " << model_.head_count_kv);
        LOG_DEBUG("  vocab_size: " << model_.vocab_size);
        LOG_DEBUG("  rope_theta: " << model_.rope_theta);
        LOG_DEBUG("  rms_norm_eps: " << model_.rms_norm_eps);
    }

    // =============================================================================
    // MULTI-PART GGUF HELPERS
    // =============================================================================

    std::string ModelLoader::generateSplitPath(const std::string &base_path, uint16_t split_no, uint16_t split_count)
    {
        // Extract directory and filename without extension
        size_t last_slash = base_path.find_last_of("/\\");
        std::string dir = (last_slash != std::string::npos) ? base_path.substr(0, last_slash + 1) : "";
        std::string filename = (last_slash != std::string::npos) ? base_path.substr(last_slash + 1) : base_path;

        // Remove .gguf extension if present
        size_t gguf_pos = filename.rfind(".gguf");
        std::string prefix = (gguf_pos != std::string::npos) ? filename.substr(0, gguf_pos) : filename;

        // Format: prefix-00001-of-00005.gguf (1-based indexing for filename)
        char buffer[512];
        snprintf(buffer, sizeof(buffer), "%s%s-%05d-of-%05d.gguf",
                 dir.c_str(), prefix.c_str(), split_no + 1, split_count);
        return std::string(buffer);
    }

    bool ModelLoader::parseSplitPath(const std::string &split_path, std::string &prefix,
                                     uint16_t &split_no, uint16_t &split_count)
    {
        // Look for pattern: prefix-NNNNN-of-MMMMM.gguf
        size_t gguf_pos = split_path.rfind(".gguf");
        if (gguf_pos == std::string::npos)
            return false;

        std::string without_ext = split_path.substr(0, gguf_pos);

        // Find last "-of-"
        size_t of_pos = without_ext.rfind("-of-");
        if (of_pos == std::string::npos)
            return false;

        // Extract split count (after "-of-")
        std::string count_str = without_ext.substr(of_pos + 4);
        split_count = static_cast<uint16_t>(std::stoul(count_str));

        // Find second-to-last dash (before split number)
        size_t second_dash = without_ext.rfind('-', of_pos - 1);
        if (second_dash == std::string::npos)
            return false;

        // Extract split number (1-based in filename, convert to 0-based)
        std::string num_str = without_ext.substr(second_dash + 1, of_pos - second_dash - 1);
        split_no = static_cast<uint16_t>(std::stoul(num_str)) - 1;

        // Extract prefix
        prefix = without_ext.substr(0, second_dash);

        return true;
    }

    bool ModelLoader::loadSplitFiles()
    {
        // Check if this is a split model
        if (model_.split_count <= 1)
            return true; // Single file, nothing to do

        LOG_INFO("[ModelLoader] Loading multi-part GGUF: " << model_.split_count << " splits");

        // Verify main file is split 0
        if (model_.split_no != 0)
        {
            LOG_ERROR("[ModelLoader] Main file must be split 0, got split " << model_.split_no);
            return false;
        }

        // Generate paths for all splits
        model_.split_paths.resize(model_.split_count);
        model_.split_data_offsets.resize(model_.split_count);

        model_.split_paths[0] = file_path_;
        model_.split_data_offsets[0] = model_.data_offset;

        // Load additional split files (1 to split_count-1)
        split_streams_.resize(model_.split_count - 1);

        for (uint16_t idx = 1; idx < model_.split_count; ++idx)
        {
            std::string split_path = generateSplitPath(file_path_, idx, model_.split_count);
            model_.split_paths[idx] = split_path;

            LOG_INFO("[ModelLoader] Loading split " << idx << ": " << split_path);

            // Open split file
            split_streams_[idx - 1].open(split_path, std::ios::binary);
            if (!split_streams_[idx - 1])
            {
                LOG_ERROR("[ModelLoader] Failed to open split file: " << split_path);
                return false;
            }

            // Parse split header to get tensor info
            std::ifstream &stream = split_streams_[idx - 1];

            // Read GGUF magic
            uint32_t magic;
            stream.read(reinterpret_cast<char *>(&magic), sizeof(magic));
            if (magic != 0x46554747) // "GGUF"
            {
                LOG_ERROR("[ModelLoader] Invalid GGUF magic in split " << idx);
                return false;
            }

            // Read version
            uint32_t version;
            stream.read(reinterpret_cast<char *>(&version), sizeof(version));

            // Read tensor and metadata counts
            uint64_t tensor_count, kv_count;
            stream.read(reinterpret_cast<char *>(&tensor_count), sizeof(tensor_count));
            stream.read(reinterpret_cast<char *>(&kv_count), sizeof(kv_count));

            // Skip metadata KV pairs
            for (uint64_t i = 0; i < kv_count; ++i)
            {
                // Skip key
                uint64_t key_len;
                stream.read(reinterpret_cast<char *>(&key_len), sizeof(key_len));
                stream.seekg(key_len, std::ios::cur);

                // Skip value type
                uint32_t value_type;
                stream.read(reinterpret_cast<char *>(&value_type), sizeof(value_type));

                // Skip value data (complex, but we can estimate)
                // For simplicity, read and discard based on type
                // This is a simplified version - full implementation would parse all types
                switch (value_type)
                {
                case 4: // UINT32
                    stream.seekg(4, std::ios::cur);
                    break;
                case 5: // INT32
                    stream.seekg(4, std::ios::cur);
                    break;
                case 6: // FLOAT32
                    stream.seekg(4, std::ios::cur);
                    break;
                case 8: // STRING
                {
                    uint64_t str_len;
                    stream.read(reinterpret_cast<char *>(&str_len), sizeof(str_len));
                    stream.seekg(str_len, std::ios::cur);
                    break;
                }
                case 9: // ARRAY
                {
                    uint32_t arr_type;
                    uint64_t arr_len;
                    stream.read(reinterpret_cast<char *>(&arr_type), sizeof(arr_type));
                    stream.read(reinterpret_cast<char *>(&arr_len), sizeof(arr_len));
                    // Skip array elements (simplified)
                    size_t elem_size = (arr_type == 4 || arr_type == 5 || arr_type == 6) ? 4 : 8;
                    stream.seekg(elem_size * arr_len, std::ios::cur);
                    break;
                }
                default:
                    stream.seekg(8, std::ios::cur); // Guess
                }
            }

            // Parse tensor info from this split
            for (uint64_t i = 0; i < tensor_count; ++i)
            {
                GGUFTensorInfo info;

                // Read tensor name
                uint64_t name_len;
                stream.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
                info.name.resize(name_len);
                stream.read(&info.name[0], name_len);

                // Read n_dims
                uint32_t n_dims;
                stream.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));

                // Read dimensions
                info.dimensions.resize(n_dims);
                for (uint32_t d = 0; d < n_dims; ++d)
                {
                    stream.read(reinterpret_cast<char *>(&info.dimensions[d]), sizeof(uint64_t));
                }

                // Read tensor type
                uint32_t type_val;
                stream.read(reinterpret_cast<char *>(&type_val), sizeof(type_val));
                info.type = static_cast<GGUFTensorType>(type_val);

                // Read offset
                stream.read(reinterpret_cast<char *>(&info.offset), sizeof(info.offset));

                // Calculate size
                info.size_bytes = 1;
                for (auto dim : info.dimensions)
                {
                    info.size_bytes *= dim;
                }
                size_t type_size = info.getTypeSize();
                size_t block_size = info.getBlockSize();
                if (block_size > 0)
                {
                    info.size_bytes = (info.size_bytes / block_size) * type_size;
                }
                else
                {
                    info.size_bytes *= type_size;
                }

                // Mark which split this tensor belongs to
                info.split_idx = idx;

                // Add to model's tensor list
                model_.tensors.push_back(info);
            }

            // Calculate data offset (32-byte aligned)
            std::streampos pos = stream.tellg();
            uint64_t cur = static_cast<uint64_t>(pos);
            uint64_t align = model_.alignment ? model_.alignment : 32;
            uint64_t aligned = (cur + align - 1) / align * align;
            model_.split_data_offsets[idx] = aligned;

            LOG_INFO("[ModelLoader] Split " << idx << " has " << tensor_count << " tensors");
        }

        LOG_INFO("[ModelLoader] Total tensors across all splits: " << model_.tensors.size());
        return true;
    }

    // =============================================================================
    // INT8 DEQUANTIZATION HELPERS
    // =============================================================================

    std::shared_ptr<TensorBase> ModelLoader::dequantizeToINT8(
        const GGUFTensorInfo *info,
        const std::vector<size_t> &shape,
        const std::vector<uint8_t> &raw)
    {
        // Step 1: Create temporary quantized tensor using factory
        std::unique_ptr<TensorBase> temp_unique;

        if (factory_)
        {
            // Use factory for NUMA-aware allocation
            switch (info->type)
            {
            case GGUFTensorType::IQ4_NL:
                temp_unique = factory_->createQuantized(TensorType::IQ4_NL, shape, raw);
                break;
            case GGUFTensorType::IQ4_XS:
                temp_unique = factory_->createQuantized(TensorType::IQ4_XS, shape, raw);
                break;
            case GGUFTensorType::Q8_0:
                temp_unique = factory_->createQuantized(TensorType::Q8_0, shape, raw);
                break;
            case GGUFTensorType::Q4_0:
                temp_unique = factory_->createQuantized(TensorType::Q4_0, shape, raw);
                break;
            case GGUFTensorType::Q4_1:
                temp_unique = factory_->createQuantized(TensorType::Q4_1, shape, raw);
                break;
            case GGUFTensorType::Q5_0:
                std::cerr << "ModelLoader calling createQuantized for Q5_0" << std::endl;
                temp_unique = factory_->createQuantized(TensorType::Q5_0, shape, raw);
                break;
            case GGUFTensorType::Q5_1:
                temp_unique = factory_->createQuantized(TensorType::Q5_1, shape, raw);
                break;
            case GGUFTensorType::Q6_K:
                temp_unique = factory_->createQuantized(TensorType::Q6_K, shape, raw);
                break;
            case GGUFTensorType::Q2_K:
                temp_unique = factory_->createQuantized(TensorType::Q2_K, shape, raw);
                break;
            case GGUFTensorType::Q3_K:
                temp_unique = factory_->createQuantized(TensorType::Q3_K, shape, raw);
                break;
            case GGUFTensorType::Q4_K:
                temp_unique = factory_->createQuantized(TensorType::Q4_K, shape, raw);
                break;
            case GGUFTensorType::Q5_K:
                temp_unique = factory_->createQuantized(TensorType::Q5_K, shape, raw);
                break;
            case GGUFTensorType::Q8_K:
                temp_unique = factory_->createQuantized(TensorType::Q8_K, shape, raw);
                break;
            case GGUFTensorType::IQ2_XXS:
                temp_unique = factory_->createQuantized(TensorType::IQ2_XXS, shape, raw);
                break;
            case GGUFTensorType::IQ2_XS:
                temp_unique = factory_->createQuantized(TensorType::IQ2_XS, shape, raw);
                break;
            case GGUFTensorType::IQ3_XXS:
                temp_unique = factory_->createQuantized(TensorType::IQ3_XXS, shape, raw);
                break;
            case GGUFTensorType::IQ2_S:
                temp_unique = factory_->createQuantized(TensorType::IQ2_S, shape, raw);
                break;
            case GGUFTensorType::IQ3_S:
                temp_unique = factory_->createQuantized(TensorType::IQ3_S, shape, raw);
                break;
            case GGUFTensorType::IQ1_S:
                temp_unique = factory_->createQuantized(TensorType::IQ1_S, shape, raw);
                break;
            case GGUFTensorType::IQ1_M:
                temp_unique = factory_->createQuantized(TensorType::IQ1_M, shape, raw);
                break;
            default:
                LOG_ERROR("[ModelLoader] INT8 dequantization not supported for type: "
                          << static_cast<int>(info->type));
                return nullptr;
            }
        }
        else
        {
            // Fallback: create without factory
            switch (info->type)
            {
            case GGUFTensorType::IQ4_NL:
                temp_unique = std::make_unique<IQ4_NLTensor>(shape, raw);
                break;
            case GGUFTensorType::IQ4_XS:
                temp_unique = std::make_unique<IQ4_XSTensor>(shape, raw);
                break;
            case GGUFTensorType::Q8_0:
                temp_unique = std::make_unique<Q8_0Tensor>(shape, raw);
                break;
            case GGUFTensorType::Q4_0:
                temp_unique = std::make_unique<Q4_0Tensor>(shape, raw);
                break;
            case GGUFTensorType::Q4_1:
                temp_unique = std::make_unique<Q4_1Tensor>(shape, raw);
                break;
            case GGUFTensorType::Q5_0:
                temp_unique = std::make_unique<Q5_0Tensor>(shape, raw);
                break;
            case GGUFTensorType::Q5_1:
                temp_unique = std::make_unique<Q5_1Tensor>(shape, raw);
                break;
            case GGUFTensorType::Q6_K:
                temp_unique = std::make_unique<Q6_KTensor>(shape, raw);
                break;
            case GGUFTensorType::Q2_K:
                temp_unique = std::make_unique<Q2_KTensor>(shape, raw);
                break;
            case GGUFTensorType::Q3_K:
                temp_unique = std::make_unique<Q3_KTensor>(shape, raw);
                break;
            case GGUFTensorType::Q4_K:
                temp_unique = std::make_unique<Q4_KTensor>(shape, raw);
                break;
            case GGUFTensorType::Q5_K:
                temp_unique = std::make_unique<Q5_KTensor>(shape, raw);
                break;
            case GGUFTensorType::Q8_K:
                temp_unique = std::make_unique<Q8_KTensor>(shape, raw);
                break;
            case GGUFTensorType::IQ2_XXS:
                temp_unique = std::make_unique<IQ2_XXSTensor>(shape, raw);
                break;
            case GGUFTensorType::IQ2_XS:
                temp_unique = std::make_unique<IQ2_XSTensor>(shape, raw);
                break;
            case GGUFTensorType::IQ3_XXS:
                temp_unique = std::make_unique<IQ3_XXSTensor>(shape, raw);
                break;
            case GGUFTensorType::IQ2_S:
                temp_unique = std::make_unique<IQ2_STensor>(shape, raw);
                break;
            case GGUFTensorType::IQ3_S:
                temp_unique = std::make_unique<IQ3_STensor>(shape, raw);
                break;
            case GGUFTensorType::IQ1_S:
                temp_unique = std::make_unique<IQ1_STensor>(shape, raw);
                break;
            case GGUFTensorType::IQ1_M:
                temp_unique = std::make_unique<IQ1_MTensor>(shape, raw);
                break;
            default:
                LOG_ERROR("[ModelLoader] INT8 dequantization not supported for type: "
                          << static_cast<int>(info->type));
                return nullptr;
            }
        }

        if (!temp_unique)
        {
            LOG_ERROR("[ModelLoader] Failed to create temporary tensor for INT8 dequantization");
            return nullptr;
        }

        // Step 2: Use tensor's built-in to_int8_blocked() method
        TensorBase *temp_tensor = temp_unique.get();
        size_t total_elements = 1;
        for (auto dim : shape)
            total_elements *= dim;

        // Use per-tensor quantization (single scale for entire tensor)
        constexpr size_t BLOCK_SIZE = 256; // Standard block size
        size_t num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        std::vector<int8_t> int8_data(total_elements);
        std::vector<float> scales(num_blocks);

        temp_tensor->to_int8_blocked(int8_data.data(), scales.data(), BLOCK_SIZE);

        // Step 3: Create INT8Tensor with per-tensor scale (average of block scales)
        float avg_scale = 0.0f;
        for (auto scale : scales)
            avg_scale += scale;
        avg_scale /= static_cast<float>(num_blocks);

        return std::make_shared<INT8Tensor>(shape, int8_data, avg_scale);
    }

    // =============================================================================
    // FP32 DEQUANTIZATION
    // =============================================================================

    std::shared_ptr<TensorBase> ModelLoader::dequantizeToFP32(
        const GGUFTensorInfo *info,
        const std::vector<size_t> &shape,
        const std::vector<uint8_t> &raw)
    {
        // Calculate total elements
        size_t total_elements = 1;
        for (auto dim : shape)
            total_elements *= dim;

        // Allocate FP32 buffer
        std::vector<float> fp32_data(total_elements);

        // Dequantize based on tensor type
        switch (info->type)
        {
        case GGUFTensorType::F32:
            // Already FP32 - just copy
            std::memcpy(fp32_data.data(), raw.data(), total_elements * sizeof(float));
            break;

        case GGUFTensorType::F16:
        {
            // FP16 -> FP32 conversion
            // Reinterpret uint8_t raw bytes as uint16_t
            const uint16_t *fp16_raw = reinterpret_cast<const uint16_t *>(raw.data());
            std::vector<uint16_t> fp16_data(fp16_raw, fp16_raw + total_elements);
            auto temp_tensor = std::make_shared<FP16Tensor>(shape, fp16_data);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::Q4_0:
        {
            auto temp_tensor = std::make_shared<Q4_0Tensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::Q4_1:
        {
            auto temp_tensor = std::make_shared<Q4_1Tensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::Q5_0:
        {
            auto temp_tensor = std::make_shared<Q5_0Tensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::Q5_1:
        {
            auto temp_tensor = std::make_shared<Q5_1Tensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::Q8_0:
        {
            auto temp_tensor = std::make_shared<Q8_0Tensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::Q6_K:
        {
            auto temp_tensor = std::make_shared<Q6_KTensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::Q2_K:
        {
            auto temp_tensor = std::make_shared<Q2_KTensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::Q3_K:
        {
            auto temp_tensor = std::make_shared<Q3_KTensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::Q4_K:
        {
            auto temp_tensor = std::make_shared<Q4_KTensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::Q5_K:
        {
            auto temp_tensor = std::make_shared<Q5_KTensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::Q8_K:
        {
            auto temp_tensor = std::make_shared<Q8_KTensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::IQ4_NL:
        {
            auto temp_tensor = std::make_shared<IQ4_NLTensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::IQ4_XS:
        {
            auto temp_tensor = std::make_shared<IQ4_XSTensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::IQ2_XXS:
        {
            auto temp_tensor = std::make_shared<IQ2_XXSTensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::IQ2_XS:
        {
            auto temp_tensor = std::make_shared<IQ2_XSTensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::IQ3_XXS:
        {
            auto temp_tensor = std::make_shared<IQ3_XXSTensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::IQ2_S:
        {
            auto temp_tensor = std::make_shared<IQ2_STensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::IQ3_S:
        {
            auto temp_tensor = std::make_shared<IQ3_STensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::IQ1_S:
        {
            auto temp_tensor = std::make_shared<IQ1_STensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        case GGUFTensorType::IQ1_M:
        {
            auto temp_tensor = std::make_shared<IQ1_MTensor>(shape, raw);
            temp_tensor->to_fp32(fp32_data.data());
            break;
        }

        default:
            LOG_ERROR("[ModelLoader] FP32 dequantization not supported for type: "
                      << static_cast<int>(info->type));
            return nullptr;
        }

        // Create FP32 tensor using factory (NUMA-aware allocation)
        if (!factory_)
        {
            LOG_ERROR("[ModelLoader] TensorFactory is required for FP32 dequantization but is null");
            return nullptr;
        }

        auto fp32_tensor = factory_->createFP32(shape);
        std::memcpy(fp32_tensor->mutable_data(), fp32_data.data(), total_elements * sizeof(float));

        return fp32_tensor;
    }

    void ModelLoader::dequantizeIQ4_NLToFP32(
        const std::vector<uint8_t> &raw,
        std::vector<float> &fp32_buffer,
        const std::vector<size_t> &shape)
    {
        // Use IQ4_NLTensor to do the decoding
        auto temp_tensor = std::make_shared<IQ4_NLTensor>(shape, raw);
        temp_tensor->to_fp32(fp32_buffer.data());
    }

    void ModelLoader::dequantizeQ8_0ToFP32(
        const std::vector<uint8_t> &raw,
        std::vector<float> &fp32_buffer,
        const std::vector<size_t> &shape)
    {
        auto temp_tensor = std::make_shared<Q8_0Tensor>(shape, raw);
        temp_tensor->to_fp32(fp32_buffer.data());
    }

    void ModelLoader::dequantizeQ4_0ToFP32(
        const std::vector<uint8_t> &raw,
        std::vector<float> &fp32_buffer,
        const std::vector<size_t> &shape)
    {
        auto temp_tensor = std::make_shared<Q4_0Tensor>(shape, raw);
        temp_tensor->to_fp32(fp32_buffer.data());
    }

    void ModelLoader::dequantizeQ6_KToFP32(
        const std::vector<uint8_t> &raw,
        std::vector<float> &fp32_buffer,
        const std::vector<size_t> &shape)
    {
        auto temp_tensor = std::make_shared<Q6_KTensor>(shape, raw);
        temp_tensor->to_fp32(fp32_buffer.data());
    }

    // =============================================================================
    // IModelLoader INTERFACE IMPLEMENTATIONS
    // =============================================================================

    std::vector<std::string> ModelLoader::tensorNames() const
    {
        std::vector<std::string> names;
        names.reserve(model_.tensors.size());
        for (const auto &t : model_.tensors)
        {
            names.push_back(t.name);
        }
        return names;
    }

    size_t ModelLoader::totalBytes() const
    {
        size_t total = 0;
        for (const auto &t : model_.tensors)
        {
            total += t.size_bytes;
        }
        return total;
    }

    int ModelLoader::getInt(const std::string &key, int default_val) const
    {
        // Try direct model fields first (common hyperparameters)
        if (key == "block_count")
            return static_cast<int>(model_.block_count);
        if (key == "head_count")
            return static_cast<int>(model_.head_count);
        if (key == "head_count_kv")
            return static_cast<int>(model_.head_count_kv);

        // Search metadata with architecture prefix
        std::string full_key = model_.architecture + "." + key;
        auto it = model_.metadata.find(full_key);
        if (it != model_.metadata.end())
        {
            if (it->second.type == GGUFValueType::UINT32)
                return static_cast<int>(it->second.asUInt32());
            if (it->second.type == GGUFValueType::INT32)
            {
                int32_t val;
                std::memcpy(&val, it->second.data.data(), 4);
                return val;
            }
        }

        // Try without architecture prefix
        it = model_.metadata.find(key);
        if (it != model_.metadata.end())
        {
            if (it->second.type == GGUFValueType::UINT32)
                return static_cast<int>(it->second.asUInt32());
        }

        return default_val;
    }

    uint64_t ModelLoader::getUInt64(const std::string &key, uint64_t default_val) const
    {
        // Try direct model fields first (common hyperparameters)
        if (key == "block_count")
            return model_.block_count;
        if (key == "embedding_length")
            return model_.embedding_length;
        if (key == "context_length")
            return model_.context_length;
        if (key == "head_count")
            return model_.head_count;
        if (key == "head_count_kv")
            return model_.head_count_kv;
        if (key == "vocab_size")
            return model_.vocab_size;

        // Search metadata with architecture prefix
        std::string full_key = model_.architecture + "." + key;
        auto it = model_.metadata.find(full_key);
        if (it != model_.metadata.end())
        {
            if (it->second.type == GGUFValueType::UINT64)
                return it->second.asUInt64();
            if (it->second.type == GGUFValueType::UINT32)
                return static_cast<uint64_t>(it->second.asUInt32());
        }

        // Try without architecture prefix
        it = model_.metadata.find(key);
        if (it != model_.metadata.end())
        {
            if (it->second.type == GGUFValueType::UINT64)
                return it->second.asUInt64();
            if (it->second.type == GGUFValueType::UINT32)
                return static_cast<uint64_t>(it->second.asUInt32());
        }

        return default_val;
    }

    float ModelLoader::getFloat(const std::string &key, float default_val) const
    {
        // Try direct model fields first
        if (key == "rope_theta")
            return model_.rope_theta;
        if (key == "rms_norm_eps")
            return model_.rms_norm_eps;

        // Search metadata with architecture prefix
        std::string full_key = model_.architecture + "." + key;
        auto it = model_.metadata.find(full_key);
        if (it != model_.metadata.end())
        {
            if (it->second.type == GGUFValueType::FLOAT32)
                return it->second.asFloat32();
        }

        // Try without architecture prefix
        it = model_.metadata.find(key);
        if (it != model_.metadata.end())
        {
            if (it->second.type == GGUFValueType::FLOAT32)
                return it->second.asFloat32();
        }

        return default_val;
    }

    std::string ModelLoader::getString(const std::string &key, const std::string &default_val) const
    {
        // Try direct model fields first
        if (key == "architecture")
            return model_.architecture;

        // Search metadata with prefix
        std::string full_key = model_.architecture + "." + key;
        auto it = model_.metadata.find(full_key);
        if (it != model_.metadata.end())
        {
            if (it->second.type == GGUFValueType::STRING)
                return it->second.asString();
        }

        // Try general prefix
        full_key = "general." + key;
        it = model_.metadata.find(full_key);
        if (it != model_.metadata.end())
        {
            if (it->second.type == GGUFValueType::STRING)
                return it->second.asString();
        }

        // Try without prefix
        it = model_.metadata.find(key);
        if (it != model_.metadata.end())
        {
            if (it->second.type == GGUFValueType::STRING)
                return it->second.asString();
        }

        return default_val;
    }

} // namespace llaminar2
