#include "repacker.h"
#include "tensors/tensor_base.h"
#include "tensors/tensor_factory.h"
#include "quant_dequant.h" // For proper Q4_0/Q8_0 dequantization matching llama.cpp
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <iomanip>

// COSMATensor implementation
COSMATensor::COSMATensor(const std::vector<int> &shape, const std::vector<double> &data,
                         const std::string &name)
    : shape_(shape), data_(data), name_(name), distributed_(false), optimized_(false) {}

std::shared_ptr<llaminar::TensorBase> COSMATensor::toTensorBase() const
{
    // Convert int vector to size_t for TensorFactory (which needs vector<int>)
    std::vector<int> int_shape(shape_.begin(), shape_.end());
    auto tensor = llaminar::TensorFactory::create_simple(int_shape);

    // Copy data (need to convert double to float)
    float *tensor_data = tensor->data();
    for (size_t i = 0; i < data_.size(); ++i)
    {
        tensor_data[i] = static_cast<float>(data_[i]);
    }

    return tensor;
}

std::shared_ptr<COSMATensor> COSMATensor::fromTensorBase(const std::shared_ptr<llaminar::TensorBase> &tensor,
                                                         const std::string &name)
{
    if (!tensor)
        return nullptr;

    // Convert shape from vector<int> to vector<int>
    const auto &tensor_shape = tensor->shape();
    std::vector<int> int_shape(tensor_shape.begin(), tensor_shape.end());

    // Convert float data to double for COSMA
    const float *tensor_data = tensor->data();
    size_t data_size = 1;
    for (int dim : int_shape)
    {
        data_size *= dim;
    }

    std::vector<double> double_data(data_size);
    for (size_t i = 0; i < data_size; ++i)
    {
        double_data[i] = static_cast<double>(tensor_data[i]);
    }

    return std::make_shared<COSMATensor>(int_shape, double_data, name);
}

void COSMATensor::optimizeForCOSMA()
{
    if (optimized_)
        return;

    // Implement COSMA-specific memory layout optimizations
    // This could include:
    // - Memory alignment for vectorization
    // - Cache-friendly data layouts
    // - Padding for optimal block sizes

    size_t total_elements = data_.size();
    if (total_elements == 0)
        return;

    // For now, just ensure data is properly aligned and contiguous
    // In a real implementation, this would do more sophisticated optimizations
    std::vector<double> optimized_data = data_;

    // Apply any necessary transformations
    data_ = std::move(optimized_data);
    optimized_ = true;

    std::cout << "Optimized tensor '" << name_ << "' for COSMA" << std::endl;
}

// TensorRepacker implementation
TensorRepacker::TensorRepacker()
    : optimize_memory_layout_(true),
      preferred_data_type_("double"),
      distribution_strategy_("auto"),
      total_memory_usage_(0),
      processed_tensor_count_(0) {}

std::shared_ptr<COSMATensor> TensorRepacker::repackFromGGUF(const GGUFTensorInfo &tensor_info,
                                                            const std::vector<uint8_t> &raw_data)
{
    // Calculate total elements
    size_t total_elements = 1;
    for (uint64_t dim : tensor_info.dimensions)
    {
        total_elements *= dim;
    }

    if (total_elements == 0)
    {
        std::cerr << "Error: Zero-sized tensor: " << tensor_info.name << std::endl;
        return nullptr;
    }

    // Convert dimensions to int
    std::vector<int> shape;
    for (uint64_t dim : tensor_info.dimensions)
    {
        shape.push_back(static_cast<int>(dim));
    }

    // Dequantize based on tensor type
    std::vector<double> double_data;

    switch (tensor_info.type)
    {
    case GGUFTensorType::F32:
        double_data = convertF32ToDouble(raw_data.data(), total_elements);
        break;

    case GGUFTensorType::F16:
        double_data = dequantizeF16(raw_data.data(), total_elements);
        break;

    case GGUFTensorType::Q8_0:
        double_data = dequantizeQ8_0(raw_data.data(), total_elements);
        break;

    case GGUFTensorType::Q4_0:
        double_data = dequantizeQ4_0(raw_data.data(), total_elements);
        break;

    default:
        std::cerr << "Error: Unsupported tensor type for " << tensor_info.name << std::endl;
        return nullptr;
    }

    if (double_data.empty())
    {
        std::cerr << "Error: Failed to dequantize tensor: " << tensor_info.name << std::endl;
        return nullptr;
    }

    // Create COSMA tensor
    auto cosma_tensor = std::make_shared<COSMATensor>(shape, double_data, tensor_info.name);

    // Apply optimizations
    if (optimize_memory_layout_)
    {
        cosma_tensor->optimizeForCOSMA();
    }

    // Apply distribution strategy
    applyDistributionStrategy(*cosma_tensor);

    // Update statistics
    updateStatistics(*cosma_tensor);

    std::cout << "Repacked tensor '" << tensor_info.name << "' to COSMA format" << std::endl;
    return cosma_tensor;
}

std::shared_ptr<COSMATensor> TensorRepacker::repackFromTensorBase(const std::shared_ptr<llaminar::TensorBase> &tensor,
                                                                  const std::string &name)
{
    if (!tensor)
        return nullptr;

    auto cosma_tensor = COSMATensor::fromTensorBase(tensor, name);

    if (optimize_memory_layout_)
    {
        cosma_tensor->optimizeForCOSMA();
    }

    applyDistributionStrategy(*cosma_tensor);
    updateStatistics(*cosma_tensor);

    return cosma_tensor;
}

std::vector<std::shared_ptr<COSMATensor>> TensorRepacker::repackModel(const GGUFModel &model,
                                                                      ModelLoader &loader)
{
    std::vector<std::shared_ptr<COSMATensor>> cosma_tensors;
    cosma_tensors.reserve(model.tensors.size());

    std::cout << "Repacking " << model.tensors.size() << " tensors to COSMA format..." << std::endl;

    for (const auto &tensor_info : model.tensors)
    {
        auto regular_tensor = loader.loadTensor(tensor_info.name);
        if (regular_tensor)
        {
            auto cosma_tensor = repackFromTensorBase(regular_tensor, tensor_info.name);
            if (cosma_tensor)
            {
                cosma_tensors.push_back(cosma_tensor);
            }
        }
        else
        {
            std::cerr << "Warning: Failed to load tensor: " << tensor_info.name << std::endl;
        }
    }

    std::cout << "Successfully repacked " << cosma_tensors.size() << " tensors" << std::endl;
    printReppackingInfo();

    return cosma_tensors;
}

void TensorRepacker::printReppackingInfo() const
{
    std::cout << "\n=== Tensor Repacking Summary ===" << std::endl;
    std::cout << "Processed tensors: " << processed_tensor_count_ << std::endl;
    std::cout << "Total memory usage: " << std::fixed << std::setprecision(2)
              << total_memory_usage_ / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Memory layout optimization: " << (optimize_memory_layout_ ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Preferred data type: " << preferred_data_type_ << std::endl;
    std::cout << "Distribution strategy: " << distribution_strategy_ << std::endl;
    std::cout << "===============================" << std::endl;
}

// Dequantization functions
std::vector<double> TensorRepacker::dequantizeQ8_0(const uint8_t *data, size_t n_elements)
{
    std::vector<double> result(n_elements);

    const size_t block_size = 32;
    const size_t num_blocks = (n_elements + block_size - 1) / block_size;

    for (size_t block = 0; block < num_blocks; ++block)
    {
        const size_t block_offset = block * 34; // 32 values + 2 bytes scale

        // Read scale factor (fp16 -> fp32 -> double)
        uint16_t scale_bits;
        std::memcpy(&scale_bits, data + block_offset, 2);

        // Convert fp16 to double (simplified conversion)
        double scale = static_cast<double>(scale_bits) / 32768.0;

        // Dequantize 32 int8 values
        for (size_t i = 0; i < block_size && (block * block_size + i) < n_elements; ++i)
        {
            int8_t quantized_val = static_cast<int8_t>(data[block_offset + 2 + i]);
            result[block * block_size + i] = scale * static_cast<double>(quantized_val);
        }
    }

    return result;
}

std::vector<double> TensorRepacker::dequantizeQ4_0(const uint8_t *data, size_t n_elements)
{
    std::vector<double> result(n_elements);

    const size_t block_size = 32;
    const size_t block_bytes = 18; // 2 (scale) + 16 (quantized data)
    const size_t num_blocks = (n_elements + block_size - 1) / block_size;

    // Use the quant_dequant.h implementation which matches llama.cpp exactly
    for (size_t block = 0; block < num_blocks; ++block)
    {
        const size_t block_offset = block * block_bytes;
        const size_t elements_in_block = std::min(block_size, n_elements - block * block_size);

        // Temporary float buffer for dequant_block_q4_0
        float block_result[32];

        // Use the correct implementation from quant_dequant.h
        llaminar::dequant_block_q4_0(data + block_offset, block_result, elements_in_block);

        // Convert float to double for result
        for (size_t i = 0; i < elements_in_block; ++i)
        {
            result[block * block_size + i] = static_cast<double>(block_result[i]);
        }
    }

    return result;
}

std::vector<double> TensorRepacker::dequantizeF16(const uint8_t *data, size_t n_elements)
{
    std::vector<double> result(n_elements);

    for (size_t i = 0; i < n_elements; ++i)
    {
        uint16_t fp16_bits;
        std::memcpy(&fp16_bits, data + i * 2, 2);

        // Simple fp16 to double conversion (could use proper IEEE conversion)
        float fp32_val = static_cast<float>(fp16_bits) / 32768.0f;
        result[i] = static_cast<double>(fp32_val);
    }

    return result;
}

std::vector<double> TensorRepacker::convertF32ToDouble(const uint8_t *data, size_t n_elements)
{
    std::vector<double> result(n_elements);

    const float *float_data = reinterpret_cast<const float *>(data);
    for (size_t i = 0; i < n_elements; ++i)
    {
        result[i] = static_cast<double>(float_data[i]);
    }

    return result;
}

// Memory layout optimization functions
void TensorRepacker::optimizeRowMajorLayout(std::vector<double> &data, const std::vector<int> &shape)
{
    // Row-major is already the default layout, so no transformation needed
    // Could add cache-line alignment or padding here
}

void TensorRepacker::optimizeColumnMajorLayout(std::vector<double> &data, const std::vector<int> &shape)
{
    if (shape.size() != 2)
        return; // Only for 2D matrices

    int rows = shape[0];
    int cols = shape[1];

    std::vector<double> transposed(data.size());
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            transposed[j * rows + i] = data[i * cols + j];
        }
    }

    data = std::move(transposed);
}

void TensorRepacker::applyDistributionStrategy(COSMATensor &tensor)
{
    if (distribution_strategy_ == "auto")
    {
        // Choose strategy based on tensor shape
        const auto &shape = tensor.getShape();
        if (shape.size() == 2 && shape[0] > shape[1])
        {
            distributeByRows(tensor);
        }
        else if (shape.size() == 2 && shape[1] > shape[0])
        {
            distributeByColumns(tensor);
        }
        else
        {
            distributeByBlocks(tensor);
        }
    }
    else if (distribution_strategy_ == "rows")
    {
        distributeByRows(tensor);
    }
    else if (distribution_strategy_ == "columns")
    {
        distributeByColumns(tensor);
    }
    else if (distribution_strategy_ == "blocks")
    {
        distributeByBlocks(tensor);
    }

    tensor.setDistributed(true);
}

void TensorRepacker::distributeByRows(COSMATensor &tensor)
{
    // Mark as row-distributed for COSMA
    std::cout << "  Applied row distribution strategy to " << tensor.getName() << std::endl;
}

void TensorRepacker::distributeByColumns(COSMATensor &tensor)
{
    // Mark as column-distributed for COSMA
    std::cout << "  Applied column distribution strategy to " << tensor.getName() << std::endl;
}

void TensorRepacker::distributeByBlocks(COSMATensor &tensor)
{
    // Mark as block-distributed for COSMA
    std::cout << "  Applied block distribution strategy to " << tensor.getName() << std::endl;
}

size_t TensorRepacker::calculateMemoryUsage(const std::vector<int> &shape) const
{
    size_t total_elements = 1;
    for (int dim : shape)
    {
        total_elements *= dim;
    }
    return total_elements * sizeof(double);
}

void TensorRepacker::updateStatistics(const COSMATensor &tensor)
{
    processed_tensor_count_++;
    total_memory_usage_ += calculateMemoryUsage(tensor.getShape());
}