#include "tensor_factory.h"
#include "tensor_base.h"
#include "simple_tensor.h"
#include "cosma_tensor.h"
#include "../tensor.h" // For legacy Tensor struct definition
#include "../logger.h"
#include <mpi.h>

namespace llaminar
{

    // Factory function implementations

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_simple(const std::vector<int> &shape)
    {
        return std::make_shared<SimpleTensor>(shape);
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_simple(const std::vector<int> &shape,
                                                                       const std::vector<float> &data)
    {
        return std::make_shared<SimpleTensor>(shape, data);
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_cosma(const std::vector<int> &shape,
                                                                      const std::string &label,
                                                                      int mpi_rank)
    {
        try
        {
            return std::make_shared<COSMATensor>(shape, label, mpi_rank);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Failed to create COSMATensor: " + std::string(e.what()) +
                      ". Falling back to SimpleTensor.");
            return create_simple(shape);
        }
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_auto(const std::vector<int> &shape,
                                                                     bool prefer_distributed)
    {
        // Heuristics for automatic tensor type selection

        // Check if MPI is available and we have multiple processes
        int mpi_size = 1;
        int mpi_rank = 0;
        int mpi_initialized;
        MPI_Initialized(&mpi_initialized);

        if (mpi_initialized)
        {
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        }

        // Decision criteria for using COSMA tensor:
        // 1. MPI environment with multiple processes
        // 2. 2D matrices (COSMA's strength)
        // 3. Large enough matrices to benefit from distribution
        // 4. User preference for distributed

        bool should_use_cosma = false;

        if (mpi_size > 1 && shape.size() == 2)
        {
            // Calculate total elements
            long total_elements = 1;
            for (int dim : shape)
            {
                total_elements *= dim;
            }

            // Use COSMA for large matrices or when explicitly preferred
            const long COSMA_THRESHOLD = 256 * 256; // 64K elements threshold
            should_use_cosma = (total_elements >= COSMA_THRESHOLD) || prefer_distributed;
        }

        if (should_use_cosma)
        {
            LOG_DEBUG("Auto-selecting COSMATensor for shape [" +
                      std::to_string(shape[0]) + ", " + std::to_string(shape[1]) +
                      "] with " + std::to_string(mpi_size) + " MPI processes");

            // Generate a label based on tensor characteristics
            std::string auto_label = "auto_" + std::to_string(shape[0]) + "x" + std::to_string(shape[1]);
            return create_cosma(shape, auto_label, mpi_rank);
        }
        else
        {
            LOG_DEBUG("Auto-selecting SimpleTensor for shape of size " + std::to_string(shape.size()));
            return create_simple(shape);
        }
    }

    std::shared_ptr<TensorBase> TensorFactory::convert_to_cosma(const std::shared_ptr<TensorBase> &tensor,
                                                                const std::string &label,
                                                                int mpi_rank)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // If already a COSMA tensor, return as-is
        if (auto cosma_tensor = std::dynamic_pointer_cast<COSMATensor>(tensor))
        {
            return tensor;
        }

        // Create new COSMA tensor and copy data
        try
        {
            auto cosma_tensor = create_cosma(tensor->shape(), label, mpi_rank);
            cosma_tensor->copy_from(*tensor);

            LOG_DEBUG("Converted " + tensor->type_name() + " to COSMATensor");
            return cosma_tensor;
        }
        catch (const std::exception &e)
        {
            LOG_WARN("Failed to convert to COSMATensor: " + std::string(e.what()) +
                     ". Returning original tensor.");
            return tensor;
        }
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::convert_to_simple(const std::shared_ptr<llaminar::TensorBase> &tensor)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // If already a simple tensor, return as-is
        if (auto simple_tensor = std::dynamic_pointer_cast<SimpleTensor>(tensor))
        {
            return tensor;
        }

        // Create new simple tensor and copy data
        auto simple_tensor = create_simple(tensor->shape());
        simple_tensor->copy_from(*tensor);

        LOG_DEBUG("Converted " + tensor->type_name() + " to SimpleTensor");
        return simple_tensor;
    }

    // Legacy compatibility methods

    std::shared_ptr<SimpleTensor> TensorFactory::to_simple_tensor(std::shared_ptr<llaminar::TensorBase> tensor)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // If already a SimpleTensor, cast and return
        if (auto simple = std::dynamic_pointer_cast<SimpleTensor>(tensor))
        {
            return simple;
        }

        // Convert to SimpleTensor
        auto simple_tensor = std::make_shared<SimpleTensor>(tensor->shape());
        std::copy(tensor->data(), tensor->data() + tensor->size(), simple_tensor->data());
        return simple_tensor;
    }

    std::shared_ptr<COSMATensor> TensorFactory::to_cosma_tensor(std::shared_ptr<llaminar::TensorBase> tensor)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // If already a COSMATensor, cast and return
        if (auto cosma = std::dynamic_pointer_cast<COSMATensor>(tensor))
        {
            return cosma;
        }

        // Convert to COSMATensor
        int mpi_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

        try
        {
            auto cosma_tensor = std::make_shared<COSMATensor>(tensor->shape(), "converted", mpi_rank);
            std::copy(tensor->data(), tensor->data() + tensor->size(), cosma_tensor->data());
            return cosma_tensor;
        }
        catch (const std::exception &e)
        {
            LOG_WARN("Failed to convert to COSMATensor: " + std::string(e.what()));
            return nullptr;
        }
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::from_tensor(std::shared_ptr<llaminar::Tensor> legacy_tensor)
    {
        if (!legacy_tensor)
        {
            return nullptr;
        }

        // Create SimpleTensor from legacy Tensor
        return std::make_shared<SimpleTensor>(legacy_tensor->shape, legacy_tensor->data);
    }

    std::shared_ptr<llaminar::Tensor> TensorFactory::to_tensor(std::shared_ptr<llaminar::TensorBase> tensor)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // Create legacy Tensor from TensorBase
        auto legacy_tensor = std::make_shared<llaminar::Tensor>(tensor->shape());
        std::copy(tensor->data(), tensor->data() + tensor->size(), legacy_tensor->data.begin());
        return legacy_tensor;
    }

    void TensorFactory::copy_from_simple_tensor(std::shared_ptr<llaminar::Tensor> source,
                                                std::shared_ptr<llaminar::TensorBase> target)
    {
        if (!source || !target)
        {
            return;
        }

        // Verify compatible shapes
        if (source->shape != target->shape())
        {
            LOG_WARN("Shape mismatch in copy_from_simple_tensor");
            return;
        }

        // Copy data from legacy Tensor to TensorBase
        std::copy(source->data.begin(), source->data.end(), target->data());
    }

    // Graph Compute Compatibility Bridge Implementation

    std::shared_ptr<TensorBase> GraphTensorBridge::optimize_for_kernel(std::shared_ptr<llaminar::Tensor> legacy_tensor,
                                                                       const std::string &operation_type)
    {
        if (!legacy_tensor)
        {
            return nullptr;
        }

        // For matrix operations (especially matmul), consider COSMA optimization
        bool prefer_cosma = (operation_type == "matmul" || operation_type == "attention") &&
                            legacy_tensor->shape.size() == 2;

        if (prefer_cosma)
        {
            // Try to create COSMA tensor for potential zero-copy operations
            auto cosma_result = TensorFactory::create_auto(legacy_tensor->shape, true);
            if (cosma_result && cosma_result->type_name() == "COSMATensor")
            {
                // Copy data to COSMA tensor
                std::copy(legacy_tensor->data.begin(), legacy_tensor->data.end(), cosma_result->data());
                LOG_DEBUG("Optimized legacy tensor to COSMATensor for " + operation_type);
                return cosma_result;
            }
        }

        // Fallback to SimpleTensor (zero-copy wrapper)
        auto simple_result = std::make_shared<SimpleTensor>(legacy_tensor->shape, legacy_tensor->data);
        LOG_DEBUG("Wrapped legacy tensor as SimpleTensor for " + operation_type);
        return simple_result;
    }

    std::shared_ptr<llaminar::Tensor> GraphTensorBridge::to_graph_tensor(std::shared_ptr<TensorBase> tensor)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // Create legacy Tensor and copy data back
        auto legacy_tensor = std::make_shared<llaminar::Tensor>(tensor->shape());
        std::copy(tensor->data(), tensor->data() + tensor->size(), legacy_tensor->data.begin());

        LOG_DEBUG("Converted " + tensor->type_name() + " back to legacy Tensor");
        return legacy_tensor;
    }

    std::shared_ptr<TensorBase> GraphTensorBridge::auto_upgrade(std::shared_ptr<llaminar::Tensor> legacy_tensor,
                                                                const std::string &kernel_type)
    {
        if (!legacy_tensor)
        {
            return nullptr;
        }

        // Zero-copy wrapper as SimpleTensor for most cases
        // This avoids data copying while enabling TensorBase interface
        auto simple_tensor = std::make_shared<SimpleTensor>(legacy_tensor->shape, legacy_tensor->data);

        // For compute-intensive kernels, consider promoting to COSMA
        if ((kernel_type == "MatMulKernel" || kernel_type == "AttentionKernel") &&
            legacy_tensor->shape.size() == 2)
        {

            // Check if matrix is large enough to benefit from COSMA
            long total_elements = 1;
            for (int dim : legacy_tensor->shape)
            {
                total_elements *= dim;
            }

            const long UPGRADE_THRESHOLD = 512 * 512; // 256K elements
            if (total_elements >= UPGRADE_THRESHOLD)
            {
                try
                {
                    int mpi_rank = 0;
                    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

                    auto cosma_tensor = std::make_shared<COSMATensor>(legacy_tensor->shape, "auto_upgrade", mpi_rank);
                    std::copy(legacy_tensor->data.begin(), legacy_tensor->data.end(), cosma_tensor->data());

                    LOG_DEBUG("Auto-upgraded large matrix to COSMATensor for " + kernel_type);
                    return cosma_tensor;
                }
                catch (const std::exception &e)
                {
                    LOG_WARN("Failed to upgrade to COSMATensor: " + std::string(e.what()) + ", using SimpleTensor");
                }
            }
        }

        return simple_tensor;
    }

} // namespace llaminar